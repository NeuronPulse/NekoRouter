use async_trait::async_trait;
use neko_core::{
    AffectiveState, ChatMessage, DropReason, EngagementType, LlmClient, LlmMessage, LlmRequest,
    LlmRole, NekoError,
};
use serde::Deserialize;
use std::sync::Arc;

use crate::actor::GateConfig;

/// Identity of the bot, used by the gate to detect messages directed at it.
#[derive(Debug, Clone, Default)]
pub struct BotIdentity {
    /// The bot's QQ id.
    pub qq_id: u64,
    /// The bot's display name.
    pub name: String,
    /// Additional aliases that refer to the bot (e.g. "猫娘", "机器人").
    pub aliases: Vec<String>,
}

impl BotIdentity {
    /// Check whether the message explicitly addresses the bot.
    pub fn is_addressed_to_me(&self, msg: &ChatMessage) -> bool {
        // Reply to a message sent by the bot.
        // Note: reply_to stores the original message id; the router resolves the
        // platform id for egress, but here we only have the internal id. The
        // sensory layer could tag self-messages in raw_payload for a stronger
        // signal; as a heuristic we also rely on text matching below.
        if msg.reply_to.is_some() {
            return true;
        }

        let text = msg.content.trim();

        // @mention in OneBot payload is stored in raw_payload; try to match
        // the bot qq_id in the at segments.
        if let Some(true) = msg
            .raw_payload
            .get("message")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter().any(|seg| {
                    seg.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| t == "at")
                        .unwrap_or(false)
                        && seg
                            .get("data")
                            .and_then(|d| d.get("qq"))
                            .and_then(|q| q.as_str())
                            .map(|q| q == self.qq_id.to_string())
                            .unwrap_or(false)
                })
            })
        {
            return true;
        }

        // Text contains the bot name or any alias.
        let names: Vec<String> = std::iter::once(self.name.clone())
            .chain(self.aliases.iter().cloned())
            .filter(|n| !n.is_empty())
            .collect();
        if names.iter().any(|n| text.contains(n)) {
            return true;
        }

        false
    }
}

/// Result of classifying an incoming message at the gate.
#[derive(Debug, Clone, PartialEq)]
pub enum GateClassification {
    /// Drop the message without reply.
    Drop(DropReason),
    /// Escalate to the council layer, specifying how the bot should engage.
    Escalate(EngagementType),
}

/// Strategy used by the gate to decide whether and how the bot should engage.
#[async_trait]
pub trait GateClassifier: Send + Sync {
    /// Classify the message into a gate decision.
    async fn classify(
        &self,
        msg: &ChatMessage,
        recent_context: &[ChatMessage],
        affective: &AffectiveState,
        config: &GateConfig,
        self_id: &BotIdentity,
    ) -> Result<GateClassification, NekoError>;
}

/// Default heuristic: filter commands, URLs, and overly long messages; detect
/// directed messages as personal replies; treat short chitchat as ambient join.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultHeuristic;

#[async_trait]
impl GateClassifier for DefaultHeuristic {
    async fn classify(
        &self,
        msg: &ChatMessage,
        _recent_context: &[ChatMessage],
        _affective: &AffectiveState,
        config: &GateConfig,
        self_id: &BotIdentity,
    ) -> Result<GateClassification, NekoError> {
        let text = &msg.content;

        if text.starts_with('/') || text.starts_with('!') {
            return Ok(GateClassification::Drop(DropReason::Other(
                "command".to_string(),
            )));
        }
        if text.contains("http://") || text.contains("https://") {
            return Ok(GateClassification::Drop(DropReason::Spam));
        }
        if msg.char_count() > config.max_message_length {
            return Ok(GateClassification::Drop(DropReason::TooLong));
        }

        if self_id.is_addressed_to_me(msg) {
            return Ok(GateClassification::Escalate(EngagementType::PersonalReply));
        }

        // Short, low-stakes chitchat is a candidate for ambient join.
        if msg.word_count() <= config.max_ambient_words * 3 {
            return Ok(GateClassification::Escalate(EngagementType::AmbientJoin));
        }

        // Everything else is treated as a potential personal reply so the
        // council can decide with full context.
        Ok(GateClassification::Escalate(EngagementType::PersonalReply))
    }
}

/// Stub heuristic that always escalates as a personal reply.
///
/// Useful for testing the full Layer 3/4/5 pipeline without relying on the
/// gate's classification.
#[derive(Debug, Default, Clone, Copy)]
pub struct EscalateAllHeuristic;

#[async_trait]
impl GateClassifier for EscalateAllHeuristic {
    async fn classify(
        &self,
        _msg: &ChatMessage,
        _recent_context: &[ChatMessage],
        _affective: &AffectiveState,
        _config: &GateConfig,
        _self_id: &BotIdentity,
    ) -> Result<GateClassification, NekoError> {
        Ok(GateClassification::Escalate(EngagementType::PersonalReply))
    }
}

/// LLM-driven classifier for nuanced engagement decisions.
///
/// Obvious drops are handled by a fast heuristic first; only messages that
/// survive the heuristic are sent to the cheap LLM.
pub struct LlmGateClassifier {
    client: Arc<dyn LlmClient>,
}

impl LlmGateClassifier {
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        Self { client }
    }
}

#[derive(Debug, Deserialize)]
struct LlmGateOutput {
    action: String,
    engagement_type: String,
    #[allow(dead_code)]
    confidence: f32,
    #[allow(dead_code)]
    reasoning: String,
}

#[async_trait]
impl GateClassifier for LlmGateClassifier {
    async fn classify(
        &self,
        msg: &ChatMessage,
        recent_context: &[ChatMessage],
        affective: &AffectiveState,
        config: &GateConfig,
        self_id: &BotIdentity,
    ) -> Result<GateClassification, NekoError> {
        // Fast heuristic pre-filter for obvious drops.
        let text = &msg.content;
        if text.starts_with('/') || text.starts_with('!') {
            return Ok(GateClassification::Drop(DropReason::Other(
                "command".to_string(),
            )));
        }
        if text.contains("http://") || text.contains("https://") {
            return Ok(GateClassification::Drop(DropReason::Spam));
        }
        if msg.char_count() > config.max_message_length {
            return Ok(GateClassification::Drop(DropReason::TooLong));
        }

        let context_text = recent_context
            .iter()
            .map(|m| format!("{}: {}", m.nickname, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let system = format!(
            "You are the engagement gate for a QQ group chat bot named {name} (qq {qq}). \
             Decide whether the bot should engage with the latest message. \
             Output JSON with fields: action ('escalate' or 'drop'), engagement_type ('personal_reply' or 'ambient_join'), confidence (0..1), reasoning (short). \
             Rules: personal_reply when directly addressed; ambient_join when the bot can naturally join group chatter; drop for commands, URLs, spam, or anything irrelevant.",
            name = self_id.name,
            qq = self_id.qq_id,
        );

        let user = format!(
            "Recent context:\n{context}\n\nLatest message from {nick}: {content}\n\nBot energy: {energy:.2}, favor: {favor:.2}",
            context = context_text,
            nick = msg.nickname,
            content = msg.content,
            energy = affective.energy,
            favor = affective.favorability,
        );

        let req = LlmRequest {
            messages: vec![
                LlmMessage {
                    role: LlmRole::System,
                    content: system,
                },
                LlmMessage {
                    role: LlmRole::User,
                    content: user,
                },
            ],
            temperature: 0.3,
            max_tokens: Some(128),
            response_format: Some(neko_core::ResponseFormat::JsonObject),
        };

        let resp = self.client.complete(req).await?;
        let parsed: LlmGateOutput = serde_json::from_str(&resp.content).map_err(|e| {
            NekoError::llm(format!("invalid gate LLM output: {e}: {}", resp.content))
        })?;

        match parsed.action.as_str() {
            "drop" => Ok(GateClassification::Drop(DropReason::Other(
                parsed.reasoning,
            ))),
            _ => match parsed.engagement_type.as_str() {
                "ambient_join" => Ok(GateClassification::Escalate(EngagementType::AmbientJoin)),
                _ => Ok(GateClassification::Escalate(EngagementType::PersonalReply)),
            },
        }
    }
}

/// Build a classifier from its configuration name.
pub fn classifier_from_name(name: &str) -> Box<dyn GateClassifier> {
    match name {
        "escalate_all" => Box::new(EscalateAllHeuristic),
        _ => Box::new(DefaultHeuristic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use uuid::Uuid;

    fn identity() -> BotIdentity {
        BotIdentity {
            qq_id: 123456789,
            name: "NekoRouter".to_string(),
            aliases: vec!["猫娘".to_string(), "机器人".to_string()],
        }
    }

    fn msg(content: &str) -> ChatMessage {
        ChatMessage {
            id: Uuid::new_v4(),
            trace_id: Uuid::new_v4(),
            group_id: "g1".to_string(),
            sender: "u1".to_string(),
            nickname: "Alice".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            reply_to: None,
            raw_payload: serde_json::Value::Null,
        }
    }

    fn at_msg(qq: u64) -> ChatMessage {
        ChatMessage {
            raw_payload: json!({
                "message": [{"type": "at", "data": {"qq": qq.to_string()}}]
            }),
            ..msg("hello")
        }
    }

    fn config() -> GateConfig {
        GateConfig {
            max_message_length: 800,
            max_ambient_words: 10,
            concurrency_limit: 8,
            classifier: "default".to_string(),
            context_messages: 10,
            cache_ttl_sec: 30,
            rate_limit_per_min: 60,
        }
    }

    #[tokio::test]
    async fn default_drops_commands_and_urls() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&msg("/help"), &[], &AffectiveState::default(), &cfg, &id)
                .await
                .unwrap(),
            GateClassification::Drop(DropReason::Other("command".to_string()))
        );
        assert_eq!(
            h.classify(
                &msg("see https://example.com"),
                &[],
                &AffectiveState::default(),
                &cfg,
                &id
            )
            .await
            .unwrap(),
            GateClassification::Drop(DropReason::Spam)
        );
    }

    #[tokio::test]
    async fn default_drops_long_messages() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = GateConfig {
            max_message_length: 10,
            ..config()
        };
        assert_eq!(
            h.classify(
                &msg("this is a long message"),
                &[],
                &AffectiveState::default(),
                &cfg,
                &id
            )
            .await
            .unwrap(),
            GateClassification::Drop(DropReason::TooLong)
        );
    }

    #[tokio::test]
    async fn default_detects_at_mention_as_personal() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(
                &at_msg(123456789),
                &[],
                &AffectiveState::default(),
                &cfg,
                &id
            )
            .await
            .unwrap(),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }

    #[tokio::test]
    async fn default_detects_reply_to_bot_as_personal() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        let mut m = msg("你说得对");
        m.reply_to = Some(Uuid::new_v4());
        assert_eq!(
            h.classify(&m, &[], &AffectiveState::default(), &cfg, &id)
                .await
                .unwrap(),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }

    #[tokio::test]
    async fn default_detects_name_mention_as_personal() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(
                &msg("NekoRouter 你怎么看"),
                &[],
                &AffectiveState::default(),
                &cfg,
                &id
            )
            .await
            .unwrap(),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
        assert_eq!(
            h.classify(&msg("猫娘出来"), &[], &AffectiveState::default(), &cfg, &id)
                .await
                .unwrap(),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }

    #[tokio::test]
    async fn default_classifies_short_chitchat_as_ambient() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&msg("你好"), &[], &AffectiveState::default(), &cfg, &id)
                .await
                .unwrap(),
            GateClassification::Escalate(EngagementType::AmbientJoin)
        );
    }

    #[tokio::test]
    async fn escalate_all_always_personal() {
        let h = EscalateAllHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&msg("/help"), &[], &AffectiveState::default(), &cfg, &id)
                .await
                .unwrap(),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }
}
