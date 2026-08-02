use neko_core::{ChatMessage, DropReason, EngagementType};

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
pub trait GateHeuristic: Send + Sync {
    /// Classify the message into a gate decision.
    fn classify(
        &self,
        msg: &ChatMessage,
        config: &GateConfig,
        self_id: &BotIdentity,
    ) -> GateClassification;
}

/// Default heuristic: filter commands, URLs, and overly long messages; detect
/// directed messages as personal replies; treat short chitchat as ambient join.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultHeuristic;

impl GateHeuristic for DefaultHeuristic {
    fn classify(
        &self,
        msg: &ChatMessage,
        config: &GateConfig,
        self_id: &BotIdentity,
    ) -> GateClassification {
        let text = &msg.content;

        if text.starts_with('/') || text.starts_with('!') {
            return GateClassification::Drop(DropReason::Other("command".to_string()));
        }
        if text.contains("http://") || text.contains("https://") {
            return GateClassification::Drop(DropReason::Spam);
        }
        if msg.char_count() > config.max_message_length {
            return GateClassification::Drop(DropReason::TooLong);
        }

        if self_id.is_addressed_to_me(msg) {
            return GateClassification::Escalate(EngagementType::PersonalReply);
        }

        // Short, low-stakes chitchat is a candidate for ambient join.
        if msg.word_count() <= config.max_ambient_words * 3 {
            return GateClassification::Escalate(EngagementType::AmbientJoin);
        }

        // Everything else is treated as a potential personal reply so the
        // council can decide with full context.
        GateClassification::Escalate(EngagementType::PersonalReply)
    }
}

/// Stub heuristic that always escalates as a personal reply.
///
/// Useful for testing the full Layer 3/4/5 pipeline without relying on the
/// gate's classification.
#[derive(Debug, Default, Clone, Copy)]
pub struct EscalateAllHeuristic;

impl GateHeuristic for EscalateAllHeuristic {
    fn classify(
        &self,
        _msg: &ChatMessage,
        _config: &GateConfig,
        _self_id: &BotIdentity,
    ) -> GateClassification {
        GateClassification::Escalate(EngagementType::PersonalReply)
    }
}

/// Build a heuristic from its configuration name.
pub fn heuristic_from_name(name: &str) -> Box<dyn GateHeuristic> {
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
            heuristic: "default".to_string(),
        }
    }

    #[test]
    fn default_drops_commands_and_urls() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&msg("/help"), &cfg, &id),
            GateClassification::Drop(DropReason::Other("command".to_string()))
        );
        assert_eq!(
            h.classify(&msg("see https://example.com"), &cfg, &id),
            GateClassification::Drop(DropReason::Spam)
        );
    }

    #[test]
    fn default_drops_long_messages() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = GateConfig {
            max_message_length: 10,
            ..config()
        };
        assert_eq!(
            h.classify(&msg("this is a long message"), &cfg, &id),
            GateClassification::Drop(DropReason::TooLong)
        );
    }

    #[test]
    fn default_detects_at_mention_as_personal() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&at_msg(123456789), &cfg, &id),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }

    #[test]
    fn default_detects_reply_to_bot_as_personal() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        let mut m = msg("你说得对");
        m.reply_to = Some(Uuid::new_v4());
        assert_eq!(
            h.classify(&m, &cfg, &id),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }

    #[test]
    fn default_detects_name_mention_as_personal() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&msg("NekoRouter 你怎么看"), &cfg, &id),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
        assert_eq!(
            h.classify(&msg("猫娘出来"), &cfg, &id),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }

    #[test]
    fn default_classifies_short_chitchat_as_ambient() {
        let h = DefaultHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&msg("你好"), &cfg, &id),
            GateClassification::Escalate(EngagementType::AmbientJoin)
        );
    }

    #[test]
    fn escalate_all_always_personal() {
        let h = EscalateAllHeuristic;
        let id = identity();
        let cfg = config();
        assert_eq!(
            h.classify(&msg("/help"), &cfg, &id),
            GateClassification::Escalate(EngagementType::PersonalReply)
        );
    }
}
