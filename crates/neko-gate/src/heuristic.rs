use neko_core::ChatMessage;

use crate::actor::GateConfig;

/// Strategy used by the gate to decide whether a message can be handled by the
/// cheap cozy path.
pub trait GateHeuristic: Send + Sync {
    /// Return `true` if the message should be handled by the cozy LLM path.
    fn should_cozy(&self, msg: &ChatMessage, config: &GateConfig) -> bool;
}

/// Default heuristic: filter commands, URLs, and overly long messages.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultHeuristic;

impl GateHeuristic for DefaultHeuristic {
    fn should_cozy(&self, msg: &ChatMessage, config: &GateConfig) -> bool {
        let text = &msg.content;
        if text.starts_with('/') || text.starts_with('!') {
            return false;
        }
        if text.contains("http://") || text.contains("https://") {
            return false;
        }
        if msg.word_count() > config.max_cozy_words * 3 {
            return false;
        }
        true
    }
}

/// Stub heuristic that always escalates to the council layer.
///
/// Useful for testing the full Layer 3/4/5 pipeline without relying on the
/// cheap model's cozy path.
#[derive(Debug, Default, Clone, Copy)]
pub struct EscalateAllHeuristic;

impl GateHeuristic for EscalateAllHeuristic {
    fn should_cozy(&self, _msg: &ChatMessage, _config: &GateConfig) -> bool {
        false
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

    fn msg(content: &str) -> ChatMessage {
        ChatMessage {
            id: uuid::Uuid::new_v4(),
            group_id: "g1".to_string(),
            sender: "u1".to_string(),
            nickname: "Alice".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            reply_to: None,
            raw_payload: serde_json::Value::Null,
        }
    }

    #[test]
    fn default_allows_short_chitchat() {
        let config = GateConfig {
            max_cozy_words: 10,
            ..Default::default()
        };
        assert!(DefaultHeuristic.should_cozy(&msg("你好"), &config));
    }

    #[test]
    fn default_escalates_commands_and_urls() {
        let config = GateConfig {
            max_cozy_words: 10,
            ..Default::default()
        };
        assert!(!DefaultHeuristic.should_cozy(&msg("/help"), &config));
        assert!(!DefaultHeuristic.should_cozy(&msg("!stats"), &config));
        assert!(!DefaultHeuristic.should_cozy(&msg("see https://example.com"), &config));
    }

    #[test]
    fn default_escalates_long_messages() {
        let config = GateConfig {
            max_cozy_words: 2,
            ..Default::default()
        };
        // word_count counts whitespace-separated tokens, so use spaced text.
        assert!(
            !DefaultHeuristic.should_cozy(&msg("this message has more than six words"), &config)
        );
    }

    #[test]
    fn escalate_all_always_false() {
        let config = GateConfig::default();
        assert!(!EscalateAllHeuristic.should_cozy(&msg("你好"), &config));
    }
}
