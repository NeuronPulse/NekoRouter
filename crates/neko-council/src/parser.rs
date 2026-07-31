use neko_core::{CouncilAction, CouncilDecision, NekoError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CouncilDecisionJson {
    action: String,
    #[serde(default)]
    reasoning: String,
    #[serde(default)]
    draft_reply: String,
}

pub fn parse_council_decision(raw: &str) -> Result<CouncilDecision, NekoError> {
    let cleaned = neko_llm::extract_json(raw).unwrap_or_else(|| raw.trim().to_string());
    let parsed: CouncilDecisionJson = serde_json::from_str(&cleaned)
        .map_err(|e| NekoError::parse(format!("cannot parse council decision: {e}")))?;

    let action = match parsed.action.as_str() {
        "reply" => CouncilAction::ReplyDirectly,
        "detective" => CouncilAction::LaunchDetective,
        "ignore" => CouncilAction::Ignore,
        other => return Err(NekoError::parse(format!("unknown council action: {other}"))),
    };

    Ok(CouncilDecision {
        action,
        reasoning: parsed.reasoning,
        draft_reply: if parsed.draft_reply.is_empty() {
            None
        } else {
            Some(parsed.draft_reply)
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use neko_core::CouncilAction;

    #[test]
    fn parses_reply_action() {
        let raw = r#"{"action":"reply","reasoning":"用户打招呼","draft_reply":"你好呀~"}"#;
        let decision = parse_council_decision(raw).unwrap();
        assert_eq!(decision.action, CouncilAction::ReplyDirectly);
        assert_eq!(decision.reasoning, "用户打招呼");
        assert_eq!(decision.draft_reply, Some("你好呀~".to_string()));
    }

    #[test]
    fn parses_detective_action() {
        let raw = r#"{"action":"detective","reasoning":"需要上下文","draft_reply":""}"#;
        let decision = parse_council_decision(raw).unwrap();
        assert_eq!(decision.action, CouncilAction::LaunchDetective);
        assert!(decision.draft_reply.is_none());
    }

    #[test]
    fn parses_ignore_action() {
        let raw = r#"{"action":"ignore","reasoning":"不相关","draft_reply":""}"#;
        let decision = parse_council_decision(raw).unwrap();
        assert_eq!(decision.action, CouncilAction::Ignore);
    }

    #[test]
    fn unknown_action_returns_error() {
        let raw = r#"{"action":"dance","reasoning":"???","draft_reply":""}"#;
        let err = parse_council_decision(raw).unwrap_err();
        assert!(format!("{err}").contains("unknown council action"));
    }

    #[test]
    fn empty_draft_reply_becomes_none() {
        let raw = r#"{"action":"reply","reasoning":"","draft_reply":""}"#;
        let decision = parse_council_decision(raw).unwrap();
        assert_eq!(decision.action, CouncilAction::ReplyDirectly);
        assert_eq!(decision.draft_reply, None);
    }

    #[test]
    fn strips_markdown_json_fence() {
        let raw =
            "```json\n{\"action\":\"ignore\",\"reasoning\":\" fenced \",\"draft_reply\":\"\"}\n```";
        let decision = parse_council_decision(raw).unwrap();
        assert_eq!(decision.action, CouncilAction::Ignore);
        assert_eq!(decision.reasoning, " fenced ");
    }

    #[test]
    fn malformed_json_returns_error() {
        let raw = "not json";
        let err = parse_council_decision(raw).unwrap_err();
        assert!(matches!(err, NekoError::Parse(_)));
    }
}
