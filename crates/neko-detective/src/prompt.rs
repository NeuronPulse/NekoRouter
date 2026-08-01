use neko_core::{ChatMessage, DetectiveInput, MemoryRecord};

/// Build the detective prompt that asks for a dehydrated JSON case report.
pub fn detective_prompt(
    input: &DetectiveInput,
    history: &[ChatMessage],
    _memory: &[MemoryRecord],
) -> String {
    let history_text = if history.is_empty() {
        "（无历史记录）".to_string()
    } else {
        history
            .iter()
            .map(|m| {
                format!(
                    "{} [{}]: {}",
                    m.timestamp.format("%H:%M"),
                    m.nickname,
                    m.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "你是一台无情的侦探机器。请根据以下信息生成一份脱水、结构化的 JSON 结案报告。\n\
         \n\
         目标用户：{}\n\
         当前消息：\"{}\"\n\
         精力值：{:.2}\n\
         好感度：{:.2}\n\
         \n\
         历史记录：\n{}\n\
         \n\
         请只输出 JSON，格式如下：\n\
         {{\n\
           \"target_user\": \"{}\",\n\
           \"summary\": \"一句话总结\",\n\
           \"historical_facts\": [{{\"text\": \"事实\", \"evidence\": [\"证据1\"]}}],\n\
           \"psychological_weaknesses\": [{{\"description\": \"弱点描述\", \"severity\": 0.5}}],\n\
           \"relationship_changes\": [{{\"from\": \"user_a\", \"to\": \"user_b\", \"kind\": \"tease\", \"delta\": -0.1, \"evidence\": [\"证据\"]}}],\n\
           \"recommended_tone\": \"warm\" | \"cold\" | \"playful\" | \"cautious\" | \"neutral\",\n\
           \"confidence\": 0.8\n\
         }}",
        input.target_user,
        input.message.content,
        input.state.energy,
        input.state.favorability,
        history_text,
        input.target_user,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use neko_core::{AffectiveState, ChatMessage, DetectiveInput};
    use uuid::Uuid;

    fn make_message(content: &str) -> ChatMessage {
        ChatMessage {
            id: Uuid::new_v4(),
            trace_id: Uuid::new_v4(),
            group_id: "12345".to_string(),
            sender: "67890".to_string(),
            nickname: "Alice".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            reply_to: None,
            raw_payload: serde_json::Value::Null,
        }
    }

    fn input(content: &str) -> DetectiveInput {
        DetectiveInput {
            message: make_message(content),
            state: AffectiveState {
                energy: 0.6,
                favorability: 0.4,
                ..Default::default()
            },
            target_user: "67890".to_string(),
        }
    }

    #[test]
    fn prompt_contains_target_and_state() {
        let input = input("你觉得呢");
        let prompt = detective_prompt(&input, &[], &[]);
        assert!(prompt.contains("目标用户：67890"));
        assert!(prompt.contains("当前消息：\"你觉得呢\""));
        assert!(prompt.contains("精力值：0.60"));
        assert!(prompt.contains("好感度：0.40"));
    }

    #[test]
    fn prompt_handles_empty_history() {
        let input = input("hi");
        let prompt = detective_prompt(&input, &[], &[]);
        assert!(prompt.contains("（无历史记录）"));
    }

    #[test]
    fn prompt_lists_history() {
        let input = input("hi");
        let history = vec![
            ChatMessage {
                content: "first".to_string(),
                nickname: "Bob".to_string(),
                ..make_message("")
            },
            ChatMessage {
                content: "second".to_string(),
                nickname: "Alice".to_string(),
                ..make_message("")
            },
        ];
        let prompt = detective_prompt(&input, &history, &[]);
        assert!(prompt.contains("[Bob]: first"));
        assert!(prompt.contains("[Alice]: second"));
        assert!(!prompt.contains("（无历史记录）"));
    }
}
