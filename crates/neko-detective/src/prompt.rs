use neko_core::{ChatMessage, DetectiveInput, MemoryRecord, TopicBurst};

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

/// Build the prompt for the detective's memory-curator mode.
///
/// Given a hot conversation window, the model decides what is worth remembering
/// and where each insight belongs (vector facts, vector culture, graph
/// relations, affective deltas).
pub fn memory_curator_prompt(burst: &TopicBurst) -> String {
    let messages_text = burst
        .messages
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
        .join("\n");

    format!(
        "你是一位群聊记忆管理员。请分析下面这段突然热起来的对话，并决定哪些信息值得长期保存。\n\
         \n\
         请只输出 JSON，顶层结构为：\n\
         {{\n\
           \"group_id\": \"{group_id}\",\n\
           \"summary\": \"这段对话的整体摘要\",\n\
           \"updates\": [\n\
             {{\"kind\": \"vector_fact\", \"content\": \"...\", \"tags\": [\"tag\"], \"related_users\": [\"user_id\"]}},\n\
             {{\"kind\": \"vector_culture\", \"content\": \"...\", \"tags\": [\"meme\"], \"related_users\": []}},\n\
             {{\"kind\": \"graph_relation\", \"from\": \"u1\", \"to\": \"u2\", \"relation\": \"互怼\", \"delta\": -0.2, \"evidence\": \"...\"}},\n\
             {{\"kind\": \"affective_delta\", \"target_user\": \"u1\", \"energy_delta\": 0.0, \"favorability_delta\": 0.1, \"mood\": null, \"reason\": \"...\"}}\n\
           ]\n\
         }}\n\
         \n\
         原则：\n\
         1. 只记录以后可能用得上的信息，避免噪音。\n\
         2. 区分个人事实（vector_fact）和群文化/梗（vector_culture）。\n\
         3. 当梗或文化与某个人相关时，把TA放进 related_users。\n\
         4. 关系变化（graph_relation）可以是友好、对立、调侃、默契等。\n\
         5. 情感变化（affective_delta）幅度要小（-0.3 到 +0.3），除非事件特别强烈。\n\
         6. 没有值得记录的内容时，updates 可以为空。\n\
         \n\
         对话记录：\n{messages}",
        group_id = burst.group_id,
        messages = messages_text,
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
