use neko_core::{ChatMessage, CouncilInput};

/// Build the Mind Council debate prompt.
///
/// Three personas debate inside a single LLM call:
/// - 猫系本能：emotional, instinctive, cat-girl reactions.
/// - 社交伪装：socially aware, camouflage, group harmony.
/// - 冷酷理智：cold rationality, risk analysis.
///
/// The Chief Routing Officer must output a JSON decision.
pub fn council_prompt(input: &CouncilInput, context: &[ChatMessage]) -> String {
    let user_nick = &input.message.nickname;
    let content = &input.message.content;
    let energy = input.state.energy;
    let favor = input.state.favorability;

    let context_text = if context.is_empty() {
        "（无历史上下文）".to_string()
    } else {
        context
            .iter()
            .map(|m| format!("{}: {}", m.nickname, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let daily_block = if input.daily_context.trim().is_empty() {
        String::new()
    } else {
        format!("当前长期关系记忆：\n{}\n\n", input.daily_context.trim())
    };

    format!(
        "你是一座赛博猫娘心智议会中的首席路由官。现在需要决定如何回复 QQ 群里的这条消息。\n\
         \n\
         目标用户：{}\n\
         当前消息：\"{}\"\n\
         精力值：{:.2}（越高越活跃）\n\
         好感度：{:.2}（越高越亲近）\n\
         \n\
         {daily_block}\
         近期历史上下文：\n{}\n\
         \n\
         议会中有三个角色在辩论：\n\
         1. 【猫系本能】：凭直觉、情绪、猫娘本能反应。\n\
         2. 【社交伪装】：考虑群聊氛围、社交礼仪、避免冲突。\n\
         3. 【冷酷理智】：理性分析风险、判断是否需要更多信息。\n\
         \n\
         请综合三方的意见，由首席路由官输出最终决策。只输出 JSON，不要解释。JSON 格式如下：\n\
         {{\n\
           \"action\": \"reply\" | \"detective\" | \"ignore\",\n\
           \"reasoning\": \"一句话决策理由\",\n\
           \"draft_reply\": \"如果 action 是 reply，填写回复内容；否则为空字符串\"\n\
         }}\n\
         \n\
         注意：回复内容不要超过 50 字。",
        user_nick, content, energy, favor, context_text
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use neko_core::{AffectiveState, ChatMessage, CouncilInput};
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

    fn input_with_context(context: Vec<ChatMessage>) -> CouncilInput {
        CouncilInput {
            message: make_message("在吗"),
            state: AffectiveState {
                energy: 0.75,
                favorability: 0.33,
                ..Default::default()
            },
            context,
            daily_context: String::new(),
        }
    }

    #[test]
    fn prompt_contains_user_and_message() {
        let input = input_with_context(vec![]);
        let prompt = council_prompt(&input, &input.context);
        assert!(prompt.contains("Alice"));
        assert!(prompt.contains("在吗"));
        assert!(prompt.contains("精力值：0.75"));
        assert!(prompt.contains("好感度：0.33"));
    }

    #[test]
    fn prompt_handles_empty_context() {
        let input = input_with_context(vec![]);
        let prompt = council_prompt(&input, &input.context);
        assert!(prompt.contains("（无历史上下文）"));
    }

    #[test]
    fn prompt_lists_context_messages() {
        let ctx = vec![
            make_message(" earlier "),
            ChatMessage {
                content: "second".to_string(),
                nickname: "Bob".to_string(),
                ..make_message("")
            },
        ];
        let input = input_with_context(ctx.clone());
        let prompt = council_prompt(&input, &input.context);
        assert!(prompt.contains("Alice:  earlier "));
        assert!(prompt.contains("Bob: second"));
        assert!(!prompt.contains("（无历史上下文）"));
    }

    #[test]
    fn prompt_contains_required_actions() {
        let input = input_with_context(vec![]);
        let prompt = council_prompt(&input, &input.context);
        assert!(prompt.contains("\"action\": \"reply\" | \"detective\" | \"ignore\""));
    }
}
