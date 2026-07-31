/// Build the COZY gate prompt.
///
/// The model is asked to reply like a slightly aloof cat-girl in at most
/// `max_words` words. Only the reply text should be output.
pub fn cozy_prompt(message: &str, max_words: usize) -> String {
    format!(
        "你是一只高冷的猫娘，正在 QQ 群里潜水。群友说：\"{}\"

请用不超过 {} 个字回复，语气简短、傲娇、可爱。只输出回复内容，不要解释、不要加引号。",
        message, max_words
    )
}

/// Build a prompt that asks the cheap model whether the message is ordinary
/// small talk or something that should be escalated.
pub fn classify_prompt(message: &str) -> String {
    format!(
        "判断下面的 QQ 群消息属于哪一类。只输出一个单词：COZY（普通闲聊，适合简短回复）、DROP（垃圾/广告/无意义）、ESCALATE（冲突/烂梗/需要上下文）。\n\n消息：\"{}\"",
        message
    )
}
