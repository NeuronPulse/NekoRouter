use chrono::{TimeZone, Utc};
use neko_core::{ChatMessage, GroupId, MessageId, UserId};
use serde_json::{json, Value};

/// Fixed namespace used to derive a stable UUID from a OneBot message id.
const NAMESPACE_ONEBOT: uuid::Uuid =
    uuid::Uuid::from_u128(0x6ba7_b810_9dad_11d1_80b4_00c0_4fd4_30c8);

/// Derive a deterministic message id from the group and platform message id.
///
/// Re-delivered messages (e.g. after a reconnect) keep the same id so the
/// history store can deduplicate them instead of inserting duplicates.
fn derive_message_id(group_id: &str, platform_message_id: &str) -> MessageId {
    let name = format!("{group_id}:{platform_message_id}");
    MessageId::new_v5(&NAMESPACE_ONEBOT, name.as_bytes())
}

/// Parse a OneBot v11 group message payload into a `ChatMessage`.
///
/// Supports both plain string `message` and single-segment array messages.
pub fn parse_onebot11_group_message(payload: &Value) -> Option<ChatMessage> {
    let post_type = payload.get("post_type")?.as_str()?;
    if post_type != "message" {
        return None;
    }

    let message_type = payload.get("message_type")?.as_str()?;
    if message_type != "group" {
        return None;
    }

    let group_id = payload.get("group_id")?.as_i64()?;
    let user_id = payload.get("user_id")?.as_i64()?;
    let platform_message_id = match payload.get("message_id")? {
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => return None,
    };

    let content = extract_text(payload.get("message")?)?;
    let nickname = payload
        .get("sender")
        .and_then(|s| s.get("nickname"))
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string();

    let timestamp = payload
        .get("time")
        .and_then(|t| t.as_i64())
        .map(|t| t * 1000)
        .unwrap_or_else(|| Utc::now().timestamp_millis());

    let timestamp = Utc.timestamp_millis_opt(timestamp).single()?;

    // Derive a stable id from the platform message id so re-delivered
    // messages after a reconnect do not get inserted twice.
    let id = derive_message_id(&group_id.to_string(), &platform_message_id);

    // A `reply` message segment references the platform id of the message
    // this one answers; map it through the same derivation as `id`.
    let reply_to = payload
        .get("message")
        .and_then(extract_reply_to)
        .map(|platform_id| derive_message_id(&group_id.to_string(), &platform_id.to_string()));

    Some(ChatMessage {
        id,
        trace_id: uuid::Uuid::new_v4(),
        group_id: GroupId::from(group_id.to_string()),
        sender: UserId::from(user_id.to_string()),
        nickname,
        content,
        timestamp,
        reply_to,
        raw_payload: payload.clone(),
    })
}

/// Extract the platform message id referenced by a `reply` segment, if any.
fn extract_reply_to(message: &Value) -> Option<u64> {
    let Value::Array(segments) = message else {
        return None;
    };
    for seg in segments {
        if seg.get("type").and_then(|t| t.as_str()) == Some("reply") {
            let id = seg.get("data")?.get("id")?;
            return match id {
                Value::Number(n) => n.as_u64(),
                Value::String(s) => s.parse().ok(),
                _ => None,
            };
        }
    }
    None
}

fn extract_text(message: &Value) -> Option<String> {
    match message {
        Value::String(s) => Some(s.clone()),
        Value::Array(segments) => {
            let mut texts = Vec::new();
            for seg in segments {
                if let Some(ty) = seg.get("type").and_then(|t| t.as_str()) {
                    if ty == "text" {
                        if let Some(text) = seg
                            .get("data")
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            texts.push(text.to_string());
                        }
                    }
                }
            }
            if texts.is_empty() {
                None
            } else {
                Some(texts.join(""))
            }
        }
        _ => None,
    }
}

/// Build a OneBot v11 `send_group_msg` action payload.
pub fn build_onebot11_group_reply(group_id: &GroupId, content: &str) -> Value {
    json!({
        "action": "send_group_msg",
        "params": {
            "group_id": group_id.parse::<i64>().unwrap_or(0),
            "message": content
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_string_message() {
        let payload = json!({
            "post_type": "message",
            "message_type": "group",
            "group_id": 12345,
            "user_id": 67890,
            "message_id": 111,
            "sender": {"nickname": "Alice"},
            "message": "hello world",
            "time": 1700000000
        });
        let msg = parse_onebot11_group_message(&payload).unwrap();
        assert_eq!(msg.group_id, "12345");
        assert_eq!(msg.sender, "67890");
        assert_eq!(msg.nickname, "Alice");
        assert_eq!(msg.content, "hello world");
    }

    #[test]
    fn parses_array_message() {
        let payload = json!({
            "post_type": "message",
            "message_type": "group",
            "group_id": 12345,
            "user_id": 67890,
            "message_id": 222,
            "sender": {"nickname": "Bob"},
            "message": [
                {"type": "text", "data": {"text": "hi "}},
                {"type": "text", "data": {"text": "there"}}
            ],
            "time": 1700000000
        });
        let msg = parse_onebot11_group_message(&payload).unwrap();
        assert_eq!(msg.content, "hi there");
    }

    #[test]
    fn derives_stable_message_id() {
        let payload = json!({
            "post_type": "message",
            "message_type": "group",
            "group_id": 12345,
            "user_id": 67890,
            "message_id": 111,
            "sender": {"nickname": "Alice"},
            "message": "hello",
            "time": 1700000000
        });
        let a = parse_onebot11_group_message(&payload).unwrap();
        let b = parse_onebot11_group_message(&payload).unwrap();
        assert_eq!(a.id, b.id);

        let other_group = json!({
            "post_type": "message",
            "message_type": "group",
            "group_id": 999,
            "user_id": 67890,
            "message_id": 111,
            "sender": {"nickname": "Alice"},
            "message": "hello",
            "time": 1700000000
        });
        let c = parse_onebot11_group_message(&other_group).unwrap();
        assert_ne!(a.id, c.id);
    }

    #[test]
    fn parses_string_message_id() {
        let payload = json!({
            "post_type": "message",
            "message_type": "group",
            "group_id": 12345,
            "user_id": 67890,
            "message_id": "abc-123",
            "sender": {"nickname": "Alice"},
            "message": "hello",
            "time": 1700000000
        });
        let a = parse_onebot11_group_message(&payload).unwrap();
        let b = parse_onebot11_group_message(&payload).unwrap();
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn parses_reply_segment_as_reply_to() {
        let payload = json!({
            "post_type": "message",
            "message_type": "group",
            "group_id": 12345,
            "user_id": 67890,
            "message_id": 222,
            "sender": {"nickname": "Bob"},
            "message": [
                {"type": "reply", "data": {"id": "111"}},
                {"type": "text", "data": {"text": "哈哈哈"}}
            ],
            "time": 1700000000
        });
        let msg = parse_onebot11_group_message(&payload).unwrap();
        assert_eq!(msg.content, "哈哈哈");
        assert_eq!(msg.reply_to, Some(derive_message_id("12345", "111")));
    }

    #[test]
    fn ignores_reply_segment_when_missing_id() {
        let payload = json!({
            "post_type": "message",
            "message_type": "group",
            "group_id": 12345,
            "user_id": 67890,
            "message_id": 222,
            "sender": {"nickname": "Bob"},
            "message": [
                {"type": "reply", "data": {}},
                {"type": "text", "data": {"text": "hi"}}
            ],
            "time": 1700000000
        });
        let msg = parse_onebot11_group_message(&payload).unwrap();
        assert_eq!(msg.content, "hi");
        assert!(msg.reply_to.is_none());
    }
}
