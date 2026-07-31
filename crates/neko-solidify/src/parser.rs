use neko_core::{GraphUpdate, NekoError};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct SolidifyJson {
    #[serde(default)]
    updates: Vec<GraphUpdateJson>,
}

#[derive(Debug, Deserialize)]
struct GraphUpdateJson {
    cypher: String,
    #[serde(default)]
    params: HashMap<String, serde_json::Value>,
}

pub fn parse_solidify_updates(raw: &str) -> Result<Vec<GraphUpdate>, NekoError> {
    let cleaned = neko_llm::extract_json(raw).unwrap_or_else(|| raw.trim().to_string());
    let parsed: SolidifyJson = serde_json::from_str(&cleaned)
        .map_err(|e| NekoError::parse(format!("cannot parse solidify updates: {e}")))?;

    Ok(parsed
        .updates
        .into_iter()
        .map(|u| GraphUpdate {
            cypher: u.cypher,
            params: u.params,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_updates_json() {
        let raw = r#"{"updates":[{"cypher":"MERGE (a:User {id: $from})","params":{"from":"u1"}}]}"#;
        let updates = parse_solidify_updates(raw).unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].cypher, "MERGE (a:User {id: $from})");
        assert_eq!(
            updates[0].params.get("from"),
            Some(&serde_json::Value::String("u1".to_string()))
        );
    }

    #[test]
    fn parses_empty_updates() {
        let raw = r#"{"updates":[]}"#;
        let updates = parse_solidify_updates(raw).unwrap();
        assert!(updates.is_empty());
    }
}
