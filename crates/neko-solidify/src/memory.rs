use async_trait::async_trait;
use neko_core::{GraphStore, GraphUpdate, NekoError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

type RelationshipKey = (String, String, String);

/// In-memory graph store fallback.
///
/// Records every Cypher update it receives and optionally aggregates simple
/// `(from)-[kind]->(to)` relationship deltas when the update params contain
/// `from`, `to`, and `delta`. It is intended for local development and
/// integration tests where a real Neo4j instance is unavailable.
#[derive(Debug, Default, Clone)]
pub struct InMemoryGraphStore {
    updates: Arc<Mutex<Vec<GraphUpdate>>>,
    relationships: Arc<Mutex<HashMap<RelationshipKey, f32>>>,
}

impl InMemoryGraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return all updates applied so far.
    pub fn applied_updates(&self) -> Vec<GraphUpdate> {
        self.updates.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Return the aggregated delta for a relationship, if any.
    pub fn relationship_delta(&self, from: &str, to: &str, kind: &str) -> Option<f32> {
        self.relationships
            .lock()
            .ok()?
            .get(&(from.to_string(), to.to_string(), kind.to_string()))
            .copied()
    }
}

fn extract_delta(value: &serde_json::Value) -> Option<f32> {
    value.as_f64().map(|d| d as f32)
}

fn extract_kind(cypher: &str) -> String {
    // Very light extraction of the relationship type from patterns like
    // `MERGE (a)-[r:SOME_KIND]->(b)`. Falls back to "REL" for hand-written
    // or complex Cypher.
    if let Some(start) = cypher.find("-[r:") {
        let rest = &cypher[start + 4..];
        if let Some(end) = rest.find("]") {
            return rest[..end].to_string();
        }
    }
    "REL".to_string()
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn apply_updates(&self, updates: &[GraphUpdate]) -> Result<(), NekoError> {
        let mut rels = self.relationships.lock().unwrap();

        for update in updates {
            info!("in-memory graph store applying: {}", update.cypher);
            debug!("params: {:?}", update.params);

            if let (Some(from), Some(to), Some(delta)) = (
                update.params.get("from").and_then(|v| v.as_str()),
                update.params.get("to").and_then(|v| v.as_str()),
                update.params.get("delta").and_then(extract_delta),
            ) {
                let kind = extract_kind(&update.cypher);
                let key = (from.to_string(), to.to_string(), kind);
                rels.entry(key)
                    .and_modify(|acc| *acc += delta)
                    .or_insert(delta);
            }
        }

        if let Ok(mut buf) = self.updates.lock() {
            buf.extend_from_slice(updates);
        }

        Ok(())
    }

    async fn relationship_summary(&self, limit: usize) -> Result<String, NekoError> {
        let rels = self.relationships.lock().unwrap();
        let mut rows: Vec<((&str, &str, &str), f32)> = rels
            .iter()
            .map(|((from, to, kind), delta)| ((from.as_str(), to.as_str(), kind.as_str()), *delta))
            .collect();
        rows.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        let lines: Vec<String> = rows
            .iter()
            .take(limit)
            .map(|((from, to, kind), delta)| format!("{from} -[{kind}]-> {to} (delta {delta:+.2})"))
            .collect();
        Ok(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn update(cypher: &str, params: &[(&str, serde_json::Value)]) -> GraphUpdate {
        GraphUpdate {
            cypher: cypher.to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    #[tokio::test]
    async fn records_updates() {
        let store = InMemoryGraphStore::new();
        let u = update("CREATE (n)", &[]);
        store.apply_updates(std::slice::from_ref(&u)).await.unwrap();
        assert_eq!(store.applied_updates().len(), 1);
        assert_eq!(store.applied_updates()[0].cypher, "CREATE (n)");
    }

    #[tokio::test]
    async fn aggregates_relationship_deltas() {
        let store = InMemoryGraphStore::new();
        let u1 = update(
            "MERGE (a)-[r:TEASE]->(b)",
            &[
                ("from", "u1".into()),
                ("to", "u2".into()),
                ("delta", 0.5.into()),
            ],
        );
        let u2 = update(
            "MERGE (a)-[r:TEASE]->(b)",
            &[
                ("from", "u1".into()),
                ("to", "u2".into()),
                ("delta", (-0.2).into()),
            ],
        );

        store.apply_updates(&[u1, u2]).await.unwrap();

        assert!((store.relationship_delta("u1", "u2", "TEASE").unwrap() - 0.3).abs() < 0.001);
    }

    #[tokio::test]
    async fn fallback_kind_for_plain_cypher() {
        let store = InMemoryGraphStore::new();
        let u = update(
            "MATCH (a),(b) ...",
            &[
                ("from", "u1".into()),
                ("to", "u2".into()),
                ("delta", 0.1.into()),
            ],
        );
        store.apply_updates(&[u]).await.unwrap();
        assert!((store.relationship_delta("u1", "u2", "REL").unwrap() - 0.1).abs() < 0.001);
    }
}
