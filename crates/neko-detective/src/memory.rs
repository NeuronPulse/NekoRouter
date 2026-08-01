use async_trait::async_trait;
use neko_core::{GroupId, MemoryRecord, NekoError, VectorStore};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::debug;

/// In-memory vector store fallback.
///
/// Stores records without computing embeddings and performs simple keyword
/// search filtered by `group_id`. It is intended for local development or
/// integration tests where a real Qdrant instance is unavailable. Semantic
/// similarity is replaced by substring matching on whitespace-separated query
/// tokens.
#[derive(Debug, Default, Clone)]
pub struct InMemoryVectorStore {
    records: Arc<Mutex<HashMap<String, MemoryRecord>>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn embed_and_upsert(&self, records: &[MemoryRecord]) -> Result<(), NekoError> {
        let mut store = self.records.lock().unwrap();
        for record in records {
            store.insert(record.id.clone(), record.clone());
        }
        debug!("in-memory vector store now holds {} records", store.len());
        Ok(())
    }

    async fn search(
        &self,
        group_id: &GroupId,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<MemoryRecord>, NekoError> {
        let scored = self.search_with_score(group_id, query_text, top_k).await?;
        Ok(scored.into_iter().map(|(_, record)| record).collect())
    }

    async fn search_with_score(
        &self,
        group_id: &GroupId,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(f32, MemoryRecord)>, NekoError> {
        let store = self.records.lock().unwrap();

        let query_lower = query_text.to_lowercase();
        let query_words: Vec<String> = query_lower
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        let mut matched: Vec<(f32, MemoryRecord)> = store
            .values()
            .filter(|record| record.group_id == *group_id)
            .filter(|record| {
                if query_words.is_empty() {
                    return true;
                }
                let text = record.text.to_lowercase();
                query_words.iter().any(|word| text.contains(word))
            })
            .map(|record| {
                // Crude similarity for the in-memory fallback: exact match or
                // full containment scores 1.0, partial word overlap scores 0.5.
                let text_lower = record.text.to_lowercase();
                let score = if text_lower == query_lower
                    || text_lower.contains(&query_lower)
                    || query_lower.contains(&text_lower)
                {
                    1.0
                } else {
                    0.5
                };
                (score, record.clone())
            })
            .collect();

        matched.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matched.truncate(top_k);

        Ok(matched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn record(group_id: &str, text: &str) -> MemoryRecord {
        MemoryRecord {
            id: Uuid::new_v4().to_string(),
            group_id: group_id.to_string(),
            speaker_id: "u1".to_string(),
            target_id: None,
            text: text.to_string(),
            timestamp: Utc::now(),
            relation_delta: None,
            tags: vec![],
            layer: "test".to_string(),
        }
    }

    #[tokio::test]
    async fn stores_and_retrieves_records_by_keyword() {
        let store = InMemoryVectorStore::new();
        let r1 = record("g1", "Alice likes cats");
        let r2 = record("g1", "Bob likes dogs");
        let r3 = record("g2", "Alice likes fish");

        store
            .embed_and_upsert(&[r1.clone(), r2.clone(), r3.clone()])
            .await
            .unwrap();

        let results = store.search(&"g1".to_string(), "cats", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Alice likes cats");

        let results = store.search(&"g2".to_string(), "likes", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Alice likes fish");
    }

    #[tokio::test]
    async fn search_filters_by_group_id() {
        let store = InMemoryVectorStore::new();
        let r1 = record("g1", "hello");
        let r2 = record("g2", "hello");

        store.embed_and_upsert(&[r1, r2]).await.unwrap();

        let results = store.search(&"g1".to_string(), "hello", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].group_id, "g1");
    }

    #[tokio::test]
    async fn empty_query_returns_all_group_records() {
        let store = InMemoryVectorStore::new();
        let r1 = record("g1", "alpha");
        let r2 = record("g1", "beta");

        store.embed_and_upsert(&[r1, r2]).await.unwrap();

        let results = store.search(&"g1".to_string(), "", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn respects_top_k() {
        let store = InMemoryVectorStore::new();
        let r1 = record("g1", "hello world");
        let r2 = record("g1", "hello again");

        store.embed_and_upsert(&[r1, r2]).await.unwrap();

        let results = store.search(&"g1".to_string(), "hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn upsert_overwrites_same_id() {
        let store = InMemoryVectorStore::new();
        let mut r = record("g1", "old");
        store.embed_and_upsert(&[r.clone()]).await.unwrap();

        r.text = "new".to_string();
        store.embed_and_upsert(&[r.clone()]).await.unwrap();

        let results = store.search(&"g1".to_string(), "", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "new");
    }

    #[tokio::test]
    async fn search_with_score_ranks_exact_matches_highest() {
        let store = InMemoryVectorStore::new();
        let r1 = record("g1", "hello world");
        let r2 = record("g1", "goodbye");

        store.embed_and_upsert(&[r1.clone(), r2]).await.unwrap();

        let scored = store
            .search_with_score(&"g1".to_string(), "hello world", 10)
            .await
            .unwrap();
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].0, 1.0);
        assert_eq!(scored[0].1.text, "hello world");
    }
}
