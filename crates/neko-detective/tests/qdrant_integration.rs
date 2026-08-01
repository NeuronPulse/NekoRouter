use async_trait::async_trait;
use chrono::Utc;
use neko_core::{EmbeddingClient, MemoryRecord, NekoError, VectorStore};
use neko_detective::QdrantVectorStore;
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

const QDRANT_URL: &str = "http://127.0.0.1:6334";

/// Deterministic, dimension-8 embedding: characters hash into vector bins.
/// Cosine similarity between two texts is 1.0 exactly when they are identical.
struct MockEmbedding;

#[async_trait]
impl EmbeddingClient for MockEmbedding {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NekoError> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; 8];
                for (i, b) in t.bytes().enumerate() {
                    v[i % 8] += b as f32;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect())
    }

    fn vector_dim(&self) -> usize {
        8
    }
}

fn qdrant_reachable() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:6334".parse().unwrap(),
        Duration::from_millis(500),
    )
    .is_ok()
}

fn sample_record(id: &str, group: &str, text: &str) -> MemoryRecord {
    MemoryRecord {
        id: id.to_string(),
        group_id: group.to_string(),
        speaker_id: "67890".to_string(),
        target_id: None,
        text: text.to_string(),
        timestamp: Utc::now(),
        relation_delta: None,
        tags: vec![],
        layer: "test".to_string(),
    }
}

#[tokio::test]
async fn qdrant_upserts_and_searches_by_group() {
    if !qdrant_reachable() {
        eprintln!("skipping: qdrant not reachable at {QDRANT_URL}");
        return;
    }

    let collection = format!("test_qdrant_{}", uuid::Uuid::new_v4().simple());
    let store = QdrantVectorStore::new(
        QDRANT_URL,
        None::<String>,
        &collection,
        Arc::new(MockEmbedding),
    )
    .unwrap();

    let id_a = uuid::Uuid::new_v4().to_string();
    let id_b = uuid::Uuid::new_v4().to_string();
    let records = vec![
        sample_record(&id_a, "g1", "loves cats"),
        sample_record(&id_b, "g1", "hates storms"),
    ];
    store.embed_and_upsert(&records).await.unwrap();

    let g1 = "g1".to_string();
    let g2 = "g2".to_string();

    // Same text as record "a" ranks it first.
    let hits = store.search(&g1, "loves cats", 1).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id_a);
    assert_eq!(hits[0].text, "loves cats");

    // Group filter: nothing stored for g2.
    let other = store.search(&g2, "loves cats", 5).await.unwrap();
    assert!(other.is_empty());

    // The most relevant record comes back first for the group.
    let all = store.search(&g1, "loves cats", 5).await.unwrap();
    assert_eq!(all[0].id, id_a);

    // Clean up the throwaway collection.
    let client = qdrant_client::Qdrant::from_url(QDRANT_URL).build().unwrap();
    let _ = client.delete_collection(&collection).await;
}
