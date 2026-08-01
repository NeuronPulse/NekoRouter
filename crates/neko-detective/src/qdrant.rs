use async_trait::async_trait;
use neko_core::{EmbeddingClient, GroupId, MemoryRecord, NekoError, VectorStore};
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, PointStruct, QueryPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use std::sync::Arc;
use tracing::{debug, info};

/// A real Qdrant-backed vector memory store.
///
/// Text is embedded via the configured `EmbeddingClient` before upsert or
/// search. The collection is created automatically on first upsert if it does
/// not already exist.
pub struct QdrantVectorStore {
    client: Qdrant,
    collection: String,
    embedding: Arc<dyn EmbeddingClient>,
}

impl QdrantVectorStore {
    pub fn new(
        url: impl AsRef<str>,
        api_key: Option<String>,
        collection: impl Into<String>,
        embedding: Arc<dyn EmbeddingClient>,
    ) -> Result<Self, NekoError> {
        let mut builder = Qdrant::from_url(url.as_ref());
        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }

        let client = builder
            .build()
            .map_err(|e| NekoError::config(format!("cannot build Qdrant client: {e}")))?;

        Ok(Self {
            client,
            collection: collection.into(),
            embedding,
        })
    }

    async fn ensure_collection(&self) -> Result<(), NekoError> {
        match self.client.collection_exists(&self.collection).await {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(e) => {
                return Err(NekoError::transport(format!(
                    "cannot check Qdrant collection existence: {e:?}"
                )))
            }
        }

        info!("creating Qdrant collection {}", self.collection);
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection).vectors_config(
                    VectorParamsBuilder::new(self.embedding.vector_dim() as u64, Distance::Cosine),
                ),
            )
            .await
            .map_err(|e| NekoError::transport(format!("cannot create Qdrant collection: {e:?}")))?;

        Ok(())
    }

    fn record_to_payload(record: &MemoryRecord) -> Result<Payload, NekoError> {
        serde_json::to_value(record)
            .map_err(|e| NekoError::parse(format!("cannot serialize memory record: {e}")))?
            .try_into()
            .map_err(|e| NekoError::parse(format!("cannot convert record to Qdrant payload: {e}")))
    }

    fn payload_to_record(payload: Payload) -> Result<MemoryRecord, NekoError> {
        let value = serde_json::Value::from(payload);
        serde_json::from_value(value)
            .map_err(|e| NekoError::parse(format!("cannot deserialize memory record: {e}")))
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn embed_and_upsert(&self, records: &[MemoryRecord]) -> Result<(), NekoError> {
        if records.is_empty() {
            return Ok(());
        }

        self.ensure_collection().await?;

        let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
        let embeddings = self.embedding.embed(&texts).await?;

        let points: Vec<PointStruct> = records
            .iter()
            .zip(embeddings.into_iter())
            .map(|(record, vector)| {
                let payload = Self::record_to_payload(record)?;
                Ok(PointStruct::new(record.id.clone(), vector, payload))
            })
            .collect::<Result<Vec<_>, NekoError>>()?;

        debug!("upserting {} points into {}", points.len(), self.collection);
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection, points))
            .await
            .map_err(|e| NekoError::transport(format!("Qdrant upsert failed: {e}")))?;

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
        self.ensure_collection().await?;

        let embedding = self.embedding.embed(&[query_text.to_string()]).await?;
        let query_vector = embedding
            .into_iter()
            .next()
            .ok_or_else(|| NekoError::llm("embedding returned empty vector"))?;

        let result = self
            .client
            .query(
                QueryPointsBuilder::new(&self.collection)
                    .query(query_vector)
                    .limit(top_k as u64)
                    .filter(Filter::all([Condition::matches(
                        "group_id",
                        group_id.clone(),
                    )]))
                    .with_payload(true),
            )
            .await
            .map_err(|e| NekoError::transport(format!("Qdrant query failed: {e}")))?;

        let mut records = Vec::with_capacity(result.result.len());
        for scored in result.result {
            let payload = Payload::from(scored.payload);
            match Self::payload_to_record(payload) {
                Ok(record) => records.push((scored.score, record)),
                Err(e) => debug!("skipping malformed Qdrant payload: {e}"),
            }
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use neko_core::{EmbeddingClient, MemoryRecord};
    use uuid::Uuid;

    struct MockEmbedding {
        dim: usize,
    }

    #[async_trait]
    impl EmbeddingClient for MockEmbedding {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NekoError> {
            Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
        }

        fn vector_dim(&self) -> usize {
            self.dim
        }
    }

    fn sample_record() -> MemoryRecord {
        MemoryRecord {
            id: Uuid::new_v4().to_string(),
            group_id: "12345".to_string(),
            speaker_id: "67890".to_string(),
            target_id: Some("11111".to_string()),
            text: "hello world".to_string(),
            timestamp: Utc::now(),
            relation_delta: None,
            tags: vec!["tag1".to_string()],
            layer: "detective".to_string(),
        }
    }

    #[test]
    fn record_payload_roundtrip() {
        let record = sample_record();
        let payload = QdrantVectorStore::record_to_payload(&record).unwrap();
        let decoded = QdrantVectorStore::payload_to_record(payload).unwrap();
        assert_eq!(record.id, decoded.id);
        assert_eq!(record.group_id, decoded.group_id);
        assert_eq!(record.text, decoded.text);
        assert_eq!(record.tags, decoded.tags);
    }

    #[tokio::test]
    async fn empty_upsert_returns_ok_without_embedding() {
        let store = QdrantVectorStore::new(
            "http://127.0.0.1:6333",
            None::<String>,
            "test_collection",
            Arc::new(MockEmbedding { dim: 4 }),
        )
        .unwrap();

        // With no records the method returns early, so no network call is made.
        store.embed_and_upsert(&[]).await.unwrap();
    }
}
