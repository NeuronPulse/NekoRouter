use crate::types::{
    AffectiveState, ChatMessage, GraphUpdate, GroupId, LlmRequest, LlmResponse, MemoryRecord,
    MessageId, ReplyOut, UserId,
};
use crate::NekoError;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, watch};

/// Abstract LLM client. Implementations decide the transport and provider.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, NekoError>;
}

/// Abstract embedding client. Implementations decide the transport and model.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Embed a batch of texts into vectors.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NekoError>;

    /// Vector dimension produced by this model.
    fn vector_dim(&self) -> usize;
}

/// Stores chat history and supports context retrieval.
#[async_trait]
pub trait HistoryStore: Send + Sync {
    async fn append_batch(&self, messages: &[ChatMessage]) -> Result<(), NekoError>;

    async fn query_context(
        &self,
        group_id: &GroupId,
        user_id: Option<&UserId>,
        before: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<ChatMessage>, NekoError>;

    /// Load persisted per-(group, user) affective states. Defaults to nothing
    /// for stores that do not support state persistence.
    async fn load_affective_states(
        &self,
    ) -> Result<Vec<(GroupId, UserId, AffectiveState)>, NekoError> {
        Ok(Vec::new())
    }

    /// Persist affective states so they survive restarts. Defaults to a no-op.
    async fn save_affective_states(
        &self,
        _states: &[(GroupId, UserId, AffectiveState)],
    ) -> Result<(), NekoError> {
        Ok(())
    }

    /// Mark messages as processed (flushed to the store). Defaults to a no-op.
    async fn mark_processed(&self, _ids: &[MessageId]) -> Result<(), NekoError> {
        Ok(())
    }
}

/// Vector memory store (Qdrant or similar).
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn embed_and_upsert(&self, records: &[MemoryRecord]) -> Result<(), NekoError>;

    async fn search(
        &self,
        group_id: &GroupId,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<MemoryRecord>, NekoError>;
}

/// Graph store (Neo4j or similar).
#[async_trait]
pub trait GraphStore: Send + Sync {
    async fn apply_updates(&self, updates: &[GraphUpdate]) -> Result<(), NekoError>;

    /// Produce a short human-readable summary of the strongest stored
    /// relationships. Used by solidify to refresh the council's long-term
    /// memory. Defaults to an empty summary.
    async fn relationship_summary(&self, _limit: usize) -> Result<String, NekoError> {
        Ok(String::new())
    }
}

/// Ingress adapter that feeds incoming events into the router.
#[async_trait]
pub trait Ingress: Send + Sync {
    async fn run(
        self,
        out: mpsc::Sender<crate::Event>,
        shutdown: watch::Receiver<bool>,
    ) -> Result<(), NekoError>;
}

/// Egress adapter that sends replies back to the chat platform.
#[async_trait]
pub trait Egress: Send + Sync {
    async fn send(&self, reply: ReplyOut) -> Result<(), NekoError>;
}
