use crate::{parser, prompt};
use chrono::Utc;
use neko_core::{
    DetectiveInput, DetectiveReport, Event, HistoryStore, LlmClient, LlmMessage, LlmRequest,
    LlmRole, NekoError, ReplyOut, ResponseFormat, VectorStore,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct DetectiveConfig {
    pub history_limit: usize,
    pub memory_top_k: usize,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,
}

impl Default for DetectiveConfig {
    fn default() -> Self {
        Self {
            history_limit: 20,
            memory_top_k: 5,
            llm_temperature: 0.7,
            llm_max_tokens: 1024,
        }
    }
}

/// Layer 4 actor: emotionless data machine that retrieves context and produces
/// a structured, dehydrated JSON report.
pub struct DetectiveActor<C: LlmClient> {
    config: DetectiveConfig,
    llm: Arc<C>,
    history_store: Arc<dyn HistoryStore>,
    vector_store: Arc<dyn VectorStore>,
}

impl<C: LlmClient + 'static> DetectiveActor<C> {
    pub fn new(
        config: DetectiveConfig,
        llm: C,
        history_store: Arc<dyn HistoryStore>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self {
            config,
            llm: Arc::new(llm),
            history_store,
            vector_store,
        }
    }

    pub async fn run(
        self,
        mut inbound: mpsc::Receiver<Event>,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        info!("detective actor started");

        while let Some(event) = inbound.recv().await {
            if let Event::DetectiveRequest(req) = event {
                let this = self.clone();
                let out = out.clone();
                tokio::spawn(async move {
                    if let Err(e) = this.handle_request(req, out).await {
                        warn!("detective handle error: {e}");
                    }
                });
            }
        }

        info!("detective actor stopped");
        Ok(())
    }

    async fn handle_request(
        &self,
        req: DetectiveInput,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        debug!("detective investigating {}", req.target_user);

        let history = self
            .history_store
            .query_context(
                &req.message.group_id,
                Some(&req.target_user),
                Utc::now(),
                self.config.history_limit,
            )
            .await
            .unwrap_or_default();

        let memory = self
            .vector_store
            .search(
                &req.message.group_id,
                &req.message.content,
                self.config.memory_top_k,
            )
            .await
            .unwrap_or_default();

        let prompt = prompt::detective_prompt(&req, &history, &memory);
        let llm_req = LlmRequest {
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: prompt,
            }],
            temperature: self.config.llm_temperature,
            max_tokens: Some(self.config.llm_max_tokens),
            response_format: Some(ResponseFormat::JsonObject),
        };

        let report = match self.llm.complete(llm_req).await {
            Ok(resp) => parser::parse_detective_report(&resp.content)?,
            Err(e) => {
                warn!("detective llm failed: {e}, producing empty report");
                DetectiveReport {
                    target_user: req.target_user.clone(),
                    ..Default::default()
                }
            }
        };

        out.send(Event::DetectiveReport(report.clone())).await.ok();

        // If the report has high confidence, generate a reply directly.
        if report.confidence >= 0.5 {
            let tone_hint = match report.recommended_tone {
                neko_core::Tone::Warm => "温柔地",
                neko_core::Tone::Cold => "冷淡地",
                neko_core::Tone::Playful => "俏皮地",
                neko_core::Tone::Sarcastic => "带讽刺地",
                neko_core::Tone::Cautious => "谨慎地",
                neko_core::Tone::Neutral => "",
            };
            let reply = format!("{} {}", tone_hint, report.summary);
            out.send(Event::ReplyOut(ReplyOut {
                id: uuid::Uuid::new_v4(),
                reply_to: req.message.id,
                group_id: req.message.group_id,
                target_user: req.target_user,
                content: reply.trim().to_string(),
                layer: "detective".to_string(),
            }))
            .await
            .map_err(|_| NekoError::transport("router channel closed"))?;
        }

        Ok(())
    }
}

impl<C: LlmClient> Clone for DetectiveActor<C> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            llm: self.llm.clone(),
            history_store: self.history_store.clone(),
            vector_store: self.vector_store.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::{DateTime, Utc};
    use neko_core::{
        AffectiveState, ChatMessage, DetectiveInput, Event, FinishReason, HistoryStore, LlmClient,
        LlmRequest, LlmResponse, MemoryRecord, ReplyOut, TokenUsage, VectorStore,
    };
    use std::sync::Mutex;
    use uuid::Uuid;

    struct MockLlm {
        responses: Mutex<Vec<String>>,
    }

    impl MockLlm {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse, NekoError> {
            let content = self
                .responses
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| "{}".to_string());
            Ok(LlmResponse {
                content,
                usage: TokenUsage::default(),
                finish_reason: FinishReason::Stop,
            })
        }
    }

    #[derive(Default, Clone)]
    struct MockHistory {
        context: Vec<ChatMessage>,
    }

    #[async_trait]
    impl HistoryStore for MockHistory {
        async fn append_batch(&self, _messages: &[ChatMessage]) -> Result<(), NekoError> {
            Ok(())
        }

        async fn query_context(
            &self,
            _group_id: &neko_core::GroupId,
            _user_id: Option<&neko_core::UserId>,
            _before: DateTime<Utc>,
            _limit: usize,
        ) -> Result<Vec<ChatMessage>, NekoError> {
            Ok(self.context.clone())
        }
    }

    #[derive(Default)]
    struct MockVectorStore {
        records: Mutex<Vec<MemoryRecord>>,
    }

    #[async_trait]
    impl VectorStore for MockVectorStore {
        async fn embed_and_upsert(&self, records: &[MemoryRecord]) -> Result<(), NekoError> {
            self.records.lock().unwrap().extend_from_slice(records);
            Ok(())
        }

        async fn search(
            &self,
            _group_id: &neko_core::GroupId,
            _query_text: &str,
            _top_k: usize,
        ) -> Result<Vec<MemoryRecord>, NekoError> {
            Ok(vec![])
        }
    }

    fn make_message(content: &str) -> ChatMessage {
        ChatMessage {
            id: Uuid::new_v4(),
            group_id: "12345".to_string(),
            sender: "67890".to_string(),
            nickname: "Alice".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            reply_to: None,
            raw_payload: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn high_confidence_produces_report_and_reply() {
        let (tx, rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let report = DetectiveReport {
            target_user: "67890".to_string(),
            summary: "用户喜欢直接表达。".to_string(),
            confidence: 0.8,
            recommended_tone: neko_core::Tone::Playful,
            ..Default::default()
        };
        let llm = MockLlm::new(vec![serde_json::to_string(&report).unwrap()]);

        let actor = DetectiveActor::new(
            DetectiveConfig::default(),
            llm,
            Arc::new(MockHistory::default()),
            Arc::new(MockVectorStore::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(rx, out_tx).await;
        });

        tx.send(Event::DetectiveRequest(DetectiveInput {
            message: make_message("你怎么看"),
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

        let event = out_rx.recv().await.expect("expected detective report");
        assert!(matches!(event, Event::DetectiveReport(_)));

        let event = out_rx.recv().await.expect("expected reply");
        match event {
            Event::ReplyOut(ReplyOut { content, layer, .. }) => {
                assert_eq!(layer, "detective");
                assert!(content.contains("用户喜欢直接表达"));
                assert!(content.starts_with("俏皮地"));
            }
            other => panic!("expected ReplyOut, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn low_confidence_produces_only_report() {
        let (tx, rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let report = DetectiveReport {
            target_user: "67890".to_string(),
            summary: "不确定。".to_string(),
            confidence: 0.3,
            recommended_tone: neko_core::Tone::Neutral,
            ..Default::default()
        };
        let llm = MockLlm::new(vec![serde_json::to_string(&report).unwrap()]);

        let actor = DetectiveActor::new(
            DetectiveConfig::default(),
            llm,
            Arc::new(MockHistory::default()),
            Arc::new(MockVectorStore::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(rx, out_tx).await;
        });

        tx.send(Event::DetectiveRequest(DetectiveInput {
            message: make_message("?"),
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

        let event = out_rx.recv().await.expect("expected detective report");
        assert!(matches!(event, Event::DetectiveReport(_)));

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(200), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }

    #[tokio::test]
    async fn llm_failure_produces_empty_report() {
        let (tx, rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        struct FailingLlm;
        #[async_trait]
        impl LlmClient for FailingLlm {
            async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse, NekoError> {
                Err(NekoError::llm("mock failure"))
            }
        }

        let actor = DetectiveActor::new(
            DetectiveConfig::default(),
            FailingLlm,
            Arc::new(MockHistory::default()),
            Arc::new(MockVectorStore::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(rx, out_tx).await;
        });

        tx.send(Event::DetectiveRequest(DetectiveInput {
            message: make_message("?"),
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

        let event = out_rx.recv().await.expect("expected empty report");
        match event {
            Event::DetectiveReport(report) => {
                assert_eq!(report.target_user, "67890");
                assert!(report.summary.is_empty());
            }
            other => panic!("expected DetectiveReport, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parse_failure_is_graceful() {
        let (tx, rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = DetectiveActor::new(
            DetectiveConfig::default(),
            MockLlm::new(vec!["not json".to_string()]),
            Arc::new(MockHistory::default()),
            Arc::new(MockVectorStore::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(rx, out_tx).await;
        });

        tx.send(Event::DetectiveRequest(DetectiveInput {
            message: make_message("?"),
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(300), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }
}
