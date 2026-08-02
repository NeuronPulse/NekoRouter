use crate::{parser, prompt};
use chrono::Utc;
use neko_core::{
    DetectiveInput, DetectiveReport, Event, HistoryStore, LlmClient, LlmMessage, LlmRequest,
    LlmRole, MemoryDecision, NekoError, ResponseFormat, TopicBurst, VectorStore,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, info_span, warn, Instrument};

#[derive(Debug, Clone)]
pub struct DetectiveConfig {
    pub history_limit: usize,
    pub memory_top_k: usize,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,
    /// Minimum cosine similarity for a learned fact to be considered a
    /// duplicate of an existing memory record and skipped.
    pub fact_dedup_threshold: f32,
}

impl Default for DetectiveConfig {
    fn default() -> Self {
        Self {
            history_limit: 20,
            memory_top_k: 5,
            llm_temperature: 0.7,
            llm_max_tokens: 1024,
            fact_dedup_threshold: 0.92,
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
            match event {
                Event::DetectiveRequest(req) => {
                    let this = self.clone();
                    let out = out.clone();
                    let trace_id = req.message.trace_id;
                    tokio::spawn(
                        async move {
                            if let Err(e) = this.handle_request(req, out).await {
                                warn!("detective handle error: {e}");
                            }
                        }
                        .instrument(info_span!("detective_request", trace_id = %trace_id)),
                    );
                }
                Event::TopicBurst(burst) => {
                    let this = self.clone();
                    let out = out.clone();
                    tokio::spawn(async move {
                        if let Err(e) = this.handle_burst(burst, out).await {
                            warn!("detective burst handle error: {e}");
                        }
                    });
                }
                _ => {}
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

        // Correlate the report with the message that triggered it so the
        // council can turn it into a final reply and solidify can persist it.
        let mut report = report;
        report.message_id = req.message.id;
        report.trace_id = req.message.trace_id;
        report.group_id = req.message.group_id.clone();
        report.target_user = req.target_user.clone();

        // Learn from the report: persist high-value facts back into vector
        // memory so future investigations of this user recall them.
        self.persist_facts(&report).await;

        out.send(Event::DetectiveReport(report)).await.ok();
        Ok(())
    }

    /// Write the report's historical facts into the vector store as memory
    /// records, closing the learning loop (facts from one investigation are
    /// retrieved by the next).
    ///
    /// Facts that are too similar to an existing memory record for the same
    /// group are skipped to avoid duplicate entries.
    async fn persist_facts(&self, report: &DetectiveReport) {
        let mut facts: Vec<neko_core::MemoryRecord> = Vec::new();

        for f in report
            .historical_facts
            .iter()
            .filter(|f| !f.text.trim().is_empty())
        {
            match self
                .vector_store
                .search_with_score(&report.group_id, &f.text, 1)
                .await
            {
                Ok(scored)
                    if scored
                        .first()
                        .map(|(s, _)| *s >= self.config.fact_dedup_threshold)
                        .unwrap_or(false) =>
                {
                    debug!("skipping duplicate fact: {}", f.text);
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    warn!("fact dedup search failed: {e}, persisting anyway");
                }
            }

            facts.push(neko_core::MemoryRecord {
                id: uuid::Uuid::new_v4().to_string(),
                group_id: report.group_id.clone(),
                speaker_id: report.target_user.clone(),
                target_id: None,
                text: f.text.clone(),
                timestamp: f.occurred_at.unwrap_or_else(Utc::now),
                relation_delta: None,
                tags: vec!["fact".to_string(), "detective".to_string()],
                layer: "detective".to_string(),
            });
        }

        if facts.is_empty() {
            return;
        }

        match self.vector_store.embed_and_upsert(&facts).await {
            Ok(()) => debug!("persisted {} learned facts", facts.len()),
            Err(e) => warn!("failed to persist learned facts: {e}"),
        }
    }

    /// Analyze a hot conversation window and produce a structured memory decision.
    async fn handle_burst(
        &self,
        burst: TopicBurst,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        debug!(
            "detective curating memory for group {} ({} messages, {:.1} mpm)",
            burst.group_id,
            burst.messages.len(),
            burst.score.messages_per_minute
        );

        let prompt = prompt::memory_curator_prompt(&burst);
        let llm_req = LlmRequest {
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: prompt,
            }],
            temperature: self.config.llm_temperature,
            max_tokens: Some(self.config.llm_max_tokens),
            response_format: Some(ResponseFormat::JsonObject),
        };

        let decision = match self.llm.complete(llm_req).await {
            Ok(resp) => match parser::parse_memory_decision(&resp.content) {
                Ok(mut d) => {
                    d.group_id = burst.group_id;
                    d
                }
                Err(e) => {
                    warn!("detective failed to parse memory decision: {e}");
                    MemoryDecision {
                        group_id: burst.group_id,
                        summary: String::new(),
                        updates: vec![],
                    }
                }
            },
            Err(e) => {
                warn!("detective llm failed during burst curation: {e}");
                MemoryDecision {
                    group_id: burst.group_id,
                    summary: String::new(),
                    updates: vec![],
                }
            }
        };

        if !decision.updates.is_empty() {
            out.send(Event::MemoryDecision(decision))
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
    use crate::InMemoryVectorStore;
    use async_trait::async_trait;
    use chrono::{DateTime, Utc};
    use neko_core::{
        AffectiveState, ChatMessage, DetectiveInput, Event, Fact, FinishReason, HistoryStore,
        LlmClient, LlmRequest, LlmResponse, MemoryRecord, TokenUsage, VectorStore,
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

    #[tokio::test]
    async fn report_is_correlated_and_no_direct_reply() {
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

        let msg = make_message("你怎么看");
        let message_id = msg.id;
        tx.send(Event::DetectiveRequest(DetectiveInput {
            message: msg,
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

        // The report carries the triggering message id/group and no direct
        // reply is emitted — the council decides the final reply.
        let event = out_rx.recv().await.expect("expected detective report");
        match event {
            Event::DetectiveReport(report) => {
                assert_eq!(report.message_id, message_id);
                assert_eq!(report.group_id, "12345");
                assert_eq!(report.target_user, "67890");
                assert_eq!(report.confidence, 0.8);
            }
            other => panic!("expected DetectiveReport, got {other:?}"),
        }

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(200), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
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
            tokio::time::timeout(std::time::Duration::from_millis(300), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }

    #[tokio::test]
    async fn report_facts_are_learned_into_vector_store() {
        let (tx, rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let report = DetectiveReport {
            target_user: "67890".to_string(),
            summary: "确认过。".to_string(),
            historical_facts: vec![
                neko_core::Fact {
                    text: "用户养了一只叫咪咪的猫。".to_string(),
                    evidence: vec!["2026-08-01 消息".to_string()],
                    occurred_at: None,
                },
                neko_core::Fact {
                    text: "   ".to_string(),
                    evidence: vec![],
                    occurred_at: None,
                },
            ],
            confidence: 0.9,
            recommended_tone: neko_core::Tone::Warm,
            ..Default::default()
        };
        let llm = MockLlm::new(vec![serde_json::to_string(&report).unwrap()]);

        let vector_store = Arc::new(MockVectorStore::default());
        let actor = DetectiveActor::new(
            DetectiveConfig::default(),
            llm,
            Arc::new(MockHistory::default()),
            vector_store.clone(),
        );

        tokio::spawn(async move {
            let _ = actor.run(rx, out_tx).await;
        });

        tx.send(Event::DetectiveRequest(DetectiveInput {
            message: make_message("你了解我吗？"),
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

        let event = out_rx.recv().await.expect("expected detective report");
        assert!(matches!(event, Event::DetectiveReport(_)));

        // Only the non-blank fact is persisted, tagged as a learned fact.
        let records = vector_store.records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].text, "用户养了一只叫咪咪的猫。");
        assert_eq!(records[0].speaker_id, "67890");
        assert!(records[0].tags.contains(&"fact".to_string()));
    }

    #[tokio::test]
    async fn duplicate_facts_are_not_persisted_twice() {
        let (tx, rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let vector_store = Arc::new(InMemoryVectorStore::new());
        vector_store
            .embed_and_upsert(&[MemoryRecord {
                id: Uuid::new_v4().to_string(),
                group_id: "12345".to_string(),
                speaker_id: "67890".to_string(),
                target_id: None,
                text: "用户养了一只叫咪咪的猫。".to_string(),
                timestamp: Utc::now(),
                relation_delta: None,
                tags: vec!["fact".to_string()],
                layer: "detective".to_string(),
            }])
            .await
            .unwrap();

        let report = DetectiveReport {
            target_user: "67890".to_string(),
            summary: "用户有宠物。".to_string(),
            confidence: 0.9,
            historical_facts: vec![Fact {
                text: "用户养了一只叫咪咪的猫。".to_string(),
                ..Default::default()
            }],
            recommended_tone: neko_core::Tone::Warm,
            ..Default::default()
        };
        let llm = MockLlm::new(vec![serde_json::to_string(&report).unwrap()]);

        let actor = DetectiveActor::new(
            DetectiveConfig::default(),
            llm,
            Arc::new(MockHistory::default()),
            vector_store.clone(),
        );

        tokio::spawn(async move {
            let _ = actor.run(rx, out_tx).await;
        });

        tx.send(Event::DetectiveRequest(DetectiveInput {
            message: make_message("你了解我吗？"),
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

        let event = out_rx.recv().await.expect("expected detective report");
        assert!(matches!(event, Event::DetectiveReport(_)));

        // The duplicate fact is skipped, so the count stays at 1.
        let records = vector_store
            .search(&"12345".to_string(), "", 10)
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
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
