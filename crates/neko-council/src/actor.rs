use crate::{parser, prompt};
use chrono::Utc;
use dashmap::DashMap;
use neko_core::{
    CouncilAction, CouncilInput, DetectiveInput, DetectiveReport, Event, HistoryStore, LlmClient,
    LlmMessage, LlmRequest, LlmRole, MessageId, NekoError, ReplyOut, ResponseFormat,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, info_span, warn, Instrument};

#[derive(Debug, Clone)]
pub struct CouncilConfig {
    pub context_limit: usize,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,
    /// How long an escalation may wait for its detective report before the
    /// council gives up on it (the pending entry is swept).
    pub detective_timeout: Duration,
}

impl Default for CouncilConfig {
    fn default() -> Self {
        Self {
            context_limit: 10,
            llm_temperature: 0.9,
            llm_max_tokens: 512,
            detective_timeout: Duration::from_secs(300),
        }
    }
}

/// An escalation that was handed to the detective and is awaiting a report.
struct PendingDetective {
    message: neko_core::ChatMessage,
    inserted_at: Instant,
}

/// Layer 3 actor: Mind Council.
pub struct CouncilActor<C: LlmClient> {
    config: CouncilConfig,
    llm: Arc<C>,
    history_store: Arc<dyn HistoryStore>,
    /// Escalations that were handed off to the detective and await a report,
    /// keyed by the triggering message id. Behind an `Arc` so that per-event
    /// task clones share the same map (`DashMap::clone` would deep-copy).
    pending_detective: Arc<DashMap<MessageId, PendingDetective>>,
    /// The latest nightly graph summary, produced by solidify. Injected into
    /// the council prompt so long-term relationships influence replies.
    daily_context: Arc<std::sync::RwLock<Option<String>>>,
}

impl<C: LlmClient + 'static> CouncilActor<C> {
    pub fn new(config: CouncilConfig, llm: C, history_store: Arc<dyn HistoryStore>) -> Self {
        Self {
            config,
            llm: Arc::new(llm),
            history_store,
            pending_detective: Arc::new(DashMap::new()),
            daily_context: Arc::new(std::sync::RwLock::new(None)),
        }
    }

    pub async fn run(
        self,
        mut inbound: mpsc::Receiver<Event>,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        info!("council actor started");

        // Sweep stale detective escalations on a regular cadence so the map
        // cannot grow unbounded when the detective never reports back.
        let sweep_every = self.config.detective_timeout.min(Duration::from_secs(30));
        let mut sweeper = tokio::time::interval(sweep_every);
        sweeper.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = sweeper.tick() => self.sweep_expired(),
                event = inbound.recv() => {
                    let Some(event) = event else { break };
                    match event {
                        Event::Escalation(_, msg, state, engagement_type) => {
                            let this = self.clone();
                            let out = out.clone();
                            let trace_id = msg.trace_id;
                            tokio::spawn(
                                async move {
                                    if let Err(e) = this.handle_escalation(msg, state, engagement_type, out).await {
                                        warn!("council handle error: {e}");
                                    }
                                }
                                .instrument(info_span!("council_escalation", trace_id = %trace_id)),
                            );
                        }
                        Event::DetectiveReport(report) => {
                            let this = self.clone();
                            let out = out.clone();
                            let trace_id = report.trace_id;
                            tokio::spawn(
                                async move {
                                    if let Err(e) = this.handle_report(report, out).await {
                                        warn!("council report error: {e}");
                                    }
                                }
                                .instrument(info_span!("council_report", trace_id = %trace_id)),
                            );
                        }
                        Event::DailyContext(ctx) => {
                            self.store_daily_context(ctx);
                        }
                        _ => {}
                    }
                }
            }
        }

        info!("council actor stopped");
        Ok(())
    }

    /// Remove detective escalations whose reports never arrived in time.
    fn sweep_expired(&self) {
        let timeout = self.config.detective_timeout;
        self.pending_detective.retain(|_, p| {
            if p.inserted_at.elapsed() >= timeout {
                debug!(
                    "sweeping stale detective escalation from {}: {:?} (waited {:?})",
                    p.message.sender, p.message.content, timeout
                );
                false
            } else {
                true
            }
        });
    }

    /// Number of escalations still waiting for a detective report.
    pub fn pending_detective_count(&self) -> usize {
        self.pending_detective.len()
    }

    fn store_daily_context(&self, context: String) {
        let trimmed = context.trim().to_string();
        let mut slot = self.daily_context.write().unwrap();
        if trimmed.is_empty() {
            debug!("daily context cleared");
            *slot = None;
        } else {
            debug!("stored daily context ({} chars)", trimmed.chars().count());
            *slot = Some(trimmed);
        }
    }

    async fn handle_escalation(
        &self,
        msg: neko_core::ChatMessage,
        state: neko_core::AffectiveState,
        engagement_type: neko_core::EngagementType,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        debug!(
            "council processing escalation from {} ({:?})",
            msg.sender, engagement_type
        );

        let user_filter = match engagement_type {
            neko_core::EngagementType::PersonalReply => Some(&msg.sender),
            neko_core::EngagementType::AmbientJoin => None,
        };
        let context = self
            .history_store
            .query_context(
                &msg.group_id,
                user_filter,
                Utc::now(),
                self.config.context_limit,
            )
            .await
            .unwrap_or_default();

        let input = CouncilInput {
            message: msg.clone(),
            state,
            context,
            daily_context: self
                .daily_context
                .read()
                .unwrap()
                .clone()
                .unwrap_or_default(),
            engagement_type,
        };

        let council_prompt = prompt::council_prompt(&input, &input.context);
        let req = LlmRequest {
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: council_prompt,
            }],
            temperature: self.config.llm_temperature,
            max_tokens: Some(self.config.llm_max_tokens),
            response_format: Some(ResponseFormat::JsonObject),
        };

        let decision = match self.llm.complete(req).await {
            Ok(resp) => parser::parse_council_decision(&resp.content)?,
            Err(e) => {
                warn!("council llm failed: {e}, falling back to ignore");
                return Ok(());
            }
        };

        out.send(Event::CouncilDecision(decision.clone()))
            .await
            .ok();

        match decision.action {
            CouncilAction::ReplyDirectly => {
                let reply = decision.draft_reply.unwrap_or_else(|| "……".to_string());
                out.send(Event::ReplyOut(ReplyOut {
                    id: uuid::Uuid::new_v4(),
                    reply_to: msg.id,
                    reply_to_platform: None,
                    group_id: msg.group_id,
                    target_user: msg.sender,
                    content: reply,
                    layer: "council".to_string(),
                    trace_id: msg.trace_id,
                }))
                .await
                .map_err(|_| NekoError::transport("router channel closed"))?;
            }
            CouncilAction::LaunchDetective => {
                self.pending_detective.insert(
                    msg.id,
                    PendingDetective {
                        message: msg.clone(),
                        inserted_at: Instant::now(),
                    },
                );
                out.send(Event::DetectiveRequest(DetectiveInput {
                    message: msg,
                    state,
                    target_user: input.message.sender.clone(),
                }))
                .await
                .map_err(|_| NekoError::transport("router channel closed"))?;
            }
            CouncilAction::Ignore => {}
        }

        Ok(())
    }

    /// Turn a detective report into a final reply for the escalation that
    /// launched the detective. This closes the loop: the council stays the
    /// single authority that decides what is actually sent out.
    async fn handle_report(
        &self,
        report: DetectiveReport,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        if self.pending_detective.remove(&report.message_id).is_none() {
            debug!(
                "detective report for unknown message {}, ignoring",
                report.message_id
            );
            return Ok(());
        }

        debug!(
            "council reviewing detective report for {}",
            report.target_user
        );

        let Some(reply) = self.compose_detective_reply(&report) else {
            return Ok(());
        };

        out.send(Event::ReplyOut(ReplyOut {
            id: uuid::Uuid::new_v4(),
            reply_to: report.message_id,
            reply_to_platform: None,
            group_id: report.group_id,
            target_user: report.target_user,
            content: reply,
            layer: "council".to_string(),
            trace_id: report.trace_id,
        }))
        .await
        .map_err(|_| NekoError::transport("router channel closed"))?;

        Ok(())
    }

    /// Compose a final reply from the detective report. Returns `None` when
    /// the report is not confident enough to act on.
    fn compose_detective_reply(&self, report: &DetectiveReport) -> Option<String> {
        if report.confidence < 0.5 || report.summary.trim().is_empty() {
            return None;
        }

        let tone_hint = match report.recommended_tone {
            neko_core::Tone::Warm => "温柔地",
            neko_core::Tone::Cold => "冷淡地",
            neko_core::Tone::Playful => "俏皮地",
            neko_core::Tone::Sarcastic => "带讽刺地",
            neko_core::Tone::Cautious => "谨慎地",
            neko_core::Tone::Neutral => "",
        };
        let reply = format!("{} {}", tone_hint, report.summary);
        Some(reply.trim().to_string())
    }
}

impl<C: LlmClient> Clone for CouncilActor<C> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            llm: self.llm.clone(),
            history_store: self.history_store.clone(),
            pending_detective: self.pending_detective.clone(),
            daily_context: self.daily_context.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use neko_core::{
        AffectiveState, ChatMessage, CouncilAction, DetectiveInput, EngagementType,
        EscalationReason, Event, FinishReason, LlmClient, LlmRequest, LlmResponse, ReplyOut,
        TokenUsage,
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
            _before: chrono::DateTime<Utc>,
            _limit: usize,
        ) -> Result<Vec<ChatMessage>, NekoError> {
            Ok(self.context.clone())
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
    async fn reply_directly_emits_decision_and_reply() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = CouncilActor::new(
            CouncilConfig::default(),
            MockLlm::new(vec![serde_json::json!({
                "action": "reply",
                "reasoning": "打招呼",
                "draft_reply": "你好呀~"
            })
            .to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        council_tx
            .send(Event::Escalation(
                EscalationReason::NeedsContext,
                make_message("在吗"),
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        let event = out_rx.recv().await.expect("expected council decision");
        match event {
            Event::CouncilDecision(decision) => {
                assert_eq!(decision.action, CouncilAction::ReplyDirectly);
            }
            other => panic!("expected CouncilDecision, got {other:?}"),
        }

        let event = out_rx.recv().await.expect("expected reply");
        match event {
            Event::ReplyOut(ReplyOut { content, layer, .. }) => {
                assert_eq!(content, "你好呀~");
                assert_eq!(layer, "council");
            }
            other => panic!("expected ReplyOut, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn launch_detective_emits_decision_and_request() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = CouncilActor::new(
            CouncilConfig::default(),
            MockLlm::new(vec![serde_json::json!({
                "action": "detective",
                "reasoning": "需要更多上下文",
                "draft_reply": ""
            })
            .to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        let msg = make_message("你怎么看");
        let group_id = msg.group_id.clone();
        council_tx
            .send(Event::Escalation(
                EscalationReason::NeedsContext,
                msg,
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        let event = out_rx.recv().await.expect("expected council decision");
        assert!(matches!(event, Event::CouncilDecision(_)));

        let event = out_rx.recv().await.expect("expected detective request");
        match event {
            Event::DetectiveRequest(DetectiveInput { message, .. }) => {
                assert_eq!(message.group_id, group_id);
            }
            other => panic!("expected DetectiveRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn ignore_action_emits_only_decision() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = CouncilActor::new(
            CouncilConfig::default(),
            MockLlm::new(vec![serde_json::json!({
                "action": "ignore",
                "reasoning": "不相关",
                "draft_reply": ""
            })
            .to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        council_tx
            .send(Event::Escalation(
                EscalationReason::Other("spam".to_string()),
                make_message("广告"),
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        let event = out_rx.recv().await.expect("expected council decision");
        assert!(matches!(event, Event::CouncilDecision(_)));

        // No further output for ignore.
        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(200), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }

    #[tokio::test]
    async fn llm_error_is_graceful() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        // MockLlm returns invalid JSON, which causes parser to fail; the actor
        // logs the error and returns Ok, producing no output.
        let actor = CouncilActor::new(
            CouncilConfig::default(),
            MockLlm::new(vec!["not json".to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        council_tx
            .send(Event::Escalation(
                EscalationReason::NeedsContext,
                make_message("??"),
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(300), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }

    #[tokio::test]
    async fn detective_report_closes_loop_with_confident_reply() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = CouncilActor::new(
            CouncilConfig::default(),
            MockLlm::new(vec![serde_json::json!({
                "action": "detective",
                "reasoning": "需要更多上下文",
                "draft_reply": ""
            })
            .to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        let msg = make_message("你怎么看");
        let message_id = msg.id;
        council_tx
            .send(Event::Escalation(
                EscalationReason::NeedsContext,
                msg,
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        // Council first emits its decision, then the detective request.
        assert!(matches!(
            out_rx.recv().await.unwrap(),
            Event::CouncilDecision(_)
        ));
        let request = out_rx.recv().await.unwrap();
        let Event::DetectiveRequest(DetectiveInput { message, .. }) = request else {
            panic!("expected DetectiveRequest, got {request:?}");
        };
        assert_eq!(message.id, message_id);

        // The detective reports back; council composes the final reply.
        let report = neko_core::DetectiveReport {
            message_id,
            group_id: "12345".to_string(),
            target_user: "67890".to_string(),
            summary: "用户喜欢直接表达。".to_string(),
            confidence: 0.8,
            recommended_tone: neko_core::Tone::Playful,
            ..Default::default()
        };
        council_tx
            .send(Event::DetectiveReport(report))
            .await
            .unwrap();

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), out_rx.recv())
            .await
            .expect("timeout waiting for council reply")
            .expect("channel closed");
        match event {
            Event::ReplyOut(ReplyOut {
                content,
                layer,
                reply_to,
                ..
            }) => {
                assert_eq!(reply_to, message_id);
                assert_eq!(layer, "council");
                assert_eq!(content, "俏皮地 用户喜欢直接表达。");
            }
            other => panic!("expected ReplyOut, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn low_confidence_report_does_not_reply() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = CouncilActor::new(
            CouncilConfig::default(),
            MockLlm::new(vec![serde_json::json!({
                "action": "detective",
                "reasoning": "需要更多上下文",
                "draft_reply": ""
            })
            .to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        let msg = make_message("?");
        let message_id = msg.id;
        council_tx
            .send(Event::Escalation(
                EscalationReason::NeedsContext,
                msg,
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        assert!(matches!(
            out_rx.recv().await.unwrap(),
            Event::CouncilDecision(_)
        ));
        assert!(matches!(
            out_rx.recv().await.unwrap(),
            Event::DetectiveRequest(_)
        ));

        let report = neko_core::DetectiveReport {
            message_id,
            group_id: "12345".to_string(),
            target_user: "67890".to_string(),
            summary: "不确定。".to_string(),
            confidence: 0.2,
            ..Default::default()
        };
        council_tx
            .send(Event::DetectiveReport(report))
            .await
            .unwrap();

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(300), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }

    #[tokio::test]
    async fn stale_detective_escalations_are_swept() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = CouncilActor::new(
            CouncilConfig {
                detective_timeout: std::time::Duration::from_millis(50),
                ..CouncilConfig::default()
            },
            MockLlm::new(vec![serde_json::json!({
                "action": "detective",
                "reasoning": "需要更多上下文",
                "draft_reply": ""
            })
            .to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        let msg = make_message("你怎么看");
        let message_id = msg.id;
        council_tx
            .send(Event::Escalation(
                EscalationReason::NeedsContext,
                msg,
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        assert!(matches!(
            out_rx.recv().await.unwrap(),
            Event::CouncilDecision(_)
        ));
        assert!(matches!(
            out_rx.recv().await.unwrap(),
            Event::DetectiveRequest(_)
        ));

        // Wait well past the timeout so the sweeper removes the escalation.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        // A late report for the swept escalation must NOT produce a reply.
        let report = neko_core::DetectiveReport {
            message_id,
            group_id: "12345".to_string(),
            target_user: "67890".to_string(),
            summary: "用户喜欢直接表达。".to_string(),
            confidence: 0.9,
            recommended_tone: neko_core::Tone::Warm,
            ..Default::default()
        };
        council_tx
            .send(Event::DetectiveReport(report))
            .await
            .unwrap();

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(200), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }

    #[tokio::test]
    async fn daily_context_reaches_council_prompt() {
        let (council_tx, council_rx) = mpsc::channel(16);
        let (out_tx, mut out_rx) = mpsc::channel(16);

        let actor = CouncilActor::new(
            CouncilConfig::default(),
            MockLlm::new(vec![serde_json::json!({
                "action": "ignore",
                "reasoning": "无",
                "draft_reply": ""
            })
            .to_string()]),
            Arc::new(MockHistory::default()),
        );

        tokio::spawn(async move {
            let _ = actor.run(council_rx, out_tx).await;
        });

        // Solidify pushes its nightly summary into the council.
        council_tx
            .send(Event::DailyContext(
                "用户 A 与 用户 B 关系亲密。".to_string(),
            ))
            .await
            .unwrap();

        council_tx
            .send(Event::Escalation(
                EscalationReason::NeedsContext,
                make_message("在吗"),
                AffectiveState::default(),
                EngagementType::PersonalReply,
            ))
            .await
            .unwrap();

        // The escalation still resolves to a council decision.
        let decision = tokio::time::timeout(std::time::Duration::from_secs(1), out_rx.recv())
            .await
            .expect("timeout waiting for council decision")
            .expect("channel closed");
        assert!(matches!(decision, Event::CouncilDecision(_)));
    }
}
