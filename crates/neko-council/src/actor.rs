use crate::{parser, prompt};
use chrono::Utc;
use neko_core::{
    CouncilAction, CouncilInput, DetectiveInput, Event, HistoryStore, LlmClient, LlmMessage,
    LlmRequest, LlmRole, NekoError, ReplyOut, ResponseFormat,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct CouncilConfig {
    pub context_limit: usize,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,
}

impl Default for CouncilConfig {
    fn default() -> Self {
        Self {
            context_limit: 10,
            llm_temperature: 0.9,
            llm_max_tokens: 512,
        }
    }
}

/// Layer 3 actor: Mind Council.
pub struct CouncilActor<C: LlmClient> {
    config: CouncilConfig,
    llm: Arc<C>,
    history_store: Arc<dyn HistoryStore>,
}

impl<C: LlmClient + 'static> CouncilActor<C> {
    pub fn new(config: CouncilConfig, llm: C, history_store: Arc<dyn HistoryStore>) -> Self {
        Self {
            config,
            llm: Arc::new(llm),
            history_store,
        }
    }

    pub async fn run(
        self,
        mut inbound: mpsc::Receiver<Event>,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        info!("council actor started");

        while let Some(event) = inbound.recv().await {
            if let Event::Escalation(_, msg, state) = event {
                let this = self.clone();
                let out = out.clone();
                tokio::spawn(async move {
                    if let Err(e) = this.handle_escalation(msg, state, out).await {
                        warn!("council handle error: {e}");
                    }
                });
            }
        }

        info!("council actor stopped");
        Ok(())
    }

    async fn handle_escalation(
        &self,
        msg: neko_core::ChatMessage,
        state: neko_core::AffectiveState,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        debug!("council processing escalation from {}", msg.sender);

        let context = self
            .history_store
            .query_context(
                &msg.group_id,
                Some(&msg.sender),
                Utc::now(),
                self.config.context_limit,
            )
            .await
            .unwrap_or_default();

        let input = CouncilInput {
            message: msg.clone(),
            state,
            context,
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
                    group_id: msg.group_id,
                    target_user: msg.sender,
                    content: reply,
                    layer: "council".to_string(),
                }))
                .await
                .map_err(|_| NekoError::transport("router channel closed"))?;
            }
            CouncilAction::LaunchDetective => {
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
}

impl<C: LlmClient> Clone for CouncilActor<C> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            llm: self.llm.clone(),
            history_store: self.history_store.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use neko_core::{
        AffectiveState, ChatMessage, CouncilAction, DetectiveInput, EscalationReason, Event,
        FinishReason, LlmClient, LlmRequest, LlmResponse, ReplyOut, TokenUsage,
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
            ))
            .await
            .unwrap();

        let timeout =
            tokio::time::timeout(std::time::Duration::from_millis(300), out_rx.recv()).await;
        assert!(timeout.is_err() || timeout.unwrap().is_none());
    }
}
