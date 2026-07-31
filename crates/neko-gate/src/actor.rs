use crate::heuristic::{heuristic_from_name, GateHeuristic};
use dashmap::DashMap;
use neko_core::{
    AffectiveState, ChatMessage, DropReason, EscalationReason, Event, FinishReason, GateDecision,
    GroupId, LlmClient, LlmMessage, LlmRequest, LlmRole, NekoError, ReplyOut, ResponseFormat,
    UserId,
};
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, info, warn};

/// Configuration for the gate layer.
#[derive(Debug, Clone)]
pub struct GateConfig {
    pub max_message_length: usize,
    pub max_cozy_words: usize,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,
    pub concurrency_limit: usize,
    /// Name of the heuristic strategy to use. Built-in names:
    /// - `"default"`: filter commands, URLs, and long messages.
    /// - `"escalate_all"`: always escalate (testing stub).
    pub heuristic: String,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            max_message_length: 800,
            max_cozy_words: 10,
            llm_temperature: 0.7,
            llm_max_tokens: 32,
            concurrency_limit: 8,
            heuristic: "default".to_string(),
        }
    }
}

/// Layer 2 actor: cheap-model gate.
pub struct GateActor<C: LlmClient> {
    config: GateConfig,
    llm: Arc<C>,
    semaphore: Arc<Semaphore>,
    states: DashMap<(GroupId, UserId), AffectiveState>,
    heuristic: Arc<dyn GateHeuristic>,
}

impl<C: LlmClient + 'static> GateActor<C> {
    pub fn new(config: GateConfig, llm: C) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.concurrency_limit));
        let heuristic = Arc::from(heuristic_from_name(&config.heuristic));
        Self {
            config,
            llm: Arc::new(llm),
            semaphore,
            states: DashMap::new(),
            heuristic,
        }
    }

    pub async fn run(
        self,
        mut inbound: mpsc::Receiver<Event>,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        info!("gate actor started");

        while let Some(event) = inbound.recv().await {
            match event {
                Event::IncomingMessage(msg) => {
                    let state = self
                        .states
                        .get(&(msg.group_id.clone(), msg.sender.clone()))
                        .map(|s| *s.value())
                        .unwrap_or_default();
                    let this = self.clone();
                    let out = out.clone();
                    tokio::spawn(async move {
                        if let Err(e) = this.handle_message(msg, state, out).await {
                            warn!("gate handle error: {e}");
                        }
                    });
                }
                Event::AffectiveUpdated(group_id, user_id, state) => {
                    self.states.insert((group_id, user_id), state);
                }
                _ => {}
            }
        }

        info!("gate actor stopped");
        Ok(())
    }

    async fn handle_message(
        &self,
        msg: ChatMessage,
        state: AffectiveState,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        debug!("gate processing message from {}", msg.sender);

        if msg.char_count() > self.config.max_message_length {
            out.send(Event::GateDecision(GateDecision::Drop(DropReason::TooLong)))
                .await
                .ok();
            return Ok(());
        }

        let decision = if self.should_cozy(&msg) {
            let _permit = self
                .semaphore
                .acquire()
                .await
                .map_err(|e| NekoError::other(format!("semaphore closed: {e}")))?;

            match self.generate_cozy(&msg).await {
                Ok(reply) => GateDecision::CozyReply(reply),
                Err(e) => {
                    warn!("cozy generation failed: {e}, escalating");
                    GateDecision::Escalate(EscalationReason::NeedsContext)
                }
            }
        } else {
            GateDecision::Escalate(EscalationReason::NeedsContext)
        };

        out.send(Event::GateDecision(decision.clone())).await.ok();

        if let GateDecision::CozyReply(reply) = decision {
            out.send(Event::ReplyOut(ReplyOut {
                id: uuid::Uuid::new_v4(),
                reply_to: msg.id,
                group_id: msg.group_id.clone(),
                target_user: msg.sender.clone(),
                content: reply,
                layer: "gate".to_string(),
            }))
            .await
            .map_err(|_| NekoError::transport("router channel closed"))?;
        } else if let GateDecision::Escalate(reason) = decision {
            out.send(Event::Escalation(reason, msg, state))
                .await
                .map_err(|_| NekoError::transport("router channel closed"))?;
        }

        Ok(())
    }

    /// Heuristic that decides whether a message can be handled by the cheap
    /// cozy path. Delegates to the configured [`GateHeuristic`].
    fn should_cozy(&self, msg: &ChatMessage) -> bool {
        self.heuristic.should_cozy(msg, &self.config)
    }

    async fn generate_cozy(&self, msg: &ChatMessage) -> Result<String, NekoError> {
        let prompt = crate::prompt::cozy_prompt(&msg.content, self.config.max_cozy_words);
        let req = LlmRequest {
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: prompt,
            }],
            temperature: self.config.llm_temperature,
            max_tokens: Some(self.config.llm_max_tokens),
            response_format: Some(ResponseFormat::Text),
        };

        let resp = self.llm.complete(req).await?;
        let mut text = resp.content.trim().to_string();

        // Strip surrounding quotes if the model added them.
        if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
            text = text[1..text.len() - 1].to_string();
        }

        // Hard truncate to the word budget.
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() > self.config.max_cozy_words {
            text = words[..self.config.max_cozy_words].join("");
        }

        if matches!(resp.finish_reason, FinishReason::Length) {
            warn!("cozy reply hit max_tokens");
        }

        Ok(text)
    }
}

impl<C: LlmClient> Clone for GateActor<C> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            llm: self.llm.clone(),
            semaphore: self.semaphore.clone(),
            states: self.states.clone(),
            heuristic: self.heuristic.clone(),
        }
    }
}
