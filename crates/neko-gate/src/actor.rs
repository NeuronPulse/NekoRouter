use crate::heuristic::{
    classifier_from_name, BotIdentity, GateClassification, GateClassifier, LlmGateClassifier,
};
use dashmap::DashMap;
use neko_core::{
    AffectiveState, ChatMessage, Event, GateDecision, GroupId, LlmClient, NekoError, RuntimeState,
    UserId,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, info_span, warn, Instrument};

/// Configuration for the gate layer.
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Messages longer than this are dropped before classification.
    pub max_message_length: usize,
    /// Word budget for ambient join candidates. The heuristic treats messages
    /// with at most `max_ambient_words * 3` whitespace-separated tokens as
    /// potential ambient joins.
    pub max_ambient_words: usize,
    /// Maximum concurrent classification tasks.
    pub concurrency_limit: usize,
    /// Classifier strategy to use. Built-in names:
    /// - `"default"`: fast heuristic classifier.
    /// - `"llm"`: cheap LLM classifier (requires `llm_client`).
    /// - `"escalate_all"`: always escalate as personal reply (testing stub).
    pub classifier: String,
    /// Number of recent messages to include in the LLM prompt context.
    pub context_messages: usize,
    /// TTL in seconds for cached classification results.
    pub cache_ttl_sec: u64,
    /// Maximum classifications per minute per group.
    pub rate_limit_per_min: u32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            max_message_length: 800,
            max_ambient_words: 10,
            concurrency_limit: 8,
            classifier: "default".to_string(),
            context_messages: 10,
            cache_ttl_sec: 30,
            rate_limit_per_min: 60,
        }
    }
}

/// Layer 2 actor: engagement classifier gate.
///
/// The gate no longer generates replies. Its only job is to decide whether the
/// bot should engage and, if so, whether the message is a personal reply or an
/// ambient group join. All engagement decisions are escalated to the council.
pub struct GateActor {
    config: GateConfig,
    self_id: BotIdentity,
    states: DashMap<(GroupId, UserId), AffectiveState>,
    classifier: Arc<dyn GateClassifier>,
    /// Recent messages per group, used as LLM context.
    recent_context: DashMap<GroupId, Vec<ChatMessage>>,
    /// Optional shared runtime metrics, updated when messages enter the gate.
    state: Option<Arc<RuntimeState>>,
}

impl GateActor {
    pub fn new(config: GateConfig, self_id: BotIdentity) -> Self {
        Self::new_with_state(config, self_id, None)
    }

    pub fn new_with_state(
        config: GateConfig,
        self_id: BotIdentity,
        state: Option<Arc<RuntimeState>>,
    ) -> Self {
        let classifier = Arc::from(classifier_from_name(&config.classifier));
        Self {
            config,
            self_id,
            states: DashMap::new(),
            classifier,
            recent_context: DashMap::new(),
            state,
        }
    }

    /// Build a gate actor with an LLM classifier.
    pub fn new_with_llm(
        config: GateConfig,
        self_id: BotIdentity,
        llm_client: Arc<dyn LlmClient>,
        state: Option<Arc<RuntimeState>>,
    ) -> Self {
        let classifier: Arc<dyn GateClassifier> = if config.classifier == "llm" {
            Arc::new(LlmGateClassifier::new(llm_client))
        } else {
            Arc::from(classifier_from_name(&config.classifier))
        };
        Self {
            config,
            self_id,
            states: DashMap::new(),
            classifier,
            recent_context: DashMap::new(),
            state,
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
                    if let Some(ref runtime) = self.state {
                        runtime
                            .messages_received
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    let state = self
                        .states
                        .get(&(msg.group_id.clone(), msg.sender.clone()))
                        .map(|s| *s.value())
                        .unwrap_or_default();
                    let this = self.clone();
                    let out = out.clone();
                    let trace_id = msg.trace_id;
                    tokio::spawn(
                        async move {
                            if let Err(e) = this.handle_message(msg, state, out).await {
                                warn!("gate handle error: {e}");
                            }
                        }
                        .instrument(info_span!("gate_handle", trace_id = %trace_id)),
                    );
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

        // Maintain a rolling window of recent messages for context-aware
        // classifiers.
        let recent_context = {
            let mut entry = self.recent_context.entry(msg.group_id.clone()).or_default();
            entry.push(msg.clone());
            if entry.len() > self.config.context_messages {
                entry.remove(0);
            }
            entry.value().clone()
        };

        let classification = self
            .classifier
            .classify(&msg, &recent_context, &state, &self.config, &self.self_id)
            .await?;

        let decision = match classification {
            GateClassification::Drop(reason) => GateDecision::Drop(reason),
            GateClassification::Escalate(engagement_type) => {
                GateDecision::Escalate(neko_core::EscalationReason::NeedsContext, engagement_type)
            }
        };

        out.send(Event::GateDecision(decision.clone())).await.ok();

        match decision {
            GateDecision::Drop(_) => {
                // Nothing further to do.
            }
            GateDecision::Escalate(reason, engagement_type) => {
                out.send(Event::Escalation(reason, msg, state, engagement_type))
                    .await
                    .map_err(|_| NekoError::transport("router channel closed"))?;
            }
        }

        Ok(())
    }
}

impl Clone for GateActor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            self_id: self.self_id.clone(),
            states: self.states.clone(),
            classifier: self.classifier.clone(),
            recent_context: self.recent_context.clone(),
            state: self.state.clone(),
        }
    }
}
