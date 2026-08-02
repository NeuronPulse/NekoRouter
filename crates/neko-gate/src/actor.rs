use crate::heuristic::{heuristic_from_name, BotIdentity, GateClassification, GateHeuristic};
use dashmap::DashMap;
use neko_core::{
    AffectiveState, ChatMessage, EngagementType, Event, GateDecision, GroupId, NekoError,
    RuntimeState, UserId,
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
    /// Name of the heuristic strategy to use. Built-in names:
    /// - `"default"`: classify commands/URLs/long messages as drop, directed
    ///   messages as personal replies, short chitchat as ambient joins.
    /// - `"escalate_all"`: always escalate as personal reply (testing stub).
    pub heuristic: String,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            max_message_length: 800,
            max_ambient_words: 10,
            concurrency_limit: 8,
            heuristic: "default".to_string(),
        }
    }
}

/// Layer 2 actor: fast heuristic gate.
///
/// The gate no longer generates replies. Its only job is to decide whether the
/// bot should engage and, if so, whether the message is a personal reply or an
/// ambient group join. All engagement decisions are escalated to the council.
pub struct GateActor {
    config: GateConfig,
    self_id: BotIdentity,
    states: DashMap<(GroupId, UserId), AffectiveState>,
    heuristic: Arc<dyn GateHeuristic>,
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
        let heuristic = Arc::from(heuristic_from_name(&config.heuristic));
        Self {
            config,
            self_id,
            states: DashMap::new(),
            heuristic,
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

        let classification = self.heuristic.classify(&msg, &self.config, &self.self_id);

        let decision = match classification {
            GateClassification::Drop(reason) => GateDecision::Drop(reason),
            GateClassification::Escalate(engagement_type) => {
                GateDecision::Escalate(reason_for(&msg, engagement_type), engagement_type)
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

/// Map an engagement classification to a human-readable escalation reason.
fn reason_for(_msg: &ChatMessage, _engagement_type: EngagementType) -> neko_core::EscalationReason {
    neko_core::EscalationReason::NeedsContext
}

impl Clone for GateActor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            self_id: self.self_id.clone(),
            states: self.states.clone(),
            heuristic: self.heuristic.clone(),
            state: self.state.clone(),
        }
    }
}
