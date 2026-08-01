use chrono::Utc;
use dashmap::DashMap;
use neko_core::{
    AffectiveState, ChatMessage, Event, GroupId, HistoryStore, MessageId, NekoError, ReplyOut,
    UserId,
};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch, Mutex};
use tracing::{debug, info_span, trace, Instrument};

/// Configuration for the sensory actor.
#[derive(Debug, Clone, Copy)]
pub struct SensoryConfig {
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub energy_decay_per_min: f32,
    pub favor_decay_per_min: f32,
    pub max_recent_messages: usize,
}

impl Default for SensoryConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            flush_interval_ms: 5000,
            energy_decay_per_min: 0.05,
            favor_decay_per_min: 0.02,
            max_recent_messages: 1024,
        }
    }
}

/// Layer 1 actor: maintains affective state, buffers messages and writes them to the history store.
pub struct SensoryActor<HS: HistoryStore> {
    config: SensoryConfig,
    history_store: HS,
    states: DashMap<(GroupId, UserId), AffectiveState>,
    buffer: Arc<Mutex<Vec<ChatMessage>>>,
    recent_senders: Arc<Mutex<HashSet<MessageId>>>,
    shutdown: watch::Receiver<bool>,
}

impl<HS: HistoryStore> SensoryActor<HS> {
    pub fn new(config: SensoryConfig, history_store: HS, shutdown: watch::Receiver<bool>) -> Self {
        Self {
            config,
            history_store,
            states: DashMap::new(),
            buffer: Arc::new(Mutex::new(Vec::with_capacity(64))),
            recent_senders: Arc::new(Mutex::new(HashSet::new())),
            shutdown,
        }
    }

    /// Run the actor loop.
    ///
    /// - `raw`: events from the ingress layer (mostly `IncomingMessage`).
    /// - `routed`: events routed back from upper layers (e.g. `ReplyOut`).
    /// - `outbound`: events forwarded to the gate layer.
    pub async fn run(
        mut self,
        mut raw: mpsc::Receiver<Event>,
        mut routed: mpsc::Receiver<Event>,
        outbound: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        // Restore persisted affective state so it survives restarts.
        for (group_id, user_id, state) in self.history_store.load_affective_states().await? {
            self.states.insert((group_id, user_id), state);
        }

        let mut flush_interval =
            tokio::time::interval(Duration::from_millis(self.config.flush_interval_ms));
        flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = self.shutdown.changed() => {
                    if *self.shutdown.borrow() {
                        break;
                    }
                }
                _ = flush_interval.tick() => {
                    self.flush().await?;
                }
                Some(event) = raw.recv() => {
                    self.handle_event(event, &outbound).await?;
                }
                Some(event) = routed.recv() => {
                    self.handle_event(event, &outbound).await?;
                }
            }
        }

        self.flush().await?;
        Ok(())
    }

    async fn handle_event(
        &self,
        event: Event,
        outbound: &mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        match event {
            Event::IncomingMessage(msg) => {
                let trace_id = msg.trace_id;
                self.handle_message(msg, outbound)
                    .instrument(info_span!("sensory_message", trace_id = %trace_id))
                    .await
            }
            Event::ReplyOut(reply) => {
                let trace_id = reply.trace_id;
                self.handle_reply(reply)
                    .instrument(info_span!("sensory_reply", trace_id = %trace_id))
                    .await
            }
            _ => Ok(()),
        }
    }

    async fn handle_message(
        &self,
        msg: ChatMessage,
        outbound: &mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        trace!("handling incoming message {}", msg.id);

        {
            let mut seen = self.recent_senders.lock().await;
            if seen.len() >= self.config.max_recent_messages {
                // Simple eviction: drop the set when it grows too large.
                seen.clear();
            }
            if !seen.insert(msg.id) {
                // Duplicate message (e.g. re-delivered after a reconnect).
                // Drop it before it reaches the gate or the history store.
                trace!("dropping duplicate message {}", msg.id);
                return Ok(());
            }
        }

        let state = self.touch_state(&msg.group_id, &msg.sender).await;

        outbound
            .send(Event::IncomingMessage(msg.clone()))
            .await
            .map_err(|_| NekoError::transport("gate channel closed"))?;

        outbound
            .send(Event::AffectiveUpdated(
                msg.group_id.clone(),
                msg.sender.clone(),
                state,
            ))
            .await
            .map_err(|_| NekoError::transport("gate channel closed"))?;

        let mut buf = self.buffer.lock().await;
        buf.push(msg);
        let should_flush = buf.len() >= self.config.batch_size;
        drop(buf);

        if should_flush {
            self.flush().await?;
        }

        Ok(())
    }

    async fn handle_reply(&self, reply: ReplyOut) -> Result<(), NekoError> {
        trace!("handling reply {} to user {}", reply.id, reply.target_user);
        let key = (reply.group_id.clone(), reply.target_user.clone());
        let mut entry = self.states.entry(key).or_default();
        let now = Utc::now();
        entry.on_reply(0.05, now);
        Ok(())
    }

    /// Update the last-seen timestamp and decay energy/favor for a user.
    async fn touch_state(&self, group_id: &GroupId, user_id: &UserId) -> AffectiveState {
        let key = (group_id.clone(), user_id.clone());
        let now = Utc::now();
        let mut entry = self.states.entry(key).or_default();
        if let Some(last) = entry.last_updated {
            let minutes = (now - last).num_seconds() as f32 / 60.0;
            entry.decay(
                minutes,
                self.config.energy_decay_per_min,
                self.config.favor_decay_per_min,
            );
        }
        entry.energy = (entry.energy + 0.02).min(1.0);
        entry.last_updated = Some(now);
        *entry.value()
    }

    async fn flush(&self) -> Result<(), NekoError> {
        let batch: Vec<ChatMessage> = {
            let mut buf = self.buffer.lock().await;
            std::mem::take(&mut *buf)
        };

        if !batch.is_empty() {
            debug!("flushing {} messages to history store", batch.len());
            let ids: Vec<MessageId> = batch.iter().map(|m| m.id).collect();
            self.history_store.append_batch(&batch).await?;
            self.history_store.mark_processed(&ids).await?;
        }

        self.persist_states().await
    }

    /// Persist the in-memory affective states to the history store.
    async fn persist_states(&self) -> Result<(), NekoError> {
        let states: Vec<(GroupId, UserId, AffectiveState)> = self
            .states
            .iter()
            .map(|entry| (entry.key().0.clone(), entry.key().1.clone(), *entry.value()))
            .collect();

        if !states.is_empty() {
            self.history_store.save_affective_states(&states).await?;
        }
        Ok(())
    }
}
