use chrono::{DateTime, Utc};
use dashmap::DashMap;
use neko_core::{
    AffectiveState, ChatMessage, Event, GroupId, HistoryStore, MessageId, NekoError, ReplyOut,
    TopicBurst, TopicBurstScore, UserId,
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
    /// Whether to detect topic bursts and forward them to the detective.
    pub burst_detection_enabled: bool,
    /// Observation window in seconds for burst detection.
    pub burst_window_sec: u64,
    /// Minimum messages-per-minute to consider a burst.
    pub burst_threshold_mpm: f32,
    /// Minimum distinct senders in the window to consider a burst.
    pub burst_threshold_participants: usize,
    /// Maximum average gap between messages (seconds) to consider a burst.
    pub burst_threshold_gap_sec: f32,
    /// Cooldown between consecutive bursts for the same group.
    pub burst_cooldown_sec: u64,
}

impl Default for SensoryConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            flush_interval_ms: 5000,
            energy_decay_per_min: 0.05,
            favor_decay_per_min: 0.02,
            max_recent_messages: 1024,
            burst_detection_enabled: true,
            burst_window_sec: 60,
            burst_threshold_mpm: 6.0,
            burst_threshold_participants: 3,
            burst_threshold_gap_sec: 30.0,
            burst_cooldown_sec: 120,
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
    /// Recent messages per group, used for topic-burst detection.
    burst_windows: DashMap<GroupId, Vec<ChatMessage>>,
    /// Last time a topic burst was emitted per group.
    last_burst: DashMap<GroupId, DateTime<Utc>>,
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
            burst_windows: DashMap::new(),
            last_burst: DashMap::new(),
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
        buf.push(msg.clone());
        let should_flush = buf.len() >= self.config.batch_size;
        drop(buf);

        if should_flush {
            self.flush().await?;
        }

        if let Some(burst) = self.detect_burst(&msg).await {
            outbound
                .send(Event::TopicBurst(burst))
                .await
                .map_err(|_| NekoError::transport("router channel closed"))?;
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

    /// Update the per-group sliding window and emit a topic burst if the
    /// conversation has heated up.
    async fn detect_burst(&self, msg: &ChatMessage) -> Option<TopicBurst> {
        if !self.config.burst_detection_enabled {
            return None;
        }

        let window = Duration::from_secs(self.config.burst_window_sec);
        let now = Utc::now();

        let mut entry = self.burst_windows.entry(msg.group_id.clone()).or_default();
        entry.push(msg.clone());
        // Evict messages older than the observation window.
        entry
            .retain(|m| now - m.timestamp < chrono::Duration::from_std(window).unwrap_or_default());

        // Clamp window size to avoid unbounded growth.
        const MAX_WINDOW_SIZE: usize = 256;
        if entry.len() > MAX_WINDOW_SIZE {
            let excess = entry.len() - MAX_WINDOW_SIZE;
            entry.drain(0..excess);
        }

        let window_messages = entry.value().clone();
        drop(entry);

        if window_messages.len() < self.config.burst_threshold_participants.max(2) {
            return None;
        }

        let unique_participants: HashSet<_> = window_messages.iter().map(|m| &m.sender).collect();
        if unique_participants.len() < self.config.burst_threshold_participants {
            return None;
        }

        let durations: Vec<_> = window_messages
            .windows(2)
            .map(|pair| {
                let gap = pair[1].timestamp - pair[0].timestamp;
                gap.num_milliseconds().max(0) as f32 / 1000.0
            })
            .collect();
        let avg_gap_seconds = if durations.is_empty() {
            0.0
        } else {
            durations.iter().sum::<f32>() / durations.len() as f32
        };

        let oldest = window_messages.first()?.timestamp;
        let newest = window_messages.last()?.timestamp;
        let elapsed_seconds = (newest - oldest).num_seconds().max(1) as f32;
        let messages_per_minute = window_messages.len() as f32 / (elapsed_seconds / 60.0);

        if messages_per_minute < self.config.burst_threshold_mpm {
            return None;
        }
        if avg_gap_seconds > self.config.burst_threshold_gap_sec {
            return None;
        }

        // Cooldown: do not fire repeatedly for the same heated conversation.
        let cooldown = chrono::Duration::seconds(self.config.burst_cooldown_sec as i64);
        if let Some(last) = self.last_burst.get(&msg.group_id) {
            let last_time = *last.value();
            if now - last_time < cooldown {
                return None;
            }
        }
        self.last_burst.insert(msg.group_id.clone(), now);

        let score = TopicBurstScore {
            messages_per_minute,
            unique_participants: unique_participants.len(),
            avg_gap_seconds,
            coherence: None,
        };

        Some(TopicBurst {
            group_id: msg.group_id.clone(),
            messages: window_messages,
            score,
            detected_at: now,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use neko_memory::SqliteStore;
    use tokio::sync::watch;
    use uuid::Uuid;

    fn make_message(group: &str, sender: &str, content: &str, ts: DateTime<Utc>) -> ChatMessage {
        ChatMessage {
            id: Uuid::new_v4(),
            trace_id: Uuid::new_v4(),
            group_id: group.to_string(),
            sender: sender.to_string(),
            nickname: sender.to_string(),
            content: content.to_string(),
            timestamp: ts,
            reply_to: None,
            raw_payload: serde_json::Value::Null,
        }
    }

    fn burst_config() -> SensoryConfig {
        SensoryConfig {
            batch_size: 100,
            flush_interval_ms: 10000,
            energy_decay_per_min: 0.05,
            favor_decay_per_min: 0.02,
            max_recent_messages: 1024,
            burst_detection_enabled: true,
            burst_window_sec: 60,
            burst_threshold_mpm: 6.0,
            burst_threshold_participants: 3,
            burst_threshold_gap_sec: 30.0,
            burst_cooldown_sec: 120,
        }
    }

    #[tokio::test]
    async fn detects_topic_burst_when_many_users_chat_quickly() {
        let sqlite = SqliteStore::connect_in_memory().await.unwrap();
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cfg = burst_config();
        cfg.burst_cooldown_sec = 0; // allow repeated bursts for testing
        let actor = SensoryActor::new(cfg, sqlite, shutdown_rx);

        let base = Utc::now();
        let group = "g1";

        // Three users rapidly send messages.
        let mut triggered = 0usize;
        for i in 0..6 {
            let sender = match i % 3 {
                0 => "u1",
                1 => "u2",
                _ => "u3",
            };
            let msg = make_message(
                group,
                sender,
                &format!("msg{i}"),
                base + chrono::Duration::seconds(i),
            );
            if let Some(burst) = actor.detect_burst(&msg).await {
                triggered += 1;
                assert_eq!(burst.group_id, group);
                assert!(burst.score.unique_participants >= 3);
                assert!(burst.score.messages_per_minute >= 6.0);
            }
        }
        assert!(triggered > 0, "topic burst should trigger at least once");
    }

    #[tokio::test]
    async fn no_burst_when_only_one_user_talks() {
        let sqlite = SqliteStore::connect_in_memory().await.unwrap();
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let actor = SensoryActor::new(burst_config(), sqlite, shutdown_rx);

        let base = Utc::now();
        let group = "g1";

        for i in 0..10 {
            let msg = make_message(
                group,
                "u1",
                &format!("msg{i}"),
                base + chrono::Duration::seconds(i),
            );
            assert!(actor.detect_burst(&msg).await.is_none());
        }
    }

    #[tokio::test]
    async fn burst_detection_can_be_disabled() {
        let sqlite = SqliteStore::connect_in_memory().await.unwrap();
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cfg = burst_config();
        cfg.burst_detection_enabled = false;
        let actor = SensoryActor::new(cfg, sqlite, shutdown_rx);

        let base = Utc::now();
        for i in 0..6 {
            let sender = match i % 3 {
                0 => "u1",
                1 => "u2",
                _ => "u3",
            };
            let msg = make_message(
                "g1",
                sender,
                &format!("msg{i}"),
                base + chrono::Duration::seconds(i),
            );
            assert!(actor.detect_burst(&msg).await.is_none());
        }
    }
}
