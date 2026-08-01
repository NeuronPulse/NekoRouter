use crate::{CooldownStore, GroupId, NekoError};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

/// Per-group reply rate limiter.
///
/// Tracks the last time a reply was sent per group and denies replies that
/// arrive within `min_interval` of the previous one. Used to avoid spamming
/// a group and to reduce the risk of triggering anti-spam wind control.
///
/// When a `CooldownStore` is provided, watermarks are loaded on construction
/// and persisted whenever a reply is allowed.
pub struct ReplyCooldown {
    min_interval: Duration,
    last_reply: Mutex<HashMap<GroupId, chrono::DateTime<Utc>>>,
    store: Option<Arc<dyn CooldownStore>>,
}

impl std::fmt::Debug for ReplyCooldown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplyCooldown")
            .field("min_interval", &self.min_interval)
            .field("last_reply", &self.last_reply)
            .field("store", &self.store.is_some())
            .finish()
    }
}

impl ReplyCooldown {
    /// A cooldown with the given minimum interval. A zero interval allows
    /// everything (no rate limiting).
    pub fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_reply: Mutex::new(HashMap::new()),
            store: None,
        }
    }

    /// Build a cooldown backed by a store. Watermarks are loaded from the
    /// store before the cooldown is returned.
    pub async fn new_with_store(
        min_interval: Duration,
        store: Arc<dyn CooldownStore>,
    ) -> Result<Self, NekoError> {
        let watermarks = store.load_cooldowns().await?;
        Ok(Self {
            min_interval,
            last_reply: Mutex::new(watermarks),
            store: Some(store),
        })
    }

    /// Whether a reply to `group_id` is currently allowed.
    ///
    /// Returns `true` (and records the timestamp) only when the previous
    /// reply for the group happened more than `min_interval` ago. When a
    /// store is attached, the new watermark is persisted asynchronously.
    pub fn allow(&self, group_id: &GroupId) -> bool {
        let mut last = self.last_reply.lock().unwrap();
        let now = Utc::now();
        if let Some(&prev) = last.get(group_id) {
            if (now - prev).to_std().unwrap_or(self.min_interval) < self.min_interval {
                return false;
            }
        }
        last.insert(group_id.clone(), now);

        if let Some(store) = &self.store {
            let group_id = group_id.clone();
            let store = store.clone();
            tokio::spawn(async move {
                if let Err(e) = store.save_cooldown(&group_id, now).await {
                    tracing::warn!("failed to persist cooldown for {}: {e}", group_id);
                }
            });
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_interval_always_allows() {
        let cd = ReplyCooldown::new(Duration::ZERO);
        for _ in 0..10 {
            assert!(cd.allow(&"g1".to_string()));
        }
    }

    #[test]
    fn blocks_replies_within_interval() {
        let cd = ReplyCooldown::new(Duration::from_millis(100));
        let g = "g1".to_string();
        assert!(cd.allow(&g));
        assert!(!cd.allow(&g));
        // Different groups are independent.
        assert!(cd.allow(&"g2".to_string()));
    }
}
