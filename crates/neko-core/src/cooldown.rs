use crate::GroupId;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Per-group reply rate limiter.
///
/// Tracks the last time a reply was sent per group and denies replies that
/// arrive within `min_interval` of the previous one. Used to avoid spamming
/// a group and to reduce the risk of triggering anti-spam wind control.
#[derive(Debug)]
pub struct ReplyCooldown {
    min_interval: Duration,
    last_reply: Mutex<HashMap<GroupId, Instant>>,
}

impl ReplyCooldown {
    /// A cooldown with the given minimum interval. A zero interval allows
    /// everything (no rate limiting).
    pub fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_reply: Mutex::new(HashMap::new()),
        }
    }

    /// Whether a reply to `group_id` is currently allowed.
    ///
    /// Returns `true` (and records the timestamp) only when the previous
    /// reply for the group happened more than `min_interval` ago.
    pub fn allow(&self, group_id: &GroupId) -> bool {
        let mut last = self.last_reply.lock().unwrap();
        let now = Instant::now();
        if let Some(&prev) = last.get(group_id) {
            if now.duration_since(prev) < self.min_interval {
                return false;
            }
        }
        last.insert(group_id.clone(), now);
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
