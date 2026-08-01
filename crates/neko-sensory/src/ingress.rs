use async_trait::async_trait;
use napcat_link::NapLink;
use neko_core::{ChatMessage, Event, Ingress, NekoError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

/// NapCat / OneBot 11 compatible ingress using the official `napcat-link` SDK.
pub struct NapCatIngress {
    url: String,
    token: Option<String>,
    reconnect_interval: Duration,
    /// Number of messages dropped because the downstream channel was full.
    drop_count: Arc<AtomicU64>,
}

impl NapCatIngress {
    pub fn new(
        url: impl Into<String>,
        token: Option<String>,
        reconnect_interval: Duration,
    ) -> Self {
        Self {
            url: url.into(),
            token,
            reconnect_interval,
            drop_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Total number of messages dropped due to backpressure.
    pub fn drop_count(&self) -> u64 {
        self.drop_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl Ingress for NapCatIngress {
    async fn run(
        self,
        out: mpsc::Sender<Event>,
        mut shutdown: watch::Receiver<bool>,
    ) -> Result<(), NekoError> {
        info!("NapCat ingress starting at {}", self.url);

        loop {
            if *shutdown.borrow() {
                info!("ingress shutting down");
                return Ok(());
            }

            match run_once(
                &self.url,
                self.token.as_deref(),
                &out,
                &mut shutdown,
                &self.drop_count,
            )
            .await
            {
                Ok(()) => {
                    if *shutdown.borrow() {
                        return Ok(());
                    }
                    warn!("NapCat connection closed, reconnecting");
                }
                Err(e) => {
                    error!(
                        "NapCat connection error: {e}, reconnecting in {:?}",
                        self.reconnect_interval
                    );
                }
            }

            tokio::time::sleep(self.reconnect_interval).await;
        }
    }
}

async fn run_once(
    url: &str,
    token: Option<&str>,
    out: &mpsc::Sender<Event>,
    shutdown: &mut watch::Receiver<bool>,
    drops: &Arc<AtomicU64>,
) -> Result<(), NekoError> {
    let mut builder = NapLink::builder(url);
    if let Some(token) = token {
        builder = builder.token(token);
    }

    let client = Arc::new(
        builder
            .build()
            .map_err(|e| NekoError::transport(format!("failed to build NapLink client: {e}")))?,
    );

    let mut sub = client.subscribe();
    let out_for_listener = out.clone();
    let drops = drops.clone();

    let listener = tokio::spawn(async move {
        loop {
            match sub.recv_filter("message").await {
                Some(event) => {
                    if event.name.starts_with("message.group") {
                        if let Some(msg) = crate::parser::parse_onebot11_group_message(&event.data)
                        {
                            debug!("parsed message from {}", msg.sender);
                            if !push_or_drop(&out_for_listener, msg, &drops) {
                                warn!("router channel closed, ingress listener exiting");
                                break;
                            }
                        }
                    }
                }
                None => {
                    warn!("NapCat event subscription ended");
                    break;
                }
            }
        }
    });

    client
        .connect()
        .await
        .map_err(|e| NekoError::transport(format!("NapCat connect failed: {e}")))?;

    info!("NapCat connected");

    // Wait for shutdown. napcat-link handles automatic reconnects internally;
    // we only exit this loop when the user requests shutdown.
    let _ = shutdown.changed().await;

    client.disconnect();
    listener.abort();
    let _ = listener.await;

    Ok(())
}

/// Push a parsed message into the downstream channel without blocking.
///
/// When the channel is full, the message is dropped (spam) and the drop
/// counter is incremented so backpressure loss is observable. Returns `false`
/// only when the channel is closed (a fatal condition for the listener).
fn push_or_drop(out: &mpsc::Sender<Event>, msg: ChatMessage, drops: &AtomicU64) -> bool {
    match out.try_send(Event::IncomingMessage(msg)) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(_)) => {
            let total = drops.fetch_add(1, Ordering::Relaxed) + 1;
            // Rate-limit the log noise: always log the first few, then one
            // per hundred.
            if total < 5 || total % 100 == 0 {
                warn!("ingress channel full, dropped message (total drops: {total})");
            }
            true
        }
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn message(content: &str) -> ChatMessage {
        ChatMessage {
            id: Uuid::new_v4(),
            group_id: "g1".to_string(),
            sender: "u1".to_string(),
            nickname: "Alice".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            reply_to: None,
            raw_payload: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn full_channel_counts_and_drops() {
        let (tx, mut rx) = mpsc::channel(1);
        let drops = AtomicU64::new(0);

        assert!(push_or_drop(&tx, message("first"), &drops));
        assert_eq!(drops.load(Ordering::Relaxed), 0);

        // Channel is full: the second message is dropped and counted.
        assert!(push_or_drop(&tx, message("second"), &drops));
        assert_eq!(drops.load(Ordering::Relaxed), 1);

        // The queued message is still delivered.
        assert!(matches!(
            rx.try_recv().unwrap(),
            Event::IncomingMessage(m) if m.content == "first"
        ));
    }

    #[tokio::test]
    async fn closed_channel_signals_fatal() {
        let (tx, _rx) = mpsc::channel::<Event>(1);
        drop(_rx);
        assert!(!push_or_drop(&tx, message("x"), &AtomicU64::new(0)));
    }
}
