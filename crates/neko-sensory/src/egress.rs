use async_trait::async_trait;
use napcat_link::{MessageSegment, NapLink};
use neko_core::{Egress, NekoError, ReplyOut};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, warn};

/// NapCat / OneBot 11 compatible egress using the official `napcat-link` SDK.
///
/// The first reply opens a WebSocket connection and it is reused for all
/// subsequent replies; the connection is transparently re-established if it
/// drops.
#[derive(Clone)]
pub struct NapCatEgress {
    url: String,
    token: Option<String>,
    client: Arc<Mutex<Option<Arc<NapLink>>>>,
}

impl NapCatEgress {
    pub fn new(url: impl Into<String>, token: Option<String>) -> Self {
        Self {
            url: url.into(),
            token,
            client: Arc::new(Mutex::new(None)),
        }
    }

    /// Return the shared client, connecting (or reconnecting) on demand.
    async fn client(&self) -> Result<Arc<NapLink>, NekoError> {
        let mut guard = self.client.lock().await;
        if let Some(client) = guard.as_ref() {
            if client.is_connected() {
                return Ok(client.clone());
            }
            warn!("NapCat egress connection lost, reconnecting");
        }

        let mut builder = NapLink::builder(&self.url);
        if let Some(token) = &self.token {
            builder = builder.token(token);
        }

        let client =
            Arc::new(builder.build().map_err(|e| {
                NekoError::transport(format!("failed to build NapLink client: {e}"))
            })?);

        client
            .connect()
            .await
            .map_err(|e| NekoError::transport(format!("NapCat connect failed: {e}")))?;

        *guard = Some(client.clone());
        Ok(client)
    }
}

#[async_trait]
impl Egress for NapCatEgress {
    async fn send(&self, reply: ReplyOut) -> Result<(), NekoError> {
        let group_id = reply
            .group_id
            .parse::<i64>()
            .map_err(|e| NekoError::transport(format!("invalid group_id: {e}")))?;

        debug!("sending reply to group {}: {}", group_id, reply.content);

        let client = self.client().await?;
        let mut segments = Vec::new();
        if let Some(platform_id) = reply.reply_to_platform {
            debug!("quoting message {platform_id}");
            segments.push(MessageSegment::reply(platform_id));
        }
        segments.push(MessageSegment::text(&reply.content));

        client
            .api()
            .message
            .send_group_message(group_id, segments)
            .await
            .map_err(|e| NekoError::transport(format!("NapCat send failed: {e}")))?;

        Ok(())
    }
}
