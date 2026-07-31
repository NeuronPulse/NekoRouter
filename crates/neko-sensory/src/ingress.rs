use async_trait::async_trait;
use napcat_link::NapLink;
use neko_core::{Event, Ingress, NekoError};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

/// NapCat / OneBot 11 compatible ingress using the official `napcat-link` SDK.
pub struct NapCatIngress {
    url: String,
    token: Option<String>,
    reconnect_interval: Duration,
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
        }
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

            match run_once(&self.url, self.token.as_deref(), &out, &mut shutdown).await {
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

    let listener = tokio::spawn(async move {
        loop {
            match sub.recv_filter("message").await {
                Some(event) => {
                    if event.name.starts_with("message.group") {
                        if let Some(msg) = crate::parser::parse_onebot11_group_message(&event.data)
                        {
                            debug!("parsed message from {}", msg.sender);
                            if out_for_listener
                                .send(Event::IncomingMessage(msg))
                                .await
                                .is_err()
                            {
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
