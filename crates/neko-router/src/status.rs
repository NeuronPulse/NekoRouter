use chrono::Utc;
use neko_core::{Ingress, RuntimeState};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{error, info};

/// Start a minimal HTTP server on `bind_addr` that exposes `/status`.
///
/// The server runs until the shutdown signal fires. Only `GET /status` is
/// handled; every other path returns 404.
pub async fn run_status_server(
    bind_addr: String,
    state: Arc<RuntimeState>,
    ingress: Arc<dyn Ingress + Send + Sync>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> Result<(), std::io::Error> {
    let listener = TcpListener::bind(&bind_addr).await?;
    info!("status server listening on http://{bind_addr}/status");

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("status server shutting down");
                    break;
                }
            }
            result = listener.accept() => {
                match result {
                    Ok((stream, addr)) => {
                        let state = state.clone();
                        let ingress = ingress.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, state, ingress).await {
                                error!("status connection from {addr} failed: {e}");
                            }
                        });
                    }
                    Err(e) => error!("status accept failed: {e}"),
                }
            }
        }
    }

    Ok(())
}

async fn handle_connection(
    mut stream: TcpStream,
    state: Arc<RuntimeState>,
    ingress: Arc<dyn Ingress + Send + Sync>,
) -> Result<(), std::io::Error> {
    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }

    let request = String::from_utf8_lossy(&buf[..n]);
    let first_line = request.lines().next().unwrap_or("");

    let (status, body) = if first_line.starts_with("GET /status ") {
        (200, status_body(&state, ingress.drop_count()))
    } else {
        (404, r#"{"error":"not found"}"#.to_string())
    };

    let response = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;
    Ok(())
}

fn status_body(state: &RuntimeState, ingress_drops: u64) -> String {
    let started_at = state.started_at.to_rfc3339();
    let uptime_sec = (Utc::now() - state.started_at).num_seconds().max(0);
    let messages_received = state
        .messages_received
        .load(std::sync::atomic::Ordering::Relaxed);
    let replies_sent = state
        .replies_sent
        .load(std::sync::atomic::Ordering::Relaxed);

    format!(
        r#"{{"status":"ok","version":"{}","started_at":"{started_at}","uptime_sec":{uptime_sec},"messages_received":{messages_received},"replies_sent":{replies_sent},"ingress_drops":{ingress_drops}}}"#,
        env!("CARGO_PKG_VERSION")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    struct StubIngress;
    #[async_trait::async_trait]
    impl Ingress for StubIngress {
        async fn run(
            self,
            _out: tokio::sync::mpsc::Sender<neko_core::Event>,
            _shutdown: tokio::sync::watch::Receiver<bool>,
        ) -> Result<(), neko_core::NekoError> {
            Ok(())
        }

        fn drop_count(&self) -> u64 {
            42
        }
    }

    #[tokio::test]
    async fn status_endpoint_returns_json() {
        let state = Arc::new(RuntimeState {
            started_at: Utc::now(),
            ..Default::default()
        });
        let ingress: Arc<dyn Ingress + Send + Sync> = Arc::new(StubIngress);
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        let bind_addr = "127.0.0.1:0";
        let listener = TcpListener::bind(bind_addr).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let bind_addr = format!("127.0.0.1:{port}");
        tokio::spawn(run_status_server(
            bind_addr.clone(),
            state,
            ingress,
            shutdown_rx,
        ));

        // Give the server a moment to start.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut stream = TcpStream::connect(&bind_addr).await.unwrap();
        stream
            .write_all(b"GET /status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();

        let mut buf = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut buf))
            .await
            .unwrap()
            .unwrap();
        let response = String::from_utf8_lossy(&buf);

        assert!(response.contains("HTTP/1.1 200 OK"));
        assert!(response.contains("\"status\":\"ok\""));
        assert!(response.contains("\"ingress_drops\":42"));
        assert!(response.contains("\"messages_received\":0"));

        shutdown_tx.send(true).ok();
    }

    #[tokio::test]
    async fn unknown_path_returns_404() {
        let state = Arc::new(RuntimeState {
            started_at: Utc::now(),
            ..Default::default()
        });
        let ingress: Arc<dyn Ingress + Send + Sync> = Arc::new(StubIngress);
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        let bind_addr = "127.0.0.1:0";
        let listener = TcpListener::bind(bind_addr).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let bind_addr = format!("127.0.0.1:{port}");
        tokio::spawn(run_status_server(
            bind_addr.clone(),
            state,
            ingress,
            shutdown_rx,
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut stream = TcpStream::connect(&bind_addr).await.unwrap();
        stream
            .write_all(b"GET /unknown HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();

        let mut buf = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut buf))
            .await
            .unwrap()
            .unwrap();
        let response = String::from_utf8_lossy(&buf);

        assert!(response.contains("HTTP/1.1 404 OK"));
        shutdown_tx.send(true).ok();
    }
}
