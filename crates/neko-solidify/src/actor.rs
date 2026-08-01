use crate::{parser, prompt};
use neko_core::{
    DetectiveReport, Event, GraphStore, LlmClient, LlmMessage, LlmRequest, LlmRole, NekoError,
    ResponseFormat,
};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

/// Configuration for the solidification layer.
#[derive(Debug, Clone)]
pub struct SolidifyConfig {
    pub report_buffer_limit: usize,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,
}

impl Default for SolidifyConfig {
    fn default() -> Self {
        Self {
            report_buffer_limit: 100,
            llm_temperature: 0.7,
            llm_max_tokens: 1024,
        }
    }
}

/// Layer 5 actor: late-night solidification center.
///
/// Buffers detective reports and, on `Event::SolidifyTick`, asks an LLM to
/// distill them into idempotent Cypher updates for the graph store.
pub struct SolidifyActor<C: LlmClient> {
    config: SolidifyConfig,
    llm: Arc<C>,
    graph_store: Arc<dyn GraphStore>,
    buffer: Arc<Mutex<Vec<DetectiveReport>>>,
}

impl<C: LlmClient + 'static> SolidifyActor<C> {
    pub fn new(config: SolidifyConfig, llm: C, graph_store: Arc<dyn GraphStore>) -> Self {
        Self {
            config,
            llm: Arc::new(llm),
            graph_store,
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn run(
        self,
        mut inbound: mpsc::Receiver<Event>,
        out: mpsc::Sender<Event>,
    ) -> Result<(), NekoError> {
        info!("solidify actor started");

        while let Some(event) = inbound.recv().await {
            match event {
                Event::DetectiveReport(report) => {
                    let mut buf = self.buffer.lock().await;
                    if buf.len() < self.config.report_buffer_limit {
                        buf.push(report);
                    } else {
                        warn!("solidify report buffer full, dropping oldest report");
                        buf.remove(0);
                        buf.push(report);
                    }
                }
                Event::SolidifyTick => {
                    let this = self.clone();
                    let out = out.clone();
                    tokio::spawn(async move {
                        if let Err(e) = this.solidify(out).await {
                            warn!("solidify tick failed: {e}");
                        }
                    });
                }
                _ => {}
            }
        }

        info!("solidify actor stopped");
        Ok(())
    }

    async fn solidify(&self, out: mpsc::Sender<Event>) -> Result<(), NekoError> {
        let reports: Vec<DetectiveReport> = {
            let mut buf = self.buffer.lock().await;
            if buf.is_empty() {
                debug!("solidify tick: no reports to process");
                return Ok(());
            }
            std::mem::take(&mut *buf)
        };

        info!("solidifying {} detective reports", reports.len());

        let prompt = prompt::solidify_prompt(&reports);
        let req = LlmRequest {
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: prompt,
            }],
            temperature: self.config.llm_temperature,
            max_tokens: Some(self.config.llm_max_tokens),
            response_format: Some(ResponseFormat::JsonObject),
        };

        let updates = match self.llm.complete(req).await {
            Ok(resp) => parser::parse_solidify_updates(&resp.content)?,
            Err(e) => {
                warn!("solidify llm failed: {e}, restoring reports to buffer");
                // Put reports back so the next tick can retry.
                let mut buf = self.buffer.lock().await;
                let remaining = self.config.report_buffer_limit - buf.len();
                buf.extend(reports.into_iter().take(remaining));
                return Ok(());
            }
        };

        if updates.is_empty() {
            debug!("solidify produced no graph updates");
            return Ok(());
        }

        self.graph_store.apply_updates(&updates).await?;
        info!("applied {} graph updates", updates.len());

        // Refresh the council's long-term memory from the graph so nightly
        // solidification actually influences the next day's replies.
        let summary = self.graph_store.relationship_summary(20).await?;
        let context = format!(
            "以下为昨夜固化的长期关系记忆（按强度排序）：\n{}",
            if summary.trim().is_empty() {
                "（暂无关系记录）"
            } else {
                summary.trim()
            }
        );
        out.send(Event::DailyContext(context))
            .await
            .map_err(|_| NekoError::transport("router channel closed"))?;

        Ok(())
    }
}

impl<C: LlmClient> Clone for SolidifyActor<C> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            llm: self.llm.clone(),
            graph_store: self.graph_store.clone(),
            buffer: self.buffer.clone(),
        }
    }
}
