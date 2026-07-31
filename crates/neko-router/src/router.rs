use chrono::Utc;
use neko_core::{Egress, Event, NekoError, ReplyOut};
use neko_memory::SqliteStore;
use tokio::sync::mpsc;
use tracing::info;

/// Route an event produced by any layer to its downstream channel and persist
/// observability records.
///
/// Every new layer or event kind must be wired here (and a channel added in
/// the `Router` setup in `main.rs`).
pub async fn dispatch_event(
    event: Event,
    egress: &dyn Egress,
    sqlite: &SqliteStore,
    sensory_routed: &mpsc::Sender<Event>,
    council_tx: &mpsc::Sender<Event>,
    detective_tx: &mpsc::Sender<Event>,
    solidify_tx: &mpsc::Sender<Event>,
) -> Result<(), NekoError> {
    match event {
        Event::ReplyOut(reply) => {
            persist_reply(sqlite, &reply).await?;
            egress.send(reply.clone()).await?;
            sensory_routed
                .send(Event::ReplyOut(reply))
                .await
                .map_err(|_| NekoError::transport("sensory routed channel closed"))?;
        }
        Event::GateDecision(decision) => {
            info!("gate decision: {decision:?}");
            let payload = serde_json::json!({"decision": format!("{:?}", decision) });
            sqlite
                .insert_event(uuid::Uuid::new_v4(), "gate_decision", &payload, Utc::now())
                .await?;
        }
        Event::Escalation(_, _, _) => {
            council_tx
                .send(event)
                .await
                .map_err(|_| NekoError::transport("council channel closed"))?;
        }
        Event::CouncilDecision(decision) => {
            info!("council decision: {decision:?}");
            let payload = serde_json::json!({
                "action": format!("{:?}", decision.action),
                "reasoning": decision.reasoning,
            });
            sqlite
                .insert_event(
                    uuid::Uuid::new_v4(),
                    "council_decision",
                    &payload,
                    Utc::now(),
                )
                .await?;
        }
        Event::DetectiveRequest(_) => {
            detective_tx
                .send(event)
                .await
                .map_err(|_| NekoError::transport("detective channel closed"))?;
        }
        Event::DetectiveReport(report) => {
            info!("detective report for {}", report.target_user);
            solidify_tx
                .send(Event::DetectiveReport(report))
                .await
                .map_err(|_| NekoError::transport("solidify channel closed"))?;
        }
        Event::SolidifyTick => {
            solidify_tx
                .send(event)
                .await
                .map_err(|_| NekoError::transport("solidify channel closed"))?;
        }
        _ => {}
    }
    Ok(())
}

pub async fn persist_reply(sqlite: &SqliteStore, reply: &ReplyOut) -> Result<(), NekoError> {
    sqlite
        .insert_reply(
            reply.id,
            reply.reply_to,
            &reply.layer,
            &reply.content,
            Utc::now(),
        )
        .await
}
