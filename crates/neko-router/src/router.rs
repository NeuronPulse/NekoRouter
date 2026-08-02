use chrono::Utc;
use neko_core::{
    Egress, Event, GraphStore, NekoError, ReplyCooldown, ReplyOut, RuntimeState, VectorStore,
};
use neko_memory::SqliteStore;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Route an event produced by any layer to its downstream channel and persist
/// observability records.
///
/// Every new layer or event kind must be wired here (and a channel added in
/// the `Router` setup in `main.rs`).
///
/// `cooldown` optionally rate-limits outgoing replies per group; pass `None`
/// to disable.
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_event(
    event: Event,
    egress: &dyn Egress,
    sqlite: &SqliteStore,
    sensory_routed: &mpsc::Sender<Event>,
    council_tx: &mpsc::Sender<Event>,
    detective_tx: &mpsc::Sender<Event>,
    solidify_tx: &mpsc::Sender<Event>,
    cooldown: Option<&ReplyCooldown>,
) -> Result<(), NekoError> {
    dispatch_event_full(
        event,
        egress,
        sqlite,
        None,
        None,
        sensory_routed,
        council_tx,
        detective_tx,
        solidify_tx,
        cooldown,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn dispatch_event_with_state(
    event: Event,
    egress: &dyn Egress,
    sqlite: &SqliteStore,
    sensory_routed: &mpsc::Sender<Event>,
    council_tx: &mpsc::Sender<Event>,
    detective_tx: &mpsc::Sender<Event>,
    solidify_tx: &mpsc::Sender<Event>,
    cooldown: Option<&ReplyCooldown>,
    runtime_state: Option<&Arc<RuntimeState>>,
) -> Result<(), NekoError> {
    dispatch_event_full(
        event,
        egress,
        sqlite,
        None,
        None,
        sensory_routed,
        council_tx,
        detective_tx,
        solidify_tx,
        cooldown,
        runtime_state,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn dispatch_event_full(
    event: Event,
    egress: &dyn Egress,
    sqlite: &SqliteStore,
    vector_store: Option<&dyn VectorStore>,
    graph_store: Option<&dyn GraphStore>,
    sensory_routed: &mpsc::Sender<Event>,
    council_tx: &mpsc::Sender<Event>,
    detective_tx: &mpsc::Sender<Event>,
    solidify_tx: &mpsc::Sender<Event>,
    cooldown: Option<&ReplyCooldown>,
    runtime_state: Option<&Arc<RuntimeState>>,
) -> Result<(), NekoError> {
    match event {
        Event::ReplyOut(mut reply) => {
            if let Some(cooldown) = cooldown {
                if !cooldown.allow(&reply.group_id) {
                    warn!(
                        "reply cooldown active for group {}, skipping reply",
                        reply.group_id
                    );
                    return Ok(());
                }
            }
            // Resolve the platform id of the replied-to message so the egress
            // can send a quote reply; degrade to plain text when unavailable.
            if reply.reply_to_platform.is_none() {
                reply.reply_to_platform = sqlite
                    .platform_message_id(reply.reply_to)
                    .await
                    .unwrap_or(None);
            }
            persist_reply(sqlite, &reply).await?;
            egress.send(reply.clone()).await?;
            if let Some(state) = runtime_state {
                state
                    .replies_sent
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
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
        Event::Escalation(_, _, _, _) => {
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
            // The report feeds the council's final reply and the nightly
            // solidify graph updates.
            council_tx
                .send(Event::DetectiveReport(report.clone()))
                .await
                .map_err(|_| NekoError::transport("council channel closed"))?;
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
        Event::DailyContext(context) => {
            info!("daily context produced ({} chars)", context.chars().count());
            // Persist for observability and feed the council so long-term
            // relationship memory influences its decisions.
            let payload = serde_json::json!({ "context": context });
            sqlite
                .insert_event(uuid::Uuid::new_v4(), "daily_context", &payload, Utc::now())
                .await?;
            council_tx
                .send(Event::DailyContext(context))
                .await
                .map_err(|_| NekoError::transport("council channel closed"))?;
        }
        Event::TopicBurst(_) => {
            detective_tx
                .send(event)
                .await
                .map_err(|_| NekoError::transport("detective channel closed"))?;
        }
        Event::MemoryDecision(decision) => {
            apply_memory_decision(decision, vector_store, graph_store, sqlite).await?;
        }
        _ => {}
    }
    Ok(())
}

async fn apply_memory_decision(
    decision: neko_core::MemoryDecision,
    vector_store: Option<&dyn VectorStore>,
    graph_store: Option<&dyn GraphStore>,
    sqlite: &SqliteStore,
) -> Result<(), NekoError> {
    use neko_core::MemoryUpdate;

    let now = Utc::now();
    for update in decision.updates {
        match update {
            MemoryUpdate::VectorFact(u) => {
                if let Some(store) = vector_store {
                    let record = neko_core::MemoryRecord {
                        id: uuid::Uuid::new_v4().to_string(),
                        group_id: decision.group_id.clone(),
                        speaker_id: u.related_users.first().cloned().unwrap_or_default(),
                        target_id: None,
                        text: u.content,
                        timestamp: now,
                        relation_delta: None,
                        tags: merge_tags(u.tags, &["fact"]),
                        layer: "detective".to_string(),
                    };
                    store.embed_and_upsert(&[record]).await?;
                }
            }
            MemoryUpdate::VectorCulture(u) => {
                if let Some(store) = vector_store {
                    let record = neko_core::MemoryRecord {
                        id: uuid::Uuid::new_v4().to_string(),
                        group_id: decision.group_id.clone(),
                        speaker_id: u.related_users.first().cloned().unwrap_or_default(),
                        target_id: None,
                        text: u.content,
                        timestamp: now,
                        relation_delta: None,
                        tags: merge_tags(u.tags, &["culture"]),
                        layer: "detective".to_string(),
                    };
                    store.embed_and_upsert(&[record]).await?;
                }
            }
            MemoryUpdate::GraphRelation(u) => {
                if let Some(store) = graph_store {
                    store
                        .merge_relation(&u.from, &u.to, &u.relation, u.delta, &u.evidence)
                        .await?;
                }
            }
            MemoryUpdate::AffectiveDelta(u) => {
                sqlite
                    .apply_affective_delta(
                        &decision.group_id,
                        &u.target_user,
                        u.energy_delta,
                        u.favorability_delta,
                        u.mood.as_deref(),
                    )
                    .await?;
            }
        }
    }
    Ok(())
}

fn merge_tags(mut tags: Vec<String>, required: &[&str]) -> Vec<String> {
    for req in required {
        if !tags.iter().any(|t| t == *req) {
            tags.push(req.to_string());
        }
    }
    tags
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
