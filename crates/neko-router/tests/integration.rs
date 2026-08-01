use async_trait::async_trait;
use chrono::Utc;
use neko_core::{
    AffectiveState, ChatMessage, DetectiveInput, DetectiveReport, Egress, Event, FinishReason,
    HistoryStore, LlmClient, LlmRequest, LlmResponse, NekoError, ReplyCooldown, ReplyOut,
    TokenUsage,
};
use neko_gate::{GateActor, GateConfig};
use neko_memory::SqliteStore;
use neko_sensory::{SensoryActor, SensoryConfig};
use neko_solidify::{InMemoryGraphStore, SolidifyActor, SolidifyConfig};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch, Mutex};
use uuid::Uuid;

/// Mock LLM that returns canned responses.
struct MockLlmClient {
    responses: Mutex<Vec<String>>,
}

impl MockLlmClient {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse, NekoError> {
        let content = self
            .responses
            .lock()
            .await
            .pop()
            .unwrap_or_else(|| "喵".to_string());
        Ok(LlmResponse {
            content,
            usage: TokenUsage::default(),
            finish_reason: FinishReason::Stop,
        })
    }
}

/// Mock egress that records every reply it is asked to send.
#[derive(Default)]
struct MockEgress {
    sent: Arc<Mutex<Vec<ReplyOut>>>,
}

#[async_trait]
impl Egress for MockEgress {
    async fn send(&self, reply: ReplyOut) -> Result<(), NekoError> {
        self.sent.lock().await.push(reply);
        Ok(())
    }
}

fn make_message(content: &str) -> ChatMessage {
    ChatMessage {
        id: Uuid::new_v4(),
        trace_id: Uuid::new_v4(),
        group_id: "12345".to_string(),
        sender: "67890".to_string(),
        nickname: "Alice".to_string(),
        content: content.to_string(),
        timestamp: Utc::now(),
        reply_to: None,
        raw_payload: serde_json::Value::Null,
    }
}

/// Spawn a tiny pipeline: sensory actor -> gate actor.
async fn spawn_pipeline(
    sqlite: SqliteStore,
    llm_responses: Vec<String>,
) -> (
    mpsc::Sender<Event>,
    mpsc::Receiver<Event>,
    watch::Sender<bool>,
) {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (raw_tx, raw_rx) = mpsc::channel(64);
    let (gate_tx, gate_rx) = mpsc::channel(64);
    let (out_tx, out_rx) = mpsc::channel(64);
    let (routed_tx, routed_rx) = mpsc::channel(64);

    let sensory = SensoryActor::new(
        SensoryConfig {
            batch_size: 10,
            flush_interval_ms: 100,
            ..Default::default()
        },
        sqlite,
        shutdown_rx.clone(),
    );

    let gate = GateActor::new(
        GateConfig {
            max_message_length: 100,
            max_cozy_words: 10,
            llm_temperature: 0.7,
            llm_max_tokens: 32,
            concurrency_limit: 2,
            ..Default::default()
        },
        MockLlmClient::new(llm_responses),
    );

    tokio::spawn(async move {
        let _ = sensory.run(raw_rx, routed_rx, gate_tx).await;
    });

    tokio::spawn(async move {
        let _ = gate.run(gate_rx, out_tx).await;
    });

    // Forward ReplyOut events back to the sensory actor so it can update
    // affective state.
    tokio::spawn(async move {
        // The test does not need to observe routed events after this point,
        // but keeping the channel alive prevents send errors.
        let _ = routed_tx;
    });

    (raw_tx, out_rx, shutdown_tx)
}

#[tokio::test]
async fn gate_replies_to_short_message_and_persists_history() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let (raw_tx, mut out_rx, shutdown_tx) =
        spawn_pipeline(sqlite.clone(), vec!["喵~".to_string()]).await;

    raw_tx
        .send(Event::IncomingMessage(make_message("你好")))
        .await
        .unwrap();

    // The gate emits a decision first, then the actual reply.
    let decision = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for gate decision")
        .expect("gate channel closed");
    assert!(
        matches!(decision, Event::GateDecision(_)),
        "expected GateDecision, got {decision:?}"
    );

    let event = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for gate reply")
        .expect("gate channel closed");

    match event {
        Event::ReplyOut(ReplyOut { content, .. }) => {
            assert_eq!(content, "喵~");
        }
        other => panic!("expected ReplyOut, got {other:?}"),
    }

    // Shutdown sensory actor and give it time to flush.
    let _ = shutdown_tx.send(true);
    tokio::time::sleep(Duration::from_millis(300)).await;

    let messages = sqlite
        .query_context(&"12345".to_string(), None, Utc::now(), 10)
        .await
        .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, "你好");
}

#[tokio::test]
async fn gate_drops_too_long_message() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let (raw_tx, mut out_rx, shutdown_tx) = spawn_pipeline(sqlite.clone(), vec![]).await;

    let long_content = "a".repeat(101);
    raw_tx
        .send(Event::IncomingMessage(make_message(&long_content)))
        .await
        .unwrap();

    let event = tokio::time::timeout(Duration::from_secs(1), out_rx.recv())
        .await
        .expect("timeout waiting for gate decision")
        .expect("gate channel closed");

    match event {
        Event::GateDecision(neko_core::GateDecision::Drop(_)) => {}
        other => panic!("expected Drop decision, got {other:?}"),
    }

    // No reply should be generated for a dropped message.
    let maybe_reply = tokio::time::timeout(Duration::from_millis(200), out_rx.recv()).await;
    assert!(maybe_reply.is_err() || maybe_reply.unwrap().is_none());

    let _ = shutdown_tx.send(true);
}

#[tokio::test]
async fn gate_escalate_all_heuristic_sends_escalation() {
    let (gate_tx, gate_rx) = mpsc::channel(16);
    let (out_tx, mut out_rx) = mpsc::channel(16);

    let gate = GateActor::new(
        GateConfig {
            max_message_length: 100,
            max_cozy_words: 10,
            llm_temperature: 0.7,
            llm_max_tokens: 32,
            concurrency_limit: 2,
            heuristic: "escalate_all".to_string(),
        },
        MockLlmClient::new(vec![]),
    );

    tokio::spawn(async move {
        let _ = gate.run(gate_rx, out_tx).await;
    });

    gate_tx
        .send(Event::IncomingMessage(make_message("hello")))
        .await
        .unwrap();

    let event = tokio::time::timeout(Duration::from_secs(1), out_rx.recv())
        .await
        .expect("timeout waiting for gate decision")
        .expect("output channel closed");
    assert!(
        matches!(
            event,
            Event::GateDecision(neko_core::GateDecision::Escalate(_))
        ),
        "expected Escalate decision, got {event:?}"
    );

    let event = tokio::time::timeout(Duration::from_secs(1), out_rx.recv())
        .await
        .expect("timeout waiting for escalation event")
        .expect("output channel closed");
    assert!(
        matches!(event, Event::Escalation(_, _, _)),
        "expected Escalation event, got {event:?}"
    );
}

#[tokio::test]
async fn council_replies_directly_when_decision_is_reply() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let history: Arc<dyn HistoryStore + Send + Sync> = Arc::new(sqlite);

    let (council_tx, council_rx) = mpsc::channel(16);
    let (out_tx, mut out_rx) = mpsc::channel(16);

    let council_llm = MockLlmClient::new(vec![serde_json::json!({
        "action": "reply",
        "reasoning": "用户只是打招呼，直接回复即可。",
        "draft_reply": "你好呀~"
    })
    .to_string()]);

    let council = neko_council::CouncilActor::new(
        neko_council::CouncilConfig {
            context_limit: 5,
            llm_temperature: 0.9,
            llm_max_tokens: 128,
            ..Default::default()
        },
        council_llm,
        history,
    );

    tokio::spawn(async move {
        let _ = council.run(council_rx, out_tx).await;
    });

    let msg = make_message("在吗");
    let state = AffectiveState::default();
    council_tx
        .send(Event::Escalation(
            neko_core::EscalationReason::NeedsContext,
            msg,
            state,
        ))
        .await
        .unwrap();

    let event = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for council output")
        .expect("output channel closed");
    assert!(
        matches!(event, Event::CouncilDecision(_)),
        "expected CouncilDecision, got {event:?}"
    );

    let event = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for council reply")
        .expect("output channel closed");
    match event {
        Event::ReplyOut(ReplyOut { content, layer, .. }) => {
            assert_eq!(content, "你好呀~");
            assert_eq!(layer, "council");
        }
        other => panic!("expected ReplyOut, got {other:?}"),
    }
}

#[tokio::test]
async fn council_launches_detective_when_decision_is_detective() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let history: Arc<dyn HistoryStore + Send + Sync> = Arc::new(sqlite);

    let (council_tx, council_rx) = mpsc::channel(16);
    let (out_tx, mut out_rx) = mpsc::channel(16);

    let council_llm = MockLlmClient::new(vec![serde_json::json!({
        "action": "detective",
        "reasoning": "需要更多上下文。",
        "draft_reply": ""
    })
    .to_string()]);

    let council = neko_council::CouncilActor::new(
        neko_council::CouncilConfig {
            context_limit: 5,
            llm_temperature: 0.9,
            llm_max_tokens: 128,
            ..Default::default()
        },
        council_llm,
        history,
    );

    tokio::spawn(async move {
        let _ = council.run(council_rx, out_tx).await;
    });

    let msg = make_message("你怎么看我？");
    let state = AffectiveState::default();
    council_tx
        .send(Event::Escalation(
            neko_core::EscalationReason::NeedsContext,
            msg,
            state,
        ))
        .await
        .unwrap();

    let event = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for council output")
        .expect("output channel closed");
    assert!(
        matches!(event, Event::CouncilDecision(_)),
        "expected CouncilDecision, got {event:?}"
    );

    let event = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for detective request")
        .expect("output channel closed");
    assert!(
        matches!(event, Event::DetectiveRequest(_)),
        "expected DetectiveRequest, got {event:?}"
    );
}

#[tokio::test]
async fn detective_produces_correlated_report() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let history: Arc<dyn HistoryStore + Send + Sync> = Arc::new(sqlite);

    let (detective_tx, detective_rx) = mpsc::channel(16);
    let (out_tx, mut out_rx) = mpsc::channel(16);

    let report = DetectiveReport {
        target_user: "67890".to_string(),
        summary: "用户喜欢简短回复。".to_string(),
        confidence: 0.8,
        recommended_tone: neko_core::Tone::Warm,
        ..Default::default()
    };
    let detective_llm = MockLlmClient::new(vec![serde_json::to_string(&report).unwrap()]);

    let detective = neko_detective::DetectiveActor::new(
        neko_detective::DetectiveConfig {
            history_limit: 5,
            memory_top_k: 2,
            llm_temperature: 0.7,
            llm_max_tokens: 256,
            fact_dedup_threshold: 0.92,
        },
        detective_llm,
        history,
        Arc::new(neko_detective::InMemoryVectorStore::new()),
    );

    tokio::spawn(async move {
        let _ = detective.run(detective_rx, out_tx).await;
    });

    let msg = make_message("你觉得我怎么样？");
    let message_id = msg.id;
    detective_tx
        .send(Event::DetectiveRequest(DetectiveInput {
            message: msg,
            state: AffectiveState::default(),
            target_user: "67890".to_string(),
        }))
        .await
        .unwrap();

    let event = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for detective report")
        .expect("channel closed");
    match event {
        Event::DetectiveReport(report) => {
            assert_eq!(report.message_id, message_id);
            assert_eq!(report.group_id, "12345");
            assert_eq!(report.target_user, "67890");
        }
        other => panic!("expected DetectiveReport, got {other:?}"),
    }

    // The detective no longer replies directly; the council owns that.
    let maybe_reply = tokio::time::timeout(Duration::from_millis(200), out_rx.recv()).await;
    assert!(maybe_reply.is_err() || maybe_reply.unwrap().is_none());
}

#[tokio::test]
async fn solidify_applies_graph_updates_on_tick() {
    let graph_store = Arc::new(InMemoryGraphStore::new());
    let (solidify_tx, solidify_rx) = mpsc::channel(16);
    let (out_tx, _out_rx) = mpsc::channel(16);

    let llm_response = serde_json::json!({
        "updates": [
            {
                "cypher": "MERGE (a:User {id: $from}) MERGE (b:User {id: $to}) MERGE (a)-[r:TEASE]->(b) SET r.delta = COALESCE(r.delta, 0) + $delta",
                "params": {"from": "67890", "to": "12345", "delta": -0.1}
            }
        ]
    })
    .to_string();

    let solidify_llm = MockLlmClient::new(vec![llm_response]);
    let solidify = SolidifyActor::new(
        SolidifyConfig {
            report_buffer_limit: 10,
            llm_temperature: 0.7,
            llm_max_tokens: 256,
        },
        solidify_llm,
        graph_store.clone(),
    );

    tokio::spawn(async move {
        let _ = solidify.run(solidify_rx, out_tx).await;
    });

    solidify_tx
        .send(Event::DetectiveReport(DetectiveReport {
            target_user: "67890".to_string(),
            summary: "测试报告".to_string(),
            relationship_changes: vec![neko_core::RelationshipChange {
                from: "67890".to_string(),
                to: "12345".to_string(),
                kind: neko_core::RelationKind::Tease,
                delta: -0.1,
                evidence: vec!["开玩笑".to_string()],
            }],
            confidence: 0.9,
            ..Default::default()
        }))
        .await
        .unwrap();

    solidify_tx.send(Event::SolidifyTick).await.unwrap();

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if !graph_store.applied_updates().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("timeout waiting for graph updates");

    let updates = graph_store.applied_updates();
    assert_eq!(updates.len(), 1);
    assert!(updates[0].cypher.contains("MERGE"));
    assert_eq!(
        updates[0].params.get("from"),
        Some(&serde_json::Value::String("67890".to_string()))
    );
}

/// A minimal dispatcher loop that mirrors the routing in `main.rs`.
async fn spawn_dispatcher(
    router_rx: mpsc::Receiver<Event>,
    egress: Arc<MockEgress>,
    sqlite: SqliteStore,
    sensory_routed: mpsc::Sender<Event>,
    council_tx: mpsc::Sender<Event>,
    detective_tx: mpsc::Sender<Event>,
    solidify_tx: mpsc::Sender<Event>,
) {
    tokio::spawn(async move {
        let mut router_rx = router_rx;
        while let Some(event) = router_rx.recv().await {
            let _ = neko_router::router::dispatch_event(
                event,
                egress.as_ref(),
                &sqlite,
                &sensory_routed,
                &council_tx,
                &detective_tx,
                &solidify_tx,
                None,
            )
            .await;
        }
    });
}

#[tokio::test]
async fn router_routes_reply_out_to_egress_sensory_and_sqlite() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let egress = Arc::new(MockEgress::default());
    let (sensory_tx, mut sensory_rx) = mpsc::channel(16);
    let (council_tx, _council_rx) = mpsc::channel(16);
    let (detective_tx, _detective_rx) = mpsc::channel(16);
    let (solidify_tx, _solidify_rx) = mpsc::channel(16);

    let reply = ReplyOut {
        id: Uuid::new_v4(),
        reply_to: Uuid::new_v4(),
        reply_to_platform: None,
        group_id: "12345".to_string(),
        target_user: "67890".to_string(),
        content: "喵".to_string(),
        layer: "gate".to_string(),
        trace_id: Uuid::new_v4(),
    };

    neko_router::router::dispatch_event(
        Event::ReplyOut(reply.clone()),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        None,
    )
    .await
    .unwrap();

    assert_eq!(egress.sent.lock().await.len(), 1);
    assert_eq!(egress.sent.lock().await[0].content, "喵");

    let routed = tokio::time::timeout(Duration::from_secs(1), sensory_rx.recv())
        .await
        .expect("timeout waiting for routed event")
        .expect("sensory channel closed");
    assert!(matches!(routed, Event::ReplyOut(_)));

    assert_eq!(sqlite.count_replies().await.unwrap(), 1);
}

#[tokio::test]
async fn router_persists_gate_and_council_decisions() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let egress = Arc::new(MockEgress::default());
    let (sensory_tx, _sensory_rx) = mpsc::channel(16);
    let (council_tx, _council_rx) = mpsc::channel(16);
    let (detective_tx, _detective_rx) = mpsc::channel(16);
    let (solidify_tx, _solidify_rx) = mpsc::channel(16);

    neko_router::router::dispatch_event(
        Event::GateDecision(neko_core::GateDecision::Drop(neko_core::DropReason::Spam)),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        None,
    )
    .await
    .unwrap();

    neko_router::router::dispatch_event(
        Event::CouncilDecision(neko_core::CouncilDecision {
            action: neko_core::CouncilAction::Ignore,
            reasoning: "no reply needed".to_string(),
            draft_reply: None,
        }),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        None,
    )
    .await
    .unwrap();

    // Neither decision produces a reply.
    assert_eq!(egress.sent.lock().await.len(), 0);
    assert_eq!(sqlite.count_replies().await.unwrap(), 0);
    assert_eq!(sqlite.count_events().await.unwrap(), 2);
}

#[tokio::test]
async fn router_forwards_events_to_downstream_layers() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let egress = Arc::new(MockEgress::default());
    let (sensory_tx, _sensory_rx) = mpsc::channel(16);
    let (council_tx, mut council_rx) = mpsc::channel(16);
    let (detective_tx, mut detective_rx) = mpsc::channel(16);
    let (solidify_tx, mut solidify_rx) = mpsc::channel(16);

    let dispatch = |event: Event| {
        neko_router::router::dispatch_event(
            event,
            egress.as_ref(),
            &sqlite,
            &sensory_tx,
            &council_tx,
            &detective_tx,
            &solidify_tx,
            None,
        )
    };

    let msg = make_message("hello");
    dispatch(Event::Escalation(
        neko_core::EscalationReason::NeedsContext,
        msg,
        AffectiveState::default(),
    ))
    .await
    .unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), council_rx.recv())
            .await
            .expect("timeout waiting for council event")
            .unwrap(),
        Event::Escalation(_, _, _)
    ));

    dispatch(Event::DetectiveRequest(DetectiveInput {
        message: make_message("hi"),
        state: AffectiveState::default(),
        target_user: "67890".to_string(),
    }))
    .await
    .unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), detective_rx.recv())
            .await
            .expect("timeout waiting for detective event")
            .unwrap(),
        Event::DetectiveRequest(_)
    ));

    dispatch(Event::DetectiveReport(DetectiveReport {
        message_id: Uuid::new_v4(),
        group_id: "12345".to_string(),
        target_user: "67890".to_string(),
        ..Default::default()
    }))
    .await
    .unwrap();
    // Reports fan out to the council (for the final reply) and to solidify.
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), council_rx.recv())
            .await
            .expect("timeout waiting for council report")
            .unwrap(),
        Event::DetectiveReport(_)
    ));
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), solidify_rx.recv())
            .await
            .expect("timeout waiting for solidify report")
            .unwrap(),
        Event::DetectiveReport(_)
    ));

    dispatch(Event::SolidifyTick).await.unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), solidify_rx.recv())
            .await
            .expect("timeout waiting for solidify tick")
            .unwrap(),
        Event::SolidifyTick
    ));
}

#[tokio::test]
async fn router_applies_reply_cooldown() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let egress = Arc::new(MockEgress::default());
    let (sensory_tx, _sensory_rx) = mpsc::channel(16);
    let (council_tx, _council_rx) = mpsc::channel(16);
    let (detective_tx, _detective_rx) = mpsc::channel(16);
    let (solidify_tx, _solidify_rx) = mpsc::channel(16);

    let cooldown = ReplyCooldown::new(Duration::from_secs(60));
    let reply = || ReplyOut {
        id: Uuid::new_v4(),
        reply_to: Uuid::new_v4(),
        reply_to_platform: None,
        group_id: "12345".to_string(),
        target_user: "67890".to_string(),
        content: "喵".to_string(),
        layer: "council".to_string(),
        trace_id: Uuid::new_v4(),
    };

    // First reply is allowed; the second, within the interval, is suppressed.
    neko_router::router::dispatch_event(
        Event::ReplyOut(reply()),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        Some(&cooldown),
    )
    .await
    .unwrap();
    neko_router::router::dispatch_event(
        Event::ReplyOut(reply()),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        Some(&cooldown),
    )
    .await
    .unwrap();

    assert_eq!(egress.sent.lock().await.len(), 1);
    assert_eq!(sqlite.count_replies().await.unwrap(), 1);
}

#[tokio::test]
async fn router_persists_reply_cooldown_to_sqlite() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let store: Arc<dyn neko_core::CooldownStore> = Arc::new(sqlite.clone());
    let egress = Arc::new(MockEgress::default());
    let (sensory_tx, _sensory_rx) = mpsc::channel(16);
    let (council_tx, _council_rx) = mpsc::channel(16);
    let (detective_tx, _detective_rx) = mpsc::channel(16);
    let (solidify_tx, _solidify_rx) = mpsc::channel(16);

    let cooldown = ReplyCooldown::new_with_store(Duration::from_secs(60), store.clone())
        .await
        .unwrap();
    let reply = ReplyOut {
        id: Uuid::new_v4(),
        reply_to: Uuid::new_v4(),
        reply_to_platform: None,
        group_id: "12345".to_string(),
        target_user: "67890".to_string(),
        content: "喵".to_string(),
        layer: "council".to_string(),
        trace_id: Uuid::new_v4(),
    };

    neko_router::router::dispatch_event(
        Event::ReplyOut(reply),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        Some(&cooldown),
    )
    .await
    .unwrap();

    // Wait for the async persistence task to finish.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // A fresh cooldown loads the persisted watermark and blocks the next reply.
    let cooldown2 = ReplyCooldown::new_with_store(Duration::from_secs(60), store)
        .await
        .unwrap();
    neko_router::router::dispatch_event(
        Event::ReplyOut(ReplyOut {
            id: Uuid::new_v4(),
            reply_to: Uuid::new_v4(),
            reply_to_platform: None,
            group_id: "12345".to_string(),
            target_user: "67890".to_string(),
            content: "喵".to_string(),
            layer: "council".to_string(),
            trace_id: Uuid::new_v4(),
        }),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        Some(&cooldown2),
    )
    .await
    .unwrap();

    assert_eq!(egress.sent.lock().await.len(), 1);
    assert_eq!(sqlite.count_replies().await.unwrap(), 1);
}

#[tokio::test]
async fn router_routes_daily_context_to_council_and_sqlite() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let egress = Arc::new(MockEgress::default());
    let (sensory_tx, _sensory_rx) = mpsc::channel(16);
    let (council_tx, mut council_rx) = mpsc::channel(16);
    let (detective_tx, _detective_rx) = mpsc::channel(16);
    let (solidify_tx, _solidify_rx) = mpsc::channel(16);

    neko_router::router::dispatch_event(
        Event::DailyContext("用户 A 与 用户 B 关系亲密。".to_string()),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        None,
    )
    .await
    .unwrap();

    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), council_rx.recv())
            .await
            .expect("timeout waiting for daily context")
            .unwrap(),
        Event::DailyContext(_)
    ));
    assert_eq!(sqlite.count_events().await.unwrap(), 1);
}

#[tokio::test]
async fn full_pipeline_replies_through_router() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let (raw_tx, raw_rx) = mpsc::channel(64);
    let (gate_tx, gate_rx) = mpsc::channel(64);
    let (router_tx, router_rx) = mpsc::channel(64);
    let (sensory_routed_tx, sensory_routed_rx) = mpsc::channel(64);
    let (council_tx, _council_rx) = mpsc::channel(16);
    let (detective_tx, _detective_rx) = mpsc::channel(16);
    let (solidify_tx, _solidify_rx) = mpsc::channel(16);

    let egress = Arc::new(MockEgress::default());

    let sensory = SensoryActor::new(
        SensoryConfig {
            batch_size: 10,
            flush_interval_ms: 100,
            ..Default::default()
        },
        sqlite.clone(),
        shutdown_rx,
    );
    let gate = GateActor::new(
        GateConfig {
            max_message_length: 100,
            max_cozy_words: 10,
            llm_temperature: 0.7,
            llm_max_tokens: 32,
            concurrency_limit: 2,
            ..Default::default()
        },
        MockLlmClient::new(vec!["喵~".to_string()]),
    );

    tokio::spawn(async move {
        let _ = sensory.run(raw_rx, sensory_routed_rx, gate_tx).await;
    });
    tokio::spawn(async move {
        let _ = gate.run(gate_rx, router_tx).await;
    });
    spawn_dispatcher(
        router_rx,
        egress.clone(),
        sqlite.clone(),
        sensory_routed_tx,
        council_tx,
        detective_tx,
        solidify_tx,
    )
    .await;

    raw_tx
        .send(Event::IncomingMessage(make_message("你好")))
        .await
        .unwrap();

    // The message eventually comes back as a reply via the real router path.
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if !egress.sent.lock().await.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("timeout waiting for reply through the router");

    let sent = egress.sent.lock().await;
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].content, "喵~");
    assert_eq!(sent[0].layer, "gate");
    drop(sent);

    assert_eq!(sqlite.count_replies().await.unwrap(), 1);
    // Gate decision is also recorded as an event.
    assert_eq!(sqlite.count_events().await.unwrap(), 1);

    let _ = shutdown_tx.send(true);
}

#[tokio::test]
async fn detective_feedback_loop_replies_through_router() {
    let sqlite = SqliteStore::connect_in_memory().await.unwrap();
    let history: Arc<dyn HistoryStore + Send + Sync> = Arc::new(sqlite.clone());

    let (router_tx, router_rx) = mpsc::channel(64);
    let (sensory_tx, _sensory_rx) = mpsc::channel(16);
    let (council_tx, council_rx) = mpsc::channel(16);
    let (detective_tx, detective_rx) = mpsc::channel(16);
    let (solidify_tx, _solidify_rx) = mpsc::channel(16);

    let egress = Arc::new(MockEgress::default());

    let council = neko_council::CouncilActor::new(
        neko_council::CouncilConfig::default(),
        MockLlmClient::new(vec![serde_json::json!({
            "action": "detective",
            "reasoning": "需要更多上下文",
            "draft_reply": ""
        })
        .to_string()]),
        history,
    );
    let detective_report = DetectiveReport {
        target_user: "67890".to_string(),
        summary: "用户喜欢简短回复。".to_string(),
        confidence: 0.9,
        recommended_tone: neko_core::Tone::Warm,
        ..Default::default()
    };
    let detective = neko_detective::DetectiveActor::new(
        neko_detective::DetectiveConfig::default(),
        MockLlmClient::new(vec![serde_json::to_string(&detective_report).unwrap()]),
        Arc::new(neko_memory::SqliteStore::connect_in_memory().await.unwrap()),
        Arc::new(neko_detective::InMemoryVectorStore::new()),
    );

    let council_tx_actor = router_tx.clone();
    tokio::spawn(async move {
        let _ = council.run(council_rx, council_tx_actor).await;
    });
    tokio::spawn(async move {
        let _ = detective.run(detective_rx, router_tx).await;
    });
    spawn_dispatcher(
        router_rx,
        egress.clone(),
        sqlite.clone(),
        sensory_tx.clone(),
        council_tx.clone(),
        detective_tx.clone(),
        solidify_tx.clone(),
    )
    .await;

    // An escalation that the council hands to the detective.
    let msg = make_message("你怎么看我？");
    neko_router::router::dispatch_event(
        Event::Escalation(
            neko_core::EscalationReason::NeedsContext,
            msg,
            AffectiveState::default(),
        ),
        egress.as_ref(),
        &sqlite,
        &sensory_tx,
        &council_tx,
        &detective_tx,
        &solidify_tx,
        None,
    )
    .await
    .unwrap();

    // council -> detective -> report -> council reply -> egress
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if !egress.sent.lock().await.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("timeout waiting for feedback-loop reply");

    let sent = egress.sent.lock().await;
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].content, "温柔地 用户喜欢简短回复。");
    assert_eq!(sent[0].layer, "council");
    drop(sent);

    assert_eq!(sqlite.count_replies().await.unwrap(), 1);
}
