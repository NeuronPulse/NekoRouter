use neko_config::{NekoConfig, Neo4jConfig, ResponseFormatConfig};
use neko_core::{
    CooldownStore, Egress, Event, GraphStore, HistoryStore, Ingress, NekoError, ResponseFormat,
    RuntimeState, VectorStore,
};
use neko_detective::{DetectiveActor, DetectiveConfig, InMemoryVectorStore, QdrantVectorStore};
use neko_gate::{BotIdentity, GateActor, GateConfig};
use neko_llm::{OpenAiCompatibleClient, OpenAiEmbeddingClient};
use neko_memory::SqliteStore;
use neko_sensory::{NapCatEgress, NapCatIngress, SensoryActor, SensoryConfig};
use neko_solidify::{InMemoryGraphStore, Neo4jGraphStore, SolidifyActor, SolidifyConfig};
use secrecy::ExposeSecret;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio_cron_scheduler::{JobBuilder, JobScheduler};
use tracing::{error, info, warn};

mod observe;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    if std::env::args().any(|a| a == "--observe") {
        return observe::run().await;
    }

    let config = NekoConfig::load()?;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let router = Router::new(config, shutdown_rx).await?;

    // Spawn signal handler before starting the router so shutdown is always
    // triggerable.
    tokio::spawn(shutdown_signal_handler(shutdown_tx));

    router.run().await?;

    Ok(())
}

async fn shutdown_signal_handler(shutdown: watch::Sender<bool>) {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("cannot install SIGTERM handler");
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .expect("cannot install SIGINT handler");

    tokio::select! {
        _ = sigterm.recv() => info!("received SIGTERM"),
        _ = sigint.recv() => info!("received SIGINT"),
    }

    let _ = shutdown.send(true);
}

struct Router {
    config: NekoConfig,
    shutdown: watch::Receiver<bool>,
    sqlite: SqliteStore,
}

impl Router {
    async fn new(config: NekoConfig, shutdown: watch::Receiver<bool>) -> Result<Self, NekoError> {
        let sqlite = SqliteStore::connect(&config.sqlite.path).await?;
        Ok(Self {
            config,
            shutdown,
            sqlite,
        })
    }

    async fn run(self) -> Result<(), NekoError> {
        // Channels between layers.
        let (raw_tx, raw_rx) = mpsc::channel::<Event>(4096);
        let (gate_tx, gate_rx) = mpsc::channel::<Event>(1024);
        let (router_tx, mut router_rx) = mpsc::channel::<Event>(1024);
        let (sensory_routed_tx, sensory_routed_rx) = mpsc::channel::<Event>(1024);
        let (council_tx, council_rx) = mpsc::channel::<Event>(1024);
        let (detective_tx, detective_rx) = mpsc::channel::<Event>(1024);
        let (solidify_tx, solidify_rx) = mpsc::channel::<Event>(1024);

        let runtime_state = Arc::new(RuntimeState {
            started_at: chrono::Utc::now(),
            ..Default::default()
        });

        let sqlite_arc: Arc<dyn HistoryStore + Send + Sync> = Arc::new(self.sqlite.clone());
        let vector_store: Arc<dyn VectorStore + Send + Sync> =
            build_vector_store(&self.config.qdrant, &self.config.embedding).await?;
        let graph_store: Arc<dyn GraphStore + Send + Sync> =
            build_graph_store(&self.config.neo4j).await?;

        // Layer 1: sensory.
        let sensory_config = SensoryConfig {
            batch_size: self.config.sqlite.batch_size,
            flush_interval_ms: self.config.sqlite.flush_interval_ms,
            energy_decay_per_min: self.config.personality.energy_decay_per_min,
            favor_decay_per_min: self.config.personality.favor_decay_per_min,
            max_recent_messages: 1024,
        };
        let sensory_actor =
            SensoryActor::new(sensory_config, self.sqlite.clone(), self.shutdown.clone());

        // Layer 2: gate.
        let gate_config = GateConfig {
            max_message_length: self.config.personality.max_message_length,
            max_ambient_words: self.config.personality.max_cozy_words,
            concurrency_limit: 8,
            heuristic: "default".to_string(),
        };
        let bot_identity = BotIdentity {
            qq_id: self.config.bot.qq_id,
            name: self.config.bot.name.clone(),
            aliases: vec!["猫娘".to_string(), "机器人".to_string()],
        };
        let gate_actor =
            GateActor::new_with_state(gate_config, bot_identity, Some(runtime_state.clone()));

        // Layer 3: council.
        let council_provider = self.config.council_provider()?;
        let council_llm = OpenAiCompatibleClient::new(
            &self.config.llm.council,
            &council_provider.base_url,
            &council_provider.model,
            council_provider.api_key.clone(),
            council_provider.temperature,
            council_provider.max_tokens,
            match council_provider.response_format {
                ResponseFormatConfig::Text => ResponseFormat::Text,
                ResponseFormatConfig::Json => ResponseFormat::JsonObject,
            },
        )?;
        let council_config = neko_council::CouncilConfig {
            context_limit: 10,
            llm_temperature: council_provider.temperature,
            llm_max_tokens: council_provider.max_tokens.unwrap_or(512),
            detective_timeout: std::time::Duration::from_secs(300),
        };
        let council_actor =
            neko_council::CouncilActor::new(council_config, council_llm, sqlite_arc.clone());

        // Layer 4: detective. Uses its own provider when configured,
        // otherwise reuses the council provider.
        let detective_provider = self.config.detective_provider()?;
        let detective_config = DetectiveConfig {
            history_limit: 20,
            memory_top_k: 5,
            llm_temperature: detective_provider.temperature,
            llm_max_tokens: detective_provider.max_tokens.unwrap_or(1024),
            fact_dedup_threshold: 0.92,
        };
        let detective_llm = OpenAiCompatibleClient::new(
            "detective",
            &detective_provider.base_url,
            &detective_provider.model,
            detective_provider.api_key.clone(),
            detective_provider.temperature,
            detective_provider.max_tokens,
            ResponseFormat::JsonObject,
        )?;
        let detective_actor = DetectiveActor::new(
            detective_config,
            detective_llm,
            sqlite_arc.clone(),
            vector_store.clone(),
        );

        // Layer 5: solidify. Uses its own provider when configured,
        // otherwise reuses the council provider.
        let solidify_provider = self.config.solidify_provider()?;
        let solidify_config = SolidifyConfig {
            report_buffer_limit: 100,
            llm_temperature: solidify_provider.temperature,
            llm_max_tokens: solidify_provider.max_tokens.unwrap_or(1024),
        };
        let solidify_llm = OpenAiCompatibleClient::new(
            "solidify",
            &solidify_provider.base_url,
            &solidify_provider.model,
            solidify_provider.api_key.clone(),
            solidify_provider.temperature,
            solidify_provider.max_tokens,
            ResponseFormat::JsonObject,
        )?;
        let solidify_actor = SolidifyActor::new(solidify_config, solidify_llm, graph_store.clone());

        // Ingress / egress.
        let napcat_token = if self.config.websocket.token.is_empty() {
            None
        } else {
            Some(self.config.websocket.token.clone())
        };
        let ingress = NapCatIngress::new(
            &self.config.websocket.url,
            napcat_token.clone(),
            Duration::from_secs(self.config.websocket.reconnect_interval_sec),
        );
        let egress: Arc<dyn Egress + Send + Sync> =
            Arc::new(NapCatEgress::new(&self.config.websocket.url, napcat_token));

        // Cron scheduler for solidify ticks.
        let mut sched =
            build_solidify_scheduler(&self.config.solidify, solidify_tx.clone()).await?;

        let shutdown_for_ingress = self.shutdown.clone();
        let ingress_for_task = ingress.clone();
        let ingress_handle = tokio::spawn(async move {
            if let Err(e) = ingress_for_task.run(raw_tx, shutdown_for_ingress).await {
                error!("ingress task error: {e}");
            }
        });

        // Runtime status HTTP endpoint.
        let status_bind_addr = format!("{}:{}", self.config.status.host, self.config.status.port);
        let status_ingress: Arc<dyn Ingress + Send + Sync> = Arc::new(ingress.clone());
        let status_state = runtime_state.clone();
        let shutdown_for_status = self.shutdown.clone();
        let _status_handle = tokio::spawn(async move {
            if let Err(e) = neko_router::status::run_status_server(
                status_bind_addr,
                status_state,
                status_ingress,
                shutdown_for_status,
            )
            .await
            {
                error!("status server error: {e}");
            }
        });

        let sensory_handle = tokio::spawn(async move {
            if let Err(e) = sensory_actor.run(raw_rx, sensory_routed_rx, gate_tx).await {
                error!("sensory actor error: {e}");
            }
        });

        let router_tx_for_gate = router_tx.clone();
        let gate_handle = tokio::spawn(async move {
            if let Err(e) = gate_actor.run(gate_rx, router_tx_for_gate).await {
                error!("gate actor error: {e}");
            }
        });

        let router_tx_for_council = router_tx.clone();
        let council_handle = tokio::spawn(async move {
            if let Err(e) = council_actor.run(council_rx, router_tx_for_council).await {
                error!("council actor error: {e}");
            }
        });

        let router_tx_for_detective = router_tx.clone();
        let detective_handle = tokio::spawn(async move {
            if let Err(e) = detective_actor
                .run(detective_rx, router_tx_for_detective)
                .await
            {
                error!("detective actor error: {e}");
            }
        });

        let router_tx_for_solidify = router_tx.clone();
        let solidify_handle = tokio::spawn(async move {
            if let Err(e) = solidify_actor
                .run(solidify_rx, router_tx_for_solidify)
                .await
            {
                error!("solidify actor error: {e}");
            }
        });

        // Dispatcher: route events from all layers to the right downstream
        // channel and persist observability records.
        let mut shutdown_for_dispatcher = self.shutdown.clone();
        let sqlite_for_dispatcher = self.sqlite.clone();
        let cooldown_store: Arc<dyn CooldownStore> = Arc::new(self.sqlite.clone());
        let reply_cooldown = neko_core::ReplyCooldown::new_with_store(
            std::time::Duration::from_secs(self.config.personality.min_reply_interval_sec),
            cooldown_store,
        )
        .await?;
        let dispatcher_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = shutdown_for_dispatcher.changed() => {
                        if *shutdown_for_dispatcher.borrow() {
                            break;
                        }
                    }
                    Some(event) = router_rx.recv() => {
                            if let Err(e) = neko_router::router::dispatch_event_with_state(
                                event,
                                egress.as_ref(),
                                &sqlite_for_dispatcher,
                                &sensory_routed_tx,
                                &council_tx,
                                &detective_tx,
                                &solidify_tx,
                                Some(&reply_cooldown),
                                Some(&runtime_state),
                            ).await {
                                error!("dispatch error: {e}");
                            }
                        }
                }
            }
        });

        info!("NekoRouter started");

        // Wait for shutdown.
        let mut shutdown = self.shutdown;
        shutdown.changed().await.ok();

        info!("NekoRouter shutting down");

        // Stop cron scheduler cleanly.
        if let Err(e) = sched.shutdown().await {
            warn!("cron scheduler shutdown error: {e}");
        }

        // Wait for layers to finish.
        let _ = tokio::join!(
            ingress_handle,
            sensory_handle,
            gate_handle,
            council_handle,
            detective_handle,
            solidify_handle,
            dispatcher_handle,
        );
        info!("NekoRouter stopped");
        Ok(())
    }
}

async fn build_vector_store(
    qdrant: &neko_config::QdrantConfig,
    embedding: &neko_config::EmbeddingConfig,
) -> Result<Arc<dyn VectorStore + Send + Sync>, NekoError> {
    if qdrant.url.is_empty() {
        info!("qdrant.url is empty, using in-memory vector store");
        return Ok(Arc::new(InMemoryVectorStore::new()));
    }

    info!("connecting to Qdrant at {}", qdrant.url);
    let embedding_client = OpenAiEmbeddingClient::new(
        "embedding",
        &embedding.base_url,
        &embedding.model,
        embedding.api_key.clone(),
        qdrant.vector_dim,
    )?;

    let store = QdrantVectorStore::new(
        &qdrant.url,
        None::<String>,
        &qdrant.collection,
        Arc::new(embedding_client),
    )?;
    Ok(Arc::new(store))
}

async fn build_graph_store(
    config: &Neo4jConfig,
) -> Result<Arc<dyn GraphStore + Send + Sync>, NekoError> {
    if config.uri.is_empty() {
        info!("neo4j.uri is empty, using in-memory graph store");
        return Ok(Arc::new(InMemoryGraphStore::new()));
    }

    info!("connecting to Neo4j at {}", config.uri);
    let store =
        Neo4jGraphStore::new(&config.uri, &config.user, config.password.expose_secret()).await?;
    Ok(Arc::new(store))
}

async fn build_solidify_scheduler(
    config: &neko_config::SolidifyConfig,
    solidify_tx: mpsc::Sender<Event>,
) -> Result<JobScheduler, NekoError> {
    let sched = JobScheduler::new()
        .await
        .map_err(|e| NekoError::other(format!("cannot create cron scheduler: {e}")))?;

    // Resolve the configured timezone. Empty means UTC (the default).
    let tz: chrono_tz::Tz = if config.timezone.is_empty() {
        chrono_tz::UTC
    } else {
        config.timezone.parse().map_err(|e| {
            NekoError::config(format!(
                "invalid solidify timezone '{}': {e}",
                config.timezone
            ))
        })?
    };

    let job = JobBuilder::new()
        .with_timezone(tz)
        .with_cron_job_type()
        .with_schedule(config.cron.as_str())
        .map_err(|e| NekoError::config(format!("invalid solidify cron expression: {e}")))?
        .with_run_async(Box::new(move |_uuid, _lock| {
            let tx = solidify_tx.clone();
            Box::pin(async move {
                info!("cron triggered SolidifyTick");
                if tx.send(Event::SolidifyTick).await.is_err() {
                    warn!("solidify channel closed, cannot send tick");
                }
            })
        }))
        .build()
        .map_err(|e| NekoError::other(format!("cannot build solidify job: {e}")))?;

    sched
        .add(job)
        .await
        .map_err(|e| NekoError::other(format!("cannot add solidify job: {e}")))?;
    sched
        .start()
        .await
        .map_err(|e| NekoError::other(format!("cannot start cron scheduler: {e}")))?;

    Ok(sched)
}
