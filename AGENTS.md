# AGENTS.md

Event-driven multi-agent QQ group chatbot. Cargo workspace of 10 crates; only binary is `neko-router` (plus a `neko_router` lib target).

## Commands

- `./scripts/test.sh` = `cargo fmt --check` → `cargo clippy --all-targets --all-features -- -D warnings` → `cargo test --all` → `cargo build --release`. Runs green in CI (`.github/workflows/ci.yml`).
- Focused verification: `cargo test -p neko-router` (integration suite), `cargo test -p neko-gate` (unit tests).
- Run the app: `cargo run -p neko-router`.
- Observability: `cargo run -p neko-router -- --observe [--limit N]` (read-only DB summary; uses `NekoConfig::load_observe`, so it works without API keys).

## Architecture

- 5 layers, each an actor consuming/producing the `Event` enum (`crates/neko-core/src/types.rs`): `neko-sensory` (NapCat/OneBot 11 WS ingress via `napcat-link`, SQLite batch write) → `neko-gate` (cheap LLM, pluggable `GateHeuristic`) → `neko-council` (advanced-model JSON decision) → `neko-detective` (context retrieval + structured report) → `neko-solidify` (cron → Cypher graph updates). See `docs/architecture.md`.
- Central routing is `dispatch_event` in `crates/neko-router/src/router.rs` (lib target, integration-testable). Events flow between actors over `tokio::sync::mpsc` channels; **new layers/event kinds must be registered there** (channel + `dispatch_event` match arm), not inside individual crates. `main.rs` wires the channels and spawns the dispatcher loop.
- Shared traits live in `neko-core` (`LlmClient`, `HistoryStore`, `VectorStore`, `GraphStore`, `Ingress`, `Egress`). New backends implement a trait; all are behind `Arc<dyn Trait + Send + Sync>`.
- LLM is deliberately ONE implementation, `OpenAiCompatibleClient` (`neko-llm`), config-driven (DeepSeek/Grok/Gemini are just TOML provider entries). Don't add per-provider clients. `llm.detective`/`llm.solidify` config keys are optional; when absent the council provider is reused.
- The **council is the single reply authority**. The detective never emits `ReplyOut`: it tags its report with `message_id`/`group_id`/`target_user` and sends `Event::DetectiveReport`, which `dispatch_event` fans out to both the council (which correlates it against `pending_detective` and composes the final reply, skipping low-confidence/empty summaries) and solidify. Council state that must survive per-event task clones lives behind `Arc` (`DashMap` clones deep-copy).
- Council detective escalations have a timeout: `pending_detective` entries are swept by a `tokio::select!` interval (`council.detective_timeout`, default 300s) so late reports are ignored and the map can't grow unbounded.
- The council consumes `Event::DailyContext` (solidify's post-cron relationship summary) into an `Arc<RwLock<Option<String>>>` injected into the council prompt as long-term background.
- Solidify, after applying graph updates, sends `Event::DailyContext` built from `GraphStore::relationship_summary(limit)` (top `|delta|` relationships).
- Reply rate limiting: `dispatch_event` takes `cooldown: Option<&ReplyCooldown>` (`neko-core/src/cooldown.rs`, per-group `Instant` watermark); an in-cooldown `ReplyOut` is dropped without persisting. Interval is `personality.min_reply_interval_sec` (default 3). **New `dispatch_event` callers must pass the cooldown arg.**
- The detective learns long-term facts: non-blank `historical_facts` in a report are written back to the vector store as `MemoryRecord`s (tags `["fact","detective"]`, speaker = target user).
- NapCat ingress pushes downstream with `try_send`; a full channel drops the message and bumps `drop_count` (observable via `ingress.drop_count()`) instead of blocking the event loop.
- New gate heuristics: implement `GateHeuristic` and register in `heuristic_from_name` (`crates/neko-gate/src/heuristic.rs`); built-ins are `"default"` and `"escalate_all"`.

## Config

- `config/local.toml` is gitignored; copy `config/local.toml.example`. Load order (later wins): `config/default.toml` → `config/{NEKO_ENV}.toml` (default `local`) → `.env` → `NEKO_SECRETS_FILE` → `NEKO__`-prefixed env vars (`__` = nesting separator).
- API keys/passwords use `secrecy::SecretString` with `${VAR}` placeholders resolved from env. Startup **hard-fails** validation if any provider api_key/base_url/model or `websocket.url` is missing.
- Qdrant/Neo4j are optional: empty `qdrant.url` → `InMemoryVectorStore`, empty `neo4j.uri` → `InMemoryGraphStore` (no external service needed to run).
- `solidify.timezone` (IANA name, e.g. `Asia/Shanghai`) is honored via `chrono-tz`; empty means UTC.
- `personality.min_reply_interval_sec` (default 3) drives the per-group reply cooldown. `council.detective_timeout` (default 300s) controls how long detective escalations wait before being swept.

## Storage

- SQLite via sqlx; migrations live at `crates/neko-memory/migrations/sqlite/001_init.sql` (resolved relative to the neko-memory crate, **not** a repo-root `migrations/` dir despite `docs/architecture.md`/plan doc suggesting one). DB file `data/neko.db` is created automatically. Migrations are versioned — if you change `001_init.sql`, existing dev DBs keep the old schema (delete `data/neko.db` to recreate).
- `replies.message_id` deliberately has **no FK**: messages are batch-written asynchronously, so a reply can be persisted before its source message is flushed.
- Affective state is persisted to `affective_snapshots` (loaded at sensory startup, saved on each flush interval).
- Dev services: `docker compose -f docker-compose.dev.yml up -d` (Qdrant REST :6333 + gRPC :6334, Neo4j :7474/:7687, creds `neo4j/password`). `qdrant.url` is the **gRPC** endpoint (6334) — `qdrant-client` is gRPC-only, pointing at 6333 fails. For embeddings: `python scripts/local_embedding_server.py` (OpenAI-compatible :8000, `BAAI/bge-small-zh-v1.5` = 512-dim; set `vector_dim` to match).
- `--observe` reads the same SQLite DB the app writes (`recent_messages`/`recent_replies`/`recent_events` + counts).
- `scripts/setup-docker-mirror.sh` configures a China Docker registry mirror (optional, dev-machine convenience).

## Testing quirks

- Integration tests (`crates/neko-router/tests/integration.rs`) need no external services: `MockLlmClient` returns canned responses from a LIFO stack, `MockEgress` records sent replies, DB is in-memory SQLite, assertions use `tokio::time::timeout`. Add pipeline tests following the `spawn_pipeline`/`spawn_dispatcher` patterns.
- The router's `dispatch_event` is tested directly; a full-pipeline test pushes `IncomingMessage` through sensory → gate → dispatcher → `MockEgress`.
- `wiremock` is declared as a dev-dep (`neko-llm`) but is not used anywhere; the plan doc's wiremock/`migrations/sqlite`-at-root claims are stale — trust the code.
- Gate `word_count()` counts whitespace-separated tokens; `char_count()` counts chars (`max_message_length` uses chars, `max_cozy_words` effectively tokens).
- Message ids are deterministic UUIDv5 derived from `group_id:message_id`; the sensory actor drops duplicates it has already seen (dedup after reconnects).
- Real-service integration tests live in `crates/neko-detective/tests/qdrant_integration.rs` and `crates/neko-solidify/tests/neo4j_integration.rs`. They **skip gracefully at runtime** (TCP probe) when the docker services are down, so `cargo test --all` stays green in CI; start `docker compose -f docker-compose.dev.yml up -d` to run them for real. Qdrant point ids must be valid UUIDs.
- `SqliteStore::connect` uses `SqliteConnectOptions::create_if_missing(true)` — a plain `sqlite:` URL parses to `create_if_missing=false` and would fail to open a fresh DB file.

## Misc

- Git repo has commits on `main` (initial commit `ac80ea4`).
- Repo docs are Chinese; code comments/commits are English. Match whatever the surrounding file uses.
