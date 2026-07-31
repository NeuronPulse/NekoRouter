# AGENTS.md

Event-driven multi-agent QQ group chatbot. Cargo workspace of 10 crates; only binary is `neko-router` (plus a `neko_router` lib target).

## Commands

- `./scripts/test.sh` = `cargo fmt --check` → `cargo clippy --all-targets --all-features -- -D warnings` → `cargo test --all` → `cargo build --release`. Runs green in CI (`.github/workflows/ci.yml`).
- Focused verification: `cargo test -p neko-router` (integration suite), `cargo test -p neko-gate` (unit tests).
- Run the app: `cargo run -p neko-router`.

## Architecture

- 5 layers, each an actor consuming/producing the `Event` enum (`crates/neko-core/src/types.rs`): `neko-sensory` (NapCat/OneBot 11 WS ingress via `napcat-link`, SQLite batch write) → `neko-gate` (cheap LLM, pluggable `GateHeuristic`) → `neko-council` (advanced-model JSON decision) → `neko-detective` (context retrieval + structured report) → `neko-solidify` (cron → Cypher graph updates). See `docs/architecture.md`.
- Central routing is `dispatch_event` in `crates/neko-router/src/router.rs` (lib target, integration-testable). Events flow between actors over `tokio::sync::mpsc` channels; **new layers/event kinds must be registered there** (channel + `dispatch_event` match arm), not inside individual crates. `main.rs` wires the channels and spawns the dispatcher loop.
- Shared traits live in `neko-core` (`LlmClient`, `HistoryStore`, `VectorStore`, `GraphStore`, `Ingress`, `Egress`). New backends implement a trait; all are behind `Arc<dyn Trait + Send + Sync>`.
- LLM is deliberately ONE implementation, `OpenAiCompatibleClient` (`neko-llm`), config-driven (DeepSeek/Grok/Gemini are just TOML provider entries). Don't add per-provider clients. `llm.detective`/`llm.solidify` config keys are optional; when absent the council provider is reused.
- New gate heuristics: implement `GateHeuristic` and register in `heuristic_from_name` (`crates/neko-gate/src/heuristic.rs`); built-ins are `"default"` and `"escalate_all"`.

## Config

- `config/local.toml` is gitignored; copy `config/local.toml.example`. Load order (later wins): `config/default.toml` → `config/{NEKO_ENV}.toml` (default `local`) → `.env` → `NEKO_SECRETS_FILE` → `NEKO__`-prefixed env vars (`__` = nesting separator).
- API keys/passwords use `secrecy::SecretString` with `${VAR}` placeholders resolved from env. Startup **hard-fails** validation if any provider api_key/base_url/model or `websocket.url` is missing.
- Qdrant/Neo4j are optional: empty `qdrant.url` → `InMemoryVectorStore`, empty `neo4j.uri` → `InMemoryGraphStore` (no external service needed to run).
- `solidify.timezone` (IANA name, e.g. `Asia/Shanghai`) is honored via `chrono-tz`; empty means UTC.

## Storage

- SQLite via sqlx; migrations live at `crates/neko-memory/migrations/sqlite/001_init.sql` (resolved relative to the neko-memory crate, **not** a repo-root `migrations/` dir despite `docs/architecture.md`/plan doc suggesting one). DB file `data/neko.db` is created automatically. Migrations are versioned — if you change `001_init.sql`, existing dev DBs keep the old schema (delete `data/neko.db` to recreate).
- `replies.message_id` deliberately has **no FK**: messages are batch-written asynchronously, so a reply can be persisted before its source message is flushed.
- Affective state is persisted to `affective_snapshots` (loaded at sensory startup, saved on each flush interval).
- Dev services: `docker compose -f docker-compose.dev.yml up -d` (Qdrant :6333, Neo4j :7474/:7687, creds `neo4j/password`). For embeddings: `python scripts/local_embedding_server.py` (OpenAI-compatible :8000, `BAAI/bge-small-zh-v1.5` = 512-dim; set `vector_dim` to match).
- `scripts/setup-docker-mirror.sh` configures a China Docker registry mirror (optional, dev-machine convenience).

## Testing quirks

- Integration tests (`crates/neko-router/tests/integration.rs`) need no external services: `MockLlmClient` returns canned responses from a LIFO stack, `MockEgress` records sent replies, DB is in-memory SQLite, assertions use `tokio::time::timeout`. Add pipeline tests following the `spawn_pipeline`/`spawn_dispatcher` patterns.
- The router's `dispatch_event` is tested directly; a full-pipeline test pushes `IncomingMessage` through sensory → gate → dispatcher → `MockEgress`.
- `wiremock` is declared as a dev-dep (`neko-llm`) but is not used anywhere; the plan doc's wiremock/`migrations/sqlite`-at-root claims are stale — trust the code.
- Gate `word_count()` counts whitespace-separated tokens; `char_count()` counts chars (`max_message_length` uses chars, `max_cozy_words` effectively tokens).
- Message ids are deterministic UUIDv5 derived from `group_id:message_id`; the sensory actor drops duplicates it has already seen (dedup after reconnects).

## Misc

- Git repo has no commits yet (all files untracked on `main`).
- Repo docs are Chinese; code comments/commits are English. Match whatever the surrounding file uses.
