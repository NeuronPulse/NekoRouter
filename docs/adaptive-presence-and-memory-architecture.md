# Adaptive Presence & Unified Memory Architecture

> 目标：让 NekoRouter 在 QQ 群里表现得像一个“局内人”，而不是一个只会被 @ 才回复的机器人。
> 版本：2026-08-02，对应当前代码实现。

## 1. 设计目标

1. **情境感知的参与**：不是每条消息都回，也不是只回 @ 自己的消息；而是由模型根据上下文判断“现在插话是否自然”。
2. **统一记忆**：把 Qdrant（事实/梗）、Neo4j（人际关系）、SQLite（历史/情感状态）打通，所有跨库联系由 LLM 判断，而不是写死的规则。
3. **持续学习**：即使没有回复需求，侦探也会在群聊“热闹”时（TopicBurst）主动整理记忆。
4. **单一发言权威**：所有最终回复仍由 council 产生，避免多层各自发声。
5. **可解释、可观测**：每个关键决策（gate、council、detective、memory）都落入 SQLite `events` 表，便于复盘。

## 2. 核心设计原则

- **LLM 优先于硬编码**：参与时机、记忆归类、关系变化尽量交给模型判断；代码只负责“结构化输入 / 解析输出 / 分发存储”。
- **事件驱动、通道解耦**：每一层都是 actor，通过 `tokio::sync::mpsc` 收发 `Event`。
- **新增事件必须在 router 注册**：`crates/neko-router/src/router.rs` 的 `dispatch_event_full` 是中央路由；新的 layer/event 必须在这里加 match arm。
- **降级友好**：Qdrant/Neo4j 未配置时自动使用 `InMemoryVectorStore` / `InMemoryGraphStore`，不需要外部服务也能跑测试。
- **状态可持久化**：情感状态、回复冷却水线都落到 SQLite，重启不丢。

## 3. 总体架构

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ OneBot 11 / NapCat WebSocket                                              │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ Event::IncomingMessage
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 1: neko-sensory                                                     │
│ - 去重、维护 affective state                                              │
│ - 批量写入 SQLite messages / affective_snapshots                          │
│ - 维护 per-group 滑动窗口，检测 TopicBurst → Event::TopicBurst            │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │ Event::IncomingMessage / Event::AffectiveUpdated
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 2: neko-gate                                                        │
│ - GateClassifier：DefaultHeuristic / EscalateAll / LlmGateClassifier      │
│ - 输出 GateDecision::Drop / Escalate(_, EngagementType)                   │
│ - Escalate 生成 Event::Escalation(_, msg, state, engagement_type)         │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │ Event::Escalation
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 3: neko-council                                                     │
│ - 唯一回复权威：决定 Reply / LaunchDetective / Ignore                     │
│ - 根据 EngagementType 决定 prompt 风格                                    │
│ - 接收 DailyContext（graph summary）作为长期背景                          │
│ - detective 报告超时清理 pending_detective                                │
└────────────┬─────────────────────────────┬───────────────────────────────┘
             │ Event::ReplyOut             │ Event::DetectiveRequest
             ▼                             ▼
┌─────────────────────────┐  ┌─────────────────────────────────────────────┐
│ neko-sensory / egress   │  │ Layer 4: neko-detective                      │
│ - 发送 QQ 消息           │  │ - 按需侦探：DetectiveInput → DetectiveReport │
│ - 更新自身 affective     │  │ - 持续策展：TopicBurst → MemoryDecision      │
│                         │  │ - 事实去重：embedding 相似度阈值             │
└─────────────────────────┘  └──────────────┬──────────────────────────────┘
                                            │ Event::DetectiveReport
                                            │ Event::MemoryDecision
                                            ▼
                           ┌──────────────────────────────────────────────┐
                           │ Layer 5: neko-solidify                        │
                           │ - cron 触发 SolidifyTick                      │
                           │ - 聚合 DetectiveReport → Neo4j graph updates  │
                           │ - 生成 DailyContext 回送 council              │
└──────────────────────────────────────────┴────────────────────────────────┘
                                    │
                                    ▼
                         ┌───────────────────────┐
                         │ SQLite / Qdrant / Neo4j│
                         └───────────────────────┘
```

## 4. 事件总线与路由

所有事件定义在 `crates/neko-core/src/types.rs`：

```rust
pub enum Event {
    IncomingMessage(ChatMessage),
    AffectiveUpdated(GroupId, UserId, AffectiveState),
    GateDecision(GateDecision),
    Escalation(EscalationReason, ChatMessage, AffectiveState, EngagementType),
    CouncilDecision(CouncilDecision),
    DetectiveRequest(DetectiveInput),
    DetectiveReport(DetectiveReport),
    ReplyOut(ReplyOut),
    SolidifyTick,
    DailyContext(String),
    TopicBurst(TopicBurst),
    MemoryDecision(MemoryDecision),
}
```

中央路由在 `crates/neko-router/src/router.rs` 的 `dispatch_event_full`：

| 事件 | 处理 |
|------|------|
| `ReplyOut` | 应用冷却、解析 quote id、持久化、调用 egress、回送 sensory |
| `Escalation` | 转发 council |
| `DetectiveRequest` | 转发 detective |
| `DetectiveReport` | 同时转发 council + solidify |
| `TopicBurst` | 转发 detective |
| `MemoryDecision` | 就地应用：向量事实/文化、图关系、情感增量 |
| `DailyContext` | 持久化 + 转发 council |
| `GateDecision` / `CouncilDecision` | 写入 `events` 表 |

## 5. Layer 1：Sensory（感知层）

职责：

1. **消息去重**：基于 `message_id` 的 UUIDv5 去重，避免 WebSocket 重连后重复处理。
2. **情感状态维护**：per `(group_id, user_id)` 维护 `AffectiveState`（energy / favorability / reply_count）。
3. **批量持久化**：把消息批量写入 SQLite `messages`，情感状态写入 `affective_snapshots`。
4. **TopicBurst 检测**：为每个群维护一个时间窗口，当满足阈值时发出 `Event::TopicBurst`。

### 5.1 TopicBurst 算法

维护每个群的最近消息窗口 `burst_windows: DashMap<GroupId, Vec<ChatMessage>>`。

触发条件（全部满足）：

- 窗口内不重复发言人数 `>= burst_threshold_participants`（默认 3）。
- 消息频率 `>= burst_threshold_mpm`（默认 6 msg/min）。
- 平均消息间隔 `<= burst_threshold_gap_sec`（默认 30s）。
- 距上次同群 burst 超过 `burst_cooldown_sec`（默认 120s）。

输出 `TopicBurst { group_id, messages, score, detected_at }`，其中 `TopicBurstScore` 包含 mpm、unique_participants、avg_gap_seconds。

## 6. Layer 2：Gate（参与门）

Gate 不再生成回复，只做“是否参与 + 以何种身份参与”的二元/三元分类。

### 6.1 分类结果

```rust
pub enum GateClassification {
    Drop(DropReason),
    Escalate(EngagementType),
}

pub enum EngagementType {
    PersonalReply,  // 被 @、回复机器人、叫到名字 → 认真回应
    AmbientJoin,    // 自然插话 → 简短、俏皮
}
```

### 6.2 分类器接口

```rust
#[async_trait]
pub trait GateClassifier: Send + Sync {
    async fn classify(
        &self,
        msg: &ChatMessage,
        recent_context: &[ChatMessage],
        affective: &AffectiveState,
        config: &GateConfig,
        self_id: &BotIdentity,
    ) -> Result<GateClassification, NekoError>;
}
```

实现：

- `DefaultHeuristic`：零成本规则；过滤命令 / URL / 超长消息；@/回复/叫名为 `PersonalReply`；短闲聊为 `AmbientJoin`。
- `EscalateAllHeuristic`：测试用，总是 `PersonalReply`。
- `LlmGateClassifier`：在硬规则快速过滤后，把最近 `context_messages` 条上下文 + 当前消息 + 机器人情感状态喂给廉价 LLM，输出 JSON `{action, engagement_type, confidence, reasoning}`。

### 6.3 BotIdentity

包含 `qq_id`、`name`、`aliases`。判断被指向的逻辑：

1. 消息 `reply_to` 非空（当前假定回复的就是机器人；后续可由 sensory 在 `raw_payload` 中标记 self-messages 增强）。
2. OneBot `message` 段里存在 `type: "at"` 且 `data.qq == bot.qq_id`。
3. 文本中包含 name 或任意 alias。

## 7. Layer 3：Council（议会层）

唯一能够产生 `ReplyOut` 的层。

### 7.1 输入

`CouncilInput` 包含：

- 触发消息
- 当前 affective state
- 最近历史（`PersonalReply` 按发送者过滤，`AmbientJoin` 取全群上下文）
- `daily_context`（solidify 每晚生成的关系摘要）
- `engagement_type`

### 7.2 输出

`CouncilDecision { action, reasoning, draft_reply }`，`action` 为 `ReplyDirectly` / `LaunchDetective` / `Ignore`。

### 7.3 Detective 协作

- 选择 `LaunchDetective` 后，把 `message_id` 加入 `pending_detective`。
- `DetectiveReport` 回来时，只有对应 `message_id` 仍在 pending 中才会被采纳。
- Pending 项按 `council.detective_timeout`（默认 300s）定期清理，防止无限增长。

### 7.4 长期背景

`solidify` 每晚生成 `DailyContext`，经 router 持久化后送入 council 的 `daily_context`，让长期关系影响回复。

## 8. Layer 4：Detective（侦探层）

两种工作模式：

### 8.1 按需侦探（On-demand）

由 council 通过 `DetectiveRequest` 触发：

1. 从 SQLite 查历史、从 VectorStore 查相关记忆。
2. LLM 生成结构化的 `DetectiveReport`（summary / facts / weaknesses / relationship_changes / recommended_tone / confidence）。
3. 把报告中的 `historical_facts` 去重后写回向量库（tags `["fact","detective"]`），形成学习闭环。

事实去重：用 `VectorStore::search_with_score` 查最相似记录，若 cosine similarity >= `fact_dedup_threshold`（默认 0.92）则跳过。

### 8.2 持续记忆策展（Memory Curator）

由 `Event::TopicBurst` 触发：

1. 把一整段热聊上下文喂给 LLM。
2. LLM 输出 `MemoryDecision { group_id, summary, updates }`。
3. `updates` 是带 `kind` 的多态数组，由 router 分发到不同存储：

```rust
pub enum MemoryUpdate {
    VectorFact(VectorFactUpdate),
    VectorCulture(VectorCultureUpdate),
    GraphRelation(GraphRelationUpdate),
    AffectiveDelta(AffectiveDeltaUpdate),
}
```

LLM 同时决定：

- 这是个人事实还是群文化/梗（`vector_fact` vs `vector_culture`）。
- 这段互动是否改变了两个人之间的关系（`graph_relation`）。
- 某个用户的情感状态是否应微调（`affective_delta`）。

代码不做语义判断，只按 `kind` 分发。

## 9. Layer 5：Solidify（固化层）

职责：

1. 接收 `SolidifyTick`（cron）和 `DetectiveReport`。
2. 把 detective 报告中的 `relationship_changes` 转成 Cypher `GraphUpdate`，写入 Neo4j（或内存图）。
3. 每晚聚合图中变化最大的关系，生成 `DailyContext` 文本摘要，通过 `Event::DailyContext` 回送 council。

## 10. 跨存储联动（MemoryDecision 路由）

`crates/neko-router/src/router.rs` 中的 `apply_memory_decision` 实现：

| Update 类型 | 目标存储 | 记录形式 |
|-------------|----------|----------|
| `VectorFact` | Qdrant / InMemoryVectorStore | `MemoryRecord`，tags 强制加入 `"fact"`，speaker 取 `related_users.first()` |
| `VectorCulture` | Qdrant / InMemoryVectorStore | `MemoryRecord`，tags 强制加入 `"culture"` |
| `GraphRelation` | Neo4j / InMemoryGraphStore | 调用 `GraphStore::merge_relation(...)` |
| `AffectiveDelta` | SQLite | 调用 `SqliteStore::apply_affective_delta(...)` |

这样，梗、人际关系、情感状态都通过同一次 LLM 决策产生，并保持语义关联（例如某个梗被标记到相关用户，图关系记录两人因这个梗产生的调侃关系）。

## 11. 三种记忆的用途

| 存储 | 内容 | 查询方 | 典型标签/关系 |
|------|------|--------|---------------|
| SQLite `messages` | 原始聊天记录 | council / detective 查上下文 | — |
| SQLite `affective_snapshots` | 用户情感状态 | gate / council 调 prompt | — |
| Qdrant / InMemoryVectorStore | 长期事实 + 群文化梗 | detective / council | `fact`, `culture`, `detective` |
| Neo4j / InMemoryGraphStore | 用户间关系边 | solidify 汇总 → council | `TEASE`, `SUPPORT`, `CONFLICT`, … |

## 12. 回复与冷却

- `ReplyOut` 经过 `ReplyCooldown` 的 per-group 水线控制，未通过则直接丢弃。
- 冷却间隔由 `personality.min_reply_interval_sec` 控制（默认 3s）。
- 冷却水线持久化到 SQLite `reply_cooldowns` 表，重启后继续生效。
- 发送前 router 会尝试把内部 `reply_to` 解析成 OneBot `message_id`，支持引用回复；解析失败则降级为纯文本。

## 13. 可观测性

- SQLite `events` 表记录 `gate_decision`、`council_decision`、`daily_context` 等。
- HTTP `/status` 暴露 `RuntimeState`：启动时间、收到消息数、已发送回复数、连接状态。
- `--observe` 模式只读浏览 SQLite，不需要 API key。
- `trace_id` 贯穿日志，便于追踪一条消息在多层之间的流转。

## 14. 配置说明

`config/default.toml` 关键新增/调整：

```toml
[gate]
classifier = "default"        # "default" | "llm" | "escalate_all"
provider = "deepseek"         # classifier="llm" 时使用哪个 provider
context_messages = 10
cache_ttl_sec = 30            # 保留字段，未来用于分类结果缓存
rate_limit_per_min = 60       # 保留字段，未来用于 per-group 限流

[burst]
detection_enabled = true
window_sec = 60
threshold_mpm = 6.0
threshold_participants = 3
threshold_gap_sec = 30.0
cooldown_sec = 120

[personality]
max_cozy_words = 10
max_message_length = 800
energy_decay_per_min = 0.05
favor_decay_per_min = 0.02
min_reply_interval_sec = 3

[llm]
council = "grok"
detective = "grok"
solidify = "grok"
# gate 字段已废弃；gate LLM 由 [gate].provider 指定
```

Qdrant URL 使用 gRPC 端口（默认 `http://127.0.0.1:6334`），空字符串则回退到内存实现。

## 15. 当前实现状态

已实现：

- [x] `GateClassifier` 接口 + `DefaultHeuristic` / `EscalateAllHeuristic` / `LlmGateClassifier`
- [x] `EngagementType::PersonalReply` / `AmbientJoin` 贯穿 gate → council
- [x] `TopicBurst` 检测与 `Event::TopicBurst`
- [x] `MemoryDecision` / `MemoryUpdate` 类型与解析
- [x] detective 的 `handle_burst` 记忆策展
- [x] router 对 `MemoryDecision` 的多存储分发
- [x] SQLite `apply_affective_delta`
- [x] detective 事实去重
- [x] `trace_id` 贯穿日志
- [x] LLM 瞬态错误重试
- [x] `/status` HTTP 端点
- [x] 冷却状态持久化到 SQLite

保留字段 / 未来扩展（已在配置中预留）：

- gate 分类结果缓存与 per-group 速率限制。
- TopicBurst 的语义连贯性（coherence）过滤。
- detective 的定时 digest 模式（不依赖 burst 也能学习）。
- council 的二级自审（`ReplyReview`），用于自动评估回复是否“像人”。

## 16. 开发/部署命令

```bash
# 启动外部服务（可选）
docker compose -f docker-compose.dev.yml up -d

# 本地嵌入服务（如使用 Qdrant）
python scripts/local_embedding_server.py

# 运行完整验证
./scripts/test.sh

# 只跑单元/集成测试
cargo test -p neko-gate
cargo test -p neko-router

# 启动机器人
cargo run -p neko-router

# 只读观察模式
cargo run -p neko-router -- --observe
```

## 17. 关键文件索引

- 事件与类型：`crates/neko-core/src/types.rs`
- 路由中枢：`crates/neko-router/src/router.rs`
- 主程序组装：`crates/neko-router/src/main.rs`
- Gate 分类器：`crates/neko-gate/src/heuristic.rs`
- Gate actor：`crates/neko-gate/src/actor.rs`
- Sensory actor / TopicBurst：`crates/neko-sensory/src/actor.rs`
- Detective actor：`crates/neko-detective/src/actor.rs`
- Detective prompt：`crates/neko-detective/src/prompt.rs`
- Detective parser：`crates/neko-detective/src/parser.rs`
- Council actor：`crates/neko-council/src/actor.rs`
- SQLite 存储：`crates/neko-memory/src/sqlite.rs`
- 配置加载：`crates/neko-config/src/lib.rs`
- 默认配置：`config/default.toml`
