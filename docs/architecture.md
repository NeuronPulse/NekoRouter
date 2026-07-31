# NekoRouter 架构文档

## 设计目标

- **事件驱动**：所有层通过异步事件通道通信，解耦生产与消费速率。
- **低成本优先**：用廉价模型拦截绝大多数流量，仅把复杂场景升级到高级模型。
- **可配置模型接入**：通过统一 `LlmClient` trait + `OpenAiCompatibleClient` 实现，支持任意兼容端点。
- **可观测与可回归**：所有原始模型输出、Gate 决策、Council 决策、Detective 报告、回复都写入 SQLite `events` / `replies` 表，便于快照回归。

## 5 层架构

### Layer 1 感官与状态层（`neko-sensory`）

职责：

- 通过 `NapCatIngress` 持续读取 NapCat / OneBot 11 WebSocket 流量（基于 `napcat-link` SDK）。
- 解析 OneBot v11 group message 为 `ChatMessage`。
- `SensoryActor` 维护内存中的 `AffectiveState`（精力值、好感度、回复次数）。
- 批量写入 SQLite，避免高频刷屏锁库。

背压：

- `raw_tx` 容量 4096，满则丢弃旧 spam。
- 批量写入按 `batch_size` 或 `flush_interval_ms` 触发。

### Layer 2 便宜模型门控（`neko-gate`）

职责：

- `GateActor` 消费 `Event::IncomingMessage`。
- 硬过滤：超过 `max_message_length` 的消息直接 `Drop`。
- 启发式判断消息是否适合 COZY 路径，通过 `GateHeuristic` trait 可插拔。
- COZY 路径调用配置中的廉价模型，生成不超过 `max_cozy_words`  的短句。
- 不适合 COZY 的场景发送 `Event::Escalation`。

并发控制：

- 使用 `tokio::sync::Semaphore` 限制并发 LLM 调用数，防止廉价模型端被压垮。

启发式扩展：

- 内置 `DefaultHeuristic`：过滤命令、URL、过长消息。
- 内置 `EscalateAllHeuristic`（stub）：总是升级到议会层，方便测试完整 pipeline。

### Layer 3 心智议会层（`neko-council`）

职责：

- 接收 `Event::Escalation`。
- 构造三角色辩论 prompt：【猫系本能】、【社交伪装】、【冷酷理智】。
- 单次调用高级模型（Grok / Gemini 等），要求其输出 JSON 决策：

  ```json
  {
    "action": "reply|detective|ignore",
    "reasoning": "...",
    "draft_reply": "..."
  }
  ```

- 首席路由官解析输出，决定直接回复或启动侦探。

### Layer 4 侦探智能体（`neko-detective`）

职责：

- 接收 `Event::DetectiveRequest`。
- 并行查询 SQLite 历史与 Qdrant 向量记忆（MVP 使用 `StubVectorStore`）。
- 将检索结果组织成结构化 prompt，调用高级模型生成脱水 JSON 报告。
- 报告通过 `Event::DetectiveReport` 发送给深夜固化中心，高置信度时直接生成 `ReplyOut`。

### Layer 5 深夜固化中心（`neko-solidify`）

职责：

- 按 `solidify.cron` 表达式触发（MVP 时区配置已读取但未生效，使用 UTC）。
- 缓冲白天产生的 `DetectiveReport`。
- 触发时将报告 batch 转换为 LLM prompt，生成 `MERGE` Cypher 语句。
- 通过 `GraphStore::apply_updates` 更新图数据库（MVP 使用 `StubGraphStore`）。
- 根据图数据库状态刷新次日系统默认提示词，固化长期偏见（刷新逻辑预留）。

## 事件流

```text
NapCatIngress --(Event::IncomingMessage)--> SensoryActor
                                                    |
                                                    v
                                             SQLite (batch)
                                                    |
                                                    v
                                            GateActor
                                           /    |    \
                                          /     |     \
                                         v      v      v
                                 GateDecision  ReplyOut  Escalation
                                     |            |          |
                                     v            v          v
                                   SQLite      NapCatEgress  Council
                                                                   |
                                                    /---------------+---------------\
                                                   v                               v
                                              ReplyOut                       DetectiveRequest
                                                   |                               |
                                                   v                               v
                                             NapCatEgress                       Detective
                                                                                     |
                                                                      /---------------+---------------\
                                                                     v                               v
                                                                ReplyOut                        DetectiveReport
                                                                     |                               |
                                                                     v                               v
                                                               NapCatEgress                     Solidify
                                                                                                     |
                                                                                                     v
                                                                                              GraphStore (Neo4j)
```

`ReplyOut` 还会被路由回 `SensoryActor`，以便更新发送者的回复次数与好感度。

## 配置与秘密管理

配置分层加载，API key 使用 `secrecy::SecretString` 避免意外日志泄露。

环境变量示例：

```bash
export NEKO_ENV=production
export NEKO__LLM__PROVIDERS__DEEPSEEK__API_KEY="sk-..."
export NEKO__LLM__PROVIDERS__GROK__API_KEY="xai-..."
export NEKO__WEBSOCKET__URL="ws://127.0.0.1:3001"
export NEKO__WEBSOCKET__TOKEN="your_napcat_token"
```

## 扩展点

- 新增 LLM 协议：实现 `LlmClient` trait。
- 新增 QQ 协议：实现 `Ingress` / `Egress` trait。
- 新增存储后端：实现 `HistoryStore` / `VectorStore` / `GraphStore` trait。
- 新增 Gate 启发式：实现 `GateHeuristic` trait 并在 `heuristic_from_name` 注册。
- 新增层：按相同的事件通道模式接入 `neko-router` 的编排逻辑。
