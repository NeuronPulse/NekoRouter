# NekoRouter

事件驱动的微型多智能体 QQ 群聊机器人。

## 快速开始

```bash
cp config/local.toml.example config/local.toml
# 编辑 config/local.toml，填入模型 API key 与 NapCat token
cargo test --all
cargo run -p neko-router
```

## 架构

- Layer 1 `neko-sensory`：基于 `napcat-link` SDK 的 NapCat / OneBot 11 入站与出站、SQLite 批量写入、情感状态。
- Layer 2 `neko-gate`：廉价模型门控，可插拔启发式。
- Layer 3 `neko-council`：心智议会，三角色 JSON 决策。
- Layer 4 `neko-detective`：侦探智能体，检索上下文并生成结构化报告。
- Layer 5 `neko-solidify`：深夜固化中心，cron 触发 Cypher 图更新。

## 协议

MIT License © 2026 NekoRouter Contributors
