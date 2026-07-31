use neko_core::DetectiveReport;

/// Build the late-night solidification prompt.
///
/// The LLM is asked to distill a batch of detective reports into a set of
/// idempotent Cypher `MERGE` statements that update the long-term relationship
/// graph in Neo4j.
pub fn solidify_prompt(reports: &[DetectiveReport]) -> String {
    let reports_text = if reports.is_empty() {
        "（无报告）".to_string()
    } else {
        reports
            .iter()
            .map(|r| {
                let changes = r
                    .relationship_changes
                    .iter()
                    .map(|c| {
                        format!(
                            "- {} -> {} ({:?}, delta {:.2})",
                            c.from, c.to, c.kind, c.delta
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "目标用户：{}\n总结：{}\n关系变化：\n{}",
                    r.target_user, r.summary, changes
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    format!(
        "你是深夜固化中心。请将以下侦探报告 batch 转换为 Neo4j Cypher 更新语句。\n\
         \n\
         规则：\n\
         1. 每个 GraphUpdate 必须包含一条可执行的 Cypher 语句和可选参数。\n\
         2. 使用 MERGE 保证幂等性，避免重复创建节点。\n\
         3. 节点标签统一为 User，关系类型使用大写（如 TEASE、SUPPORT、CONFLICT）。\n\
         4. 只输出 JSON，不要解释。\n\
         \n\
         侦探报告：\n{}\n\
         \n\
         输出 JSON 格式：\n\
         {{\n\
           \"updates\": [\n\
             {{\n\
               \"cypher\": \"MERGE (a:User {{id: $from}}) MERGE (b:User {{id: $to}}) MERGE (a)-[r:TEASE]->(b) SET r.delta = COALESCE(r.delta, 0) + $delta\",\n\
               \"params\": {{\"from\": \"user_a\", \"to\": \"user_b\", \"delta\": -0.1}}\n\
             }}\n\
           ]\n\
         }}",
        reports_text
    )
}
