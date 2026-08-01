use neko_core::{GraphStore, GraphUpdate};
use neko_solidify::Neo4jGraphStore;
use neo4rs::{query, Graph};
use std::collections::HashMap;
use std::net::TcpStream;
use std::time::Duration;

const NEO4J_URI: &str = "neo4j://127.0.0.1:7687";
const NEO4J_USER: &str = "neo4j";
const NEO4J_PASSWORD: &str = "password";

fn neo4j_reachable() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:7687".parse().unwrap(),
        Duration::from_millis(500),
    )
    .is_ok()
}

fn update(from: &str, to: &str, delta: f64) -> GraphUpdate {
    let mut params = HashMap::new();
    params.insert("from".to_string(), serde_json::json!(from));
    params.insert("to".to_string(), serde_json::json!(to));
    params.insert("delta".to_string(), serde_json::json!(delta));
    GraphUpdate {
        cypher: "MERGE (a:User {id: $from}) MERGE (b:User {id: $to}) \
                 MERGE (a)-[r:TEASE]->(b) SET r.delta = COALESCE(r.delta, 0) + $delta"
            .to_string(),
        params,
    }
}

#[tokio::test]
async fn neo4j_applies_graph_updates() {
    if !neo4j_reachable() {
        eprintln!("skipping: neo4j not reachable at {NEO4J_URI}");
        return;
    }

    let from = format!("u_{}", uuid::Uuid::new_v4().simple());
    let to = format!("u_{}", uuid::Uuid::new_v4().simple());

    let store = Neo4jGraphStore::new(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        .await
        .unwrap();

    // Two applications must accumulate the relationship delta.
    store
        .apply_updates(&[update(&from, &to, 2.0)])
        .await
        .unwrap();
    store
        .apply_updates(&[update(&from, &to, 3.0)])
        .await
        .unwrap();

    // Read back through a separate connection.
    let graph = Graph::new(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        .await
        .unwrap();
    let mut stream = graph
        .execute(
            query("MATCH (a:User {id: $from})-[r:TEASE]->(b:User {id: $to}) RETURN r.delta")
                .param("from", from.clone())
                .param("to", to.clone()),
        )
        .await
        .unwrap();
    let row = stream
        .next()
        .await
        .unwrap()
        .expect("expected a relationship");
    let delta: f64 = row.get("r.delta").unwrap();
    assert_eq!(delta, 5.0);

    // Clean up the throwaway nodes.
    graph
        .run(
            query("MATCH (a:User {id: $from})-[r:TEASE]->(b:User {id: $to}) DETACH DELETE a, r, b")
                .param("from", from)
                .param("to", to),
        )
        .await
        .map_err(|e| eprintln!("cleanup failed: {e:?}"))
        .ok();
}
