use async_trait::async_trait;
use neko_core::{GraphStore, GraphUpdate, NekoError};
use neo4rs::{query, Graph};
use std::sync::Arc;
use tracing::{debug, info};

/// A real Neo4j-backed graph store.
pub struct Neo4jGraphStore {
    graph: Arc<Graph>,
}

impl Neo4jGraphStore {
    pub async fn new(
        uri: impl AsRef<str>,
        user: impl AsRef<str>,
        password: impl AsRef<str>,
    ) -> Result<Self, NekoError> {
        let graph = Graph::new(uri.as_ref(), user.as_ref(), password.as_ref())
            .await
            .map_err(|e| NekoError::transport(format!("cannot connect to Neo4j: {e:?}")))?;

        Ok(Self {
            graph: Arc::new(graph),
        })
    }
}

#[async_trait]
impl GraphStore for Neo4jGraphStore {
    async fn apply_updates(&self, updates: &[GraphUpdate]) -> Result<(), NekoError> {
        if updates.is_empty() {
            return Ok(());
        }

        for update in updates {
            info!("applying graph update");
            debug!("cypher: {}", update.cypher);

            let mut q = query(&update.cypher);
            for (key, value) in &update.params {
                let bolt_value = neo4rs::BoltType::try_from(value.clone())
                    .map_err(|e| NekoError::parse(format!("cannot convert param {key}: {e}")))?;
                q = q.param(key, bolt_value);
            }

            self.graph
                .run(q)
                .await
                .map_err(|e| NekoError::transport(format!("Neo4j run failed: {e:?}")))?;
        }

        Ok(())
    }
}
