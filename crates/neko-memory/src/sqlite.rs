use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use neko_core::{AffectiveState, ChatMessage, GroupId, HistoryStore, NekoError, UserId};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Pool, QueryBuilder, Sqlite, SqlitePool};
use std::path::Path;
use std::str::FromStr;
use tracing::{debug, info};

/// SQLite-backed implementation of [`HistoryStore`].
#[derive(Debug, Clone)]
pub struct SqliteStore {
    pool: Pool<Sqlite>,
}

impl SqliteStore {
    /// Connect to the SQLite database and run pending migrations.
    pub async fn connect<P: AsRef<Path>>(path: P) -> Result<Self, NekoError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| NekoError::database(format!("cannot create data dir: {e}")))?;
        }

        let url = format!("sqlite:{}", path.display());
        // `sqlite:` URLs parse to create_if_missing=false, so a fresh DB file
        // would never be created; opt in explicitly.
        let options = SqliteConnectOptions::from_str(&url)
            .map_err(|e| NekoError::database(format!("invalid sqlite url: {e}")))?
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(options)
            .await
            .map_err(|e| NekoError::database(format!("cannot connect to sqlite: {e}")))?;

        sqlx::migrate!("./migrations/sqlite")
            .run(&pool)
            .await
            .map_err(|e| NekoError::database(format!("migration failed: {e}")))?;

        info!("sqlite store ready at {}", path.display());
        Ok(Self { pool })
    }

    /// Connect to an in-memory database for testing.
    pub async fn connect_in_memory() -> Result<Self, NekoError> {
        let pool = SqlitePool::connect("sqlite::memory:")
            .await
            .map_err(|e| NekoError::database(format!("cannot connect to sqlite: {e}")))?;

        sqlx::migrate!("./migrations/sqlite")
            .run(&pool)
            .await
            .map_err(|e| NekoError::database(format!("migration failed: {e}")))?;

        Ok(Self { pool })
    }

    /// Persist a batch of messages in a single transaction.
    pub async fn insert_messages(&self, messages: &[ChatMessage]) -> Result<(), NekoError> {
        if messages.is_empty() {
            return Ok(());
        }

        let mut txn = self
            .pool
            .begin()
            .await
            .map_err(|e| NekoError::database(format!("begin transaction failed: {e}")))?;

        let mut builder = QueryBuilder::new(
            "INSERT OR IGNORE INTO messages \
             (id, group_id, sender_id, sender_nick, content, reply_to, created_at, raw_payload) ",
        );
        builder.push_values(messages, |mut b, msg| {
            b.push_bind(msg.id.to_string())
                .push_bind(&msg.group_id)
                .push_bind(&msg.sender)
                .push_bind(&msg.nickname)
                .push_bind(&msg.content)
                .push_bind(msg.reply_to.map(|id| id.to_string()))
                .push_bind(msg.timestamp.timestamp_millis())
                .push_bind(msg.raw_payload.to_string());
        });

        builder
            .build()
            .execute(&mut *txn)
            .await
            .map_err(|e| NekoError::database(format!("batch insert messages failed: {e}")))?;

        txn.commit()
            .await
            .map_err(|e| NekoError::database(format!("commit transaction failed: {e}")))?;

        debug!("inserted {} messages", messages.len());
        Ok(())
    }

    /// Record a sent reply.
    pub async fn insert_reply(
        &self,
        id: neko_core::MessageId,
        message_id: neko_core::MessageId,
        layer: &str,
        content: &str,
        sent_at: DateTime<Utc>,
    ) -> Result<(), NekoError> {
        sqlx::query(
            "INSERT INTO replies (id, message_id, layer, content, sent_at) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(id.to_string())
        .bind(message_id.to_string())
        .bind(layer)
        .bind(content)
        .bind(sent_at.timestamp_millis())
        .execute(&self.pool)
        .await
        .map_err(|e| NekoError::database(format!("insert reply failed: {e}")))?;
        Ok(())
    }

    /// Record an internal event for observability / regression testing.
    pub async fn insert_event(
        &self,
        id: neko_core::MessageId,
        kind: &str,
        payload: &serde_json::Value,
        created_at: DateTime<Utc>,
    ) -> Result<(), NekoError> {
        sqlx::query("INSERT INTO events (id, kind, payload, created_at) VALUES (?, ?, ?, ?)")
            .bind(id.to_string())
            .bind(kind)
            .bind(payload.to_string())
            .bind(created_at.timestamp_millis())
            .execute(&self.pool)
            .await
            .map_err(|e| NekoError::database(format!("insert event failed: {e}")))?;
        Ok(())
    }

    /// Load all persisted affective snapshots.
    pub async fn load_affective_states(
        &self,
    ) -> Result<Vec<(GroupId, UserId, AffectiveState)>, NekoError> {
        let rows: Vec<AffectiveRow> = sqlx::query_as(
            "SELECT group_id, user_id, energy, favorability, reply_count, updated_at \
             FROM affective_snapshots",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NekoError::database(format!("load affective states failed: {e}")))?;

        rows.into_iter()
            .map(|row| {
                let last_updated = Utc
                    .timestamp_millis_opt(row.updated_at)
                    .single()
                    .ok_or_else(|| NekoError::database("invalid snapshot timestamp"))?;
                Ok((
                    row.group_id,
                    row.user_id,
                    AffectiveState {
                        energy: row.energy,
                        favorability: row.favorability,
                        reply_count: row.reply_count as u64,
                        last_updated: Some(last_updated),
                    },
                ))
            })
            .collect()
    }

    /// Upsert affective snapshots in a single transaction.
    pub async fn save_affective_states(
        &self,
        states: &[(GroupId, UserId, AffectiveState)],
    ) -> Result<(), NekoError> {
        if states.is_empty() {
            return Ok(());
        }

        let mut txn = self
            .pool
            .begin()
            .await
            .map_err(|e| NekoError::database(format!("begin transaction failed: {e}")))?;

        for (group_id, user_id, state) in states {
            let updated_at = state.last_updated.unwrap_or_else(Utc::now);
            sqlx::query(
                "INSERT INTO affective_snapshots \
                 (group_id, user_id, energy, favorability, reply_count, updated_at) \
                 VALUES (?, ?, ?, ?, ?, ?) \
                 ON CONFLICT(group_id, user_id) DO UPDATE SET \
                   energy = excluded.energy, \
                   favorability = excluded.favorability, \
                   reply_count = excluded.reply_count, \
                   updated_at = excluded.updated_at",
            )
            .bind(group_id)
            .bind(user_id)
            .bind(state.energy)
            .bind(state.favorability)
            .bind(state.reply_count as i64)
            .bind(updated_at.timestamp_millis())
            .execute(&mut *txn)
            .await
            .map_err(|e| NekoError::database(format!("upsert affective state failed: {e}")))?;
        }

        txn.commit()
            .await
            .map_err(|e| NekoError::database(format!("commit transaction failed: {e}")))?;
        Ok(())
    }

    /// Number of persisted replies (used for tests and observability).
    pub async fn count_replies(&self) -> Result<i64, NekoError> {
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM replies")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| NekoError::database(format!("count replies failed: {e}")))?;
        Ok(count)
    }

    /// Number of persisted events (used for tests and observability).
    pub async fn count_events(&self) -> Result<i64, NekoError> {
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM events")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| NekoError::database(format!("count events failed: {e}")))?;
        Ok(count)
    }

    /// Number of persisted messages (used for observability).
    pub async fn count_messages(&self) -> Result<i64, NekoError> {
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM messages")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| NekoError::database(format!("count messages failed: {e}")))?;
        Ok(count)
    }

    /// Mark messages as flushed to the store (sets `processed_at`).
    pub async fn mark_processed(&self, ids: &[neko_core::MessageId]) -> Result<(), NekoError> {
        if ids.is_empty() {
            return Ok(());
        }
        let now = Utc::now().timestamp_millis();
        for id in ids {
            sqlx::query("UPDATE messages SET processed_at = ? WHERE id = ?")
                .bind(now)
                .bind(id.to_string())
                .execute(&self.pool)
                .await
                .map_err(|e| NekoError::database(format!("mark processed failed: {e}")))?;
        }
        Ok(())
    }

    /// Number of persisted affective snapshots (used for observability).
    pub async fn count_snapshots(&self) -> Result<i64, NekoError> {
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM affective_snapshots")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| NekoError::database(format!("count snapshots failed: {e}")))?;
        Ok(count)
    }

    /// Most recent messages, newest first.
    pub async fn recent_messages(&self, limit: usize) -> Result<Vec<StoredMessage>, NekoError> {
        let rows: Vec<StoredMessageRow> = sqlx::query_as(
            "SELECT id, group_id, sender_id, sender_nick, content, reply_to, created_at \
             FROM messages ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NekoError::database(format!("query recent messages failed: {e}")))?;

        rows.into_iter()
            .map(|r| {
                Ok(StoredMessage {
                    id: r.id,
                    group_id: r.group_id,
                    sender_id: r.sender_id,
                    sender_nick: r.sender_nick,
                    content: r.content,
                    reply_to: r.reply_to,
                    created_at: timestamp_to_utc(r.created_at)?,
                })
            })
            .collect()
    }

    /// Most recent replies, newest first.
    pub async fn recent_replies(&self, limit: usize) -> Result<Vec<StoredReply>, NekoError> {
        let rows: Vec<StoredReplyRow> = sqlx::query_as(
            "SELECT id, message_id, layer, content, sent_at \
             FROM replies ORDER BY sent_at DESC LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NekoError::database(format!("query recent replies failed: {e}")))?;

        rows.into_iter()
            .map(|r| {
                Ok(StoredReply {
                    id: r.id,
                    message_id: r.message_id,
                    layer: r.layer,
                    content: r.content,
                    sent_at: timestamp_to_utc(r.sent_at)?,
                })
            })
            .collect()
    }

    /// Most recent internal events, newest first.
    pub async fn recent_events(&self, limit: usize) -> Result<Vec<StoredEvent>, NekoError> {
        let rows: Vec<StoredEventRow> = sqlx::query_as(
            "SELECT id, kind, payload, created_at FROM events ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NekoError::database(format!("query recent events failed: {e}")))?;

        rows.into_iter()
            .map(|r| {
                let payload = serde_json::from_str(&r.payload)
                    .unwrap_or_else(|_| serde_json::Value::String(r.payload.clone()));
                Ok(StoredEvent {
                    id: r.id,
                    kind: r.kind,
                    payload,
                    created_at: timestamp_to_utc(r.created_at)?,
                })
            })
            .collect()
    }
}

/// A single message as stored in the `messages` table (observability).
#[derive(Debug, Clone)]
pub struct StoredMessage {
    pub id: String,
    pub group_id: GroupId,
    pub sender_id: UserId,
    pub sender_nick: Option<String>,
    pub content: String,
    pub reply_to: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A single reply as stored in the `replies` table (observability).
#[derive(Debug, Clone)]
pub struct StoredReply {
    pub id: String,
    pub message_id: String,
    pub layer: String,
    pub content: String,
    pub sent_at: DateTime<Utc>,
}

/// A single internal event as stored in the `events` table (observability).
#[derive(Debug, Clone)]
pub struct StoredEvent {
    pub id: String,
    pub kind: String,
    pub payload: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

fn timestamp_to_utc(millis: i64) -> Result<DateTime<Utc>, NekoError> {
    Utc.timestamp_millis_opt(millis)
        .single()
        .ok_or_else(|| NekoError::database("invalid stored timestamp"))
}

#[async_trait]
impl HistoryStore for SqliteStore {
    async fn append_batch(&self, messages: &[ChatMessage]) -> Result<(), NekoError> {
        self.insert_messages(messages).await
    }

    async fn query_context(
        &self,
        group_id: &GroupId,
        user_id: Option<&UserId>,
        before: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<ChatMessage>, NekoError> {
        let before_ms = before.timestamp_millis();
        let rows: Vec<MessageRow> = match user_id {
            Some(uid) => {
                sqlx::query_as(
                    "SELECT * FROM messages \
                     WHERE group_id = ? AND sender_id = ? AND created_at < ? \
                     ORDER BY created_at DESC LIMIT ?",
                )
                .bind(group_id)
                .bind(uid)
                .bind(before_ms)
                .bind(limit as i64)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query_as(
                    "SELECT * FROM messages \
                     WHERE group_id = ? AND created_at < ? \
                     ORDER BY created_at DESC LIMIT ?",
                )
                .bind(group_id)
                .bind(before_ms)
                .bind(limit as i64)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| NekoError::database(format!("query context failed: {e}")))?;

        rows.into_iter()
            .map(|row| row.try_into())
            .rev()
            .collect::<Result<Vec<_>, _>>()
    }

    async fn load_affective_states(
        &self,
    ) -> Result<Vec<(GroupId, UserId, AffectiveState)>, NekoError> {
        self.load_affective_states().await
    }

    async fn save_affective_states(
        &self,
        states: &[(GroupId, UserId, AffectiveState)],
    ) -> Result<(), NekoError> {
        self.save_affective_states(states).await
    }

    async fn mark_processed(&self, ids: &[neko_core::MessageId]) -> Result<(), NekoError> {
        self.mark_processed(ids).await
    }
}

#[derive(sqlx::FromRow)]
struct AffectiveRow {
    group_id: String,
    user_id: String,
    energy: f32,
    favorability: f32,
    reply_count: i64,
    updated_at: i64,
}

#[derive(sqlx::FromRow)]
struct MessageRow {
    id: String,
    group_id: String,
    sender_id: String,
    sender_nick: Option<String>,
    content: String,
    reply_to: Option<String>,
    created_at: i64,
    raw_payload: String,
    #[allow(dead_code)]
    #[sqlx(rename = "processed_at")]
    processed_at: Option<i64>,
}

impl TryFrom<MessageRow> for ChatMessage {
    type Error = NekoError;

    fn try_from(row: MessageRow) -> Result<Self, Self::Error> {
        let id = row
            .id
            .parse()
            .map_err(|e| NekoError::database(format!("invalid message id: {e}")))?;
        let reply_to = match row.reply_to {
            Some(s) => Some(
                s.parse()
                    .map_err(|e| NekoError::database(format!("invalid reply_to id: {e}")))?,
            ),
            None => None,
        };
        let timestamp = Utc
            .timestamp_millis_opt(row.created_at)
            .single()
            .ok_or_else(|| NekoError::database("invalid timestamp"))?;
        let raw_payload = serde_json::from_str(&row.raw_payload)
            .map_err(|e| NekoError::database(format!("invalid raw_payload json: {e}")))?;

        Ok(ChatMessage {
            id,
            group_id: row.group_id,
            sender: row.sender_id,
            nickname: row.sender_nick.unwrap_or_default(),
            content: row.content,
            timestamp,
            reply_to,
            raw_payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[tokio::test]
    async fn affective_states_round_trip() {
        let store = SqliteStore::connect_in_memory().await.unwrap();
        let now = Utc
            .timestamp_millis_opt(1_700_000_000_000)
            .single()
            .unwrap();
        let states = vec![
            (
                "g1".to_string(),
                "u1".to_string(),
                AffectiveState {
                    energy: 0.7,
                    favorability: 0.3,
                    reply_count: 5,
                    last_updated: Some(now),
                },
            ),
            (
                "g2".to_string(),
                "u2".to_string(),
                AffectiveState {
                    energy: 0.2,
                    favorability: -0.1,
                    reply_count: 1,
                    last_updated: Some(now),
                },
            ),
        ];

        store.save_affective_states(&states).await.unwrap();
        let loaded = store.load_affective_states().await.unwrap();
        assert_eq!(loaded.len(), 2);

        let (g, u, s) = loaded
            .into_iter()
            .find(|(g, u, _)| g == "g1" && u == "u1")
            .unwrap();
        assert_eq!(g, "g1");
        assert_eq!(u, "u1");
        assert_eq!(s.energy, 0.7);
        assert_eq!(s.favorability, 0.3);
        assert_eq!(s.reply_count, 5);
        assert_eq!(s.last_updated, Some(now));

        // Re-saving upserts instead of duplicating.
        store
            .save_affective_states(&[(
                "g1".to_string(),
                "u1".to_string(),
                AffectiveState {
                    energy: 0.9,
                    ..states[0].2
                },
            )])
            .await
            .unwrap();
        let loaded = store.load_affective_states().await.unwrap();
        assert_eq!(loaded.len(), 2);
        let (_, _, s) = loaded
            .into_iter()
            .find(|(g, u, _)| g == "g1" && u == "u1")
            .unwrap();
        assert_eq!(s.energy, 0.9);
    }
}

#[derive(sqlx::FromRow)]
struct StoredMessageRow {
    id: String,
    group_id: String,
    sender_id: String,
    sender_nick: Option<String>,
    content: String,
    reply_to: Option<String>,
    created_at: i64,
}

#[derive(sqlx::FromRow)]
struct StoredReplyRow {
    id: String,
    message_id: String,
    layer: String,
    content: String,
    sent_at: i64,
}

#[derive(sqlx::FromRow)]
struct StoredEventRow {
    id: String,
    kind: String,
    payload: String,
    created_at: i64,
}
