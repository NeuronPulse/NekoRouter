CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    sender_nick TEXT,
    content TEXT NOT NULL,
    reply_to TEXT,
    created_at INTEGER NOT NULL,
    raw_payload TEXT,
    processed_at INTEGER
);

CREATE TABLE IF NOT EXISTS replies (
    id TEXT PRIMARY KEY,
    -- The replied-to message id. No FK: messages are batch-written asynchronously,
    -- so a reply can be persisted before its source message is flushed.
    message_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    content TEXT NOT NULL,
    sent_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS affective_snapshots (
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    energy REAL NOT NULL,
    favorability REAL NOT NULL,
    reply_count INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, user_id)
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_group_time ON messages(group_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_kind_time ON events(kind, created_at);
