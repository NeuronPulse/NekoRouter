use neko_config::NekoConfig;
use neko_memory::SqliteStore;

/// Read-only observability mode: `neko-router --observe [--limit N]`.
///
/// Prints database counts plus the most recent messages, replies and internal
/// events. Loads config without secret validation so it works even when API
/// keys are not configured.
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let limit = std::env::args()
        .position(|a| a == "--limit")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    let config = NekoConfig::load_observe()?;
    let sqlite = SqliteStore::connect(&config.sqlite.path).await?;

    println!("== NekoRouter observability ==");
    println!("database:  {}", config.sqlite.path);
    println!("messages:  {}", sqlite.count_messages().await?);
    println!("replies:   {}", sqlite.count_replies().await?);
    println!("events:    {}", sqlite.count_events().await?);
    println!("snapshots: {}", sqlite.count_snapshots().await?);
    println!();

    if limit > 0 {
        println!("-- last {limit} messages --");
        for m in sqlite.recent_messages(limit).await? {
            let nick = m.sender_nick.as_deref().unwrap_or(m.sender_id.as_str());
            let reply = m
                .reply_to
                .as_deref()
                .map(|r| format!(" (reply to {})", &r[..r.len().min(8)]))
                .unwrap_or_default();
            println!(
                "{} [{}] {}: {}{}",
                m.created_at.format("%m-%d %H:%M:%S"),
                m.group_id,
                nick,
                truncate(&m.content, 80),
                reply,
            );
        }
        println!();

        println!("-- last {limit} replies --");
        for r in sqlite.recent_replies(limit).await? {
            println!(
                "{} [{}] <{}> {}",
                r.sent_at.format("%m-%d %H:%M:%S"),
                r.layer,
                &r.message_id[..r.message_id.len().min(8)],
                truncate(&r.content, 80),
            );
        }
        println!();

        println!("-- last {limit} events --");
        for e in sqlite.recent_events(limit).await? {
            println!(
                "{} [{}] {}",
                e.created_at.format("%m-%d %H:%M:%S"),
                e.kind,
                truncate(&e.payload.to_string(), 120),
            );
        }
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
