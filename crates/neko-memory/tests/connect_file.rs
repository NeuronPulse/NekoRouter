#[tokio::test]
async fn connect_to_file_path() {
    let path = format!("/tmp/neko_connect_test_{}.db", std::process::id());
    let store = neko_memory::SqliteStore::connect(&path).await.unwrap();
    assert_eq!(store.count_messages().await.unwrap(), 0);
    std::fs::remove_file(&path).ok();
}
