use neko_core::{
    FinishReason, LlmClient, LlmMessage, LlmRequest, LlmRole, NekoError, ResponseFormat,
};
use neko_llm::OpenAiCompatibleClient;
use secrecy::SecretString;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const OK_BODY: &str = r#"{
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "你好呀"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}"#;

fn request() -> LlmRequest {
    LlmRequest {
        messages: vec![LlmMessage {
            role: LlmRole::User,
            content: "hi".to_string(),
        }],
        temperature: 0.9,
        max_tokens: Some(16),
        response_format: Some(ResponseFormat::Text),
    }
}

fn client(base_url: &str, max_retries: u32) -> OpenAiCompatibleClient {
    OpenAiCompatibleClient::new(
        "test",
        format!("{base_url}/v1"),
        "test-model",
        SecretString::from("sk-test".to_string()),
        0.9,
        Some(16),
        ResponseFormat::Text,
    )
    .unwrap()
    .with_retry(max_retries)
}

#[tokio::test]
async fn retries_after_http_500_and_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(OK_BODY))
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server.uri(), 1);
    let resp = client.complete(request()).await.expect("should succeed");
    assert_eq!(resp.content, "你好呀");
    assert_eq!(resp.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
        .expect(2) // initial + 1 retry
        .mount(&server)
        .await;

    let client = client(&server.uri(), 1);
    let err = client.complete(request()).await.expect_err("should fail");
    assert!(matches!(err, NekoError::Transport(_)));
}

#[tokio::test]
async fn does_not_retry_client_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server.uri(), 3);
    let err = client.complete(request()).await.expect_err("should fail");
    assert!(matches!(err, NekoError::Llm(_)));
}

#[tokio::test]
async fn retries_on_429_throttling() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(OK_BODY))
        .mount(&server)
        .await;

    let client = client(&server.uri(), 2);
    let resp = client
        .complete(request())
        .await
        .expect("should succeed after 429");
    assert_eq!(resp.content, "你好呀");
}
