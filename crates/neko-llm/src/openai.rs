use async_trait::async_trait;
use neko_core::{
    FinishReason, LlmClient, LlmMessage, LlmRequest, LlmResponse, LlmRole, NekoError,
    ResponseFormat, TokenUsage,
};
use reqwest::header::{self, HeaderMap, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, instrument};

/// An OpenAI-compatible LLM client.
/// Works with DeepSeek, Grok, Gemini OpenAI-compatible endpoints, etc.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient {
    name: String,
    http: reqwest::Client,
    base_url: reqwest::Url,
    model: String,
    #[allow(dead_code)]
    api_key: SecretString,
    #[allow(dead_code)]
    default_temperature: f32,
    default_max_tokens: Option<u32>,
    default_response_format: ResponseFormat,
}

impl OpenAiCompatibleClient {
    pub fn new(
        name: impl Into<String>,
        base_url: impl AsRef<str>,
        model: impl Into<String>,
        api_key: SecretString,
        default_temperature: f32,
        default_max_tokens: Option<u32>,
        default_response_format: ResponseFormat,
    ) -> Result<Self, NekoError> {
        let base_url = base_url.as_ref().trim_end_matches('/');
        let base_url = reqwest::Url::parse(base_url)
            .map_err(|e| NekoError::config(format!("invalid base_url: {e}")))?;

        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(|e| NekoError::config(format!("invalid api key header: {e}")))?,
        );
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| NekoError::config(format!("cannot build http client: {e}")))?;

        Ok(Self {
            name: name.into(),
            http,
            base_url,
            model: model.into(),
            api_key,
            default_temperature,
            default_max_tokens,
            default_response_format,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl LlmClient for OpenAiCompatibleClient {
    #[instrument(skip(self, req), fields(provider = %self.name, model = %self.model))]
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, NekoError> {
        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| NekoError::transport(format!("invalid endpoint url: {e}")))?;

        let body = ChatCompletionRequest {
            model: self.model.clone(),
            messages: req.messages.into_iter().map(Into::into).collect(),
            temperature: Some(req.temperature),
            max_tokens: req.max_tokens.or(self.default_max_tokens),
            response_format: match req.response_format.unwrap_or(self.default_response_format) {
                ResponseFormat::Text => None,
                ResponseFormat::JsonObject => Some(ResponseFormatBody {
                    r#type: "json_object".to_string(),
                }),
            },
        };

        debug!("sending request to {}", url);
        let resp = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| NekoError::transport(format!("llm request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            error!("llm returned {}: {}", status, text);
            return Err(NekoError::llm(format!("llm returned {status}: {text}")));
        }

        let parsed: ChatCompletionResponse = resp
            .json()
            .await
            .map_err(|e| NekoError::parse(format!("cannot decode llm response: {e}")))?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| NekoError::llm("llm returned no choices"))?;

        let finish_reason = choice.finish_reason.unwrap_or_else(|| "stop".to_string());
        let finish_reason = match finish_reason.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "content_filter" => FinishReason::ContentFilter,
            other => FinishReason::Other(other.to_string()),
        };

        Ok(LlmResponse {
            content: choice.message.content,
            usage: parsed.usage.map(Into::into).unwrap_or_default(),
            finish_reason,
        })
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormatBody>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

impl From<LlmMessage> for OpenAiMessage {
    fn from(msg: LlmMessage) -> Self {
        Self {
            role: match msg.role {
                LlmRole::System => "system",
                LlmRole::User => "user",
                LlmRole::Assistant => "assistant",
            }
            .to_string(),
            content: msg.content,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ResponseFormatBody {
    r#type: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<OpenAiUsage> for TokenUsage {
    fn from(u: OpenAiUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

/// Extract JSON from a model response that may be wrapped in markdown fences.
pub fn extract_json(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if let Some(inner) = trimmed.strip_prefix("```json") {
        let inner = inner.strip_suffix("```").unwrap_or(inner);
        return Some(inner.trim().to_string());
    }
    if let Some(inner) = trimmed.strip_prefix("```") {
        let inner = inner.strip_suffix("```").unwrap_or(inner);
        return Some(inner.trim().to_string());
    }
    Some(trimmed.to_string())
}

/// Try to parse a JSON string; return the raw text if parsing fails.
pub fn parse_or_fallback<T: serde::de::DeserializeOwned>(
    raw: &str,
) -> Result<T, (serde_json::Error, String)> {
    let cleaned = extract_json(raw).unwrap_or_else(|| raw.to_string());
    serde_json::from_str(&cleaned).map_err(|e| (e, cleaned))
}
