use async_trait::async_trait;
use neko_core::{EmbeddingClient, NekoError};
use reqwest::header::{self, HeaderMap, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, instrument};

/// An OpenAI-compatible embedding client.
/// Works with any provider exposing `/embeddings` (OpenAI, text-embedding-3,
/// some local proxies, etc.).
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddingClient {
    name: String,
    http: reqwest::Client,
    base_url: reqwest::Url,
    model: String,
    #[allow(dead_code)]
    api_key: SecretString,
    vector_dim: usize,
}

impl OpenAiEmbeddingClient {
    pub fn new(
        name: impl Into<String>,
        base_url: impl AsRef<str>,
        model: impl Into<String>,
        api_key: SecretString,
        vector_dim: usize,
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
            vector_dim,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl EmbeddingClient for OpenAiEmbeddingClient {
    #[instrument(skip(self, texts), fields(provider = %self.name, model = %self.model, batch = texts.len()))]
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NekoError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| NekoError::transport(format!("invalid endpoint url: {e}")))?;

        let body = EmbeddingsRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        debug!("sending embedding request to {}", url);
        let resp = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| NekoError::transport(format!("embedding request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            return Err(NekoError::llm(format!(
                "embeddings returned {status}: {text}"
            )));
        }

        let parsed: EmbeddingsResponse = resp
            .json()
            .await
            .map_err(|e| NekoError::parse(format!("cannot decode embeddings response: {e}")))?;

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for item in parsed.data {
            embeddings.push(item.embedding);
        }

        if embeddings.len() != texts.len() {
            return Err(NekoError::llm(format!(
                "embeddings returned {} vectors for {} texts",
                embeddings.len(),
                texts.len()
            )));
        }

        Ok(embeddings)
    }

    fn vector_dim(&self) -> usize {
        self.vector_dim
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}
