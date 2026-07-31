use neko_core::NekoError;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct NekoConfig {
    pub bot: BotConfig,
    pub websocket: WebSocketConfig,
    pub sqlite: SqliteConfig,
    pub qdrant: QdrantConfig,
    pub embedding: EmbeddingConfig,
    pub neo4j: Neo4jConfig,
    pub llm: LlmConfig,
    pub personality: PersonalityConfig,
    pub solidify: SolidifyConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BotConfig {
    pub name: String,
    pub qq_id: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WebSocketConfig {
    pub url: String,
    pub reconnect_interval_sec: u64,
    /// Optional access token for NapCat / OneBot 11 WebSocket authentication.
    #[serde(default)]
    pub token: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SqliteConfig {
    pub path: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub collection: String,
    pub vector_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: SecretString,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub user: String,
    pub password: SecretString,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub gate: String,
    pub council: String,
    /// Optional provider for the detective layer; falls back to `council`.
    #[serde(default)]
    pub detective: Option<String>,
    /// Optional provider for the solidify layer; falls back to `council`.
    #[serde(default)]
    pub solidify: Option<String>,
    pub providers: HashMap<String, LlmProviderConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmProviderConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: SecretString,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub response_format: ResponseFormatConfig,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResponseFormatConfig {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PersonalityConfig {
    pub max_cozy_words: usize,
    pub max_message_length: usize,
    pub energy_decay_per_min: f32,
    pub favor_decay_per_min: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SolidifyConfig {
    pub cron: String,
    pub timezone: String,
}

impl NekoConfig {
    /// Load configuration from the layered sources.
    ///
    /// Order (later overrides earlier):
    /// 1. `config/default.toml`
    /// 2. `config/{NEKO_ENV}.toml` (optional, defaults to `local`)
    /// 3. `.env` file (optional)
    /// 4. `{NEKO_SECRETS_FILE}` (optional)
    /// 5. Environment variables prefixed with `NEKO__` using `__` as separator.
    pub fn load() -> Result<Self, NekoError> {
        let _ = dotenvy::dotenv();

        let root = project_root()?;
        let env_name = env::var("NEKO_ENV").unwrap_or_else(|_| "local".to_string());

        let mut builder = config::Config::builder()
            .add_source(config::File::from(root.join("config/default.toml")))
            .add_source(
                config::File::from(root.join(format!("config/{env_name}.toml"))).required(false),
            );

        if let Ok(secrets_path) = env::var("NEKO_SECRETS_FILE") {
            builder =
                builder.add_source(config::File::from(PathBuf::from(secrets_path)).required(false));
        }

        builder = builder.add_source(
            config::Environment::with_prefix("NEKO")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder
            .build()
            .map_err(|e| NekoError::config(format!("failed to build config: {e}")))?;

        let mut config: NekoConfig = cfg
            .try_deserialize()
            .map_err(|e| NekoError::config(format!("failed to deserialize config: {e}")))?;

        config.expand_env_placeholders();
        config.validate()?;
        Ok(config)
    }

    /// Expand `${VAR}` placeholders in provider API keys, embedding key and Neo4j password.
    fn expand_env_placeholders(&mut self) {
        self.embedding.api_key = expand_secret(&self.embedding.api_key);
        self.neo4j.password = expand_secret(&self.neo4j.password);
        for provider in self.llm.providers.values_mut() {
            provider.api_key = expand_secret(&provider.api_key);
        }
    }

    fn validate(&self) -> Result<(), NekoError> {
        if self.websocket.url.is_empty() {
            return Err(NekoError::config("websocket.url is required"));
        }
        if self.sqlite.path.is_empty() {
            return Err(NekoError::config("sqlite.path is required"));
        }

        self.gate_provider()?;
        self.council_provider()?;
        self.detective_provider()?;
        self.solidify_provider()?;

        if !self.qdrant.url.is_empty() {
            if self.embedding.base_url.is_empty() {
                return Err(NekoError::config(
                    "embedding.base_url is required when qdrant is enabled",
                ));
            }
            if self.embedding.model.is_empty() {
                return Err(NekoError::config(
                    "embedding.model is required when qdrant is enabled",
                ));
            }
            if self.embedding.api_key.expose_secret().is_empty() {
                return Err(NekoError::config(
                    "embedding.api_key is required when qdrant is enabled",
                ));
            }
        }

        for (name, provider) in &self.llm.providers {
            if provider.api_key.expose_secret().is_empty() {
                return Err(NekoError::config(format!(
                    "llm.providers.{name}.api_key is required"
                )));
            }
            if provider.base_url.is_empty() {
                return Err(NekoError::config(format!(
                    "llm.providers.{name}.base_url is required"
                )));
            }
            if provider.model.is_empty() {
                return Err(NekoError::config(format!(
                    "llm.providers.{name}.model is required"
                )));
            }
        }

        Ok(())
    }

    pub fn gate_provider(&self) -> Result<&LlmProviderConfig, NekoError> {
        self.llm.providers.get(&self.llm.gate).ok_or_else(|| {
            NekoError::config(format!("gate provider '{}' not found", self.llm.gate))
        })
    }

    pub fn council_provider(&self) -> Result<&LlmProviderConfig, NekoError> {
        self.llm.providers.get(&self.llm.council).ok_or_else(|| {
            NekoError::config(format!("council provider '{}' not found", self.llm.council))
        })
    }

    /// Provider for the detective layer. Falls back to the council provider
    /// when no dedicated `detective` provider is configured.
    pub fn detective_provider(&self) -> Result<&LlmProviderConfig, NekoError> {
        self.provider_or_council("detective", &self.llm.detective)
    }

    /// Provider for the solidify layer. Falls back to the council provider
    /// when no dedicated `solidify` provider is configured.
    pub fn solidify_provider(&self) -> Result<&LlmProviderConfig, NekoError> {
        self.provider_or_council("solidify", &self.llm.solidify)
    }

    fn provider_or_council(
        &self,
        layer: &str,
        name: &Option<String>,
    ) -> Result<&LlmProviderConfig, NekoError> {
        match name {
            Some(name) => {
                self.llm.providers.get(name).ok_or_else(|| {
                    NekoError::config(format!("{layer} provider '{name}' not found"))
                })
            }
            None => self.council_provider(),
        }
    }
}

fn expand_secret(secret: &SecretString) -> SecretString {
    let value = secret.expose_secret();
    if let Some(inner) = value.strip_prefix("${").and_then(|s| s.strip_suffix("}")) {
        if let Ok(env_value) = env::var(inner) {
            return SecretString::from(env_value);
        }
    }
    SecretString::from(value.to_string())
}

fn project_root() -> Result<PathBuf, NekoError> {
    let current =
        env::current_dir().map_err(|e| NekoError::config(format!("cannot get cwd: {e}")))?;
    let mut dir = current.as_path();
    loop {
        if dir.join("Cargo.toml").exists() {
            return Ok(dir.to_path_buf());
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return Ok(current),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_env_placeholder() {
        let secret = SecretString::from("${NEKO_TEST_KEY}".to_string());
        env::set_var("NEKO_TEST_KEY", "real-key");
        let expanded = expand_secret(&secret);
        assert_eq!(expanded.expose_secret(), "real-key");
    }

    #[test]
    fn leaves_literal_secret_unchanged() {
        let secret = SecretString::from("literal-key".to_string());
        let expanded = expand_secret(&secret);
        assert_eq!(expanded.expose_secret(), "literal-key");
    }
}
