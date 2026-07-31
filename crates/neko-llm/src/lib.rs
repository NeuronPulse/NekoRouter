pub mod embedding;
pub mod openai;

pub use embedding::OpenAiEmbeddingClient;
pub use openai::{extract_json, parse_or_fallback, OpenAiCompatibleClient};
