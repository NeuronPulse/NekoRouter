use std::fmt;

/// Top-level error type used across NekoRouter crates.
#[derive(Debug)]
pub enum NekoError {
    /// Configuration loading or validation failed.
    Config(String),
    /// Database operation failed.
    Database(String),
    /// LLM provider returned an error or produced invalid output.
    Llm(String),
    /// Network / transport failure.
    Transport(String),
    /// Message parsing failed.
    Parse(String),
    /// A requested resource or handler is missing.
    Missing(String),
    /// Catch-all for unexpected conditions.
    Other(String),
}

impl fmt::Display for NekoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NekoError::Config(msg) => write!(f, "config error: {msg}"),
            NekoError::Database(msg) => write!(f, "database error: {msg}"),
            NekoError::Llm(msg) => write!(f, "llm error: {msg}"),
            NekoError::Transport(msg) => write!(f, "transport error: {msg}"),
            NekoError::Parse(msg) => write!(f, "parse error: {msg}"),
            NekoError::Missing(msg) => write!(f, "missing: {msg}"),
            NekoError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for NekoError {}

impl NekoError {
    pub fn config<S: Into<String>>(msg: S) -> Self {
        NekoError::Config(msg.into())
    }

    pub fn database<S: Into<String>>(msg: S) -> Self {
        NekoError::Database(msg.into())
    }

    pub fn llm<S: Into<String>>(msg: S) -> Self {
        NekoError::Llm(msg.into())
    }

    pub fn transport<S: Into<String>>(msg: S) -> Self {
        NekoError::Transport(msg.into())
    }

    pub fn parse<S: Into<String>>(msg: S) -> Self {
        NekoError::Parse(msg.into())
    }

    pub fn missing<S: Into<String>>(msg: S) -> Self {
        NekoError::Missing(msg.into())
    }

    pub fn other<S: Into<String>>(msg: S) -> Self {
        NekoError::Other(msg.into())
    }
}
