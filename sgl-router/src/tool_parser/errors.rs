use thiserror::Error;

/// Result type for tool parser operations
pub type ParserResult<T> = Result<T, ParserError>;

/// Errors that can occur during tool parsing
#[derive(Debug, Error)]
pub enum ParserError {
    #[error("Parsing failed: {0}")]
    ParsingFailed(String),

    #[error("Model not supported: {0}")]
    ModelNotSupported(String),

    #[error("Parse depth exceeded: max {0}")]
    DepthExceeded(usize),

    #[error("Invalid JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),

    #[error("Incomplete tool call")]
    Incomplete,

    #[error("Invalid tool name: {0}")]
    InvalidToolName(String),

    #[error("Token not found: {0}")]
    TokenNotFound(String),
}
