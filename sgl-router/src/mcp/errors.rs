use std::fmt;

#[derive(Debug, Clone)]
pub enum MCPError {
    ConnectionError(String),
    ExecutionError(String),
    TimeoutError(String),
    ParseError(String),
    ValidationError(String),
    ConfigurationError(String),
    ProtocolError(String),
}

impl fmt::Display for MCPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MCPError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            MCPError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            MCPError::TimeoutError(msg) => write!(f, "Timeout error: {}", msg),
            MCPError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            MCPError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            MCPError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            MCPError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl std::error::Error for MCPError {}

pub type MCPResult<T> = Result<T, MCPError>;