//! MCP error types.
//!
//! Defines error variants for MCP operations including connection, tool execution,
//! and configuration errors.

use thiserror::Error;

pub type McpResult<T> = Result<T, McpError>;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("Server not found: {0}")]
    ServerNotFound(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Tool execution failed: {0}")]
    ToolExecution(String),

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    #[error("Prompt not found: {0}")]
    PromptNotFound(String),

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error(transparent)]
    Sdk(#[from] Box<rmcp::RmcpError>),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Http(#[from] reqwest::Error),
}
