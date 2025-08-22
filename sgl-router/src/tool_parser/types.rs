use serde::{Deserialize, Serialize};

/// Parsed tool call from model output (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for the tool call
    pub id: String,
    /// Type of tool call (currently always "function")
    #[serde(rename = "type")]
    pub r#type: String,
    /// Function call details
    pub function: FunctionCall,
}

/// Function call within a tool call
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// Arguments as JSON string
    pub arguments: String,
}

/// Streaming parse result
#[derive(Debug, Clone)]
pub enum StreamResult {
    /// Need more data to continue parsing
    Incomplete,
    /// Found a tool name (for streaming)
    ToolName { index: usize, name: String },
    /// Found incremental arguments (for streaming)
    ToolArguments { index: usize, arguments: String },
    /// Completed parsing a tool
    ToolComplete(ToolCall),
    /// Normal text (not part of tool call)
    NormalText(String),
}

/// Token configuration for parsing
#[derive(Debug, Clone)]
pub struct TokenConfig {
    /// Start tokens for tool calls
    pub start_tokens: Vec<String>,
    /// End tokens for tool calls
    pub end_tokens: Vec<String>,
    /// Separator between multiple tool calls
    pub separator: String,
}

impl TokenConfig {
    /// Iterate over start/end token pairs
    pub fn iter_pairs(&self) -> impl Iterator<Item = (&str, &str)> {
        self.start_tokens
            .iter()
            .zip(self.end_tokens.iter())
            .map(|(s, e)| (s.as_str(), e.as_str()))
    }
}

/// Simple partial tool call for streaming
#[derive(Debug, Clone)]
pub struct PartialToolCall {
    /// Tool name (if parsed)
    pub name: Option<String>,
    /// Buffer for accumulating arguments
    pub arguments_buffer: String,
    /// Start position in the input buffer
    pub start_position: usize,
    /// Whether the name has been sent (for streaming)
    pub name_sent: bool,
    /// Arguments already streamed
    pub streamed_args: String,
}
