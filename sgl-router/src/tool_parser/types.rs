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
/// Can return both normal text AND tool calls in the same response
#[derive(Debug, Clone)]
pub struct StreamingParseResult {
    /// Normal text content (equivalent to Python's normal_text field)
    pub normal_text: String,
    /// Tool calls found during parsing (equivalent to Python's calls field)
    pub tool_calls: Vec<ToolCallItem>,
}

impl StreamingParseResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self {
            normal_text: String::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Create result with only normal text
    pub fn with_normal_text(text: String) -> Self {
        Self {
            normal_text: text,
            tool_calls: Vec::new(),
        }
    }

    /// Create result with only tool calls
    pub fn with_tool_calls(calls: Vec<ToolCallItem>) -> Self {
        Self {
            normal_text: String::new(),
            tool_calls: calls,
        }
    }

    /// Create result with both normal text and tool calls
    pub fn with_both(text: String, calls: Vec<ToolCallItem>) -> Self {
        Self {
            normal_text: text,
            tool_calls: calls,
        }
    }
}

impl Default for StreamingParseResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool call item for streaming (matching Python's ToolCallItem)
#[derive(Debug, Clone)]
pub struct ToolCallItem {
    /// Tool index for this call
    pub tool_index: usize,
    /// Tool ID (sent only on first chunk)
    pub id: Option<String>,
    /// Function name (sent only on first chunk)
    pub name: Option<String>,
    /// Incremental arguments chunk
    pub arguments_delta: String,
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
    /// Tool index (for multiple concurrent tools)
    pub index: usize,
    /// Tool name (if parsed)
    pub name: Option<String>,
    /// Whether name has been emitted to stream
    pub name_sent: bool,
    /// Buffer for accumulating arguments
    pub arguments_buffer: String,
    /// Arguments already streamed
    pub streamed_arguments: String,
    /// Tool ID (generated once per tool)
    pub id: Option<String>,
}
