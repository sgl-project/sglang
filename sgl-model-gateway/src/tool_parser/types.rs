use serde::{Deserialize, Serialize};

/// Parsed tool call from model output
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
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

/// Result of streaming parse operation (matches Python StreamingParseResult)
#[derive(Debug, Clone, Default)]
pub struct StreamingParseResult {
    /// Normal text that's not part of tool calls
    pub normal_text: String,
    /// Tool call items parsed from the chunk
    pub calls: Vec<ToolCallItem>,
}

/// Simple encapsulation of parsed tool call for streaming (matches Python ToolCallItem)
#[derive(Debug, Clone)]
pub struct ToolCallItem {
    /// Tool index in the array
    pub tool_index: usize,
    /// Tool name (only present on first chunk)
    pub name: Option<String>,
    /// Incremental JSON arguments
    pub parameters: String,
}

/// Format information for building structural tags
/// Contains the patterns needed to construct structural tags for constrained generation
pub struct FormatInfo {
    /// Function to generate begin pattern for a specific tool name and index
    /// Parameters: (tool_name, tool_index)
    pub begin_pattern: Box<dyn Fn(&str, usize) -> String + Send + Sync>,
    /// End pattern string
    pub end_pattern: String,
    /// Trigger token that starts the tool call sequence
    pub trigger: String,
    /// Optional function to generate begin pattern for subsequent tool calls (after the first one)
    /// Used for parsers with dual triggers (e.g., KimiK2, DeepSeek V3.1)
    /// If None, uses begin_pattern for all tool calls
    pub begin_pattern_subsequent: Option<Box<dyn Fn(&str, usize) -> String + Send + Sync>>,
    /// Optional second trigger token for subsequent tool calls
    /// If None, uses trigger for all tool calls
    pub trigger_subsequent: Option<String>,
}
