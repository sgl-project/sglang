//! Built-in tool type definitions
//!
//! Built-in tools are high-level abstractions that delegate to MCP servers.
//! They provide a cleaner, more user-friendly API for common operations.

use crate::protocols::responses::{ResponseTool, ResponseToolType};

/// Built-in tool types that map to dedicated MCP servers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinToolType {
    WebSearch,
    FileSearch,
    CodeInterpreter,
}

impl BuiltinToolType {
    /// Get the fixed label for this built-in tool type
    ///
    /// Labels are used to resolve which static MCP server to use.
    /// Each tool type has exactly ONE fixed label.
    pub fn fixed_label(&self) -> &'static str {
        match self {
            BuiltinToolType::WebSearch => "web_search",
            BuiltinToolType::FileSearch => "file_search",
            BuiltinToolType::CodeInterpreter => "code_interpreter",
        }
    }

    /// Get the synthetic name prefix for this built-in tool type
    ///
    /// Synthetic names are used internally to track built-in tool calls.
    pub fn synthetic_name(&self) -> String {
        format!("{}_builtin", self.fixed_label())
    }

    /// Convert from ResponseToolType to BuiltinToolType
    ///
    /// Returns Some if the tool type is a built-in tool, None otherwise
    pub fn from_response_tool_type(tool_type: &ResponseToolType) -> Option<Self> {
        match tool_type {
            ResponseToolType::WebSearch => Some(BuiltinToolType::WebSearch),
            ResponseToolType::FileSearch => Some(BuiltinToolType::FileSearch),
            ResponseToolType::CodeInterpreter => Some(BuiltinToolType::CodeInterpreter),
            _ => None,
        }
    }

    /// Convert from ResponseTool to BuiltinToolType
    ///
    /// Returns Some if the tool is a built-in tool, None otherwise
    pub fn from_response_tool(tool: &ResponseTool) -> Option<Self> {
        Self::from_response_tool_type(&tool.r#type)
    }

    /// Infer a BuiltinToolType from its synthetic tool name.
    /// Synthetic names are generated via `synthetic_name()` and prefixed with the fixed label.
    pub fn from_synthetic_name(name: &str) -> Option<Self> {
        if name.starts_with(&format!(
            "{}_builtin",
            BuiltinToolType::WebSearch.fixed_label()
        )) {
            Some(BuiltinToolType::WebSearch)
        } else if name.starts_with(&format!(
            "{}_builtin",
            BuiltinToolType::FileSearch.fixed_label()
        )) {
            Some(BuiltinToolType::FileSearch)
        } else if name.starts_with(&format!(
            "{}_builtin",
            BuiltinToolType::CodeInterpreter.fixed_label()
        )) {
            Some(BuiltinToolType::CodeInterpreter)
        } else {
            None
        }
    }
}

/// Represents a detected built-in tool call from LLM output
#[derive(Debug, Clone)]
pub struct BuiltinToolCall {
    pub tool_type: BuiltinToolType,
    pub call_id: String,
    pub synthetic_name: String,
    pub arguments: String,
}

/// Result from executing a built-in tool via MCP
#[derive(Debug, Clone)]
pub struct BuiltinToolResult {
    /// The built-in tool type that was executed
    pub tool_type: BuiltinToolType,
    /// The call ID from the function call
    pub call_id: String,
    /// The arguments sent to the MCP tool (JSON string)
    pub arguments: String,
    /// The raw MCP output
    pub mcp_output: serde_json::Value,
    /// Whether the call resulted in an error
    pub is_error: bool,
}
