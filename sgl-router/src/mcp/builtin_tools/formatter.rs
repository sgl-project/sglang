//! Built-in tool output formatting
//!
//! Transforms raw MCP results into user-friendly, tool-specific output items.

use super::types::{BuiltinToolResult, BuiltinToolType};
use crate::protocols::responses::{ResponseOutputItem, WebSearchAction};

/// Formats MCP results as built-in tool output items
pub struct BuiltinToolFormatter;

impl BuiltinToolFormatter {
    /// Format MCP result as built-in tool output item
    pub fn format_output(result: BuiltinToolResult) -> Result<ResponseOutputItem, String> {
        match result.tool_type {
            BuiltinToolType::WebSearch => Self::format_web_search(result),
            BuiltinToolType::FileSearch => Self::format_file_search(result),
            BuiltinToolType::CodeInterpreter => Self::format_code_interpreter(result),
        }
    }

    /// Format web search result
    ///
    /// Extracts query from the original arguments sent to the MCP tool.
    /// Note: No results or error fields - only id, status, and action.
    fn format_web_search(result: BuiltinToolResult) -> Result<ResponseOutputItem, String> {
        // Parse arguments to extract the query
        let args: serde_json::Value = serde_json::from_str(&result.arguments)
            .map_err(|e| format!("Failed to parse arguments: {}", e))?;

        // Extract query from arguments
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown query")
            .to_string();

        // Build action
        let action = WebSearchAction {
            action_type: "search".to_string(),
            query,
        };

        // Build output item (no results or error fields)
        Ok(ResponseOutputItem::WebSearchCall {
            id: generate_web_search_id(&result.call_id),
            status: if result.is_error {
                "failed"
            } else {
                "completed"
            }
            .to_string(),
            action,
        })
    }

    /// Format file search result (placeholder)
    fn format_file_search(_result: BuiltinToolResult) -> Result<ResponseOutputItem, String> {
        Err("File search formatting not yet implemented".to_string())
    }

    /// Format code interpreter result (placeholder)
    fn format_code_interpreter(_result: BuiltinToolResult) -> Result<ResponseOutputItem, String> {
        Err("Code interpreter formatting not yet implemented".to_string())
    }
}

/// Generate web search call ID
///
/// Strips "call_" prefix if present and prefixes with "ws_" to indicate it's a web search call.
fn generate_web_search_id(call_id: &str) -> String {
    let id_without_prefix = call_id.strip_prefix("call_").unwrap_or(call_id);
    format!("ws_{}", id_without_prefix)
}
