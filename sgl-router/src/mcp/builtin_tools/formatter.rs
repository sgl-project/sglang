//! Built-in tool output formatting
//!
//! Transforms raw MCP results into user-friendly, tool-specific output items.
//! IMPORTANT: Built-in tools NEVER produce `mcp_call` or `mcp_list_tools` outputs.

use super::types::{BuiltinToolResult, BuiltinToolType};
use crate::protocols::responses::{ResponseOutputItem, WebSearchAction};

/// Formats MCP results as built-in tool output items
///
/// IMPORTANT: Built-in tools NEVER produce `mcp_call` or `mcp_list_tools` outputs.
/// They always produce tool-specific outputs like `web_search_call`, `file_search_call`, etc.
pub struct BuiltinToolFormatter;

impl BuiltinToolFormatter {
    /// Format MCP result as built-in tool output item
    ///
    /// # Arguments
    /// * `result` - Built-in tool result from MCP execution
    ///
    /// # Returns
    /// * `Ok(ResponseOutputItem)` - Formatted output item
    /// * `Err(String)` - Formatting error
    ///
    /// # Example
    /// ```ignore
    /// let result = BuiltinToolResult {
    ///     tool_type: BuiltinToolType::WebSearch,
    ///     call_id: "call_123".to_string(),
    ///     mcp_output: json!({"query": "restaurants"}),
    ///     is_error: false,
    /// };
    /// let output = BuiltinToolFormatter::format_output(result)?;
    /// // Returns: ResponseOutputItem::WebSearchCall { ... }
    /// ```
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
/// Examples:
/// - "call_abc123" -> "ws_abc123"
/// - "abc123" -> "ws_abc123"
fn generate_web_search_id(call_id: &str) -> String {
    let id_without_prefix = call_id.strip_prefix("call_").unwrap_or(call_id);
    format!("ws_{}", id_without_prefix)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_format_web_search_success() {
        let result = BuiltinToolResult {
            tool_type: BuiltinToolType::WebSearch,
            call_id: "call_123".to_string(),
            arguments: r#"{"query": "restaurants Seattle"}"#.to_string(),
            mcp_output: json!({"results": []}),
            is_error: false,
        };

        let output = BuiltinToolFormatter::format_output(result).unwrap();

        match output {
            ResponseOutputItem::WebSearchCall { id, status, action } => {
                assert_eq!(id, "ws_123");
                assert_eq!(status, "completed");
                assert_eq!(action.action_type, "search");
                assert_eq!(action.query, "restaurants Seattle");
            }
            _ => panic!("Expected WebSearchCall"),
        }
    }

    #[test]
    fn test_format_web_search_error() {
        let result = BuiltinToolResult {
            tool_type: BuiltinToolType::WebSearch,
            call_id: "call_456".to_string(),
            arguments: r#"{"query": "test"}"#.to_string(),
            mcp_output: json!({"error": "Connection failed"}),
            is_error: true,
        };

        let output = BuiltinToolFormatter::format_output(result).unwrap();

        match output {
            ResponseOutputItem::WebSearchCall { id, status, .. } => {
                assert_eq!(id, "ws_456");
                assert_eq!(status, "failed");
            }
            _ => panic!("Expected WebSearchCall"),
        }
    }

    #[test]
    fn test_format_web_search_missing_query() {
        let result = BuiltinToolResult {
            tool_type: BuiltinToolType::WebSearch,
            call_id: "call_789".to_string(),
            arguments: r#"{}"#.to_string(),
            mcp_output: json!({}),
            is_error: false,
        };

        let output = BuiltinToolFormatter::format_output(result).unwrap();

        match output {
            ResponseOutputItem::WebSearchCall { action, .. } => {
                assert_eq!(action.query, "unknown query");
            }
            _ => panic!("Expected WebSearchCall"),
        }
    }

    #[test]
    fn test_generate_web_search_id() {
        assert_eq!(generate_web_search_id("abc123"), "ws_abc123");
        assert_eq!(generate_web_search_id("call_456"), "ws_456");
        assert_eq!(
            generate_web_search_id("call_cbbc3f10587a467db0ca6691"),
            "ws_cbbc3f10587a467db0ca6691"
        );
    }

    #[test]
    fn test_file_search_not_implemented() {
        let result = BuiltinToolResult {
            tool_type: BuiltinToolType::FileSearch,
            call_id: "call_123".to_string(),
            arguments: r#"{}"#.to_string(),
            mcp_output: json!({}),
            is_error: false,
        };

        let err = BuiltinToolFormatter::format_output(result).unwrap_err();
        assert!(err.contains("not yet implemented"));
    }

    #[test]
    fn test_code_interpreter_not_implemented() {
        let result = BuiltinToolResult {
            tool_type: BuiltinToolType::CodeInterpreter,
            call_id: "call_123".to_string(),
            arguments: r#"{}"#.to_string(),
            mcp_output: json!({}),
            is_error: false,
        };

        let err = BuiltinToolFormatter::format_output(result).unwrap_err();
        assert!(err.contains("not yet implemented"));
    }
}
