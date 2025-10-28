//! Web Search Preview Integration (MVP - Minimal)
//!
//! This module handles transformation between OpenAI's web_search_preview format
//! and MCP-based web search tool calls.
//!
//! MVP Scope:
//! - Detect web_search_preview tool in requests
//! - Check MCP server availability
//! - Transform to/from function calls
//! - Build minimal web_search_call output items (status only)
//!
//! Future: search_context_size, user_location, results exposure

use std::sync::Arc;

use serde_json::{json, Value};

use crate::mcp::McpManager;
use crate::protocols::responses::{ResponseTool, ResponseToolType};

use super::utils::generate_id;

// ============================================================================
// Tool Detection & Transformation
// ============================================================================

/// Detect if request has web_search_preview tool
pub fn has_web_search_preview_tool(tools: &[ResponseTool]) -> bool {
    tools
        .iter()
        .any(|t| matches!(t.r#type, ResponseToolType::WebSearchPreview))
}

/// Check if MCP server "web_search_preview" is available
pub async fn is_web_search_mcp_available(mcp_manager: &Arc<McpManager>) -> bool {
    mcp_manager
        .get_client("web_search_preview")
        .await
        .is_some()
}

/// Transform web_search_preview tool to MCP function tools
/// Returns function tools from the "web_search_preview" MCP server
pub fn transform_web_search_to_mcp_functions(mcp_manager: &Arc<McpManager>) -> Vec<Value> {
    // Get tools from the "web_search_preview" MCP server
    let tools = mcp_manager.list_tools();

    tools
        .iter()
        .filter(|t| t.server == "web_search_preview")
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.clone().unwrap_or_else(|| json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }))
            })
        })
        .collect()
}

// ============================================================================
// Output Item Builders (MVP - Status Only)
// ============================================================================

/// Build a web_search_call output item (MVP - status only)
///
/// The MCP search results are passed to the LLM internally via function_call_output,
/// but we don't expose them in the web_search_call item to the client.
pub fn build_web_search_call_item() -> Value {
    json!({
        "id": generate_id("ws"),
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search"
        }
    })
}

/// Build a failed web_search_call output item
pub fn build_web_search_call_item_failed(error: &str) -> Value {
    json!({
        "id": generate_id("ws"),
        "type": "web_search_call",
        "status": "failed",
        "action": {
            "type": "search"
        },
        "error": error
    })
}

/// Convert mcp_call item to web_search_call item (MVP - minimal)
pub fn mcp_call_to_web_search_call(mcp_call_item: &Value) -> Value {
    let status = mcp_call_item
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("completed");

    if status != "completed" {
        // Return failed web_search_call
        let error = mcp_call_item
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        return build_web_search_call_item_failed(error);
    }

    // Return successful web_search_call (status only, no results)
    build_web_search_call_item()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_web_search_preview_tool() {
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearchPreview,
            server_url: None,
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];
        assert!(has_web_search_preview_tool(&tools));

        let empty_tools: Vec<ResponseTool> = vec![];
        assert!(!has_web_search_preview_tool(&empty_tools));

        let other_tools = vec![ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some("http://example.com".to_string()),
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];
        assert!(!has_web_search_preview_tool(&other_tools));
    }

    #[test]
    fn test_build_web_search_call_item() {
        let item = build_web_search_call_item();
        assert_eq!(item["type"], "web_search_call");
        assert_eq!(item["status"], "completed");
        assert_eq!(item["action"]["type"], "search");
        assert!(item.get("results").is_none()); // No results in MVP
        assert!(item.get("id").is_some());
    }

    #[test]
    fn test_build_web_search_call_item_failed() {
        let item = build_web_search_call_item_failed("Test error");
        assert_eq!(item["type"], "web_search_call");
        assert_eq!(item["status"], "failed");
        assert_eq!(item["error"], "Test error");
        assert_eq!(item["action"]["type"], "search");
    }

    #[test]
    fn test_mcp_call_to_web_search_call_success() {
        let mcp_call = json!({
            "type": "mcp_call",
            "status": "completed",
            "server_label": "web_search_preview",
            "output": "search results here"
        });

        let ws_call = mcp_call_to_web_search_call(&mcp_call);
        assert_eq!(ws_call["type"], "web_search_call");
        assert_eq!(ws_call["status"], "completed");
        assert!(ws_call.get("results").is_none()); // MVP: no results
    }

    #[test]
    fn test_mcp_call_to_web_search_call_failed() {
        let mcp_call = json!({
            "type": "mcp_call",
            "status": "failed",
            "server_label": "web_search_preview",
            "error": "Search failed"
        });

        let ws_call = mcp_call_to_web_search_call(&mcp_call);
        assert_eq!(ws_call["type"], "web_search_call");
        assert_eq!(ws_call["status"], "failed");
        assert_eq!(ws_call["error"], "Search failed");
    }
}
