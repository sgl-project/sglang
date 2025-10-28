//! Web Search Preview Integration (MVP - Minimal)
//!
//! This module handles the web_search_preview tool type, which provides a simplified
//! interface for web search capabilities via MCP servers.
//!
//! Key responsibilities:
//! - Detect web_search_preview tool in requests
//! - Check MCP server availability
//! - Build web_search_call output items (status only, MVP)
//!
//! The actual transformation logic (MCP tools → function tools, mcp_call → web_search_call)
//! happens in other modules (prepare_mcp_payload_for_streaming, build_mcp_call_item).
//!
//! Future: search_context_size, user_location, results exposure

use std::sync::Arc;

use serde_json::{json, Value};

use crate::mcp::McpManager;
use crate::protocols::responses::{generate_id, ResponseTool, ResponseToolType};

// ============================================================================
// Tool Detection
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
}
