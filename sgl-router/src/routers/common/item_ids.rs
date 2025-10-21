//! Item ID Generation
//!
//! Utilities for generating unique item IDs with proper prefixes.
//! Used by both OpenAI and gRPC routers.

use uuid::Uuid;

// ============================================================================
// ID Prefixes
// ============================================================================

pub const PREFIX_MCP_LIST_TOOLS: &str = "mcpl";
pub const PREFIX_MCP_CALL: &str = "mcp";
pub const PREFIX_MESSAGE: &str = "msg";
pub const PREFIX_REASONING: &str = "rs";
pub const PREFIX_FUNCTION_CALL: &str = "fc";

// ============================================================================
// ID Generation
// ============================================================================

/// Generate a unique item ID with the given prefix
pub fn generate_item_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

/// Generate a specific type of item ID
pub fn generate_mcp_list_tools_id() -> String {
    generate_item_id(PREFIX_MCP_LIST_TOOLS)
}

pub fn generate_mcp_call_id() -> String {
    generate_item_id(PREFIX_MCP_CALL)
}

pub fn generate_message_id() -> String {
    generate_item_id(PREFIX_MESSAGE)
}

pub fn generate_reasoning_id() -> String {
    generate_item_id(PREFIX_REASONING)
}

pub fn generate_function_call_id() -> String {
    generate_item_id(PREFIX_FUNCTION_CALL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_item_id() {
        let id = generate_item_id("test");
        assert!(id.starts_with("test_"));
        assert_eq!(id.len(), 5 + 36); // "test_" + UUID
    }

    #[test]
    fn test_generate_specific_ids() {
        assert!(generate_mcp_call_id().starts_with("mcp_"));
        assert!(generate_message_id().starts_with("msg_"));
        assert!(generate_reasoning_id().starts_with("rs_"));
    }
}
