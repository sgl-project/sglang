//! Common utilities for routers

/// Strip server_label prefix from qualified MCP tool name
///
/// MCP tools are stored internally with qualified names (server_label__tool_name)
/// to support multiple servers with the same tool names. This function extracts
/// the original tool name for display purposes.
pub fn strip_server_label(qualified_name: &str) -> &str {
    qualified_name
        .split_once("__")
        .map(|(_, tool_name)| tool_name)
        .unwrap_or(qualified_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_server_label() {
        assert_eq!(
            strip_server_label("brave__brave_web_search"),
            "brave_web_search"
        );
        assert_eq!(strip_server_label("server__tool"), "tool");
        assert_eq!(strip_server_label("no_prefix"), "no_prefix");
    }
}
