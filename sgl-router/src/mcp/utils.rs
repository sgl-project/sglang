//! MCP utility functions for tool name formatting and parsing.

use once_cell::sync::Lazy;
use regex::Regex;

/// Separator for formatted tool names: `server_label__tool_name`
pub const TOOL_NAME_SEPARATOR: &str = "__";

/// Regex pattern: `server_label__tool_name`
/// Captures: (server_label, tool_name)
static TOOL_NAME_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^([^_]+(?:_[^_]+)*)__(.+)$").unwrap());

/// Format tool name: `server_label__tool_name`
pub fn format_tool_name(server_label: &str, tool_name: &str) -> String {
    format!("{}{}{}", server_label, TOOL_NAME_SEPARATOR, tool_name)
}

/// Parse formatted tool name into (server_label, tool_name)
pub fn parse_tool_name(formatted_name: &str) -> Option<(String, String)> {
    TOOL_NAME_PATTERN
        .captures(formatted_name)
        .map(|caps| (caps[1].to_string(), caps[2].to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_and_parse() {
        let formatted = format_tool_name("my-server", "get_weather");
        assert_eq!(formatted, "my-server__get_weather");

        let (server, tool) = parse_tool_name(&formatted).unwrap();
        assert_eq!(server, "my-server");
        assert_eq!(tool, "get_weather");
    }

    #[test]
    fn test_parse_with_underscores() {
        let (server, tool) = parse_tool_name("my_server__get_weather_info").unwrap();
        assert_eq!(server, "my_server");
        assert_eq!(tool, "get_weather_info");
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_tool_name("no_separator").is_none());
    }
}
