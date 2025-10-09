//! Common Test Suite Trait
//!
//! Provides a trait-based approach for testing common parser behaviors.
//! Each parser implements this trait to provide format-specific inputs,
//! then inherits all common test implementations.

use serde_json::Value as JsonValue;
use sglang_router_rs::tool_parser::ToolParser;

/// Common test suite that all parsers should implement
#[allow(async_fn_in_trait)]
pub trait CommonParserTests {
    type Parser: ToolParser;

    // =============================================================================
    // FACTORY METHODS (Required - each parser must implement)
    // =============================================================================

    /// Create a new parser instance
    fn create_parser() -> Self::Parser;

    // =============================================================================
    // FORMAT-SPECIFIC INPUT GENERATORS (Required - format-specific)
    // =============================================================================

    /// Input with empty/no arguments: {"name": "ping", "arguments": {}}
    fn format_empty_args_input() -> &'static str;

    /// Input with a single tool call
    fn format_single_tool_input(name: &str, args_json: &str) -> String;

    /// Input with deeply nested JSON arguments
    fn format_nested_json_input() -> String {
        let nested_args = r#"{"level1": {"level2": {"level3": {"value": "deep"}}}}"#;
        Self::format_single_tool_input("deep_process", nested_args)
    }

    /// Input with unicode characters in arguments
    fn format_unicode_input(unicode_text: &str) -> String {
        let args = format!(r#"{{"text": "{}"}}"#, unicode_text);
        Self::format_single_tool_input("echo", &args)
    }

    /// Input with special JSON values (null, bool, numbers)
    fn format_special_json_values_input() -> String {
        let args = r#"{"null_val": null, "bool_true": true, "bool_false": false, "int": 42, "float": 3.14}"#;
        Self::format_single_tool_input("process", args)
    }

    /// Input with escaped characters in strings
    fn format_escaped_chars_input() -> String {
        let args = r#"{"text": "Line1\nLine2\tTab", "path": "C:\\Users\\test"}"#;
        Self::format_single_tool_input("test", args)
    }

    /// Input with very long string argument (10KB)
    fn format_long_string_input() -> String {
        let long_text = "A".repeat(10000);
        let args = format!(r#"{{"text": "{}"}}"#, long_text);
        Self::format_single_tool_input("process_text", &args)
    }

    /// Input with whitespace variations (compact, spaced, multiline)
    fn format_compact_json_input() -> &'static str;

    // =============================================================================
    // COMMON TEST IMPLEMENTATIONS (Inherited by all parsers)
    // =============================================================================

    /// Test: Empty input should return no tools
    async fn test_empty_input_impl() {
        let parser = Self::create_parser();
        let (_normal_text, tools) = parser.parse_complete("").await.unwrap();
        assert_eq!(
            tools.len(),
            0,
            "Parser should return no tools for empty input"
        );
    }

    /// Test: Plain text with no tool markers should return no tools
    async fn test_plain_text_impl() {
        let parser = Self::create_parser();
        let text = "This is just a regular response with no tool calls whatsoever.";
        let (normal_text, tools) = parser.parse_complete(text).await.unwrap();
        assert_eq!(
            tools.len(),
            0,
            "Parser should return no tools for plain text"
        );
        assert!(
            !normal_text.is_empty(),
            "Parser should preserve normal text"
        );
    }

    /// Test: Empty arguments should be parsed correctly
    async fn test_empty_arguments_impl() {
        let parser = Self::create_parser();
        let input = Self::format_empty_args_input();
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "ping");

        let args: JsonValue = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert!(args.is_object());
        assert!(args.as_object().unwrap().is_empty());
    }

    /// Test: Deeply nested JSON structures should be parsed
    async fn test_nested_json_impl() {
        let parser = Self::create_parser();
        let input = Self::format_nested_json_input();
        let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "deep_process");

        let args: JsonValue = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert!(args["level1"]["level2"]["level3"]["value"].is_string());
        assert_eq!(args["level1"]["level2"]["level3"]["value"], "deep");
    }

    /// Test: Unicode characters should be handled correctly
    async fn test_unicode_impl() {
        let parser = Self::create_parser();

        let test_cases = vec![
            ("emoji", "🎉🎊✨"),
            ("chinese", "你好世界"),
            ("japanese", "こんにちは"),
            ("arabic", "مرحبا"),
            ("mixed", "Hello 世界 🌍"),
        ];

        for (test_name, unicode_text) in test_cases {
            let input = Self::format_unicode_input(unicode_text);
            let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
            assert_eq!(tools.len(), 1, "Failed on test: {}", test_name);

            let args: JsonValue = serde_json::from_str(&tools[0].function.arguments).unwrap();
            assert_eq!(args["text"], unicode_text, "Failed on test: {}", test_name);
        }
    }

    /// Test: Special JSON values (null, boolean, numbers) should be parsed
    async fn test_special_json_values_impl() {
        let parser = Self::create_parser();
        let input = Self::format_special_json_values_input();
        let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(tools.len(), 1);

        let args: JsonValue = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert!(args["null_val"].is_null());
        assert_eq!(args["bool_true"], true);
        assert_eq!(args["bool_false"], false);
        assert_eq!(args["int"], 42);
        assert_eq!(args["float"], 3.14);
    }

    /// Test: Escaped characters in strings should be preserved
    async fn test_escaped_chars_impl() {
        let parser = Self::create_parser();
        let input = Self::format_escaped_chars_input();
        let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(tools.len(), 1);

        let args: JsonValue = serde_json::from_str(&tools[0].function.arguments).unwrap();
        let text = args["text"].as_str().unwrap();
        assert!(text.contains('\n'), "Should have newline");
        assert!(text.contains('\t'), "Should have tab");
    }

    /// Test: Very long strings should be handled
    async fn test_long_string_impl() {
        let parser = Self::create_parser();
        let input = Self::format_long_string_input();
        let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(tools.len(), 1);

        let args: JsonValue = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert_eq!(args["text"].as_str().unwrap().len(), 10000);
    }

    /// Test: Compact JSON (no extra whitespace) should be parsed
    async fn test_compact_json_impl() {
        let parser = Self::create_parser();
        let input = Self::format_compact_json_input();
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1, "Failed to parse compact JSON");
    }

    /// Test: Incomplete/malformed JSON should be handled gracefully
    async fn test_malformed_json_impl() {
        let parser = Self::create_parser();
        // This is intentionally malformed - parser should not panic
        let input = r#"{"name": "incomplete""#;
        let result = parser.parse_complete(input).await;
        // Should either return error or return 0 tools, but not panic
        match result {
            Ok((_text, tools)) => assert_eq!(tools.len(), 0),
            Err(_) => {} // Also acceptable
        }
    }
}
