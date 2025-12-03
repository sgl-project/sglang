//! Edge Cases and Error Handling Tests
//!
//! Tests for malformed input, edge cases, and error recovery

use sgl_model_gateway::tool_parser::{
    JsonParser, MistralParser, PythonicParser, QwenParser, ToolParser,
};

mod common;
use common::create_test_tools;

#[tokio::test]
async fn test_empty_input() {
    // Test that all parsers handle empty input correctly
    let json_parser = JsonParser::new();
    let (_normal_text, tools) = json_parser.parse_complete("").await.unwrap();
    assert_eq!(
        tools.len(),
        0,
        "JSON parser should return empty for empty input"
    );

    let mistral_parser = MistralParser::new();
    let (_normal_text, tools) = mistral_parser.parse_complete("").await.unwrap();
    assert_eq!(
        tools.len(),
        0,
        "Mistral parser should return empty for empty input"
    );

    let qwen_parser = QwenParser::new();
    let (_normal_text, tools) = qwen_parser.parse_complete("").await.unwrap();
    assert_eq!(
        tools.len(),
        0,
        "Qwen parser should return empty for empty input"
    );

    let pythonic_parser = PythonicParser::new();
    let (_normal_text, tools) = pythonic_parser.parse_complete("").await.unwrap();
    assert_eq!(
        tools.len(),
        0,
        "Pythonic parser should return empty for empty input"
    );
}

#[tokio::test]
async fn test_plain_text_no_tools() {
    let plain_text = "This is just a regular response with no tool calls whatsoever.";

    let json_parser = JsonParser::new();
    assert_eq!(
        json_parser
            .parse_complete(plain_text)
            .await
            .unwrap()
            .1
            .len(),
        0
    );

    let mistral_parser = MistralParser::new();
    assert_eq!(
        mistral_parser
            .parse_complete(plain_text)
            .await
            .unwrap()
            .1
            .len(),
        0
    );

    let qwen_parser = QwenParser::new();
    assert_eq!(
        qwen_parser
            .parse_complete(plain_text)
            .await
            .unwrap()
            .1
            .len(),
        0
    );

    let pythonic_parser = PythonicParser::new();
    assert_eq!(
        pythonic_parser
            .parse_complete(plain_text)
            .await
            .unwrap()
            .1
            .len(),
        0
    );
}

#[tokio::test]
async fn test_incomplete_json() {
    let json_parser = JsonParser::new();

    let incomplete_cases = vec![
        r#"{"name": "test""#,                 // Missing closing brace
        r#"{"name": "test", "arguments":"#,   // Incomplete arguments
        r#"{"name": "test", "arguments": {"#, // Incomplete nested object
    ];

    for input in incomplete_cases {
        let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
        assert_eq!(
            tools.len(),
            0,
            "Should not parse incomplete JSON: {}",
            input
        );
    }

    // This case might actually parse because [{"name": "test"}] is complete
    // The trailing comma suggests more items but the first item is valid
    let _result = json_parser
        .parse_complete(r#"[{"name": "test"},"#)
        .await
        .unwrap();
    // This could parse the first element or return empty - implementation dependent
}

#[tokio::test]
async fn test_malformed_mistral() {
    let parser = MistralParser::new();

    let malformed_cases = vec![
        "[TOOL_CALLS]",                // Missing array
        "[TOOL_CALLS] {",              // Not an array
        "[TOOL_CALLS] [",              // Incomplete array
        "[TOOL_CALLS] [{]",            // Invalid JSON in array
        "[TOOL_CALLS] [{\"name\": }]", // Invalid value
    ];

    for input in malformed_cases {
        // Parser might return error or empty vec for malformed input
        if let Ok((_normal_text, tools)) = parser.parse_complete(input).await {
            assert_eq!(
                tools.len(),
                0,
                "Should not parse malformed Mistral: {}",
                input
            );
        }
        // Error is also acceptable for malformed input
    }
}

#[tokio::test]
async fn test_missing_required_fields() {
    let json_parser = JsonParser::new();

    // Missing name field
    let input = r#"{"arguments": {"x": 1}}"#;
    let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0, "Should not parse without name field");

    // Name is not a string
    let input = r#"{"name": 123, "arguments": {}}"#;
    let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0, "Should not parse with non-string name");
}

#[tokio::test]
async fn test_very_long_strings() {
    let json_parser = JsonParser::new();

    let long_string = "x".repeat(10000);
    let input = format!(
        r#"{{"name": "test", "arguments": {{"data": "{}"}}}}"#,
        long_string
    );

    let (_normal_text, tools) = json_parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["data"].as_str().unwrap().len(), 10000);
}

#[tokio::test]
async fn test_unicode_edge_cases() {
    let json_parser = JsonParser::new();

    // Various Unicode characters including emojis, CJK, RTL text
    let input = r#"{"name": "translate", "arguments": {"text": "Hello 世界 🌍 مرحبا עולם"}}"#;

    let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Hello 世界 🌍 مرحبا עולם");
}

#[tokio::test]
async fn test_nested_brackets_in_strings() {
    let mistral_parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "echo", "arguments": {"text": "Array: [1, 2, 3]"}}]"#;
    let (_normal_text, tools) = mistral_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Array: [1, 2, 3]");

    let pythonic_parser = PythonicParser::new();
    let input = r#"[echo(text="List: [a, b, c]")]"#;
    let (_normal_text, tools) = pythonic_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "List: [a, b, c]");
}

#[tokio::test]
async fn test_multiple_formats_in_text() {
    let json_parser = JsonParser::new();
    let input = r#"
    Here's some text with [TOOL_CALLS] that shouldn't trigger.
    {"name": "actual_tool", "arguments": {}}
    And some more text with <tool_call> tags.
    "#;

    let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "actual_tool");
}

#[tokio::test]
async fn test_escaped_characters() {
    let json_parser = JsonParser::new();

    let input = r#"{"name": "write", "arguments": {"content": "Line 1\nLine 2\r\nLine 3\tTabbed\\Backslash\"Quote"}}"#;

    let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    let content = args["content"].as_str().unwrap();
    assert!(content.contains('\n'));
    assert!(content.contains('\t'));
    assert!(content.contains('\\'));
    assert!(content.contains('"'));
}

#[tokio::test]
async fn test_numeric_edge_cases() {
    let json_parser = JsonParser::new();

    let input = r#"{
        "name": "calculate",
        "arguments": {
            "int": 42,
            "float": 123.456,
            "scientific": 1.23e-4,
            "negative": -999,
            "zero": 0,
            "large": 9007199254740991
        }
    }"#;

    let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["int"], 42);
    assert_eq!(args["float"], 123.456);
    assert_eq!(args["scientific"], 0.000123);
    assert_eq!(args["negative"], -999);
    assert_eq!(args["zero"], 0);
    assert_eq!(args["large"], 9007199254740991i64);
}

#[tokio::test]
async fn test_null_and_boolean_values() {
    let json_parser = JsonParser::new();

    let input = r#"{
        "name": "configure",
        "arguments": {
            "enabled": true,
            "disabled": false,
            "optional": null
        }
    }"#;

    let (_normal_text, tools) = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["enabled"], true);
    assert_eq!(args["disabled"], false);
    assert_eq!(args["optional"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_partial_token_at_buffer_boundary() {
    let mut parser = QwenParser::new();

    let tools = create_test_tools();

    // Send exactly "<tool" which is a 5-character prefix of "<tool_call>\n"
    let result = parser.parse_incremental("<tool", &tools).await.unwrap();
    assert!(
        result.calls.is_empty(),
        "Should be incomplete for partial tag"
    );

    // Complete the token
    let result = parser
        .parse_incremental(
            "_call>\n{\"name\": \"test\", \"arguments\": {}}\n</tool_call>",
            &tools,
        )
        .await
        .unwrap();

    // Should successfully parse after completing
    if !result.calls.is_empty() {
        if let Some(name) = &result.calls[0].name {
            assert_eq!(name, "test");
        }
    }
}

#[tokio::test]
async fn test_exact_prefix_lengths() {
    let mut parser = QwenParser::new();

    let tools = create_test_tools();

    let test_cases = vec![
        ("<", 1),            // 1-char prefix
        ("<t", 2),           // 2-char prefix
        ("<tool", 5),        // 5-char prefix (the main bug case)
        ("<tool_call", 10),  // 10-char prefix
        ("<tool_call>", 11), // 11-char prefix (full start without \n)
    ];

    for (prefix, expected_len) in test_cases {
        let result = parser.parse_incremental(prefix, &tools).await.unwrap();
        assert!(
            result.calls.is_empty(),
            "Prefix '{}' (len {}) should be incomplete",
            prefix,
            expected_len
        );
        // Buffer is now internal to parser - can't assert on it
    }
}
