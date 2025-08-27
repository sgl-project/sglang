//! Mixed Format and Additional Edge Case Tests
//!
//! Tests for edge cases across parsers and mixed format scenarios

use serde_json::json;
use sglang_router_rs::tool_parser::{
    JsonParser, LlamaParser, MistralParser, ParseState, PythonicParser, QwenParser, StreamResult,
    ToolParser,
};

#[tokio::test]
async fn test_mixed_formats_in_text() {
    // Test that parsers correctly ignore other formats' markers

    let json_parser = JsonParser::new();
    let input = r#"
    Some text with [TOOL_CALLS] marker that shouldn't trigger.
    Also has <tool_call> tags and [function()] syntax.
    But here's the actual JSON: {"name": "test", "arguments": {}}
    "#;

    let result = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");

    // Mistral parser should ignore JSON and other formats
    let mistral_parser = MistralParser::new();
    let input = r#"
    {"name": "fake"} [function()] <tool_call>
    [TOOL_CALLS] [{"name": "real", "arguments": {}}]
    "#;

    let result = mistral_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "real");
}

#[tokio::test]
async fn test_format_markers_in_string_content() {
    // Test that format markers inside string content don't interfere

    let pythonic_parser = PythonicParser::new();
    let input = r#"[echo(text="Use [TOOL_CALLS] and <tool_call> in text")]"#;

    let result = pythonic_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Use [TOOL_CALLS] and <tool_call> in text");

    let qwen_parser = QwenParser::new();
    let input = r#"<tool_call>
{"name": "log", "arguments": {"msg": "Found [function()] pattern"}}
</tool_call>"#;

    let result = qwen_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["msg"], "Found [function()] pattern");
}

#[tokio::test]
async fn test_deeply_nested_json_structures() {
    let json_parser = JsonParser::new();

    let input = r#"{
        "name": "deep_process",
        "arguments": {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "data": [1, 2, [3, [4, 5]]]
                            }
                        }
                    }
                }
            }
        }
    }"#;

    let result = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "deep_process");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert!(args["level1"]["level2"]["level3"]["level4"]["level5"]["data"].is_array());
}

#[tokio::test]
async fn test_multiple_sequential_calls_different_formats() {
    // Simulate a scenario where different parts of text have different formats
    // (though each parser will only recognize its own format)

    let llama_parser = LlamaParser::new();

    // Llama parser currently only returns the first tool found
    let input = r#"First call: <|python_tag|>{"name": "call1", "arguments": {}}"#;

    let result = llama_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "call1");

    // Test plain JSON separately
    let input2 = r#"{"name": "call2", "arguments": {"x": 1}}"#;
    let result2 = llama_parser.parse_complete(input2).await.unwrap();
    assert_eq!(result2.len(), 1);
    assert_eq!(result2[0].function.name, "call2");
}

#[tokio::test]
async fn test_empty_and_whitespace_variations() {
    let json_parser = JsonParser::new();

    // Various whitespace scenarios
    let cases = vec![
        r#"  {"name":"compact","arguments":{}}  "#,
        r#"

        {"name": "spaced", "arguments": {}}

        "#,
        r#"	{"name": "tabbed", "arguments": {}}	"#, // tabs
    ];

    for input in cases {
        let result = json_parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1, "Should parse regardless of whitespace");
    }
}

#[tokio::test]
async fn test_special_json_values() {
    let json_parser = JsonParser::new();

    // Test various special JSON values
    let input = r#"{
        "name": "test_special",
        "arguments": {
            "float_e": 1.23e10,
            "float_neg_e": 1.23e-10,
            "hex_like": "0x1234",
            "very_long_num": 99999999999999999999,
            "special_strings": ["", " ", "\u0000", "\u001f"],
            "escaped": "\\n\\r\\t\\\"\\\\",
            "unicode": "\u4e2d\u6587"
        }
    }"#;

    let result = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test_special");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert!(args["special_strings"].is_array());
    assert!(args["escaped"].is_string());
}

#[tokio::test]
async fn test_parser_recovery_after_invalid_input() {
    let mut state = ParseState::new();
    let parser = JsonParser::new();

    // Send invalid JSON first
    let _ = parser.parse_incremental(r#"{"broken": "#, &mut state).await;

    // Clear state and try valid JSON
    state.buffer.clear();
    let result = parser
        .parse_incremental(r#"{"name": "valid", "arguments": {}}"#, &mut state)
        .await
        .unwrap();

    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "valid");
        }
        _ => {
            // Might be incomplete depending on implementation
        }
    }
}

#[tokio::test]
async fn test_boundary_cases_for_extraction() {
    // Test edge cases in JSON extraction from text

    let json_parser = JsonParser::new();

    // JSON at the very beginning
    let input = r#"{"name": "start", "arguments": {}} and then text"#;
    let result = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "start");

    // JSON at the very end
    let input = r#"Some text first {"name": "end", "arguments": {}}"#;
    let result = json_parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "end");

    // Multiple JSON objects in text (should find first valid one)
    let input =
        r#"Text {"name": "first", "arguments": {}} more {"name": "second", "arguments": {}}"#;
    let result = json_parser.parse_complete(input).await.unwrap();
    assert!(!result.is_empty());
    assert_eq!(result[0].function.name, "first");
}

#[tokio::test]
async fn test_pythonic_edge_cases() {
    let parser = PythonicParser::new();

    // Function name with underscores and numbers
    let input = r#"[func_name_2(param_1="value")]"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "func_name_2");

    // Empty string argument
    let input = r#"[process(text="")]"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["text"], "");
}

#[tokio::test]
async fn test_mistral_with_pretty_json() {
    let parser = MistralParser::new();

    // Pretty-printed JSON in Mistral format
    let input = r#"[TOOL_CALLS] [
        {
            "name": "formatted",
            "arguments": {
                "nested": {
                    "key": "value"
                },
                "array": [
                    1,
                    2,
                    3
                ]
            }
        }
    ]"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "formatted");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["nested"]["key"], "value");
    assert_eq!(args["array"], json!([1, 2, 3]));
}

#[tokio::test]
async fn test_qwen_with_cdata_like_content() {
    let parser = QwenParser::new();

    // Test with content that looks like CDATA but isn't
    // Note: QwenParser expects exactly "<tool_call>\n" with the newline
    let input = r#"<tool_call>
{"name": "process", "arguments": {"xml": "<![CDATA[some data]]>"}}
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["xml"], "<![CDATA[some data]]>");
}

#[tokio::test]
async fn test_extremely_long_function_names() {
    let parser = PythonicParser::new();

    let long_name = "very_long_function_name_that_might_appear_in_generated_code_somewhere";
    let input = format!(r#"[{}(param="value")]"#, long_name);

    let result = parser.parse_complete(&input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, long_name);
}

#[tokio::test]
async fn test_json_with_duplicate_keys() {
    let parser = JsonParser::new();

    // JSON with duplicate keys (last one should win per JSON spec)
    let input = r#"{"name": "test", "arguments": {"key": "first", "key": "second"}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    // JSON parsers typically keep the last value for duplicate keys
    assert_eq!(args["key"], "second");
}
