use super::*;
use crate::tool_parser::{
    parsers::JsonParser,
    partial_json::{compute_diff, find_common_prefix, is_complete_json, PartialJson},
    traits::ToolParser,
};

#[tokio::test]
async fn test_tool_parser_factory() {
    let factory = ParserFactory::new();

    // Test that we can get a pooled parser
    let pooled_parser = factory.get_pooled("gpt-4");
    let parser = pooled_parser.lock().await;
    assert!(parser.has_tool_markers(r#"{"name": "test", "arguments": {}}"#));
}

#[tokio::test]
async fn test_tool_parser_factory_model_mapping() {
    let factory = ParserFactory::new();

    // Test model mapping
    factory.registry().map_model("test-model", "json");

    // Get parser for the test model
    let pooled_parser = factory.get_pooled("test-model");
    let parser = pooled_parser.lock().await;
    assert!(parser.has_tool_markers(r#"{"name": "test", "arguments": {}}"#));
}

#[test]
fn test_tool_call_serialization() {
    let tool_call = ToolCall {
        function: FunctionCall {
            name: "search".to_string(),
            arguments: r#"{"query": "rust programming"}"#.to_string(),
        },
    };

    let json = serde_json::to_string(&tool_call).unwrap();
    assert!(json.contains("search"));
    assert!(json.contains("rust programming"));

    let parsed: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.function.name, "search");
    assert_eq!(
        parsed.function.arguments,
        r#"{"query": "rust programming"}"#
    );
}

#[test]
fn test_partial_json_parser() {
    let parser = PartialJson::default();

    let input = r#"{"name": "test", "value": 42}"#;
    let (value, consumed) = parser.parse_value(input, true).unwrap();
    assert_eq!(value["name"], "test");
    assert_eq!(value["value"], 42);
    assert_eq!(consumed, input.len());

    let input = r#"{"name": "test", "value": "#;
    let (value, _consumed) = parser.parse_value(input, true).unwrap();
    assert_eq!(value["name"], "test");
    assert!(value["value"].is_null());

    let input = r#"{"name": "tes"#;
    let (value, _consumed) = parser.parse_value(input, true).unwrap();
    assert_eq!(value["name"], "tes");

    let input = r#"[1, 2, "#;
    let (value, _consumed) = parser.parse_value(input, true).unwrap();
    assert!(value.is_array());
    assert_eq!(value[0], 1);
    assert_eq!(value[1], 2);
}

#[test]
fn test_partial_json_depth_limit() {
    // max_depth of 3 allows nesting up to 3 levels
    // Set allow_incomplete to false to get errors instead of partial results
    let parser = PartialJson::new(3, false);

    // This should work (simple object)
    let input = r#"{"a": 1}"#;
    let result = parser.parse_value(input, true);
    assert!(result.is_ok());

    // This should work (nested to depth 3)
    let input = r#"{"a": {"b": {"c": 1}}}"#;
    let result = parser.parse_value(input, true);
    assert!(result.is_ok());

    // This should fail (nested to depth 4, exceeds limit)
    let input = r#"{"a": {"b": {"c": {"d": 1}}}}"#;
    let result = parser.parse_value(input, true);
    assert!(result.is_err());
}

#[test]
fn test_is_complete_json() {
    assert!(is_complete_json(r#"{"name": "test"}"#));
    assert!(is_complete_json(r#"[1, 2, 3]"#));
    assert!(is_complete_json(r#""string""#));
    assert!(is_complete_json("42"));
    assert!(is_complete_json("true"));
    assert!(is_complete_json("null"));

    assert!(!is_complete_json(r#"{"name": "#));
    assert!(!is_complete_json(r#"[1, 2, "#));
    assert!(!is_complete_json(r#""unclosed"#));
}

#[test]
fn test_find_common_prefix() {
    assert_eq!(find_common_prefix("hello", "hello"), 5);
    assert_eq!(find_common_prefix("hello", "help"), 3);
    assert_eq!(find_common_prefix("hello", "world"), 0);
    assert_eq!(find_common_prefix("", "hello"), 0);
    assert_eq!(find_common_prefix("hello", ""), 0);
}

#[test]
fn test_compute_diff() {
    assert_eq!(compute_diff("hello", "hello world"), " world");
    assert_eq!(compute_diff("", "hello"), "hello");
    assert_eq!(compute_diff("hello", "hello"), "");
    assert_eq!(compute_diff("test", "hello"), "hello");
}

// NOTE: test_stream_result_variants removed - StreamResult enum replaced by StreamingParseResult

#[test]
fn test_partial_tool_call() {
    let mut partial = PartialToolCall {
        name: None,
        arguments_buffer: String::new(),
        start_position: 0,
        name_sent: false,
        streamed_args: String::new(),
    };

    // Set name
    partial.name = Some("test_function".to_string());
    assert_eq!(partial.name.as_ref().unwrap(), "test_function");

    // Append arguments
    partial.arguments_buffer.push_str(r#"{"key": "value"}"#);
    assert_eq!(partial.arguments_buffer, r#"{"key": "value"}"#);

    // Update streaming state
    partial.name_sent = true;
    partial.streamed_args = r#"{"key": "#.to_string();
    assert!(partial.name_sent);
    assert_eq!(partial.streamed_args, r#"{"key": "#);
}

#[tokio::test]
async fn test_json_parser_complete_single() {
    let parser = JsonParser::new();

    let input = r#"{"name": "get_weather", "arguments": {"location": "San Francisco", "units": "celsius"}}"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
    assert!(tools[0].function.arguments.contains("San Francisco"));
    assert!(tools[0].function.arguments.contains("celsius"));
}

#[tokio::test]
async fn test_json_parser_complete_array() {
    let parser = JsonParser::new();

    let input = r#"[
        {"name": "get_weather", "arguments": {"location": "SF"}},
        {"name": "get_news", "arguments": {"query": "technology"}}
    ]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "get_news");
}

#[tokio::test]
async fn test_json_parser_with_parameters() {
    let parser = JsonParser::new();

    let input = r#"{"name": "calculate", "parameters": {"x": 10, "y": 20, "operation": "add"}}"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "calculate");
    assert!(tools[0].function.arguments.contains("10"));
    assert!(tools[0].function.arguments.contains("20"));
    assert!(tools[0].function.arguments.contains("add"));
}

// Tests removed - TokenConfig no longer supported in JsonParser

#[tokio::test]
async fn test_multiline_json_array() {
    let parser = JsonParser::new();

    let input = r#"[
    {
        "name": "function1",
        "arguments": {
            "param1": "value1",
            "param2": 42
        }
    },
    {
        "name": "function2",
        "parameters": {
            "data": [1, 2, 3],
            "flag": false
        }
    }
]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "function1");
    assert_eq!(tools[1].function.name, "function2");
    assert!(tools[0].function.arguments.contains("value1"));
    assert!(tools[1].function.arguments.contains("[1,2,3]"));
}

#[test]
fn test_json_parser_format_detection() {
    let parser = JsonParser::new();

    // Should detect valid tool call formats
    assert!(parser.has_tool_markers(r#"{"name": "test", "arguments": {}}"#));
    assert!(parser.has_tool_markers(r#"{"name": "test", "parameters": {"x": 1}}"#));
    assert!(parser.has_tool_markers(r#"[{"name": "test"}]"#));

    // Should not detect non-tool formats
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_factory_with_json_parser() {
    let factory = ParserFactory::new();

    // Should get JSON parser for OpenAI models
    let pooled_parser = factory.get_pooled("gpt-4-turbo");
    let parser = pooled_parser.lock().await;

    let input = r#"{"name": "test", "arguments": {"x": 1}}"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");
}

#[tokio::test]
async fn test_json_parser_invalid_input() {
    let parser = JsonParser::new();

    // Invalid JSON should return empty results
    assert_eq!(parser.parse_complete("not json").await.unwrap().1.len(), 0);
    assert_eq!(parser.parse_complete("{invalid}").await.unwrap().1.len(), 0);
    assert_eq!(parser.parse_complete("").await.unwrap().1.len(), 0);
}

#[tokio::test]
async fn test_json_parser_empty_arguments() {
    let parser = JsonParser::new();

    // Tool call with no arguments
    let input = r#"{"name": "get_time"}"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_time");
    assert_eq!(tools[0].function.arguments, "{}");
}

#[cfg(test)]
mod failure_cases {
    use super::*;

    #[tokio::test]
    async fn test_malformed_tool_missing_name() {
        let parser = JsonParser::new();

        // Missing name field
        let input = r#"{"arguments": {"x": 1}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 0, "Should return empty for tool without name");

        // Empty name
        let input = r#"{"name": "", "arguments": {"x": 1}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1, "Should accept empty name string");
        assert_eq!(tools[0].function.name, "");
    }

    #[tokio::test]
    async fn test_invalid_arguments_json() {
        let parser = JsonParser::new();

        // Arguments is a string instead of object
        let input = r#"{"name": "test", "arguments": "not an object"}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        // Should serialize the string as JSON
        assert!(tools[0].function.arguments.contains("not an object"));

        // Arguments is a number
        let input = r#"{"name": "test", "arguments": 42}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.arguments, "42");

        // Arguments is null
        let input = r#"{"name": "test", "arguments": null}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.arguments, "null");
    }

    // Test removed - wrapper token functionality moved to specific parsers

    #[tokio::test]
    async fn test_invalid_json_structures() {
        let parser = JsonParser::new();

        // Trailing comma
        let input = r#"{"name": "test", "arguments": {"x": 1,}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 0, "Should reject JSON with trailing comma");

        // Missing quotes on keys
        let input = r#"{name: "test", arguments: {}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 0, "Should reject invalid JSON syntax");

        // Unclosed object
        let input = r#"{"name": "test", "arguments": {"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 0, "Should reject incomplete JSON");
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[tokio::test]
    async fn test_unicode_in_names_and_arguments() {
        let parser = JsonParser::new();

        // Unicode in function name
        let input = r#"{"name": "获取天气", "arguments": {"location": "北京"}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "获取天气");
        assert!(tools[0].function.arguments.contains("北京"));

        // Emoji in arguments
        let input = r#"{"name": "send_message", "arguments": {"text": "Hello 👋 World 🌍"}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("👋"));
        assert!(tools[0].function.arguments.contains("🌍"));
    }

    #[tokio::test]
    async fn test_escaped_characters() {
        let parser = JsonParser::new();

        // Escaped quotes in arguments
        let input = r#"{"name": "echo", "arguments": {"text": "He said \"hello\""}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains(r#"\"hello\""#));

        // Escaped backslashes
        let input = r#"{"name": "path", "arguments": {"dir": "C:\\Users\\test"}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("\\\\"));

        // Newlines and tabs
        let input = r#"{"name": "format", "arguments": {"text": "line1\nline2\ttabbed"}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("\\n"));
        assert!(tools[0].function.arguments.contains("\\t"));
    }

    #[tokio::test]
    async fn test_very_large_payloads() {
        let parser = JsonParser::new();

        // Large arguments object
        let mut large_args = r#"{"name": "process", "arguments": {"#.to_string();
        for i in 0..1000 {
            large_args.push_str(&format!(r#""field_{}": "value_{}","#, i, i));
        }
        large_args.push_str(r#""final": "value"}}"#);

        let (_normal_text, tools) = parser.parse_complete(&large_args).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "process");
        assert!(tools[0].function.arguments.contains("field_999"));

        // Large array of tool calls
        let mut large_array = "[".to_string();
        for i in 0..100 {
            if i > 0 {
                large_array.push(',');
            }
            large_array.push_str(&format!(r#"{{"name": "func_{}", "arguments": {{}}}}"#, i));
        }
        large_array.push(']');

        let (_normal_text, tools) = parser.parse_complete(&large_array).await.unwrap();
        assert_eq!(tools.len(), 100);
        assert_eq!(tools[99].function.name, "func_99");
    }

    #[tokio::test]
    async fn test_mixed_array_tools_and_non_tools() {
        let parser = JsonParser::new();

        // Array with both tool calls and non-tool objects
        let input = r#"[
            {"name": "tool1", "arguments": {}},
            {"not_a_tool": "just_data"},
            {"name": "tool2", "parameters": {"x": 1}},
            {"key": "value", "another": "field"}
        ]"#;

        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 2, "Should only parse valid tool calls");
        assert_eq!(tools[0].function.name, "tool1");
        assert_eq!(tools[1].function.name, "tool2");
    }

    #[tokio::test]
    async fn test_duplicate_keys_in_json() {
        let parser = JsonParser::new();

        // JSON with duplicate keys (last one wins in most parsers)
        let input = r#"{"name": "first", "name": "second", "arguments": {"x": 1, "x": 2}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0].function.name, "second",
            "Last duplicate key should win"
        );
        assert!(
            tools[0].function.arguments.contains("2"),
            "Last duplicate value should win"
        );
    }

    #[tokio::test]
    async fn test_null_values_in_arguments() {
        let parser = JsonParser::new();

        // Null values in arguments
        let input = r#"{"name": "test", "arguments": {"required": "value", "optional": null}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("null"));

        // Array with null
        let input = r#"{"name": "test", "arguments": {"items": [1, null, "three"]}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("null"));
    }

    #[tokio::test]
    async fn test_special_json_values() {
        let parser = JsonParser::new();

        // Boolean values
        let input = r#"{"name": "toggle", "arguments": {"enabled": true, "disabled": false}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("true"));
        assert!(tools[0].function.arguments.contains("false"));

        // Numbers (including float and negative)
        let input = r#"{"name": "calc", "arguments": {"int": 42, "float": 3.14, "negative": -17}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("42"));
        assert!(tools[0].function.arguments.contains("3.14"));
        assert!(tools[0].function.arguments.contains("-17"));

        // Empty arrays and objects
        let input = r#"{"name": "test", "arguments": {"empty_arr": [], "empty_obj": {}}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("[]"));
        assert!(tools[0].function.arguments.contains("{}"));
    }

    #[tokio::test]
    async fn test_function_field_alternative() {
        let parser = JsonParser::new();

        // Using "function" instead of "name"
        let input = r#"{"function": "test_func", "arguments": {"x": 1}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "test_func");

        // Both "name" and "function" present (name should take precedence)
        let input = r#"{"name": "primary", "function": "secondary", "arguments": {}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "primary");
    }

    #[tokio::test]
    async fn test_whitespace_handling() {
        let parser = JsonParser::new();

        // Extra whitespace everywhere
        let input = r#"  {
            "name"   :   "test"  ,
            "arguments"   :   {
                "key"   :   "value"
            }
        }  "#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "test");

        // Minified JSON (no whitespace)
        let input = r#"{"name":"compact","arguments":{"a":1,"b":2}}"#;
        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "compact");
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_deeply_nested_arguments() {
        let parser = JsonParser::new();

        // Deeply nested structure
        let input = r#"{
            "name": "nested",
            "arguments": {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "level5": {
                                    "value": "deep"
                                }
                            }
                        }
                    }
                }
            }
        }"#;

        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].function.arguments.contains("deep"));
    }

    #[tokio::test]
    async fn test_concurrent_parser_usage() {
        let parser = std::sync::Arc::new(JsonParser::new());

        let mut handles = vec![];

        for i in 0..10 {
            let parser_clone = parser.clone();
            let handle = tokio::spawn(async move {
                let input = format!(r#"{{"name": "func_{}", "arguments": {{}}}}"#, i);
                let (_normal_text, tools) = parser_clone.parse_complete(&input).await.unwrap();
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0].function.name, format!("func_{}", i));
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }
}
