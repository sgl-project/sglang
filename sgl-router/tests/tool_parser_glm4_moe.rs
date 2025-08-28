//! GLM-4 MoE Parser Integration Tests

use sglang_router_rs::tool_parser::{Glm4MoeParser, ParseState, StreamResult, ToolParser};

#[tokio::test]
async fn test_glm4_complete_parsing() {
    let parser = Glm4MoeParser::new();

    // Test single tool call
    let input = r#"Let me search for that.
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2024-12-25</arg_value>
</tool_call>
The weather will be..."#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");

    // Verify arguments
    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["date"], "2024-12-25");
}

#[tokio::test]
async fn test_glm4_multiple_tools() {
    let parser = Glm4MoeParser::new();

    let input = r#"<tool_call>search
<arg_key>query</arg_key>
<arg_value>rust tutorials</arg_value>
</tool_call>
<tool_call>translate
<arg_key>text</arg_key>
<arg_value>Hello World</arg_value>
<arg_key>target_lang</arg_key>
<arg_value>zh</arg_value>
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "search");
    assert_eq!(result[1].function.name, "translate");
}

#[tokio::test]
async fn test_glm4_type_conversion() {
    let parser = Glm4MoeParser::new();

    // Test various value types
    let input = r#"<tool_call>process
<arg_key>count</arg_key>
<arg_value>42</arg_value>
<arg_key>rate</arg_key>
<arg_value>1.5</arg_value>
<arg_key>enabled</arg_key>
<arg_value>true</arg_value>
<arg_key>data</arg_key>
<arg_value>null</arg_value>
<arg_key>text</arg_key>
<arg_value>string value</arg_value>
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["count"], 42);
    assert_eq!(args["rate"], 1.5);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["data"], serde_json::Value::Null);
    assert_eq!(args["text"], "string value");
}

#[tokio::test]
async fn test_glm4_streaming() {
    let parser = Glm4MoeParser::new();
    let mut state = ParseState::new();

    // Simulate streaming chunks
    let chunks = vec![
        "<tool_call>",
        "get_weather\n",
        "<arg_key>city</arg_key>\n",
        "<arg_value>Shanghai</arg_value>\n",
        "<arg_key>units</arg_key>\n",
        "<arg_value>celsius</arg_value>\n",
        "</tool_call>",
    ];

    let mut found_name = false;
    let mut found_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();

        match result {
            StreamResult::ToolName { name, .. } => {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            StreamResult::ToolComplete(tool) => {
                assert_eq!(tool.function.name, "get_weather");
                found_complete = true;
            }
            _ => {}
        }
    }

    assert!(found_name || found_complete);
}

#[test]
fn test_glm4_format_detection() {
    let parser = Glm4MoeParser::new();

    // Should detect GLM-4 format
    assert!(parser.detect_format("<tool_call>"));
    assert!(parser.detect_format("text with <tool_call> marker"));

    // Should not detect other formats
    assert!(!parser.detect_format("[TOOL_CALLS]"));
    assert!(!parser.detect_format("<｜tool▁calls▁begin｜>"));
    assert!(!parser.detect_format("plain text"));
}

#[tokio::test]
async fn test_glm4_python_literal_values() {
    let parser = Glm4MoeParser::new();

    // Test Python-style boolean values
    let input = r#"<tool_call>config
<arg_key>debug</arg_key>
<arg_value>True</arg_value>
<arg_key>verbose</arg_key>
<arg_value>False</arg_value>
<arg_key>optional</arg_key>
<arg_value>None</arg_value>
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["debug"], true);
    assert_eq!(args["verbose"], false);
    assert_eq!(args["optional"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_python_literals() {
    let parser = Glm4MoeParser::new();

    let input = r#"<tool_call>test_func
<arg_key>bool_true</arg_key>
<arg_value>True</arg_value>
<arg_key>bool_false</arg_key>
<arg_value>False</arg_value>
<arg_key>none_val</arg_key>
<arg_value>None</arg_value>
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test_func");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["bool_true"], true);
    assert_eq!(args["bool_false"], false);
    assert_eq!(args["none_val"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_nested_values() {
    let parser = Glm4MoeParser::new();

    let input = r#"<tool_call>process
<arg_key>data</arg_key>
<arg_value>{"nested": {"key": "value"}}</arg_value>
<arg_key>list</arg_key>
<arg_value>[1, 2, 3]</arg_value>
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert!(args["data"].is_object());
    assert!(args["list"].is_array());
}
