//! DeepSeek V3 Parser Integration Tests

use sgl_model_gateway::tool_parser::{DeepSeekParser, ToolParser};

mod common;
use common::create_test_tools;

#[tokio::test]
async fn test_deepseek_complete_parsing() {
    let parser = DeepSeekParser::new();

    let input = r#"Let me help you with that.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo", "units": "celsius"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
The weather in Tokyo is..."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me help you with that.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_deepseek_multiple_tools() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{"query": "rust programming"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>translate
```json
{"text": "Hello World", "to": "ja"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_deepseek_streaming() {
    let tools = create_test_tools();

    let mut parser = DeepSeekParser::new();

    // Simulate streaming chunks
    let chunks = vec![
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "function<｜tool▁sep｜>get_weather\n",
        "```json\n",
        r#"{"location": "#,
        r#""Beijing", "#,
        r#""units": "metric"}"#,
        "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ];

    let mut found_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
}

#[tokio::test]
async fn test_deepseek_nested_json() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>process
```json
{
    "data": {
        "nested": {
            "deep": [1, 2, 3]
        }
    }
}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"]["deep"].is_array());
}

#[test]
fn test_deepseek_format_detection() {
    let parser = DeepSeekParser::new();

    // Should detect DeepSeek format
    assert!(parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(parser.has_tool_markers("text with <｜tool▁calls▁begin｜> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_deepseek_malformed_json_handling() {
    let parser = DeepSeekParser::new();

    // Malformed JSON should be skipped
    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>broken
```json
{invalid json}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>valid
```json
{"key": "value"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    // Only the valid tool call should be parsed
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "valid");
}

#[tokio::test]
async fn test_multiple_tool_calls() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Paris"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "get_weather");
}
