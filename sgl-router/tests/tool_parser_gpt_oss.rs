//! GPT-OSS Parser Integration Tests

use sglang_router_rs::tool_parser::{GptOssParser, ToolParser};

mod common;
use common::create_test_tools;

#[tokio::test]
async fn test_gpt_oss_complete_parsing() {
    let parser = GptOssParser::new();

    let input = r#"Let me search for that information.
<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query": "rust programming", "limit": 10}<|call|>
Here are the results..."#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust programming");
    assert_eq!(args["limit"], 10);
}

#[tokio::test]
async fn test_gpt_oss_multiple_tools() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "Paris"}<|call|>commentary
<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query": "Paris tourism"}<|call|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "search");
}

#[tokio::test]
async fn test_gpt_oss_with_namespace() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary to=api.users.create<|constrain|>json<|message|>{"name": "John", "email": "john@example.com"}<|call|>
<|channel|>commentary to=tools.calculator.add<|constrain|>json<|message|>{"x": 10, "y": 20}<|call|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "create"); // Should extract last part
    assert_eq!(tools[1].function.name, "add");
}

#[tokio::test]
async fn test_gpt_oss_with_assistant_prefix() {
    let parser = GptOssParser::new();

    let input = r#"<|start|>assistant<|channel|>commentary to=functions.test<|constrain|>json<|message|>{"key": "value"}<|call|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");
}

#[tokio::test]
async fn test_gpt_oss_empty_args() {
    let parser = GptOssParser::new();

    let input =
        r#"<|channel|>commentary to=functions.get_time<|constrain|>json<|message|>{}<|call|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_time");
    assert_eq!(tools[0].function.arguments, "{}");
}

#[tokio::test]
async fn test_gpt_oss_streaming() {
    let tools = create_test_tools();

    let mut parser = GptOssParser::new();

    // Simulate streaming chunks
    let chunks = vec![
        "<|channel|>commentary to=",
        "functions.calculate",
        "<|constrain|>json<|message|>",
        r#"{"x": 10"#,
        r#", "y": 20}"#,
        "<|call|>",
    ];

    let mut found_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        if !result.calls.is_empty() {
            if let Some(name) = &result.calls[0].name {
                assert_eq!(name, "calculate");
                found_complete = true;
            }
        }
    }

    assert!(found_complete);
}

#[test]
fn test_gpt_oss_format_detection() {
    let parser = GptOssParser::new();

    // Should detect GPT-OSS format
    assert!(parser.has_tool_markers("<|channel|>commentary to="));
    assert!(parser.has_tool_markers("<|channel|>commentary"));
    assert!(parser.has_tool_markers("text with <|channel|>commentary to= marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_gpt_oss_with_whitespace() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary to=functions.test  <|constrain|>json<|message|>{"key": "value"}<|call|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");
}

#[tokio::test]
async fn test_gpt_oss_complex_json() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary to=functions.process<|constrain|>json<|message|>{
    "nested": {
        "data": [1, 2, 3],
        "config": {
            "enabled": true
        }
    }
}<|call|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["nested"]["data"].is_array());
    assert_eq!(args["nested"]["config"]["enabled"], true);
}

#[tokio::test]
async fn test_commentary_without_function() {
    let parser = GptOssParser::new();

    // Python should extract commentary as normal text
    let input = r#"<|channel|>commentary<|message|>**Action plan**: 1. Do X 2. Do Y<|end|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0); // No tool calls
                                // TODO: Verify normal text = "**Action plan**: 1. Do X 2. Do Y"
}

#[tokio::test]
async fn test_final_channel() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary to=functions.test<|constrain|>json<|message|>{"x": 1}<|call|>
<|channel|>final<|message|>The result is calculated.<|return|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");
    // TODO: Verify normal text = "The result is calculated."
}

#[tokio::test]
async fn test_mixed_commentary_and_calls() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary<|message|>Let me think<|end|>
<|channel|>commentary to=functions.calc<|constrain|>json<|message|>{"x": 5}<|call|>
<|channel|>commentary<|message|>Processing...<|end|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "calc");
    // TODO: Verify normal text = "Let me think Processing..."
}
