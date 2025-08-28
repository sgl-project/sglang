//! GPT-OSS Parser Integration Tests

use sglang_router_rs::tool_parser::{GptOssParser, ParseState, StreamResult, ToolParser};

#[tokio::test]
async fn test_gpt_oss_complete_parsing() {
    let parser = GptOssParser::new();

    // Test single tool call
    let input = r#"Let me search for that information.
<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query": "rust programming", "limit": 10}<|call|>
Here are the results..."#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "search");

    // Verify arguments
    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust programming");
    assert_eq!(args["limit"], 10);
}

#[tokio::test]
async fn test_gpt_oss_multiple_tools() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "Paris"}<|call|>commentary
<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query": "Paris tourism"}<|call|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "get_weather");
    assert_eq!(result[1].function.name, "search");
}

#[tokio::test]
async fn test_gpt_oss_with_namespace() {
    let parser = GptOssParser::new();

    // Test with different namespace patterns
    let input = r#"<|channel|>commentary to=api.users.create<|constrain|>json<|message|>{"name": "John", "email": "john@example.com"}<|call|>
<|channel|>commentary to=tools.calculator.add<|constrain|>json<|message|>{"x": 10, "y": 20}<|call|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "create"); // Should extract last part
    assert_eq!(result[1].function.name, "add");
}

#[tokio::test]
async fn test_gpt_oss_with_assistant_prefix() {
    let parser = GptOssParser::new();

    // Test with <|start|>assistant prefix
    let input = r#"<|start|>assistant<|channel|>commentary to=functions.test<|constrain|>json<|message|>{"key": "value"}<|call|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");
}

#[tokio::test]
async fn test_gpt_oss_empty_args() {
    let parser = GptOssParser::new();

    // Test with empty arguments
    let input =
        r#"<|channel|>commentary to=functions.get_time<|constrain|>json<|message|>{}<|call|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_time");
    assert_eq!(result[0].function.arguments, "{}");
}

#[tokio::test]
async fn test_gpt_oss_streaming() {
    let parser = GptOssParser::new();
    let mut state = ParseState::new();

    // Simulate streaming chunks
    let chunks = vec![
        "<|channel|>commentary to=",
        "functions.calculate",
        "<|constrain|>json<|message|>",
        r#"{"x": 10"#,
        r#", "y": 20}"#,
        "<|call|>",
    ];

    let mut found_name = false;
    let mut found_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();

        match result {
            StreamResult::ToolName { name, .. } => {
                assert_eq!(name, "calculate");
                found_name = true;
            }
            StreamResult::ToolComplete(tool) => {
                assert_eq!(tool.function.name, "calculate");
                found_complete = true;
            }
            _ => {}
        }
    }

    assert!(found_name || found_complete);
}

#[test]
fn test_gpt_oss_format_detection() {
    let parser = GptOssParser::new();

    // Should detect GPT-OSS format
    assert!(parser.detect_format("<|channel|>commentary to="));
    assert!(parser.detect_format("<|channel|>commentary"));
    assert!(parser.detect_format("text with <|channel|>commentary to= marker"));

    // Should not detect other formats
    assert!(!parser.detect_format("[TOOL_CALLS]"));
    assert!(!parser.detect_format("<tool_call>"));
    assert!(!parser.detect_format("plain text"));
}

#[tokio::test]
async fn test_gpt_oss_with_whitespace() {
    let parser = GptOssParser::new();

    // Test with whitespace after function name
    let input = r#"<|channel|>commentary to=functions.test  <|constrain|>json<|message|>{"key": "value"}<|call|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");
}

#[tokio::test]
async fn test_gpt_oss_complex_json() {
    let parser = GptOssParser::new();

    // Test with complex nested JSON
    let input = r#"<|channel|>commentary to=functions.process<|constrain|>json<|message|>{
    "nested": {
        "data": [1, 2, 3],
        "config": {
            "enabled": true
        }
    }
}<|call|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert!(args["nested"]["data"].is_array());
    assert_eq!(args["nested"]["config"]["enabled"], true);
}

#[tokio::test]
async fn test_commentary_without_function() {
    let parser = GptOssParser::new();

    // Python should extract commentary as normal text
    let input = r#"<|channel|>commentary<|message|>**Action plan**: 1. Do X 2. Do Y<|end|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 0); // No tool calls
                                 // TODO: Verify normal text = "**Action plan**: 1. Do X 2. Do Y"
}

#[tokio::test]
async fn test_final_channel() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary to=functions.test<|constrain|>json<|message|>{"x": 1}<|call|>
<|channel|>final<|message|>The result is calculated.<|return|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");
    // TODO: Verify normal text = "The result is calculated."
}

#[tokio::test]
async fn test_mixed_commentary_and_calls() {
    let parser = GptOssParser::new();

    let input = r#"<|channel|>commentary<|message|>Let me think<|end|>
<|channel|>commentary to=functions.calc<|constrain|>json<|message|>{"x": 5}<|call|>
<|channel|>commentary<|message|>Processing...<|end|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "calc");
    // TODO: Verify normal text = "Let me think Processing..."
}
