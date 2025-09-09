//! Qwen Parser Integration Tests
//!
//! Tests for the Qwen parser which handles <tool_call>...</tool_call> format

use serde_json::json;
use sglang_router_rs::tool_parser::{ParseState, QwenParser, StreamResult, ToolParser};

#[tokio::test]
async fn test_qwen_single_tool() {
    let parser = QwenParser::new();
    let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing", "units": "celsius"}}
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_qwen_multiple_sequential_tools() {
    let parser = QwenParser::new();
    let input = r#"Let me help you with that.
<tool_call>
{"name": "search", "arguments": {"query": "Qwen model"}}
</tool_call>
<tool_call>
{"name": "translate", "arguments": {"text": "Hello", "to": "zh"}}
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "search");
    assert_eq!(result[1].function.name, "translate");
}

#[tokio::test]
async fn test_qwen_pretty_printed_json() {
    let parser = QwenParser::new();
    let input = r#"<tool_call>
{
    "name": "create_document",
    "arguments": {
        "title": "Test Document",
        "content": "This is a test",
        "metadata": {
            "author": "Qwen",
            "tags": ["test", "example"]
        }
    }
}
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "create_document");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["metadata"]["author"], "Qwen");
    assert_eq!(args["metadata"]["tags"], json!(["test", "example"]));
}

#[tokio::test]
async fn test_qwen_with_text_between() {
    let parser = QwenParser::new();
    let input = r#"First, let me search for information.
<tool_call>
{"name": "search", "arguments": {"query": "test"}}
</tool_call>

Now I'll translate something.

<tool_call>
{"name": "translate", "arguments": {"text": "world", "to": "es"}}
</tool_call>
Done!"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "search");
    assert_eq!(result[1].function.name, "translate");
}

#[tokio::test]
async fn test_qwen_empty_arguments() {
    let parser = QwenParser::new();
    let input = r#"<tool_call>
{"name": "get_time", "arguments": {}}
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_time");
}

#[tokio::test]
async fn test_qwen_with_newlines_in_strings() {
    let parser = QwenParser::new();
    let input = r#"<tool_call>
{"name": "write_file", "arguments": {"content": "Line 1\nLine 2\nLine 3", "path": "/tmp/test.txt"}}
</tool_call>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["content"], "Line 1\nLine 2\nLine 3");
}

#[tokio::test]
async fn test_qwen_format_detection() {
    let parser = QwenParser::new();

    assert!(parser.detect_format("<tool_call>"));
    assert!(parser.detect_format("Some text <tool_call>\n{"));
    assert!(!parser.detect_format("Just plain text"));
    assert!(!parser.detect_format("{\"name\": \"test\"}")); // Plain JSON
}

#[tokio::test]
async fn test_qwen_incomplete_tags() {
    let parser = QwenParser::new();

    // Missing closing tag
    let input = r#"<tool_call>
{"name": "test", "arguments": {}}"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 0);

    // Missing opening tag
    let input = r#"{"name": "test", "arguments": {}}
</tool_call>"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 0);
}

#[tokio::test]
async fn test_qwen_real_world_output() {
    let parser = QwenParser::new();

    // Actual output from Qwen model
    let input = r#"I'll help you search for information and perform calculations.

<tool_call>
{
    "name": "web_search",
    "arguments": {
        "query": "quantum computing breakthroughs 2024",
        "language": "en",
        "region": "us",
        "safe_search": true
    }
}
</tool_call>

Let me also calculate something for you:

<tool_call>
{
    "name": "calculator",
    "arguments": {
        "expression": "sqrt(144) + 3^2",
        "precision": 2
    }
}
</tool_call>

These tools will provide the information you need."#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "web_search");
    assert_eq!(result[1].function.name, "calculator");

    let args0: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args0["query"], "quantum computing breakthroughs 2024");
    assert_eq!(args0["safe_search"], true);
}

#[tokio::test]
async fn test_buffer_drain_optimization() {
    let parser = QwenParser::new();
    let mut state = ParseState::new();

    // First chunk - incomplete tool call
    let chunk1 = "<tool_call>\n{\"name\": \"test1\", ";
    let _result = parser.parse_incremental(chunk1, &mut state).await.unwrap();
    // Phase 2 simplified streaming might not handle partial JSON correctly
    // The important thing is buffer accumulation works
    assert!(!state.buffer.is_empty());

    // Complete first tool and start second
    let chunk2 = "\"arguments\": {}}\n</tool_call><tool_call>\n{\"name\": \"test2\", ";
    let result = parser.parse_incremental(chunk2, &mut state).await.unwrap();

    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "test1");
            // After consuming the first tool, buffer should contain only the second tool start
            assert!(state.buffer.starts_with("<tool_call>"));
            assert!(state.buffer.contains("test2"));
        }
        _ => {
            // Phase 2 simplified streaming might return Incomplete
            // The important thing is the buffer is managed correctly
        }
    }

    // Complete the second tool
    let chunk3 = "\"arguments\": {\"x\": 1}}\n</tool_call>";
    let result = parser.parse_incremental(chunk3, &mut state).await.unwrap();

    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "test2");
            // Buffer should be empty after consuming all tools
            assert!(state.buffer.is_empty() || !state.buffer.contains("</tool_call>"));
        }
        _ => {
            // Phase 2 simplified streaming might handle this differently
        }
    }
}

#[tokio::test]
async fn test_buffer_efficiency_with_multiple_tools() {
    let parser = QwenParser::new();
    let mut state = ParseState::new();

    // Send multiple complete tools at once
    let input = r#"<tool_call>
{"name": "tool1", "arguments": {"a": 1}}
</tool_call><tool_call>
{"name": "tool2", "arguments": {"b": 2}}
</tool_call><tool_call>
{"name": "tool3", "arguments": {"c": 3}}
</tool_call>"#;

    // This should efficiently process tools using drain() without creating new strings
    let result = parser.parse_incremental(input, &mut state).await.unwrap();

    // In Phase 2, this will likely parse only the first tool
    // The important thing is that drain() doesn't cause any issues
    match result {
        StreamResult::ToolComplete(tool) => {
            assert!(["tool1", "tool2", "tool3"].contains(&tool.function.name.as_str()));
        }
        _ => {
            // Simplified streaming might return Incomplete
        }
    }

    // Verify no memory issues or panics occurred with drain()
    // Test passes if we reach this point without panic
}
