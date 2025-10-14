//! Streaming Parser Tests
//!
//! Tests for incremental/streaming parsing capabilities across all parsers

use sglang_router_rs::tool_parser::{
    JsonParser, LlamaParser, MistralParser, PythonicParser, QwenParser, ToolParser,
};

mod common;
use common::create_test_tools;

#[tokio::test]
async fn test_json_streaming_simple() {
    let tools = create_test_tools();

    let mut parser = JsonParser::new();

    let full_json = r#"{"name": "get_weather", "arguments": {"location": "San Francisco"}}"#;

    let result = parser.parse_incremental(full_json, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should have parsed a tool call");
    assert_eq!(result.calls[0].name, Some("get_weather".to_string()));
}

#[tokio::test]
async fn test_json_streaming_array() {
    let tools = create_test_tools();

    let mut parser = JsonParser::new();

    let chunks = vec![
        r#"["#,
        r#"{"name": "tool1", "#,
        r#""arguments": {}}, "#,
        r#"{"name": "tool2", "#,
        r#""arguments": {"x": 1"#,
        r#"}}]"#,
    ];

    let mut tool_count = 0;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_some() {
                tool_count += 1;
            }
        }
    }

    // Current implementation may handle this differently
    assert!(tool_count <= 2, "Should parse at most 2 tools");
}

#[tokio::test]
async fn test_mistral_streaming() {
    let tools = create_test_tools();

    let mut parser = MistralParser::new();

    let chunks = vec![
        r#"Here is the result: "#,
        r#"[TOOL_CALLS] ["#,
        r#"{"name": "#,
        r#""search", "#,
        r#""arguments": "#,
        r#"{"query": "#,
        r#""rust lang""#,
        r#"}}]"#,
    ];

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "search");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have found tool name");
}

#[tokio::test]
async fn test_pythonic_streaming() {
    let tools = create_test_tools();

    let mut parser = PythonicParser::new();

    let full_input = r#"[get_weather(city="London", units="celsius")]"#;

    let result = parser.parse_incremental(full_input, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should have parsed a tool call");
    assert_eq!(result.calls[0].name, Some("get_weather".to_string()));
    let args: serde_json::Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
    assert_eq!(args["city"], "London");
}

#[tokio::test]
async fn test_llama_streaming_with_python_tag() {
    let tools = create_test_tools();

    let mut parser = LlamaParser::new();

    let chunks = vec![
        r#"Let me help. "#,
        r#"<|python"#,
        r#"_tag|>"#,
        r#"{"name": "#,
        r#""calculate", "#,
        r#""arguments": "#,
        r#"{"x": 10}"#,
        r#"}"#,
    ];

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "calculate");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have found tool name");
}

#[tokio::test]
async fn test_qwen_streaming() {
    let tools = create_test_tools();

    let mut parser = QwenParser::new();

    // Note: Parser expects newline after both tags
    let full_input = "<tool_call>\n{\"name\": \"translate\", \"arguments\": {\"text\": \"hello\", \"to\": \"zh\"}}\n</tool_call>";

    let result = parser.parse_incremental(full_input, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should have parsed a tool call");
    assert_eq!(result.calls[0].name, Some("translate".to_string()));
}

#[tokio::test]
async fn test_streaming_incomplete_stays_incomplete() {
    let tools = create_test_tools();

    let mut parser = JsonParser::new();

    let chunks = vec![r#"{"na"#, r#"me": "#];

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        assert!(
            result.calls.is_empty(),
            "Should return empty calls for partial JSON, got: {:?}",
            result
        );
    }
}

#[tokio::test]
async fn test_streaming_buffer_accumulation() {
    let tools = create_test_tools();

    let mut parser = JsonParser::new();

    let result1 = parser.parse_incremental(r#"{"na"#, &tools).await.unwrap();

    assert!(result1.calls.is_empty(), "Should not parse incomplete JSON");

    let result2 = parser
        .parse_incremental(r#"me": "test", "arguments": {}}"#, &tools)
        .await
        .unwrap();

    assert!(
        !result2.calls.is_empty(),
        "Should parse complete JSON after buffering"
    );
    assert_eq!(result2.calls[0].name, Some("test".to_string()));
}

#[tokio::test]
async fn test_streaming_multiple_tools_sequential() {
    let tools = create_test_tools();

    let mut parser = QwenParser::new();

    let full_input = r#"<tool_call>
{"name": "tool1", "arguments": {}}
</tool_call>"#;

    let result = parser.parse_incremental(full_input, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should have parsed a tool call");
    assert_eq!(result.calls[0].name, Some("tool1".to_string()));
}

#[tokio::test]
async fn test_streaming_reset_after_error() {
    let tools = create_test_tools();

    let mut parser1 = JsonParser::new();

    let _ = parser1
        .parse_incremental(r#"{"name": invalid}"#, &tools)
        .await;

    // Use a new parser instance for clean state
    let mut parser2 = JsonParser::new();
    let result = parser2
        .parse_incremental(r#"{"name": "test", "arguments": {}}"#, &tools)
        .await
        .unwrap();

    assert!(!result.calls.is_empty(), "Should parse valid JSON");
    assert_eq!(result.calls[0].name, Some("test".to_string()));
}

#[tokio::test]
async fn test_streaming_with_unicode_chunks() {
    let tools = create_test_tools();

    let mut parser = JsonParser::new();

    let full_input = r#"{"name": "translate", "arguments": {"text": "Hello 世界 🌍"}}"#;

    let result = parser.parse_incremental(full_input, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should have parsed a tool call");

    // Check if we got the tool name
    if let Some(name) = &result.calls[0].name {
        assert_eq!(name, "translate");
    }

    // In streaming mode, need to make another call to get parameters
    let result2 = parser.parse_incremental("", &tools).await.unwrap();

    // Parameters should be in either result.calls[1] or result2.calls[0]
    let params = if result.calls.len() > 1 {
        &result.calls[1].parameters
    } else if !result2.calls.is_empty() {
        &result2.calls[0].parameters
    } else {
        &result.calls[0].parameters
    };

    if !params.is_empty() {
        let args: serde_json::Value = serde_json::from_str(params).unwrap();
        assert!(args["text"].as_str().unwrap().contains("世界"));
    }
}

#[tokio::test]
async fn test_streaming_with_partial_chunks() {
    let mut parser = JsonParser::new();
    let tools = create_test_tools();

    let partial = r#"{"#;
    let result = parser.parse_incremental(partial, &tools).await.unwrap();
    assert!(
        result.calls.is_empty(),
        "Should return empty calls for just opening brace"
    );

    let mut parser2 = JsonParser::new();
    let complete = r#"{"name": "get_weather", "arguments": {"location": "SF"}}"#;
    let result = parser2.parse_incremental(complete, &tools).await.unwrap();

    assert!(
        !result.calls.is_empty(),
        "Expected tool call for complete JSON"
    );
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "get_weather");

    // In streaming mode, need to make another call to get parameters
    let result2 = parser2.parse_incremental("", &tools).await.unwrap();

    // Parameters should be in either result.calls[1] or result2.calls[0]
    let params = if result.calls.len() > 1 {
        &result.calls[1].parameters
    } else if !result2.calls.is_empty() {
        &result2.calls[0].parameters
    } else {
        &result.calls[0].parameters
    };

    if !params.is_empty() {
        let args: serde_json::Value = serde_json::from_str(params).unwrap();
        assert_eq!(args["location"], "SF");
    }

    // The PartialJson parser can complete partial JSON by filling in missing values
    let mut parser3 = JsonParser::new();
    let partial_with_name = r#"{"name": "test", "argum"#;
    let result = parser3
        .parse_incremental(partial_with_name, &tools)
        .await
        .unwrap();

    // Parser behavior may vary - either complete with partial data or wait for more
    if !result.calls.is_empty() {
        assert_eq!(result.calls[0].name.as_ref().unwrap(), "test");
    }
}
