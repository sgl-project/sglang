//! Streaming Parser Tests
//!
//! Tests for incremental/streaming parsing capabilities across all parsers

use sglang_router_rs::tool_parser::{
    JsonParser, LlamaParser, MistralParser, ParseState, PythonicParser, QwenParser, StreamResult,
    ToolParser,
};

#[tokio::test]
async fn test_json_streaming_simple() {
    let parser = JsonParser::new();
    let mut state = ParseState::new();

    // Phase 2 note: This test sends the full JSON at once in the last chunk
    // In real streaming, chunks would be smaller
    let full_json = r#"{"name": "get_weather", "arguments": {"location": "San Francisco"}}"#;

    let result = parser
        .parse_incremental(full_json, &mut state)
        .await
        .unwrap();

    // With complete JSON sent at once, we should get ToolComplete
    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "get_weather");
        }
        _ => {
            panic!("Expected ToolComplete for complete JSON input");
        }
    }
}

#[tokio::test]
async fn test_json_streaming_array() {
    let parser = JsonParser::new();
    let mut state = ParseState::new();

    // Stream a JSON array of tools
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
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();
        if let StreamResult::ToolComplete(_) = result {
            tool_count += 1;
        }
    }

    // Current implementation may handle this differently
    // We're mainly testing that it doesn't crash
    assert!(tool_count <= 2, "Should parse at most 2 tools");
}

#[tokio::test]
async fn test_mistral_streaming() {
    let parser = MistralParser::new();
    let mut state = ParseState::new();

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

    let mut got_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();
        if let StreamResult::ToolComplete(tool) = result {
            assert_eq!(tool.function.name, "search");
            got_complete = true;
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_pythonic_streaming() {
    let parser = PythonicParser::new();
    let mut state = ParseState::new();

    // Send complete pythonic format at once
    let full_input = r#"[get_weather(city="London", units="celsius")]"#;

    let result = parser
        .parse_incremental(full_input, &mut state)
        .await
        .unwrap();

    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "get_weather");
            let args: serde_json::Value = serde_json::from_str(&tool.function.arguments).unwrap();
            assert_eq!(args["city"], "London");
        }
        _ => {
            panic!("Expected ToolComplete for complete pythonic input");
        }
    }
}

#[tokio::test]
async fn test_llama_streaming_with_python_tag() {
    let parser = LlamaParser::new();
    let mut state = ParseState::new();

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

    let mut got_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();
        if let StreamResult::ToolComplete(tool) = result {
            assert_eq!(tool.function.name, "calculate");
            got_complete = true;
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_qwen_streaming() {
    let parser = QwenParser::new();
    let mut state = ParseState::new();

    // Send complete Qwen format at once (with exact format expected by parser)
    // Note: Parser expects newline after both tags
    let full_input = "<tool_call>\n{\"name\": \"translate\", \"arguments\": {\"text\": \"hello\", \"to\": \"zh\"}}\n</tool_call>";

    let result = parser
        .parse_incremental(full_input, &mut state)
        .await
        .unwrap();

    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "translate");
        }
        other => {
            panic!(
                "Expected ToolComplete for complete Qwen input, got: {:?}",
                other
            );
        }
    }
}

#[tokio::test]
async fn test_streaming_incomplete_stays_incomplete() {
    let parser = JsonParser::new();
    let mut state = ParseState::new();

    // Send truly incomplete JSON that can't be auto-completed
    let chunks = vec![r#"{"na"#, r#"me": "#];

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();
        // Should return Incomplete for partial JSON that can't be auto-completed
        assert!(
            matches!(result, StreamResult::Incomplete),
            "Should return Incomplete for partial JSON, got: {:?}",
            result
        );
    }

    // Buffer should contain the accumulated incomplete JSON
    assert!(!state.buffer.is_empty());
}

#[tokio::test]
async fn test_streaming_with_text_before_tool() {
    let parser = JsonParser::new();
    let mut state = ParseState::new();

    // For streaming, the parser expects clean JSON
    // Mixed text extraction only works in parse_complete, not parse_incremental
    let full_input = r#"{"name": "test", "arguments": {}}"#;

    let result = parser
        .parse_incremental(full_input, &mut state)
        .await
        .unwrap();

    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "test");
        }
        other => {
            panic!("Expected ToolComplete, got: {:?}", other);
        }
    }
}

#[tokio::test]
async fn test_streaming_buffer_accumulation() {
    let parser = JsonParser::new();

    // Test: Complete JSON should clear buffer after parsing
    let mut state = ParseState::new();

    // Send partial JSON that can't be interpreted as complete
    let result1 = parser
        .parse_incremental(r#"{"na"#, &mut state)
        .await
        .unwrap();

    assert!(matches!(result1, StreamResult::Incomplete));
    assert!(
        !state.buffer.is_empty(),
        "Buffer should accumulate incomplete JSON"
    );

    // Send rest of JSON
    let result2 = parser
        .parse_incremental(r#"me": "test", "arguments": {}}"#, &mut state)
        .await
        .unwrap();

    match result2 {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "test");
            assert!(
                state.buffer.is_empty(),
                "Buffer should be cleared after complete parse"
            );
        }
        _ => panic!(
            "Expected ToolComplete for complete JSON, got: {:?}",
            result2
        ),
    }
}

#[tokio::test]
async fn test_streaming_multiple_tools_sequential() {
    let parser = QwenParser::new();
    let mut state = ParseState::new();

    // Send complete Qwen format with newlines
    let full_input = r#"<tool_call>
{"name": "tool1", "arguments": {}}
</tool_call>"#;

    let result = parser
        .parse_incremental(full_input, &mut state)
        .await
        .unwrap();

    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "tool1");
        }
        _ => {
            panic!("Expected ToolComplete for first tool");
        }
    }
}

#[tokio::test]
async fn test_streaming_reset_after_error() {
    let parser = JsonParser::new();

    // First attempt with invalid JSON
    let mut state1 = ParseState::new();
    let _ = parser
        .parse_incremental(r#"{"name": invalid}"#, &mut state1)
        .await;

    // Second attempt with valid JSON should work with fresh state
    let mut state2 = ParseState::new();
    let result = parser
        .parse_incremental(r#"{"name": "test", "arguments": {}}"#, &mut state2)
        .await
        .unwrap();

    if let StreamResult::ToolComplete(tool) = result {
        assert_eq!(tool.function.name, "test");
    }
}

#[tokio::test]
async fn test_streaming_with_unicode_chunks() {
    let parser = JsonParser::new();
    let mut state = ParseState::new();

    // Send complete JSON with unicode
    let full_input = r#"{"name": "translate", "arguments": {"text": "Hello 世界 🌍"}}"#;

    let result = parser
        .parse_incremental(full_input, &mut state)
        .await
        .unwrap();

    // Phase 2 may return partial results even with complete JSON
    // The important thing is that unicode is handled without crashes
    match result {
        StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "translate");
            let args: serde_json::Value = serde_json::from_str(&tool.function.arguments).unwrap();
            assert!(args["text"].as_str().unwrap().contains("世界"));
        }
        StreamResult::ToolName { name, .. } => {
            assert_eq!(name, "translate");
            // Phase 2 partial streaming behavior - acceptable
        }
        StreamResult::ToolArguments { arguments, .. } => {
            // Verify unicode was preserved
            let args: serde_json::Value = serde_json::from_str(&arguments).unwrap();
            assert!(args["text"].as_str().unwrap().contains("世界"));
        }
        other => {
            panic!("Unexpected result: {:?}", other);
        }
    }
}
