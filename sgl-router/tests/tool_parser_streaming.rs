//! Realistic Streaming Parser Tests
//!
//! Tests incremental parsing with realistic char-level chunks (2-5 chars)
//! that simulate how LLM tokens actually arrive.
//!
//! These tests are designed to catch bugs like `{"name": "` being parsed
//! as an empty tool name.

use sglang_router_rs::tool_parser::{JsonParser, LlamaParser, QwenParser, ToolParser};

mod common;
use common::{create_test_tools, streaming_helpers::*};

// =============================================================================
// THE BUG SCENARIO - Most Critical Test
// =============================================================================

#[tokio::test]
async fn test_json_bug_incomplete_tool_name_string() {
    let tools = create_test_tools();
    let mut parser = JsonParser::new();

    // This exact sequence triggered the bug:
    // Parser receives {"name": " and must NOT parse it as empty name
    let chunks = vec![
        r#"{"#,
        r#"""#,
        r#"name"#,
        r#"""#,
        r#":"#,
        r#" "#,
        r#"""#, // ← Critical moment: parser has {"name": "
        // At this point, partial_json should NOT allow incomplete strings
        // when current_tool_name_sent=false
        r#"search"#, // Use valid tool name from create_test_tools()
        r#"""#,
        r#", "#,
        r#"""#,
        r#"arguments"#,
        r#"""#,
        r#": {"#,
        r#"""#,
        r#"query"#,
        r#"""#,
        r#": "#,
        r#"""#,
        r#"rust programming"#,
        r#"""#,
        r#"}}"#,
    ];

    let mut got_tool_name = false;
    let mut saw_empty_name = false;

    for chunk in chunks.iter() {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = &call.name {
                if name.is_empty() {
                    saw_empty_name = true;
                }
                if name == "search" {
                    got_tool_name = true;
                }
            }
        }
    }

    assert!(
        !saw_empty_name,
        "Parser should NEVER return empty tool name"
    );
    assert!(got_tool_name, "Should have parsed tool name correctly");
}

// =============================================================================
// JSON PARSER REALISTIC STREAMING
// =============================================================================

#[tokio::test]
async fn test_json_realistic_chunks_simple_tool() {
    let tools = create_test_tools();
    let mut parser = JsonParser::new();

    let input = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
    let chunks = create_realistic_chunks(input);

    assert!(chunks.len() > 10, "Should have many small chunks");

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_json_strategic_chunks_with_quotes() {
    let tools = create_test_tools();
    let mut parser = JsonParser::new();

    let input = r#"{"name": "search", "arguments": {"query": "rust programming"}}"#;
    let chunks = create_strategic_chunks(input);

    // Strategic chunks break after quotes and colons
    assert!(chunks.iter().any(|c| c.ends_with('"')));

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_some() {
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_json_incremental_arguments_streaming() {
    let tools = create_test_tools();
    let mut parser = JsonParser::new();

    let input = r#"{"name": "search", "arguments": {"query": "test", "limit": 10}}"#;
    let chunks = create_realistic_chunks(input);

    let mut tool_name_sent = false;
    let mut got_arguments = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_some() {
                tool_name_sent = true;
            }
            if tool_name_sent && !call.parameters.is_empty() {
                got_arguments = true;
            }
        }
    }

    assert!(tool_name_sent, "Should have sent tool name");
    assert!(got_arguments, "Should have sent arguments");
}

// =============================================================================
// LLAMA PARSER REALISTIC STREAMING
// =============================================================================

#[tokio::test]
async fn test_llama_realistic_chunks_with_python_tag() {
    let tools = create_test_tools();
    let mut parser = LlamaParser::new();

    let input = r#"<|python_tag|>{"name": "calculate", "parameters": {"x": 10, "y": 20}}"#;
    let chunks = create_realistic_chunks(input);

    assert!(chunks.len() > 15, "Should have many small chunks");

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "calculate");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_llama_python_tag_arrives_in_parts() {
    let tools = create_test_tools();
    let mut parser = LlamaParser::new();

    // Python tag itself arrives in small chunks
    let chunks = vec![
        "<|p", "yth", "on_", "tag", "|>{", r#"""#, "na", r#"me""#, ": ", r#"""#, "sea", "rch",
        r#"""#, ", ", r#"""#, "par", "ame", "ter", "s", r#"""#, ": {", r#"""#, "q", r#"""#, ": ",
        r#"""#, "tes", "t", r#"""#, "}}",
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

    assert!(got_tool_name, "Should have parsed tool name");
}

// =============================================================================
// QWEN PARSER REALISTIC STREAMING
// =============================================================================

#[tokio::test]
async fn test_qwen_realistic_chunks_with_xml_tags() {
    let tools = create_test_tools();
    let mut parser = QwenParser::new();

    let input = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>";
    let chunks = create_realistic_chunks(input);

    assert!(chunks.len() > 20, "Should have many small chunks");

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_qwen_xml_tag_arrives_in_parts() {
    let tools = create_test_tools();
    let mut parser = QwenParser::new();

    let chunks = vec![
        "<to", "ol_", "cal", "l>\n", "{", r#"""#, "na", "me", r#"""#, ": ", r#"""#, "tra", "nsl",
        "ate", r#"""#, ", ", r#"""#, "arg", "ume", "nts", r#"""#, ": {", r#"""#, "tex", "t",
        r#"""#, ": ", r#"""#, "hel", "lo", r#"""#, "}}\n", "</t", "ool", "_ca", "ll>",
    ];

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "translate");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

// =============================================================================
// EDGE CASES WITH REALISTIC CHUNKS
// =============================================================================

#[tokio::test]
async fn test_json_very_long_url_in_arguments() {
    let tools = create_test_tools();
    let mut parser = JsonParser::new();

    // Simulate long URL arriving in many chunks
    let long_url = "https://example.com/very/long/path/".to_string() + &"segment/".repeat(50);
    let input = format!(
        r#"{{"name": "search", "arguments": {{"query": "{}"}}}}"#,
        long_url
    );
    let chunks = create_realistic_chunks(&input);

    assert!(chunks.len() > 100, "Long URL should create many chunks");

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_some() {
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_json_unicode_arrives_byte_by_byte() {
    let tools = create_test_tools();
    let mut parser = JsonParser::new();

    let input = r#"{"name": "search", "arguments": {"query": "Hello 世界 🌍"}}"#;
    let chunks = create_realistic_chunks(input);

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_some() {
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed with unicode");
}
