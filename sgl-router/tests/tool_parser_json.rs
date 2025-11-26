//! JSON Parser Integration Tests
//!
//! Tests for the JSON parser which handles OpenAI, Claude, and generic JSON formats

use serde_json::json;
use sglang_router_rs::tool_parser::{JsonParser, ToolParser};

mod common;
use common::{create_test_tools, streaming_helpers::*};

#[tokio::test]
async fn test_simple_json_tool_call() {
    let parser = JsonParser::new();
    let input = r#"{"name": "get_weather", "arguments": {"location": "San Francisco"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "San Francisco");
}

#[tokio::test]
async fn test_json_array_of_tools() {
    let parser = JsonParser::new();
    let input = r#"Hello, here are the results: [
        {"name": "get_weather", "arguments": {"location": "SF"}},
        {"name": "search", "arguments": {"query": "news"}}
    ]"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "Hello, here are the results: ");
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "search");
}

#[tokio::test]
async fn test_json_with_parameters_key() {
    let parser = JsonParser::new();
    let input = r#"{"name": "calculate", "parameters": {"x": 10, "y": 20}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "calculate");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["x"], 10);
    assert_eq!(args["y"], 20);
}

#[tokio::test]
async fn test_json_extraction_from_text() {
    let parser = JsonParser::new();
    let input = r#"I'll help you with that. {"name": "search", "arguments": {"query": "rust"}} Let me search for that."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(
        normal_text,
        "I'll help you with that.  Let me search for that."
    );
    assert_eq!(tools[0].function.name, "search");
}

#[tokio::test]
async fn test_json_with_nested_objects() {
    let parser = JsonParser::new();
    let input = r#"{
        "name": "update_config",
        "arguments": {
            "settings": {
                "theme": "dark",
                "language": "en",
                "notifications": {
                    "email": true,
                    "push": false
                }
            }
        }
    }"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "update_config");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["settings"]["theme"], "dark");
    assert_eq!(args["settings"]["notifications"]["email"], true);
}

#[tokio::test]
async fn test_json_with_special_characters() {
    let parser = JsonParser::new();
    let input = r#"{"name": "echo", "arguments": {"text": "Line 1\nLine 2\tTabbed", "path": "C:\\Users\\test"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Line 1\nLine 2\tTabbed");
    assert_eq!(args["path"], "C:\\Users\\test");
}

#[tokio::test]
async fn test_json_with_unicode() {
    let parser = JsonParser::new();
    let input = r#"{"name": "translate", "arguments": {"text": "Hello 世界 🌍", "emoji": "😊"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Hello 世界 🌍");
    assert_eq!(args["emoji"], "😊");
}

#[tokio::test]
async fn test_json_empty_arguments() {
    let parser = JsonParser::new();
    let input = r#"{"name": "ping", "arguments": {}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "ping");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args, json!({}));
}

#[tokio::test]
async fn test_json_invalid_format() {
    let parser = JsonParser::new();

    // Missing closing brace
    let input = r#"{"name": "test", "arguments": {"key": "value""#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(
        normal_text,
        "{\"name\": \"test\", \"arguments\": {\"key\": \"value\""
    );

    // Not JSON at all
    let input = "This is just plain text";
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
}

#[tokio::test]
async fn test_json_format_detection() {
    let parser = JsonParser::new();

    assert!(parser.has_tool_markers(r#"{"name": "test", "arguments": {}}"#));
    assert!(parser.has_tool_markers(r#"[{"name": "test"}]"#));
    assert!(!parser.has_tool_markers("plain text"));
}

// Streaming tests for JSON array format
#[tokio::test]
async fn test_json_array_streaming_required_mode() {
    use sglang_router_rs::protocols::common::Tool;

    // Test that simulates the exact streaming pattern from required mode
    let mut parser = JsonParser::new();

    // Define test tools
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: sglang_router_rs::protocols::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({}),
            strict: None,
        },
    }];

    // Simulate the EXACT chunks from the debug log
    let chunks = vec![
        "[{",
        " \"",
        "name",
        "\":",
        " \"",
        "get",
        "_weather",
        "\",",
        " \"",
        "parameters",
        "\":",
        " {",
        " \"",
        "city",
        "\":",
        " \"",
        "Paris",
        "\"",
        " }",
        " }]",
    ];

    let mut all_results = Vec::new();
    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_results.extend(result.calls);
        all_normal_text.push_str(&result.normal_text);
    }

    // We should have gotten tool call chunks
    assert!(
        !all_results.is_empty(),
        "Should have emitted tool call chunks"
    );

    // Should not have emitted any normal text (including the closing ])
    assert_eq!(
        all_normal_text, "",
        "Should not emit normal text for JSON array format"
    );

    // Check that we got the function name
    let has_name = all_results
        .iter()
        .any(|item| item.name.as_ref().is_some_and(|n| n == "get_weather"));
    assert!(has_name, "Should have emitted function name");

    // Check that we got the parameters
    let has_params = all_results.iter().any(|item| !item.parameters.is_empty());
    assert!(has_params, "Should have emitted parameters");
}

#[tokio::test]
async fn test_json_array_multiple_tools_streaming() {
    use sglang_router_rs::protocols::common::Tool;

    // Test with multiple tools in array
    let mut parser = JsonParser::new();

    let tools = vec![
        Tool {
            tool_type: "function".to_string(),
            function: sglang_router_rs::protocols::common::Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: sglang_router_rs::protocols::common::Function {
                name: "get_news".to_string(),
                description: Some("Get news".to_string()),
                parameters: serde_json::json!({}),
                strict: None,
            },
        },
    ];

    // Split into smaller, more realistic chunks
    let chunks = vec![
        "[{",
        "\"name\":",
        "\"get_weather\"",
        ",\"parameters\":",
        "{\"city\":",
        "\"SF\"}",
        "}",
        ",",
        "{\"name\":",
        "\"get_news\"",
        ",\"parameters\":",
        "{\"topic\":",
        "\"tech\"}",
        "}]",
    ];

    let mut all_results = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_results.extend(result.calls);
    }

    // Should have gotten tool calls for both functions
    let has_weather = all_results
        .iter()
        .any(|item| item.name.as_ref().is_some_and(|n| n == "get_weather"));
    let has_news = all_results
        .iter()
        .any(|item| item.name.as_ref().is_some_and(|n| n == "get_news"));

    assert!(has_weather, "Should have get_weather tool call");
    assert!(has_news, "Should have get_news tool call");
}

#[tokio::test]
async fn test_json_array_closing_bracket_separate_chunk() {
    use sglang_router_rs::protocols::common::Tool;

    // Test case where the closing ] comes as a separate chunk
    let mut parser = JsonParser::new();

    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: sglang_router_rs::protocols::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: json!({}),
            strict: None,
        },
    }];

    // Closing ] as separate chunk, followed by normal text
    let chunks = vec![
        "[{",
        "\"",
        "name",
        "\":",
        "\"",
        "get",
        "_weather",
        "\",",
        "\"",
        "parameters",
        "\":",
        "{",
        "\"",
        "city",
        "\":",
        "\"",
        "Paris",
        "\"",
        "}",
        "}",
        "]",
        " Here's",
        " the",
        " weather",
        " info",
    ];

    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);
    }

    // Should emit only the third chunk as normal text, NOT the ]
    assert_eq!(
        all_normal_text, " Here's the weather info",
        "Should emit only normal text without ], got: '{}'",
        all_normal_text
    );
}

#[tokio::test]
async fn test_json_single_object_with_trailing_text() {
    use sglang_router_rs::protocols::common::Tool;

    // Test single object format (no array) with trailing text
    let mut parser = JsonParser::new();

    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: sglang_router_rs::protocols::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({}),
            strict: None,
        },
    }];

    let chunks = vec![
        "{",
        "\"",
        "name",
        "\":",
        "\"",
        "get_weather",
        "\",",
        "\"",
        "parameters",
        "\":",
        "{",
        "\"city",
        "\":",
        "\"Paris",
        "\"}",
        "}",
        " Here's",
        " the",
        " weather",
    ];

    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);
    }

    // Should emit the trailing text as normal_text (no ] to strip for single object)
    assert_eq!(
        all_normal_text, " Here's the weather",
        "Should emit normal text for single object format, got: '{}'",
        all_normal_text
    );
}

#[tokio::test]
async fn test_json_single_object_with_bracket_in_text() {
    use sglang_router_rs::protocols::common::Tool;

    // Test that ] in normal text is NOT stripped for single object format
    let mut parser = JsonParser::new();

    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: sglang_router_rs::protocols::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({}),
            strict: None,
        },
    }];

    let chunks = vec![
        "{",
        "\"name",
        "\":",
        "\"get_weather",
        "\",",
        "\"parameters",
        "\":",
        "{",
        "\"city",
        "\":",
        "\"Paris",
        "\"}",
        "}",
        "]",
        " Here's",
        " the",
        " weather",
    ];

    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);
    }

    // For single object format, ] should NOT be stripped (it's part of normal text)
    assert_eq!(
        all_normal_text, "] Here's the weather",
        "Should preserve ] in normal text for single object format, got: '{}'",
        all_normal_text
    );
}

#[tokio::test]
async fn test_json_array_bracket_in_text_after_tools() {
    use sglang_router_rs::protocols::common::Tool;

    // Test that ] in normal text AFTER array tools is preserved
    let mut parser = JsonParser::new();

    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: sglang_router_rs::protocols::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({}),
            strict: None,
        },
    }];

    let chunks = vec![
        "[",
        "{",
        "\"name",
        "\":",
        "\"get_weather",
        "\",",
        "\"parameters",
        "\":",
        "{",
        "\"city",
        "\":",
        "\"Paris",
        "\"}",
        "}",
        "]",
        " Array",
        " notation:",
        " arr",
        "[",
        "0",
        "]",
    ];

    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);
    }

    // Should preserve ] in normal text after array tools complete
    assert_eq!(
        all_normal_text, " Array notation: arr[0]",
        "Should preserve ] in normal text after array tools, got: '{}'",
        all_normal_text
    );
}
// =============================================================================
// REALISTIC STREAMING TESTS
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
async fn test_json_unicode() {
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
