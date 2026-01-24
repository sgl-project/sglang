//! Qwen Coder Parser Integration Tests
//!
//! Tests for the Qwen Coder parser which handles XML format:
//! <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>

use serde_json::json;
use smg::tool_parser::{parsers::QwenCoderParser, traits::ToolParser};

use crate::common::{create_test_tools, streaming_helpers::*};

#[tokio::test]
async fn test_qwen_coder_single_tool() {
    let parser = QwenCoderParser::new();
    let input = r#"<tool_call>
<function=get_weather>
<parameter=city>Beijing</parameter>
<parameter=units>celsius</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_qwen_coder_multiple_sequential_tools() {
    let parser = QwenCoderParser::new();
    let input = r#"Let me help you with that.
<tool_call>
<function=search>
<parameter=query>Qwen model</parameter>
</function>
</tool_call>
<tool_call>
<function=translate>
<parameter=text>Hello</parameter>
<parameter=to>zh</parameter>
</function>
</tool_call>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "Let me help you with that.\n");
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_qwen_coder_nested_json_in_parameters() {
    let parser = QwenCoderParser::new();
    let input = r#"<tool_call>
<function=process_data>
<parameter=config>{"nested": {"value": [1, 2, 3]}}</parameter>
<parameter=enabled>true</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process_data");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // JSON values should be parsed
    assert_eq!(args["config"]["nested"]["value"], json!([1, 2, 3]));
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_qwen_coder_string_parameters() {
    let parser = QwenCoderParser::new();
    let input = r#"<tool_call>
<function=process>
<parameter=text>Hello World</parameter>
<parameter=number>42</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Hello World");
    // JSON numbers should be parsed as numbers (consistent with Python's json.loads)
    assert_eq!(args["number"], 42);
}

#[tokio::test]
async fn test_qwen_coder_empty_arguments() {
    let parser = QwenCoderParser::new();
    let input = r#"<tool_call>
<function=get_time>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_time");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args, json!({}));
}

#[tokio::test]
async fn test_qwen_coder_multiline_parameter_values() {
    let parser = QwenCoderParser::new();
    let input = r#"<tool_call>
<function=write_file>
<parameter=content>Line 1
Line 2
Line 3</parameter>
<parameter=path>/tmp/test.txt</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["content"], "Line 1\nLine 2\nLine 3");
    assert_eq!(args["path"], "/tmp/test.txt");
}

#[tokio::test]
async fn test_qwen_coder_format_detection() {
    let parser = QwenCoderParser::new();

    assert!(parser.has_tool_markers("<tool_call>"));
    assert!(parser.has_tool_markers("Some text <tool_call>"));
    assert!(!parser.has_tool_markers("Just plain text"));
    assert!(!parser.has_tool_markers("<function=test>")); // Without tool_call tags
}

#[tokio::test]
async fn test_qwen_coder_incomplete_tags() {
    let parser = QwenCoderParser::new();

    // Missing closing tag
    let input = r#"<tool_call>
<function=get_weather>
<parameter=city>Beijing</parameter>"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);

    // Missing opening tag
    let input = r#"<parameter=city>Beijing</parameter>
</function>
</tool_call>"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
}

#[tokio::test]
async fn test_qwen_coder_streaming_basic() {
    let mut parser = QwenCoderParser::new();
    let tools = create_test_tools();

    // Simulate streaming chunks
    let chunks = vec![
        "<tool_call>",
        r#"<function=get_weather>"#,
        r#"<parameter=city>Shanghai</parameter>"#,
        r#"<parameter=units>celsius</parameter>"#,
        "</function>",
        "</tool_call>",
    ];

    let mut found_name = false;
    let mut found_params = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if !call.parameters.is_empty() {
                found_params = true;
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(found_params, "Should have streamed parameters");
}

#[tokio::test]
async fn test_qwen_coder_streaming_incremental_json() {
    let mut parser = QwenCoderParser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<tool_call>",
        r#"<function=get_weather>"#,
        r#"<parameter=city>Paris</parameter>"#,
        r#"<parameter=units>metric</parameter>"#,
        "</function></tool_call>",
    ];

    let mut json_fragments = Vec::new();
    let mut found_function = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(_name) = call.name {
                found_function = true;
            }
            if !call.parameters.is_empty() {
                json_fragments.push(call.parameters.clone());
            }
        }
    }

    assert!(found_function);

    // Verify JSON was built incrementally
    assert!(!json_fragments.is_empty());

    // First fragment should start with opening brace
    if let Some(first) = json_fragments.first() {
        assert!(
            first.starts_with('{'),
            "First JSON fragment should start with '{{': {}",
            first
        );
    }
}

#[tokio::test]
async fn test_qwen_coder_streaming_partial_tags() {
    let mut parser = QwenCoderParser::new();
    let tools = create_test_tools();

    // Chunks split mid-tag
    let chunks = vec![
        "<tool_c",
        "all><function=",
        r#"get_weather><param"#,
        r#"eter=city>Bei"#,
        "jing</parameter></func",
        "tion></tool_call>",
    ];

    let mut found_name = false;
    let mut buffer = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        buffer.push_str(&result.normal_text);

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
        }
    }

    assert!(
        found_name,
        "Should have parsed function name from partial chunks"
    );
}

#[tokio::test]
async fn test_qwen_coder_multiple_tools_boundary() {
    let mut parser = QwenCoderParser::new();
    let tools = create_test_tools();

    // Tool boundary at chunk boundary
    let chunks = vec![
        r#"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>"#,
        r#"<tool_call><function=search><parameter=query>weather forecast</parameter></function></tool_call>"#,
    ];

    let mut tool_names = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
        }
    }

    assert_eq!(tool_names.len(), 2);
    assert_eq!(tool_names[0], "get_weather");
    assert_eq!(tool_names[1], "search");
}

#[tokio::test]
async fn test_qwen_coder_invalid_function_name() {
    let mut parser = QwenCoderParser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<tool_call>",
        r#"<function=invalid_function>"#,
        r#"<parameter=param>value</parameter>"#,
        "</function></tool_call>",
    ];

    let mut found_invalid = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        // Invalid function should be skipped
        for call in result.calls {
            if let Some(name) = call.name {
                if name == "invalid_function" {
                    found_invalid = true;
                }
            }
        }
    }

    assert!(!found_invalid, "Invalid function should not be parsed");
}

#[tokio::test]
async fn test_qwen_coder_type_conversion() {
    let parser = QwenCoderParser::new();

    let input = r#"<tool_call>
<function=process>
<parameter=count>42</parameter>
<parameter=rate>1.5</parameter>
<parameter=enabled>true</parameter>
<parameter=data>null</parameter>
<parameter=text>string value</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // JSON values should be parsed
    assert_eq!(args["count"], 42);
    assert_eq!(args["rate"], 1.5);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["data"], serde_json::Value::Null);
    assert_eq!(args["text"], "string value");
}

#[tokio::test]
async fn test_qwen_coder_special_characters_in_values() {
    let parser = QwenCoderParser::new();

    let input = r#"<tool_call>
<function=process>
<parameter=text>Special chars: @#$%^&*()</parameter>
<parameter=emoji>🦀 Rust 🚀</parameter>
<parameter=quotes>"double" and 'single' quotes</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Special chars: @#$%^&*()");
    assert_eq!(args["emoji"], "🦀 Rust 🚀");
    assert_eq!(args["quotes"], "\"double\" and 'single' quotes");
}

#[tokio::test]
async fn test_qwen_coder_whitespace_handling() {
    let parser = QwenCoderParser::new();

    // Test with various whitespace scenarios
    let input = r#"<tool_call>
    <function=process>
        <parameter=trimmed>  spaces around  </parameter>
        <parameter=newlines>
            Line 1
            Line 2
        </parameter>
    </function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // Values should preserve internal whitespace but trim edges
    assert_eq!(args["trimmed"], "spaces around");
    assert!(args["newlines"].as_str().unwrap().contains("Line 1"));
    assert!(args["newlines"].as_str().unwrap().contains("Line 2"));
}

#[tokio::test]
async fn test_qwen_coder_no_tools() {
    // Test input with no tool calls at all
    let parser = QwenCoderParser::new();

    let input = r#"This is just a normal response without any tool calls.
I can provide information directly without using any tools.
Even if I mention function names like get_weather or search,
they are not actual tool calls unless properly formatted."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // No tools should be extracted
    assert_eq!(
        tools.len(),
        0,
        "Should not extract any tools from plain text"
    );

    // All content should be returned as normal text
    assert_eq!(
        normal_text, input,
        "All content should be returned as normal text when no tools present"
    );
}

#[tokio::test]
async fn test_qwen_coder_streaming_state_reset() {
    let mut parser = QwenCoderParser::new();
    let tools = create_test_tools();

    // First tool
    let chunks1 = vec![
        r#"<tool_call><function=get_weather>"#,
        r#"<parameter=city>London</parameter>"#,
        "</function></tool_call>",
    ];

    for chunk in chunks1 {
        parser.parse_incremental(chunk, &tools).await.unwrap();
    }

    // Second tool - state should be reset
    let chunks2 = vec![
        r#"<tool_call><function=search>"#,
        r#"<parameter=query>rust</parameter>"#,
        "</function></tool_call>",
    ];

    let mut second_tool_name = None;
    for chunk in chunks2 {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                second_tool_name = Some(name);
            }
        }
    }

    assert_eq!(second_tool_name, Some("search".to_string()));
}

#[tokio::test]
async fn test_qwen_coder_realistic_chunks() {
    let tools = create_test_tools();
    let mut parser = QwenCoderParser::new();

    let input = r#"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
<parameter=units>celsius</parameter>
</function>
</tool_call>"#;
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
async fn test_qwen_coder_xml_tag_arrives_in_parts() {
    let tools = create_test_tools();
    let mut parser = QwenCoderParser::new();

    let chunks = vec![
        "<to", "ol_", "cal", "l>", "<fun", "cti", "on=", "get", "_we", "ath", "er>", "<par", "ame",
        "ter=", "cit", "y>", "Tok", "yo", "</", "par", "ame", "ter>", "</", "func", "tion>", "</",
        "too", "l_c", "all>",
    ];

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
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
async fn test_qwen_coder_content_before_and_after_tool_calls() {
    let parser = QwenCoderParser::new();

    let input = r#"I'll analyze the weather for you now.
<tool_call>
<function=get_weather>
<parameter=city>Boston</parameter>
<parameter=state>MA</parameter>
</function>
</tool_call>
Based on the analysis, here's what I found."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Verify tool extraction
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    // Verify content preservation (only text before tool call is returned)
    assert!(normal_text.contains("I'll analyze the weather for you now."));
    // Text after tool call is not included in parse_complete
    assert!(!normal_text.contains("Based on the analysis, here's what I found."));

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Boston");
    assert_eq!(args["state"], "MA");
}

#[tokio::test]
async fn test_qwen_coder_incomplete_tool_call() {
    let parser = QwenCoderParser::new();

    // Incomplete tool call - missing closing tag
    let input = r#"<tool_call>
<function=get_weather>
<parameter=city>Chicago</parameter>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should not extract incomplete tool calls
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return as normal text
}

#[tokio::test]
async fn test_qwen_coder_malformed_function_tag() {
    let parser = QwenCoderParser::new();

    // Malformed function tag - missing name attribute
    let input = r#"<tool_call>
<function>
<parameter=city>Miami</parameter>
</function>
</tool_call>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should not extract tool calls with malformed function tags
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_qwen_coder_many_parameters() {
    let parser = QwenCoderParser::new();

    let mut params_xml = String::new();
    for i in 1..=20 {
        params_xml.push_str(&format!(
            r#"<parameter=param{}>value{}</parameter>
"#,
            i, i
        ));
    }

    let input = format!(
        r#"<tool_call>
<function=complex_func>
{}
</function>
</tool_call>"#,
        params_xml
    );

    let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "complex_func");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();

    // Verify all 20 parameters are parsed
    for i in 1..=20 {
        let key = format!("param{}", i);
        let expected_value = format!("value{}", i);
        assert_eq!(args[key], expected_value);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[tokio::test]
async fn test_qwen_coder_malformed_xml_missing_parameter_close() {
    let parser = QwenCoderParser::new();

    // Missing </parameter> closing tag - parser regex won't match incomplete parameter
    let input = r#"<tool_call>
<function=get_weather>
<parameter=city>Beijing
</function>
</tool_call>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // The parser extracts the tool call but with empty arguments since
    // the parameter block is malformed (no </parameter>)
    // This is acceptable behavior - we extract what we can
    if tools.is_empty() {
        // If no tools extracted, input returned as normal text
        assert_eq!(normal_text, input);
    } else {
        // If tool extracted, it should have the function name
        assert_eq!(tools[0].function.name, "get_weather");
    }
}

#[tokio::test]
async fn test_qwen_coder_malformed_xml_unclosed_function() {
    let parser = QwenCoderParser::new();

    // Missing </function> closing tag
    let input = r#"<tool_call>
<function=get_weather>
<parameter=city>Beijing</parameter>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Parser should still extract the tool since it has complete tool_call tags
    // and the function name + parameters are present
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_qwen_coder_malformed_xml_nested_tool_calls() {
    let parser = QwenCoderParser::new();

    // Nested tool_call tags (invalid)
    let input = r#"<tool_call>
<function=outer>
<tool_call>
<function=inner>
</function>
</tool_call>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should handle gracefully - may parse first complete tool_call
    // The exact behavior depends on regex matching
    assert!(tools.len() <= 1);
}

#[tokio::test]
async fn test_qwen_coder_unicode_parameter_names() {
    let parser = QwenCoderParser::new();

    // Unicode characters in parameter names (Chinese, Japanese, emoji)
    let input = r#"<tool_call>
<function=process>
<parameter=城市>北京</parameter>
<parameter=天気>晴れ</parameter>
<parameter=emoji_key>🌍🌎🌏</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["城市"], "北京");
    assert_eq!(args["天気"], "晴れ");
    assert_eq!(args["emoji_key"], "🌍🌎🌏");
}

#[tokio::test]
async fn test_qwen_coder_unicode_function_name() {
    let parser = QwenCoderParser::new();

    // Unicode function name
    let input = r#"<tool_call>
<function=获取天气>
<parameter=location>上海</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "获取天气");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "上海");
}

#[tokio::test]
async fn test_qwen_coder_very_large_parameter_value() {
    let parser = QwenCoderParser::new();

    // Generate a large parameter value (100KB)
    let large_value: String = "x".repeat(100_000);

    let input = format!(
        r#"<tool_call>
<function=process_large>
<parameter=data>{}</parameter>
</function>
</tool_call>"#,
        large_value
    );

    let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process_large");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["data"].as_str().unwrap().len(), 100_000);
}

#[tokio::test]
async fn test_qwen_coder_very_large_nested_json_parameter() {
    let parser = QwenCoderParser::new();

    // Generate moderately nested JSON structure (10 levels to avoid stack overflow)
    let mut nested_json = String::from(r#"{"level": 0}"#);
    for i in 1..=10 {
        nested_json = format!(r#"{{"level": {}, "child": {}}}"#, i, nested_json);
    }

    let input = format!(
        r#"<tool_call>
<function=process_nested>
<parameter=config>{}</parameter>
</function>
</tool_call>"#,
        nested_json
    );

    let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process_nested");

    // Verify the nested JSON was parsed correctly
    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["config"].is_object());
    assert_eq!(args["config"]["level"], 10);
}

#[tokio::test]
async fn test_qwen_coder_streaming_malformed_recovery() {
    let mut parser = QwenCoderParser::new();
    let tools = create_test_tools();

    // First: malformed tool call (invalid function name)
    // Second: valid tool call
    let chunks = vec![
        r#"<tool_call><function=invalid_func><parameter=x>1</parameter></function></tool_call>"#,
        r#"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>"#,
    ];

    let mut valid_tool_found = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                if name == "get_weather" {
                    valid_tool_found = true;
                }
            }
        }
    }

    assert!(
        valid_tool_found,
        "Should recover and parse valid tool after invalid one"
    );
}

#[tokio::test]
async fn test_qwen_coder_parameter_with_xml_like_content() {
    let parser = QwenCoderParser::new();

    // Parameter value contains XML-like content that shouldn't be parsed as tags
    let input = r#"<tool_call>
<function=process>
<parameter=html_content><div class="test"><span>Hello</span></div></parameter>
<parameter=xml_snippet><root><child attr="value"/></root></parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["html_content"]
        .as_str()
        .unwrap()
        .contains("<div class=\"test\">"));
    assert!(args["xml_snippet"]
        .as_str()
        .unwrap()
        .contains("<root><child"));
}

#[tokio::test]
async fn test_qwen_coder_empty_parameter_value() {
    let parser = QwenCoderParser::new();

    let input = r#"<tool_call>
<function=process>
<parameter=empty></parameter>
<parameter=whitespace>   </parameter>
<parameter=normal>value</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["empty"], "");
    assert_eq!(args["whitespace"], ""); // Trimmed
    assert_eq!(args["normal"], "value");
}

// ============================================================================
// HTML Entity and Python Literal Tests
// ============================================================================

#[tokio::test]
async fn test_qwen_coder_html_entity_decoding() {
    let parser = QwenCoderParser::new();

    // Test HTML entities in parameter values
    let input = r#"<tool_call>
<function=process>
<parameter=ampersand>Tom &amp; Jerry</parameter>
<parameter=comparison>5 &lt; 10 &amp;&amp; 10 &gt; 5</parameter>
<parameter=quotes>&quot;Hello&quot; &amp; &apos;World&apos;</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["ampersand"], "Tom & Jerry");
    assert_eq!(args["comparison"], "5 < 10 && 10 > 5");
    assert_eq!(args["quotes"], "\"Hello\" & 'World'");
}

#[tokio::test]
async fn test_qwen_coder_html_numeric_entities() {
    let parser = QwenCoderParser::new();

    // Test numeric HTML entities
    let input = r#"<tool_call>
<function=process>
<parameter=decimal>&#60;tag&#62;</parameter>
<parameter=hex>&#x3C;tag&#x3E;</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["decimal"], "<tag>");
    assert_eq!(args["hex"], "<tag>");
}

#[tokio::test]
async fn test_qwen_coder_python_literals() {
    let parser = QwenCoderParser::new();

    // Test Python-style literals (True, False, None)
    let input = r#"<tool_call>
<function=process>
<parameter=py_true>True</parameter>
<parameter=py_false>False</parameter>
<parameter=py_none>None</parameter>
<parameter=json_true>true</parameter>
<parameter=json_false>false</parameter>
<parameter=json_null>null</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // Python literals should be converted
    assert_eq!(args["py_true"], true);
    assert_eq!(args["py_false"], false);
    assert_eq!(args["py_none"], serde_json::Value::Null);
    // JSON literals should also work
    assert_eq!(args["json_true"], true);
    assert_eq!(args["json_false"], false);
    assert_eq!(args["json_null"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_qwen_coder_mixed_html_and_json() {
    let parser = QwenCoderParser::new();

    // Test HTML entities within JSON structures
    let input = r#"<tool_call>
<function=search>
<parameter=query>price &lt; 100 &amp;&amp; rating &gt; 4</parameter>
<parameter=config>{"operator": "&amp;&amp;", "escape": true}</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "price < 100 && rating > 4");
    // JSON with HTML entity inside - the entity gets decoded first, then JSON parsed
    assert!(args["config"].is_object());
    assert_eq!(args["config"]["operator"], "&&");
}
