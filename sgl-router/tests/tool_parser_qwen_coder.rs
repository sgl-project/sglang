//! Qwen Coder Parser Integration Tests
//!
//! Tests for the Qwen Coder parser which handles XML format:
//! <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>

use serde_json::json;
use sglang_router_rs::tool_parser::{
    parsers::QwenCoderParser,
    traits::ToolParser,
};

mod common;
use common::{create_test_tools, streaming_helpers::*};

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
        "<to", "ol_", "cal", "l>", "<fun", "cti", "on=", "get", "_we", "ath", "er>", "<par",
        "ame", "ter=", "cit", "y>", "Tok", "yo", "</", "par", "ame", "ter>", "</", "func",
        "tion>", "</", "too", "l_c", "all>",
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

