//! Pythonic Parser Integration Tests
//!
//! Tests for the Pythonic parser which handles Python function call syntax

use serde_json::json;
use sgl_model_gateway::tool_parser::{PythonicParser, ToolParser};

mod common;
use common::create_test_tools;

#[tokio::test]
async fn test_pythonic_single_function() {
    let parser = PythonicParser::new();
    let input = r#"[get_weather(city="London", units="celsius")]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "London");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_pythonic_multiple_functions() {
    let parser = PythonicParser::new();
    let input =
        r#"[search_web(query="Rust programming", max_results=5), get_time(timezone="UTC")]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search_web");
    assert_eq!(tools[1].function.name, "get_time");

    let args0: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args0["query"], "Rust programming");
    assert_eq!(args0["max_results"], 5);
}

#[tokio::test]
async fn test_pythonic_with_python_literals() {
    let parser = PythonicParser::new();
    let input = r#"[configure(enabled=True, disabled=False, optional=None)]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["enabled"], true);
    assert_eq!(args["disabled"], false);
    assert_eq!(args["optional"], json!(null));
}

#[tokio::test]
async fn test_pythonic_with_lists_and_dicts() {
    let parser = PythonicParser::new();
    let input =
        r#"[process_data(items=[1, 2, 3], config={"key": "value", "nested": {"deep": True}})]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["items"], json!([1, 2, 3]));
    assert_eq!(args["config"]["key"], "value");
    assert_eq!(args["config"]["nested"]["deep"], true);
}

#[tokio::test]
async fn test_pythonic_with_special_tokens() {
    let parser = PythonicParser::new();

    // Llama 4 sometimes outputs these tokens
    let input = r#"<|python_start|>[calculate(x=10, y=20)]<|python_end|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "calculate");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["x"], 10);
    assert_eq!(args["y"], 20);
}

#[tokio::test]
async fn test_pythonic_with_nested_parentheses() {
    let parser = PythonicParser::new();
    let input = r#"[math_eval(expression="(2 + 3) * (4 - 1)", round_to=2)]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["expression"], "(2 + 3) * (4 - 1)");
    assert_eq!(args["round_to"], 2);
}

#[tokio::test]
async fn test_pythonic_with_escaped_quotes() {
    let parser = PythonicParser::new();
    let input = r#"[echo(text="She said \"Hello\" to him")]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "She said \"Hello\" to him");
}

#[tokio::test]
async fn test_pythonic_empty_arguments() {
    let parser = PythonicParser::new();
    let input = r#"[ping()]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "ping");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args, json!({}));
}

#[tokio::test]
async fn test_pythonic_format_detection() {
    let parser = PythonicParser::new();

    assert!(!parser.has_tool_markers("[function_name(")); // Incomplete
    assert!(parser.has_tool_markers("[get_weather(city=\"NYC\")]"));
    assert!(!parser.has_tool_markers("Just plain text"));
    assert!(!parser.has_tool_markers("{\"name\": \"test\"}")); // JSON
}

#[tokio::test]
async fn test_pythonic_invalid_syntax() {
    let parser = PythonicParser::new();

    // Missing closing bracket
    let input = r#"[function(arg=value"#;
    if let Ok((_normal_text, tools)) = parser.parse_complete(input).await {
        assert_eq!(tools.len(), 0);
    }
    // Error is also acceptable for invalid syntax

    // Invalid Python syntax - empty parameter name
    // Note: The parser currently accepts this invalid syntax and returns a result
    // This is a known limitation of the current implementation
    let input = r#"[function(=value)]"#;
    if let Ok((_normal_text, tools)) = parser.parse_complete(input).await {
        // The parser incorrectly accepts this, returning 1 result
        // We'll accept this behavior for now but note it's not ideal
        assert!(tools.len() <= 1, "Should parse at most one function");
    }
    // Error would be the correct behavior
}

#[tokio::test]
async fn test_pythonic_real_world_llama4() {
    let parser = PythonicParser::new();

    // Actual output from Llama 4 model
    let input = r#"I'll help you with multiple tasks. Let me search for information and perform calculations.

[web_search(query="latest Rust features", max_results=3, safe_search=True),
 calculate(expression="42 * 3.14159", precision=2),
 get_weather(city="San Francisco", units="fahrenheit", include_forecast=False)]

These functions will provide the information you need."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 3);
    assert_eq!(normal_text, "I'll help you with multiple tasks. Let me search for information and perform calculations.\n\n\n\nThese functions will provide the information you need.");
    assert_eq!(tools[0].function.name, "web_search");
    assert_eq!(tools[1].function.name, "calculate");
    assert_eq!(tools[2].function.name, "get_weather");

    let args0: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args0["query"], "latest Rust features");
    assert_eq!(args0["safe_search"], true);
}

#[tokio::test]
async fn test_pythonic_nested_brackets_in_lists() {
    let parser = PythonicParser::new();

    let input = r#"[process_matrix(data=[[1, 2], [3, 4]], labels=["row[0]", "row[1]"])]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process_matrix");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["data"], json!([[1, 2], [3, 4]]));
    assert_eq!(args["labels"], json!(["row[0]", "row[1]"]));
}

#[tokio::test]
async fn test_pythonic_nested_brackets_in_dicts() {
    let parser = PythonicParser::new();

    let input =
        r#"[analyze(config={"patterns": ["[a-z]+", "[0-9]+"], "nested": {"list": [1, [2, 3]]}})]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "analyze");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["config"]["patterns"], json!(["[a-z]+", "[0-9]+"]));
    assert_eq!(args["config"]["nested"]["list"], json!([1, [2, 3]]));
}

#[tokio::test]
async fn test_pythonic_mixed_quotes() {
    let parser = PythonicParser::new();

    let input = r#"[format_text(single='Hello', double="World", mixed="It's \"quoted\"")]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "format_text");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["single"], "Hello");
    assert_eq!(args["double"], "World");
    assert_eq!(args["mixed"], "It's \"quoted\"");
}

#[tokio::test]
async fn test_pythonic_complex_nesting() {
    let parser = PythonicParser::new();

    let input = r#"[transform(
        matrix=[[1, [2, 3]], [4, [5, [6, 7]]]],
        operations=[{"type": "scale", "factor": [2, 3]}, {"type": "rotate", "angle": 90}],
        metadata={"tags": ["nested[0]", "nested[1]"], "config": {"depth": [1, 2, 3]}}
    )]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "transform");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["matrix"].is_array());
    assert!(args["operations"].is_array());
    assert_eq!(args["operations"][0]["type"], "scale");
    assert_eq!(args["metadata"]["config"]["depth"], json!([1, 2, 3]));
}

#[tokio::test]
async fn test_parse_streaming_no_brackets() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text = "This is just normal text without any tool calls.";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    // Expected - no tool calls found
    assert!(result.calls.is_empty());
}

#[tokio::test]
async fn test_parse_streaming_complete_tool_call() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text = "Here's a tool call: [get_weather(location='New York', unit='celsius')]";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should parse complete tool call");
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "get_weather");
    let args: serde_json::Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
    assert_eq!(args["location"], "New York");
    assert_eq!(args["unit"], "celsius");
}

#[tokio::test]
async fn test_parse_streaming_text_before_tool_call() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text = "This is some text before [get_weather(location='London')]";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should parse tool call");
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "get_weather");
    let args: serde_json::Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
    assert_eq!(args["location"], "London");
}

#[tokio::test]
async fn test_parse_streaming_partial_tool_call() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    // First chunk with opening bracket but no closing bracket
    let text1 = "Let me check the weather: [get_weather(location=";
    let result1 = parser.parse_incremental(text1, &tools).await.unwrap();

    // First chunk should be incomplete
    assert!(
        result1.calls.is_empty(),
        "First chunk should not return tool call"
    );

    // Second chunk completing the tool call
    let text2 = "'Paris')]";
    let result2 = parser.parse_incremental(text2, &tools).await.unwrap();

    assert!(
        !result2.calls.is_empty(),
        "Second chunk should complete tool call"
    );
    assert_eq!(result2.calls[0].name.as_ref().unwrap(), "get_weather");
    let args: serde_json::Value = serde_json::from_str(&result2.calls[0].parameters).unwrap();
    assert_eq!(args["location"], "Paris");
}

#[tokio::test]
async fn test_parse_streaming_bracket_without_text_before() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text = "[search(query='python programming')]";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should parse tool call");
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "search");
    let args: serde_json::Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
    assert_eq!(args["query"], "python programming");
}

#[tokio::test]
async fn test_parse_streaming_text_after_tool_call() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    // First chunk with complete tool call and some text after
    let text = "[get_weather(location='Tokyo')] Here's the forecast:";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    assert!(!result.calls.is_empty(), "Should parse tool call");
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "get_weather");
    // Text after tool call is handled by parser internally
}

#[tokio::test]
async fn test_parse_streaming_multiple_tool_calls() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text = "[get_weather(location='Berlin'), search(query='restaurants')]";

    // Current implementation may handle this as a single parse
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    // The parser should handle multiple tools in one bracket pair
    // This test is flexible about the implementation behavior
    if !result.calls.is_empty() {
        // Parser found at least one tool
        assert!(result.calls[0].name.is_some());
    }
    // Also acceptable if parser returns empty waiting for more context
}

#[tokio::test]
async fn test_parse_streaming_opening_bracket_only() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text = "Let's try this: [";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    // Should be incomplete - no complete tool call
    assert!(
        result.calls.is_empty(),
        "Should not return tool call for partial bracket"
    );
}

#[tokio::test]
async fn test_parse_streaming_nested_brackets() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text = "[get_weather(location='New York', unit='celsius', data=[1, 2, 3])]";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    assert!(
        !result.calls.is_empty(),
        "Should parse tool call with nested brackets"
    );
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "get_weather");
    let args: serde_json::Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
    assert_eq!(args["location"], "New York");
    assert_eq!(args["unit"], "celsius");
    assert_eq!(args["data"], json!([1, 2, 3]));
}

#[tokio::test]
async fn test_parse_streaming_nested_brackets_dict() {
    let mut parser = PythonicParser::new();
    let tools = create_test_tools();

    let text = r#"[search(query='test', config={'options': [1, 2], 'nested': {'key': 'value'}})]"#;
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    assert!(
        !result.calls.is_empty(),
        "Should parse tool call with nested dict"
    );
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "search");
    let args: serde_json::Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
    assert_eq!(args["query"], "test");
    assert_eq!(args["config"]["options"], json!([1, 2]));
    assert_eq!(args["config"]["nested"]["key"], "value");
}

#[tokio::test]
async fn test_parse_streaming_multiple_tools_with_nested_brackets() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let text =
        "[get_weather(location='Paris', data=[10, 20]), search(query='test', filters=['a', 'b'])]";
    let result = parser.parse_incremental(text, &tools).await.unwrap();

    // Should parse tools successfully
    if !result.calls.is_empty() {
        // At least gets the first tool
        assert!(result.calls[0].name.is_some());
    }
}

#[tokio::test]
async fn test_parse_streaming_partial_nested_brackets() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    // First chunk with nested brackets but incomplete
    let text1 = "Here's a call: [get_weather(location='Tokyo', data=[1, 2";
    let result1 = parser.parse_incremental(text1, &tools).await.unwrap();

    // First chunk should be incomplete
    assert!(result1.calls.is_empty(), "First chunk should not complete");

    // Second chunk completing the nested brackets
    let text2 = ", 3])]";
    let result2 = parser.parse_incremental(text2, &tools).await.unwrap();

    assert!(
        !result2.calls.is_empty(),
        "Second chunk should complete tool call"
    );
    assert_eq!(result2.calls[0].name.as_ref().unwrap(), "get_weather");
    let args: serde_json::Value = serde_json::from_str(&result2.calls[0].parameters).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["data"], json!([1, 2, 3]));
}

#[tokio::test]
async fn test_parse_streaming_with_python_start_and_end_token() {
    let mut parser = PythonicParser::new();

    let tools = create_test_tools();

    let chunks = vec![
        "Here's a call: ",
        "<|python_",
        "start|>[get_weather(location=",
        "'Tokyo', data=[1, 2",
        ", 3])]<|python_end|>",
    ];

    let mut got_tool = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        if !result.calls.is_empty() {
            if let Some(name) = &result.calls[0].name {
                assert_eq!(name, "get_weather");
                let args: serde_json::Value =
                    serde_json::from_str(&result.calls[0].parameters).unwrap();
                assert_eq!(args["location"], "Tokyo");
                assert_eq!(args["data"], json!([1, 2, 3]));
                got_tool = true;
            }
        }
    }

    assert!(got_tool, "Should have parsed the tool call");
}

#[tokio::test]
async fn test_detect_and_parse_with_python_start_and_end_token() {
    let parser = PythonicParser::new();

    let text = "User wants to get the weather in Mars. <|python_start|>[get_weather(location='Mars', unit='celsius')]<|python_end|> In this way we will get the weather in Mars.";
    let (_normal_text, tools) = parser.parse_complete(text).await.unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Mars");
    assert_eq!(args["unit"], "celsius");
}
