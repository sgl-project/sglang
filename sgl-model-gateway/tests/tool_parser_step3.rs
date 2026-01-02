//! Step3 Parser Integration Tests

use sgl_model_gateway::tool_parser::{Step3Parser, ToolParser};

mod common;
use common::create_test_tools;

#[tokio::test]
async fn test_step3_complete_parsing() {
    let parser = Step3Parser::new();

    let input = r#"Let me help you.
<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="search">
<steptml:parameter name="query">rust programming</steptml:parameter>
<steptml:parameter name="limit">10</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>
Here are the results..."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me help you.\n");
    assert_eq!(tools[0].function.name, "search");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust programming");
    assert_eq!(args["limit"], 10);
}

#[tokio::test]
async fn test_step3_multiple_tools() {
    let parser = Step3Parser::new();

    let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="get_weather">
<steptml:parameter name="location">Tokyo</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="get_news">
<steptml:parameter name="category">tech</steptml:parameter>
<steptml:parameter name="limit">5</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "get_news");
}

#[tokio::test]
async fn test_step3_type_conversion() {
    let parser = Step3Parser::new();

    let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="process">
<steptml:parameter name="count">100</steptml:parameter>
<steptml:parameter name="rate">2.5</steptml:parameter>
<steptml:parameter name="active">true</steptml:parameter>
<steptml:parameter name="optional">null</steptml:parameter>
<steptml:parameter name="text">hello world</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["count"], 100);
    assert_eq!(args["rate"], 2.5);
    assert_eq!(args["active"], true);
    assert_eq!(args["optional"], serde_json::Value::Null);
    assert_eq!(args["text"], "hello world");
}

#[tokio::test]
async fn test_step3_streaming() {
    let mut parser = Step3Parser::new();

    let tools = create_test_tools();

    // Simulate streaming chunks
    let chunks = vec![
        "<｜tool_calls_begin｜>\n",
        "<｜tool_call_begin｜>function",
        "<｜tool_sep｜><steptml:invoke name=\"calc\">",
        "\n<steptml:parameter name=\"x\">10</steptml:parameter>",
        "\n<steptml:parameter name=\"y\">20</steptml:parameter>",
        "\n</steptml:invoke><｜tool_call_end｜>",
        "\n<｜tool_calls_end｜>",
    ];

    let mut found_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        if !result.calls.is_empty() {
            if let Some(name) = &result.calls[0].name {
                assert_eq!(name, "calc");
                found_complete = true;
            }
        }
    }

    assert!(found_complete);
}

#[test]
fn test_step3_format_detection() {
    let parser = Step3Parser::new();

    // Should detect Step3 format
    assert!(parser.has_tool_markers("<｜tool_calls_begin｜>"));
    assert!(parser.has_tool_markers("text with <｜tool_calls_begin｜> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_step3_nested_steptml() {
    let parser = Step3Parser::new();

    let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="config">
<steptml:parameter name="settings">{"nested": {"key": "value"}}</steptml:parameter>
<steptml:parameter name="array">[1, 2, 3]</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "config");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["settings"].is_object());
    assert!(args["array"].is_array());
}

#[tokio::test]
async fn test_step3_python_literals() {
    let parser = Step3Parser::new();

    let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="test">
<steptml:parameter name="bool_true">True</steptml:parameter>
<steptml:parameter name="bool_false">False</steptml:parameter>
<steptml:parameter name="none_value">None</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["bool_true"], true);
    assert_eq!(args["bool_false"], false);
    assert_eq!(args["none_value"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_steptml_format() {
    let parser = Step3Parser::new();

    let input = r#"Text before.
<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="search">
<steptml:parameter name="query">rust lang</steptml:parameter>
<steptml:parameter name="limit">10</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>Text after."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Text before.\n");
    assert_eq!(tools[0].function.name, "search");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust lang");
    assert_eq!(args["limit"], 10);
    // TODO: Verify normal text extraction
}

#[tokio::test]
async fn test_json_parameter_values() {
    let parser = Step3Parser::new();

    let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="config">
<steptml:parameter name="settings">{"nested": {"value": true}}</steptml:parameter>
<steptml:parameter name="items">[1, 2, 3]</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["settings"].is_object());
    assert!(args["items"].is_array());
}

#[tokio::test]
async fn test_step3_parameter_with_angle_brackets() {
    let parser = Step3Parser::new();

    let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="compare">
<steptml:parameter name="expression">a < b && b > c</steptml:parameter>
<steptml:parameter name="context">comparison test</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "compare");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["expression"], "a < b && b > c");
    assert_eq!(args["context"], "comparison test");
}

#[tokio::test]
async fn test_step3_empty_function_name() {
    let parser = Step3Parser::new();

    let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="">
<steptml:parameter name="param">value</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0); // Should reject empty function name
}
