//! Tests for constraint generation according to GitHub issue #13032
//!
//! Tests `constraints::build_tool_call_constraint()` with various tool_choice modes
//! and parallel_tool_calls configurations

use smg::protocols::common::{Tool, ToolChoice, ToolChoiceValue};
use smg::tool_parser::{constraints, factory::ParserFactory};

/// Helper to create test tools
fn create_test_tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: smg::protocols::common::Function {
                name: "get_weather".to_string(),
                description: Some("Get current weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: smg::protocols::common::Function {
                name: "search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }),
                strict: None,
            },
        },
    ]
}

/// Helper to create tool with complex schema
fn create_tool_with_defs() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: sgl_model_gateway::protocols::common::Function {
            name: "complex_tool".to_string(),
            description: Some("Tool with $defs".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "item": {"$ref": "#/$defs/Item"}
                },
                "required": ["item"],
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "number"}
                        }
                    }
                }
            }),
            strict: None,
        },
    }
}

// ============================================================================
// Auto mode tests - structural_tag generation
// ============================================================================

#[test]
fn test_auto_mode_generates_structural_tag() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "structural_tag");

    let tag_json: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    assert_eq!(tag_json["format"], "triggered_tags");
}

#[test]
fn test_auto_mode_with_configured_parser() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));
    let configured_parser = &"mistral".to_string();

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        Some(configured_parser),
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "structural_tag");

    let tag_json: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    let triggers = tag_json["triggers"].as_array().unwrap();
    assert!(triggers.iter().any(|t| t == "[TOOL_CALLS]"));
}

#[test]
fn test_auto_mode_auto_detection() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "llama-3.2",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "structural_tag");

    let tag_json: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    let triggers = tag_json["triggers"].as_array().unwrap();
    assert!(triggers.iter().any(|t| t == "<|python_tag|>"));
}

// ============================================================================
// Required mode tests - json_schema generation
// ============================================================================

#[test]
fn test_required_mode_generates_json_schema() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "json_schema");

    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    assert_eq!(schema["type"], "array");
    assert_eq!(schema["minItems"], 1);
    assert!(schema["maxItems"], serde_json::json!(null));

    let items = &schema["items"];
    assert_eq!(items["type"], "object");
    assert!(items["anyOf"].is_array());
}

#[test]
fn test_required_mode_single_tool() {
    let factory = ParserFactory::new();
    let tools = vec![create_test_tools()[0].clone()];
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "json_schema");

    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    let any_of = schema["items"]["anyOf"].as_array().unwrap();
    assert_eq!(any_of.len(), 1);
    assert_eq!(any_of[0]["properties"]["name"]["enum"][0], "get_weather");
}

// ============================================================================
// Specific function tests
// ============================================================================

#[test]
fn test_specific_function_generates_json_schema() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Function {
        name: "search".to_string(),
    });

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "json_schema");

    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    assert_eq!(schema["type"], "array");
    assert_eq!(schema["minItems"], 1);
    assert_eq!(schema["maxItems"], 1);

    let items = &schema["items"];
    assert_eq!(items["properties"]["name"]["enum"][0], "search");
}

// ============================================================================
// parallel_tool_calls parameter tests
// ============================================================================

#[test]
fn test_parallel_true_no_maxitems() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();

    assert!(schema.get("maxItems").is_none());
}

#[test]
fn test_parallel_false_sets_maxitems() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        false,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();

    assert_eq!(schema["maxItems"], 1);
}

#[test]
fn test_auto_mode_stop_after_first() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        false,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();

    assert_eq!(tag_json["stop_after_first"], true);
}

// ============================================================================
// Configured parser precedence tests
// ============================================================================

#[test]
fn test_configured_parser_takes_precedence() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));
    let configured_parser = &"llama".to_string();

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        Some(configured_parser),
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();

    let tag_json: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    let triggers = tag_json["triggers"].as_array().unwrap();

    assert!(
        triggers.iter().any(|t| t == "<|python_tag|>"),
        "Should use llama parser, not json parser for gpt-4"
    );
}

#[test]
fn test_auto_detection_when_no_configured_parser() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();

    let tag_json: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    let triggers = tag_json["triggers"].as_array().unwrap();

    assert!(
        triggers.iter().any(|t| t == "[TOOL_CALLS]"),
        "Should auto-detect json parser for gpt-4"
    );
}

#[test]
fn test_configured_parser_invalid_falls_back() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));
    let configured_parser = &"invalid_parser".to_string();

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        Some(configured_parser),
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();

    let tag_json: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    let triggers = tag_json["triggers"].as_array().unwrap();

    assert!(
        triggers.iter().any(|t| t == "[TOOL_CALLS]"),
        "Should fall back to auto-detection when configured parser is invalid"
    );
}

// ============================================================================
// AllowedTools mode tests
// ============================================================================

#[test]
fn test_allowed_tools_auto_mode() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::AllowedTools {
        mode: "auto".to_string(),
        allowed_tools: vec!["get_weather".to_string(), "search".to_string()],
    });

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "structural_tag");
}

#[test]
fn test_allowed_tools_required_mode() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::AllowedTools {
        mode: "required".to_string(),
        allowed_tools: vec!["get_weather".to_string(), "search".to_string()],
    });

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    assert_eq!(constraint.0, "json_schema");
}

// ============================================================================
// Edge cases and error handling
// ============================================================================

#[test]
fn test_empty_tools_returns_none() {
    let factory = ParserFactory::new();
    let tools: Vec<Tool> = vec![];
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap();
    assert!(constraint.is_none());
}

#[test]
fn test_none_tool_choice_returns_none() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice: Option<ToolChoice> = None;

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap();
    assert!(constraint.is_none());
}

#[test]
fn test_schema_serialization_valid_json() {
    let factory = ParserFactory::new();
    let tools = vec![create_tool_with_defs()];
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();

    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    assert!(schema["$defs"].is_object());
    assert_eq!(schema["$defs"]["Item"]["type"], "object");
}

#[test]
fn test_structural_tag_valid_json() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();

    let tag: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();
    assert_eq!(tag["format"], "triggered_tags");
    assert!(tag["tags"].is_array());
    assert!(tag["triggers"].is_array());
}

// ============================================================================
// Multiple tools in schema
// ============================================================================

#[test]
fn test_required_mode_multiple_tools() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();

    let any_of = schema["items"]["anyOf"].as_array().unwrap();
    assert_eq!(any_of.len(), 2);

    let names: Vec<&str> = any_of
        .iter()
        .map(|item| item["properties"]["name"]["enum"][0].as_str().unwrap())
        .collect();

    assert!(names.contains(&"get_weather"));
    assert!(names.contains(&"search"));
}

#[test]
fn test_required_mode_with_all_properties() {
    let factory = ParserFactory::new();
    let tools = create_test_tools();
    let tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));

    let result = constraints::build_tool_call_constraint(
        &tools,
        &tool_choice,
        true,
        &factory,
        None,
        "gpt-4",
    );

    assert!(result.is_ok());
    let constraint = result.unwrap().unwrap();
    let schema: serde_json::Value = serde_json::from_str(&constraint.1).unwrap();

    let any_of = schema["items"]["anyOf"].as_array().unwrap();

    for item in any_of {
        assert!(item["properties"]["name"].is_object());
        assert!(item["properties"]["parameters"].is_object());
        assert!(item["required"]
            .as_array()
            .unwrap()
            .contains(&"name".into()));
        assert!(item["required"]
            .as_array()
            .unwrap()
            .contains(&"parameters".into()));
    }
}
