use crate::{
    protocols::common::{Tool, ToolChoice, ToolChoiceValue},
    tool_parser::{factory::ParserFactory, traits::ToolParser},
};

/// Build tool call constraint based on tool_choice and parallel_tool_calls.
///
/// This function follows the same pattern as get_tool_parser():
/// - Respects user's explicit --tool-call-parser flag (configured_parser)
/// - Falls back to auto-detection by model name
///
/// Strategy:
/// - tool_choice="auto": Use structural_tag from parser (if available)
/// - tool_choice="required": Use json_schema
/// - tool_choice=specific function: Use json_schema
///
/// # Arguments
/// * `tools` - Available tools
/// * `tool_choice` - Tool choice mode
/// * `parallel_tool_calls` - Whether to allow multiple tool calls
/// * `tool_parser_factory` - Factory to get parser (from SharedComponents)
/// * `configured_parser` - User's explicit --tool-call-parser choice (from processor's configured_tool_parser)
/// * `model` - Model name for auto-detection fallback
///
/// # Returns
/// Option<(constraint_type, constraint_value)>
pub fn build_tool_call_constraint(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
    parallel_tool_calls: bool,
    tool_parser_factory: &ParserFactory,
    configured_parser: Option<&String>,
    model: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(choice) = tool_choice.as_ref() else {
        return Ok(None);
    };

    match choice {
        ToolChoice::Value(ToolChoiceValue::Auto) => {
            let parser = get_tool_parser(tool_parser_factory, configured_parser, model);

            let tag = parser.build_structural_tag(tools, false, !parallel_tool_calls)?;
            Ok(Some(("structural_tag".to_string(), tag)))
        }

        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_json_schema(tools, parallel_tool_calls)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return Ok(None);
            }
            let schema = build_specific_function_json_schema(&tools[0], parallel_tool_calls)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                let schema = build_required_json_schema(tools, parallel_tool_calls)?;
                Ok(Some(("json_schema".to_string(), schema)))
            } else {
                let parser = get_tool_parser(tool_parser_factory, configured_parser, model);
                let tag = parser.build_structural_tag(tools, false, !parallel_tool_calls)?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
        }

        _ => Ok(None),
    }
}

/// Get a tool parser for given model, respecting configured_parser.
fn get_tool_parser(
    factory: &ParserFactory,
    configured_parser: Option<&String>,
    model: &str,
) -> Box<dyn ToolParser> {
    if let Some(parser_name) = configured_parser {
        if let Some(parser) = factory.registry().create_parser(parser_name) {
            return parser;
        }
    }

    factory
        .create_for_model(model)
        .unwrap_or_else(|| factory.registry().create_parser("json").unwrap())
}

/// Build JSON schema for required tool choice mode.
fn build_required_json_schema(tools: &[Tool], parallel_tool_calls: bool) -> Result<String, String> {
    let mut any_of_schemas = Vec::new();
    for tool in tools {
        let tool_schema = serde_json::json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        });
        any_of_schemas.push(tool_schema);
    }

    let mut array_schema = serde_json::json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    if !parallel_tool_calls {
        array_schema["maxItems"] = serde_json::json!(1);
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize json schema: {}", e))
}

/// Build JSON schema for specific function choice.
fn build_specific_function_json_schema(
    tool: &Tool,
    _parallel_tool_calls: bool,
) -> Result<String, String> {
    let schema = serde_json::json!({
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        }
    });

    serde_json::to_string(&schema).map_err(|e| format!("Failed to serialize json schema: {}", e))
}
