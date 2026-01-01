//! Tool constraint generation for structured output
//!
//! This module provides functions to build tool call constraints based on tool_choice,
//! supporting both structural tags (for parsers that support them) and JSON schema constraints.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::{
    protocols::common::{Tool, ToolChoice, ToolChoiceValue},
    tool_parser::factory::ParserFactory,
};

/// Build tool call constraint based on tool_choice and parser capabilities
///
/// # Arguments
/// * `tools` - List of available tools (should already be filtered if needed)
/// * `tool_choice` - Tool choice setting from request
/// * `parallel_tool_calls` - Whether parallel tool calls are enabled
/// * `tool_parser_factory` - Factory for getting parser instances
/// * `configured_parser` - Optional configured parser name (overrides model-based selection)
/// * `model` - Model name for auto-detection fallback
///
/// # Returns
/// * `Ok(Some((constraint_type, constraint_value)))` - Constraint to apply
/// * `Ok(None)` - No constraint needed
/// * `Err(String)` - Error building constraint
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
        // Auto mode: Try structural tag if parser supports it, otherwise no constraint
        ToolChoice::Value(ToolChoiceValue::Auto) => {
            // Get a fresh parser instance to check if it supports structural tags
            // (get_format_info doesn't need state, so we don't need the pooled instance)
            let parser = if let Some(parser_name) = configured_parser {
                tool_parser_factory
                    .registry()
                    .create_parser(parser_name)
                    .or_else(|| {
                        tracing::warn!(
                            "Configured parser '{}' not found, falling back to model-based selection",
                            parser_name
                        );
                        tool_parser_factory.registry().create_for_model(model)
                    })
            } else {
                tool_parser_factory.registry().create_for_model(model)
            };

            if let Some(parser) = parser {
                if let Some(format_info) = parser.get_format_info() {
                    // Build structural tag using format info
                    let structural_tag = build_structural_tag_from_format_info(tools, &format_info)?;
                    return Ok(Some(("structural_tag".to_string(), structural_tag)));
                }
            }

            // Parser doesn't support structural tags, no constraint for auto mode
            Ok(None)
        }

        // Required mode: Build JSON schema with minItems: 1
        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_json_schema(tools, parallel_tool_calls)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

        // Specific function: Return parameters schema directly
        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return Ok(None);
            }
            let tool = &tools[0];
            let params_schema = serde_json::to_string(&tool.function.parameters)
                .map_err(|e| format!("Failed to serialize tool parameters: {}", e))?;
            Ok(Some(("json_schema".to_string(), params_schema)))
        }

        // AllowedTools with required mode: tools are already filtered
        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                if tools.is_empty() {
                    return Ok(None);
                }
                let schema = build_required_json_schema(tools, parallel_tool_calls)?;
                Ok(Some(("json_schema".to_string(), schema)))
            } else {
                // "auto" mode - try structural tag if parser supports it
                let parser = if let Some(parser_name) = configured_parser {
                    tool_parser_factory
                        .registry()
                        .create_parser(parser_name)
                        .or_else(|| {
                            tracing::warn!(
                                "Configured parser '{}' not found, falling back to model-based selection",
                                parser_name
                            );
                            tool_parser_factory.registry().create_for_model(model)
                        })
                } else {
                    tool_parser_factory.registry().create_for_model(model)
                };

                if let Some(parser) = parser {
                    if let Some(format_info) = parser.get_format_info() {
                        let structural_tag = build_structural_tag_from_format_info(tools, &format_info)?;
                        return Ok(Some(("structural_tag".to_string(), structural_tag)));
                    }
                }

                // No constraint for auto mode
                Ok(None)
            }
        }

        // "none" - no constraint
        _ => Ok(None),
    }
}

/// Build structural tag from format info
fn build_structural_tag_from_format_info(
    tools: &[Tool],
    format_info: &crate::tool_parser::types::FormatInfo,
) -> Result<String, String> {
    use serde_json::json;

    let mut structures = Vec::new();
    let mut trigger_set = std::collections::HashSet::new();

    // Add primary trigger
    trigger_set.insert(format_info.trigger.clone());
    
    // Add secondary trigger if it exists
    if let Some(ref trigger_subsequent) = format_info.trigger_subsequent {
        trigger_set.insert(trigger_subsequent.clone());
    }

    for (index, tool) in tools.iter().enumerate() {
        let tool_name = &tool.function.name;
        let end = &format_info.end_pattern;
        
        // Use tool's parameters schema if available
        let schema = &tool.function.parameters;

        // Create structure for first tool call (or all if no subsequent pattern)
        let begin_first = (format_info.begin_pattern)(tool_name, index);
        structures.push(json!({
            "begin": begin_first,
            "schema": schema,
            "end": end,
        }));

        // If there's a subsequent pattern, create a second structure for subsequent calls
        if let Some(ref begin_pattern_subsequent) = format_info.begin_pattern_subsequent {
            let begin_subsequent = begin_pattern_subsequent(tool_name, index);
            structures.push(json!({
                "begin": begin_subsequent,
                "schema": schema,
                "end": end,
            }));
        }
    }

    let structural_tag = json!({
        "type": "structural_tag",
        "structures": structures,
        "triggers": trigger_set.into_iter().collect::<Vec<_>>(),
    });

    serde_json::to_string(&structural_tag)
        .map_err(|e| format!("Failed to serialize structural tag: {}", e))
}

/// Build JSON schema for required tool calls (array with minItems: 1)
/// Includes $defs consolidation from all tools (matching Python's behavior)
fn build_required_json_schema(tools: &[Tool], parallel_tool_calls: bool) -> Result<String, String> {
    // Build anyOf schemas for each tool
    let mut any_of_schemas = Vec::new();
    for tool in tools {
        let tool_schema = json!({
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

    // Consolidate $defs from all tools (matching Python's _get_tool_schema_defs)
    let mut all_defs: HashMap<String, Value> = HashMap::new();
    for tool in tools {
        if let Value::Object(params) = &tool.function.parameters {
            if let Some(Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                        // Check for conflicts
                        if existing != def_schema {
                            let error_msg = format!(
                                "Tool definition '{}' has multiple conflicting schemas, which is not supported",
                                def_name
                            );
                            tracing::error!("{}", error_msg);
                            return Err(error_msg);
                        }
                    } else {
                        all_defs.insert(def_name.clone(), def_schema.clone());
                    }
                }
            }
        }
    }

    // Build the full array schema
    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    // Add maxItems if parallel_tool_calls is false
    if !parallel_tool_calls {
        if let Value::Object(ref mut schema_obj) = array_schema {
            schema_obj.insert("maxItems".to_string(), json!(1));
        }
    }

    // Add $defs if any were found (matching Python's behavior)
    if !all_defs.is_empty() {
        if let Value::Object(ref mut schema_obj) = array_schema {
            let defs_value = Value::Object(all_defs.into_iter().collect::<Map<String, Value>>());
            schema_obj.insert("$defs".to_string(), defs_value);
        }
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize tool schema: {}", e))
}

