//! The tool schemas shared by output parsing and constrained generation.

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::{
    StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext, ToolChoice,
    ToolDefinition, TriggeredTagsConfig,
};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCallChunk, ChatCompletionRequestMessage, ChatCompletionTool,
    ChatCompletionToolChoiceOption, FunctionObject,
};

use super::ChatToolCallDelta;
use crate::ProcessorError;

/// Collect top-level and dynamic system-message tools without moving their
/// declarations in the prompt. Dynamic tools accept wrapped OpenAI declarations
/// or bare function schemas, as Dynamo's Kimi formatter does.
pub fn chat_tool_definitions(
    tools: Option<&[ChatCompletionTool]>,
    messages: &[ChatCompletionRequestMessage],
) -> Result<Vec<ToolDefinition>, ProcessorError> {
    let mut definitions: Vec<_> = tools
        .into_iter()
        .flatten()
        .map(|tool| definition(tool.function.clone()))
        .collect();
    for message in messages {
        let ChatCompletionRequestMessage::System(system) = message else {
            continue;
        };
        for tool in system.tools.iter().flatten() {
            let function = if tool.get("type").is_some() || tool.get("function").is_some() {
                serde_json::from_value::<ChatCompletionTool>(tool.clone()).map(|tool| tool.function)
            } else {
                serde_json::from_value::<FunctionObject>(tool.clone())
            }
            .map_err(|error| {
                ProcessorError::InvalidRequest(format!("invalid system-message tool: {error}"))
            })?;
            definitions.push(definition(function));
        }
    }
    Ok(definitions)
}

fn definition(function: FunctionObject) -> ToolDefinition {
    ToolDefinition {
        name: function.name,
        parameters: function.parameters,
        strict: function.strict,
    }
}

/// Map SGLang tool-parser aliases onto Dynamo's tool-parser names.
pub fn dynamo_tool_parser_name(parser: &str) -> &str {
    match parser {
        "llama3" => "llama3_json",
        "qwen" => "qwen25",
        "glm" | "glm45" => "glm47",
        other => other,
    }
}

/// The sampling constraint a `tool_choice` turns into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolConstraint {
    StructuralTag(String),
    JsonSchema(String),
}

/// Map the OpenAI `tool_choice` onto Dynamo's; a missing choice is `auto`.
pub fn dynamo_tool_choice(choice: &Option<ChatCompletionToolChoiceOption>) -> ToolChoice {
    match choice {
        Some(ChatCompletionToolChoiceOption::None) => ToolChoice::None,
        Some(ChatCompletionToolChoiceOption::Required) => ToolChoice::Required,
        Some(ChatCompletionToolChoiceOption::Named(choice)) => {
            ToolChoice::Named(choice.function.name.clone())
        }
        Some(ChatCompletionToolChoiceOption::Auto) | None => ToolChoice::Auto,
    }
}

/// Validate `tool_choice` against `tools`, then build its constraint when a
/// tool `parser` is configured: the parser's structural tag if it has one,
/// else a JSON-schema array of `{name, parameters}` for `required`/named.
pub fn tool_constraint(
    parser: Option<&str>,
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
) -> Result<Option<ToolConstraint>, ProcessorError> {
    if *tool_choice == ToolChoice::None {
        return Ok(None);
    }
    if *tool_choice == ToolChoice::Required && tools.is_empty() {
        return Err("tool_choice is \"required\" but tools is empty".into());
    }
    if let ToolChoice::Named(name) = tool_choice
        && !tools.iter().any(|tool| &tool.name == name)
    {
        return Err(format!("tool named \"{name}\" in tool_choice is not present in tools").into());
    }

    let Some(parser) = parser else {
        return Ok(None);
    };
    let parser = dynamo_tool_parser_name(parser);
    let config = get_tool_parser_map()
        .get(parser)
        .ok_or_else(|| format!("tool-call parser `{parser}` is not supported by Dynamo"))?;
    let builder = config.structural_tag_builder.clone().or_else(|| {
        (parser == "llama3_json"
            && *tool_choice == ToolChoice::Auto
            && tools.iter().any(|tool| tool.strict.unwrap_or(false)))
        .then(|| {
            StructuralTagBuilder::TriggeredTags(TriggeredTagsConfig {
                begin_template: r#"<|python_tag|>{"name":"{name}", "arguments":"#.to_string(),
                end_template: "}".to_string(),
                triggers: vec!["<|python_tag|>".to_string()],
                content_style: Default::default(),
                tool_call_ban_tokens: Vec::new(),
                reasoning_end: None,
            })
        })
    });
    if let Some(builder) = builder
        && let Some(tag) = builder
            .build_tool_call_format(&ToolCallFormatBuildContext {
                tool_choice,
                tools,
                parallel_tool_calls,
                schema_mode: StructuralTagSchemaMode::Auto,
                starts_in_reasoning: false,
            })
            .map_err(|error| ProcessorError::InvalidRequest(error.to_string()))?
    {
        return Ok(Some(ToolConstraint::StructuralTag(tag.to_string())));
    }

    let selected: Vec<_> = match tool_choice {
        ToolChoice::Required => tools.iter().collect(),
        ToolChoice::Named(name) => tools.iter().filter(|tool| tool.name == *name).collect(),
        _ => return Ok(None),
    };
    let schemas = selected
        .into_iter()
        .map(|tool| {
            serde_json::json!({
                "properties": {
                    "name": {"type": "string", "enum": [tool.name]},
                    "parameters": tool.parameters.clone().unwrap_or_else(|| {
                        serde_json::json!({"type": "object", "properties": {}})
                    }),
                },
                "required": ["name", "parameters"],
            })
        })
        .collect::<Vec<_>>();
    let items = if schemas.len() == 1 {
        schemas.into_iter().next().expect("one schema")
    } else {
        serde_json::json!({"type": "object", "anyOf": schemas})
    };
    let mut schema = serde_json::json!({"type": "array", "minItems": 1, "items": items});
    if parallel_tool_calls == Some(false) {
        schema["maxItems"] = serde_json::json!(1);
    }
    Ok(Some(ToolConstraint::JsonSchema(schema.to_string())))
}

/// Special tokens a parser leaves as content after a call; they are dropped.
pub(super) fn post_tool_terminal_markers(parser: Option<&str>) -> &'static [&'static str] {
    match parser.map(dynamo_tool_parser_name) {
        Some("qwen25") => &["<|im_end|>"],
        Some("glm47") => &["<|user|>", "<|endoftext|>", "<|observation|>"],
        _ => &[],
    }
}

pub(super) fn tool_call_delta(call: ChatCompletionMessageToolCallChunk) -> ChatToolCallDelta {
    ChatToolCallDelta {
        index: call.index,
        id: call.id,
        name: call
            .function
            .as_ref()
            .and_then(|function| function.name.clone()),
        arguments: call.function.and_then(|function| function.arguments),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, strict: bool) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: Some(strict),
        }
    }

    fn json_schema(constraint: Option<ToolConstraint>) -> serde_json::Value {
        match constraint {
            Some(ToolConstraint::JsonSchema(schema)) => serde_json::from_str(&schema).unwrap(),
            other => panic!("expected a JSON schema, got {other:?}"),
        }
    }

    #[test]
    fn tool_parser_aliases() {
        assert_eq!(dynamo_tool_parser_name("llama3"), "llama3_json");
        assert_eq!(dynamo_tool_parser_name("qwen"), "qwen25");
        assert_eq!(dynamo_tool_parser_name("glm45"), "glm47");
        assert_eq!(dynamo_tool_parser_name("deepseekv4"), "deepseekv4");
    }

    #[test]
    fn required_and_named_choices_build_a_call_array_schema() {
        let tools = [tool("get_weather", false), tool("get_time", false)];
        let schema = json_schema(
            tool_constraint(Some("llama3"), &ToolChoice::Required, &tools, Some(false)).unwrap(),
        );
        assert_eq!(
            (schema["type"].as_str(), schema["minItems"].as_u64()),
            (Some("array"), Some(1))
        );
        assert_eq!(schema["maxItems"], 1);
        assert_eq!(schema["items"]["anyOf"].as_array().unwrap().len(), 2);

        let named = ToolChoice::Named("get_time".into());
        let schema = json_schema(tool_constraint(Some("llama3"), &named, &tools, None).unwrap());
        assert_eq!(schema["items"]["properties"]["name"]["enum"][0], "get_time");
        assert!(schema["items"].get("anyOf").is_none());
        assert!(schema.get("maxItems").is_none());
    }

    #[test]
    fn strict_auto_llama_tool_uses_triggered_tags() {
        let tools = [tool("get_weather", true)];
        let Some(ToolConstraint::StructuralTag(tag)) =
            tool_constraint(Some("llama3"), &ToolChoice::Auto, &tools, None).unwrap()
        else {
            panic!("expected a structural tag");
        };
        let tag: serde_json::Value = serde_json::from_str(&tag).unwrap();
        assert_eq!(tag["type"], "structural_tag");
        assert_eq!(tag["format"]["type"], "triggered_tags");
        assert_eq!(tag["format"]["at_least_one"], false);
        assert_eq!(
            tag["format"]["tags"][0]["content"]["json_schema"]["required"][0],
            "city"
        );
    }

    #[test]
    fn unconstrained_and_invalid_choices() {
        let tools = [tool("get_weather", false)];
        assert_eq!(
            tool_constraint(Some("llama3"), &ToolChoice::Auto, &tools, None),
            Ok(None)
        );
        assert_eq!(
            tool_constraint(Some("llama3"), &ToolChoice::None, &[], None),
            Ok(None)
        );
        assert_eq!(
            tool_constraint(None, &ToolChoice::Auto, &tools, None),
            Ok(None)
        );
        // Validation runs even without a parser.
        let error = |parser, choice, tools| {
            tool_constraint(parser, choice, tools, None)
                .unwrap_err()
                .to_string()
        };
        assert!(error(None, &ToolChoice::Required, &[]).contains("required"));
        let missing = ToolChoice::Named("missing".into());
        assert!(error(None, &missing, &tools).contains("missing"));
        assert!(error(Some("not-a-parser"), &ToolChoice::Auto, &tools).contains("not supported"));
    }

    #[test]
    fn openai_tool_choices_map_to_dynamo() {
        assert_eq!(dynamo_tool_choice(&None), ToolChoice::Auto);
        let wire = |value| serde_json::from_value(serde_json::json!(value)).unwrap();
        assert_eq!(dynamo_tool_choice(&Some(wire("none"))), ToolChoice::None);
        assert_eq!(
            dynamo_tool_choice(&Some(wire("required"))),
            ToolChoice::Required
        );
        let named = serde_json::from_value(serde_json::json!(
            {"type": "function", "function": {"name": "get_weather"}}
        ))
        .unwrap();
        assert_eq!(
            dynamo_tool_choice(&Some(named)),
            ToolChoice::Named("get_weather".into())
        );
    }
}
