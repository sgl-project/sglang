//! The tool schemas shared by output parsing and constrained generation.

use std::collections::{HashMap, HashSet};
use std::pin::Pin;

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
use dynamo_parsers::{
    CalledFunction, StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext,
    ToolCallResponse, ToolCallType, ToolChoice, ToolDefinition, TriggeredTagsConfig,
    try_tool_call_parse_aggregate_finalize,
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk, ChatCompletionRequestMessage,
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionStreamResponse,
    FinishReason, FunctionCallStream, FunctionObject, FunctionType,
};
use futures::{Stream, StreamExt};
use serde_json::Value;

use super::ChatToolCallDelta;
use crate::ProcessorError;
use crate::schema_types::normalize_json_schema_types;
use crate::tool_call::{ToolDetector, tool_call_id, tool_detector};

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
/// `reasoning_parser`: one owns the think tokens, so a tag does not exclude
/// them from free text (Python's `strip_structural_tag_excludes`).
pub fn tool_constraint(
    parser: Option<&str>,
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    reasoning_parser: bool,
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
    if parser == "deepseekv41" {
        let tag = deepseek_v41_tag(tool_choice, tools, parallel_tool_calls, reasoning_parser);
        return Ok(tag.map(ToolConstraint::StructuralTag));
    }
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

type ChunkStream =
    Pin<Box<dyn Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send>>;

/// The tools as the request lists them, which SGLang's detectors read.
fn tool_values(tools: Option<&[ToolDefinition]>) -> Vec<Value> {
    tools
        .into_iter()
        .flatten()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {"name": tool.name, "parameters": tool.parameters},
            })
        })
        .collect()
}

/// Parse tool calls out of OpenAI stream chunks: SGLang's port for the names
/// `models/` lists, else Dynamo's jail.
pub fn tool_call_stream<S>(
    parser: &str,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    tools: Option<Vec<ToolDefinition>>,
    uses_tool_call_structural_tag: bool,
    stream: S,
) -> ChunkStream
where
    S: Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tool_values = tool_values(tools.as_deref());
    let Some(_) = tool_detector(parser, &tool_values) else {
        let parser = dynamo_tool_parser_name(parser).to_owned();
        return Box::pin(apply_tool_calling_jail(
            Some(parser),
            tool_choice,
            tools,
            uses_tool_call_structural_tag,
            stream,
        ));
    };
    let parser = parser.to_owned();
    Box::pin(async_stream::stream! {
        let mut detectors: HashMap<u32, Box<dyn ToolDetector>> = HashMap::new();
        let mut called = HashSet::new();
        futures::pin_mut!(stream);
        while let Some(mut item) = stream.next().await {
            for choice in item.data.iter_mut().flat_map(|chunk| chunk.choices.iter_mut()) {
                let detector = detectors
                    .entry(choice.index)
                    .or_insert_with(|| tool_detector(&parser, &tool_values).expect("checked"));
                let text = match choice.delta.content.take() {
                    Some(ChatCompletionMessageContent::Text(text)) => text,
                    _ => String::new(),
                };
                let (normal, calls) = detector.parse_stream(&text, choice.finish_reason.is_some());
                choice.delta.content =
                    (!normal.is_empty()).then_some(ChatCompletionMessageContent::Text(normal));
                if !calls.is_empty() {
                    called.insert(choice.index);
                    // Python streams an id and name only for a non-empty name.
                    let calls = calls.into_iter().map(|call| {
                        let named = !call.name.is_empty();
                        ChatCompletionMessageToolCallChunk {
                            index: call.tool_index as u32,
                            id: named.then(tool_call_id),
                            r#type: Some(FunctionType::Function),
                            function: Some(FunctionCallStream {
                                name: named.then_some(call.name),
                                arguments: Some(call.parameters),
                            }),
                        }
                    });
                    choice.delta.tool_calls = Some(calls.collect());
                }
                if choice.finish_reason == Some(FinishReason::Stop) && called.contains(&choice.index) {
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                }
            }
            yield item;
        }
    })
}

/// Parse a whole output's tool calls into `(calls, normal_text)`: SGLang's
/// port for the names `models/` lists, else Dynamo's.
pub async fn parse_tool_calls(
    parser: &str,
    text: &str,
    tools: Option<&[ToolDefinition]>,
) -> Result<(Vec<ToolCallResponse>, Option<String>), String> {
    let Some(detector) = tool_detector(parser, &tool_values(tools)) else {
        let parser = dynamo_tool_parser_name(parser);
        return try_tool_call_parse_aggregate_finalize(text, Some(parser), tools)
            .await
            .map_err(|error| error.to_string());
    };
    if !detector.has_tool_call(text) {
        return Ok((Vec::new(), Some(text.to_owned())));
    }
    let (normal, calls) = detector.parse_non_stream(text);
    let calls = calls.into_iter().map(|call| ToolCallResponse {
        id: tool_call_id(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: call.name,
            arguments: call.parameters,
        },
    });
    Ok((calls.collect(), Some(normal)))
}

/// xgrammar's `deepseek_v4_1` tag as SGLang builds it: `required` and named
/// choices, and `auto` once a tool is strict; otherwise `auto` is
/// unconstrained.
fn deepseek_v41_tag(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    reasoning_parser: bool,
) -> Option<String> {
    let invoke = |tool: &ToolDefinition| {
        // xgrammar's `_get_function_parameters`: only a strict tool's schema
        // constrains its arguments. `_validate_request` has normalized the
        // schema types in place, as the prompt shows them.
        let schema = match (tool.strict, tool.parameters.clone()) {
            (Some(true), Some(mut schema)) => {
                normalize_json_schema_types(&mut schema);
                schema
            }
            _ => Value::Bool(true),
        };
        serde_json::json!({
            "type": "tag",
            "begin": format!("<｜DSML｜ invoke name=\"{}\">\n", tool.name),
            "content": {
                "type": "json_schema",
                "json_schema": schema,
                "style": "deepseek_v4_1_xml",
                "any_order": false,
                "max_whitespace_cnt": null
            },
            "end": "</｜DSML｜ invoke>\n"
        })
    };
    let stop_after_first = parallel_tool_calls == Some(false);
    let all_calls = || {
        serde_json::json!({
            "type": "tags_with_separator",
            "tags": tools.iter().map(invoke).collect::<Vec<_>>(),
            "separator": "",
            "at_least_one": true,
            "stop_after_first": stop_after_first
        })
    };
    let calls = match tool_choice {
        ToolChoice::Required => all_calls(),
        ToolChoice::Named(name) => invoke(tools.iter().find(|tool| &tool.name == name)?),
        ToolChoice::Auto if tools.iter().any(|tool| tool.strict == Some(true)) => {
            let format = serde_json::json!({
                "type": "triggered_tags",
                "triggers": ["<｜DSML｜ calls>"],
                "tags": [{
                    "type": "tag",
                    "begin": "<｜DSML｜ calls>\n",
                    "content": all_calls(),
                    "end": "</｜DSML｜ calls>"
                }],
                "at_least_one": false,
                "stop_after_first": stop_after_first,
                "excludes": if reasoning_parser { vec![] } else { vec!["<think>", "</think>"] }
            });
            return Some(
                serde_json::json!({"type": "structural_tag", "format": format}).to_string(),
            );
        }
        _ => return None,
    };
    let elements = [
        serde_json::json!({"type": "const_string", "value": "\n\n<｜DSML｜ calls>\n"}),
        calls,
        serde_json::json!({"type": "const_string", "value": "</｜DSML｜ calls>"}),
    ];
    let format = serde_json::json!({"type": "sequence", "elements": elements});
    Some(serde_json::json!({"type": "structural_tag", "format": format}).to_string())
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

    /// The tags SGLang sends xgrammar for each choice, from
    /// `FunctionCallParser(tools, "deepseekv41").get_structure_constraint`,
    /// excludes stripped for a reasoning parser unless noted: a non-strict
    /// tool's arguments are free (`json_schema: true`), and `auto` is
    /// constrained only once a tool is strict.
    #[test]
    fn deepseek_v41_constraint_matches_xgrammar() {
        let tag = |choice: ToolChoice, tools: &[ToolDefinition], parallel: Option<bool>| {
            match tool_constraint(Some("deepseekv41"), &choice, tools, parallel, true).unwrap() {
                Some(ToolConstraint::StructuralTag(tag)) => {
                    serde_json::from_str::<Value>(&tag).ok()
                }
                None => None,
                other => panic!("expected a structural tag, got {other:?}"),
            }
        };
        let python = |json: &str| serde_json::from_str::<Value>(json).ok();
        let lenient = |name: &str| {
            let mut tool = tool(name, false);
            tool.strict = None;
            tool
        };

        assert_eq!(tag(ToolChoice::Auto, &[lenient("get_weather")], None), None);
        assert_eq!(
            tag(ToolChoice::Required, &[lenient("get_weather")], Some(false)),
            python(
                r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"const_string","value":"\n\n<｜DSML｜ calls>\n"},{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<｜DSML｜ invoke name=\"get_weather\">\n","content":{"type":"json_schema","json_schema":true,"style":"deepseek_v4_1_xml","any_order":false,"max_whitespace_cnt":null},"end":"</｜DSML｜ invoke>\n"}],"separator":"","at_least_one":true,"stop_after_first":true},{"type":"const_string","value":"</｜DSML｜ calls>"}]}}"#
            )
        );
        let named = ToolChoice::Named("get_weather".into());
        let tools = [tool("get_weather", false), tool("t", true)];
        assert_eq!(
            tag(named, &tools, Some(true)),
            python(
                r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"const_string","value":"\n\n<｜DSML｜ calls>\n"},{"type":"tag","begin":"<｜DSML｜ invoke name=\"get_weather\">\n","content":{"type":"json_schema","json_schema":true,"style":"deepseek_v4_1_xml","any_order":false,"max_whitespace_cnt":null},"end":"</｜DSML｜ invoke>\n"},{"type":"const_string","value":"</｜DSML｜ calls>"}]}}"#
            )
        );
        let tools = [tool("get_weather", true), lenient("t")];
        assert_eq!(
            tag(ToolChoice::Auto, &tools, None),
            python(
                r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<｜DSML｜ calls>"],"tags":[{"type":"tag","begin":"<｜DSML｜ calls>\n","content":{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<｜DSML｜ invoke name=\"get_weather\">\n","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"style":"deepseek_v4_1_xml","any_order":false,"max_whitespace_cnt":null},"end":"</｜DSML｜ invoke>\n"},{"type":"tag","begin":"<｜DSML｜ invoke name=\"t\">\n","content":{"type":"json_schema","json_schema":true,"style":"deepseek_v4_1_xml","any_order":false,"max_whitespace_cnt":null},"end":"</｜DSML｜ invoke>\n"}],"separator":"","at_least_one":true,"stop_after_first":false},"end":"</｜DSML｜ calls>"}],"at_least_one":false,"stop_after_first":false,"excludes":[]}}"#
            )
        );

        // Without a reasoning parser, xgrammar's think excludes stay.
        let tools = [tool("get_weather", true)];
        let Some(ToolConstraint::StructuralTag(tag)) = tool_constraint(
            Some("deepseekv41"),
            &ToolChoice::Auto,
            &tools,
            Some(false),
            false,
        )
        .unwrap() else {
            panic!("expected a structural tag");
        };
        assert_eq!(
            serde_json::from_str::<Value>(&tag).ok(),
            python(
                r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<｜DSML｜ calls>"],"tags":[{"type":"tag","begin":"<｜DSML｜ calls>\n","content":{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<｜DSML｜ invoke name=\"get_weather\">\n","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"style":"deepseek_v4_1_xml","any_order":false,"max_whitespace_cnt":null},"end":"</｜DSML｜ invoke>\n"}],"separator":"","at_least_one":true,"stop_after_first":true},"end":"</｜DSML｜ calls>"}],"at_least_one":false,"stop_after_first":true,"excludes":["<think>","</think>"]}}"#
            )
        );
    }

    /// A DB-style alias the prompt shows as `string` is `string` in a strict
    /// tool's constraint too; xgrammar rejects `varchar`.
    #[test]
    fn deepseek_v41_constraint_normalizes_schema_types() {
        let tools = [ToolDefinition {
            name: "lookup".into(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"id": {"type": "varchar(255)"}, "n": {"type": "bigint"}}
            })),
            strict: Some(true),
        }];
        let named = ToolChoice::Named("lookup".into());
        let Some(ToolConstraint::StructuralTag(tag)) =
            tool_constraint(Some("deepseekv41"), &named, &tools, None, true).unwrap()
        else {
            panic!("expected a structural tag");
        };
        let tag: Value = serde_json::from_str(&tag).unwrap();
        let schema = &tag["format"]["elements"][1]["content"]["json_schema"];
        assert_eq!(schema["properties"]["id"]["type"], "string");
        assert_eq!(schema["properties"]["n"]["type"], "integer");
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
            tool_constraint(
                Some("llama3"),
                &ToolChoice::Required,
                &tools,
                Some(false),
                true,
            )
            .unwrap(),
        );
        assert_eq!(
            (schema["type"].as_str(), schema["minItems"].as_u64()),
            (Some("array"), Some(1))
        );
        assert_eq!(schema["maxItems"], 1);
        assert_eq!(schema["items"]["anyOf"].as_array().unwrap().len(), 2);

        let named = ToolChoice::Named("get_time".into());
        let schema =
            json_schema(tool_constraint(Some("llama3"), &named, &tools, None, true).unwrap());
        assert_eq!(schema["items"]["properties"]["name"]["enum"][0], "get_time");
        assert!(schema["items"].get("anyOf").is_none());
        assert!(schema.get("maxItems").is_none());
    }

    #[test]
    fn strict_auto_llama_tool_uses_triggered_tags() {
        let tools = [tool("get_weather", true)];
        let Some(ToolConstraint::StructuralTag(tag)) =
            tool_constraint(Some("llama3"), &ToolChoice::Auto, &tools, None, true).unwrap()
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
            tool_constraint(Some("llama3"), &ToolChoice::Auto, &tools, None, true),
            Ok(None)
        );
        assert_eq!(
            tool_constraint(Some("llama3"), &ToolChoice::None, &[], None, true),
            Ok(None)
        );
        assert_eq!(
            tool_constraint(None, &ToolChoice::Auto, &tools, None, true),
            Ok(None)
        );
        // Validation runs even without a parser.
        let error = |parser, choice, tools| {
            tool_constraint(parser, choice, tools, None, true)
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
