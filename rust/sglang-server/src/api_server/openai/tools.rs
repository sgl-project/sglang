//! Tool-choice constraints and streaming tool-call parsing.

use std::collections::HashMap;

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_parsers::{
    StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext,
    ToolChoice as DynamoToolChoice, ToolDefinition, TriggeredTagsConfig, detect_tool_call_start,
    find_tool_call_end_position, try_tool_call_parse_aggregate,
    try_tool_call_parse_aggregate_finalize,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FinishReason as OpenAIFinishReason, FunctionCall,
    FunctionCallStream, FunctionType, Role,
};
use futures::StreamExt;

use crate::message::{ChunkEvent, SamplingParams};

#[derive(Default)]
struct StreamingToolState {
    buffered: String,
    parsing_tool: bool,
    emitted_calls: usize,
}

fn partial_marker_suffix_len(text: &str, markers: &[String]) -> usize {
    markers
        .iter()
        .flat_map(|marker| text.char_indices().map(move |(start, _)| (marker, start)))
        .filter_map(|(marker, start)| {
            let suffix = &text[start..];
            (suffix.len() < marker.len() && marker.starts_with(suffix)).then_some(suffix.len())
        })
        .max()
        .unwrap_or(0)
}

pub(super) fn parse_streaming_tool_calls<S>(
    stream: S,
    parser: String,
    tools: Option<Vec<ToolDefinition>>,
    starts_immediately: bool,
) -> impl futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send
where
    S: futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send + 'static,
{
    let start_markers = get_tool_parser_map()
        .get(parser.as_str())
        .map(|config| {
            config
                .parser_config
                .tool_call_start_tokens()
                .into_iter()
                .filter(|marker| !marker.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    async_stream::stream! {
        let mut states = HashMap::<u32, StreamingToolState>::new();
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            let Some(response) = item.data.as_ref() else {
                yield item;
                continue;
            };
            if response.choices.is_empty() {
                yield item;
                continue;
            }

            let mut emitted = Vec::new();
            for choice in &response.choices {
                let Some(ChatCompletionMessageContent::Text(content)) =
                    choice.delta.content.as_ref()
                else {
                    if choice.finish_reason.is_some()
                        && let Some(state) = states.get_mut(&choice.index)
                        && !state.buffered.is_empty()
                    {
                        let text = std::mem::take(&mut state.buffered);
                        state.parsing_tool = false;
                        emitted.push(ChatChoiceStream {
                            index: choice.index,
                            delta: chat_delta(Some(text), choice.delta.role, None, None),
                            finish_reason: Some(if state.emitted_calls == 0 {
                                choice.finish_reason.unwrap_or(OpenAIFinishReason::Stop)
                            } else {
                                OpenAIFinishReason::ToolCalls
                            }),
                            logprobs: choice.logprobs.clone(),
                        });
                        continue;
                    }
                    let mut choice = choice.clone();
                    if choice.finish_reason.is_some()
                        && states
                            .get(&choice.index)
                            .is_some_and(|state| state.emitted_calls != 0)
                    {
                        choice.finish_reason = Some(OpenAIFinishReason::ToolCalls);
                    }
                    emitted.push(choice);
                    continue;
                };
                let state = states.entry(choice.index).or_insert_with(|| StreamingToolState {
                    parsing_tool: starts_immediately,
                    ..Default::default()
                });
                state.buffered.push_str(content);

                if !state.parsing_tool {
                    if let Some(position) = start_markers
                        .iter()
                        .filter_map(|marker| state.buffered.find(marker))
                        .min()
                    {
                        if position != 0 {
                            let prefix = state.buffered[..position].to_owned();
                            state.buffered.drain(..position);
                            emitted.push(ChatChoiceStream {
                                index: choice.index,
                                delta: chat_delta(Some(prefix), choice.delta.role, None, None),
                                finish_reason: None,
                                logprobs: choice.logprobs.clone(),
                            });
                        }
                        state.parsing_tool = true;
                    } else if start_markers.is_empty()
                        && detect_tool_call_start(&state.buffered, Some(&parser)).unwrap_or(false)
                    {
                        state.parsing_tool = true;
                    } else {
                        let held = if choice.finish_reason.is_some() {
                            0
                        } else {
                            partial_marker_suffix_len(&state.buffered, &start_markers)
                        };
                        let safe_len = state.buffered.len().saturating_sub(held);
                        if safe_len != 0 {
                            let text = state.buffered[..safe_len].to_owned();
                            state.buffered.drain(..safe_len);
                            emitted.push(ChatChoiceStream {
                                index: choice.index,
                                delta: chat_delta(Some(text), choice.delta.role, None, None),
                                finish_reason: choice.finish_reason.map(|reason| {
                                    if state.emitted_calls == 0 {
                                        reason
                                    } else {
                                        OpenAIFinishReason::ToolCalls
                                    }
                                }),
                                logprobs: choice.logprobs.clone(),
                            });
                        }
                    }
                }

                if state.parsing_tool {
                    if choice.finish_reason.is_none()
                        && find_tool_call_end_position(&state.buffered, Some(&parser)).is_none()
                    {
                        continue;
                    }
                    let parsed = if choice.finish_reason.is_some() {
                        try_tool_call_parse_aggregate_finalize(
                            &state.buffered,
                            Some(&parser),
                            tools.as_deref(),
                        )
                        .await
                    } else {
                        try_tool_call_parse_aggregate(
                            &state.buffered,
                            Some(&parser),
                            tools.as_deref(),
                        )
                        .await
                    };
                    if let Ok((calls, normal)) = parsed
                        && !calls.is_empty()
                    {
                        let tool_calls = calls
                            .into_iter()
                            .enumerate()
                            .map(|(index, call)| ChatCompletionMessageToolCallChunk {
                                index: u32::try_from(state.emitted_calls + index)
                                    .unwrap_or(u32::MAX),
                                id: Some(call.id),
                                r#type: Some(FunctionType::Function),
                                function: Some(FunctionCallStream {
                                    name: Some(call.function.name),
                                    arguments: Some(call.function.arguments),
                                }),
                            })
                            .collect::<Vec<_>>();
                        state.emitted_calls += tool_calls.len();
                        state.buffered.clear();
                        state.parsing_tool = false;
                        emitted.push(ChatChoiceStream {
                            index: choice.index,
                            delta: chat_delta(
                                normal.filter(|text| !text.is_empty()),
                                choice.delta.role,
                                Some(tool_calls),
                                None,
                            ),
                            finish_reason: choice
                                .finish_reason
                                .map(|_| OpenAIFinishReason::ToolCalls),
                            logprobs: choice.logprobs.clone(),
                        });
                    } else if choice.finish_reason.is_some() {
                        let text = std::mem::take(&mut state.buffered);
                        state.parsing_tool = false;
                        emitted.push(ChatChoiceStream {
                            index: choice.index,
                            delta: chat_delta(
                                (!text.is_empty()).then_some(text),
                                choice.delta.role,
                                None,
                                None,
                            ),
                            finish_reason: choice.finish_reason.map(|reason| {
                                if state.emitted_calls == 0 {
                                    reason
                                } else {
                                    OpenAIFinishReason::ToolCalls
                                }
                            }),
                            logprobs: choice.logprobs.clone(),
                        });
                    }
                } else if choice.finish_reason.is_some()
                    && !emitted.iter().any(|emission| emission.index == choice.index)
                {
                    emitted.push(choice.clone());
                }
            }

            for choice in emitted {
                let mut output = response.clone();
                output.choices = vec![choice];
                yield Annotated {
                    data: Some(output),
                    id: item.id.clone(),
                    event: item.event.clone(),
                    comment: item.comment.clone(),
                    error: item.error.clone(),
                };
            }
        }
    }
}

pub(super) fn dynamo_parser_name(parser: &str) -> &str {
    match parser {
        "llama3" => "llama3_json",
        // SGLang canonicalizes these legacy CLI names in the opposite
        // direction; Dynamo 5.0 still registers the older parser keys.
        "qwen" => "qwen25",
        "glm" | "glm45" => "glm47",
        other => other,
    }
}

pub(super) fn apply_tool_constraint(
    sampling: &mut SamplingParams,
    parser: &str,
    tool_choice: &DynamoToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
) -> Result<(), String> {
    if *tool_choice == DynamoToolChoice::None {
        return Ok(());
    }
    if *tool_choice == DynamoToolChoice::Required && tools.is_empty() {
        return Err("tool_choice is \"required\" but tools is empty".into());
    }
    if let DynamoToolChoice::Named(name) = tool_choice
        && !tools.iter().any(|tool| &tool.name == name)
    {
        return Err(format!(
            "tool named \"{name}\" in tool_choice is not present in tools"
        ));
    }

    let parser = dynamo_parser_name(parser);
    let config = get_tool_parser_map()
        .get(parser)
        .ok_or_else(|| format!("tool-call parser `{parser}` is not supported by Dynamo"))?;
    let builder = config.structural_tag_builder.clone().or_else(|| {
        (parser == "llama3_json"
            && *tool_choice == DynamoToolChoice::Auto
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
            .map_err(|error| error.to_string())?
    {
        sampling.structural_tag = Some(tag.to_string());
        return Ok(());
    }

    if matches!(
        tool_choice,
        DynamoToolChoice::Required | DynamoToolChoice::Named(_)
    ) {
        let selected = match tool_choice {
            DynamoToolChoice::Named(name) => tools
                .iter()
                .filter(|tool| tool.name == *name)
                .collect::<Vec<_>>(),
            _ => tools.iter().collect(),
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
        let mut schema = serde_json::json!({
            "type": "array",
            "minItems": 1,
            "items": items,
        });
        if parallel_tool_calls == Some(false) {
            schema["maxItems"] = serde_json::json!(1);
        }
        sampling.json_schema = Some(schema.to_string());
    }
    Ok(())
}

#[allow(deprecated)]
pub(super) fn chat_delta(
    content: Option<String>,
    role: Option<Role>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    reasoning_content: Option<String>,
) -> ChatCompletionStreamResponseDelta {
    ChatCompletionStreamResponseDelta {
        content: content.map(ChatCompletionMessageContent::Text),
        function_call: None,
        tool_calls,
        role,
        refusal: None,
        reasoning_content,
    }
}

pub(super) async fn parse_chat_tool_calls(
    content: String,
    parser: Option<&str>,
    tools: Option<&[ToolDefinition]>,
    parallel_tool_calls: bool,
) -> (String, Option<Vec<ChatCompletionMessageToolCall>>) {
    let Some(parser) = parser else {
        return (content, None);
    };
    let parser = dynamo_parser_name(parser);
    match try_tool_call_parse_aggregate_finalize(&content, Some(parser), tools).await {
        Ok((mut calls, normal)) if !calls.is_empty() => {
            if !parallel_tool_calls {
                calls.truncate(1);
            }
            (
                normal.unwrap_or_default(),
                Some(
                    calls
                        .into_iter()
                        .map(|call| ChatCompletionMessageToolCall {
                            id: call.id,
                            r#type: FunctionType::Function,
                            function: FunctionCall {
                                name: call.function.name,
                                arguments: call.function.arguments,
                            },
                        })
                        .collect(),
                ),
            )
        }
        _ => (content, None),
    }
}

pub(super) fn chat_finish_reason(output: &ChunkEvent) -> Option<OpenAIFinishReason> {
    let kind = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.kind_name());
    kind.map(|kind| match kind {
        "length" => OpenAIFinishReason::Length,
        "content_filter" => OpenAIFinishReason::ContentFilter,
        _ => OpenAIFinishReason::Stop,
    })
}
