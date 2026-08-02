//! Tool-choice constraints and streaming tool-call parsing.
//!
//! Two complementary mechanisms, mirroring the Python frontend
//! (`serving_chat.py` + dynamo-parsers):
//!
//! - [`apply_tool_constraint`] turns `tool_choice` into a sampling constraint
//!   *before* submission: a `structural_tag` when the parser supports one (or
//!   when strict tools need the llama3 triggered-tag format), otherwise a
//!   `json_schema` array restricting the output to tool calls. It validates
//!   `tool_choice`/`tools` agreement first.
//! - [`parse_streaming_tool_calls`] / [`parse_chat_tool_calls`] strip the
//!   model's tool-call markers out of the *output*: the streaming form buffers
//!   text until a complete call is available (so a marker split across chunks
//!   never leaks), the unary form parses the finished content. Both return the
//!   calls alongside any non-tool text.
//!
//! [`dynamo_parser_name`] canonicalizes the SGLang CLI parser names onto the
//! dynamo-parsers registry keys, [`chat_delta`] builds the stream deltas these
//! paths emit, and [`chat_finish_reason`] maps the scheduler's finish reason
//! onto the OpenAI wire values.
//!
//! # Test coverage
//!
//! The `tests` module below covers each item directly: the marker-suffix
//! holdback ([`partial_marker_suffix_len`]), parser-name canonicalization,
//! every `tool_choice` branch of [`apply_tool_constraint`] (required/named
//! schemas, the strict llama3 triggered tag, validation failures, no-ops),
//! the streaming state machine (whole-call buffering, split markers, release
//! at done, `tool_calls` finish rewriting), unary parsing (marker formats and
//! the no-parser passthrough), and the finish-reason mapping. The end-to-end
//! wiring is covered where the pieces are assembled in `chat.rs` (streaming
//! deltas, role framing, and reasoning).

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
    ChatCompletionToolChoiceOption, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason, FunctionCall, FunctionCallStream, FunctionType, Role,
};
use futures::StreamExt;

use crate::message::{ChunkEvent, SamplingParams};

/// Per-choice state of the streaming parser: the text buffered since the last
/// emission, whether the current buffer has entered a tool-call marker, and
/// how many calls were already emitted (which rewrites the terminal
/// `finish_reason` to `tool_calls`).
#[derive(Default)]
struct StreamingToolState {
    buffered: String,
    parsing_tool: bool,
    emitted_calls: usize,
}

/// How many trailing characters of `text` form a proper prefix of one of the
/// start markers. Those characters must be held back until the next chunk (or
/// released at `done`), so a marker split across chunks is never flushed into
/// the output as literal text. Returns 0 when nothing is a partial marker.
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

/// Map a stream of chat deltas through the Dynamo tool-call parser, splitting
/// tool calls out of `content` into `tool_calls` deltas.
///
/// `parser` is the configured tool-call parser name (see
/// [`dynamo_parser_name`]); `tools` are the request's definitions, used to
/// recover from malformed calls. `starts_immediately` is true when the request
/// carried no structural-tag constraint, so generation can begin in the middle
/// of a call and parsing must start before any start marker arrives.
///
/// The stream is buffered until a complete call parses — each emitted item
/// carries either plain text (possibly truncated at a partial start marker)
/// or one or more tool-call chunks. When the underlying stream finished
/// (`finish_reason` set), the terminal item's reason becomes `tool_calls` if
/// any call was emitted, and an incomplete marker is released as literal text.
///
/// `parser` must already be canonicalized (the callers apply
/// [`dynamo_parser_name`] before entering this function).
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
                    let partial_marker_len = if choice.finish_reason.is_some() {
                        0
                    } else {
                        partial_marker_suffix_len(&state.buffered, &start_markers)
                    };
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
                    } else if partial_marker_len == 0
                        && detect_tool_call_start(&state.buffered, Some(&parser))
                            .unwrap_or(false)
                    {
                        state.parsing_tool = true;
                    } else {
                        let safe_len = state.buffered.len().saturating_sub(partial_marker_len);
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

/// Canonicalize a tool-call parser name onto the dynamo-parsers registry keys.
///
/// SGLang canonicalizes these legacy CLI names in the opposite direction;
/// Dynamo 5.0 still registers the older parser keys.
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

/// Map the OpenAI wire `tool_choice` onto the Dynamo choice. Shared by the
/// chat and responses handlers; a missing/auto choice reads as `Auto`.
pub(super) fn dynamo_tool_choice(
    choice: &Option<ChatCompletionToolChoiceOption>,
) -> DynamoToolChoice {
    match choice {
        Some(ChatCompletionToolChoiceOption::None) => DynamoToolChoice::None,
        Some(ChatCompletionToolChoiceOption::Required) => DynamoToolChoice::Required,
        Some(ChatCompletionToolChoiceOption::Named(choice)) => {
            DynamoToolChoice::Named(choice.function.name.clone())
        }
        Some(ChatCompletionToolChoiceOption::Auto) | None => DynamoToolChoice::Auto,
    }
}

/// Validate `tool_choice` against `tools`, then — when a tool-call `parser`
/// is configured — turn it into a sampling constraint, mirroring Python's
/// `serving_chat` logic. Validation runs even without a parser, so an invalid
/// choice (required/named with nothing to select) is rejected before
/// submission in every mode.
///
/// Prefers a structural-tag constraint: the parser's own registered builder,
/// or — for llama3 with strict tools under `auto` — a triggered-tag builder
/// so the model emits calls in the exact `<|python_tag|>` format. Otherwise
/// `required`/`named` choices fall back to a JSON-schema array constraining
/// the output to `{"name", "parameters"}` objects (`maxItems: 1` when
/// `parallel_tool_calls` is false).
pub(super) fn apply_tool_constraint(
    sampling: &mut SamplingParams,
    parser: Option<&str>,
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

    let Some(parser) = parser else {
        return Ok(()); // validation only
    };
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

/// Build a chat-streaming delta carrying any of the optional columns.
///
/// The deprecated `function_call` field stays `None` — tool calls go through
/// the `tool_calls` array.
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

/// Parse tool calls out of a completed (unary) generation's content.
///
/// Returns `(content, None)` when no parser is configured or no call parses —
/// the content passes through untouched. With a parser, a successful parse
/// returns the leftover non-tool text and the calls; `parallel_tool_calls`
/// false truncates the batch to the first call, mirroring Python.
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

/// Map the scheduler's finish kind onto the OpenAI wire values. Length and
/// content-filter keep their names; everything else (including a bare abort)
/// reports as `stop`, matching Python's fallback.
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

#[cfg(test)]
mod tests {
    use super::{
        apply_tool_constraint, chat_delta, chat_finish_reason, dynamo_parser_name,
        dynamo_tool_choice, parse_chat_tool_calls, parse_streaming_tool_calls,
        partial_marker_suffix_len,
    };
    use crate::message::{ChunkEvent, SamplingParams};
    use dynamo_parsers::tool_calling::jail::Annotated;
    use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
    use dynamo_protocols::types::CreateChatCompletionStreamResponse as StreamResponse;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionNamedToolChoice, ChatCompletionToolChoiceOption, ChatCompletionToolType,
        FinishReason as OpenAIFinishReason, FunctionCallStream, FunctionName, FunctionType, Role,
    };
    use futures::{StreamExt, stream};

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

    fn stream_item(text: &str, finish: Option<OpenAIFinishReason>) -> Annotated<StreamResponse> {
        Annotated {
            data: Some(StreamResponse {
                id: "chatcmpl-test".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: chat_delta(Some(text.into()), Some(Role::Assistant), None, None),
                    finish_reason: finish,
                    logprobs: None,
                }],
                created: 1,
                model: "model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    /// A terminal chunk with no text — what the upstream path emits for an
    /// empty `Done` frame (content `None`, not an empty string).
    fn stream_done(finish: OpenAIFinishReason) -> Annotated<StreamResponse> {
        Annotated {
            data: Some(StreamResponse {
                id: "chatcmpl-test".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: chat_delta(None, Some(Role::Assistant), None, None),
                    finish_reason: Some(finish),
                    logprobs: None,
                }],
                created: 1,
                model: "model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn choice(item: &Annotated<StreamResponse>) -> &ChatChoiceStream {
        item.data.as_ref().unwrap().choices.first().unwrap()
    }

    fn delta_text(item: &Annotated<StreamResponse>) -> String {
        match choice(item).delta.content.as_ref().unwrap() {
            ChatCompletionMessageContent::Text(text) => text.clone(),
            _ => panic!("expected a text delta"),
        }
    }

    async fn parse(
        items: Vec<Annotated<StreamResponse>>,
        parser: &str,
        starts_immediately: bool,
    ) -> Vec<Annotated<StreamResponse>> {
        parse_streaming_tool_calls(stream::iter(items), parser.into(), None, starts_immediately)
            .collect()
            .await
    }

    #[test]
    fn partial_marker_suffix_len_holds_only_a_real_prefix() {
        let markers = vec!["<|python_tag|>".to_string(), "<tool_call>".to_string()];
        // A trailing "<|python_" (9 chars) is a proper prefix of the marker.
        assert_eq!(partial_marker_suffix_len("Before <|python_", &markers), 9);
        // A complete marker is not a partial one.
        assert_eq!(partial_marker_suffix_len("a<|python_tag|>b", &markers), 0);
        // A suffix that matches both candidates picks the longest.
        assert_eq!(partial_marker_suffix_len("x<tool_", &markers), 6);
        // Nothing marker-shaped.
        assert_eq!(partial_marker_suffix_len("plain text", &markers), 0);
        assert_eq!(partial_marker_suffix_len("", &markers), 0);
        assert_eq!(partial_marker_suffix_len("x", &[]), 0);
    }

    #[test]
    fn dynamo_parser_name_canonicalizes_cli_names() {
        assert_eq!(dynamo_parser_name("llama3"), "llama3_json");
        assert_eq!(dynamo_parser_name("qwen"), "qwen25");
        assert_eq!(dynamo_parser_name("glm"), "glm47");
        assert_eq!(dynamo_parser_name("glm45"), "glm47");
        assert_eq!(dynamo_parser_name("qwen25"), "qwen25");
    }

    #[test]
    fn required_tool_choice_builds_python_compatible_constraint() {
        let mut sampling = SamplingParams::default();
        apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Required,
            &[tool("get_weather", true)],
            Some(false),
        )
        .unwrap();
        let schema: serde_json::Value =
            serde_json::from_str(sampling.json_schema.as_deref().unwrap()).unwrap();
        assert_eq!(schema["type"], "array");
        assert_eq!(schema["minItems"], 1);
        assert_eq!(schema["maxItems"], 1);
        assert_eq!(
            schema["items"]["properties"]["name"]["enum"][0],
            "get_weather"
        );
    }

    #[test]
    fn required_choice_without_parallel_flag_has_no_max_items() {
        let mut sampling = SamplingParams::default();
        apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Required,
            &[tool("get_weather", false)],
            None,
        )
        .unwrap();
        let schema: serde_json::Value =
            serde_json::from_str(sampling.json_schema.as_deref().unwrap()).unwrap();
        assert_eq!(schema["type"], "array");
        assert!(schema.get("maxItems").is_none());
    }

    #[test]
    fn named_tool_choice_restricts_the_schema_enum() {
        let tools = [tool("get_weather", false), tool("get_time", false)];
        let mut sampling = SamplingParams::default();
        apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Named("get_time".into()),
            &tools,
            None,
        )
        .unwrap();
        let schema: serde_json::Value =
            serde_json::from_str(sampling.json_schema.as_deref().unwrap()).unwrap();
        // One candidate → a single schema, not an anyOf.
        assert_eq!(schema["items"]["properties"]["name"]["enum"][0], "get_time");
        assert!(schema["items"].get("anyOf").is_none());
    }

    #[test]
    fn strict_auto_llama_tool_uses_python_compatible_constraint() {
        let mut sampling = SamplingParams::default();
        apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Auto,
            &[tool("get_weather", true)],
            None,
        )
        .unwrap();
        let schema: serde_json::Value =
            serde_json::from_str(sampling.structural_tag.as_deref().unwrap()).unwrap();
        assert_eq!(schema["type"], "structural_tag");
        assert_eq!(schema["format"]["type"], "triggered_tags");
        assert_eq!(schema["format"]["at_least_one"], false);
        assert_eq!(
            schema["format"]["tags"][0]["content"]["json_schema"]["required"][0],
            "city"
        );
    }

    #[test]
    fn auto_without_strict_tools_stays_unconstrained() {
        let mut sampling = SamplingParams::default();
        apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Auto,
            &[tool("get_weather", false)],
            None,
        )
        .unwrap();
        assert!(sampling.json_schema.is_none());
        assert!(sampling.structural_tag.is_none());
    }

    #[test]
    fn tool_choice_none_is_a_no_op() {
        let mut sampling = SamplingParams::default();
        apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::None,
            &[],
            None,
        )
        .unwrap();
        assert!(sampling.json_schema.is_none());
        assert!(sampling.structural_tag.is_none());
    }

    #[test]
    fn invalid_tool_choices_are_rejected_before_submission() {
        let mut sampling = SamplingParams::default();
        let error = apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Required,
            &[],
            None,
        )
        .unwrap_err();
        assert!(error.contains("required"));

        let error = apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Named("missing".into()),
            &[tool("get_weather", false)],
            None,
        )
        .unwrap_err();
        assert!(error.contains("missing"));
    }

    /// Validation runs even without a parser (the handler calls this in every
    /// mode), so an invalid choice is rejected before submission there too.
    #[test]
    fn missing_parser_still_validates_the_tool_choice() {
        let mut sampling = SamplingParams::default();
        let error =
            apply_tool_constraint(&mut sampling, None, &DynamoToolChoice::Required, &[], None)
                .unwrap_err();
        assert!(error.contains("required"));
        let error = apply_tool_constraint(
            &mut sampling,
            None,
            &DynamoToolChoice::Named("missing".into()),
            &[tool("get_weather", false)],
            None,
        )
        .unwrap_err();
        assert!(error.contains("missing"));
        // A valid choice with no parser stays unconstrained.
        apply_tool_constraint(
            &mut sampling,
            None,
            &DynamoToolChoice::Auto,
            &[tool("get_weather", false)],
            None,
        )
        .unwrap();
        assert!(sampling.json_schema.is_none());
        assert!(sampling.structural_tag.is_none());
    }

    #[test]
    fn dynamo_tool_choice_maps_the_openai_wire_values() {
        let named = |name: &str| {
            Some(ChatCompletionToolChoiceOption::Named(
                ChatCompletionNamedToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: FunctionName { name: name.into() },
                },
            ))
        };
        assert!(matches!(dynamo_tool_choice(&None), DynamoToolChoice::Auto));
        assert!(matches!(
            dynamo_tool_choice(&Some(ChatCompletionToolChoiceOption::Auto)),
            DynamoToolChoice::Auto
        ));
        assert!(matches!(
            dynamo_tool_choice(&Some(ChatCompletionToolChoiceOption::Required)),
            DynamoToolChoice::Required
        ));
        assert!(matches!(
            dynamo_tool_choice(&Some(ChatCompletionToolChoiceOption::None)),
            DynamoToolChoice::None
        ));
        assert!(matches!(
            dynamo_tool_choice(&named("get_weather")),
            DynamoToolChoice::Named(name) if name == "get_weather"
        ));
    }

    #[test]
    fn unsupported_parser_is_rejected() {
        let mut sampling = SamplingParams::default();
        let error = apply_tool_constraint(
            &mut sampling,
            Some("not-a-parser"),
            &DynamoToolChoice::Auto,
            &[tool("get_weather", false)],
            None,
        )
        .unwrap_err();
        assert!(error.contains("not supported"));
        assert!(sampling.json_schema.is_none());
    }

    #[tokio::test]
    async fn streaming_parser_emits_plain_text_without_buffering() {
        let items = parse(
            vec![
                stream_item("Par", None),
                stream_item("is", Some(OpenAIFinishReason::Stop)),
            ],
            "llama3_json",
            false,
        )
        .await;
        assert_eq!(items.len(), 2);
        assert_eq!(delta_text(&items[0]), "Par");
        assert_eq!(choice(&items[0]).finish_reason, None);
        assert_eq!(delta_text(&items[1]), "is");
        assert_eq!(
            choice(&items[1]).finish_reason,
            Some(OpenAIFinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn streaming_parser_buffers_a_whole_call_until_done() {
        let items = parse(
            vec![stream_item(
                r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                Some(OpenAIFinishReason::Stop),
            )],
            "llama3_json",
            false,
        )
        .await;
        assert_eq!(items.len(), 1);
        let terminal = choice(&items[0]);
        assert!(terminal.delta.content.is_none());
        assert_eq!(
            terminal.delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        // The terminal reason is rewritten: calls were emitted.
        assert_eq!(terminal.finish_reason, Some(OpenAIFinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn streaming_parser_detects_bare_json_without_a_start_marker() {
        let items = parse(
            vec![
                stream_item(r#"{"name":"get_weather","parameters":{"#, None),
                stream_item(r#""city":"Paris"}}"#, Some(OpenAIFinishReason::Stop)),
            ],
            "llama3_json",
            false,
        )
        .await;
        assert_eq!(items.len(), 1);
        let terminal = choice(&items[0]);
        assert!(terminal.delta.content.is_none());
        assert_eq!(
            terminal.delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        assert_eq!(terminal.finish_reason, Some(OpenAIFinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn streaming_parser_holds_only_a_split_marker() {
        let items = parse(
            vec![
                stream_item("Before <|python_", None),
                stream_item(
                    r#"tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                    Some(OpenAIFinishReason::Stop),
                ),
            ],
            "llama3_json",
            false,
        )
        .await;
        // The safe prefix streams immediately; the held marker suffix joins
        // the next chunk, which parses into a tool call.
        assert_eq!(delta_text(&items[0]), "Before ");
        let tool_call = choice(&items[1]);
        assert!(tool_call.delta.content.is_none());
        assert_eq!(
            tool_call.delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        assert_eq!(tool_call.finish_reason, Some(OpenAIFinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn streaming_parser_releases_an_incomplete_marker_at_done() {
        let items = parse(
            vec![
                stream_item("Before <|python_", None),
                stream_done(OpenAIFinishReason::Stop),
            ],
            "llama3_json",
            false,
        )
        .await;
        let text = items
            .iter()
            .filter_map(|item| choice(item).delta.content.as_ref())
            .filter_map(|content| match content {
                ChatCompletionMessageContent::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "Before <|python_");
        assert_eq!(
            choice(&items[1]).finish_reason,
            Some(OpenAIFinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn streaming_parser_emits_a_complete_tool_call_before_done() {
        let items = parse(
            vec![
                stream_item(
                    r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                    None,
                ),
                stream_done(OpenAIFinishReason::Stop),
            ],
            "llama3_json",
            false,
        )
        .await;
        let tool_position = items
            .iter()
            .position(|item| choice(item).delta.tool_calls.is_some())
            .expect("tool call chunk");
        let terminal_position = items
            .iter()
            .position(|item| choice(item).finish_reason.is_some())
            .expect("terminal chunk");
        assert!(tool_position < terminal_position);
        // Calls were emitted, so the terminal reason is rewritten.
        assert_eq!(
            choice(&items[terminal_position]).finish_reason,
            Some(OpenAIFinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn canonical_qwen_parser_name_uses_dynamo_qwen25() {
        let (content, calls) = parse_chat_tool_calls(
            r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#.into(),
            Some("qwen"),
            None,
            true,
        )
        .await;
        assert!(content.is_empty());
        assert_eq!(calls.unwrap()[0].function.name, "get_weather");
    }

    #[tokio::test]
    async fn unary_parse_without_a_parser_passes_content_through() {
        let (content, calls) =
            parse_chat_tool_calls("<|python_tag|>call".into(), None, None, true).await;
        assert_eq!(content, "<|python_tag|>call");
        assert!(calls.is_none());
    }

    #[test]
    fn chat_finish_reason_maps_scheduler_kinds() {
        let output = |finish: serde_json::Value| ChunkEvent {
            rid: "r".into(),
            text: "x".into(),
            token_ids: vec![1],
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: Some(serde_json::from_value(finish).unwrap()),
            ..Default::default()
        };
        assert_eq!(
            chat_finish_reason(&output(
                serde_json::json!({"type": "stop", "matched": "</s>"})
            )),
            Some(OpenAIFinishReason::Stop)
        );
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "length", "length": 8}))),
            Some(OpenAIFinishReason::Length)
        );
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "content_filter"}))),
            Some(OpenAIFinishReason::ContentFilter)
        );
        // Unknown kinds (including a bare abort) fall back to `stop`.
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "abort"}))),
            Some(OpenAIFinishReason::Stop)
        );
        assert_eq!(
            chat_finish_reason(&ChunkEvent {
                finish_reason: None,
                ..Default::default()
            }),
            None
        );
    }

    #[test]
    fn chat_delta_carries_the_optional_columns() {
        let delta = chat_delta(
            Some("hi".into()),
            Some(Role::Assistant),
            Some(vec![ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call_1".into()),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some("get_weather".into()),
                    arguments: Some("{}".into()),
                }),
            }]),
            Some("thinking".into()),
        );
        assert_eq!(
            delta.content,
            Some(ChatCompletionMessageContent::Text("hi".into()))
        );
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(delta.reasoning_content, Some("thinking".into()));
        assert_eq!(
            delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        assert!(chat_delta(None, None, None, None).content.is_none());
    }
}
