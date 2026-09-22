//! Tool-choice constraints and unary tool-call parsing.
//!
//! Two complementary mechanisms, mirroring the Python frontend
//! (`serving_chat.py` + dynamo-parsers):
//!
//! - [`apply_tool_constraint`] turns `tool_choice` into a sampling constraint
//!   *before* submission: a `structural_tag` when the parser supports one (or
//!   when strict tools need the llama3 triggered-tag format), otherwise a
//!   `json_schema` array restricting the output to tool calls. It validates
//!   `tool_choice`/`tools` agreement first.
//! - [`parse_chat_tool_calls`] strips the model's tool-call markers out of a
//!   finished response. Streaming responses use Dynamo's
//!   `apply_tool_calling_jail` directly in `chat.rs`.
//!
//! [`dynamo_parser_name`] canonicalizes the SGLang CLI parser names onto the
//! dynamo-parsers registry keys, [`chat_delta`] builds the stream deltas these
//! paths emit, and [`chat_finish_reason`] maps the scheduler's finish reason
//! onto the OpenAI wire values.
//!
//! # Test coverage
//!
//! The tests cover parser-name canonicalization, every `tool_choice` branch of
//! [`apply_tool_constraint`], Dynamo's streaming jail integration, unary
//! parsing, and finish-reason mapping.

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::{
    StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext,
    ToolChoice as DynamoToolChoice, ToolDefinition, TriggeredTagsConfig,
    try_tool_call_parse_aggregate_finalize,
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption, FinishReason as OpenAIFinishReason, FunctionCall, FunctionType,
    Role,
};

use crate::message::response::ChunkEvent;
use crate::message::sampling::SamplingParams;

/// Canonicalize a tool-call parser name onto the dynamo-parsers registry keys.
///
/// SGLang canonicalizes these legacy CLI names in the opposite direction from
/// the current Dynamo parser registry.
pub(super) fn dynamo_parser_name(parser: &str) -> &str {
    match parser {
        "llama3" => "llama3_json",
        "qwen" => "qwen25",
        "glm" | "glm45" => "glm47",
        other => other,
    }
}

/// Map the OpenAI wire `tool_choice` onto the Dynamo choice. A missing/auto
/// choice reads as `Auto`.
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
        dynamo_tool_choice, parse_chat_tool_calls,
    };
    use crate::message::response::ChunkEvent;
    use crate::message::sampling::SamplingParams;
    use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
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

    async fn apply_jail(
        items: Vec<Annotated<StreamResponse>>,
        parser: &str,
    ) -> Vec<Annotated<StreamResponse>> {
        apply_tool_calling_jail(
            Some(parser.into()),
            Some(ChatCompletionToolChoiceOption::Auto),
            None,
            false,
            stream::iter(items),
        )
        .collect()
        .await
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
    async fn streaming_jail_emits_plain_text_without_buffering() {
        let items = apply_jail(
            vec![
                stream_item("Par", None),
                stream_item("is", Some(OpenAIFinishReason::Stop)),
            ],
            "llama3_json",
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
    async fn streaming_jail_buffers_a_whole_call_until_done() {
        let items = apply_jail(
            vec![stream_item(
                r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                Some(OpenAIFinishReason::Stop),
            )],
            "llama3_json",
        )
        .await;
        assert_eq!(items.len(), 1);
        let terminal = choice(&items[0]);
        assert!(matches!(
            terminal.delta.content.as_ref(),
            Some(ChatCompletionMessageContent::Text(text)) if text.is_empty()
        ));
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
    async fn streaming_jail_detects_bare_json_without_a_start_marker() {
        let items = apply_jail(
            vec![
                stream_item(r#"{"name":"get_weather","parameters":{"#, None),
                stream_item(r#""city":"Paris"}}"#, Some(OpenAIFinishReason::Stop)),
            ],
            "llama3_json",
        )
        .await;
        assert_eq!(items.len(), 1);
        let terminal = choice(&items[0]);
        assert!(matches!(
            terminal.delta.content.as_ref(),
            Some(ChatCompletionMessageContent::Text(text)) if text.is_empty()
        ));
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
    async fn streaming_jail_holds_only_a_split_marker() {
        let items = apply_jail(
            vec![
                stream_item("Before <|python_", None),
                stream_item(
                    r#"tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                    Some(OpenAIFinishReason::Stop),
                ),
            ],
            "llama3_json",
        )
        .await;
        // The safe prefix streams immediately; the held marker suffix joins
        // the next chunk, which parses into a tool call.
        assert_eq!(delta_text(&items[0]), "Before ");
        let tool_call = choice(&items[1]);
        assert!(matches!(
            tool_call.delta.content.as_ref(),
            Some(ChatCompletionMessageContent::Text(text)) if text.is_empty()
        ));
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
    async fn streaming_jail_releases_an_incomplete_marker_at_done() {
        let items = apply_jail(
            vec![
                stream_item("Before <|python_", None),
                stream_done(OpenAIFinishReason::Stop),
            ],
            "llama3_json",
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
    async fn streaming_jail_emits_a_complete_tool_call_before_done() {
        let items = apply_jail(
            vec![
                stream_item(
                    r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                    None,
                ),
                stream_done(OpenAIFinishReason::Stop),
            ],
            "llama3_json",
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
