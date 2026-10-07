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
//! [`chat_delta`] builds the stream deltas these paths emit, and
//! [`chat_finish_reason`] maps the scheduler's finish reason onto the OpenAI
//! wire values.

use dynamo_parsers::{
    ToolChoice as DynamoToolChoice, ToolDefinition, try_tool_call_parse_aggregate_finalize,
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    FinishReason as OpenAIFinishReason, FunctionCall, FunctionType, Role,
};
use sglang_processor::{ToolConstraint, dynamo_tool_parser_name, tool_constraint};

use crate::api_server::core::CoreOutput;
use crate::message::sampling::SamplingParams;

/// Set the `tool_choice` constraint `sglang_processor::tool_constraint` builds.
pub(super) fn apply_tool_constraint(
    sampling: &mut SamplingParams,
    parser: Option<&str>,
    tool_choice: &DynamoToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
) -> Result<(), String> {
    match tool_constraint(parser, tool_choice, tools, parallel_tool_calls)? {
        Some(ToolConstraint::StructuralTag(tag)) => sampling.structural_tag = Some(tag),
        Some(ToolConstraint::JsonSchema(schema)) => sampling.json_schema = Some(schema),
        None => {}
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
    let parser = dynamo_tool_parser_name(parser);
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
pub(super) fn chat_finish_reason(output: &CoreOutput) -> Option<OpenAIFinishReason> {
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
    use super::{chat_delta, chat_finish_reason, parse_chat_tool_calls};
    use crate::api_server::core::CoreOutput;
    use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
    use dynamo_protocols::types::CreateChatCompletionStreamResponse as StreamResponse;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionToolChoiceOption, FinishReason as OpenAIFinishReason, FunctionCallStream,
        FunctionType, Role,
    };
    use futures::{StreamExt, stream};

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
        let output = |finish: serde_json::Value| CoreOutput {
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
            chat_finish_reason(&CoreOutput {
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
