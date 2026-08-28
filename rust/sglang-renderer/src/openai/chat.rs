//! OpenAI Chat Completions endpoint.

use std::sync::Arc;
use std::{collections::BTreeMap, convert::Infallible};

use crate::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    ResponseError,
};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::post,
};
use dynamo_protocols::types::{
    ChatChoice, ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta,
    ChatCompletionStreamResponseDeltaFunctionCall, ChatCompletionTokenLogprob, CompletionUsage,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason, FunctionCall, FunctionCallStream, FunctionType, Role,
    ServiceTier as ChatServiceTier, TopLogprobs,
};
use futures::StreamExt;
use serde::Serialize;

use super::{
    ChatCompletionRequest, OpenAIHttpFrontend, completion_usage,
    error::{error_payload, json_rejection_response, openai_error, renderer_status},
    protocol::lower_chat_request,
    submission::{merge_indexed, submit_generate_requests},
    unix_seconds_u32,
};

pub(super) fn routes() -> Router<Arc<OpenAIHttpFrontend>> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

struct ChatStreamWireContext {
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    include_usage: bool,
    service_tier: Option<ChatServiceTier>,
}

async fn chat_completions(
    State(state): State<Arc<OpenAIHttpFrontend>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let extended = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return json_rejection_response(rejection),
    };
    let stream = extended.stream.unwrap_or(false);
    let model = extended.model.clone();
    let want_logprobs = extended.logprobs.unwrap_or(false);
    let include_usage = extended
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage)
        || state
            .renderer
            .config()
            .stream_response_default_include_usage;
    let service_tier = extended.service_tier.clone();
    let (response_id, chat_request) = match lower_chat_request(state.renderer.config(), extended) {
        Ok(request) => request,
        Err(error) => {
            let status = renderer_status(&error);
            return openai_error(status, error.to_string(), false);
        }
    };
    let chat = match state.renderer.prepare_chat(chat_request).await {
        Ok(chat) => chat,
        Err(error) => {
            let status = renderer_status(&error);
            return openai_error(status, error.to_string(), false);
        }
    };
    let response_processor = chat.response_processor;
    let created = unix_seconds_u32();
    let streams = match submit_generate_requests(&state, chat.requests, stream).await {
        Ok(streams) => streams,
        Err(response) => return response,
    };
    let submitted = streams.into_iter().enumerate().collect();
    if stream {
        let event_stream = chat_event_stream(
            submitted,
            response_processor,
            ChatStreamWireContext {
                response_id,
                model,
                created,
                want_logprobs,
                include_usage,
                service_tier,
            },
        )
        .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(event_stream).into_response()
    } else {
        unary_chat(
            submitted,
            response_processor,
            response_id,
            model,
            created,
            want_logprobs,
            service_tier,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn unary_chat(
    submitted: Vec<(usize, GenerationStream)>,
    response_processor: ChatResponseProcessor,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    service_tier: Option<ChatServiceTier>,
) -> Response {
    let choice_count = submitted.len();
    let mut accumulated = (0..choice_count)
        .map(|_| UnaryChatChoice::default())
        .collect::<Vec<_>>();
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u64;
    let parsed = semantic_chat_stream(submitted, response_processor, want_logprobs);
    futures::pin_mut!(parsed);
    while let Some(item) = parsed.next().await {
        match item {
            Ok(ChatEvent::Role { .. }) => {}
            Ok(ChatEvent::Delta {
                choice,
                content,
                reasoning_content,
                tool_calls,
                finish_reason,
                logprobs,
            }) => {
                let Some(choice) = accumulated.get_mut(choice) else {
                    return openai_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "chat response choice is out of range",
                        false,
                    );
                };
                if let Some(content) = content {
                    choice.content.push_str(&content);
                }
                if let Some(reasoning) = reasoning_content {
                    choice.reasoning_content.push_str(&reasoning);
                }
                if let Some(tool_calls) = tool_calls {
                    choice.extend_tool_calls(tool_calls);
                }
                if finish_reason.is_some() {
                    choice.finish_reason = finish_reason;
                }
                merge_chat_logprobs(&mut choice.logprobs, logprobs);
            }
            Ok(ChatEvent::Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
            }) => {
                prompt_tokens = prompt;
                completion_tokens = completion;
            }
            Err(error) => {
                let status = StatusCode::from_u16(error.status_code)
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return openai_error(status, error.message, false);
            }
        }
    }

    let choices = accumulated
        .into_iter()
        .enumerate()
        .map(|(index, parsed)| {
            #[allow(deprecated)]
            let message = ChatCompletionResponseMessage {
                content: (!parsed.content.is_empty())
                    .then_some(ChatCompletionMessageContent::Text(parsed.content)),
                refusal: None,
                tool_calls: (!parsed.tool_calls.is_empty()).then(|| {
                    parsed
                        .tool_calls
                        .into_values()
                        .map(|call| ChatCompletionMessageToolCall {
                            id: call.id,
                            r#type: FunctionType::Function,
                            function: FunctionCall {
                                name: call.name,
                                arguments: call.arguments,
                            },
                        })
                        .collect()
                }),
                role: Role::Assistant,
                function_call: None,
                audio: None,
                // Python: `reasoning_text if reasoning_text else None`.
                reasoning_content: (!parsed.reasoning_content.is_empty())
                    .then_some(parsed.reasoning_content),
            };
            ChatChoice {
                index: u32::try_from(index).unwrap_or(u32::MAX),
                message,
                finish_reason: parsed.finish_reason.map(openai_finish_reason),
                logprobs: parsed.logprobs,
            }
        })
        .collect();

    Json(CreateChatCompletionResponse {
        id: response_id,
        choices,
        created,
        model,
        service_tier,
        system_fingerprint: None,
        object: "chat.completion".into(),
        usage: Some(completion_usage(
            prompt_tokens,
            u32::try_from(completion_tokens).unwrap_or(u32::MAX),
        )),
    })
    .into_response()
}

#[derive(Default)]
struct UnaryChatChoice {
    content: String,
    reasoning_content: String,
    tool_calls: BTreeMap<u32, UnaryToolCall>,
    finish_reason: Option<ChatFinishReason>,
    logprobs: Option<ChatChoiceLogprobs>,
}

#[derive(Default)]
struct UnaryToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl UnaryChatChoice {
    fn extend_tool_calls(&mut self, deltas: Vec<ChatToolCallDelta>) {
        for delta in deltas {
            let call = self.tool_calls.entry(delta.index).or_default();
            if let Some(id) = delta.id {
                call.id = id;
            }
            if let Some(name) = delta.name {
                call.name = name;
            }
            if let Some(arguments) = delta.arguments {
                call.arguments.push_str(&arguments);
            }
        }
    }
}

fn merge_chat_logprobs(
    collected: &mut Option<ChatChoiceLogprobs>,
    delta: Option<ChatChoiceLogprobs>,
) {
    let Some(mut delta) = delta else {
        return;
    };
    let collected = collected.get_or_insert_with(|| ChatChoiceLogprobs {
        content: Some(Vec::new()),
        refusal: None,
    });
    if let Some(content) = delta.content.take() {
        collected
            .content
            .get_or_insert_with(Vec::new)
            .extend(content);
    }
}

fn chat_event_stream(
    submitted: Vec<(usize, GenerationStream)>,
    response_processor: ChatResponseProcessor,
    context: ChatStreamWireContext,
) -> impl futures::Stream<Item = String> {
    let parsed = semantic_chat_stream(submitted, response_processor, context.want_logprobs);

    async_stream::stream! {
        futures::pin_mut!(parsed);
        while let Some(item) = parsed.next().await {
            match item {
                Ok(ChatEvent::Role { choice }) => {
                    yield serialize_chat_stream_response(chat_stream_response(
                        &context.response_id,
                        &context.model,
                        context.created,
                        context.service_tier.clone(),
                        vec![ChatChoiceStream {
                            index: choice as u32,
                            delta: chat_delta(None, Some(Role::Assistant), None, None),
                            finish_reason: None,
                            logprobs: None,
                        }],
                        None,
                    ));
                }
                Ok(ChatEvent::Delta {
                    choice,
                    content,
                    reasoning_content,
                    tool_calls,
                    finish_reason,
                    logprobs,
                }) => {
                    yield serialize_chat_stream_response(chat_stream_response(
                        &context.response_id,
                        &context.model,
                        context.created,
                        context.service_tier.clone(),
                        vec![ChatChoiceStream {
                            index: choice as u32,
                            delta: chat_delta(
                                content,
                                None,
                                tool_calls.map(|calls| {
                                    calls.into_iter().map(openai_tool_call_delta).collect()
                                }),
                                reasoning_content,
                            ),
                            finish_reason: finish_reason.map(openai_finish_reason),
                            logprobs,
                        }],
                        None,
                    ));
                }
                Ok(ChatEvent::Usage {
                    prompt_tokens,
                    completion_tokens,
                }) if context.include_usage => {
                    yield serialize_chat_stream_response(chat_stream_response(
                        &context.response_id,
                        &context.model,
                        context.created,
                        context.service_tier.clone(),
                        Vec::new(),
                        Some((prompt_tokens, completion_tokens)),
                    ));
                }
                Ok(ChatEvent::Usage { .. }) => {}
                Err(error) => {
                    let status = StatusCode::from_u16(error.status_code)
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    yield error_payload(status, error.message).to_string();
                }
            }
        }
        yield "[DONE]".to_string();
    }
}

fn semantic_chat_stream(
    submitted: Vec<(usize, GenerationStream)>,
    response_processor: ChatResponseProcessor,
    want_logprobs: bool,
) -> impl futures::Stream<Item = Result<ChatEvent, ResponseError>> {
    let raw = async_stream::stream! {
        let streams = submitted.into_iter().map(|(_, events)| events).collect();
        let mut events = merge_indexed(streams);
        while let Some((index, item)) = events.next().await {
            let output = match item {
                Ok(output) => output,
                Err(error) => {
                    yield Err(error);
                    break;
                }
            };
            let finish_reason = chat_finish_reason(&output);
            let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
            yield Ok(DecodedChatEvent {
                choice: index,
                text: output.text,
                token_ids: output.token_ids,
                finish_reason,
                logprobs,
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            });
        }
    };
    response_processor.process_stream(raw)
}

fn chat_finish_reason(output: &GenerationOutput) -> Option<ChatFinishReason> {
    output.finish_reason.as_ref().map(|reason| match reason {
        GenerationFinishReason::Length => ChatFinishReason::Length,
        GenerationFinishReason::ContentFilter => ChatFinishReason::ContentFilter,
        GenerationFinishReason::Stop(_)
        | GenerationFinishReason::Abort
        | GenerationFinishReason::Other(_) => ChatFinishReason::Stop,
    })
}

#[allow(deprecated)]
fn chat_logprobs(extras: Option<&GenerationOutputExtras>) -> ChatChoiceLogprobs {
    let mut content = Vec::new();
    let Some(extras) = extras else {
        return ChatChoiceLogprobs {
            content: Some(content),
            refusal: None,
        };
    };
    for position in &extras.output_logprobs {
        let selected = &position.token;
        let token = selected
            .text
            .clone()
            .unwrap_or_else(|| format!("token_id:{}", selected.token_id));
        let top_logprobs = position
            .top
            .iter()
            .map(|candidate| {
                let text = candidate
                    .text
                    .clone()
                    .unwrap_or_else(|| format!("token_id:{}", candidate.token_id));
                TopLogprobs {
                    bytes: Some(text.as_bytes().to_vec()),
                    token: text,
                    logprob: candidate.logprob.unwrap_or(f32::NAN),
                }
            })
            .collect();
        content.push(ChatCompletionTokenLogprob {
            bytes: Some(token.as_bytes().to_vec()),
            token,
            logprob: selected.logprob.unwrap_or(f32::NAN),
            token_id: u32::try_from(selected.token_id).ok(),
            top_logprobs,
        });
    }
    ChatChoiceLogprobs {
        content: Some(content),
        refusal: None,
    }
}

#[allow(deprecated)]
fn chat_delta(
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

fn chat_stream_response(
    response_id: &str,
    model: &str,
    created: u32,
    service_tier: Option<ChatServiceTier>,
    choices: Vec<ChatChoiceStream>,
    usage: Option<(u32, u64)>,
) -> CreateChatCompletionStreamResponse {
    CreateChatCompletionStreamResponse {
        id: response_id.to_owned(),
        choices,
        created,
        model: model.to_owned(),
        service_tier,
        system_fingerprint: None,
        object: "chat.completion.chunk".into(),
        usage: usage.map(|(prompt, completion)| {
            completion_usage(prompt, u32::try_from(completion).unwrap_or(u32::MAX))
        }),
    }
}

fn openai_finish_reason(reason: ChatFinishReason) -> OpenAIFinishReason {
    match reason {
        ChatFinishReason::Stop => OpenAIFinishReason::Stop,
        ChatFinishReason::Length => OpenAIFinishReason::Length,
        ChatFinishReason::ContentFilter => OpenAIFinishReason::ContentFilter,
        ChatFinishReason::ToolCalls => OpenAIFinishReason::ToolCalls,
    }
}

fn openai_tool_call_delta(call: ChatToolCallDelta) -> ChatCompletionMessageToolCallChunk {
    ChatCompletionMessageToolCallChunk {
        index: call.index,
        id: call.id,
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: call.name,
            arguments: call.arguments,
        }),
    }
}

fn serialize_chat_stream_response(response: CreateChatCompletionStreamResponse) -> String {
    serde_json::to_string(&ChatStreamResponseWire::from(&response))
        .expect("OpenAI response must serialize")
}

/// The Dynamo response type omits an absent `reasoning_content`. SGLang's
/// streaming contract emits it explicitly as `null`, so use a borrowed wire
/// view instead of building and patching a `serde_json::Value` tree.
#[derive(Serialize)]
struct ChatStreamResponseWire<'a> {
    id: &'a str,
    choices: Vec<ChatChoiceStreamWire<'a>>,
    created: u32,
    model: &'a str,
    service_tier: &'a Option<ChatServiceTier>,
    system_fingerprint: &'a Option<String>,
    object: &'a str,
    usage: &'a Option<CompletionUsage>,
}

impl<'a> From<&'a CreateChatCompletionStreamResponse> for ChatStreamResponseWire<'a> {
    fn from(response: &'a CreateChatCompletionStreamResponse) -> Self {
        Self {
            id: &response.id,
            choices: response
                .choices
                .iter()
                .map(ChatChoiceStreamWire::from)
                .collect(),
            created: response.created,
            model: &response.model,
            service_tier: &response.service_tier,
            system_fingerprint: &response.system_fingerprint,
            object: &response.object,
            usage: &response.usage,
        }
    }
}

#[derive(Serialize)]
struct ChatChoiceStreamWire<'a> {
    index: u32,
    delta: ChatDeltaWire<'a>,
    finish_reason: &'a Option<OpenAIFinishReason>,
    logprobs: &'a Option<ChatChoiceLogprobs>,
}

impl<'a> From<&'a ChatChoiceStream> for ChatChoiceStreamWire<'a> {
    fn from(choice: &'a ChatChoiceStream) -> Self {
        Self {
            index: choice.index,
            delta: ChatDeltaWire::from(&choice.delta),
            finish_reason: &choice.finish_reason,
            logprobs: &choice.logprobs,
        }
    }
}

#[derive(Serialize)]
struct ChatDeltaWire<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a ChatCompletionMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<&'a ChatCompletionStreamResponseDeltaFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a Vec<ChatCompletionMessageToolCallChunk>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal: Option<&'a String>,
    reasoning_content: Option<&'a str>,
}

impl<'a> From<&'a ChatCompletionStreamResponseDelta> for ChatDeltaWire<'a> {
    fn from(delta: &'a ChatCompletionStreamResponseDelta) -> Self {
        Self {
            content: delta.content.as_ref(),
            function_call: delta.function_call.as_ref(),
            tool_calls: delta.tool_calls.as_ref(),
            role: delta.role.as_ref(),
            refusal: delta.refusal.as_ref(),
            reasoning_content: delta.reasoning_content.as_deref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatStreamWireContext, chat_event_stream, chat_logprobs, unary_chat};
    use crate::openai::protocol::ChatCompletionRequest;
    use crate::openai::protocol::{chat_sampling_params, lower_chat_request};
    use crate::openai::test_utils::{chat_submitted, chunk};
    use crate::{
        ChatPreprocessor, GenerationOutputExtras, PositionLogprobs, RendererConfig, RendererLimits,
        ResponseError, SamplingDefaults, TokenLogprob,
    };
    use axum::http::StatusCode;
    use futures::StreamExt;

    fn request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    fn response_processor(
        reasoning_parser: Option<&str>,
        choices: usize,
    ) -> crate::ChatResponseProcessor {
        let config = RendererConfig {
            model_path: String::new(),
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: reasoning_parser.map(str::to_owned),
            default_chat_template_kwargs: Default::default(),
            revision: None,
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": choices
        }))
        .unwrap();
        let (_, chat) = lower_chat_request(&config, request).unwrap();
        ChatPreprocessor::new(
            &config,
            Some(crate::preprocessing::load_test_chat_formatter("chatml")),
        )
        .preprocess(chat)
        .unwrap()
        .response_processor
    }

    fn wire_context(include_usage: bool) -> ChatStreamWireContext {
        ChatStreamWireContext {
            response_id: "chatcmpl-test".into(),
            model: "model".into(),
            created: 1,
            want_logprobs: false,
            include_usage,
            service_tier: None,
        }
    }

    /// Python `to_sampling_params` priority: user value > model generation
    /// config (`--sampling-defaults model`) > OpenAI terminal default.
    #[test]
    fn sampling_defaults_follow_python_priority_chain() {
        let model = SamplingDefaults {
            temperature: Some(0.6),
            top_p: Some(0.9),
            top_k: Some(32),
            min_p: Some(0.1),
            repetition_penalty: Some(1.1),
        };
        // Omitted → model defaults, not the 1.0 OpenAI terminals.
        let sampling = chat_sampling_params(&request(), &model).unwrap();
        assert_eq!(sampling.temperature, 0.6);
        assert_eq!(sampling.top_p, 0.9);
        assert_eq!(sampling.top_k, 32);
        assert_eq!(sampling.min_p, 0.1);
        assert_eq!(sampling.repetition_penalty, 1.1);
        // Explicit request values win. `Option<f32>` loses precision in f64 —
        // compare with tolerance.
        let mut request = request();
        request.temperature = Some(0.2);
        request.top_p = Some(0.5);
        let sampling = chat_sampling_params(&request, &model).unwrap();
        assert!((sampling.temperature - 0.2).abs() < 1e-6);
        assert!((sampling.top_p - 0.5).abs() < 1e-6);
    }

    /// `--sampling-defaults openai` resolves an empty model-config slice, so the
    /// conversion falls back to the OpenAI terminal defaults.
    #[test]
    fn sampling_defaults_fall_back_to_openai_terminals_in_openai_mode() {
        let openai_mode = SamplingDefaults::default();
        let sampling = chat_sampling_params(&request(), &openai_mode).unwrap();
        assert_eq!(sampling.temperature, 1.0);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.top_k, 1 << 30);
        assert_eq!(sampling.min_p, 0.0);
        assert_eq!(sampling.repetition_penalty, 1.0);
    }

    /// A request with no `max_tokens`/`max_completion_tokens` stays unbounded —
    /// no terminal default is imposed.
    #[test]
    fn chat_without_a_token_limit_stays_unbounded() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        assert_eq!(
            chat_sampling_params(&request, &SamplingDefaults::default())
                .unwrap()
                .max_new_tokens,
            None
        );
    }

    #[test]
    fn chat_logprobs_use_dynamo_wire_types() {
        let extras = GenerationOutputExtras {
            output_logprobs: vec![PositionLogprobs {
                token: TokenLogprob {
                    logprob: Some(-0.25),
                    token_id: 7,
                    text: Some("x".into()),
                },
                top: vec![
                    TokenLogprob {
                        logprob: Some(-0.25),
                        token_id: 7,
                        text: Some("x".into()),
                    },
                    TokenLogprob {
                        logprob: Some(-1.0),
                        token_id: 8,
                        text: Some("y".into()),
                    },
                ],
            }],
            ..Default::default()
        };
        let logprobs = chat_logprobs(Some(&extras));
        let token = &logprobs.content.unwrap()[0];
        assert_eq!(token.token, "x");
        assert_eq!(token.token_id, Some(7));
        assert_eq!(token.top_logprobs.len(), 2);
        assert_eq!(token.top_logprobs[1].token, "y");
    }

    #[tokio::test]
    async fn unary_chat_fans_in_choices_and_usage() {
        let (choice0, tx0) = chat_submitted(0);
        let (choice1, tx1) = chat_submitted(1);
        tx0.send(chunk("Paris", true)).await.unwrap();
        tx1.send(chunk("Paris", true)).await.unwrap();

        let response = unary_chat(
            vec![choice0, choice1],
            response_processor(None, 2),
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert_eq!(value["choices"][1]["index"], 1);
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn unary_chat_separates_reasoning_content_with_parser_configured() {
        let (choice, tx) = chat_submitted(0);
        tx.send(chunk("<think>because Paris is famous</think>Paris", true))
            .await
            .unwrap();

        let response = unary_chat(
            vec![choice],
            response_processor(Some("deepseek-r1"), 1),
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "because Paris is famous"
        );
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert!(value["choices"][0]["message"]["reasoning_content"].is_string());
    }

    #[tokio::test]
    async fn streaming_chat_separates_reasoning_into_own_deltas() {
        let (choice, tx) = chat_submitted(0);
        // Force mode starts in reasoning, so the opener is stripped and the first
        // reasoning fragment streams immediately.
        tx.send(chunk("<think>be", false)).await.unwrap();
        tx.send(chunk("cause</think>Par", false)).await.unwrap();
        tx.send(chunk("is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            response_processor(Some("deepseek-r1"), 1),
            wire_context(true),
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let first_reasoning: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let second_reasoning: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let content: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[4]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            first_reasoning["choices"][0]["delta"]["reasoning_content"],
            "be"
        );
        assert!(first_reasoning["choices"][0]["delta"]["content"].is_null());
        assert_eq!(
            second_reasoning["choices"][0]["delta"]["reasoning_content"],
            "cause"
        );
        assert_eq!(content["choices"][0]["delta"]["content"], "Par");
        assert!(content["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(frames.len(), 7);
    }

    #[tokio::test]
    async fn streaming_chat_emits_role_deltas_usage_and_done() {
        let (choice, tx) = chat_submitted(0);
        tx.send(chunk("Par", false)).await.unwrap();
        tx.send(chunk("is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            response_processor(None, 1),
            wire_context(true),
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        assert_eq!(frames.len(), 5);
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let delta: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let usage: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert!(role["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(delta["choices"][0]["delta"]["content"], "Par");
        assert!(delta["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[4], "[DONE]");
    }

    #[tokio::test]
    async fn streaming_chat_stops_all_choices_after_error() {
        let (choice0, tx0) = chat_submitted(0);
        let (choice1, tx1) = chat_submitted(1);
        let stream = chat_event_stream(
            vec![choice0, choice1],
            response_processor(None, 2),
            wire_context(true),
        );
        futures::pin_mut!(stream);

        // Chat streams announce every choice before polling engine output.
        stream.next().await.unwrap();
        stream.next().await.unwrap();
        tx0.send(Err(ResponseError {
            status_code: 503,
            message: "out of memory".into(),
        }))
        .await
        .unwrap();
        let error: serde_json::Value = serde_json::from_str(&stream.next().await.unwrap()).unwrap();
        assert_eq!(error["error"]["code"], 503);

        // The other choice may already be ready, but it must not be polled after
        // the aggregate request has emitted an error.
        tx1.send(chunk("late", true)).await.unwrap();
        let remaining = stream.collect::<Vec<_>>().await;
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[1], "[DONE]");
        assert!(remaining.iter().all(|frame| !frame.contains("late")));
    }
}
