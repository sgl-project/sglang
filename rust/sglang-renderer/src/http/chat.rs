//! OpenAI Chat Completions endpoint.

use std::convert::Infallible;
use std::sync::Arc;

use crate::{
    ChatEvent, ChatFinishReason, ChatResponseError, ChatResponseInput, ChatResponseItem,
    ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent, GenerationEvent,
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationSubmission,
    InferenceBackend, InferenceSession,
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
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, ChatCompletionTokenLogprob,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason, FunctionCall, FunctionCallStream, FunctionType, Role,
    ServiceTier as ChatServiceTier, TopLogprobs,
};
use futures::StreamExt;

use super::completions::completion_usage;
use super::{
    OpenAIHttpFrontend, collect_output, error_payload, openai_error, renderer_status,
    submit_generation, unix_seconds_u32,
};

pub(super) fn routes<B: InferenceBackend>() -> Router<Arc<OpenAIHttpFrontend<B>>> {
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

async fn chat_completions<B: InferenceBackend>(
    State(state): State<Arc<OpenAIHttpFrontend<B>>>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let stream = request.stream.unwrap_or(false);
    let model = request.model.clone();
    let want_logprobs = request.logprobs.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage)
        || state
            .renderer
            .config()
            .stream_response_default_include_usage;
    let service_tier = request.service_tier.clone();
    let lowered = match state.renderer.lower_chat(request, &response_id).await {
        Ok(lowered) => lowered,
        Err(error) => {
            let status = renderer_status(error.kind());
            return openai_error(status, error.to_string(), false);
        }
    };
    let generation_inputs = lowered.generation_inputs;
    let response_processor = lowered.response_processor;
    let created = unix_seconds_u32();
    let mut session = state.backend.begin_session();
    let mut submitted = Vec::with_capacity(generation_inputs.len());

    for (index, generation_input) in generation_inputs.into_iter().enumerate() {
        let prepared = match state
            .renderer
            .prepare_generation_input(generation_input)
            .await
        {
            Ok(prepared) => prepared,
            Err(error) => {
                return openai_error(renderer_status(error.kind()), error.to_string(), false);
            }
        };
        let submission = match submit_generation(&mut session, prepared, stream).await {
            Ok(submission) => submission,
            Err(response) => return response,
        };
        submitted.push((index, submission));
    }
    if stream {
        let event_stream = chat_event_stream(
            submitted,
            session,
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
            session,
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
pub(super) async fn unary_chat<S: InferenceSession>(
    submitted: Vec<(usize, GenerationSubmission)>,
    mut session: S,
    mut response_processor: ChatResponseProcessor,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    service_tier: Option<ChatServiceTier>,
) -> Response {
    let mut choices = Vec::with_capacity(submitted.len());
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0u64;

    for (index, submission) in submitted {
        let output = match collect_output(&mut session, submission).await {
            Ok(output) => output,
            Err(error) => {
                let status = StatusCode::from_u16(error.status_code)
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return openai_error(status, error.message, false);
            }
        };

        if prompt_tokens == 0 {
            prompt_tokens = output.prompt_tokens;
        }
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
        let finish_reason = chat_finish_reason(&output).map(openai_finish_reason);
        let parsed = response_processor
            .process_unary(output.text, &output.token_ids)
            .await;
        let finish_reason = if parsed.tool_calls.is_some() {
            Some(OpenAIFinishReason::ToolCalls)
        } else {
            finish_reason
        };
        #[allow(deprecated)]
        let message = ChatCompletionResponseMessage {
            content: (!parsed.content.is_empty())
                .then_some(ChatCompletionMessageContent::Text(parsed.content)),
            refusal: None,
            tool_calls: parsed.tool_calls.map(|calls| {
                calls
                    .into_iter()
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
        choices.push(ChatChoice {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            message,
            finish_reason,
            logprobs,
        });
    }

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

fn chat_event_stream(
    submitted: Vec<(usize, GenerationSubmission)>,
    session: impl InferenceSession,
    response_processor: ChatResponseProcessor,
    context: ChatStreamWireContext,
) -> impl futures::Stream<Item = String> {
    let parsed = semantic_chat_stream(
        submitted,
        session,
        response_processor,
        context.want_logprobs,
    );

    async_stream::stream! {
        futures::pin_mut!(parsed);
        while let Some(item) = parsed.next().await {
            match item {
                ChatResponseItem::Event(ChatEvent::Role { choice }) => {
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
                ChatResponseItem::Event(ChatEvent::Delta {
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
                ChatResponseItem::Event(ChatEvent::Usage {
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
                ChatResponseItem::Event(ChatEvent::Usage { .. }) => {}
                ChatResponseItem::Error(error) => {
                    let status = StatusCode::from_u16(error.status_code)
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    yield error_payload(status, error.message).to_string();
                }
            }
        }
        yield "[DONE]".to_string();
    }
}

fn semantic_chat_stream<S: InferenceSession>(
    submitted: Vec<(usize, GenerationSubmission)>,
    mut session: S,
    response_processor: ChatResponseProcessor,
    want_logprobs: bool,
) -> impl futures::Stream<Item = ChatResponseItem> {
    let raw = async_stream::stream! {
        let mut streams = Vec::with_capacity(submitted.len());
        for (index, submission) in submitted {
            let id = submission.id;
            streams.push(submission.events.map(move |event| (index, id.clone(), event)).boxed());
        }
        let mut events = futures::stream::select_all(streams);
        while let Some((index, id, item)) = events.next().await {
            let (output, terminal) = match item {
                Ok(GenerationEvent::Frame(output)) => (output, false),
                Ok(GenerationEvent::Done(output)) => (output, true),
                Err(error) => {
                    session.complete(&id);
                    yield ChatResponseInput::Error(ChatResponseError {
                        status_code: error.status_code,
                        message: error.message,
                    });
                    continue;
                }
            };
            if terminal {
                session.complete(&id);
            }
            let finish_reason = chat_finish_reason(&output);
            let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
            yield ChatResponseInput::Decoded(DecodedChatEvent {
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
    let mut top_offset = 0usize;
    for (position, (&logprob, &token_id)) in extras
        .output_logprobs
        .iter()
        .zip(&extras.output_logprob_token_ids)
        .enumerate()
    {
        let token = extras
            .output_logprob_text
            .get(position)
            .cloned()
            .unwrap_or_else(|| format!("token_id:{token_id}"));
        let top_len = extras
            .output_top_logprob_lengths
            .get(position)
            .copied()
            .unwrap_or(0) as usize;
        let top_logprobs = extras.output_top_logprobs[top_offset..]
            .iter()
            .zip(&extras.output_top_logprob_token_ids[top_offset..])
            .take(top_len)
            .enumerate()
            .map(|(offset, (&logprob, &id))| {
                let text = extras
                    .output_top_logprob_text
                    .get(top_offset + offset)
                    .cloned()
                    .unwrap_or_else(|| format!("token_id:{id}"));
                TopLogprobs {
                    bytes: Some(text.as_bytes().to_vec()),
                    token: text,
                    logprob,
                }
            })
            .collect();
        top_offset = top_offset.saturating_add(top_len);
        content.push(ChatCompletionTokenLogprob {
            bytes: Some(token.as_bytes().to_vec()),
            token,
            logprob,
            token_id: u32::try_from(token_id).ok(),
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
    let mut response = serde_json::to_value(response).expect("OpenAI response must serialize");
    if let Some(delta) = response
        .pointer_mut("/choices/0/delta")
        .and_then(serde_json::Value::as_object_mut)
    {
        delta
            .entry("reasoning_content")
            .or_insert(serde_json::Value::Null);
    }
    response.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{TestSession, chat_submitted, chunk};
    use super::{ChatStreamWireContext, chat_event_stream, chat_logprobs, unary_chat};
    use crate::openai::{ChatSamplingDefaults as SamplingDefaults, chat_sampling_params};
    use crate::{
        GenerationOutputExtras, OpenAIRequestLowerer, RendererConfig, RendererLimits,
        SamplingDefaults as ModelSamplingDefaults,
    };
    use axum::http::StatusCode;
    use dynamo_protocols::types::CreateChatCompletionRequest;
    use futures::StreamExt;

    fn request() -> CreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    async fn response_processor(
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
            revision: None,
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 128,
            default_sampling_params: ModelSamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": choices
        }))
        .unwrap();
        OpenAIRequestLowerer::new(config)
            .lower_chat(request, "chatcmpl-test")
            .await
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
        let model = ModelSamplingDefaults {
            temperature: Some(0.6),
            top_p: Some(0.9),
        };
        // Omitted → model defaults, not the 1.0 OpenAI terminals.
        let sampling = chat_sampling_params(
            &request(),
            &SamplingDefaults::CHAT.with_model_defaults(&model),
        )
        .unwrap();
        assert_eq!(sampling.temperature, 0.6);
        assert_eq!(sampling.top_p, 0.9);
        // Explicit request values win. `Option<f32>` loses precision in f64 —
        // compare with tolerance.
        let mut request = request();
        request.temperature = Some(0.2);
        request.top_p = Some(0.5);
        let sampling = chat_sampling_params(
            &request,
            &SamplingDefaults::CHAT.with_model_defaults(&model),
        )
        .unwrap();
        assert!((sampling.temperature - 0.2).abs() < 1e-6);
        assert!((sampling.top_p - 0.5).abs() < 1e-6);
    }

    /// `--sampling-defaults openai` resolves an empty model-config slice, so the
    /// conversion falls back to the OpenAI terminal defaults.
    #[test]
    fn sampling_defaults_fall_back_to_openai_terminals_in_openai_mode() {
        let openai_mode = ModelSamplingDefaults::default();
        let sampling = chat_sampling_params(
            &request(),
            &SamplingDefaults::CHAT.with_model_defaults(&openai_mode),
        )
        .unwrap();
        assert_eq!(sampling.temperature, 1.0);
        assert_eq!(sampling.top_p, 1.0);
    }

    /// A request with no `max_tokens`/`max_completion_tokens` stays unbounded —
    /// no terminal default is imposed.
    #[test]
    fn chat_without_a_token_limit_stays_unbounded() {
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        assert_eq!(
            chat_sampling_params(&request, &SamplingDefaults::CHAT)
                .unwrap()
                .max_new_tokens,
            None
        );
    }

    #[test]
    fn chat_logprobs_use_dynamo_wire_types() {
        let extras = GenerationOutputExtras {
            output_logprobs: vec![-0.25],
            output_logprob_token_ids: vec![7],
            output_logprob_text: vec!["x".into()],
            output_top_logprobs: vec![-0.25, -1.0],
            output_top_logprob_token_ids: vec![7, 8],
            output_top_logprob_lengths: vec![2],
            output_top_logprob_text: vec!["x".into(), "y".into()],
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
        let (choice0, tx0) = chat_submitted(0, "r0");
        let (choice1, tx1) = chat_submitted(1, "r1");
        tx0.send(chunk("r0", "Paris", true)).await.unwrap();
        tx1.send(chunk("r1", "Paris", true)).await.unwrap();

        let response = unary_chat(
            vec![choice0, choice1],
            TestSession,
            response_processor(None, 2).await,
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
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk(
            "r0",
            "<think>because Paris is famous</think>Paris",
            true,
        ))
        .await
        .unwrap();

        let response = unary_chat(
            vec![choice],
            TestSession,
            response_processor(Some("deepseek-r1"), 1).await,
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
        let (choice, tx) = chat_submitted(0, "r0");
        // Force mode starts in reasoning, so the opener is stripped and the first
        // reasoning fragment streams immediately.
        tx.send(chunk("r0", "<think>be", false)).await.unwrap();
        tx.send(chunk("r0", "cause</think>Par", false))
            .await
            .unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            TestSession,
            response_processor(Some("deepseek-r1"), 1).await,
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
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            TestSession,
            response_processor(None, 1).await,
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
}
