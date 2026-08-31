//! OpenAI-compatible generation endpoints: chat completions, the legacy
//! text-completion endpoint, and model discovery.
//!
//! The HTTP adapter stays deliberately thin: Dynamo owns the standard OpenAI
//! request and response primitives. Native [`ChunkEvent`] values remain the one
//! backend output type for both unary and streaming responses.

use std::sync::Arc;

use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};
use futures::StreamExt;
use http::StatusCode;

use super::app::AppState;
use super::response::{
    HttpResponse, error_response, json_response, json_typed_response, read_json, sse_encode,
};
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::openai::chat::{
    SamplingDefaults, chat_event_stream, chat_sampling, chat_sse_payload, prepare_chat_request,
    unary_chat,
};
use crate::api_server::core::openai::completions::{
    PromptSpec, SubmittedChoice, completion_event_stream, completion_prompt_specs,
    completion_sampling_params, completion_sse_payload, decode_prompt_echo, unary_completion,
};
use crate::api_server::core::openai::models::model_card;
use crate::api_server::core::openai::tools::dynamo_tool_choice;
use crate::api_server::core::openai::{error_payload_value, submit_generation, unix_seconds_u32};
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;

const MAX_OPENAI_CHOICES: usize = 4096;

/// The OpenAI error payload.
pub(super) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    error_payload_value(code.as_u16(), &message.into())
}

/// Form an OpenAI error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(super) fn openai_error(
    code: StatusCode,
    message: impl Into<String>,
    stream: bool,
) -> super::response::HttpResponse {
    error_response(code, error_payload(code, message), stream)
}

fn contains_media(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(contains_media),
        serde_json::Value::Object(object) => {
            object.keys().any(|key| {
                matches!(
                    key.as_str(),
                    "image_url" | "video_url" | "input_audio" | "audio_url" | "file"
                )
            }) || object.values().any(contains_media)
        }
        _ => false,
    }
}

/// `POST /v1/chat/completions` — chat-template preparation + generation.
pub(in crate::api_server) async fn chat_completions<B: http_body::Body>(
    state: Arc<AppState>,
    req: http::Request<B>,
) -> HttpResponse {
    let request = match read_json::<CreateChatCompletionRequest, _>(req).await {
        Ok(request) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text, false);
        }
    };
    if request.model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("The model `{}` does not exist", request.model),
            false,
        );
    }
    if request.messages.is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "messages cannot be empty", false);
    }
    if serde_json::to_value(&request.messages).is_ok_and(|messages| contains_media(&messages)) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "image, audio, video, and file message content is not supported",
            false,
        );
    }
    if request.n == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "n must be at least 1", false);
    }
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_completion_tokens must be positive",
            false,
        );
    }
    if request.modalities.as_ref().is_some_and(|modalities| {
        serde_json::to_value(modalities).is_ok_and(|value| value.to_string().contains("\"audio\""))
    }) || request.audio.is_some()
        || request.prediction.is_some()
        || request.web_search_options.is_some()
        || request.mm_processor_kwargs.is_some()
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "audio, prediction, web search, and multimodal inputs are not supported",
            false,
        );
    }
    #[allow(deprecated)]
    if request.function_call.is_some() || request.functions.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "deprecated function_call/functions are not supported; use tools and tool_choice",
            false,
        );
    }

    let tool_choice = dynamo_tool_choice(&request.tool_choice);
    let tools_enabled = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && tool_choice != DynamoToolChoice::None;
    let parser = tools_enabled
        .then(|| state.server_args.tool_call_parser.clone())
        .flatten();
    if tools_enabled && parser.is_none() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "tool calls require --tool-call-parser",
            false,
        );
    }
    // Python gates the split on `request.separate_reasoning` (default true);
    // the Dynamo request type has no such field, so it is always on when the
    // server was launched with `--reasoning-parser`.
    let reasoning_parser = state.server_args.reasoning_parser.clone();
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.function.name.clone(),
                parameters: tool.function.parameters.clone(),
                strict: tool.function.strict,
            })
            .collect::<Vec<_>>()
    });
    let tools_slice = tools.as_deref().unwrap_or_default();

    let (request, prompt) = match prepare_chat_request(&state, request).await {
        Ok(prepared) => prepared,
        Err(e) => return openai_error(e.http_status(), e.message, false),
    };

    let sampling = match chat_sampling(
        &request,
        SamplingDefaults::CHAT,
        parser.as_deref(),
        &tool_choice,
        tools_slice,
        request.parallel_tool_calls,
        &state.server_args,
    ) {
        Ok(sampling) => sampling,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, false);
        }
    };

    let stream = request.stream.unwrap_or(false);
    let n = request.n.unwrap_or(1) as usize;
    let want_logprobs = request.logprobs.unwrap_or(false);
    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let stream_tool_choice = request.tool_choice.clone();
    let uses_tool_call_structural_tag = sampling.structural_tag.is_some();
    let service_tier = request.service_tier;
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let model = request.model;
    let include_usage = request
        .stream_options
        .is_some_and(|options| options.include_usage)
        || state.server_args.stream_response_default_include_usage;
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(n);

    let mut prompt = Some(prompt);
    for index in 0..n {
        let rid = Rid::from_client(&format!("{response_id}-{index}"));
        let choice_prompt = if index + 1 == n {
            prompt.take().expect("last chat choice owns the prompt")
        } else {
            prompt
                .as_ref()
                .expect("chat prompt exists until the last choice")
                .clone()
        };
        let native = GenerateRequest {
            rid: rid.clone(),
            text: Some(choice_prompt),
            // Rendered templates own their special tokens — the pool must not
            // add another BOS/EOS (Python's `add_special_tokens=False`).
            skip_special_tokens: true,
            sampling_params: sampling.clone(),
            stream,
            return_logprob: want_logprobs,
            logprob_start_len: -1,
            top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
            return_text_in_logprobs: want_logprobs.then_some(true),
            ..Default::default()
        };
        let rx = match submit_generation(&state, native, &mut guard).await {
            Ok(rx) => rx,
            Err(e) => return openai_error(e.http_status(), e.message, stream),
        };
        submitted.push((index, rid, rx));
    }

    if stream {
        let event_stream = chat_event_stream(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            include_usage,
            parser,
            reasoning_parser,
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            parallel_tool_calls,
            service_tier,
        );
        sse_encode(event_stream.map(chat_sse_payload))
    } else {
        match unary_chat(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            parser,
            reasoning_parser,
            tools,
            parallel_tool_calls,
            service_tier,
        )
        .await
        {
            Ok(response) => json_typed_response(StatusCode::OK, &response),
            Err(e) => openai_error(e.http_status(), e.message, false),
        }
    }
}

/// `POST /v1/completions` — the legacy text-completion endpoint.
pub(in crate::api_server) async fn completions<B: http_body::Body>(
    state: Arc<AppState>,
    req: http::Request<B>,
) -> HttpResponse {
    let request = match read_json::<CreateCompletionRequest, _>(req).await {
        Ok(request) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text, false);
        }
    };
    let stream = request.stream.unwrap_or(false);
    let echo = request.echo.unwrap_or(false);
    let model = request.model.clone();
    if model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("The model `{model}` does not exist"),
            false,
        );
    }

    if request.prompt_embeds.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "prompt_embeds is not supported by the Rust frontend",
            false,
        );
    }
    if request.suffix.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "suffix is not supported by this model",
            false,
        );
    }
    if request.best_of.is_some_and(|best_of| best_of != 1) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "best_of values greater than 1 are not supported",
            false,
        );
    }
    if request.max_tokens == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_tokens must be positive",
            false,
        );
    }
    if request.n == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "n must be at least 1", false);
    }
    let prompts = match completion_prompt_specs(&request.prompt) {
        Ok(prompts) => prompts,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, &message, false);
        }
    };
    let mut sampling = match completion_sampling_params(&request) {
        Ok(sampling) => sampling,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, &message, false);
        }
    };
    if let Err(error) = sampling.normalize(
        state.server_args.skip_tokenizer_init,
        state.server_args.model_config.vocab_size,
    ) {
        return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false);
    }

    let n = request.n.unwrap_or(1) as usize;
    let choice_count = match prompts.len().checked_mul(n) {
        Some(count) if count <= MAX_OPENAI_CHOICES => count,
        _ => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("prompt count times n exceeds the maximum of {MAX_OPENAI_CHOICES}"),
                false,
            );
        }
    };
    let response_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(choice_count);

    for (prompt_index, prompt) in prompts.into_iter().enumerate() {
        let (text, input_ids, mut prompt_echo) = match prompt {
            PromptSpec::Text(text) => {
                let prompt_echo = if echo { text.clone() } else { String::new() };
                (Some(text), None, prompt_echo)
            }
            PromptSpec::TokenIds(input_ids) => (None, Some(input_ids), String::new()),
        };
        for sample_index in 0..n {
            let index = prompt_index * n + sample_index;
            let rid = Rid::from_client(&format!("{response_id}-{index}"));
            if echo
                && sample_index == 0
                && let Some(token_ids) = &input_ids
            {
                prompt_echo = match decode_prompt_echo(&state, token_ids.clone()).await {
                    Ok(echo) => echo,
                    Err(e) => return openai_error(e.http_status(), e.message, false),
                };
            }
            let native = GenerateRequest {
                rid: rid.clone(),
                text: text.clone(),
                input_ids: input_ids.clone(),
                sampling_params: sampling.clone(),
                stream,
                return_logprob: request.logprobs.is_some(),
                logprob_start_len: if echo && request.logprobs.is_some() {
                    0
                } else {
                    -1
                },
                top_logprobs_num: request.logprobs.unwrap_or(0) as i64,
                return_text_in_logprobs: request.logprobs.map(|_| true),
                ..Default::default()
            };
            let rx = match submit_generation(&state, native, &mut guard).await {
                Ok(rx) => rx,
                Err(e) => return openai_error(e.http_status(), e.message, stream),
            };
            submitted.push(SubmittedChoice {
                index,
                prompt_index,
                rid,
                echo: prompt_echo.clone(),
                rx,
            });
        }
    }

    if stream {
        let include_usage = request
            .stream_options
            .map(|o| o.include_usage)
            .unwrap_or(false)
            || state.server_args.stream_response_default_include_usage;
        let continuous_usage = request
            .stream_options
            .map(|o| o.continuous_usage_stats)
            .unwrap_or(false);
        let want_logprobs = request.logprobs.is_some();
        let s = completion_event_stream(
            submitted,
            guard,
            response_id,
            model,
            created,
            echo,
            want_logprobs,
            include_usage,
            continuous_usage,
        );
        sse_encode(s.map(completion_sse_payload))
    } else {
        match unary_completion(
            submitted,
            guard,
            response_id,
            model,
            created,
            echo,
            request.logprobs.is_some(),
        )
        .await
        {
            Ok(value) => json_response(StatusCode::OK, &value),
            Err(e) => openai_error(e.http_status(), e.message, false),
        }
    }
}

/// `GET /v1/models` — OpenAI-compatible model list. Served from `server_args`;
/// no scheduler round-trip.
pub(in crate::api_server) async fn available_models(state: Arc<AppState>) -> HttpResponse {
    let base = model_card(&state.server_args);
    json_response(
        StatusCode::OK,
        &serde_json::json!({ "object": "list", "data": [base] }),
    )
}

/// `GET /v1/models/{model}` — `model` arrives percent-decoded (the router owns
/// the decode, matching axum's `Path<String>`).
pub(in crate::api_server) async fn retrieve_model(
    state: Arc<AppState>,
    model: &str,
) -> HttpResponse {
    if model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("The model `{model}` does not exist"),
            false,
        );
    }
    json_response(StatusCode::OK, &model_card(&state.server_args))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::json;
    use tower::util::ServiceExt;

    use super::super::response::{HttpBody, HttpResponse, empty, full};
    use super::openai_error;
    use crate::api_server::core::openai::test_utils::senders;
    use crate::message::config::ServerArgs;
    use crate::tokenizer_manager::wiring::Senders;

    pub(super) fn server_args() -> Arc<ServerArgs> {
        Arc::new(ServerArgs {
            served_model_name: "model".into(),
            ..Default::default()
        })
    }

    pub(super) fn app_state(senders: Senders) -> Arc<super::AppState> {
        Arc::new(super::AppState {
            senders,
            response_buf: 8,
            api_key: None,
            server_args: server_args(),
            chat_formatter: None,
            response_activity: Default::default(),
        })
    }

    /// The full route table over the given state — no auth / access-log layers
    /// (those have their own tests in `api_server::layers`).
    pub(super) fn app(state: Arc<super::AppState>) -> axum::Router {
        super::super::app::router(state, None)
    }

    pub(super) fn senders_closed() -> Senders {
        // Dropping the receivers disconnects the channels; the senders stay
        // valid (moveable) but every send reports `Err`, the shutdown state
        // `submit` surfaces as a 503.
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(tm_rx);
        let (abort_tx, abort_rx) = flume::unbounded();
        drop(abort_rx);
        let (tok_tx, tok_rx) = flume::unbounded();
        drop(tok_rx);
        Senders {
            tok_manager_tx: tm_tx,
            abort_tx,
            tokenizer_tx: tok_tx,
            detokenizer_tx: vec![],
        }
    }

    /// Serve one request through the full service (extraction, routing); the
    /// axum response body boxes back into [`HttpBody`] so downstream helpers
    /// keep their types.
    pub(super) async fn oneshot(app: axum::Router, req: Request<HttpBody>) -> HttpResponse {
        let response: http::Response<axum::body::Body> = app.oneshot(req).await.unwrap();
        response.map(|body| {
            HttpBody::new(body.map_err(|e| -> std::convert::Infallible {
                panic!("test response body failed: {e}")
            }))
        })
    }

    pub(super) async fn post_json(
        app: axum::Router,
        path: &str,
        body: serde_json::Value,
    ) -> HttpResponse {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(full(body.to_string()))
            .unwrap();
        oneshot(app, req).await
    }

    pub(super) async fn body_json(response: HttpResponse) -> serde_json::Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The common StatusCode→error helper follows `error_response`'s shape:
    /// unary requests get the JSON error with its status; a committed stream gets
    /// 200 + one SSE error frame + `[DONE]`, and the frame carries the OpenAI
    /// error fields (`type`, `param`, `code`) that the SDKs dispatch on.
    #[tokio::test]
    async fn openai_error_response_covers_unary_and_sse() {
        let unary = openai_error(StatusCode::BAD_REQUEST, "bad input", false);
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let value = body_json(unary).await;
        assert_eq!(value["error"]["message"], "bad input");
        assert_eq!(value["error"]["type"], "BadRequestError");
        assert_eq!(value["error"]["code"], 400);
        assert!(value["error"]["param"].is_null());

        let streamed = openai_error(StatusCode::BAD_REQUEST, "bad input", true);
        assert_eq!(streamed.status(), StatusCode::OK);
        let bytes = streamed.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frame = text
            .split("\n\n")
            .next()
            .unwrap()
            .strip_prefix("data: ")
            .unwrap();
        let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
        assert_eq!(frame["error"]["message"], "bad input");
        assert_eq!(frame["error"]["type"], "BadRequestError");
        assert!(text.contains("[DONE]"));
    }

    #[tokio::test]
    async fn completions_handler_validates_before_submit() {
        let app_ = app(app_state(senders()));
        let cases = [
            (json!({"model": "other", "prompt": "hi"}), "unknown model"),
            (json!({"model": "model", "prompt": "hi", "n": 0}), "n=0"),
            (
                json!({"model": "model", "prompt": "hi", "max_tokens": 0}),
                "max_tokens=0",
            ),
            (json!({"model": "model", "prompt": ""}), "empty prompt"),
            (
                json!({"model": "model", "prompt": "hi", "best_of": 2}),
                "best_of>1",
            ),
            (
                json!({"model": "model", "prompt": "hi", "suffix": "x"}),
                "suffix",
            ),
            (
                json!({"model": "model", "prompt": "hi", "prompt_embeds": [[1.0]]}),
                "prompt_embeds",
            ),
        ];
        for (body, label) in cases {
            let response = post_json(app_.clone(), "/v1/completions", body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
        }
        // Malformed JSON → 400 (JsonRejection path).
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(full("not json"))
            .unwrap();
        let response = oneshot(app_.clone(), req).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        // A closed tm inbox (shutdown) surfaces as 503.
        let app_ = app(app_state(senders_closed()));
        let response = post_json(
            app_.clone(),
            "/v1/completions",
            json!({"model": "model", "prompt": "hi"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn chat_handler_validates_before_submit() {
        let app_ = app(app_state(senders()));
        let cases = [
            (
                json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}),
                "unknown model",
            ),
            (json!({"model": "model", "messages": []}), "empty messages"),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 0}),
                "n=0",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}]}),
                "media content",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "function_call": "auto"}),
                "deprecated function_call",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "audio": {"input_audio": {"data": "x", "format": "wav"}}}),
                "audio",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}),
                "max_completion_tokens=0",
            ),
        ];
        for (body, label) in cases {
            let response = post_json(app_.clone(), "/v1/chat/completions", body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
        }
        // A valid request with no loaded chat template → 400 (template gate).
        let response = post_json(
            app_.clone(),
            "/v1/chat/completions",
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn basic_openai_router_excludes_responses_api() {
        let app_ = app(app_state(senders()));
        let response = post_json(app_, "/v1/responses", json!({"input": "hi"})).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// A closed tm inbox with a *streaming* request must answer inside the
    /// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
    /// same `error_response` rule the native API applies), not a unary 503.
    #[tokio::test]
    async fn streaming_submit_failure_answers_inside_the_stream() {
        let app_ = app(app_state(senders_closed()));
        let response = post_json(
            app_,
            "/v1/completions",
            json!({"model": "model", "prompt": "hi", "stream": true}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frame = text
            .split("\n\n")
            .next()
            .unwrap()
            .strip_prefix("data: ")
            .unwrap();
        let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
        assert_eq!(frame["error"]["message"], "service unavailable");
        assert_eq!(frame["error"]["type"], "InternalServerError");
        assert_eq!(frame["error"]["code"], 503);
        assert!(text.contains("[DONE]"));
    }

    /// The HTTP-edge wire contract the de-axum swap must reproduce byte-for-byte:
    /// the extractor rejection texts (clients see them in 400 bodies), the SSE
    /// response headers, and the no-route / wrong-method statuses.
    #[tokio::test]
    async fn http_edge_wire_contract() {
        use http::header::CONTENT_TYPE;

        let mk_app = || app(app_state(senders()));

        // Malformed JSON: 400 with axum's syntax text incl. serde's position.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(full("{\"model\": }"))
            .unwrap();
        let res = oneshot(mk_app(), req).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let v = body_json(res).await;
        assert_eq!(
            v["error"]["message"],
            "Failed to parse the request body as JSON: model: expected value at line 1 column 11"
        );

        // Type mismatch: 400 with axum's data-error text.
        let res = post_json(
            mk_app(),
            "/v1/completions",
            serde_json::json!({"model": 3, "prompt": "hi"}),
        )
        .await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let v = body_json(res).await;
        assert_eq!(
            v["error"]["message"],
            "Failed to deserialize the JSON body into the target type: model: invalid type: integer `3`, expected a string at line 1 column 10"
        );

        // Missing JSON content type: 400 with axum's content-type text.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .body(full("{}"))
            .unwrap();
        let res = oneshot(mk_app(), req).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let v = body_json(res).await;
        assert_eq!(
            v["error"]["message"],
            "Expected request with `Content-Type: application/json`"
        );

        // Unknown path: bare 404. Known path, wrong method: bare 405 + Allow.
        let res = oneshot(
            mk_app(),
            Request::builder().uri("/nope").body(empty()).unwrap(),
        )
        .await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let res = oneshot(
            mk_app(),
            Request::builder()
                .method("GET")
                .uri("/v1/completions")
                .body(empty())
                .unwrap(),
        )
        .await;
        assert_eq!(res.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(
            res.headers()
                .get(http::header::ALLOW)
                .and_then(|v| v.to_str().ok()),
            Some("POST")
        );

        // A committed stream: 200 with the SSE headers axum set.
        let app_ = app(app_state(senders_closed()));
        let res = post_json(
            app_,
            "/v1/completions",
            serde_json::json!({"model": "model", "prompt": "hi", "stream": true}),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(
            res.headers()
                .get(http::header::CACHE_CONTROL)
                .and_then(|v| v.to_str().ok()),
            Some("no-cache")
        );
    }
}
