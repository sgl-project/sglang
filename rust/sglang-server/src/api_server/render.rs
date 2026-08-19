//! Standalone text preprocessing HTTP surface. OpenAI adapters lower requests
//! into SGLang's native text-generate shape, then tokenizer primitives shared
//! with inference produce a request accepted by `/generate`.

use std::{collections::BTreeMap, sync::Arc};

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};
use futures::future::try_join_all;
use serde::Serialize;
use tokio::sync::Semaphore;

use super::openai::{ChatFormatter, lower_chat_requests, lower_completion_requests, openai_error};
use crate::message::config::ServerArgs;
use crate::message::request::GenerateRequest;
use crate::message::sampling::SamplingParams;
use crate::tokenizer_manager::to_scheduler::{
    Limits, check_total_tokens, validate_generate_request,
};
use crate::tokenizer_manager::tokenizer::{TextTokenizer, tokenize_generate_request};
use crate::utils::error::Error;

#[derive(Clone)]
pub(super) struct RenderState {
    server_args: Arc<ServerArgs>,
    chat_formatter: Option<ChatFormatter>,
    tokenizer: Arc<dyn TextTokenizer>,
    auto_specials: Arc<[i32]>,
    limits: Limits,
    permits: Arc<Semaphore>,
}

impl RenderState {
    pub(super) fn new(
        server_args: Arc<ServerArgs>,
        chat_formatter: Option<ChatFormatter>,
        tokenizer: Arc<dyn TextTokenizer>,
        limits: Limits,
    ) -> Self {
        let permits = Arc::new(Semaphore::new(server_args.tokenizer_worker_num.max(1)));
        let auto_specials = tokenizer.auto_specials().into();
        Self {
            server_args,
            chat_formatter,
            tokenizer,
            auto_specials,
            limits,
            permits,
        }
    }
}

pub(super) fn routes(state: RenderState) -> Router<()> {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions/render", post(render_chat))
        .route("/v1/completions/render", post(render_completions))
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::OK
}

/// Text-only token-in request for the current public `/generate` endpoint.
/// Renderer scope exclusions such as multimodal features, cache identity,
/// priority, and extra keys are documented at the HTTP boundary.
#[derive(Debug, Serialize)]
struct PreparedGenerateRequest {
    rid: String,
    input_ids: Vec<i32>,
    sampling_params: RenderedSamplingParams,
    stream: bool,
    return_logprob: bool,
    logprob_start_len: i64,
    top_logprobs_num: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_ids_logprob: Option<Vec<i32>>,
    return_hidden_states: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_text_in_logprobs: Option<bool>,
}

/// Public `/generate` sampling shape. The internal `SamplingParams` serializer
/// contains scheduler-only normalized fields, so the render boundary folds
/// normalized stops back into the aliases accepted by the public endpoint.
#[derive(Debug, Serialize)]
struct RenderedSamplingParams {
    max_new_tokens: Option<i64>,
    stop: Vec<String>,
    stop_token_ids: Option<Vec<i64>>,
    stop_regex: Vec<String>,
    temperature: f64,
    top_p: f64,
    top_k: i64,
    min_p: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    repetition_penalty: f64,
    min_new_tokens: i64,
    n: i64,
    json_schema: Option<String>,
    regex: Option<String>,
    ebnf: Option<String>,
    structural_tag: Option<String>,
    ignore_eos: bool,
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
    no_stop_trim: bool,
    stream_interval: Option<i64>,
    logit_bias: Option<BTreeMap<String, f64>>,
    sampling_seed: Option<i64>,
    custom_params: Option<serde_json::Value>,
}

impl From<SamplingParams> for RenderedSamplingParams {
    fn from(params: SamplingParams) -> Self {
        Self {
            max_new_tokens: params.max_new_tokens,
            stop: params.stop_strs,
            stop_token_ids: params.stop_token_ids,
            stop_regex: params.stop_regex_strs,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            min_p: params.min_p,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            repetition_penalty: params.repetition_penalty,
            min_new_tokens: params.min_new_tokens,
            n: params.n,
            json_schema: params.json_schema,
            regex: params.regex,
            ebnf: params.ebnf,
            structural_tag: params.structural_tag,
            ignore_eos: params.ignore_eos,
            skip_special_tokens: params.skip_special_tokens,
            spaces_between_special_tokens: params.spaces_between_special_tokens,
            no_stop_trim: params.no_stop_trim,
            stream_interval: params.stream_interval,
            logit_bias: params.logit_bias,
            sampling_seed: params.sampling_seed,
            custom_params: params.custom_params,
        }
    }
}

async fn render_chat(
    State(state): State<RenderState>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Response {
    let mut request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    if request.n.is_some_and(|n| n > 1) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "the standalone chat renderer currently requires n=1",
            false,
        );
    }
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let native_requests = match lower_chat_requests(
        &state.server_args,
        state.chat_formatter.clone(),
        &mut request,
        &response_id,
    )
    .await
    {
        Ok(requests) => requests,
        Err(response) => return response,
    };
    let native = native_requests
        .into_iter()
        .next()
        .expect("lowered chat request contains one choice");
    match prepare_one(&state, native).await {
        Ok(prepared) => Json(prepared).into_response(),
        Err(response) => response,
    }
}

async fn render_completions(
    State(state): State<RenderState>,
    body: Result<Json<CreateCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    let response_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let native_requests =
        match lower_completion_requests(&state.server_args, &request, &response_id) {
            Ok(requests) => requests,
            Err(message) => return openai_error(StatusCode::BAD_REQUEST, message, false),
        };
    let futures = native_requests
        .into_iter()
        .map(|request| prepare_one(&state, request));
    match try_join_all(futures).await {
        Ok(prepared) => Json(prepared).into_response(),
        Err(response) => response,
    }
}

async fn prepare_one(
    state: &RenderState,
    request: GenerateRequest,
) -> Result<PreparedGenerateRequest, Response> {
    let permit = state.permits.clone().acquire_owned().await.map_err(|_| {
        openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "renderer is shutting down",
            false,
        )
    })?;
    let tokenizer = state.tokenizer.clone();
    let auto_specials = state.auto_specials.clone();
    let limits = state.limits.clone();
    let mut prepared = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        prepare_text_request(request, tokenizer.as_ref(), &auto_specials, &limits)
    })
    .await
    .map_err(|error| {
        tracing::error!(%error, "render preprocessing task failed");
        openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "render preprocessing task failed",
            false,
        )
    })?
    .map_err(|error| {
        let status =
            StatusCode::from_u16(error.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        openai_error(status, error.to_string(), false)
    })?;
    Ok(PreparedGenerateRequest {
        rid: prepared.rid.client_facing().to_owned(),
        input_ids: prepared
            .input_ids
            .take()
            .expect("text preparation always produces input_ids"),
        sampling_params: prepared.sampling_params.into(),
        stream: prepared.stream,
        return_logprob: prepared.return_logprob,
        logprob_start_len: prepared.logprob_start_len,
        top_logprobs_num: prepared.top_logprobs_num,
        token_ids_logprob: prepared.token_ids_logprob,
        return_hidden_states: prepared.return_hidden_states,
        return_text_in_logprobs: prepared.return_text_in_logprobs,
    })
}

fn prepare_text_request(
    mut request: GenerateRequest,
    tokenizer: &dyn TextTokenizer,
    auto_specials: &[i32],
    limits: &Limits,
) -> Result<GenerateRequest, Error> {
    validate_generate_request(&request.rid, &request, limits)?;
    if request.has_multimodal() {
        return Err(Error::Validation(
            "multimodal inputs are not supported by the standalone renderer".into(),
        ));
    }
    request
        .sampling_params
        .normalize(limits.skip_tokenizer_init, limits.vocab_size)?;
    if !request.already_tokenized() {
        tokenize_generate_request(&mut request, tokenizer, auto_specials)?;
    }
    check_total_tokens(&mut request, limits)?;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt;

    use crate::message::request::GenerateBody;
    use crate::message::types::TokenIds;

    struct WordTokenizer;

    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    fn test_app(enable_chat: bool) -> Router<()> {
        let server_args = ServerArgs {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            tokenizer_worker_num: 2,
            chat_template: Some("chatml".into()),
            model_config: crate::message::config::ModelConfig {
                context_len: 64,
                vocab_size: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let chat_formatter = enable_chat
            .then(|| super::super::openai::load_chat_support(&server_args))
            .flatten();
        let limits = Limits::from(&server_args);
        routes(RenderState::new(
            Arc::new(server_args),
            chat_formatter,
            Arc::new(WordTokenizer),
            limits,
        ))
    }

    #[tokio::test]
    async fn completion_render_replays_as_generate_body() {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions/render")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "model",
                    "prompt": ["one two", "three"],
                    "max_tokens": 5,
                    "stream": true,
                    "stop": "END"
                })
                .to_string(),
            ))
            .unwrap();
        let response = test_app(false).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        let rendered = body.as_array().expect("completion render is an array");
        assert_eq!(rendered.len(), 2);
        assert_eq!(rendered[0]["input_ids"], serde_json::json!([7, 7]));
        assert_eq!(rendered[1]["input_ids"], serde_json::json!([7]));
        assert_eq!(rendered[0]["stream"], true);

        for value in rendered {
            let generate: GenerateBody = serde_json::from_value(value.clone()).unwrap();
            let (requests, is_batch) = generate.into_requests().unwrap();
            assert!(!is_batch);
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].sampling_params.max_new_tokens, Some(5));
            assert!(matches!(
                requests[0].sampling_params.stop.as_ref().unwrap(),
                crate::message::types::OneOrMany::Many(stops) if stops.len() == 1
            ));
        }
    }

    #[tokio::test]
    async fn chat_render_returns_one_replayable_generate_request() {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions/render")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "model",
                    "messages": [{"role": "user", "content": "hello renderer"}],
                    "max_completion_tokens": 5
                })
                .to_string(),
            ))
            .unwrap();
        let response = test_app(true).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert!(value.is_object());
        assert!(value["rid"].as_str().unwrap().starts_with("chatcmpl-"));
        assert!(!value["input_ids"].as_array().unwrap().is_empty());
        let generate: GenerateBody = serde_json::from_value(value).unwrap();
        assert_eq!(generate.into_requests().unwrap().0.len(), 1);
    }

    #[tokio::test]
    async fn chat_render_rejects_multiple_choices() {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions/render")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "n": 2
                })
                .to_string(),
            ))
            .unwrap();
        let response = test_app(true).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(
            body["error"]["message"],
            "the standalone chat renderer currently requires n=1"
        );
    }

    #[tokio::test]
    async fn completion_render_enforces_context_length() {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions/render")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "model",
                    "prompt": "one two three four",
                    "max_tokens": 64
                })
                .to_string(),
            ))
            .unwrap();
        let response = test_app(false).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("maximum context length"))
        );
    }

    #[tokio::test]
    async fn normal_inference_routes_are_not_mounted() {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = test_app(false).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
