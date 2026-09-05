//! Standalone HTTP transport for token-only generation requests.

use std::sync::Arc;

use crate::{RendererService, engine::HttpGenerateClient};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dynamo_protocols::types::Prompt;

use super::{
    ChatCompletionRequest, CompletionRequest,
    error::{json_rejection_response, openai_error, renderer_status},
    protocol::{
        lower_chat_request, lower_text_completion_request, lower_token_ids_completion_request,
    },
};

pub(super) fn routes(renderer: Arc<RendererService>) -> Router<()> {
    Router::new()
        .route("/v1/chat/completions/render", post(render_chat))
        .route("/v1/completions/render", post(render_completions))
        .with_state(renderer)
}

pub(super) fn health_route() -> Router<()> {
    Router::new().route("/health", get(health))
}

pub(super) fn engine_health_route(generate_client: HttpGenerateClient) -> Router<()> {
    Router::new()
        .route("/health", get(engine_health))
        .with_state(generate_client)
}

pub(super) fn readiness_route() -> Router<()> {
    Router::new().route("/_sglang_renderer/ready", get(readiness))
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn engine_health(State(generate_client): State<HttpGenerateClient>) -> StatusCode {
    match generate_client.health_status().await {
        Ok(status) => status,
        Err(error) => {
            tracing::warn!(message = %error.message, "engine health check failed");
            StatusCode::SERVICE_UNAVAILABLE
        }
    }
}

async fn readiness() -> impl IntoResponse {
    (StatusCode::NO_CONTENT, [("x-sglang-renderer", "ready")])
}

async fn render_chat(
    State(renderer): State<Arc<RendererService>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let extended = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return json_rejection_response(rejection),
    };
    if extended.n.is_some_and(|n| n > 1) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "the standalone chat renderer currently requires n=1",
            false,
        );
    }
    let (_, chat_request) = match lower_chat_request(renderer.config(), extended) {
        Ok(request) => request,
        Err(error) => {
            return openai_error(renderer_status(&error), error.to_string(), false);
        }
    };
    match renderer.prepare_chat(chat_request).await {
        Ok(mut chat) => Json(
            chat.requests
                .pop()
                .expect("chat generation contains one request"),
        )
        .into_response(),
        Err(error) => openai_error(renderer_status(&error), error.to_string(), false),
    }
}

async fn render_completions(
    State(renderer): State<Arc<RendererService>>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let extended = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return json_rejection_response(rejection),
    };
    let request = extended;
    let text_prompt = matches!(&request.prompt, Prompt::String(_) | Prompt::StringArray(_));
    let requests = if text_prompt {
        let (_, requests) = match lower_text_completion_request(renderer.config(), &request) {
            Ok(requests) => requests,
            Err(error) => {
                return openai_error(renderer_status(&error), error.to_string(), false);
            }
        };
        renderer.prepare_text_request_groups(requests).await
    } else {
        let (_, requests) = match lower_token_ids_completion_request(renderer.config(), &request) {
            Ok(requests) => requests,
            Err(error) => {
                return openai_error(renderer_status(&error), error.to_string(), false);
            }
        };
        renderer.prepare_token_ids_requests(requests)
    };
    match requests {
        Ok(requests) => Json(requests).into_response(),
        Err(error) => openai_error(renderer_status(&error), error.to_string(), false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt;

    use crate::{RendererConfig, RendererError, RendererLimits, SamplingDefaults, TextTokenizer};

    struct WordTokenizer;

    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<i32>, RendererError> {
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    fn app() -> Router<()> {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 100,
                context_len: 64,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        routes(Arc::new(RendererService::with_tokenizer(
            config,
            Arc::new(WordTokenizer),
            2,
            2,
        )))
    }

    #[tokio::test]
    async fn completion_render_returns_token_only_generate_requests() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions/render")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "model",
                            "prompt": ["one two", "three"],
                            "n": 2,
                            "max_tokens": 5,
                            "top_k": 17,
                            "min_p": 0.2,
                            "min_tokens": 3,
                            "stop_regex": "END[0-9]",
                            "rid": "request-id",
                            "cache_salt": "tenant-a",
                            "extra_key": "interactive",
                            "priority": 7,
                            "bootstrap_host": "prefill",
                            "bootstrap_port": 8998,
                            "bootstrap_room": 42,
                            "routed_dp_rank": 2,
                            "disagg_prefill_dp_rank": 1
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(body[0]["input_ids"], serde_json::json!([7, 7]));
        assert_eq!(body[1]["input_ids"], serde_json::json!([7, 7]));
        assert_eq!(body[2]["input_ids"], serde_json::json!([7]));
        assert_eq!(body[3]["input_ids"], serde_json::json!([7]));
        assert!(
            body.as_array()
                .unwrap()
                .iter()
                .all(|request| request.get("text").is_none())
        );
        assert_eq!(body[0]["sampling_params"]["top_k"], 17);
        assert_eq!(body[0]["sampling_params"]["min_p"], 0.2);
        assert_eq!(body[0]["sampling_params"]["min_new_tokens"], 3);
        assert_eq!(
            body[0]["sampling_params"]["stop_regex"],
            serde_json::json!(["END[0-9]"])
        );
        assert_eq!(body[0]["rid"], "request-id-0");
        assert_eq!(body[0]["model"], "model");
        assert_eq!(body[0]["cache_salt"], "tenant-a");
        assert_eq!(body[0]["extra_key"], "interactive");
        assert_eq!(body[0]["priority"], 7);
        assert_eq!(body[0]["bootstrap_host"], "prefill");
        assert_eq!(body[0]["bootstrap_port"], 8998);
        assert_eq!(body[0]["bootstrap_room"], 42);
        assert_eq!(body[0]["routed_dp_rank"], 2);
        assert_eq!(body[0]["disagg_prefill_dp_rank"], 1);
        assert_eq!(body[1]["rid"], "request-id-1");
        assert_eq!(body[2]["rid"], "request-id-2");
        assert_eq!(body[3]["rid"], "request-id-3");
        assert_eq!(body[3]["cache_salt"], "tenant-a");
    }

    #[tokio::test]
    async fn chat_render_rejects_multiple_choices() {
        let response = app()
            .oneshot(
                Request::builder()
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
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn render_rejects_unimplemented_stateful_fields() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions/render")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "model",
                            "prompt": "hello",
                            "session_id": "session"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
