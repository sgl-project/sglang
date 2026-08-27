//! Standalone HTTP transport for prepared generation requests.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};

use crate::RendererService;

use super::error::{openai_error, renderer_status};

#[derive(Clone)]
struct RenderState {
    renderer: Arc<RendererService>,
}

pub(super) fn routes(renderer: Arc<RendererService>) -> Router<()> {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions/render", post(render_chat))
        .route("/v1/completions/render", post(render_completions))
        .with_state(RenderState { renderer })
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn render_chat(
    State(state): State<RenderState>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
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
    match state.renderer.prepare_chat(request, &response_id).await {
        Ok(mut prepared_requests) => Json(
            prepared_requests
                .pop()
                .expect("prepared chat request contains one choice"),
        )
        .into_response(),
        Err(error) => openai_error(renderer_status(&error), error.to_string(), false),
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
    match state
        .renderer
        .prepare_completions(request, &response_id)
        .await
    {
        Ok(prepared_requests) => Json(prepared_requests).into_response(),
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
    use futures::future::BoxFuture;
    use tower::ServiceExt;

    use crate::{
        OpenAIRequestLowerer, RendererConfig, RendererError, RendererLimits, SamplingDefaults,
        TextPrompt, TextRequest, TokenIdsRequest, TokenizationBackend,
    };

    struct WordTokenizer;

    impl TokenizationBackend for WordTokenizer {
        fn tokenize(
            &self,
            request: TextRequest,
        ) -> BoxFuture<'static, Result<TokenIdsRequest, RendererError>> {
            Box::pin(async move {
                Ok(TokenIdsRequest {
                    rid: request.rid,
                    input_ids: match request.prompt {
                        TextPrompt::Text(text) => text.split_whitespace().map(|_| 7).collect(),
                        TextPrompt::TokenIds(_) => {
                            panic!("token-ID prompts bypass the tokenizer backend")
                        }
                    },
                    options: request.options,
                })
            })
        }
    }

    fn app() -> Router<()> {
        let lowerer = OpenAIRequestLowerer::new(RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 100,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 100,
                context_len: 64,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        });
        routes(Arc::new(RendererService::new(
            lowerer,
            Arc::new(WordTokenizer),
        )))
    }

    #[tokio::test]
    async fn completion_render_returns_prepared_token_inputs() {
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
                            "max_tokens": 5
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
        assert_eq!(body[1]["input_ids"], serde_json::json!([7]));
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
}
