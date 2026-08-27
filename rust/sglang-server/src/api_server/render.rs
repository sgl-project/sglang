//! Standalone HTTP transport for the shared engine-free renderer service.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};

use super::openai::openai_error;
use crate::renderer::{
    PreparedGenerateRequest, RenderServiceError, RendererService, render_http_status,
};

#[derive(Clone)]
pub(super) struct RenderState {
    renderer: Arc<RendererService>,
}

impl RenderState {
    pub(super) fn new(renderer: Arc<RendererService>) -> Self {
        Self { renderer }
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

fn render_error(error: RenderServiceError) -> Response {
    let status = StatusCode::from_u16(render_http_status(&error))
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    openai_error(status, error.to_string(), false)
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
    match state
        .renderer
        .prepare_chat(&mut request, &response_id)
        .await
    {
        Ok(prepared) => {
            // Render-only callers need the prepared generation request, not the
            // live parser state used after inference. Dropping `output` here is
            // an explicit property of the preprocess-only contract.
            let sglang_renderer::LoweredChat {
                mut requests,
                output: _,
            } = prepared;
            Json(PreparedGenerateRequest::from(
                requests
                    .pop()
                    .expect("lowered chat request contains one choice"),
            ))
            .into_response()
        }
        Err(error) => render_error(error),
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
        .prepare_completions(&request, &response_id)
        .await
    {
        Ok(requests) => Json(
            requests
                .into_iter()
                .map(PreparedGenerateRequest::from)
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(error) => render_error(error),
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

    use crate::message::config::{ModelConfig, ServerArgs};
    use crate::message::request::GenerateBody;
    use crate::message::types::TokenIds;
    use crate::renderer::{PreprocessJob, new_renderer_service};
    use crate::runtime::Runnable;
    use crate::tokenizer_manager::to_scheduler::Limits;
    use crate::tokenizer_manager::tokenizer::{TextTokenizer, TokenizerWorker};

    struct WordTokenizer;

    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, sglang_renderer::RendererError> {
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    struct TestApp {
        router: Option<Router<()>>,
        workers: Vec<std::thread::JoinHandle<()>>,
    }

    impl TestApp {
        async fn request(mut self, request: Request<Body>) -> Response {
            let response = self
                .router
                .take()
                .expect("test router")
                .oneshot(request)
                .await
                .unwrap();
            for worker in self.workers.drain(..) {
                worker.join().unwrap();
            }
            response
        }
    }

    fn test_app() -> TestApp {
        let server_args = Arc::new(ServerArgs {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            tokenizer_worker_num: 2,
            chat_template: Some("chatml".into()),
            model_config: ModelConfig {
                context_len: 64,
                vocab_size: 100,
                ..Default::default()
            },
            ..Default::default()
        });
        let limits = Limits::from(&*server_args);
        let tokenizer: Arc<dyn TextTokenizer> = Arc::new(WordTokenizer);
        let (jobs, worker_jobs) = flume::bounded::<PreprocessJob>(8);
        let workers = (0..server_args.tokenizer_worker_num)
            .map(|worker_index| {
                let worker = TokenizerWorker::new(
                    worker_jobs.clone(),
                    None,
                    tokenizer.clone(),
                    limits.clone(),
                );
                std::thread::Builder::new()
                    .name(format!("test-renderer-{worker_index}"))
                    .spawn(move || worker.run())
                    .unwrap()
            })
            .collect();
        let renderer = Arc::new(new_renderer_service(server_args, jobs));
        TestApp {
            router: Some(routes(RenderState::new(renderer))),
            workers,
        }
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
        let response = test_app().request(request).await;
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
        let response = test_app().request(request).await;
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
        let response = test_app().request(request).await;
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
        let response = test_app().request(request).await;
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
        let response = test_app().request(request).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
