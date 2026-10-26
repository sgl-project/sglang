//! Ollama-compatible routes (`/api/chat`, `/api/generate`, `/api/tags`,
//! `/api/show`, and the root) over the shared `api_server::ollama` adapter:
//! JSON bodies in, NDJSON frames out. Route paths are the
//! `SGLANG_OLLAMA_*_ROUTE` env overrides Python reads.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Body,
    extract::{State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::StreamExt;
use serde::Serialize;
use sglang_api_types::api::v1 as api;

use super::app::AppState;
use super::openai::openai_error;
use crate::api_server::core::CoreError;
use crate::api_server::ollama::{Frames, Ollama};

fn route_path(env: &str, default: &str) -> String {
    std::env::var(env)
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

pub(super) fn routes() -> Router<Arc<AppState>> {
    let mut router = Router::new()
        .route(
            &route_path("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat"),
            post(chat),
        )
        .route(
            &route_path("SGLANG_OLLAMA_GENERATE_ROUTE", "/api/generate"),
            post(generate),
        )
        .route(
            &route_path("SGLANG_OLLAMA_TAGS_ROUTE", "/api/tags"),
            get(tags),
        )
        .route(
            &route_path("SGLANG_OLLAMA_SHOW_ROUTE", "/api/show"),
            post(show),
        );
    // Python mounts an Ollama root only when asked for; the default root
    // names SGLang. FastAPI renders a `str` body as JSON, quotes included.
    router = match std::env::var("SGLANG_OLLAMA_ROOT_ROUTE") {
        Ok(path) if !path.is_empty() => {
            router.route(&path, get(|| async { Json("Ollama is running") }))
        }
        _ => router.route("/", get(|| async { Json("SGLang is running") })),
    };
    router
}

fn adapter(state: &AppState) -> Ollama<'_> {
    Ollama {
        core: &state.core,
        chat_formatter: state.chat_formatter.as_ref(),
        server_args: &state.server_args,
    }
}

fn error(error: CoreError) -> Response {
    openai_error(error.http_status(), error.to_string(), false)
}

fn bad_request(rejection: JsonRejection) -> Response {
    openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false)
}

/// Streamed: one NDJSON line per frame; a failure ends the stream where
/// Python's generator raised. Unary: the one frame as the JSON body.
async fn respond<T: Serialize + Send + 'static>(frames: Frames<T>, stream: bool) -> Response {
    if stream {
        let lines = frames
            .take_while(|frame| futures::future::ready(frame.is_ok()))
            .map(|frame| {
                let frame = frame.expect("errors end the stream above");
                Ok::<_, Infallible>(format!(
                    "{}\n",
                    serde_json::to_string(&frame).expect("api frame serializes")
                ))
            });
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            Body::from_stream(lines),
        )
            .into_response();
    }
    let mut frames = frames;
    match frames.next().await {
        Some(Ok(frame)) => Json(frame).into_response(),
        Some(Err(failure)) => error(failure),
        None => error(CoreError::ResponseTruncated),
    }
}

async fn chat(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::OllamaChatRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return bad_request(rejection),
    };
    let stream = request.stream_or_default();
    match adapter(&state).chat(request).await {
        Ok(frames) => respond(frames, stream).await,
        Err(failure) => error(failure),
    }
}

async fn generate(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::OllamaGenerateRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return bad_request(rejection),
    };
    let stream = request.stream_or_default();
    match adapter(&state).generate(request).await {
        Ok(frames) => respond(frames, stream).await,
        Err(failure) => error(failure),
    }
}

async fn tags(State(state): State<Arc<AppState>>) -> Response {
    Json(adapter(&state).tags().await).into_response()
}

async fn show(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::OllamaShowRequest>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(request)) => Json(adapter(&state).show(request)).into_response(),
        Err(rejection) => bad_request(rejection),
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::to_bytes,
        http::{Request as HttpRequest, header::CONTENT_TYPE},
    };
    use tower::ServiceExt;

    use super::*;
    use crate::api_server::http::openai::test_utils::answering_app_state;

    async fn post(state: Arc<AppState>, uri: &str, body: &str) -> (StatusCode, String) {
        let response = routes()
            .with_state(state)
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(uri)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    /// A unary chat answers Python's `OllamaChatResponse` shape with the
    /// token counts, and the streamed form is one NDJSON frame per delta plus
    /// a `done` frame.
    #[tokio::test]
    async fn chat_answers_ollama_shapes_unary_and_streamed() {
        let state = answering_app_state();
        let (status, body) = post(
            state.clone(),
            "/api/chat",
            r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": false}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            json["message"],
            serde_json::json!({"role": "assistant", "content": "ok"})
        );
        assert_eq!(json["done"], true);
        assert_eq!(json["done_reason"], "stop");
        assert_eq!(json["prompt_eval_count"], 5);
        assert_eq!(json["eval_count"], 1);
        assert!(json["total_duration"].is_i64());
        assert!(
            json.get("load_duration").is_none(),
            "unknown durations are left out"
        );

        let (status, body) = post(
            state,
            "/api/chat",
            r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let frames: Vec<serde_json::Value> = body
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(frames.len(), 2, "{body}");
        assert_eq!(frames[0]["message"]["content"], "ok");
        assert_eq!(frames[0]["done"], false);
        assert!(frames[0].get("eval_count").is_none());
        assert_eq!(frames[1]["message"]["content"], "");
        assert_eq!(frames[1]["done"], true);
        assert_eq!(frames[1]["done_reason"], "stop");
        assert_eq!(
            frames[1]["eval_count"], 1,
            "the done frame carries the counts"
        );
    }

    /// `/api/generate` prepends `system`, answers an empty prompt with a bare
    /// done frame without generating, and `/api/show` and `/api/tags` render
    /// the generated card types.
    #[tokio::test]
    async fn generate_show_and_tags_follow_python() {
        let state = answering_app_state();
        let (status, body) = post(
            state.clone(),
            "/api/generate",
            r#"{"model": "m", "prompt": "  ", "stream": false}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["response"], "");
        assert_eq!(json["done"], true);
        assert!(json.get("eval_count").is_none());

        let (status, body) = post(
            state.clone(),
            "/api/generate",
            r#"{"model": "m", "prompt": "hi", "system": "be brief", "stream": false}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["response"], "ok");
        assert_eq!(json["prompt_eval_count"], 5);

        let (status, body) = post(
            state.clone(),
            "/api/show",
            r#"{"model": "org/Llama-3-Instruct"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["details"]["family"], "Llama-3");
        assert_eq!(json["model_info"]["general.name"], "org/Llama-3-Instruct");
        assert!(json["model_info"]["Llama-3.context_length"].is_u64());

        let (status, body) = post(state.clone(), "/api/show", r#"{"model": 3}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

        let response = routes()
            .with_state(state)
            .oneshot(
                HttpRequest::builder()
                    .method("GET")
                    .uri("/api/tags")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["models"][0]["details"]["format"], "sglang");
        assert!(
            json["models"][0].get("families").is_none()
                && json["models"][0]["details"].get("families").is_none(),
            "tags carry the three details Python fills"
        );
        assert!(
            json["models"][0]["digest"]
                .as_str()
                .unwrap()
                .starts_with("sha256:sglang")
        );
    }
}
