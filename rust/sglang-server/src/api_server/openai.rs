//! HTTP extraction and JSON/SSE framing for the shared OpenAI operations.

use super::app::AppState;
pub(super) use crate::openai::unix_seconds_u32;
use crate::openai::{self, ChatRequest, OpenAiResponse};
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
use dynamo_protocols::types::CreateCompletionRequest;
use futures::StreamExt;
use std::convert::Infallible;
use std::sync::Arc;

mod models;

pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .merge(models::routes())
        .merge(chat_routes())
        .merge(completion_routes())
}

pub(crate) fn chat_routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

pub(crate) fn completion_routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/completions", post(completions))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(request)) => openai::chat_completions(&state, request)
            .await
            .into_response(),
        Err(error) => openai_error(StatusCode::BAD_REQUEST, error.body_text(), false),
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CreateCompletionRequest>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(request)) => openai::completions(&state, request).await.into_response(),
        Err(error) => openai_error(StatusCode::BAD_REQUEST, error.body_text(), false),
    }
}

impl IntoResponse for OpenAiResponse {
    fn into_response(self) -> Response {
        match self {
            Self::Json(bytes) => (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                bytes,
            )
                .into_response(),
            Self::Stream(stream) => {
                Sse::new(stream.map(|item| Ok::<_, Infallible>(Event::default().data(item.data))))
                    .into_response()
            }
            Self::Error {
                code,
                message,
                stream,
                ..
            } => openai_error(code, message, stream),
        }
    }
}

pub(crate) fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    crate::utils::response::error_response(code, openai::error_payload(code, message), stream)
}
