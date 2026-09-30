//! HTTP completion adapter.

use super::{
    CompletionRequest,
    error::{json_rejection_response, response_error},
    response::sse_response,
};
use crate::openai::{OpenAIService, OperationResponse};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    response::{IntoResponse, Response},
    routing::post,
};
use std::sync::Arc;

pub(super) fn routes() -> Router<Arc<OpenAIService>> {
    Router::new().route("/v1/completions", post(completions))
}

async fn completions(
    State(state): State<Arc<OpenAIService>>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => return json_rejection_response(error),
    };
    match state.complete(request).await {
        Ok(OperationResponse::Unary(response)) => Json(response).into_response(),
        Ok(OperationResponse::Stream(chunks)) => sse_response(chunks, |chunk| {
            serde_json::to_string(&chunk).expect("OpenAI response must serialize")
        }),
        Err(error) => response_error(error),
    }
}
