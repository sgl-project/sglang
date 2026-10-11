//! HTTP chat completion adapter.

use super::{
    ChatCompletionRequest,
    error::{json_rejection_response, response_error},
    response::sse_response,
};
use crate::openai::chat::serialize_chat_stream_response;
use crate::openai::{OpenAIService, OperationResponse};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    response::{IntoResponse, Response},
    routing::post,
};
use std::sync::Arc;

pub(super) fn routes() -> Router<Arc<OpenAIService>> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

async fn chat_completions(
    State(state): State<Arc<OpenAIService>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => return json_rejection_response(error),
    };
    match state.chat(request).await {
        Ok(OperationResponse::Unary(response)) => Json(response).into_response(),
        Ok(OperationResponse::Stream(chunks)) => {
            sse_response(chunks, serialize_chat_stream_response)
        }
        Err(error) => response_error(error),
    }
}
