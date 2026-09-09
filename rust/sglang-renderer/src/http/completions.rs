//! HTTP completion adapter.

use super::{
    CompletionRequest, OpenAIHttpFrontend,
    error::{json_rejection_response, response_error},
    response::sse_response,
};
use crate::openai::{
    completions::{attach_streams, completion_event_stream, prepare_request, unary_completion},
    submission::submit_generate_requests,
};
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    response::{IntoResponse, Response},
    routing::post,
};
use std::sync::Arc;

pub(super) fn routes() -> Router<Arc<OpenAIHttpFrontend>> {
    Router::new().route("/v1/completions", post(completions))
}

async fn completions(
    State(state): State<Arc<OpenAIHttpFrontend>>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => return json_rejection_response(error),
    };
    let stream = request.stream.unwrap_or(false);
    let prepared = match prepare_request(&state.renderer, &state.generate_client, request).await {
        Ok(prepared) => prepared,
        Err(error) => return response_error(error, false),
    };
    let streams = match submit_generate_requests(&state.generate_client, prepared.requests).await {
        Ok(streams) => streams,
        Err(error) => return response_error(error, stream),
    };
    let submitted = attach_streams(prepared.metadata, streams);
    if stream {
        sse_response(
            completion_event_stream(
                submitted,
                prepared.response_id,
                prepared.model,
                prepared.created,
                prepared.echo,
                prepared.want_logprobs,
                prepared.include_usage,
                prepared.continuous_usage,
            ),
            |chunk| serde_json::to_string(&chunk).expect("OpenAI response must serialize"),
        )
    } else {
        match unary_completion(
            submitted,
            prepared.response_id,
            prepared.model,
            prepared.created,
            prepared.echo,
            prepared.want_logprobs,
        )
        .await
        {
            Ok(response) => Json(response).into_response(),
            Err(error) => response_error(error, false),
        }
    }
}
