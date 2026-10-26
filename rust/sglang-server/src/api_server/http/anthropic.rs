//! Anthropic Messages routes (`/v1/messages`, `/v1/messages/count_tokens`)
//! over the shared `api_server::anthropic` adapter: JSON bodies in, the
//! Anthropic error envelope out, and SSE events named by their type.

use std::convert::Infallible;
use std::sync::Arc;

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
use futures::StreamExt;
use sglang_api_types::api::v1 as api;

use super::app::AppState;
use crate::api_server::anthropic::{Anthropic, AnthropicFailure, event_name};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
}

fn adapter(state: &AppState) -> Anthropic<'_> {
    Anthropic {
        core: &state.core,
        chat_formatter: state.chat_formatter.as_ref(),
        server_args: &state.server_args,
    }
}

fn failure(failure: &AnthropicFailure) -> Response {
    (failure.status, Json(failure.envelope())).into_response()
}

/// A body that does not decode is Python's 400 `invalid_request_error`.
fn rejected(rejection: JsonRejection) -> Response {
    failure(&AnthropicFailure::from_status(
        StatusCode::BAD_REQUEST,
        rejection.body_text(),
    ))
}

async fn messages(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::AnthropicMessagesRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return rejected(rejection),
    };
    if !request.stream_or_default() {
        return match adapter(&state).messages(request).await {
            Ok(response) => Json(response).into_response(),
            Err(error) => failure(&error),
        };
    }
    match adapter(&state).messages_stream(request).await {
        Ok(events) => Sse::new(events.map(|event| {
            Ok::<_, Infallible>(
                Event::default()
                    .event(event_name(&event))
                    .data(serde_json::to_string(&event).expect("api event serializes")),
            )
        }))
        .into_response(),
        Err(error) => failure(&error),
    }
}

async fn count_tokens(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::AnthropicCountTokensRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return rejected(rejection),
    };
    match adapter(&state).count_tokens(request).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => failure(&error),
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
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

    /// A unary call answers Python's `AnthropicMessagesResponse` shape with
    /// the usage split, and a malformed body gets the Anthropic envelope.
    #[tokio::test]
    async fn messages_answers_anthropic_shapes_and_errors() {
        let state = answering_app_state();
        let (status, body) = post(
            state.clone(),
            "/v1/messages",
            r#"{"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert!(json["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(
            json["content"],
            serde_json::json!([{"type": "text", "text": "ok"}])
        );
        assert_eq!(json["stop_reason"], "end_turn");
        assert_eq!(
            json["usage"],
            serde_json::json!({"input_tokens": 5, "output_tokens": 1})
        );
        assert!(json.get("stop_sequence").is_none());

        let (status, body) = post(state.clone(), "/v1/messages", r#"{"model": "m"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "max_tokens must be positive");

        let (status, body) = post(state, "/v1/messages", r#"{"model": 1}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    /// The streamed form is SSE with the event name on each frame, in the
    /// Anthropic order.
    #[tokio::test]
    async fn messages_streams_named_sse_events() {
        let state = answering_app_state();
        let (status, body) = post(
            state,
            "/v1/messages",
            r#"{"model": "m", "max_tokens": 8, "stream": true, "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let names: Vec<&str> = body
            .lines()
            .filter_map(|line| line.strip_prefix("event: "))
            .collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ],
            "{body}"
        );
        let data: Vec<serde_json::Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            data[2]["delta"],
            serde_json::json!({"type": "text_delta", "text": "ok"})
        );
        assert_eq!(data[4]["delta"]["stop_reason"], "end_turn");
        assert_eq!(data[4]["usage"]["output_tokens"], 1);
    }
}
