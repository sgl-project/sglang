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
use crate::api_server::http::auth::DeclaredRoute;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .declared_route("/v1/messages", post(messages))
        .declared_route("/v1/messages/count_tokens", post(count_tokens))
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

    /// `/v1/messages` decodes on its own, so Gemma 4's markers must survive
    /// for its reasoning parser to split the thinking block from the text.
    #[tokio::test]
    async fn gemma4_messages_keep_reasoning_markers() {
        use crate::api_server::http::openai::test_utils::{GEMMA4_STRIPPED, gemma4_marker_state};

        for (model_type, expected) in [
            (
                "gemma4",
                serde_json::json!([
                    {"type": "thinking", "thinking": "Check the calculation."},
                    {"type": "text", "text": "The answer is 4."},
                ]),
            ),
            (
                "llama",
                serde_json::json!([{"type": "text", "text": GEMMA4_STRIPPED}]),
            ),
        ] {
            let state = gemma4_marker_state(model_type, Some("gemma4"), |_| {});
            let (status, body) = post(
                state,
                "/v1/messages",
                r#"{"model": "model", "max_tokens": 8, "messages": [{"role": "user", "content": "What is 2 + 2?"}]}"#,
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            let content: Vec<serde_json::Value> = json["content"]
                .as_array()
                .unwrap()
                .iter()
                .map(|block| {
                    let mut block = block.clone();
                    block.as_object_mut().unwrap().remove("signature");
                    block
                })
                .collect();
            assert_eq!(
                serde_json::Value::from(content),
                expected,
                "{model_type}: {body}"
            );
        }
    }

    /// After a `tool_result`, a Gemma 4 template can leave the thought channel
    /// open in the prompt; the streamed thinking block must still split off.
    #[tokio::test]
    async fn gemma4_stream_separates_thinking_after_tool_result() {
        use crate::api_server::http::openai::test_utils::{
            answering_app_state_with, checkpoint_formatter,
        };
        use crate::message::config::{ModelConfig, ServerArgs};

        let args = Arc::new(ServerArgs {
            served_model_name: "m".into(),
            reasoning_parser: Some("gemma4".into()),
            model_config: ModelConfig {
                model_type: Some("gemma4".into()),
                ..Default::default()
            },
            ..Default::default()
        });
        let formatter = checkpoint_formatter(
            &serde_json::json!({"model_type": "gemma4"}),
            "{% if messages[-1]['role'] == 'tool' and enable_thinking %}<|channel>thought\n\
             {% else %}<|turn>model\n{% endif %}",
        );
        let state =
            answering_app_state_with(args, formatter, None, "Check the result.<channel|>Done.");
        let (status, body) = post(
            state,
            "/v1/messages",
            &serde_json::json!({
                "model": "m", "max_tokens": 8, "stream": true,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [
                    {"role": "user", "content": "Check my booking."},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "call_1", "content": "Confirmed."}
                    ]},
                ],
            })
            .to_string(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let deltas: Vec<serde_json::Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|event| event["type"] == "content_block_delta")
            .collect();
        let text = |kind: &str, key: &str| -> String {
            deltas
                .iter()
                .filter(|event| event["delta"]["type"] == kind)
                .filter_map(|event| event["delta"][key].as_str())
                .collect()
        };
        assert_eq!(
            text("thinking_delta", "thinking"),
            "Check the result.",
            "{body}"
        );
        assert_eq!(text("text_delta", "text"), "Done.", "{body}");
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
