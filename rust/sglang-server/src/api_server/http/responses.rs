//! OpenAI Responses routes over the shared `api_server::responses` adapter.
//! `POST /v1/responses` plans the chat request and sends it back through
//! the router to `/v1/chat/completions`, so the served model's own chat
//! path (the token-level encoders included) answers it, exactly as
//! Python's Responses handler inherits the chat pipeline; the chat reply or
//! stream is then re-shaped into Responses output. The retrieve and cancel
//! routes need the response store, which this server does not have; they
//! answer Python's refusal.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    extract::{Path, Request, State},
    http::{StatusCode, Uri, header},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures::{Stream, StreamExt};
use serde_json::Value;
use sglang_api_types::api::v1 as api;
use sglang_processor::dynamo_protocols::types::CreateChatCompletionResponse;
use tower::ServiceExt;

use super::app::AppState;
use crate::api_server::responses::{Responses, ResponsesFailure};

/// The store-backed routes, mounted with the state router.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/responses/{response_id}", get(retrieve))
        .route("/v1/responses/{response_id}/cancel", post(cancel))
}

/// Mount `POST /v1/responses` over `app`, which it dispatches the planned
/// chat request into.
pub(super) fn mount(app: Router, state: Arc<AppState>) -> Router {
    let inner = app.clone();
    app.route(
        "/v1/responses",
        post(move |request: Request| create(inner.clone(), state.clone(), request)),
    )
}

fn failure(failure: &ResponsesFailure) -> Response {
    (failure.status, Json(failure.envelope())).into_response()
}

/// The chat route's refusal, `{"error": {"message", "code", ...}}`, as a
/// Responses failure: Python answers every failure of the chat phase as
/// a 400 with the message.
fn chat_failure(status: StatusCode, body: &[u8]) -> ResponsesFailure {
    let message = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| value["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| String::from_utf8_lossy(body).into_owned());
    ResponsesFailure {
        status: if status == StatusCode::NOT_FOUND || status.is_server_error() {
            status
        } else {
            StatusCode::BAD_REQUEST
        },
        ..ResponsesFailure::invalid(message)
    }
}

/// The `data:` payloads of an SSE body, one per frame.
fn sse_frames(body: Body) -> impl Stream<Item = String> + Send + 'static {
    async_stream::stream! {
        let mut data = body.into_data_stream();
        let mut buffer = String::new();
        while let Some(chunk) = data.next().await {
            let Ok(chunk) = chunk else { break };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(end) = buffer.find("\n\n") {
                let frame = buffer[..end].to_owned();
                buffer.drain(..end + 2);
                let payload: Vec<&str> = frame
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:"))
                    .map(str::trim_start)
                    .collect();
                if !payload.is_empty() {
                    yield payload.join("\n");
                }
            }
        }
    }
}

async fn create(inner: Router, state: Arc<AppState>, request: Request) -> Response {
    let (mut parts, body) = request.into_parts();
    let bytes = match to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(error) => return failure(&ResponsesFailure::bad_request(error.to_string())),
    };
    let request: api::OpenAiResponsesRequest = match serde_json::from_slice(&bytes) {
        Ok(request) => request,
        Err(error) => return failure(&ResponsesFailure::bad_request(error.to_string())),
    };
    let stream = request.stream == Some(true);
    let adapter = Responses {
        core: &state.core,
        chat_formatter: state.chat_formatter.as_ref(),
        server_args: &state.server_args,
    };
    let plan = match adapter.plan(&request, stream) {
        Ok(plan) => plan,
        Err(error) => return failure(&error),
    };
    parts.uri = "/v1/chat/completions".parse::<Uri>().expect("static path");
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::CONTENT_ENCODING);
    parts.headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    let chat_request = Request::from_parts(parts, Body::from(plan.chat_body.to_string()));
    let response = match inner.oneshot(chat_request).await {
        Ok(response) => response,
        Err(never) => match never {},
    };
    let status = response.status();
    let is_sse = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    if status != StatusCode::OK {
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_else(|_| Bytes::new());
        return failure(&chat_failure(status, &body));
    }
    if !is_sse {
        let body = match to_bytes(response.into_body(), usize::MAX).await {
            Ok(body) => body,
            Err(error) => return failure(&ResponsesFailure::invalid(error.to_string())),
        };
        return match serde_json::from_slice::<CreateChatCompletionResponse>(&body) {
            Ok(chat) => Json(plan.shape_unary(chat)).into_response(),
            Err(error) => failure(&ResponsesFailure::invalid(format!(
                "unreadable chat completion: {error}"
            ))),
        };
    }
    let events = plan.events(sse_frames(response.into_body()));
    let mut response = Sse::new(events.map(|(kind, payload)| {
        Ok::<_, Infallible>(Event::default().event(kind).data(payload.to_string()))
    }))
    .into_response();
    response
        .headers_mut()
        .insert("cache-control", "no-cache".parse().expect("static header"));
    response
}

/// `GET /v1/responses/{id}` — Python's store-disabled refusal.
async fn retrieve(
    State(_state): State<Arc<AppState>>,
    Path(_response_id): Path<String>,
) -> Response {
    failure(&ResponsesFailure::store_disabled("response_id"))
}

/// `POST /v1/responses/{id}/cancel` — Python's store-disabled refusal.
async fn cancel(State(_state): State<Arc<AppState>>, Path(_response_id): Path<String>) -> Response {
    failure(&ResponsesFailure::store_disabled("response_id"))
}

#[cfg(test)]
mod tests {
    use axum::http::{Request as HttpRequest, header::CONTENT_TYPE};

    use super::*;
    use crate::api_server::http::openai::test_utils::answering_app_state;

    /// The Responses routes over the chat route, as the app mounts them.
    fn app(state: Arc<AppState>) -> Router {
        let inner = Router::new()
            .merge(super::super::openai::routes())
            .merge(routes())
            .with_state(state.clone());
        mount(inner, state)
    }

    async fn post_json(app: &Router, path: &str, body: &str) -> (StatusCode, Vec<u8>) {
        let response = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(path)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, bytes.to_vec())
    }

    /// A unary request answers Python's `ResponsesResponse`: one message
    /// item holding the generation, usage in the Responses shape, the
    /// echoed request fields, and a text-only `text` format.
    #[tokio::test]
    async fn create_answers_the_responses_shape() {
        let state = answering_app_state();
        let model = state.server_args.served_model_name.clone();
        let app = app(state);
        let body = serde_json::json!({
            "model": model, "input": "hi", "instructions": "be brief", "max_output_tokens": 10,
            "metadata": {"k": "v"}, "reasoning": {"effort": "low"}
        })
        .to_string();
        let (status, body) = post_json(&app, "/v1/responses", &body).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "response");
        assert!(json["id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(json["model"], model);
        assert_eq!(json["status"], "completed");
        assert_eq!(json["output"].as_array().unwrap().len(), 1);
        let item = &json["output"][0];
        assert_eq!(item["type"], "message");
        assert!(item["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(item["phase"], "final_answer");
        assert_eq!(item["content"][0]["type"], "output_text");
        assert_eq!(item["content"][0]["text"], "ok");
        assert_eq!(item["content"][0]["annotations"], serde_json::json!([]));
        assert_eq!(
            json["usage"]["input_tokens_details"]["cache_write_tokens"],
            0
        );
        assert_eq!(
            json["usage"]["output_tokens_details"]["reasoning_tokens"],
            0
        );
        assert_eq!(json["instructions"], "be brief");
        assert_eq!(json["max_output_tokens"], 10);
        assert_eq!(json["metadata"], serde_json::json!({"k": "v"}));
        assert_eq!(
            json["reasoning"],
            serde_json::json!({"effort": "low", "summary": null})
        );
        assert_eq!(
            json["text"],
            serde_json::json!({"format": {"type": "text"}})
        );
        assert_eq!(json["incomplete_details"], serde_json::Value::Null);
        assert_eq!(json["store"], true);
        assert_eq!(json["truncation"], "disabled");
        assert_eq!(json["tool_choice"], "auto");

        // No `reasoning` sent: Python echoes a null effort.
        let (_, body) = post_json(&app, "/v1/responses", r#"{"input": "hi"}"#).await;
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["reasoning"],
            serde_json::json!({"effort": null, "summary": null})
        );
    }

    /// The refusals Python gives without the response store, a harmony
    /// model or a tool server, and its validation 400s, with its envelope.
    #[tokio::test]
    async fn refusals_follow_python() {
        let state = answering_app_state();
        let app = app(state);
        for (body, status, param, needle) in [
            (
                r#"{"input": "hi", "previous_response_id": "resp_x"}"#,
                400,
                Some("previous_response_id"),
                "Response store is disabled",
            ),
            (
                r#"{"input": "hi", "background": true}"#,
                400,
                Some("background"),
                "Response store is disabled",
            ),
            (
                r#"{"input": "hi", "model": "other"}"#,
                404,
                Some("model"),
                "does not exist",
            ),
            (
                r#"{"input": "hi", "tools": [{"type": "shell"}]}"#,
                400,
                Some("tools"),
                "Shell tools",
            ),
            (
                r#"{"input": "hi", "tool_choice": "required"}"#,
                400,
                None,
                "tool_choice=\"required\"",
            ),
            (
                r#"{"input": "hi", "tools": [{"type": "function", "name": "f"}], "tool_choice": {"type": "function", "name": "g"}}"#,
                400,
                Some("tool_choice"),
                "not declared",
            ),
            (
                r#"{"input": "hi", "stream": true, "include": ["message.output_text.logprobs"]}"#,
                400,
                Some("logprobs"),
                "streaming",
            ),
            (
                r#"{"input": [{"type": "computer_call"}]}"#,
                400,
                None,
                "Unsupported Responses API input item type",
            ),
            (
                r#"{"input": "hi", "include": ["bogus"]}"#,
                400,
                None,
                "include",
            ),
        ] {
            let (got, bytes) = post_json(&app, "/v1/responses", body).await;
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(got.as_u16(), status, "{body}: {json}");
            assert_eq!(json["error"]["code"], status, "{body}");
            assert_eq!(
                json["error"]["param"],
                param
                    .map(serde_json::Value::from)
                    .unwrap_or(serde_json::Value::Null),
                "{body}"
            );
            assert!(
                json["error"]["message"].as_str().unwrap().contains(needle),
                "{body}: {json}"
            );
        }
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("GET")
                    .uri("/v1/responses/resp_1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1 << 16).await.unwrap())
                .unwrap();
        assert_eq!(json["error"]["param"], "response_id");
    }

    /// A stream is Python's event sequence: created and in_progress, the
    /// message item's added/part/delta/done events, and a terminal
    /// `response.completed` carrying the full response; each frame names
    /// its event and numbers it.
    #[tokio::test]
    async fn stream_emits_the_responses_event_sequence() {
        let state = answering_app_state();
        let app = app(state);
        let (status, body) =
            post_json(&app, "/v1/responses", r#"{"input": "hi", "stream": true}"#).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let text = String::from_utf8(body).unwrap();
        let frames: Vec<(String, serde_json::Value)> = text
            .split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .map(|frame| {
                let mut lines = frame.lines();
                let event = lines
                    .next()
                    .unwrap()
                    .strip_prefix("event: ")
                    .unwrap()
                    .to_owned();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (event, serde_json::from_str(data).unwrap())
            })
            .collect();
        let kinds: Vec<&str> = frames.iter().map(|(kind, _)| kind.as_str()).collect();
        assert_eq!(
            kinds,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
            "{text}"
        );
        for (i, (kind, payload)) in frames.iter().enumerate() {
            assert_eq!(payload["type"], *kind);
            assert_eq!(payload["sequence_number"], i, "{kind}");
        }
        assert_eq!(frames[0].1["response"]["status"], "in_progress");
        assert_eq!(frames[4].1["delta"], "ok");
        assert_eq!(frames[4].1["output_index"], 0);
        let done = &frames[8].1["response"];
        assert_eq!(done["status"], "completed");
        assert_eq!(done["output"][0]["content"][0]["text"], "ok");
        assert!(done["usage"]["total_tokens"].is_number());
        assert!(!text.contains("[DONE]"));
    }
}
