//! Shared HTTP test harness and `openai.rs`-level handler tests.
//!
//! Submodule tests live next to the code they cover: `chat`, `completions`,
//! `responses`, `tools`, and `reasoning` each carry their own
//! `#[cfg(test)] mod tests`. This module keeps the fixtures they all share —
//! channel fixtures (`senders`, `chunk`, `submitted`, `chat_submitted`,
//! `response_request`) and the full-router harness (`server_args`, `app_state`,
//! `oneshot`, `post_json`, `body_json`) — plus the handler-level tests that
//! exercise [`routes`] end to end. The helpers are `pub(super)` so sibling
//! test modules can import them via `super::super::test_utils::*`.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use dynamo_protocols::types::responses::{CreateResponse, Status};
use serde_json::json;
use tower::util::ServiceExt;

use super::response_stream::response_object;
use super::responses::{StoredResponse, new_response_store, routes_with_store};
use super::{routes, unix_seconds};
use crate::ids::Rid;
use crate::message::{ChunkEvent, EgressItem};
use crate::runtime::ServerArgs;
use crate::tokenizer_manager::{AbortSource, Senders};

pub(super) fn senders() -> Senders {
    Senders {
        tm: flume::unbounded().0,
        abort: flume::unbounded().0,
        tok: flume::unbounded().0,
        detok: vec![],
    }
}

pub(super) fn chunk(rid: &str, text: &str, done: bool) -> EgressItem {
    let output = ChunkEvent {
        rid: rid.into(),
        text: text.into(),
        token_ids: vec![1],
        prompt_tokens: 5,
        completion_tokens: 1,
        finish_reason: done.then(|| {
            serde_json::from_value(serde_json::json!({
                "type": "stop",
                "matched": "</s>"
            }))
            .unwrap()
        }),
        ..Default::default()
    };
    if done {
        EgressItem::Done(output)
    } else {
        EgressItem::Frame(output)
    }
}

/// A submitted legacy completion choice with its egress channel.
pub(super) fn submitted(
    index: usize,
    prompt_index: usize,
    rid: &str,
) -> (
    super::completions::SubmittedChoice,
    tokio::sync::mpsc::Sender<EgressItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    (
        super::completions::SubmittedChoice {
            index,
            prompt_index,
            rid: rid.into(),
            echo: String::new(),
            rx,
        },
        tx,
    )
}

/// A submitted chat choice (the tuple `chat_event_stream` consumes) with its
/// egress channel.
pub(super) fn chat_submitted(
    index: usize,
    rid: &str,
) -> (
    (usize, Rid, tokio::sync::mpsc::Receiver<EgressItem>),
    tokio::sync::mpsc::Sender<EgressItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    ((index, rid.into(), rx), tx)
}

/// A minimal Responses request, shared by the responses tests and the
/// handler-level lifecycle test.
pub(super) fn response_request(stream: bool) -> CreateResponse {
    serde_json::from_value(serde_json::json!({
        "model": "model",
        "input": "The capital of France is",
        "stream": stream,
        "temperature": 0.0,
        "max_output_tokens": 8
    }))
    .unwrap()
}

// ---------------------------------------------------------------------
// Handler-level tests: full router, real extractors, no scheduler. A
// request that reaches `submit` with an OPEN tm lane would wait on the
// egress receiver forever, so submission-reaching cases use `senders_closed`
// (503) and everything else fails validation before submit.
// ---------------------------------------------------------------------

pub(super) fn server_args() -> Arc<ServerArgs> {
    Arc::new(
        serde_json::from_value(serde_json::json!({ "served_model_name": "model" }))
            .expect("ServerArgs must deserialize"),
    )
}

pub(super) fn app_state(senders: Senders) -> super::AppState {
    super::AppState {
        senders,
        egress_buf: 8,
        server_args: server_args(),
        chat_formatter: None,
        egress_activity: Default::default(),
    }
}

pub(super) fn senders_closed() -> Senders {
    // Dropping the receivers disconnects the channels; the senders stay
    // valid (moveable) but every send reports `Err`, the shutdown state
    // `submit` surfaces as a 503.
    let (tm_tx, tm_rx) = flume::unbounded();
    drop(tm_rx);
    let (abort_tx, abort_rx) = flume::unbounded();
    drop(abort_rx);
    let (tok_tx, tok_rx) = flume::unbounded();
    drop(tok_rx);
    Senders {
        tm: tm_tx,
        abort: abort_tx,
        tok: tok_tx,
        detok: vec![],
    }
}

pub(super) fn senders_with_abort_rx() -> (Senders, flume::Receiver<AbortSource>) {
    let (tm_tx, _tm_rx) = flume::unbounded();
    let (abort_tx, abort_rx) = flume::unbounded();
    let (tok_tx, _tok_rx) = flume::unbounded();
    (
        Senders {
            tm: tm_tx,
            abort: abort_tx,
            tok: tok_tx,
            detok: vec![],
        },
        abort_rx,
    )
}

pub(super) fn request(method: &str, path: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .unwrap()
}

/// Serve one request through the full router (extractors, auth, routing).
/// `with_state` consumes the state into a `Router<()>`, which is what
/// implements `tower::Service`.
pub(super) async fn oneshot(app: Router<()>, req: Request<Body>) -> Response {
    app.oneshot(req).await.unwrap()
}

pub(super) async fn post_json(app: Router<()>, path: &str, body: serde_json::Value) -> Response {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    oneshot(app, req).await
}

pub(super) async fn body_json(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// The common StatusCode→error helper follows `pre_submit_error`'s shape:
/// unary requests get the JSON error with its status; a committed stream gets
/// 200 + one SSE error frame + `[DONE]`, and the frame carries the OpenAI
/// error fields (`type`, `param`, `code`) that the SDKs dispatch on.
#[tokio::test]
async fn openai_error_response_covers_unary_and_sse() {
    let unary = super::openai_error_response(StatusCode::BAD_REQUEST, "bad input", false);
    assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
    let value = body_json(unary).await;
    assert_eq!(value["error"]["message"], "bad input");
    assert_eq!(value["error"]["type"], "BadRequestError");
    assert_eq!(value["error"]["code"], 400);
    assert!(value["error"]["param"].is_null());

    let streamed = super::openai_error_response(StatusCode::BAD_REQUEST, "bad input", true);
    assert_eq!(streamed.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame = text
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
    assert_eq!(frame["error"]["message"], "bad input");
    assert_eq!(frame["error"]["type"], "BadRequestError");
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn completions_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (json!({"model": "other", "prompt": "hi"}), "unknown model"),
        (json!({"model": "model", "prompt": "hi", "n": 0}), "n=0"),
        (
            json!({"model": "model", "prompt": "hi", "max_tokens": 0}),
            "max_tokens=0",
        ),
        (json!({"model": "model", "prompt": ""}), "empty prompt"),
        (
            json!({"model": "model", "prompt": "hi", "best_of": 2}),
            "best_of>1",
        ),
        (
            json!({"model": "model", "prompt": "hi", "suffix": "x"}),
            "suffix",
        ),
        (
            json!({"model": "model", "prompt": "hi", "prompt_embeds": [[1.0]]}),
            "prompt_embeds",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // Malformed JSON → 400 (JsonRejection path).
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("not json"))
        .unwrap();
    let response = oneshot(app.clone(), req).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // A closed tm inbox (shutdown) surfaces as 503.
    let app = routes().with_state(app_state(senders_closed()));
    let response = post_json(
        app.clone(),
        "/v1/completions",
        json!({"model": "model", "prompt": "hi"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn chat_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (
            json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}),
            "unknown model",
        ),
        (json!({"model": "model", "messages": []}), "empty messages"),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 0}),
            "n=0",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}]}),
            "media content",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "function_call": "auto"}),
            "deprecated function_call",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "audio": {"input_audio": {"data": "x", "format": "wav"}}}),
            "audio",
        ),
        (
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}),
            "max_completion_tokens=0",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/chat/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // A valid request with no loaded chat template → 400 (template gate).
    let response = post_json(
        app.clone(),
        "/v1/chat/completions",
        json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn responses_handler_validates_before_submit() {
    let app = routes().with_state(app_state(senders()));
    let cases = [
        (json!({"model": "other", "input": "hi"}), "unknown model"),
        (
            json!({"input": "hi", "max_output_tokens": 0}),
            "max_output_tokens=0",
        ),
        (json!({"input": "hi", "conversation": {}}), "conversation"),
        (json!({"input": "hi", "prompt": "x"}), "prompt template"),
        (json!({"input": "hi", "include": ["reasoning"]}), "include"),
        (
            json!({"input": "hi", "max_tool_calls": 3}),
            "max_tool_calls",
        ),
        (
            json!({"input": "hi", "truncation": "auto"}),
            "truncation auto",
        ),
        (
            json!({"input": "hi", "reasoning": {"summary": "x"}}),
            "reasoning summary",
        ),
        (
            json!({"input": "hi", "previous_response_id": "nope"}),
            "bad previous_response_id",
        ),
        (
            json!({"input": [{"type": "item_reference", "item_id": "x"}]}),
            "item reference",
        ),
        (json!({"input": []}), "empty input"),
        (
            json!({"input": "hi", "background": true, "stream": true}),
            "background+stream",
        ),
    ];
    for (body, label) in cases {
        let response = post_json(app.clone(), "/v1/responses", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // Unknown previous_response_id → 404 (store lookup).
    let response = post_json(
        app.clone(),
        "/v1/responses",
        json!({"input": "hi", "previous_response_id": "resp_missing"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    // A valid request without a chat template → 400 (template gate).
    let response = post_json(app.clone(), "/v1/responses", json!({"input": "hi"})).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// A closed tm inbox with a *streaming* request must answer inside the
/// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
/// same rule `pre_submit_error` applies to the native API), not a unary 503.
#[tokio::test]
async fn streaming_submit_failure_answers_inside_the_stream() {
    let app = routes().with_state(app_state(senders_closed()));
    let response = post_json(
        app,
        "/v1/completions",
        json!({"model": "model", "prompt": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame = text
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
    assert_eq!(frame["error"]["message"], "service unavailable");
    assert_eq!(frame["error"]["type"], "InternalServerError");
    assert_eq!(frame["error"]["code"], 503);
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn response_retrieve_and_cancel_lifecycle() {
    let (senders, abort_rx) = senders_with_abort_rx();
    let state = app_state(senders);
    let store = new_response_store();
    let app = routes_with_store(store.clone()).with_state(state);

    // Unknown / malformed ids.
    let response = oneshot(app.clone(), request("GET", "/v1/responses/resp_missing")).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let response = oneshot(app.clone(), request("GET", "/v1/responses/nope")).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Seed an in-progress response directly: a background request would
    // need a real scheduler behind the tm lane. `Rid::from_client` mints a
    // fresh uniquified rid per call, so capture the one the store holds.
    let seeded_rid = Rid::from_client("resp_seeded");
    store.write().await.insert(
        "resp_seeded".into(),
        StoredResponse {
            response: response_object(
                "resp_seeded",
                "model",
                &response_request(false),
                unix_seconds(),
                Status::InProgress,
                vec![],
                None,
            ),
            messages: vec![],
            rid: Some(seeded_rid.clone()),
        },
    );

    // Retrieve returns the stored object.
    let response = oneshot(app.clone(), request("GET", "/v1/responses/resp_seeded")).await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = body_json(response).await;
    assert_eq!(value["object"], "response");
    assert_eq!(value["status"], "in_progress");

    // Cancel → Cancelled, and the abort reaches the scheduler lane.
    let response = oneshot(
        app.clone(),
        request("POST", "/v1/responses/resp_seeded/cancel"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = body_json(response).await;
    assert_eq!(value["status"], "cancelled");
    assert!(matches!(abort_rx.try_recv(), Ok(AbortSource::Guard(rid)) if rid == seeded_rid));

    // The store reflects the cancellation; cancelling again is a 400.
    let response = oneshot(app.clone(), request("GET", "/v1/responses/resp_seeded")).await;
    let value = body_json(response).await;
    assert_eq!(value["status"], "cancelled");
    let response = oneshot(
        app.clone(),
        request("POST", "/v1/responses/resp_seeded/cancel"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
