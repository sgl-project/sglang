//! Shared HTTP test harness and `openai.rs`-level handler tests.
//!
//! Submodule tests live next to the code they cover: `chat`, `completions`,
//! `tools`, and `reasoning` each carry their own
//! `#[cfg(test)] mod tests`. This module keeps the fixtures they all share —
//! channel fixtures (`senders`, `chunk`, `submitted`, `chat_submitted`) and the
//! full-service harness (`server_args`, `app_state`,
//! `oneshot`, `post_json`, `body_json`) — plus the handler-level tests that
//! exercise the [`HttpApi`] route table end to end. The helpers are
//! `pub(super)` so sibling test modules can import them via
//! `super::super::test_utils::*`.

use std::sync::Arc;

use http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use tower::util::ServiceExt;

pub(super) use crate::api_server::core::openai::test_utils::senders;

use super::super::app::HttpApi;
use super::super::plumbing::{HttpBody, HttpResponse, empty, full};
use super::openai_error;
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::wiring::Senders;

pub(super) fn server_args() -> Arc<ServerArgs> {
    Arc::new(ServerArgs {
        served_model_name: "model".into(),
        ..Default::default()
    })
}

pub(super) fn app_state(senders: Senders) -> Arc<super::AppState> {
    Arc::new(super::AppState {
        senders,
        response_buf: 8,
        api_key: None,
        server_args: server_args(),
        chat_formatter: None,
        response_activity: Default::default(),
    })
}

/// The full route table over the given state — no auth / access-log layers
/// (those have their own tests in `api_server::layers`).
pub(super) fn app(state: Arc<super::AppState>) -> HttpApi {
    HttpApi::new(state, None)
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
        tok_manager_tx: tm_tx,
        abort_tx,
        tokenizer_tx: tok_tx,
        detokenizer_tx: vec![],
    }
}

/// Serve one request through the full service (extraction, routing).
pub(super) async fn oneshot(app: HttpApi, req: Request<HttpBody>) -> HttpResponse {
    app.oneshot(req).await.unwrap()
}

pub(super) async fn post_json(app: HttpApi, path: &str, body: serde_json::Value) -> HttpResponse {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(full(body.to_string()))
        .unwrap();
    oneshot(app, req).await
}

pub(super) async fn body_json(response: HttpResponse) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

/// The common StatusCode→error helper follows `error_response`'s shape:
/// unary requests get the JSON error with its status; a committed stream gets
/// 200 + one SSE error frame + `[DONE]`, and the frame carries the OpenAI
/// error fields (`type`, `param`, `code`) that the SDKs dispatch on.
#[tokio::test]
async fn openai_error_response_covers_unary_and_sse() {
    let unary = openai_error(StatusCode::BAD_REQUEST, "bad input", false);
    assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
    let value = body_json(unary).await;
    assert_eq!(value["error"]["message"], "bad input");
    assert_eq!(value["error"]["type"], "BadRequestError");
    assert_eq!(value["error"]["code"], 400);
    assert!(value["error"]["param"].is_null());

    let streamed = openai_error(StatusCode::BAD_REQUEST, "bad input", true);
    assert_eq!(streamed.status(), StatusCode::OK);
    let bytes = streamed.into_body().collect().await.unwrap().to_bytes();
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
    let app_ = app(app_state(senders()));
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
        let response = post_json(app_.clone(), "/v1/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // Malformed JSON → 400 (JsonRejection path).
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(full("not json"))
        .unwrap();
    let response = oneshot(app_.clone(), req).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // A closed tm inbox (shutdown) surfaces as 503.
    let app_ = app(app_state(senders_closed()));
    let response = post_json(
        app_.clone(),
        "/v1/completions",
        json!({"model": "model", "prompt": "hi"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn chat_handler_validates_before_submit() {
    let app_ = app(app_state(senders()));
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
        let response = post_json(app_.clone(), "/v1/chat/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
    }
    // A valid request with no loaded chat template → 400 (template gate).
    let response = post_json(
        app_.clone(),
        "/v1/chat/completions",
        json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn basic_openai_router_excludes_responses_api() {
    let app_ = app(app_state(senders()));
    let response = post_json(app_, "/v1/responses", json!({"input": "hi"})).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// A closed tm inbox with a *streaming* request must answer inside the
/// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
/// same `error_response` rule the native API applies), not a unary 503.
#[tokio::test]
async fn streaming_submit_failure_answers_inside_the_stream() {
    let app_ = app(app_state(senders_closed()));
    let response = post_json(
        app_,
        "/v1/completions",
        json!({"model": "model", "prompt": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
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

/// The HTTP-edge wire contract the de-axum swap must reproduce byte-for-byte:
/// the extractor rejection texts (clients see them in 400 bodies), the SSE
/// response headers, and the no-route / wrong-method statuses.
#[tokio::test]
async fn http_edge_wire_contract() {
    use http::header::CONTENT_TYPE;

    let mk_app = || app(app_state(senders()));

    // Malformed JSON: 400 with axum's syntax text incl. serde's position.
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(full("{\"model\": }"))
        .unwrap();
    let res = oneshot(mk_app(), req).await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let v = body_json(res).await;
    assert_eq!(
        v["error"]["message"],
        "Failed to parse the request body as JSON: model: expected value at line 1 column 11"
    );

    // Type mismatch: 400 with axum's data-error text.
    let res = post_json(
        mk_app(),
        "/v1/completions",
        serde_json::json!({"model": 3, "prompt": "hi"}),
    )
    .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let v = body_json(res).await;
    assert_eq!(
        v["error"]["message"],
        "Failed to deserialize the JSON body into the target type: model: invalid type: integer `3`, expected a string at line 1 column 10"
    );

    // Missing JSON content type: 400 with axum's content-type text.
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .body(full("{}"))
        .unwrap();
    let res = oneshot(mk_app(), req).await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let v = body_json(res).await;
    assert_eq!(
        v["error"]["message"],
        "Expected request with `Content-Type: application/json`"
    );

    // Unknown path: bare 404. Known path, wrong method: bare 405 + Allow.
    let res = oneshot(
        mk_app(),
        Request::builder().uri("/nope").body(empty()).unwrap(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let res = oneshot(
        mk_app(),
        Request::builder()
            .method("GET")
            .uri("/v1/completions")
            .body(empty())
            .unwrap(),
    )
    .await;
    assert_eq!(res.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        res.headers()
            .get(http::header::ALLOW)
            .and_then(|v| v.to_str().ok()),
        Some("POST")
    );

    // A committed stream: 200 with the SSE headers axum set.
    let app_ = app(app_state(senders_closed()));
    let res = post_json(
        app_,
        "/v1/completions",
        serde_json::json!({"model": "model", "prompt": "hi", "stream": true}),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    assert_eq!(
        res.headers()
            .get(http::header::CACHE_CONTROL)
            .and_then(|v| v.to_str().ok()),
        Some("no-cache")
    );
}
