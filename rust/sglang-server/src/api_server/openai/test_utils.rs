//! Shared HTTP test harness and `openai.rs`-level handler tests.
//!
//! Submodule tests live next to the code they cover: `chat`, `completions`,
//! `tools`, and `reasoning` each carry their own
//! `#[cfg(test)] mod tests`. This module keeps the fixtures they all share —
//! frontend/call fixtures (`frontend`, `chunk`, `submitted`, `chat_submitted`) and the
//! full-router harness (`server_args`, `app_state`,
//! `oneshot`, `post_json`, `body_json`) — plus the handler-level tests that
//! exercise [`routes`] end to end. The helpers are `pub(super)` so sibling
//! test modules can import them via `super::super::test_utils::*`.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use futures::StreamExt;
use serde_json::json;
use tower::util::ServiceExt;

use super::{indexed_decode_stream, openai_error, routes};
use crate::frontend::{FrontendCall, FrontendEvent, FrontendHandle};
use crate::message::config::ServerArgs;
use crate::message::response::{ChunkEvent, ResponseItem};
pub(super) fn frontend() -> FrontendHandle {
    FrontendHandle::new(
        flume::unbounded().0,
        flume::unbounded().0,
        crate::frontend::FrontendConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            metadata: crate::frontend::FrontendMetadata::from(server_args().as_ref()),
        },
    )
}

pub(super) fn chunk(rid: &str, text: &str, done: bool) -> ResponseItem {
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
        ResponseItem::Done(output)
    } else {
        ResponseItem::Frame(output)
    }
}

/// A submitted legacy completion choice.
pub(super) fn submitted(
    index: usize,
    prompt_index: usize,
    rid: &str,
) -> (
    super::completions::SubmittedChoice,
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    (
        super::completions::SubmittedChoice {
            index,
            prompt_index,
            echo: String::new(),
            call: FrontendCall::from_test_generation_parts(rid.into(), rx, flume::unbounded().0),
        },
        tx,
    )
}

/// A submitted chat choice (the tuple `chat_event_stream` consumes).
pub(super) fn chat_submitted(
    index: usize,
    rid: &str,
) -> (
    (usize, FrontendCall),
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    (
        (
            index,
            FrontendCall::from_test_generation_parts(rid.into(), rx, flume::unbounded().0),
        ),
        tx,
    )
}

pub(super) fn server_args() -> Arc<ServerArgs> {
    Arc::new(ServerArgs {
        served_model_name: "model".into(),
        ..Default::default()
    })
}

pub(super) fn app_state(frontend: FrontendHandle) -> Arc<super::AppState> {
    Arc::new(super::AppState {
        frontend,
        server_args: server_args(),
        chat_formatter: None,
    })
}

pub(super) fn frontend_closed() -> FrontendHandle {
    // Dropping the receivers makes frontend admission report the shutdown
    // state as a 503.
    let (tm_tx, tm_rx) = flume::unbounded();
    drop(tm_rx);
    let (abort_tx, abort_rx) = flume::unbounded();
    drop(abort_rx);
    FrontendHandle::new(
        tm_tx,
        abort_tx,
        crate::frontend::FrontendConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            metadata: crate::frontend::FrontendMetadata::from(server_args().as_ref()),
        },
    )
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

#[tokio::test]
async fn dropping_indexed_stream_aborts_its_live_call() {
    use crate::tokenizer_manager::wiring::AbortSource;

    let (response_tx, response_rx) = tokio::sync::mpsc::channel(8);
    let (abort_tx, abort_rx) = flume::unbounded();
    let call = FrontendCall::from_test_generation_parts("live".into(), response_rx, abort_tx);
    let mut stream = indexed_decode_stream(0, call);

    response_tx.send(chunk("live", "x", false)).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some((0, FrontendEvent::Delta(_)))
    ));

    drop(stream);
    assert!(matches!(
        abort_rx.recv().unwrap(),
        AbortSource::Guard(rid) if rid.as_str() == "live"
    ));
    assert!(abort_rx.try_recv().is_err());
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
    let app = routes().with_state(app_state(frontend()));
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
    let app = routes().with_state(app_state(frontend_closed()));
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
    let app = routes().with_state(app_state(frontend()));
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
    for path in ["/v1/chat/completions", "/invocations"] {
        for (body, label) in &cases {
            let response = post_json(app.clone(), path, body.clone()).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{path}: {label}"
            );
        }
        // A valid request with no loaded chat template → 400 (template gate).
        let response = post_json(
            app.clone(),
            path,
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let response = oneshot(
            app.clone(),
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

async fn invoke_sagemaker(stream: bool) -> Response {
    use super::template::{LegacyFormatter, builtin_template};
    use crate::message::request::RequestKind;
    use crate::tokenizer_manager::wiring::TmEvent;

    let (tm_tx, tm_rx) = flume::unbounded();
    let mut state = app_state(senders());
    let state_mut = Arc::get_mut(&mut state).unwrap();
    state_mut.senders.tok_manager_tx = tm_tx;
    state_mut.chat_formatter = Some(super::ChatFormatter::Legacy(Box::new(LegacyFormatter {
        spec: builtin_template("chatml").unwrap(),
    })));
    let app = routes().with_state(state);

    let backend = tokio::spawn(async move {
        let TmEvent::Intake(request) = tm_rx.recv_async().await.unwrap() else {
            panic!("expected a generation request");
        };
        let RequestKind::Generate(generate) = &request.kind else {
            panic!("expected a chat request lowered to generation");
        };
        assert!(generate.text.as_ref().unwrap().contains("Say hello"));
        assert_eq!(generate.stream, stream);
        assert_eq!(generate.sampling_params.max_new_tokens, Some(8));
        request
            .sink
            .try_send(chunk(request.rid.client_facing(), "Hello!", true))
            .unwrap();
    });
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        post_json(
            app,
            "/invocations",
            json!({
                "model": "model",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 8,
                "stream": stream
            }),
        ),
    )
    .await
    .expect("invocation did not complete");
    assert_eq!(response.status(), StatusCode::OK);
    backend.await.unwrap();
    response
}

#[tokio::test]
async fn sagemaker_invocations_returns_chat_completion() {
    let response = invoke_sagemaker(false).await;
    let body = body_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "model");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn sagemaker_invocations_streams_chat_completion() {
    let response = invoke_sagemaker(true).await;
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.ends_with("data: [DONE]\n\n"));
    let chunks: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk["object"] == "chat.completion.chunk")
    );
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Hello!");
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[tokio::test]
async fn basic_openai_router_excludes_responses_api() {
    let app = routes().with_state(app_state(frontend()));
    let response = post_json(app, "/v1/responses", json!({"input": "hi"})).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// A closed tm inbox with a *streaming* request must answer inside the
/// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
/// same `error_response` rule the native API applies), not a unary 503.
#[tokio::test]
async fn streaming_submit_failure_answers_inside_the_stream() {
    let app = routes().with_state(app_state(frontend_closed()));
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
