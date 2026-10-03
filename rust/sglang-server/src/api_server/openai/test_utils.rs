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

async fn assert_bootstrap_requests(
    path: &str,
    body: serde_json::Value,
    expected: Vec<(Option<&str>, Option<i64>, Option<i64>)>,
) {
    use crate::message::request::RequestKind;
    use crate::tokenizer_manager::wiring::TmEvent;

    let (intake_tx, intake_rx) = flume::unbounded();
    let (abort_tx, _abort_rx) = flume::unbounded();
    let frontend = FrontendHandle::new(
        intake_tx,
        abort_tx,
        crate::frontend::FrontendConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: true,
            is_disaggregation: true,
            mm_limits: Default::default(),
            metadata: crate::frontend::FrontendMetadata::default(),
        },
    );
    let formatter = super::ChatFormatter::Legacy(Box::new(super::template::LegacyFormatter {
        spec: super::template::builtin_template("chatml").unwrap(),
    }));
    let state = Arc::new(super::AppState {
        frontend,
        server_args: server_args(),
        chat_formatter: Some(formatter),
    });
    let response = post_json(routes().with_state(state), path, body);
    let responder = async {
        for (host, port, room) in expected {
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("OpenAI generation must enter through frontend intake");
            };
            assert!(admission.try_accept());
            let RequestKind::Generate(generate) = &request.kind else {
                panic!("OpenAI endpoint must submit a generation request");
            };
            assert_eq!(generate.bootstrap_host.as_deref(), host);
            assert_eq!(generate.bootstrap_port, port);
            assert_eq!(generate.bootstrap_room, room);
            request
                .sink
                .try_send(chunk(request.rid.as_str(), "ok", true))
                .unwrap();
        }
    };
    let (response, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::join!(response, responder)
    })
    .await
    .expect("OpenAI request must complete after its scheduler response");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(intake_rx.try_recv().is_err());
}

#[tokio::test]
async fn chat_forwards_pd_bootstrap_fields_to_every_choice() {
    for stream in [false, true] {
        assert_bootstrap_requests(
            "/v1/chat/completions",
            json!({
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2,
                "stream": stream,
                "bootstrap_host": "prefill",
                "bootstrap_port": 8998,
                "bootstrap_room": 9007199254740993_i64
            }),
            vec![
                (Some("prefill"), Some(8998), Some(9007199254740993)),
                (Some("prefill"), Some(8998), Some(9007199254740994)),
            ],
        )
        .await;
    }
}

#[tokio::test]
async fn chat_preserves_list_bootstrap_room_for_all_choices() {
    assert_bootstrap_requests(
        "/v1/chat/completions",
        json!({
            "model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 2,
            "bootstrap_host": ["prefill"], "bootstrap_port": [null], "bootstrap_room": [41]
        }),
        vec![(Some("prefill"), None, Some(41)); 2],
    )
    .await;
}

#[tokio::test]
async fn completions_preserve_per_prompt_bootstrap_fields_for_all_choices() {
    assert_bootstrap_requests(
        "/v1/completions",
        json!({
            "model": "model", "prompt": ["one", "two"], "n": 2,
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null], "bootstrap_room": [41, 52]
        }),
        vec![
            (Some("prefill-a"), Some(8998), Some(41)),
            (Some("prefill-a"), Some(8998), Some(41)),
            (Some("prefill-b"), None, Some(52)),
            (Some("prefill-b"), None, Some(52)),
        ],
    )
    .await;
}

#[tokio::test]
async fn token_completions_expand_scalar_bootstrap_rooms_in_sample_major_order() {
    assert_bootstrap_requests(
        "/v1/completions",
        json!({
            "model": "model", "prompt": [[1], [2]], "n": 2,
            "bootstrap_host": "prefill", "bootstrap_room": 41
        }),
        vec![
            (Some("prefill"), None, Some(41)),
            (Some("prefill"), None, Some(43)),
            (Some("prefill"), None, Some(42)),
            (Some("prefill"), None, Some(44)),
        ],
    )
    .await;
}

#[tokio::test]
async fn mismatched_bootstrap_lists_are_rejected_before_submission() {
    let app = routes().with_state(app_state(frontend_closed()));
    for (field, value) in [
        ("bootstrap_host", json!(["prefill"])),
        ("bootstrap_port", json!([null])),
        ("bootstrap_room", json!([41])),
    ] {
        let mut body = json!({"model": "model", "prompt": ["one", "two"]});
        body[field] = value;
        let response = post_json(app.clone(), "/v1/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
        assert!(
            body_json(response).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains(field)
        );
    }
    let response = post_json(
        app,
        "/v1/chat/completions",
        json!({
            "model": "model", "messages": [{"role": "user", "content": "hi"}],
            "bootstrap_room": [41, 42]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        body_json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("bootstrap_room")
    );
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
