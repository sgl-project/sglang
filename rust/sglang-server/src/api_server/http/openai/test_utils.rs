//! Shared HTTP test harness and `openai.rs`-level handler tests.
//!
//! Submodule tests live next to the code they cover: `chat`, `completions`,
//! `tools`, and `reasoning` each carry their own
//! `#[cfg(test)] mod tests`. This module keeps the fixtures they all share —
//! core/call fixtures (`core_handle`, `chunk`, `submitted`, `chat_submitted`) and the
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
use crate::api_server::core::{CoreCall, CoreEvent, CoreHandle};
use crate::message::config::ServerArgs;
use crate::message::request::RequestKind;
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::wiring::TmEvent;
pub(super) fn core_handle() -> CoreHandle {
    CoreHandle::new(
        flume::unbounded().0,
        flume::unbounded().0,
        crate::api_server::core::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            metadata: crate::api_server::core::CoreMetadata::from(server_args().as_ref()),
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
            call: CoreCall::from_test_generation_parts(rid.into(), rx, flume::unbounded().0),
        },
        tx,
    )
}

/// A submitted chat choice (the tuple `chat_event_stream` consumes).
pub(super) fn chat_submitted(
    index: usize,
    rid: &str,
) -> ((usize, CoreCall), tokio::sync::mpsc::Sender<ResponseItem>) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    (
        (
            index,
            CoreCall::from_test_generation_parts(rid.into(), rx, flume::unbounded().0),
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

pub(super) fn app_state(core: CoreHandle) -> Arc<super::AppState> {
    Arc::new(super::AppState {
        core,
        server_args: server_args(),
        chat_formatter: None,
    })
}

pub(super) fn frontend_closed() -> CoreHandle {
    // Dropping the receivers makes core admission report the shutdown
    // state as a 503.
    let (tm_tx, tm_rx) = flume::unbounded();
    drop(tm_rx);
    let (abort_tx, abort_rx) = flume::unbounded();
    drop(abort_rx);
    CoreHandle::new(
        tm_tx,
        abort_tx,
        crate::api_server::core::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            metadata: crate::api_server::core::CoreMetadata::from(server_args().as_ref()),
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

fn routing_app() -> (Router<()>, flume::Receiver<TmEvent>) {
    let (intake_tx, intake_rx) = flume::unbounded();
    let args = server_args();
    let core = CoreHandle::new(
        intake_tx,
        flume::unbounded().0,
        crate::api_server::core::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            metadata: crate::api_server::core::CoreMetadata::from(args.as_ref()),
        },
    );
    let state = Arc::new(super::AppState {
        core,
        server_args: args,
        chat_formatter: Some(super::ChatFormatter::Legacy(Box::new(
            super::template::LegacyFormatter {
                spec: super::template::builtin_template("chatml").unwrap(),
            },
        ))),
    });
    (routes().with_state(state), intake_rx)
}

// Router-supplied host, port, room, decode DP rank, and prefill DP rank.
type ExpectedRouting = (
    Option<&'static str>,
    Option<i64>,
    Option<i64>,
    Option<i64>,
    Option<i64>,
);

async fn assert_routing_requests(
    path: &str,
    body: serde_json::Value,
    expected: Vec<ExpectedRouting>,
) {
    let (app, intake_rx) = routing_app();
    let response = post_json(app, path, body);
    let responder = async {
        for (host, port, room, dp_rank, prefill_dp_rank) in expected {
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("OpenAI generation must enter through core intake");
            };
            assert!(admission.try_accept());
            let RequestKind::Generate(generate) = &request.kind else {
                panic!("OpenAI endpoint must submit a generation request");
            };
            assert_eq!(generate.bootstrap_host.as_deref(), host);
            assert_eq!(generate.bootstrap_port, port);
            assert_eq!(generate.bootstrap_room, room);
            assert_eq!(generate.routed_dp_rank, dp_rank);
            assert_eq!(generate.disagg_prefill_dp_rank, prefill_dp_rank);
            assert_eq!(generate.sampling_params.n, 1);
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
    // Drain the response so every call completes; response formatting is
    // covered by the chat/completions tests next to those adapters.
    axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    assert!(intake_rx.try_recv().is_err());
}

#[tokio::test]
async fn chat_preserves_pd_fields_through_admission() {
    // The standard Dynamo type used to silently discard these extensions.
    assert_routing_requests(
        "/v1/chat/completions",
        json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "n": 2,
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            "bootstrap_room": 9007199254740993_i64,
            "routed_dp_rank": 1,
            "disagg_prefill_dp_rank": 3,
        }),
        vec![
            (
                Some("prefill"),
                Some(8998),
                Some(9007199254740993),
                Some(1),
                Some(3)
            );
            2
        ],
    )
    .await;
}

#[tokio::test]
async fn completions_preserve_per_prompt_pd_pairing() {
    // Python TokenizerManager copies each prompt's routing across its samples:
    // scalar room 41 -> [41, 41, 42, 42], list [41, 52] -> [41, 41, 52, 52].
    for (stream, list, token_ids) in [(true, false, true), (false, true, false)] {
        let prompt = if token_ids {
            json!([[1], [2]])
        } else {
            json!(["one", "two"])
        };
        let mut body = json!({"model": "model", "prompt": prompt, "n": 2, "stream": stream, "routed_dp_rank": 1, "disagg_prefill_dp_rank": 3});
        body["bootstrap_host"] = if list {
            json!(["prefill-a", "prefill-b"])
        } else {
            json!("prefill-a")
        };
        body["bootstrap_port"] = if list {
            json!([8998, null])
        } else {
            json!(8998)
        };
        body["bootstrap_room"] = if list { json!([41, 52]) } else { json!(41) };
        let per_prompt = [
            (Some("prefill-a"), Some(8998), Some(41), Some(1), Some(3)),
            (
                Some(if list { "prefill-b" } else { "prefill-a" }),
                (!list).then_some(8998),
                Some(if list { 52 } else { 42 }),
                Some(1),
                Some(3),
            ),
        ];
        assert_routing_requests(
            "/v1/completions",
            body,
            per_prompt
                .into_iter()
                .flat_map(|routing| std::iter::repeat_n(routing, 2))
                .collect(),
        )
        .await;
    }
}

async fn assert_routing_rejected(path: &str, body: serde_json::Value) {
    let (app, intake_rx) = routing_app();
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        post_json(app, path, body),
    )
    .await
    .expect("invalid routing must be rejected before core admission");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(body_json(response).await["error"]["message"].is_string());
    assert!(intake_rx.try_recv().is_err());
}

#[tokio::test]
async fn invalid_pd_metadata_is_rejected_before_any_prompt_is_submitted() {
    // These null elements are accepted by native /generate, but not OpenAI.
    for field in ["bootstrap_host", "bootstrap_room"] {
        let mut body = json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]});
        body[field] = json!([null]);
        assert_routing_rejected("/v1/chat/completions", body).await;
    }
    assert_routing_rejected(
        "/v1/completions",
        json!({"model": "model", "prompt": ["one", "two"], "n": 2, "stream": true, "bootstrap_room": [41]}),
    )
    .await;
    let body = json!({"model": "model", "prompt": vec!["hi"; 2048], "n": 2, "bootstrap_host": "x".repeat(16385)});
    assert_routing_rejected("/v1/completions", body).await;
    // A one-element list must not bypass the scalar host's 64 MiB clone
    // budget when that prompt fans out to 255 choices.
    assert_routing_rejected(
        "/v1/chat/completions",
        json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 255, "bootstrap_host": ["x".repeat(263173)]}),
    )
    .await;
}

#[tokio::test]
async fn dropping_indexed_stream_aborts_its_live_call() {
    use crate::tokenizer_manager::wiring::AbortSource;

    let (response_tx, response_rx) = tokio::sync::mpsc::channel(8);
    let (abort_tx, abort_rx) = flume::unbounded();
    let call = CoreCall::from_test_generation_parts("live".into(), response_rx, abort_tx);
    let mut stream = indexed_decode_stream(0, call);

    response_tx.send(chunk("live", "x", false)).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some((0, CoreEvent::Delta(_)))
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
    let app = routes().with_state(app_state(core_handle()));
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
    let app = routes().with_state(app_state(core_handle()));
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
    let app = routes().with_state(app_state(core_handle()));
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
