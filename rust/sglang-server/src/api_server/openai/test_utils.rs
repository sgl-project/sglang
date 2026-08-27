//! Shared HTTP test harness and `openai.rs`-level handler tests.
//!
//! Submodule tests live next to the code they cover: `chat`, `completions`,
//! `tools`, and `reasoning` each carry their own
//! `#[cfg(test)] mod tests`. This module keeps the fixtures they all share —
//! channel fixtures (`senders`, `chunk`, `submitted`, `chat_submitted`) and the
//! full-router harness (`server_args`, `app_state`,
//! `oneshot`, `post_json`, `body_json`) — plus the handler-level tests that
//! exercise [`routes`] end to end. The helpers are `pub(super)` so sibling
//! test modules can import them via `super::super::test_utils::*`.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use serde_json::json;
use tower::util::ServiceExt;

use super::{openai_error, routes};
use crate::frontend::FrontendHandle;
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::message::types::TokenIds;
use crate::renderer::{PreprocessJob, new_renderer_service};
use crate::runtime::Runnable;
use crate::tokenizer_manager::to_scheduler::Limits;
use crate::tokenizer_manager::tokenizer::{TextTokenizer, TokenizerWorker};
use crate::tokenizer_manager::wiring::{Senders, TmEvent};
use crate::utils::error::Error;

pub(super) fn senders() -> Senders {
    Senders {
        tok_manager_tx: flume::unbounded().0,
        abort_tx: flume::unbounded().0,
        tokenizer_tx: flume::unbounded().0,
        detokenizer_tx: vec![],
    }
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
            rid: rid.into(),
            echo: String::new(),
            rx,
        },
        tx,
    )
}

/// A submitted chat choice (the tuple `chat_event_stream` consumes).
pub(super) fn chat_submitted(
    index: usize,
    rid: &str,
) -> (
    (usize, Rid, tokio::sync::mpsc::Receiver<ResponseItem>),
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    ((index, rid.into(), rx), tx)
}

pub(super) fn server_args() -> Arc<ServerArgs> {
    Arc::new(ServerArgs {
        served_model_name: "model".into(),
        ..Default::default()
    })
}

pub(super) fn app_state(senders: Senders) -> Arc<super::AppState> {
    struct TestTokenizer;
    impl TextTokenizer for TestTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, sglang_renderer::RendererError> {
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    let server_args = server_args();
    let limits = Limits::from(&*server_args);
    let (jobs, worker_jobs) = flume::unbounded::<PreprocessJob>();
    std::thread::spawn(move || {
        TokenizerWorker::new(worker_jobs, None, Arc::new(TestTokenizer), limits).run()
    });
    let renderer = Arc::new(new_renderer_service(server_args.clone(), jobs));
    Arc::new(super::AppState {
        frontend: FrontendHandle::new(senders, 8),
        server_args,
        renderer,
        response_activity: Default::default(),
    })
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

/// OpenAI inference lowers protocol fields before submission, but the request
/// must still enter the shared FSM as text. The FSM, rather than the renderer's
/// prepare-only backend, owns inference tokenization.
#[tokio::test]
async fn completions_enter_the_fsm_before_tokenization() {
    let (tm_tx, tm_rx) = flume::unbounded();
    let mut frontend_senders = senders();
    frontend_senders.tok_manager_tx = tm_tx;
    let app = routes().with_state(app_state(frontend_senders));

    let response = tokio::spawn(post_json(
        app,
        "/v1/completions",
        json!({"model": "model", "prompt": "two words"}),
    ));
    let TmEvent::Intake(request) = tm_rx.recv_async().await.unwrap() else {
        panic!("OpenAI request must enter through the FSM intake lane")
    };
    let crate::message::request::RequestKind::Generate(generate) = &request.kind else {
        panic!("completion must lower to a generate request")
    };
    assert_eq!(generate.text.as_deref(), Some("two words"));
    assert!(
        generate.input_ids.is_none(),
        "full inference must leave tokenization to the shared FSM"
    );
    assert!(
        !generate.sampling_params.is_normalized,
        "the shared FSM must own sampling normalization"
    );

    request
        .sink
        .try_send(ResponseItem::Error(Error::Validation(
            "test terminal".into(),
        )))
        .unwrap();
    assert_eq!(response.await.unwrap().status(), StatusCode::BAD_REQUEST);
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
async fn basic_openai_router_excludes_responses_api() {
    let app = routes().with_state(app_state(senders()));
    let response = post_json(app, "/v1/responses", json!({"input": "hi"})).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// A closed tm inbox with a *streaming* request must answer inside the
/// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
/// same `error_response` rule the native API applies), not a unary 503.
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
