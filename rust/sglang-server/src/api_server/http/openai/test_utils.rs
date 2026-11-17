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

use super::{batch_n_error, indexed_decode_stream, model_error, openai_error, routes};
use crate::api_server::core::CoreEvent;
use crate::api_server::handlers::{CoreCall, CoreHandle};
use crate::message::config::ServerArgs;
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::wiring::TmEvent;

/// The formatter a checkpoint with `config` and Jinja `template` selects.
pub(super) fn checkpoint_formatter(
    config: &serde_json::Value,
    template: &str,
) -> super::ChatFormatter {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("jinja-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    let (formatter, error) =
        sglang_processor::select_chat_formatter(&sglang_processor::ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
    std::fs::remove_dir_all(&dir).unwrap();
    formatter.unwrap_or_else(|| panic!("{error:?}"))
}

pub(super) fn core_handle() -> CoreHandle {
    CoreHandle::new(
        flume::unbounded().0,
        flume::unbounded().0,
        crate::api_server::handlers::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            multimodal: false,
            peer_endpoints: Vec::new(),
            metadata: crate::api_server::handlers::CoreMetadata::from(server_args().as_ref()),
            metrics: None,
            request_log: None,
            generate_validator: None,
            abort_hook: None,
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

/// `item` with `cached` prefix-cache hits.
pub(super) fn with_cached_tokens(mut item: ResponseItem, cached: u32) -> ResponseItem {
    if let ResponseItem::Frame(output) | ResponseItem::Done(output) = &mut item {
        output.cached_tokens = cached;
    }
    item
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
    let server_args = server_args();
    Arc::new(super::AppState {
        core,
        generation_policy: crate::message::request::GenerationPolicy::from(server_args.as_ref()),
        server_args,
        chat_formatter: None,
        generate_hook: None,
        openai_chat: None,
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
        crate::api_server::handlers::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            multimodal: false,
            peer_endpoints: Vec::new(),
            metadata: crate::api_server::handlers::CoreMetadata::from(server_args().as_ref()),
            metrics: None,
            request_log: None,
            generate_validator: None,
            abort_hook: None,
        },
    )
}

/// App state whose core intake answers every generation with one finished
/// `ok` chunk, with the ChatML template so chat can render.
fn answering_app_state() -> Arc<super::AppState> {
    let formatter = sglang_processor::load_chat_formatter(None, None, None, Some("chatml"));
    answering_app_state_with(server_args(), formatter.unwrap(), None, "ok")
}

/// [`answering_app_state`], logging its requests to `request_log`.
fn answering_app_state_logged(
    request_log: Arc<crate::api_server::core::RequestLog>,
) -> Arc<super::AppState> {
    let formatter = sglang_processor::load_chat_formatter(None, None, None, Some("chatml"));
    answering_app_state_built(
        server_args(),
        formatter.unwrap(),
        None,
        |request| chunk(request.rid.as_str(), "ok", true),
        Some(request_log),
    )
}

/// App state whose core intake answers every generation with one finished `text` chunk.
pub(super) fn answering_app_state_with(
    server_args: Arc<ServerArgs>,
    formatter: super::ChatFormatter,
    tokenizer: Option<sglang_processor::dynamo_tokenizers::Tokenizer>,
    text: &'static str,
) -> Arc<super::AppState> {
    answering_app_state_with_item(server_args, formatter, tokenizer, move |rid| {
        chunk(rid, text, true)
    })
}

/// [`answering_app_state_with`], answering each request with `answer(rid)`.
pub(super) fn answering_app_state_with_item(
    server_args: Arc<ServerArgs>,
    formatter: super::ChatFormatter,
    tokenizer: Option<sglang_processor::dynamo_tokenizers::Tokenizer>,
    answer: impl Fn(&str) -> ResponseItem + Send + 'static,
) -> Arc<super::AppState> {
    answering_app_state_built(
        server_args,
        formatter,
        tokenizer,
        move |request| answer(request.rid.as_str()),
        None,
    )
}

/// [`answering_app_state_with`], answering each request with `answer(request)`.
pub(super) fn answering_app_state_with_request(
    server_args: Arc<ServerArgs>,
    formatter: super::ChatFormatter,
    tokenizer: Option<sglang_processor::dynamo_tokenizers::Tokenizer>,
    answer: impl Fn(&crate::message::request::Request) -> ResponseItem + Send + 'static,
) -> Arc<super::AppState> {
    answering_app_state_built(server_args, formatter, tokenizer, answer, None)
}

fn answering_app_state_built(
    server_args: Arc<ServerArgs>,
    formatter: super::ChatFormatter,
    tokenizer: Option<sglang_processor::dynamo_tokenizers::Tokenizer>,
    answer: impl Fn(&crate::message::request::Request) -> ResponseItem + Send + 'static,
    request_log: Option<Arc<crate::api_server::core::RequestLog>>,
) -> Arc<super::AppState> {
    let (intake_tx, intake_rx) = flume::unbounded();
    tokio::spawn(async move {
        // Held open, so an unfinished answer stays unfinished.
        let mut open = Vec::new();
        while let Ok(TmEvent::Intake { request, admission }) = intake_rx.recv_async().await {
            assert!(admission.try_accept());
            request.sink.try_send(answer(&request)).unwrap();
            open.push(request);
        }
    });
    let core = CoreHandle::new(
        intake_tx,
        flume::unbounded().0,
        crate::api_server::handlers::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            multimodal: false,
            peer_endpoints: Vec::new(),
            metadata: crate::api_server::handlers::CoreMetadata::from(server_args.as_ref()),
            metrics: None,
            request_log,
            generate_validator: None,
            abort_hook: None,
        },
    );
    // The test tokenizers add no specials, so one handle serves both modes.
    let tokenizer = tokenizer.map(|tokenizer| {
        Arc::new(sglang_processor::DynamoTokenizer::new(
            tokenizer.clone(),
            tokenizer,
        ))
    });
    Arc::new(super::AppState {
        core,
        generation_policy: crate::message::request::GenerationPolicy::from(server_args.as_ref()),
        openai_chat: tokenizer.map(|t| super::OpenAiChat::new(&server_args, &formatter, t)),
        server_args,
        chat_formatter: Some(formatter),
        generate_hook: None,
    })
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

/// The parsed `data:` frames of an SSE `response`, which must end with the
/// `[DONE]` sentinel the OpenAI SDKs stop on.
pub(super) async fn sse_frames(response: Response) -> Vec<serde_json::Value> {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    text.split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
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

/// Python's request models bound `n` by `OPENAI_MAX_N`: both routes refuse a
/// larger one, streamed or not, before rendering or submitting any choice.
#[tokio::test]
async fn openai_handlers_bound_n() {
    let served = routes().with_state(answering_app_state());
    let closed = routes().with_state(app_state(frontend_closed()));
    let messages = json!([{"role": "user", "content": "hi"}]);
    for (path, body) in [
        ("/v1/completions", json!({"model": "model", "prompt": "hi"})),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages}),
        ),
    ] {
        let mut body = body.clone();
        body["n"] = json!(128);
        let response = post_json(served.clone(), path, body.clone()).await;
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let choices = body_json(response).await["choices"]
            .as_array()
            .unwrap()
            .len();
        assert_eq!(choices, 128, "{path}");
        for n in [129, 255] {
            for stream in [false, true] {
                body["n"] = json!(n);
                body["stream"] = json!(stream);
                for app in [&served, &closed] {
                    let response = post_json(app.clone(), path, body.clone()).await;
                    assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path} {body}");
                    let error = body_json(response).await["error"].clone();
                    assert_eq!(error["message"], "n must be at most 128", "{path} {body}");
                }
            }
        }
    }
    // The total across prompts keeps its own bound.
    let body = json!({"model": "model", "prompt": vec!["hi"; 33], "n": 128});
    let response = post_json(closed, "/v1/completions", body).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["error"]["message"],
        "prompt count times n exceeds the maximum of 4096"
    );
}

/// Post the JSON text `body` to `path`, which refuses it for its `n`.
async fn assert_n_refused(app: &Router, path: &str, body: &str, message: &str) {
    let (status, refusal) = post_raw(app, path, body.as_bytes().to_vec()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{path} {body}");
    assert_eq!(refusal, message, "{path} {body}");
}

/// Python's request model bounds `n` before the handler runs, so an `n` out
/// of range is the refusal, whatever else the chat request gets wrong.
#[tokio::test]
async fn chat_refuses_n_before_other_checks() {
    let app = routes().with_state(app_state(core_handle()));
    let text = json!([{"role": "user", "content": "hi"}]);
    // The model is text-only.
    let image = json!([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]);
    // The model check and the media check bracket the handler's other checks.
    for mut body in [
        json!({"model": "model:adapter", "messages": text}),
        json!({"model": "model", "messages": image}),
    ] {
        body["n"] = json!(129);
        let body = body.to_string();
        assert_n_refused(&app, "/v1/chat/completions", &body, "n must be at most 128").await;
    }
}

/// [`chat_refuses_n_before_other_checks`] for completions. `max_tokens=0` is
/// left out: Python's request model refuses it along with `n`.
#[tokio::test]
async fn completions_refuse_n_before_other_checks() {
    let app = routes().with_state(app_state(core_handle()));
    for mut body in [
        json!({"model": "model:adapter", "prompt": "hi"}),
        json!({"model": "model", "prompt": "hi", "suffix": "x"}),
        json!({"model": "model", "prompt": ""}),
    ] {
        body["n"] = json!(129);
        let body = body.to_string();
        assert_n_refused(&app, "/v1/completions", &body, "n must be at most 128").await;
    }
}

/// Python's request models take any integer, or an integral float, as `n` and
/// refuse it by the same bound past the `u8` the typed requests hold.
#[tokio::test]
async fn openai_handlers_bound_n_past_u8() {
    let app = routes().with_state(app_state(core_handle()));
    for (path, fields) in [
        (
            "/v1/chat/completions",
            r#""messages": [{"role": "user", "content": "hi"}]"#,
        ),
        ("/v1/completions", r#""prompt": "hi""#),
    ] {
        for (n, message) in [
            ("256", "n must be at most 128"),
            ("-1", "n must be at least 1"),
            ("0", "n must be at least 1"),
            ("200.0", "n must be at most 128"),
            // serde_json reads an integer past 64 bits as `f64`.
            ("100000000000000000000", "n must be at most 128"),
            ("-100000000000000000000", "n must be at least 1"),
        ] {
            let body = format!(r#"{{"model": "model", {fields}, "n": {n}}}"#);
            assert_n_refused(&app, path, &body, message).await;
        }
    }
}

/// The chat and completions routes with a valid body's members but `n`.
const N_ROUTES: [(&str, &str); 2] = [
    (
        "/v1/chat/completions",
        r#""messages": [{"role": "user", "content": "hi"}]"#,
    ),
    ("/v1/completions", r#""prompt": "hi""#),
];

/// Post the bytes `body` to `path`: the status and the error message.
async fn post_raw(app: &Router, path: &str, body: Vec<u8>) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let response = oneshot(app.clone(), request).await;
    let status = response.status();
    (
        status,
        body_json(response).await["error"]["message"].clone(),
    )
}

/// Python's orjson reads an integer past 64 bits as a float, so an `n` past
/// `u64` but within `f64` is still refused by the bound.
#[tokio::test]
async fn openai_handlers_bound_n_past_u64() {
    let app = routes().with_state(app_state(core_handle()));
    let past_u64 = ["9".repeat(20), format!("1{}", "0".repeat(24))];
    let near_max = format!("1{}", "0".repeat(300));
    for (path, fields) in N_ROUTES {
        for n in past_u64.iter().chain([&near_max]) {
            for (n, message) in [
                (n.clone(), "n must be at most 128"),
                (format!("-{n}"), "n must be at least 1"),
            ] {
                let body = format!(r#"{{"model": "model", {fields}, "n": {n}}}"#);
                assert_n_refused(&app, path, &body, message).await;
            }
        }
    }
}

/// Python's orjson fails to decode a number past `f64`, before any field is
/// validated, so such an `n` gets the parse error the same body gets with
/// another key in its place, at the same position and naming `n`.
#[tokio::test]
async fn openai_handlers_leave_n_past_f64_to_the_parse() {
    let app = routes().with_state(app_state(core_handle()));
    let huge = format!("1{}", "0".repeat(309));
    let long = "1".repeat(4300);
    for (path, fields) in N_ROUTES {
        for n in [
            huge.clone(),
            format!("-{huge}"),
            long.clone(),
            "1e400".to_owned(),
        ] {
            let body = |key: &str| format!(r#"{{"model": "model", {fields}, "{key}": {n}}}"#);
            let (status, message) = post_raw(&app, path, body("x").into_bytes()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path} {message}");
            let prefix = "Failed to parse the request body as JSON: x: number out of range ";
            let message = message.as_str().unwrap_or_default();
            assert!(message.starts_with(prefix), "{path} {message}");
            let parse = json!(message.replacen("JSON: x: ", "JSON: n: ", 1));
            let refused = post_raw(&app, path, body("n").into_bytes()).await;
            assert_eq!(refused, (status, parse), "{path} n of {} chars", n.len());
        }
    }
}

/// Python decodes the body before reading any member, so a body that is not
/// UTF-8 gets the parse error, not the `n` refusal.
#[tokio::test]
async fn openai_handlers_refuse_a_non_utf8_body_before_n() {
    let app = routes().with_state(app_state(core_handle()));
    for (path, member) in [
        (
            "/v1/chat/completions",
            &b"\"messages\": [{\"role\": \"user\", \"content\": \"\xff\"}]"[..],
        ),
        ("/v1/completions", &b"\"prompt\": \"\xff\""[..]),
    ] {
        let body = |n: &[u8]| [&br#"{"model": "model", "#[..], member, n, b"}"].concat();
        let (status, message) = post_raw(&app, path, body(b"")).await;
        assert!(status.is_client_error(), "{path} {status}");
        assert!(
            message
                .as_str()
                .is_some_and(|m| !m.is_empty() && !m.starts_with("n must be")),
            "{path} {message}"
        );
        for n in [&br#", "n": 200"#[..], br#", "n": 0"#] {
            let refused = post_raw(&app, path, body(n)).await;
            assert_eq!(refused, (status, message.clone()), "{path}");
        }
    }
}

/// Only the top-level `n` is read, as Python's orjson keeps it: the
/// last duplicate, by its decoded key; nested ones are other fields' values.
#[tokio::test]
async fn openai_handlers_read_the_top_level_n() {
    let app = routes().with_state(app_state(core_handle()));
    let refused =
        |message: &serde_json::Value| message.as_str().is_some_and(|m| m.starts_with("n must be"));
    for (path, fields) in N_ROUTES {
        // Nested `n` members are not the request's `n`.
        let nested = if path == "/v1/chat/completions" {
            r#""messages": [{"role": "user", "content": "hi", "n": 200}], "logit_bias": {"n": 200}"#
        } else {
            r#""prompt": "hi", "logit_bias": {"n": 200}"#
        };
        let body = format!(r#"{{"model": "model", {nested}}}"#);
        let (_, message) = post_raw(&app, path, body.into_bytes()).await;
        assert!(!refused(&message), "{path} nested: {message}");
        // The last duplicate counts, as it does for orjson and so for Python.
        let body = format!(r#"{{"model": "model", {fields}, "n": 200, "n": 1}}"#);
        let (_, message) = post_raw(&app, path, body.clone().into_bytes()).await;
        assert!(!refused(&message), "{path} duplicate: {message}");
        // Past `n`, the two differ: the chat body reaches the handler through a
        // `Value`, which keeps the last of a repeated key as Python's dict does,
        // while the completions request type is a derive, which refuses one.
        // Python serves both. Pre-existing, and not what this reads `n` for.
        let served = routes().with_state(answering_app_state());
        let (status, message) = post_raw(&served, path, body.clone().into_bytes()).await;
        if path == "/v1/chat/completions" {
            assert_eq!(status, StatusCode::OK, "{path} duplicate: {message}");
        } else {
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path} duplicate");
            let message = message.as_str().unwrap_or_default();
            assert!(message.contains("duplicate field `n`"), "{path}: {message}");
        }
        let body = format!(r#"{{"model": "model", {fields}, "n": 1, "n": 200}}"#);
        assert_n_refused(&app, path, &body, "n must be at most 128").await;
        // An escaped key is decoded first.
        let body = format!(r#"{{"model": "model", {fields}, "\u006e": 200}}"#);
        assert_n_refused(&app, path, &body, "n must be at most 128").await;
        // A float within the bound, integral or not, is left to the parse.
        for n in ["128.0", "128.5"] {
            let body = format!(r#"{{"model": "model", {fields}, "n": {n}}}"#);
            let (_, message) = post_raw(&app, path, body.into_bytes()).await;
            assert!(!refused(&message), "{path} {n}: {message}");
        }
    }
}

/// Python's request models refuse an explicit `null` `n`, which is not an
/// integer, before the handler runs; an omitted `n` is one choice.
#[tokio::test]
async fn openai_handlers_refuse_a_null_n() {
    let served = routes().with_state(answering_app_state());
    let closed = routes().with_state(app_state(frontend_closed()));
    for (path, fields) in N_ROUTES {
        let body = format!(r#"{{"model": "model", {fields}}}"#);
        let response = post_json(served.clone(), path, serde_json::from_str(&body).unwrap()).await;
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let choices = body_json(response).await["choices"]
            .as_array()
            .unwrap()
            .len();
        assert_eq!(choices, 1, "{path}");
        for stream in ["false", "true"] {
            let body = format!(r#"{{"model": "model", {fields}, "n": null, "stream": {stream}}}"#);
            for app in [&served, &closed] {
                assert_n_refused(app, path, &body, "n must be an integer").await;
            }
        }
    }
}

/// Python's `validate_n` bound: `n` up to the batch cap is served, and a
/// negative cap is none.
#[test]
fn batch_cap_bounds_n_unless_negative() {
    assert_eq!(batch_n_error(4, 4), None);
    assert_eq!(
        batch_n_error(5, 4).as_deref(),
        Some("n must be an integer in [1, 4], got 5.")
    );
    assert_eq!(
        batch_n_error(1, 0).as_deref(),
        Some("n must be an integer in [1, 0], got 1.")
    );
    assert_eq!(batch_n_error(u64::MAX, -1), None);
}

/// Values outside the request types' integer ranges are refused while the
/// body is parsed.
#[tokio::test]
async fn chat_refuses_values_its_request_type_cannot_hold() {
    let app = routes().with_state(app_state(core_handle()));
    let prefix = "Failed to deserialize the JSON body into the target type: ";
    for (field, value, error) in [
        (
            "top_logprobs",
            json!(-1),
            "invalid value: integer `-1`, expected u8",
        ),
        (
            "max_tokens",
            json!(-1),
            "invalid value: integer `-1`, expected u32",
        ),
        (
            "max_tokens",
            json!(true),
            "invalid type: boolean `true`, expected u32",
        ),
    ] {
        let mut body = json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]});
        body[field] = value;
        let response = post_json(app.clone(), "/v1/chat/completions", body.clone()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        let message = body_json(response).await["error"]["message"].clone();
        assert_eq!(message, format!("{prefix}{error}"), "{body}");
    }
}

/// Python's chat and completions handlers serve any `model` name and echo it
/// back. Only a `base-model:adapter` name is refused: it asks for a LoRA
/// adapter, which this server does not support.
#[tokio::test]
async fn openai_handlers_serve_any_model_name() {
    let app = routes().with_state(answering_app_state());
    let messages = json!([{"role": "user", "content": "hi"}]);
    for (path, body) in [
        ("/v1/completions", json!({"prompt": "hi"})),
        ("/v1/chat/completions", json!({"messages": messages})),
    ] {
        for model in ["model", "other-model"] {
            let mut body = body.clone();
            body["model"] = json!(model);
            let response = post_json(app.clone(), path, body).await;
            assert_eq!(response.status(), StatusCode::OK, "{path} {model}");
            assert_eq!(body_json(response).await["model"], model, "{path}");
        }
        let mut body = body.clone();
        body["model"] = json!("model:adapter");
        let response = post_json(app.clone(), path, body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        let message = body_json(response).await["error"]["message"].clone();
        assert!(message.as_str().unwrap().contains("`adapter`"), "{message}");
    }
    // As in Python, an empty adapter (`name:`) is none. The served name is
    // always accepted, even with a colon in it.
    assert_eq!(model_error("other-model: ", "model"), None);
    assert!(model_error("org/model:v1", "model").is_some());
    assert_eq!(model_error("org/model:v1", "org/model:v1"), None);
}

/// Both OpenAI streams declare UTF-8, as Python's `StreamingResponse` does.
#[tokio::test]
async fn openai_streams_declare_utf8_event_stream() {
    let app = routes().with_state(answering_app_state());
    let messages = json!([{"role": "user", "content": "hi"}]);
    for (path, body) in [
        (
            "/v1/completions",
            json!({"model": "model", "prompt": "hi", "stream": true}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages, "stream": true}),
        ),
    ] {
        let response = post_json(app.clone(), path, body).await;
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        assert_eq!(
            response.headers()["content-type"],
            crate::utils::response::SSE_CONTENT_TYPE,
            "{path}"
        );
    }
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
    assert_eq!(
        response.headers()["content-type"],
        crate::utils::response::SSE_CONTENT_TYPE
    );
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

/// The OpenAI routes log their requests with the whitelisted headers of the
/// HTTP request, as Python's tokenizer manager does, and from level 2 the
/// OpenAI request first, as Python's `serving_base` does.
#[tokio::test]
async fn openai_request_lines_carry_whitelisted_headers() {
    let log = crate::api_server::core::RequestLog::for_test(
        Some(crate::message::config::RequestLogConfig {
            max_length: 1 << 30,
            skip_names: vec!["text".into()],
            out_skip_names: vec!["text".into(), "output_ids".into()],
            json: false,
            class_name: "GenerateReqInput".into(),
            fields: vec![("rid".into(), "None".into())],
            batch_item_fields: Vec::new(),
            batch_shared_fields: Vec::new(),
            headers: vec!["x-sgl-routing-key".into()],
            exceeded_ms: -1,
            openai_requests: true,
            chat_completion_fields: vec![("model".into(), None)],
            completion_fields: vec![("model".into(), None)],
            stdout: true,
            files: Vec::new(),
        }),
        false,
    );
    let app = routes().with_state(answering_app_state_logged(log.clone()));
    let messages = json!([{"role": "user", "content": "hi"}]);
    for (path, body) in [
        ("/v1/completions", json!({"model": "model", "prompt": "hi"})),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages, "max_completion_tokens": 0}),
        ),
    ] {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .header("x-sgl-routing-key", "conv-1")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = oneshot(app.clone(), req).await;
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let lines = log.logged_lines();
        assert_eq!(lines.len(), 3, "{path}: {lines:?}");
        assert!(
            lines[0].starts_with("Receive OpenAI: obj={'model': 'model'}"),
            "{path}"
        );
        assert!(
            lines[1].starts_with("Receive: obj=GenerateReqInput(rid="),
            "{path}"
        );
        assert!(
            lines[2].starts_with("Finish: obj=GenerateReqInput(rid="),
            "{path}"
        );
        for line in &lines {
            assert!(
                line.contains(", headers={'x-sgl-routing-key': 'conv-1'}"),
                "{path}: {line}"
            );
        }
    }

    // As in Python, a request its model and `_validate_request` accept is
    // logged even when a later check refuses it; one they refuse is not.
    for (path, body, logged) in [
        (
            "/v1/completions",
            json!({"model": "model", "prompt": "hi", "temperature": -1.0}),
            true,
        ),
        (
            "/v1/completions",
            json!({"model": "model", "prompt": ["", ""]}),
            false,
        ),
        // Python's request models refuse an `n` out of range, or one that is
        // not an integer, before it is logged.
        (
            "/v1/completions",
            json!({"model": "model", "prompt": "hi", "n": 129}),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages, "n": 129}),
            false,
        ),
        (
            "/v1/completions",
            json!({"model": "model", "prompt": "hi", "n": null}),
            false,
        ),
        // Chat reads `n` at its own call site, ahead of the lowerer, so cover
        // it as well as the completions one above.
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages, "n": null}),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": messages, "tool_choice": "required"}),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]}]}),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
            ]}]}),
            false,
        ),
        // Media only as Python finds it, in message content parts: not a tool
        // parameter named `file`, which this server refuses after logging.
        (
            "/v1/chat/completions",
            json!({"model": "model", "messages": [
                {"role": "system", "content": "", "tools": [{"type": "function", "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {"file": {"type": "string"}}},
                }}]},
                {"role": "user", "content": "hi"},
            ], "tool_choice": "none"}),
            true,
        ),
    ] {
        let response = post_json(app.clone(), path, body.clone()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        let lines = log.logged_lines();
        assert_eq!(lines.len(), usize::from(logged), "{body}: {lines:?}");
    }

    // A body that is not JSON is refused before it is read, as `Json` does.
    let unreadable = futures::stream::once(async {
        Err::<axum::body::Bytes, _>(std::io::Error::other("reset"))
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .body(Body::from_stream(unreadable))
        .unwrap();
    let response = body_json(oneshot(app.clone(), req).await).await;
    let message = response["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("Content-Type: application/json"),
        "{message}"
    );
}
