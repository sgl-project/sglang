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
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::wiring::Senders;

pub(super) fn senders() -> Senders {
    Senders {
        tok_manager_tx: flume::unbounded().0,
        lifecycle_tx: flume::unbounded().0,
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
            rx: rx.into(),
        },
        tx,
    )
}

/// A submitted chat choice (the tuple `chat_event_stream` consumes).
pub(super) fn chat_submitted(
    index: usize,
    rid: &str,
) -> (
    (usize, Rid, super::ResponseReceiver),
    tokio::sync::mpsc::Sender<ResponseItem>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    ((index, rid.into(), rx.into()), tx)
}

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
        server_args: server_args(),
        chat_formatter: None,
        http_extension: None,
        response_activity: Default::default(),
        startup_ready: Arc::new(true.into()),
        frontend_metrics: None,
    })
}

pub(super) fn senders_closed() -> Senders {
    // Dropping the receivers disconnects the channels; the senders stay
    // valid (moveable) but every send reports `Err`, the shutdown state
    // `submit` surfaces as a 503.
    let (tm_tx, tm_rx) = flume::unbounded();
    drop(tm_rx);
    let (lifecycle_tx, lifecycle_rx) = flume::unbounded();
    drop(lifecycle_rx);
    let (tok_tx, tok_rx) = flume::unbounded();
    drop(tok_rx);
    Senders {
        tok_manager_tx: tm_tx,
        lifecycle_tx,
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

#[tokio::test]
async fn model_chat_preserves_custom_fields_and_prefetches_media_before_admission() {
    use crate::message::request::RequestKind;
    use crate::tokenizer_manager::wiring::TmEvent;
    use crate::{ChatInput, GenerateRequest, HttpExtension, MmData, MmItem, RequestPreparation};

    #[derive(Debug)]
    struct ModelChat;
    impl HttpExtension for ModelChat {
        fn apply(&self, router: Router) -> Router {
            router
        }

        fn render_chat(&self, request: &serde_json::Value) -> Result<Option<ChatInput>, String> {
            assert_eq!(
                request["messages"][0]["custom_metadata"]["recipient"],
                "reader"
            );
            assert_eq!(request["chat_template_kwargs"]["thinking"], false);
            Ok(Some(ChatInput {
                text: "<user>describe <image></user>".into(),
                mm: Some(Box::new(MmData {
                    image_data: vec![MmItem::Source(
                        request["messages"][0]["content"][1]["image_url"]["url"]
                            .as_str()
                            .unwrap()
                            .into(),
                    )],
                    ..Default::default()
                })),
            }))
        }

        fn prepare_request<'a>(
            &'a self,
            request: &'a mut GenerateRequest,
        ) -> RequestPreparation<'a> {
            Box::pin(async move {
                assert!(request.mm.as_ref().unwrap().prefetched.is_empty());
                request.priority = Some(17);
                Ok(())
            })
        }
    }
    let (tx, rx) = flume::unbounded();
    let mut senders = senders_closed();
    senders.tok_manager_tx = tx;
    let state = Arc::new(super::AppState {
        senders,
        response_buf: 8,
        server_args: Arc::new(ServerArgs {
            served_model_name: "model".into(),
            ..Default::default()
        }),
        chat_formatter: None,
        http_extension: Some(Arc::new(ModelChat)),
        response_activity: Default::default(),
        startup_ready: Arc::new(true.into()),
        frontend_metrics: None,
    });
    let scheduler = tokio::spawn(async move {
        for _ in 0..2 {
            let TmEvent::Intake(request) = rx.recv_async().await.unwrap() else {
                panic!("intake")
            };
            let RequestKind::Generate(payload) = &request.kind else {
                panic!("generate")
            };
            assert_eq!(
                payload.text.as_deref(),
                Some("<user>describe <image></user>")
            );
            assert!(payload.skip_special_tokens);
            assert_eq!(payload.priority, Some(17));
            assert_eq!(&payload.mm.as_ref().unwrap().prefetched[0][..], b"fixture");
            request
                .sink
                .try_send(chunk(request.rid.client_facing(), "described", true))
                .unwrap();
        }
    });
    let image_path = std::env::temp_dir().join(format!("chat-image-{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&image_path, b"fixture").unwrap();
    let response = post_json(routes().with_state(state), "/v1/chat/completions", json!({
        "model": "org/model-alias", "n": 2, "messages": [{
            "role": "user", "custom_metadata": {"recipient": "reader"},
            "content": [{"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": format!("file://{}", image_path.display())}}]
        }], "chat_template_kwargs": {"thinking": false}
    })).await;
    let status = response.status();
    let body = body_json(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "org/model-alias");
    assert_eq!(body["choices"].as_array().unwrap().len(), 2);
    assert_eq!(body["choices"][0]["message"]["content"], "described");
    assert_eq!(body["choices"][1]["message"]["content"], "described");
    scheduler.await.unwrap();
    std::fs::remove_file(image_path).unwrap();
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
    assert_eq!(value["message"], "bad input");
    assert_eq!(value["type"], "BadRequestError");
    assert_eq!(value["code"], 400);
    assert!(value["param"].is_null());

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
        (json!({"model": "model", "prompt": "hi", "n": 0}), "n=0"),
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

#[tokio::test]
async fn prompt_only_completion_accepts_model_alias_and_echoes_without_output_tokens() {
    use crate::message::request::RequestKind;
    use crate::tokenizer_manager::wiring::TmEvent;

    let (tx, rx) = flume::unbounded();
    let mut senders = senders_closed();
    senders.tok_manager_tx = tx;
    let scheduler = tokio::spawn(async move {
        let TmEvent::Intake(request) = rx.recv_async().await.unwrap() else {
            panic!("intake")
        };
        let RequestKind::Generate(payload) = &request.kind else {
            panic!("generate")
        };
        assert_eq!(payload.sampling_params.max_new_tokens, Some(0));
        assert_eq!(payload.text.as_deref(), Some("hello"));
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent {
                rid: request.rid.client_facing().into(),
                prompt_tokens: 1,
                finish_reason: Some(
                    serde_json::from_value(json!({"type":"length", "length":0})).unwrap(),
                ),
                ..Default::default()
            }))
            .unwrap();
    });
    let response = post_json(
        routes().with_state(app_state(senders)),
        "/v1/completions",
        json!({
            "model":"org/alias", "prompt":"hello", "max_tokens":0, "echo":true
        }),
    )
    .await;
    let status = response.status();
    let body = body_json(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "org/alias");
    assert_eq!(body["choices"][0]["text"], "hello");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["prompt_tokens"], 1);
    assert_eq!(body["usage"]["completion_tokens"], 0);
    scheduler.await.unwrap();
}

#[tokio::test]
async fn openai_extensions_match_python_adapters_through_http_admission() {
    use crate::message::request::RequestKind;
    use crate::tokenizer_manager::wiring::TmEvent;

    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../testdata/openai_requests_python.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let (tx, rx) = flume::unbounded();
        let mut senders = senders_closed();
        senders.tok_manager_tx = tx;
        let mut state = app_state(senders);
        let mut args = ServerArgs {
            dp_size: 8,
            enable_request_header_overrides: true,
            ..Default::default()
        };
        args.model_config.vocab_size = 100;
        args.model_config.default_sampling_params.top_k = case["model_defaults"]["top_k"].as_i64();
        args.model_config.default_sampling_params.min_p = case["model_defaults"]["min_p"].as_f64();
        args.model_config.default_sampling_params.repetition_penalty =
            case["model_defaults"]["repetition_penalty"].as_f64();
        Arc::get_mut(&mut state).unwrap().server_args = Arc::new(args);
        let expected = case["expected"].as_array().unwrap().clone();
        let request_fields = fixture["request_fields"].as_array().unwrap().clone();
        let sampling_fields = fixture["sampling_fields"].as_array().unwrap().clone();
        let samples = case["body"]["n"].as_u64().unwrap_or(1);
        let scheduler = tokio::spawn(async move {
            for (index, expected) in expected.into_iter().enumerate() {
                let TmEvent::Intake(request) = rx.recv_async().await.unwrap() else {
                    panic!("intake")
                };
                let RequestKind::Generate(payload) = &request.kind else {
                    panic!("generate")
                };
                let actual = serde_json::to_value(payload).unwrap();
                let sampling = serde_json::to_value(&payload.sampling_params).unwrap();
                for field in &request_fields {
                    let name = field.as_str().unwrap();
                    let expected_field = if name == "bootstrap_room" && samples > 1 {
                        // Native parallel samples reserve distinct P/D rooms;
                        // the Python adapter supplies each prompt's base room.
                        expected[name]
                            .as_i64()
                            .map(|room| {
                                json!(room * samples as i64 + (index as u64 % samples) as i64)
                            })
                            .unwrap_or(serde_json::Value::Null)
                    } else {
                        expected[name].clone()
                    };
                    assert_eq!(actual[name], expected_field, "{name}: {actual}");
                }
                for field in &sampling_fields {
                    let name = field.as_str().unwrap();
                    let comparable = |value: &serde_json::Value| {
                        if matches!(name, "json_schema" | "structural_tag") {
                            value
                                .as_str()
                                .map(|text| serde_json::from_str(text).unwrap())
                                .unwrap_or(serde_json::Value::Null)
                        } else {
                            value.clone()
                        }
                    };
                    assert_eq!(
                        comparable(&sampling[name]),
                        comparable(&expected["sampling"][name]),
                        "sampling {name}: {actual}"
                    );
                }
                request
                    .sink
                    .try_send(ResponseItem::Done(ChunkEvent {
                        rid: request.rid.client_facing().into(),
                        text: "ok".into(),
                        token_ids: vec![1],
                        prompt_tokens: 3,
                        completion_tokens: 1,
                        finish_reason: Some(
                            serde_json::from_value(json!({"type":"length", "length":1})).unwrap(),
                        ),
                        ..Default::default()
                    }))
                    .unwrap();
            }
        });
        let path = if case["endpoint"] == "chat" {
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        };
        let mut request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json");
        for (key, value) in case["headers"].as_object().unwrap() {
            request = request.header(key, value.as_str().unwrap());
        }
        let response = oneshot(
            routes().with_state(state),
            request.body(Body::from(case["body"].to_string())).unwrap(),
        )
        .await;
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        scheduler.await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}: {}",
            case,
            String::from_utf8_lossy(&body)
        );
        assert!(!String::from_utf8_lossy(&body).contains("\"error\""));
        if samples == 1
            && let Some(rid) = case["body"]["rid"].as_str()
        {
            let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(response["id"], rid);
        }
    }
    for case in fixture["invalid"].as_array().unwrap() {
        let path = if case["endpoint"] == "chat" {
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        };
        let mut request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json");
        for (key, value) in case["headers"].as_object().unwrap() {
            request = request.header(key, value.as_str().unwrap());
        }
        let response = oneshot(
            routes().with_state(app_state(senders_closed())),
            request.body(Body::from(case["body"].to_string())).unwrap(),
        )
        .await;
        let status = response.status();
        let body = body_json(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case}: {body}");
        assert_eq!(body["code"], 400);
    }
}

#[tokio::test]
async fn chat_without_tokenizer_fails_with_http_500_before_opening_a_stream() {
    let (tx, rx) = flume::unbounded();
    let mut senders = senders_closed();
    senders.tok_manager_tx = tx;
    let mut state = app_state(senders);
    Arc::get_mut(&mut state).unwrap().server_args = Arc::new(ServerArgs {
        skip_tokenizer_init: true,
        ..Default::default()
    });
    let app = routes().with_state(state);
    for stream in [false, true] {
        let response = post_json(
            app.clone(),
            "/v1/chat/completions",
            json!({
                "model":"synthetic", "messages":[{"role":"user", "content":"hi"}], "stream":stream
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(response).await;
        assert_eq!(body["code"], 500);
        assert_eq!(body["type"], "InternalServerError");
        assert!(body["message"].as_str().unwrap().contains("tokenizer"));
    }
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn basic_openai_router_excludes_responses_api() {
    let app = routes().with_state(app_state(senders()));
    let response = post_json(app, "/v1/responses", json!({"input": "hi"})).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn stream_validation_status_and_error_shape_match_python() {
    use crate::tokenizer_manager::wiring::TmEvent;
    use crate::utils::error::Error;
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("../../../testdata/openai_errors_python.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let (tx, rx) = flume::unbounded();
        let mut channels = senders_closed();
        channels.tok_manager_tx = tx;
        let mut state = app_state(channels);
        Arc::get_mut(&mut state).unwrap().server_args = Arc::new(ServerArgs {
            return_input_ids: true,
            return_output_ids: true,
            ..Default::default()
        });
        let late = case["late"].as_bool().unwrap();
        let worker = tokio::spawn(async move {
            let TmEvent::Intake(request) = rx.recv_async().await.unwrap() else {
                panic!("intake")
            };
            if late {
                request
                    .sink
                    .try_send(ResponseItem::Frame(ChunkEvent {
                        rid: request.rid.clone(),
                        text: "x".into(),
                        token_ids: vec![10],
                        prompt_tokens: 2,
                        completion_tokens: 1,
                        extras: Some(Box::new(crate::message::response::ChunkExtras {
                            prompt_token_ids: Some(vec![1, 2].into()),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }))
                    .unwrap();
            }
            request
                .sink
                .try_send(ResponseItem::Error(Error::Validation("bad input".into())))
                .unwrap();
        });
        let path = if case["endpoint"] == "chat" {
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        };
        let response = post_json(routes().with_state(state), path, case["body"].clone()).await;
        assert_eq!(
            response.status().as_u16(),
            case["status"].as_u64().unwrap() as u16
        );
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        if late {
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(text.contains("data: [DONE]"));
            let frames: Vec<serde_json::Value> = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).unwrap())
                .collect();
            let errors: Vec<_> = frames
                .iter()
                .filter(|frame| frame.get("error").is_some())
                .collect();
            assert_eq!(errors, vec![&case["error"]]);
            assert!(frames.iter().all(|frame| frame.get("sglext").is_none()));
        } else {
            let actual: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(actual, case["error"]);
        }
        worker.await.unwrap();
    }
}

#[tokio::test]
async fn stream_priming_accepts_the_first_ready_parallel_choice() {
    use crate::tokenizer_manager::wiring::TmEvent;
    use crate::utils::error::Error;
    let (tx, rx) = flume::unbounded();
    let mut channels = senders_closed();
    channels.tok_manager_tx = tx;
    let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
    let worker = tokio::spawn(async move {
        let TmEvent::Intake(first) = rx.recv_async().await.unwrap() else {
            panic!("intake")
        };
        let TmEvent::Intake(second) = rx.recv_async().await.unwrap() else {
            panic!("intake")
        };
        second
            .sink
            .try_send(ResponseItem::Frame(ChunkEvent {
                rid: second.rid.clone(),
                text: "ready".into(),
                token_ids: vec![10],
                prompt_tokens: 2,
                completion_tokens: 1,
                ..Default::default()
            }))
            .unwrap();
        finish_rx.await.unwrap();
        first
            .sink
            .try_send(ResponseItem::Error(Error::Validation("late".into())))
            .unwrap();
        second
            .sink
            .try_send(ResponseItem::Done(ChunkEvent {
                rid: second.rid.clone(),
                prompt_tokens: 2,
                finish_reason: Some(
                    serde_json::from_value(json!({"type":"length","length":1})).unwrap(),
                ),
                ..Default::default()
            }))
            .unwrap();
    });
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        post_json(
            routes().with_state(app_state(channels)),
            "/v1/completions",
            json!({"model":"model","prompt":[1,2],"n":2,"stream":true}),
        ),
    )
    .await
    .expect("priming must not wait for the first indexed choice");
    assert_eq!(response.status(), StatusCode::OK);
    finish_tx.send(()).unwrap();
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("ready"));
    assert!(text.contains("validation failed: late"));
    assert!(text.contains("[DONE]"));
    worker.await.unwrap();
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
