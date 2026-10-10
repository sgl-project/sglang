// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! OpenAI chat and completions served through the engine's `/generate`; exact
//! parity with SGLang is checked by `rust/sglang-processor/tests/openai_parity.rs`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use sgl_router::discovery::WorkerMode;
use sgl_router::state::kv_events::HashTree;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{openai_router, radix_router, MODEL};
use crate::common::mock_worker::MockWorker;
use crate::common::streaming::{collect_body, parse_sse_data};

async fn complete(app: &Router, body: Value) -> (StatusCode, Vec<u8>) {
    post(app, "/v1/completions", body).await
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Vec<u8>) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    (
        response.status(),
        collect_body(response.into_body()).await.to_vec(),
    )
}

#[tokio::test]
async fn completions_go_through_generate() {
    let engine = MockWorker::start(vec![]).await;
    let app = openai_router(&[(&engine, WorkerMode::Plain)]);

    let (status, body) = complete(
        &app,
        json!({"model": MODEL, "prompt": "hi", "max_tokens": 4}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let sent = engine.captured_json().await;
    assert_eq!(sent["sampling_params"]["max_new_tokens"], 4);
    assert_eq!(sent["return_text_in_logprobs"], true);
    let response: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["object"], "text_completion");
    assert_eq!(response["id"], sent["rid"]);
    assert_eq!(response["choices"][0]["text"], "ok");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn streamed_generate_frames_become_completion_chunks() {
    let frames = vec![
        "data: {\"text\":\"o\",\"meta_info\":{\"id\":\"r\",\"finish_reason\":null}}\n\n",
        "data: {\"text\":\"ok\",\"meta_info\":{\"id\":\"r\",\"finish_reason\":{\"type\":\"stop\"}}}\n\n",
        "data: [DONE]\n\n",
    ];
    let engine = MockWorker::start(frames).await;
    let app = openai_router(&[(&engine, WorkerMode::Plain)]);

    let request = json!({"model": MODEL, "prompt": "hi", "stream": true});
    let (status, body) = complete(&app, request).await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse_data(&body);
    let chunk = |i: usize| serde_json::from_str::<Value>(&events[i]).unwrap();
    assert_eq!(chunk(0)["choices"][0]["text"], "o");
    assert_eq!(chunk(1)["choices"][0]["text"], "k");
    assert_eq!(chunk(1)["choices"][0]["finish_reason"], "stop");
    assert_eq!(events[2], "[DONE]");
}

/// Python ends the stream at an aborted choice, without waiting for the others.
#[tokio::test]
async fn an_aborted_choice_ends_the_stream_at_once() {
    let abort = "data: {\"index\":1,\"text\":\"\",\"meta_info\":{\"id\":\"r\",\
        \"finish_reason\":{\"type\":\"abort\",\"message\":\"OOM\",\"status_code\":503}}}\n\n";
    let pending = "data: {\"text\":\"o\",\"meta_info\":{\"id\":\"r\",\"finish_reason\":null}}\n\n";
    let frames = [vec![abort], vec![pending; 20], vec!["data: [DONE]\n\n"]].concat();
    let engine = MockWorker::start_slow_stream(frames, Duration::from_millis(100)).await;
    let app = openai_router(&[(&engine, WorkerMode::Plain)]);

    let request = json!({"model": MODEL, "prompt": "hi", "n": 2, "stream": true});
    let reply = tokio::time::timeout(Duration::from_secs(1), complete(&app, request));
    let (status, body) = reply.await.expect("the stream outlived the abort");
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse_data(&body);
    assert_eq!(
        serde_json::from_str::<Value>(&events[0]).unwrap()["error"]["code"],
        503
    );
    assert_eq!(events[1..], ["[DONE]"]);
}

/// The engine's `[DONE]` finishes the request, so its stream is read to the
/// end rather than dropped, which would abort the finished request.
#[tokio::test]
async fn a_finished_stream_sends_no_abort() {
    let frames = vec![
        "data: {\"text\":\"ok\",\"meta_info\":{\"id\":\"r\",\"finish_reason\":{\"type\":\"stop\"}}}\n\n",
        "data: [DONE]\n\n",
        "", // the engine closes its body a moment after `[DONE]`
    ];
    let engine = MockWorker::start_slow_stream(frames, Duration::from_millis(50)).await;
    let app = openai_router(&[(&engine, WorkerMode::Plain)]);

    let request = json!({"model": MODEL, "prompt": "hi", "stream": true});
    let (status, body) = complete(&app, request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parse_sse_data(&body).last().unwrap(), "[DONE]");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(engine.abort_log.lock().unwrap().is_empty());
}

#[tokio::test]
async fn completions_go_to_the_engine_route_when_unsupported() {
    let engine = MockWorker::start(vec![]).await;
    // Without OpenAI settings from `/server_info`, the router cannot match SGLang.
    let app = radix_router(&[(&engine, WorkerMode::Plain)], HashTree::new());
    let request = json!({"model": MODEL, "prompt": "hi"});
    assert_eq!(complete(&app, request.clone()).await.0, StatusCode::OK);
    let mut sent = engine.captured_json().await;
    sent.as_object_mut().unwrap().remove("rid"); // for abort-on-disconnect, as chat
    assert_eq!(sent, request);
}

#[tokio::test]
async fn chat_goes_through_generate_with_rendered_ids() {
    let engine = MockWorker::start(vec![]).await;
    let app = openai_router(&[(&engine, WorkerMode::Plain)]);

    let messages = json!([{"role": "user", "content": "hi"}]);
    // No `max_tokens`: the fixture has no `config.json` to check it against.
    let request = json!({"model": MODEL, "messages": messages});
    let (status, body) = post(&app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK);
    let sent = engine.captured_json().await;
    assert!(sent["input_ids"]
        .as_array()
        .is_some_and(|ids| !ids.is_empty()));
    assert_eq!(sent["require_reasoning"], false);
    let response: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["choices"][0]["message"]["content"], "ok");
}

#[tokio::test]
async fn an_engine_stream_without_frames_ends_the_response() {
    let engine = MockWorker::start(vec![]).await;
    let app = openai_router(&[(&engine, WorkerMode::Plain)]);
    let request = json!({"model": MODEL, "prompt": "hi", "stream": true});
    let (status, body) = complete(&app, request).await;
    assert_eq!((status, body.len()), (StatusCode::OK, 0));
}

#[tokio::test]
async fn pd_legs_both_get_the_lowered_body() {
    let (prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let app = openai_router(&[
        (&prefill, WorkerMode::Prefill),
        (&decode, WorkerMode::Decode),
    ]);
    let (status, _) = complete(&app, json!({"model": MODEL, "prompt": "hi"})).await;
    assert_eq!(status, StatusCode::OK);
    let (p, d) = (prefill.captured_json().await, decode.captured_json().await);
    assert_eq!(p, d);
    assert_eq!(
        (
            p["bootstrap_port"].clone(),
            p["return_text_in_logprobs"].clone()
        ),
        (json!(8997), json!(true))
    );
}
