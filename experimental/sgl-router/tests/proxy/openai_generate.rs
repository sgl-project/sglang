// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! OpenAI completions served through the engine's `/generate`; exact parity with
//! SGLang is checked by `rust/sglang-processor/tests/openai_parity.rs`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use sgl_router::discovery::WorkerMode;
use sgl_router::state::kv_events::HashTree;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{openai_router, radix_router, MODEL};
use crate::common::mock_worker::MockWorker;
use crate::common::streaming::{collect_body, parse_sse_data};

async fn complete(app: &Router, body: Value) -> (StatusCode, Vec<u8>) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
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
