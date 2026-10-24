// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Native `/generate`: the engine's schema, with `text` forwarded as router tokens.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::Router;
use serde_json::{json, Value};
use sgl_router::discovery::WorkerMode;
use sgl_router::state::kv_events::{compute_block_hashes, HashTree, KvWorkerId};
use sgl_router::tokenizer::TokenizerRegistry;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, radix_router, MODEL};
use crate::common::mock_worker::MockWorker;
use crate::common::streaming::{collect_body, parse_sse_data};

async fn send(app: &Router, method: &str, body: &Value) -> Response {
    let request = Request::builder()
        .method(method)
        .uri("/generate")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    app.clone().oneshot(request).await.unwrap()
}

fn prompt_ids(text: &str) -> Vec<u32> {
    let tokenizers = TokenizerRegistry::load_from_config(&config()).unwrap();
    tokenizers.encode_prompt(MODEL, text).unwrap()
}

#[tokio::test]
async fn generate_forwards_text_as_input_ids_and_the_engine_response() {
    let engine = MockWorker::start(vec![]).await;
    let app = radix_router(&[(&engine, WorkerMode::Plain)], HashTree::new());
    let sent = json!({"text": "hi", "sampling_params": {"max_new_tokens": 8}});

    let res = send(&app, "POST", &sent).await;
    assert_eq!(res.status(), StatusCode::OK);
    let response: Value = serde_json::from_slice(&collect_body(res.into_body()).await).unwrap();

    // `text` becomes `input_ids`, and an rid is added for abort-on-disconnect.
    let mut forwarded = engine.captured_json().await;
    let rid = forwarded.as_object_mut().unwrap().remove("rid").unwrap();
    assert!(rid
        .as_str()
        .is_some_and(crate::common::is_engine_shaped_rid));
    let expected = json!({"input_ids": prompt_ids("hi"), "sampling_params": {"max_new_tokens": 8}});
    assert_eq!(forwarded, expected);
    assert_eq!(response["meta_info"]["id"], rid);
}

#[tokio::test]
async fn generate_accepts_put_and_keeps_caller_inputs() {
    let engine = MockWorker::start(vec![]).await;
    let app = radix_router(&[(&engine, WorkerMode::Plain)], HashTree::new());
    let sent = json!({"input_ids": [[1], [2]], "rid": ["r0", "r1"]});

    assert_eq!(send(&app, "PUT", &sent).await.status(), StatusCode::OK);
    assert_eq!(engine.captured_json().await, sent);
}

#[tokio::test]
async fn generate_streams_engine_events_through() {
    let engine = MockWorker::start(vec!["data: {\"text\":\"ok\"}\n\n", "data: [DONE]\n\n"]).await;
    let app = radix_router(&[(&engine, WorkerMode::Plain)], HashTree::new());

    let res = send(&app, "POST", &json!({"text": "hi", "stream": true})).await;
    assert_eq!(
        res.headers()["content-type"],
        "text/event-stream; charset=utf-8"
    );
    let events = parse_sse_data(&collect_body(res.into_body()).await);
    assert_eq!(events, ["{\"text\":\"ok\"}", "[DONE]"]);
}

#[tokio::test]
async fn pd_generate_sends_one_bootstrap_room_to_both_workers() {
    let (prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let workers = [
        (&prefill, WorkerMode::Prefill),
        (&decode, WorkerMode::Decode),
    ];
    let app = radix_router(&workers, HashTree::new());

    let res = send(&app, "POST", &json!({"text": "hi"})).await;
    assert_eq!(res.status(), StatusCode::OK);
    let (p, d) = (prefill.captured_json().await, decode.captured_json().await);
    assert_eq!(p, d);
    assert_eq!(p["bootstrap_port"], 8997);
}

#[tokio::test]
async fn generate_routes_text_to_the_worker_caching_its_prefix() {
    let (cached, uncached) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let tree = HashTree::new();
    let hashes = compute_block_hashes(&prompt_ids("local radix cache hit"), 1);
    tree.insert(&KvWorkerId::new(cached.url.clone(), 0), None, &hashes);
    let workers = [(&uncached, WorkerMode::Plain), (&cached, WorkerMode::Plain)];
    let app = radix_router(&workers, tree);

    // Without tokens the pick is a load tie broken at random, so one request proves nothing.
    let body = json!({"text": "local radix cache hit"});
    for _ in 0..16 {
        assert_eq!(send(&app, "POST", &body).await.status(), StatusCode::OK);
    }
    assert!(uncached.captured.lock().unwrap().last_body.is_none());
}
