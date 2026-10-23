// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! OpenAI `/v1/embeddings` and SGLang's `/v1/classify`: the engine's schema, with
//! text input forwarded as router tokens.

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
use crate::common::streaming::collect_body;

async fn send(app: &Router, body: Value) -> Response {
    let request = Request::post("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(request).await.unwrap()
}

fn prompt_ids(text: &str) -> Vec<u32> {
    let tokenizers = TokenizerRegistry::load_from_config(&config()).unwrap();
    tokenizers.encode_prompt(MODEL, text).unwrap()
}

#[tokio::test]
async fn embeddings_forward_text_as_input_ids() {
    let engine = MockWorker::start(vec![]).await;
    let app = radix_router(&[(&engine, WorkerMode::Plain)], HashTree::new());

    let res = send(&app, json!({"model": MODEL, "input": "hi"})).await;
    let response: Value = serde_json::from_slice(&collect_body(res.into_body()).await).unwrap();
    assert_eq!(response["data"][0]["embedding"], json!([0.5]));
    // A single prompt also gets an rid for abort-on-disconnect.
    let forwarded = engine.captured_json().await;
    assert_eq!(forwarded["input"], json!(prompt_ids("hi")));
    assert!(forwarded["rid"]
        .as_str()
        .is_some_and(crate::common::is_engine_shaped_rid));

    // A batch needs a list of rids, so none is minted.
    send(&app, json!({"model": MODEL, "input": ["hi", "yo"]})).await;
    let expected = json!({"model": MODEL, "input": [prompt_ids("hi"), prompt_ids("yo")]});
    assert_eq!(engine.captured_json().await, expected);
}

#[tokio::test]
async fn classify_forwards_token_ids_for_one_prompt_only() {
    let engine = MockWorker::start(vec![]).await;
    let app = radix_router(&[(&engine, WorkerMode::Plain)], HashTree::new());
    // `ClassifyRequest` takes no batch of token-ID lists.
    for (input, forwarded) in [
        (json!("hi"), json!(prompt_ids("hi"))),
        (json!(["hi", "yo"]), json!(["hi", "yo"])),
    ] {
        let request = Request::post("/v1/classify")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model": MODEL, "input": input}).to_string(),
            ))
            .unwrap();
        let res = app.clone().oneshot(request).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(engine.captured_json().await["input"], forwarded);
    }
}

#[tokio::test]
async fn embeddings_route_text_to_the_worker_caching_its_prefix() {
    let (cached, uncached) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let tree = HashTree::new();
    let hashes = compute_block_hashes(&prompt_ids("local radix cache hit"), 1);
    tree.insert(&KvWorkerId::new(cached.url.clone(), 0), None, &hashes);
    let app = radix_router(
        &[(&uncached, WorkerMode::Plain), (&cached, WorkerMode::Plain)],
        tree,
    );

    // Without tokens the pick is a load tie broken at random, so one request proves nothing.
    for _ in 0..16 {
        let res = send(
            &app,
            json!({"model": MODEL, "input": "local radix cache hit"}),
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
    }
    assert!(uncached.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn embeddings_reject_prefill_decode_workers_and_other_models() {
    let (prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let workers = [
        (&prefill, WorkerMode::Prefill),
        (&decode, WorkerMode::Decode),
    ];
    let app = radix_router(&workers, HashTree::new());

    let res = send(&app, json!({"model": MODEL, "input": "hi"})).await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let res = send(&app, json!({"model": "other", "input": "hi"})).await;
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
}
