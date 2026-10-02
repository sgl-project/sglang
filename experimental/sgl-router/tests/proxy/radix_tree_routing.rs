// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_router::discovery::{ModelId, WorkerMode};
use sgl_router::policies::request_tokens_for;
use sgl_router::state::kv_events::{compute_block_hashes, HashTree, KvWorkerId};
use sgl_router::tokenizer::TokenizerRegistry;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, radix_router, MODEL};
use crate::common::mock_worker::MockWorker;

#[tokio::test]
async fn radix_tree_routes_cache_aware_request_to_cached_worker() {
    let cached = MockWorker::start(vec![]).await;
    let uncached = MockWorker::start(vec![]).await;
    let tokenizers = TokenizerRegistry::load_from_config(&config()).unwrap();
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "local radix cache hit"}],
    });
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
        .expect("test prompt tokenizes");
    let hashes = compute_block_hashes(&tokens.ids, 1);
    assert!(!hashes.is_empty());

    let tree = HashTree::new();
    tree.insert(&KvWorkerId::new(cached.url.clone(), 0), None, &hashes);
    let workers = [(&cached, WorkerMode::Plain), (&uncached, WorkerMode::Plain)];

    let response = radix_router(&workers, tree)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(cached.captured.lock().unwrap().last_body.is_some());
    assert!(uncached.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn pending_prefix_keeps_a_cold_burst_on_one_worker() {
    let a = MockWorker::start(vec![]).await;
    let b = MockWorker::start(vec![]).await;
    let tree = HashTree::new();
    tree.pending().enable(Duration::from_secs(60));
    let router = radix_router(&[(&a, WorkerMode::Plain), (&b, WorkerMode::Plain)], tree);
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "shared cold prefix"}],
    });

    let mut picks = Vec::new();
    for _ in 0..8 {
        for worker in [&a, &b] {
            worker.captured.lock().unwrap().last_body = None;
        }
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        picks.push(a.captured.lock().unwrap().last_body.is_some());
    }
    assert!(
        picks.iter().all(|&p| p == picks[0]),
        "burst scattered: {picks:?}"
    );
}
