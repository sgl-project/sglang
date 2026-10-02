// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_router::discovery::{ModelId, WorkerMode};
use sgl_router::policies::request_tokens_for;
use sgl_router::state::kv_events::{compute_block_hashes, HashTree, KvWorkerId};
use sgl_router::tokenizer::TokenizerRegistry;
use sha2::Digest;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, radix_router, MODEL};
use crate::common::mock_worker::MockWorker;

#[tokio::test]
async fn cache_salt_routes_to_its_namespace_not_the_unsalted_cache() {
    let salted = MockWorker::start(vec![]).await;
    let unsalted = MockWorker::start(vec![]).await;
    let tokens = [1, 2, 3, 4];
    // Engine event hashes: SHA256("sglang-cache-salt-v1\0" + salt)
    // seeds the chain; each page contains one little-endian u32 token.
    let mut prior = sha2::Sha256::digest(b"sglang-cache-salt-v1\0tenant-a");
    let salted_hashes: Vec<i64> = tokens
        .iter()
        .map(|token: &u32| {
            let mut hash = sha2::Sha256::new();
            hash.update(prior);
            hash.update(token.to_le_bytes());
            prior = hash.finalize();
            i64::from_be_bytes(prior[..8].try_into().unwrap())
        })
        .collect();
    let tree = HashTree::new();
    tree.insert(
        &KvWorkerId::new(salted.url.clone(), 0),
        None,
        &salted_hashes,
    );
    tree.insert(
        &KvWorkerId::new(unsalted.url.clone(), 0),
        None,
        &compute_block_hashes(&tokens, 1),
    );
    let app = radix_router(
        &[(&salted, WorkerMode::Plain), (&unsalted, WorkerMode::Plain)],
        tree,
    );
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "salted prefix"}],
        "input_ids": tokens,
        "cache_salt": "tenant-a",
    });
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        salted.captured.lock().unwrap().last_body.is_some(),
        "a salted request must select the worker holding that salted prefix"
    );
    assert!(unsalted.captured.lock().unwrap().last_body.is_none());
    assert_eq!(salted.captured_json().await["cache_salt"], "tenant-a");
}

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
