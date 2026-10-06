// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_router::discovery::{ModelId, WorkerMode};
use sgl_router::policies::request_tokens_for;
use sgl_router::state::kv_events::{
    compute_block_hashes, CacheNamespace, HashTree, KvEventIndex, KvWorkerId,
};
use sgl_router::tokenizer::TokenizerRegistry;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, radix_router, reorg_radix_router, MODEL};
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
async fn lora_requests_match_only_their_adapter_blocks() {
    let base = MockWorker::start(vec![]).await;
    let lora = MockWorker::start(vec![]).await;
    let tokenizers = TokenizerRegistry::load_from_config(&config()).unwrap();
    let messages = json!([{"role": "user", "content": "same prompt, two adapters"}]);
    let body = json!({"model": MODEL, "messages": messages});
    let ids = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
        .expect("test prompt tokenizes")
        .ids;
    let adapter_a = CacheNamespace {
        lora_name: Some("adapter-a".into()),
        ..Default::default()
    };

    // Same tokens: `base` holds the base-model blocks, `lora` the adapter's.
    let tree = HashTree::new();
    tree.insert(
        &KvWorkerId::new(base.url.clone(), 0),
        None,
        &compute_block_hashes(&ids, 1),
    );
    let lora_hashes = adapter_a.block_hashes(&ids, 1, false);
    tree.insert(&KvWorkerId::new(lora.url.clone(), 0), None, &lora_hashes);
    let router = radix_router(
        &[(&base, WorkerMode::Plain), (&lora, WorkerMode::Plain)],
        tree,
    );

    let chat = "/v1/chat/completions";
    let embedding = json!({"model": format!("{MODEL}:adapter-a"), "input": ids});
    for (uri, body, expected) in [
        (
            chat,
            json!({"model": format!("{MODEL}:adapter-a"), "messages": messages}),
            &lora,
        ),
        (
            chat,
            json!({"model": MODEL, "messages": messages, "lora_path": ["adapter-a"]}),
            &lora,
        ),
        ("/v1/embeddings", embedding, &lora),
        (chat, body, &base),
    ] {
        for worker in [&base, &lora] {
            worker.captured.lock().unwrap().last_body = None;
        }
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            expected.captured.lock().unwrap().last_body.is_some(),
            "{body}"
        );
    }
}

/// Route eight identical cold requests and return, per request, whether `a` got it.
async fn burst_picks(router: axum::Router, a: &MockWorker, b: &MockWorker) -> Vec<bool> {
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "shared cold prefix"}],
    });
    let mut picks = Vec::new();
    for _ in 0..8 {
        for worker in [a, b] {
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
    picks
}

#[tokio::test]
async fn pending_prefix_keeps_a_cold_burst_on_one_worker() {
    let a = MockWorker::start(vec![]).await;
    let b = MockWorker::start(vec![]).await;
    let tree = HashTree::new();
    tree.pending().enable(Duration::from_secs(60));
    let router = radix_router(&[(&a, WorkerMode::Plain), (&b, WorkerMode::Plain)], tree);
    let picks = burst_picks(router, &a, &b).await;
    assert!(
        picks.iter().all(|&p| p == picks[0]),
        "burst scattered: {picks:?}"
    );
}

#[tokio::test]
async fn pending_prefix_keeps_a_cold_burst_on_one_worker_under_reorg() {
    let a = MockWorker::start(vec![]).await;
    let b = MockWorker::start(vec![]).await;
    let state = KvEventIndex::new();
    state.tree().pending().enable(Duration::from_secs(60));
    let router = reorg_radix_router(&[(&a, WorkerMode::Plain), (&b, WorkerMode::Plain)], &state);
    let picks = burst_picks(router, &a, &b).await;
    assert!(
        picks.iter().all(|&p| p == picks[0]),
        "burst scattered: {picks:?}"
    );
}
