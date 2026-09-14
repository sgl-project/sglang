// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_router::config::{AffinityConfig, CachePrefixProvider, PolicyKind};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{
    compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId,
};
use sgl_router::policies::prefix_provider::RadixTreePrefixProvider;
use sgl_router::policies::request_tokens_for;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};
use crate::common::mock_worker::MockWorker;

#[tokio::test]
async fn radix_tree_routes_cache_aware_request_to_cached_worker() {
    let cached = MockWorker::start(vec![]).await;
    let uncached = MockWorker::start(vec![]).await;
    let mut cfg = config();
    cfg.model.policy = PolicyKind::CacheAware;
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::RadixTree;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        cache_candidate_min_workers: 1,
        cache_candidate_ratio: 1.0,
        cache_candidate_max_workers: 1,
        ..Default::default()
    });
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "local radix cache hit"}],
    });
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
        .expect("test prompt tokenizes");
    let hashes = compute_block_hashes(&tokens.ids, 1);
    assert!(!hashes.is_empty());

    let tree = Arc::new(HashTree::new());
    tree.insert(&KvWorkerId::new(cached.url.clone(), 0), None, &hashes);
    let registry = Arc::new(WorkerRegistry::default());
    for url in [&cached.url, &uncached.url] {
        registry
            .add(WorkerSpec {
                id: WorkerId(url.clone()),
                url: url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    let policies = Arc::new(build_registry(&cfg, Arc::clone(&tree), Arc::clone(&oracle)).unwrap());
    let mut ctx = AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    );
    ctx.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(tree, Arc::clone(&oracle)));
    ctx.block_size_oracle = oracle;

    let response = build_router(Arc::new(ctx))
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
