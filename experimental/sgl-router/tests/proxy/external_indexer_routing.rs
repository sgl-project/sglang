// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Full HTTP routing path backed by a real in-memory Indexer gRPC server.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::{
    ConfigureExpectedWorkersRequest, ExpectedWorker, ReplaceExternalKvSnapshotRequest, TierHashes,
    TierType,
};
use sgl_kv_indexer::{
    server_builder, GrpcPrefixIndex, InMemoryKvIndexerBackend, KvIndexerService, PrefixIndexConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree};
use sgl_router::policies::request_tokens_for;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use tokio_stream::wrappers::TcpListenerStream;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};
use crate::common::mock_worker::MockWorker;

#[tokio::test]
async fn external_indexer_routes_to_the_cached_worker() {
    let cached = MockWorker::start(vec![]).await;
    let uncached = MockWorker::start(vec![]).await;
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello there friend"}],
    });
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
        .expect("test prompt tokenizes");
    let hashes = compute_block_hashes(&tokens.ids, 1);
    assert!(!hashes.is_empty());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        server_builder()
            .add_service(KvIndexerService::new(InMemoryKvIndexerBackend::new()).into_server())
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    let mut indexer = KvIndexerClient::connect(endpoint.clone()).await.unwrap();
    indexer
        .configure_expected_workers(ConfigureExpectedWorkersRequest {
            workers: vec![
                ExpectedWorker {
                    worker_id: "cached-worker".into(),
                    worker_address: cached.url.clone(),
                    ..Default::default()
                },
                ExpectedWorker {
                    worker_id: "uncached-worker".into(),
                    worker_address: uncached.url.clone(),
                    ..Default::default()
                },
            ],
        })
        .await
        .unwrap();
    for (worker_id, worker_address, worker_hashes, worker_generation) in [
        (
            "cached-worker",
            cached.url.clone(),
            hashes.clone(),
            "cached-generation",
        ),
        (
            "uncached-worker",
            uncached.url.clone(),
            Vec::new(),
            "uncached-generation",
        ),
    ] {
        indexer
            .replace_external_kv_snapshot(ReplaceExternalKvSnapshotRequest {
                worker_id: worker_id.into(),
                worker_address,
                worker_epoch: "epoch".into(),
                applied_seq: 0,
                hashes_by_tier: (!worker_hashes.is_empty())
                    .then_some(TierHashes {
                        tier: TierType::TierHbm as i32,
                        hashes: worker_hashes,
                        component_masks: Vec::new(),
                        block_sizes: Vec::new(),
                    })
                    .into_iter()
                    .collect(),
                cache_spec: None,
                stream_id: None,
                worker_generation: worker_generation.into(),
            })
            .await
            .unwrap();
    }

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
    let policies = Arc::new(
        build_registry(
            &cfg,
            Arc::new(HashTree::new()),
            Arc::clone(&tokenizers),
            Arc::clone(&oracle),
        )
        .unwrap(),
    );
    let mut ctx = AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    );
    ctx.prefix_provider = Some(Arc::new(
        GrpcPrefixIndex::new(PrefixIndexConfig {
            endpoint,
            query_deadline: Duration::from_secs(1),
            max_inflight: 4,
        })
        .unwrap(),
    ));
    ctx.block_size_oracle = oracle;
    ctx.worker_loads
        .update(&cached.url, "cached-generation", 1, 0.1);
    ctx.worker_loads
        .update(&uncached.url, "uncached-generation", 0, 0.0);

    let app = build_router(Arc::new(ctx));
    let response = app
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

    server.abort();
}
