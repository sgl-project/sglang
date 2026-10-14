// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Salt namespaces must survive every request preparation and cache lookup path.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType, TierType,
};
use sgl_kv_indexer::{
    server_builder, GrpcPrefixIndex, InMemoryKvIndexerBackend, KvIndexerService, PrefixIndexConfig,
};
use sgl_router::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
use sgl_router::discovery::{ModelId, WorkerMode};
use sgl_router::policies::request_tokens_for;
use sgl_router::policies_reorg::cache_aware::{CacheAwarePolicy, CacheSource};
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::ChatRouting;
use sgl_router::state::kv_events::{
    compute_block_hashes_bigram_with_salt, compute_block_hashes_with_salt, HashTree, KvWorkerId,
};
use sgl_router::tokenizer::TokenizerRegistry;
use tokio_stream::wrappers::TcpListenerStream;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, radix_context, MODEL};
use crate::common::mock_worker::MockWorker;

async fn check_namespaces(remote: bool, reorg: bool, bigram: bool) {
    let workers = [
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    ];
    let salts = [None, Some("tenant-a"), Some("租户-B")];
    let tokenizers = TokenizerRegistry::load_from_config(&config()).unwrap();
    let chat = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello world"}]});
    let rendered = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &chat)
        .unwrap()
        .ids;
    let text_ids = tokenizers.encode_prompt(MODEL, "hello world").unwrap();
    let prompts = [
        (
            "/v1/chat/completions",
            json!({"model": MODEL, "messages": [], "input_ids": [1,2,3,4]}),
            vec![1, 2, 3, 4],
        ),
        ("/v1/chat/completions", chat, rendered),
        (
            "/generate",
            json!({"model": MODEL, "input_ids": [1,2,3,4]}),
            vec![1, 2, 3, 4],
        ),
        (
            "/generate",
            json!({"model": MODEL, "text": "hello world"}),
            text_ids,
        ),
        (
            "/v1/embeddings",
            json!({"model": MODEL, "input": [1, 2, 3, 4]}),
            vec![1, 2, 3, 4],
        ),
    ];
    let hash = if bigram {
        compute_block_hashes_bigram_with_salt
    } else {
        compute_block_hashes_with_salt
    };
    let tree = HashTree::new();
    for (worker, salt) in workers.iter().zip(salts) {
        for (_, _, ids) in &prompts {
            let hashes = hash(ids, 1, salt);
            assert!(!hashes.is_empty());
            tree.insert(&KvWorkerId::new(worker.url.clone(), 0), None, &hashes);
        }
    }
    let specs: Vec<_> = workers
        .iter()
        .map(|worker| (worker, WorkerMode::Plain))
        .collect();
    let mut ctx = radix_context(&specs, tree);
    ctx.block_size_oracle.set_bigram(bigram);
    let mut server = None;
    let source = if remote {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        server = Some(tokio::spawn(async move {
            server_builder()
                .add_service(KvIndexerService::new(InMemoryKvIndexerBackend::new()).into_server())
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        }));
        let mut indexer = KvIndexerClient::connect(endpoint.clone()).await.unwrap();
        for (worker, salt) in workers.iter().zip(salts) {
            indexer
                .apply_external_kv_batch(ApplyExternalKvBatchRequest {
                    worker_id: worker.url.clone(),
                    worker_address: worker.url.clone(),
                    seq: 1,
                    actions: prompts
                        .iter()
                        .map(|(_, _, ids)| ExternalKvAction {
                            r#type: ExternalKvActionType::ActionReport as i32,
                            tier: TierType::TierHbm as i32,
                            hashes: hash(ids, 1, salt),
                            component_masks: vec![],
                            block_sizes: vec![],
                            parent_block_hash: None,
                        })
                        .collect(),
                    cache_spec: None,
                })
                .await
                .unwrap();
        }
        let index = Arc::new(
            GrpcPrefixIndex::new(PrefixIndexConfig {
                endpoint,
                query_deadline: Duration::from_secs(1),
                max_inflight: 4,
            })
            .unwrap(),
        );
        ctx.prefix_index = Some(index.clone());
        // Ensure these cases cannot silently succeed via the local provider.
        ctx.radix_tree_prefix_provider = None;
        CacheSource::Remote {
            index,
            block_size: ctx.block_size_oracle.clone(),
        }
    } else {
        CacheSource::Local(ctx.radix_tree_prefix_provider.clone().unwrap())
    };
    if reorg {
        let policy = CacheAwarePolicy::new(
            Arc::new(source),
            ctx.engine_reported_load.clone(),
            ctx.config.model.affinity.clone().unwrap(),
        )
        .unwrap();
        ctx.chat_routing = ChatRouting::Reorg(
            [(
                ModelId(MODEL.into()),
                BucketResolver::new(vec![Bucket::new(
                    "plain",
                    BucketGroups::Plain(EngineGroup::new(Arc::new(policy))),
                )])
                .unwrap(),
            )]
            .into(),
        );
    }
    let app = build_router(Arc::new(ctx));
    for (path, prompt, _) in prompts {
        for (salt, namespace) in [
            (None, None),
            (Some(Value::Null), None),
            (Some(json!("")), None),
            (Some(json!("tenant-a")), Some("tenant-a")),
            (Some(json!("租户-B")), Some("租户-B")),
            (Some(json!([""])), None),
            (Some(json!(["tenant-a"])), Some("tenant-a")),
            (Some(json!(["租户-B"])), Some("租户-B")),
        ] {
            let mut body = prompt.clone();
            if let Some(salt) = &salt {
                body["cache_salt"] = salt.clone();
            }
            // A single prompt accepts a one-element salt array when n > 1.
            // The engine replicates that namespace across the samples.
            if salt.as_ref().is_some_and(Value::is_array) {
                if path == "/generate" {
                    body["sampling_params"] = json!({"n": 2});
                } else if path == "/v1/chat/completions" {
                    body["n"] = json!(2);
                }
            }
            // The engine's embedding input does not consume cache_salt.
            let namespace = if path == "/v1/embeddings" {
                None
            } else {
                namespace
            };
            let expected = salts.iter().position(|s| *s == namespace).unwrap();
            for worker in &workers {
                worker.captured.lock().unwrap().last_body = None;
            }
            let response = app
                .clone()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert!(
                response.status().is_success(),
                "{path}: {}",
                response.status()
            );
            for (i, worker) in workers.iter().enumerate() {
                assert_eq!(worker.captured.lock().unwrap().last_body.is_some(), i == expected,
                    "remote={remote} reorg={reorg} bigram={bigram} path={path} salt={salt:?} worker={i}");
            }
            let forwarded = workers[expected].captured_json().await;
            assert_eq!(forwarded.get("cache_salt"), body.get("cache_salt"));
        }
    }
    if let Some(server) = server {
        server.abort();
    }
}

/// A valid array salt must not replace the known prompt length with a body-size
/// estimate, nor bypass the bucket's per-sequence context limit.
#[tokio::test]
async fn parallel_sampling_array_salt_preserves_bucket_length_checks() {
    let worker = MockWorker::start(vec![]).await;
    let mut ctx = radix_context(&[(&worker, WorkerMode::Plain)], HashTree::new());
    let source = CacheSource::Local(ctx.radix_tree_prefix_provider.clone().unwrap());
    let policy = CacheAwarePolicy::new(
        Arc::new(source),
        ctx.engine_reported_load.clone(),
        ctx.config.model.affinity.clone().unwrap(),
    )
    .unwrap();
    let mut bucket = Bucket::new(
        "short",
        BucketGroups::Plain(EngineGroup::new(Arc::new(policy))),
    );
    bucket.limits.max = Some(8);
    bucket.max_context_tokens = Some(8);
    ctx.chat_routing = ChatRouting::Reorg(
        [(
            ModelId(MODEL.into()),
            BucketResolver::new(vec![bucket]).unwrap(),
        )]
        .into(),
    );
    let app = build_router(Arc::new(ctx));
    for path in ["/v1/chat/completions", "/generate"] {
        for salt in [json!("tenant-a"), json!(["tenant-a"])] {
            for (length, expected) in [(7, StatusCode::OK), (8, StatusCode::BAD_REQUEST)] {
                let ids = vec![1_u32; length];
                let mut body = json!({"model": MODEL, "input_ids": ids, "cache_salt": salt});
                if path == "/generate" {
                    body["sampling_params"] = json!({"n": 2, "max_new_tokens": 1});
                } else {
                    body["messages"] = json!([]);
                    body["n"] = json!(2);
                    body["max_tokens"] = json!(1);
                }
                worker.captured.lock().unwrap().last_body = None;
                let response = app
                    .clone()
                    .oneshot(
                        Request::post(path)
                            .header("content-type", "application/json")
                            .body(Body::from(body.to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    expected,
                    "{path} salt={salt} length={length}"
                );
                if expected == StatusCode::OK {
                    let forwarded = worker.captured_json().await;
                    assert_eq!(forwarded["input_ids"], json!(ids));
                    assert_eq!(forwarded["cache_salt"], salt);
                } else {
                    assert!(worker.captured.lock().unwrap().last_body.is_none());
                }
            }
        }
    }
}

#[tokio::test]
async fn cache_salt_local_legacy() {
    for bigram in [false, true] {
        check_namespaces(false, false, bigram).await;
    }
}

#[tokio::test]
async fn cache_salt_local_reorg() {
    for bigram in [false, true] {
        check_namespaces(false, true, bigram).await;
    }
}

#[tokio::test]
async fn cache_salt_remote_legacy() {
    for bigram in [false, true] {
        check_namespaces(true, false, bigram).await;
    }
}

#[tokio::test]
async fn cache_salt_remote_reorg() {
    for bigram in [false, true] {
        check_namespaces(true, true, bigram).await;
    }
}
