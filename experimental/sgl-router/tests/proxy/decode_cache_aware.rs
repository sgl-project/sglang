// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;

use sgl_router::config::{AffinityConfig, AffinityMode, CachePrefixProvider};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::PolicyRegistry;
use sgl_router::policies_reorg::factory::build_resolver;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::kv_events::{compute_block_hashes, KvEventIndex, KvWorkerId, Tiers};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;

use crate::common::cache_aware_fixture::{config, MODEL};
use crate::common::mock_worker::MockWorker;

fn configured() -> sgl_router::config::Config {
    let mut cfg = config();
    cfg.model.affinity = Some(AffinityConfig {
        mode: AffinityMode::Prefer,
        cache_affinity_min_matched_tokens: Some(1),
        ..Default::default()
    });
    cfg.model.reorg_buckets = Some(
        serde_json::from_value(json!({"buckets":[{
            "id":"pd", "prefill":{"policy":"cache_aware"}, "decode":{"policy":"cache_aware"}
        }]}))
        .unwrap(),
    );
    cfg
}

#[tokio::test]
async fn pd_generate_selects_a_resident_decode_in_the_prefills_version_group() {
    let p = MockWorker::start(vec![]).await;
    let cold = MockWorker::start(vec![]).await;
    let warm = MockWorker::start(vec![]).await;
    let wrong_version = MockWorker::start(vec![]).await;
    let workers = Arc::new(WorkerRegistry::default());
    for (worker, mode, version) in [
        (&p, WorkerMode::Prefill, "v1"),
        (&cold, WorkerMode::Decode, "v1"),
        (&warm, WorkerMode::Decode, "v1"),
        (&wrong_version, WorkerMode::Decode, "v2"),
    ] {
        workers
            .add(WorkerSpec {
                id: WorkerId(worker.url.clone()),
                url: worker.url.clone(),
                mode,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
                version_group: Some(version.into()),
                ..Default::default()
            })
            .unwrap();
    }
    let state = KvEventIndex::new();
    state.block_size_oracle().try_set(2).unwrap();
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let hashes = compute_block_hashes(&tokens, 2);
    state.tree().insert_tiered(
        &KvWorkerId::new(warm.url.clone(), 0),
        None,
        &hashes[..3],
        Tiers::HOST,
    );
    state.tree().insert_tiered(
        &KvWorkerId::new(wrong_version.url.clone(), 0),
        None,
        &hashes,
        Tiers::DEVICE,
    );
    let cfg = configured();
    let (resolver, _) = build_resolver(&cfg.model, &state, None).unwrap();
    let mut ctx = AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        workers,
        Arc::new(PolicyRegistry::default()),
    );
    ctx.chat_routing = ChatRouting::Reorg([(ModelId(MODEL.into()), resolver)].into());
    ctx.block_size_oracle = state.block_size_oracle();
    let app = build_router(Arc::new(ctx));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"input_ids":tokens,"sampling_params":{"max_new_tokens":1}}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let decode_body = warm.captured_json().await;
    let prefill_body = p.captured_json().await;
    assert_eq!(
        decode_body["bootstrap_room"],
        prefill_body["bootstrap_room"]
    );
    assert_eq!(decode_body["input_ids"], json!(tokens));
    assert!(cold.captured.lock().unwrap().last_body.is_none());
    assert!(wrong_version.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn decode_cache_group_rejects_tier_blind_remote_index_and_prefill_balancing() {
    let state = KvEventIndex::new();
    let mut cfg = configured();
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::Indexer;
    assert!(build_resolver(&cfg.model, &state, None)
        .err()
        .unwrap()
        .to_string()
        .contains("storage tiers"));
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::RadixTree;
    cfg.model.reorg_buckets = Some(
        serde_json::from_value(json!({"buckets":[{
            "id":"pd", "prefill":{}, "decode":{"policy":"cache_aware", "affinity":{
                "mode":"balanced", "balanced_by":"prefill_tokens"
            }}
        }]}))
        .unwrap(),
    );
    assert!(build_resolver(&cfg.model, &state, None)
        .err()
        .unwrap()
        .to_string()
        .contains("running_requests"));
}
