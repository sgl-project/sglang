// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sgl_router::config::{AffinityConfig, AffinityMode, BalancedBy};
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::AdmissionLimits;
use sgl_router::policies_reorg::decode_cache_aware::DecodeCacheAwarePolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::kv_events::{
    compute_block_hashes, BlockSizeOracle, CacheNamespace, HashTree, KvWorkerId,
    RadixTreePrefixProvider, Tiers,
};
use sgl_router::state::load_monitor::engine_reported_load::{EngineReportedLoadTable, LoadStat};
use sgl_router::workers::{EngineProfile, WireProtocol, Worker};

const TOKENS: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

fn engine(id: &str, active: usize) -> Arc<Worker> {
    let engine = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}"),
        mode: Stage::Decode,
        model_ids: vec![ModelId("m".into())],
        ..Default::default()
    }));
    engine.active_requests.store(active, Ordering::Relaxed);
    engine
}

fn policy(tree: Arc<HashTree>, load: Arc<EngineReportedLoadTable>) -> DecodeCacheAwarePolicy {
    let oracle = BlockSizeOracle::new();
    oracle.try_set(2).unwrap();
    DecodeCacheAwarePolicy::new(
        RadixTreePrefixProvider::new(tree, oracle),
        load,
        AffinityConfig {
            mode: AffinityMode::Prefer,
            balanced_by: BalancedBy::RunningRequests,
            cache_affinity_min_matched_tokens: Some(1),
            ..Default::default()
        },
    )
    .unwrap()
}

fn store(tree: &HashTree, worker: &Worker, blocks: usize, tier: Tiers) {
    tree.insert_tiered(
        &KvWorkerId::new(worker.url.clone(), 0),
        None,
        &compute_block_hashes(&TOKENS, 2)[..blocks],
        tier,
    );
}

fn request(model: &ModelId) -> PickRequest<'_> {
    PickRequest {
        token_ids: Some(&TOKENS),
        ..PickRequest::new(model, Stage::Decode, 8)
    }
}

#[tokio::test]
async fn decode_prefers_longest_resident_prefix_then_device_residency() {
    let engines = [engine("host", 4), engine("device", 0)];
    let tree = Arc::new(HashTree::new());
    store(&tree, &engines[0], 4, Tiers::HOST);
    store(&tree, &engines[1], 3, Tiers::DEVICE);
    let policy = policy(tree.clone(), EngineReportedLoadTable::new());
    let model = ModelId("m".into());
    assert_eq!(
        policy
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "host"
    );
    store(&tree, &engines[1], 4, Tiers::DEVICE);
    engines[1].active_requests.store(9, Ordering::Relaxed);
    assert_eq!(
        policy
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "device"
    );
    tree.remove_tiered(
        &KvWorkerId::new(engines[1].url.clone(), 0),
        &compute_block_hashes(&TOKENS, 2),
        Tiers::DEVICE,
    );
    assert_eq!(
        policy
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "host"
    );
}

#[tokio::test]
async fn decode_ignores_disk_external_and_pending_placements() {
    let engines = [engine("cold", 0), engine("storage", 9)];
    let tree = Arc::new(HashTree::new());
    store(&tree, &engines[1], 4, Tiers::DISK.union(Tiers::EXTERNAL));
    tree.pending().enable(Duration::from_secs(30));
    tree.pending()
        .record(&engines[1].url, &compute_block_hashes(&TOKENS, 2));
    let policy = policy(tree.clone(), EngineReportedLoadTable::new());
    let model = ModelId("m".into());
    let pick = policy.pick(&engines, &request(&model)).await.unwrap();
    assert_eq!(pick.engine.id.0, "cold");
    assert_eq!(pick.reason, "no_decode_cache_candidate");
    // An interior nonresident block stops the prefix even when later blocks reside on GPU.
    store(&tree, &engines[1], 4, Tiers::DEVICE);
    let hashes = compute_block_hashes(&TOKENS, 2);
    tree.remove_tiered(
        &KvWorkerId::new(engines[1].url.clone(), 0),
        &hashes[1..2],
        Tiers::DEVICE,
    );
    let depths = tree.resident_prefix_depths(&hashes);
    assert_eq!(depths[&KvWorkerId::new(engines[1].url.clone(), 0)], (1, 1));
}

#[tokio::test]
async fn decode_cache_hit_cannot_bypass_admission_or_candidate_membership() {
    let engines = [engine("owner", 2), engine("cold", 0), engine("outside", 0)];
    let tree = Arc::new(HashTree::new());
    store(&tree, &engines[0], 3, Tiers::DEVICE);
    store(&tree, &engines[2], 4, Tiers::DEVICE);
    let mut policy = policy(tree, EngineReportedLoadTable::new());
    policy.admission = Arc::new(AdmissionLimits {
        max_inflight_requests: Some(2),
        ..Default::default()
    });
    let model = ModelId("m".into());
    assert_eq!(
        policy
            .pick(&engines[..2], &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "cold"
    );
    engines[1].active_requests.store(2, Ordering::Relaxed);
    assert!(matches!(
        policy.pick(&engines[..2], &request(&model)).await,
        Err(PickError::NoAdmissibleEngine(_))
    ));
}

#[tokio::test]
async fn decode_ties_use_load_then_stable_identity() {
    let engines = [engine("b", 2), engine("a", 0)];
    let tree = Arc::new(HashTree::new());
    for e in &engines {
        store(&tree, e, 4, Tiers::HOST);
    }
    let policy = policy(tree, EngineReportedLoadTable::new());
    let model = ModelId("m".into());
    assert_eq!(
        policy
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "a"
    );
    engines[0].active_requests.store(0, Ordering::Relaxed);
    assert_eq!(
        policy
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "a"
    );
}

#[tokio::test]
async fn balanced_decode_uses_fresh_running_load() {
    let engines = [engine("owner", 0), engine("cold", 0)];
    let tree = Arc::new(HashTree::new());
    store(&tree, &engines[0], 4, Tiers::DEVICE);
    let model = ModelId("m".into());
    for (stale, expected) in [(false, "cold"), (true, "owner")] {
        let load = EngineReportedLoadTable::new();
        for (e, running) in [(&engines[0], 20), (&engines[1], 1)] {
            load.set(
                &e.url,
                0,
                LoadStat {
                    num_running_reqs: running,
                    num_waiting_reqs: 0,
                    num_tokens: 8,
                    max_total_num_tokens: 100,
                    native_cache: None,
                },
                Instant::now()
                    - if stale {
                        Duration::from_secs(3600)
                    } else {
                        Duration::ZERO
                    },
            );
        }
        let oracle = BlockSizeOracle::new();
        oracle.try_set(2).unwrap();
        let policy = DecodeCacheAwarePolicy::new(
            RadixTreePrefixProvider::new(tree.clone(), oracle),
            load,
            AffinityConfig {
                mode: AffinityMode::Balanced,
                balanced_by: BalancedBy::RunningRequests,
                cache_affinity_min_matched_tokens: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            policy
                .pick(&engines, &request(&model))
                .await
                .unwrap()
                .engine
                .id
                .0,
            expected
        );
    }
}

#[tokio::test]
async fn multiple_dp_ranks_do_not_turn_a_url_hit_into_decode_affinity() {
    let dp = Arc::new(Worker::with_cb_config(
        WorkerSpec {
            id: WorkerId("dp".into()),
            url: "http://dp".into(),
            mode: Stage::Decode,
            model_ids: vec![ModelId("m".into())],
            ..Default::default()
        },
        None,
        EngineProfile {
            dp_ranks: 2,
            ..WireProtocol::default().into()
        },
    ));
    dp.active_requests.store(9, Ordering::Relaxed);
    let engines = [dp, engine("cold", 0)];
    let tree = Arc::new(HashTree::new());
    store(&tree, &engines[0], 4, Tiers::DEVICE);
    let policy = policy(tree, EngineReportedLoadTable::new());
    let model = ModelId("m".into());
    assert_eq!(
        policy
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "cold"
    );
}

#[tokio::test]
async fn decode_thresholds_count_only_matched_tokens_in_a_partial_prompt() {
    let engines = [engine("owner", 9), engine("cold", 0)];
    let tree = Arc::new(HashTree::new());
    store(&tree, &engines[0], 3, Tiers::DEVICE);
    let model = ModelId("m".into());
    // Seven tokens with two-token pages: three cached pages cover six tokens.
    let request = PickRequest {
        token_ids: Some(&TOKENS[..7]),
        ..PickRequest::new(&model, Stage::Decode, 7)
    };
    for (minimum, ratio, expected) in [(6, 0.8, "owner"), (7, 0.0, "cold"), (1, 0.9, "cold")] {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(2).unwrap();
        let policy = DecodeCacheAwarePolicy::new(
            RadixTreePrefixProvider::new(tree.clone(), oracle),
            EngineReportedLoadTable::new(),
            AffinityConfig {
                mode: AffinityMode::Prefer,
                cache_affinity_min_matched_tokens: Some(minimum),
                cache_affinity_min_match_ratio: Some(ratio),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            policy.pick(&engines, &request).await.unwrap().engine.id.0,
            expected
        );
    }
}

#[tokio::test]
async fn decode_missing_signals_and_other_namespaces_fall_back_to_load() {
    let engines = [engine("owner", 9), engine("cold", 0)];
    let tree = Arc::new(HashTree::new());
    store(&tree, &engines[0], 4, Tiers::DEVICE);
    let model = ModelId("m".into());
    let unknown = DecodeCacheAwarePolicy::new(
        RadixTreePrefixProvider::new(tree.clone(), BlockSizeOracle::new()),
        EngineReportedLoadTable::new(),
        AffinityConfig {
            mode: AffinityMode::Prefer,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        unknown
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "cold"
    );
    let policy = policy(tree, EngineReportedLoadTable::new());
    let salt = CacheNamespace {
        cache_salt: Some("another-session".into()),
        ..Default::default()
    };
    let lora = CacheNamespace {
        lora_name: Some("another-adapter".into()),
        ..Default::default()
    };
    for req in [
        PickRequest::new(&model, Stage::Decode, 8),
        PickRequest {
            cache_namespace: &salt,
            ..request(&model)
        },
        PickRequest {
            cache_namespace: &lora,
            ..request(&model)
        },
    ] {
        let pick = policy.pick(&engines, &req).await.unwrap();
        assert_eq!(pick.engine.id.0, "cold");
        assert_eq!(pick.reason, "no_decode_cache_candidate");
    }
}
