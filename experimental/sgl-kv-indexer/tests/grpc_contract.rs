// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! gRPC contract tests: exercise all three RPCs of the `KVIndexer` service
//! over the wire (real tonic server + client), not just the backend trait.
//!
//! Like the backend integration tests these require a live store and are
//! opt-in via `KV_INDEXER_REDIS_URL` (or `KV_INDEXER_REDIS_CLUSTER_NODES`);
//! when neither is set every test skips. Each test uses a unique namespace and
//! unique worker/hash ids so a shared store never causes collisions.
#![cfg(feature = "redis-backend")]

#[path = "common/require.rs"]
mod require;
#[path = "common/id.rs"]
mod test_id;
#[path = "common/kv.rs"]
mod test_kv;
#[path = "common/net.rs"]
mod test_net;

use std::time::Duration;

use tonic::transport::Server;
use tonic::Code;

use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType,
    GetExternalKvHitCountsRequest, MatchExternalKvPrefixRequest, MatchExternalKvRequest,
};
use sgl_kv_indexer::{KvIndexerService, RedisKvIndexerBackend};
use test_id::nanos;
use test_kv::{action, apply_request, dram, hbm};
use test_net::free_addr;

async fn start_backend(
    backend: RedisKvIndexerBackend,
) -> KvIndexerClient<tonic::transport::Channel> {
    let svc = KvIndexerServer::new(KvIndexerService::new(backend));
    let addr = free_addr();
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve(addr)
            .await
            .expect("server serve");
    });

    let endpoint = format!("http://{addr}");
    for _ in 0..50 {
        if let Ok(c) = KvIndexerClient::connect(endpoint.clone()).await {
            return c;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("client failed to connect to {endpoint}");
}

/// Starts a real gRPC server backed by Redis on a unique namespace and returns
/// a connected client, or `None` (skip) when no store env is configured.
async fn start(test: &str) -> Option<KvIndexerClient<tonic::transport::Channel>> {
    let url = match std::env::var("KV_INDEXER_REDIS_URL") {
        Ok(u) => u,
        Err(_) => {
            require::skip(test, "KV_INDEXER_REDIS_URL is not set");
            return None;
        }
    };
    let ns = format!("grpc:{test}:{}", nanos());
    let backend = RedisKvIndexerBackend::connect_single(&url, ns)
        .await
        .expect("connect redis");
    Some(start_backend(backend).await)
}

fn apply(
    worker: &str,
    addr: &str,
    seq: u64,
    action_type: ExternalKvActionType,
    tier: i32,
    hashes: &[&str],
) -> ApplyExternalKvBatchRequest {
    apply_request(worker, addr, seq, vec![action(action_type, tier, hashes)])
}

fn apply_report(
    worker: &str,
    addr: &str,
    seq: u64,
    tier: i32,
    hashes: &[&str],
) -> ApplyExternalKvBatchRequest {
    apply(
        worker,
        addr,
        seq,
        ExternalKvActionType::ActionReport,
        tier,
        hashes,
    )
}

#[tokio::test]
async fn disjoint_workers_scale_across_two_indexer_servers() {
    let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") else {
        require::skip(
            "disjoint_workers_scale_across_two_indexer_servers",
            "KV_INDEXER_REDIS_URL is not set",
        );
        return;
    };
    let namespace = format!("grpc:horizontal:{}", nanos());
    let backend_0 = RedisKvIndexerBackend::connect_single(&url, namespace.clone())
        .await
        .expect("connect indexer-0 backend");
    let backend_1 = RedisKvIndexerBackend::connect_single(&url, namespace)
        .await
        .expect("connect indexer-1 backend");
    let mut indexer_0 = start_backend(backend_0).await;
    let mut indexer_1 = start_backend(backend_1).await;

    let suffix = nanos();
    let worker_0 = format!("worker-0-{suffix}");
    let worker_1 = format!("worker-1-{suffix}");
    let hash_0 = format!("horizontal-h0-{suffix}");
    let hash_1 = format!("horizontal-h1-{suffix}");
    let shared_hash = format!("horizontal-shared-{suffix}");

    let apply_0 = indexer_0.apply_external_kv_batch(apply_report(
        &worker_0,
        "10.0.0.1:9000",
        1,
        hbm(),
        &[&hash_0, &shared_hash],
    ));
    let apply_1 = indexer_1.apply_external_kv_batch(apply_report(
        &worker_1,
        "10.0.0.2:9000",
        1,
        hbm(),
        &[&hash_1, &shared_hash],
    ));
    let (result_0, result_1) = tokio::join!(apply_0, apply_1);
    result_0.expect("indexer-0 applies worker-0");
    result_1.expect("indexer-1 applies worker-1");

    let request = || MatchExternalKvRequest {
        hashes: vec![hash_0.clone(), hash_1.clone(), shared_hash.clone()],
        count_as_hit: false,
    };
    let from_0 = indexer_0
        .match_external_kv(request())
        .await
        .expect("query indexer-0")
        .into_inner();
    let from_1 = indexer_1
        .match_external_kv(request())
        .await
        .expect("query indexer-1")
        .into_inner();

    for response in [&from_0, &from_1] {
        assert!(
            response
                .matches
                .iter()
                .any(|entry| entry.worker_id == worker_0),
            "either indexer must see worker-0 through shared Redis"
        );
        assert!(
            response
                .matches
                .iter()
                .any(|entry| entry.worker_id == worker_1),
            "either indexer must see worker-1 through shared Redis"
        );
    }
}

#[tokio::test]
async fn apply_match_and_hit_counts_over_grpc() {
    let Some(mut c) = start("apply_match").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let (h1, h2, miss) = ("am-h1", "am-h2", "am-miss");

    c.apply_external_kv_batch(apply_report(&w, "10.0.0.1:9000", 1, hbm(), &[h1, h2]))
        .await
        .expect("apply ok");

    let resp = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h1.into(), h2.into(), miss.into()],
            count_as_hit: true,
        })
        .await
        .expect("match ok")
        .into_inner();

    let m = resp
        .matches
        .iter()
        .find(|m| m.worker_id == w)
        .expect("worker present in matches");
    assert_eq!(m.address, "10.0.0.1:9000");
    let tier = m
        .hashes_by_tier
        .iter()
        .find(|t| t.tier == hbm())
        .expect("HBM tier present");
    let mut got: Vec<&String> = tier.hashes.iter().collect();
    got.sort();
    assert_eq!(got, vec![&h1.to_string(), &h2.to_string()]);

    let hc = c
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: vec![h1.into(), h2.into(), miss.into()],
        })
        .await
        .expect("hit counts ok")
        .into_inner();
    let count = |h: &str| {
        hc.entries
            .iter()
            .find(|e| e.hash == h)
            .map(|e| e.hit_count_total)
            .unwrap_or(0)
    };
    assert!(count(h1) >= 1, "h1 should have a hit");
    assert!(count(h2) >= 1, "h2 should have a hit");
    assert_eq!(count(miss), 0, "unmatched hash must not be counted");
}

#[tokio::test]
async fn diagnostic_match_does_not_count_hits_over_grpc() {
    let Some(mut c) = start("diag_match").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let h = "diag-h1";
    c.apply_external_kv_batch(apply_report(&w, "10.0.0.2:9000", 1, hbm(), &[h]))
        .await
        .expect("apply ok");

    // count_as_hit=false must not bump counters
    c.match_external_kv(MatchExternalKvRequest {
        hashes: vec![h.into()],
        count_as_hit: false,
    })
    .await
    .expect("match ok");

    let hc = c
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: vec![h.into()],
        })
        .await
        .expect("hit counts ok")
        .into_inner();
    let count = hc
        .entries
        .iter()
        .find(|e| e.hash == h)
        .map(|e| e.hit_count_total)
        .unwrap_or(0);
    assert_eq!(count, 0, "diagnostic match must not increase hit count");
}

#[tokio::test]
async fn apply_report_then_revoke_over_grpc() {
    let Some(mut c) = start("apply_rr").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let h = "apply-rr-h1";

    c.apply_external_kv_batch(apply_report(&w, "10.0.0.3:9000", 1, hbm(), &[h]))
        .await
        .expect("report apply ok");

    let before = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h.into()],
            count_as_hit: false,
        })
        .await
        .expect("match ok")
        .into_inner();
    assert!(
        before.matches.iter().any(|m| m.worker_id == w),
        "reported hash should match"
    );

    c.apply_external_kv_batch(apply(
        &w,
        "10.0.0.3:9000",
        2,
        ExternalKvActionType::ActionRevoke,
        hbm(),
        &[h],
    ))
    .await
    .expect("revoke apply ok");

    let after = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h.into()],
            count_as_hit: false,
        })
        .await
        .expect("match ok")
        .into_inner();
    assert!(
        !after.matches.iter().any(|m| m.worker_id == w),
        "revoked hash must not match"
    );
}

#[tokio::test]
async fn revoke_all_at_tier_over_grpc() {
    let Some(mut c) = start("revoke_all").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let h = "ra-h1";

    // same hash present in both HBM and DRAM
    c.apply_external_kv_batch(apply_report(&w, "10.0.0.3:9000", 1, hbm(), &[h]))
        .await
        .expect("apply hbm");
    c.apply_external_kv_batch(apply_report(&w, "10.0.0.3:9000", 2, dram(), &[h]))
        .await
        .expect("apply dram");

    c.apply_external_kv_batch(apply(
        &w,
        "10.0.0.3:9000",
        3,
        ExternalKvActionType::ActionClearAllAtTier,
        hbm(),
        &[],
    ))
    .await
    .expect("clear-all apply ok");

    let resp = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h.into()],
            count_as_hit: false,
        })
        .await
        .expect("match ok")
        .into_inner();
    let m = resp
        .matches
        .iter()
        .find(|m| m.worker_id == w)
        .expect("worker still present via DRAM");
    let tiers: Vec<i32> = m.hashes_by_tier.iter().map(|t| t.tier).collect();
    assert!(tiers.contains(&dram()), "DRAM tier must remain");
    assert!(!tiers.contains(&hbm()), "HBM tier must be cleared");
}

#[tokio::test]
async fn match_miss_returns_empty_over_grpc() {
    let Some(mut c) = start("match_miss").await else {
        return;
    };
    let resp = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![format!("never-reported-{}", nanos())],
            count_as_hit: true,
        })
        .await
        .expect("match ok")
        .into_inner();
    assert!(resp.matches.is_empty(), "unknown hash yields no matches");
}

#[tokio::test]
async fn validation_errors_map_to_invalid_argument_over_grpc() {
    let Some(mut c) = start("validation").await else {
        return;
    };

    // empty worker_id
    let err = c
        .apply_external_kv_batch(apply_report("", "addr", 1, hbm(), &["h"]))
        .await
        .expect_err("empty worker_id must be rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "empty worker_id -> InvalidArgument"
    );

    // REPORT action with no hashes
    let err = c
        .apply_external_kv_batch(apply(
            "w",
            "addr",
            1,
            ExternalKvActionType::ActionReport,
            hbm(),
            &[],
        ))
        .await
        .expect_err("empty hashes must be rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "empty hashes -> InvalidArgument"
    );

    // unknown action type
    let bad = ApplyExternalKvBatchRequest {
        worker_id: "w".into(),
        seq: 1,
        worker_address: String::new(),
        cache_spec: None,
        actions: vec![ExternalKvAction {
            r#type: 999,
            tier: hbm(),
            hashes: vec!["h".into()],
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
        }],
    };
    let err = c
        .apply_external_kv_batch(bad)
        .await
        .expect_err("unknown action type rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "unknown action type -> InvalidArgument"
    );

    // invalid tier in an ApplyBatch action
    let err = c
        .apply_external_kv_batch(apply(
            "w",
            "addr",
            1,
            ExternalKvActionType::ActionReport,
            999,
            &["h"],
        ))
        .await
        .expect_err("bad tier rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "bad tier -> InvalidArgument"
    );
}

#[tokio::test]
async fn match_prefix_over_grpc() {
    let Some(mut c) = start("match_prefix").await else {
        return;
    };
    let (w_long, w_short) = (format!("long-{}", nanos()), format!("short-{}", nanos()));
    let (a, b, d) = ("mp-a", "mp-b", "mp-c");

    c.apply_external_kv_batch(apply_report(&w_long, "10.0.0.1:9000", 1, hbm(), &[a, b, d]))
        .await
        .expect("apply long");
    c.apply_external_kv_batch(apply_report(&w_short, "10.0.0.2:9000", 1, hbm(), &[a]))
        .await
        .expect("apply short");

    let resp = c
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![a.into(), b.into(), d.into()],
            max_blocks: 0,
        })
        .await
        .expect("prefix ok")
        .into_inner();

    assert_eq!(resp.best_prefix_blocks, 3);
    assert_eq!(resp.blocks_read, 3);
    // Descending by prefix length: long (3) before short (1).
    assert_eq!(resp.matches.len(), 2);
    assert_eq!(resp.matches[0].worker_id, w_long);
    assert_eq!(resp.matches[0].matched_prefix_blocks, 3);
    assert_eq!(resp.matches[0].worker_address, "10.0.0.1:9000");
    assert_eq!(resp.matches[1].worker_id, w_short);
    assert_eq!(resp.matches[1].matched_prefix_blocks, 1);
}
