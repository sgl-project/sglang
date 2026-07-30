// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the Redis backend.
//!
//! These require a live store and are opt-in via environment:
//!   * `KV_INDEXER_REDIS_URL`            → single Redis/Dragonfly, or
//!   * `KV_INDEXER_REDIS_CLUSTER_NODES`  → Redis Cluster (comma-separated seeds)
//!
//! When neither is set every test skips (prints a note and returns), so the
//! default `cargo test --features redis-backend` run stays green without a store.
//!
//! Each test uses a unique namespace so a shared store never causes collisions.
#![cfg(feature = "redis-backend")]

#[path = "common/require.rs"]
mod require;
#[path = "common/id.rs"]
mod test_id;
#[path = "common/kv.rs"]
mod test_kv;

use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse,
};
use sgl_kv_indexer::{KvIndexerBackend, RedisKvIndexerBackend};
use test_id::nanos;
use test_kv::{action, apply_request as apply_req, dram, hashes, hbm};
use tonic::Status;

/// Builds a backend against the configured store with a unique namespace, or
/// returns `None` (skip) when no store env is set.
async fn backend(test: &str) -> Option<RedisKvIndexerBackend> {
    let ns = format!("itest:{test}:{}", nanos());
    if let Ok(nodes) = std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES") {
        let nodes: Vec<String> = nodes
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        Some(
            RedisKvIndexerBackend::connect_cluster(nodes, ns)
                .await
                .expect("connect cluster"),
        )
    } else if let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") {
        Some(
            RedisKvIndexerBackend::connect_single(&url, ns)
                .await
                .expect("connect single"),
        )
    } else {
        require::skip(
            test,
            "neither KV_INDEXER_REDIS_URL nor KV_INDEXER_REDIS_CLUSTER_NODES is set",
        );
        None
    }
}

fn match_req(hs: &[&str], count_as_hit: bool) -> MatchExternalKvRequest {
    MatchExternalKvRequest {
        hashes: hashes(hs),
        count_as_hit,
    }
}

/// Returns the tiers a worker holds a hash at, per the match response.
fn tiers_for(resp: &MatchExternalKvResponse, worker: &str, hash: &str) -> Vec<i32> {
    let mut tiers = Vec::new();
    for m in &resp.matches {
        if m.worker_id != worker {
            continue;
        }
        for th in &m.hashes_by_tier {
            if th.hashes.iter().any(|h| h == hash) {
                tiers.push(th.tier);
            }
        }
    }
    tiers.sort_unstable();
    tiers
}

macro_rules! itest {
    ($name:ident, $b:ident, $body:block) => {
        #[tokio::test]
        async fn $name() {
            let Some($b) = backend(stringify!($name)).await else {
                return;
            };
            $body
        }
    };
}

itest!(report_then_match_returns_worker_and_address, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "10.0.0.1:9000",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &["1", "2"],
        )],
    ))
    .await
    .unwrap();

    let resp = b
        .match_external_kv(match_req(&["1", "2", "3"], false))
        .await
        .unwrap();
    assert_eq!(resp.matches.len(), 1);
    let m = &resp.matches[0];
    assert_eq!(m.worker_id, "w1");
    assert_eq!(m.address, "10.0.0.1:9000");
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
    assert_eq!(tiers_for(&resp, "w1", "2"), vec![hbm()]);
    assert!(tiers_for(&resp, "w1", "3").is_empty());
});

itest!(large_request_crosses_redis_fanout_chunks, b, {
    // The backend fan-out chunk is 256. Exercise more than one chunk on both
    // write and read paths while preserving complete, ordered results.
    let hashes: Vec<String> = (0..300).map(|i| format!("chunk-{i}")).collect();
    let hash_refs: Vec<&str> = hashes.iter().map(String::as_str).collect();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &hash_refs,
        )],
    ))
    .await
    .unwrap();

    let resp = b
        .match_external_kv(match_req(&hash_refs, false))
        .await
        .unwrap();
    let worker = resp
        .matches
        .iter()
        .find(|m| m.worker_id == "w1")
        .expect("worker must match");
    let tier = worker
        .hashes_by_tier
        .iter()
        .find(|t| t.tier == hbm())
        .expect("HBM tier must match");
    assert_eq!(tier.hashes, hashes);
});

itest!(duplicate_report_is_idempotent, b, {
    for _ in 0..3 {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    }
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
});

itest!(identical_batch_replay_is_idempotent, b, {
    // A batch that stores then removes then stores again the same hash; the net
    // state is "stored". Re-delivering the identical batch must not change it,
    // since every individual mutation is idempotent.
    let batch = apply_req(
        "w1",
        "a",
        7,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["9"]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &["9"]),
            action(ExternalKvActionType::ActionReport, hbm(), &["9"]),
        ],
    );
    b.apply_external_kv_batch(batch.clone()).await.unwrap();
    let first = b.match_external_kv(match_req(&["9"], false)).await.unwrap();
    b.apply_external_kv_batch(batch).await.unwrap();
    let second = b.match_external_kv(match_req(&["9"], false)).await.unwrap();
    assert_eq!(tiers_for(&first, "w1", "9"), vec![hbm()]);
    assert_eq!(tiers_for(&second, "w1", "9"), vec![hbm()]);
});

itest!(recomputed_full_node_restores_hbm_placement, b, {
    // HiRadixCache lifecycle for an exact-match recomputation:
    // BlockStored(GPU) -> BlockRemoved(GPU) -> BlockStored(GPU).
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &["full"])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["full"])],
    ))
    .await
    .unwrap();
    let evicted = b
        .match_external_kv(match_req(&["full"], false))
        .await
        .unwrap();
    assert!(tiers_for(&evicted, "w1", "full").is_empty());

    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        3,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &["full"])],
    ))
    .await
    .unwrap();
    let restored = b
        .match_external_kv(match_req(&["full"], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&restored, "w1", "full"), vec![hbm()]);
});

itest!(recomputed_split_reports_only_materialized_hashes, b, {
    // An evicted [prefix -> old suffix] is partially recomputed as
    // [prefix -> new suffix]. The old suffix must remain absent.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &["prefix", "old-suffix"],
        )],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(
            ExternalKvActionType::ActionRevoke,
            hbm(),
            &["prefix", "old-suffix"],
        )],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        3,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["prefix"]),
            action(ExternalKvActionType::ActionReport, hbm(), &["new-suffix"]),
        ],
    ))
    .await
    .unwrap();

    let result = b
        .match_external_kv(match_req(&["prefix", "old-suffix", "new-suffix"], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&result, "w1", "prefix"), vec![hbm()]);
    assert!(tiers_for(&result, "w1", "old-suffix").is_empty());
    assert_eq!(tiers_for(&result, "w1", "new-suffix"), vec![hbm()]);
});

itest!(recomputed_batch_replay_keeps_cpu_copy, b, {
    // Re-materializing on GPU must not revoke the existing host backup, and
    // re-delivering the same batch must leave both tiers unchanged because the
    // individual mutations are idempotent.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            dram(),
            &["tiered"],
        )],
    ))
    .await
    .unwrap();
    let recomputed = apply_req(
        "w1",
        "a",
        2,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &["tiered"],
        )],
    );
    b.apply_external_kv_batch(recomputed.clone()).await.unwrap();
    b.apply_external_kv_batch(recomputed).await.unwrap();

    let result = b
        .match_external_kv(match_req(&["tiered"], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&result, "w1", "tiered"), vec![hbm(), dram()]);
});

itest!(cluster_client_follows_ask_redirect, b, {
    if std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES").is_err() {
        eprintln!("skipping ASK redirect test outside Redis Cluster");
        return;
    }
    let Ok(hash) = std::env::var("KV_INDEXER_ASK_HASH") else {
        eprintln!("skipping ASK redirect test: set KV_INDEXER_ASK_HASH");
        return;
    };

    // The external harness marks this hash's slot MIGRATING on its owner and
    // IMPORTING on another master before starting the test. This namespace is
    // unique, so the placement key is absent on the source and Redis responds
    // with ASK. The cluster client must issue ASKING on the target and retry.
    b.apply_external_kv_batch(apply_req(
        "ask-worker",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[&hash])],
    ))
    .await
    .unwrap();
    let result = b
        .match_external_kv(match_req(&[&hash], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&result, "ask-worker", &hash), vec![hbm()]);
});

itest!(revoke_partial_tier_keeps_other_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["1"]),
            action(ExternalKvActionType::ActionReport, dram(), &["1"]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &["1"]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![dram()]);
});

itest!(revoke_missing_hash_is_idempotent, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["404"])],
    ))
    .await
    .unwrap();
    let resp = b
        .match_external_kv(match_req(&["404"], false))
        .await
        .unwrap();
    assert!(resp.matches.is_empty());
});

itest!(multi_worker_multi_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a1",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w2",
        "a2",
        1,
        vec![action(ExternalKvActionType::ActionReport, dram(), &["1"])],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(resp.matches.len(), 2);
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
    assert_eq!(tiers_for(&resp, "w2", "1"), vec![dram()]);
});

itest!(clear_all_at_tier_removes_only_that_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["1", "2", "3"]),
            action(ExternalKvActionType::ActionReport, dram(), &["1"]),
        ],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(
            ExternalKvActionType::ActionClearAllAtTier,
            hbm(),
            &[],
        )],
    ))
    .await
    .unwrap();
    let resp = b
        .match_external_kv(match_req(&["1", "2", "3"], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![dram()]);
    assert!(tiers_for(&resp, "w1", "2").is_empty());
    assert!(tiers_for(&resp, "w1", "3").is_empty());
});

itest!(
    count_as_hit_only_counts_matched_and_replay_does_not_double,
    b,
    {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();

        // Diagnostic match (count_as_hit=false) must not count.
        b.match_external_kv(match_req(&["1", "2"], false))
            .await
            .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
                hashes: hashes(&["1", "2"]),
            })
            .await
            .unwrap();
        assert!(counts.entries.is_empty());

        // Counting match: only the matched hash "1" is counted, "2" (a miss) is not.
        b.match_external_kv(match_req(&["1", "2"], true))
            .await
            .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
                hashes: hashes(&["1", "2"]),
            })
            .await
            .unwrap();
        assert_eq!(counts.entries.len(), 1);
        assert_eq!(counts.entries[0].hash, "1");
        assert_eq!(counts.entries[0].hit_count_total, 1);

        // Replaying the apply batch must not touch hit counts.
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
                hashes: hashes(&["1"]),
            })
            .await
            .unwrap();
        assert_eq!(counts.entries[0].hit_count_total, 1);
    }
);

itest!(full_revoke_drops_hit_key, b, {
    // Report a block, count a hit (creates the co-located :h key), then fully
    // revoke it. The hit key must be removed together with placement; otherwise a
    // matched-then-evicted block leaks its :h key forever (slow Redis memory growth).
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
    ))
    .await
    .unwrap();

    // Counting match creates the hit key with c=1.
    b.match_external_kv(match_req(&["1"], true)).await.unwrap();
    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: hashes(&["1"]),
        })
        .await
        .unwrap();
    assert_eq!(counts.entries.len(), 1);
    assert_eq!(counts.entries[0].hit_count_total, 1);

    // Fully revoke the block: placement empties, so the hit key must go too.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
    ))
    .await
    .unwrap();

    // Placement is gone.
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert!(resp.matches.is_empty());

    // Hit key is gone too: a leaked :h would still report a count here.
    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: hashes(&["1"]),
        })
        .await
        .unwrap();
    assert!(
        counts.entries.is_empty(),
        "hit key leaked after full revoke: {:?}",
        counts.entries
    );
});

itest!(partial_revoke_keeps_hit_key, b, {
    // Block present at two tiers; count a hit, then revoke only one tier. Placement
    // is still non-empty, so the hit key must survive (guard against over-deletion).
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["1"]),
            action(ExternalKvActionType::ActionReport, dram(), &["1"]),
        ],
    ))
    .await
    .unwrap();

    b.match_external_kv(match_req(&["1"], true)).await.unwrap();

    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
    ))
    .await
    .unwrap();

    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: hashes(&["1"]),
        })
        .await
        .unwrap();
    assert_eq!(counts.entries.len(), 1);
    assert_eq!(counts.entries[0].hit_count_total, 1);
});

itest!(batch_action_order_is_preserved, b, {
    // revoke-then-report on the same hash within one batch must net to "stored".
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionRevoke, hbm(), &["5"]),
            action(ExternalKvActionType::ActionReport, hbm(), &["5"]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["5"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "5"), vec![hbm()]);

    // report-then-revoke on the same hash must net to "absent".
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["6"]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &["6"]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["6"], false)).await.unwrap();
    assert!(tiers_for(&resp, "w1", "6").is_empty());
});

// --- server-side seq gate (durable idempotency) -----------------------------

// --- prefix query: Redis fast path vs. the trait's default implementation ----
//
// The default `match_external_kv_prefix` (composed from `match_external_kv`) is
// the written semantics; the Redis override is a command-count optimization. It
// must stay field-for-field identical on the parts that ARE the contract
// (per-worker prefix set and best_prefix_blocks); `blocks_read` is observability
// and legitimately differs, so it is not compared.

/// Delegates every RPC to a Redis backend EXCEPT `match_external_kv_prefix`,
/// which it leaves to the trait default — giving a reference answer computed from
/// the same store the fast path reads.
struct DefaultViaRedis(RedisKvIndexerBackend);

#[tonic::async_trait]
impl KvIndexerBackend for DefaultViaRedis {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        self.0.apply_external_kv_batch(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        self.0.match_external_kv(request).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.0.get_external_kv_hit_counts(request).await
    }
}

/// Two backends on one shared namespace, or `None` (skip) with no store.
async fn shared_ns_pair(test: &str) -> Option<(RedisKvIndexerBackend, RedisKvIndexerBackend)> {
    let ns = format!("itest:{test}:{}", nanos());
    let connect = |ns: String| async move {
        if let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") {
            Some(
                RedisKvIndexerBackend::connect_single(&url, ns)
                    .await
                    .expect("connect single"),
            )
        } else if let Ok(nodes) = std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES") {
            let nodes: Vec<String> = nodes
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            Some(
                RedisKvIndexerBackend::connect_cluster(nodes, ns)
                    .await
                    .expect("connect cluster"),
            )
        } else {
            None
        }
    };
    match (connect(ns.clone()).await, connect(ns).await) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => {
            require::skip(
                test,
                "neither KV_INDEXER_REDIS_URL nor KV_INDEXER_REDIS_CLUSTER_NODES is set",
            );
            None
        }
    }
}

fn prefix_req(hs: &[&str]) -> MatchExternalKvPrefixRequest {
    MatchExternalKvPrefixRequest {
        hashes: hashes(hs),
        max_blocks: 0,
    }
}

/// Sorted `(worker_id, matched_prefix_blocks)` — the semantic content of a
/// prefix response, independent of `blocks_read`.
fn prefix_pairs(resp: &MatchExternalKvPrefixResponse) -> Vec<(String, u32)> {
    let mut pairs: Vec<(String, u32)> = resp
        .matches
        .iter()
        .map(|m| (m.worker_id.clone(), m.matched_prefix_blocks))
        .collect();
    pairs.sort();
    pairs
}

fn report(worker: &str, addr: &str, seq: u64, hs: &[&str]) -> ApplyExternalKvBatchRequest {
    apply_req(
        worker,
        addr,
        seq,
        vec![action(ExternalKvActionType::ActionReport, hbm(), hs)],
    )
}

#[tokio::test]
async fn prefix_fast_path_matches_default_impl() {
    let Some((fast, reference)) = shared_ns_pair("prefix_parity").await else {
        return;
    };
    let reference = DefaultViaRedis(reference);

    // Nested prefixes (hole-free), a diverging branch, and a hole.
    fast.apply_external_kv_batch(report("w-long", "10.0.0.1:1", 1, &["a", "b", "c", "d"]))
        .await
        .unwrap();
    fast.apply_external_kv_batch(report("w-short", "10.0.0.2:1", 1, &["a", "b"]))
        .await
        .unwrap();
    // w-hole holds a, c, d but not b: strict prefix must be 1.
    fast.apply_external_kv_batch(report("w-hole", "10.0.0.3:1", 1, &["a", "c", "d"]))
        .await
        .unwrap();
    // w-noaddr is unroutable and must be excluded by both paths.
    fast.apply_external_kv_batch(report("w-noaddr", "", 1, &["a", "b"]))
        .await
        .unwrap();

    let query = ["a", "b", "c", "d"];
    let fast_resp = fast
        .match_external_kv_prefix(prefix_req(&query))
        .await
        .unwrap();
    let ref_resp = reference
        .match_external_kv_prefix(prefix_req(&query))
        .await
        .unwrap();

    assert_eq!(prefix_pairs(&fast_resp), prefix_pairs(&ref_resp));
    assert_eq!(fast_resp.best_prefix_blocks, ref_resp.best_prefix_blocks);
    assert_eq!(fast_resp.best_prefix_blocks, 4);
    assert_eq!(
        prefix_pairs(&fast_resp),
        vec![
            ("w-hole".to_string(), 1),
            ("w-long".to_string(), 4),
            ("w-short".to_string(), 2),
        ]
    );
    assert!(fast_resp
        .matches
        .iter()
        .all(|m| !m.worker_address.is_empty()));
    // Descending order and first-block read are part of the response contract.
    assert_eq!(fast_resp.matches[0].matched_prefix_blocks, 4);
    assert!(fast_resp.blocks_read >= 1);
}

#[tokio::test]
async fn prefix_first_block_miss_reads_one_block() {
    let Some(b) = backend("prefix_first_miss").await else {
        return;
    };
    // No worker holds the first queried block; the scan stops after one read.
    b.apply_external_kv_batch(report("w1", "10.0.0.1:1", 1, &["y", "z"]))
        .await
        .unwrap();
    let resp = b
        .match_external_kv_prefix(prefix_req(&["x", "y", "z"]))
        .await
        .unwrap();
    assert!(resp.matches.is_empty());
    assert_eq!(resp.best_prefix_blocks, 0);
    assert_eq!(resp.blocks_read, 1);
}

#[tokio::test]
async fn prefix_max_blocks_caps_the_scan() {
    let Some(b) = backend("prefix_max_blocks").await else {
        return;
    };
    b.apply_external_kv_batch(report("w1", "10.0.0.1:1", 1, &["a", "b", "c", "d"]))
        .await
        .unwrap();
    let resp = b
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: hashes(&["a", "b", "c", "d"]),
            max_blocks: 2,
        })
        .await
        .unwrap();
    // Capped at 2 even though the worker holds all four.
    assert_eq!(resp.best_prefix_blocks, 2);
    assert_eq!(resp.blocks_read, 2);
    assert_eq!(resp.matches.len(), 1);
    assert_eq!(resp.matches[0].matched_prefix_blocks, 2);
}
