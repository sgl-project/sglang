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
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType,
    GetExternalKvHitCountsRequest, MatchExternalKvRequest, MatchExternalKvResponse,
};
use sgl_kv_indexer::{KvIndexerBackend, RedisKvIndexerBackend};
use test_id::nanos;
use test_kv::{action, apply_request as apply_req, dram, hashes, hbm};

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

/// Like [`apply_req`] but tags the batch with a worker incarnation, so a change
/// across calls simulates a worker restart.
fn apply_req_inc(
    worker: &str,
    addr: &str,
    incarnation: &str,
    seq: u64,
    actions: Vec<ExternalKvAction>,
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        incarnation: incarnation.to_string(),
        ..apply_req(worker, addr, seq, actions)
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

itest!(same_seq_batch_replay_is_idempotent, b, {
    // A batch that stores then removes then stores again the same hash; the net
    // state is "stored". Replaying the identical batch (same seq) must not change it.
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

itest!(
    recomputed_batch_replay_is_idempotent_and_keeps_cpu_copy,
    b,
    {
        // Re-materializing on GPU must not revoke the existing host backup. If the
        // bridge replays the same sequence, the durable sequence gate rejects the
        // duplicate without changing either tier.
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
        let first = b.apply_external_kv_batch(recomputed.clone()).await.unwrap();
        let replay = b.apply_external_kv_batch(recomputed).await.unwrap();
        assert!(!first.duplicate);
        assert!(replay.duplicate);

        let result = b
            .match_external_kv(match_req(&["tiered"], false))
            .await
            .unwrap();
        assert_eq!(tiers_for(&result, "w1", "tiered"), vec![hbm(), dram()]);
    }
);

itest!(cluster_client_recovers_after_master_failover, b, {
    if std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES").is_err() {
        eprintln!("skipping cluster failover test outside Redis Cluster");
        return;
    }
    let Ok(hash) = std::env::var("KV_INDEXER_FAILOVER_HASH") else {
        eprintln!("skipping cluster failover test: set KV_INDEXER_FAILOVER_HASH");
        return;
    };
    let pause_secs = std::env::var("KV_INDEXER_FAILOVER_PAUSE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(8);

    b.apply_external_kv_batch(apply_req(
        "failover-worker",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[&hash])],
    ))
    .await
    .unwrap();
    eprintln!("FAILOVER_READY hash={hash}; stop its Redis master within {pause_secs}s");
    tokio::time::sleep(std::time::Duration::from_secs(pause_secs)).await;

    // Revoke then report after the external harness has stopped the slot's
    // original master. This exercises topology refresh, replica promotion,
    // Lua reload and both forward/reverse writes on the existing client.
    b.apply_external_kv_batch(apply_req(
        "failover-worker",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &[&hash])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "failover-worker",
        "a",
        3,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[&hash])],
    ))
    .await
    .unwrap();

    let result = b
        .match_external_kv(match_req(&[&hash], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&result, "failover-worker", &hash), vec![hbm()]);
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

itest!(cluster_heartbeat_follows_ask_redirect, b, {
    if std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES").is_err() {
        eprintln!("skipping heartbeat ASK test outside Redis Cluster");
        return;
    }
    let Ok(worker) = std::env::var("KV_INDEXER_ASK_HEARTBEAT_WORKER") else {
        eprintln!("skipping heartbeat ASK test: set KV_INDEXER_ASK_HEARTBEAT_WORKER");
        return;
    };

    // The harness migrates the `{w:<worker>}` slot before this call. TOUCH_META
    // is a multi-key Lua heartbeat in that slot and must follow ASK even when
    // the importing master has not cached the script.
    let response = b
        .apply_external_kv_batch(apply_req_inc(&worker, "a", "ask-heartbeat", 0, vec![]))
        .await
        .unwrap();
    assert!(
        !response.has_applied_seq,
        "a fresh heartbeat must not invent a durable event checkpoint"
    );
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

itest!(duplicate_seq_batch_is_skipped_not_reapplied, b, {
    // Apply a report at seq=1, then re-send seq=1 carrying a *different*
    // mutation (a revoke). Because seq=1 was already applied, the whole batch
    // is a duplicate and must be skipped, so the revoke has no effect.
    let applied = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert_eq!(applied.last_applied_seq, 1);
    assert!(!applied.duplicate);

    let dup = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert!(dup.duplicate, "re-sent seq must be reported as a duplicate");
    assert_eq!(dup.last_applied_seq, 1);

    // The revoke was skipped: the hash is still present.
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
});

itest!(lower_seq_batch_is_skipped_as_stale, b, {
    // Advance to seq=5, then a late/out-of-order seq=3 arrives. It must be
    // treated as stale (<= last) and skipped, leaving the ack at 5.
    for seq in [1u64, 5] {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            seq,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    }
    let stale = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            3,
            vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert!(stale.duplicate);
    assert_eq!(stale.last_applied_seq, 5);
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
});

itest!(seq_is_per_worker_independent, b, {
    // seq counters are independent per worker: the same seq value applied to a
    // different worker is not a duplicate.
    let a = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    let c = b
        .apply_external_kv_batch(apply_req(
            "w2",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert!(!a.duplicate && !c.duplicate);
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(resp.matches.len(), 2);
});

// --- worker liveness / stale-entry cleanup ----------------------------------

itest!(
    stale_worker_is_dropped_from_match_then_revived_by_heartbeat,
    b,
    {
        // Arm a short per-worker TTL so a worker that stops talking expires quickly.
        let b = b.with_worker_ttl(Some(std::time::Duration::from_secs(1)));

        // Report a block: immediately matchable while the worker is live.
        b.apply_external_kv_batch(apply_req(
            "w1",
            "10.0.0.1:9000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
        let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
        assert_eq!(resp.matches.len(), 1, "live worker must be matchable");

        // Let the worker go silent past its TTL: match must drop it so the router
        // never targets a dead node, even though placement/reverse entries linger.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
        assert!(
            resp.matches.is_empty(),
            "stale worker must be dropped from match, got {:?}",
            resp.matches
        );
        let resp = b.match_external_kv(match_req(&["1"], true)).await.unwrap();
        assert!(resp.matches.is_empty());
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
                hashes: hashes(&["1"]),
            })
            .await
            .unwrap();
        assert!(
            counts.entries.is_empty(),
            "a placement held only by dead workers must not count as a returned hit"
        );

        // A heartbeat (empty-actions apply) refreshes liveness; the placement was
        // never deleted, so the old block is instantly routable again.
        b.apply_external_kv_batch(apply_req("w1", "10.0.0.1:9000", 0, vec![]))
            .await
            .unwrap();
        let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
        assert_eq!(
            tiers_for(&resp, "w1", "1"),
            vec![hbm()],
            "heartbeat must revive the worker with its existing placement intact"
        );
    }
);

// --- worker restart / incarnation reset -------------------------------------

itest!(
    worker_restart_new_incarnation_wipes_stale_state_and_resets_seq,
    b,
    {
        // Incarnation "A": report h1 at seq 5, so the durable seq climbs to 5.
        b.apply_external_kv_batch(apply_req_inc(
            "w1",
            "10.0.0.1:9000",
            "A",
            5,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
        let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
        assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);

        // Same incarnation, a stale/replayed low seq is still skipped as duplicate.
        let dup = b
            .apply_external_kv_batch(apply_req_inc(
                "w1",
                "10.0.0.1:9000",
                "A",
                2,
                vec![action(ExternalKvActionType::ActionReport, hbm(), &["2"])],
            ))
            .await
            .unwrap();
        assert!(
            dup.duplicate,
            "low seq within same incarnation is a duplicate"
        );

        // Incarnation "B" (a restart) with a reset sequence (seq 1) reporting a new
        // block. The restart must: (a) wipe the old placement for h1, (b) reset the
        // durable seq so seq=1 is accepted (not skipped), (c) index the new block.
        let after = b
            .apply_external_kv_batch(apply_req_inc(
                "w1",
                "10.0.0.1:9000",
                "B",
                1,
                vec![action(ExternalKvActionType::ActionReport, hbm(), &["9"])],
            ))
            .await
            .unwrap();
        assert!(
            !after.duplicate,
            "restarted worker's reset sequence must be accepted, not skipped"
        );

        let resp = b
            .match_external_kv(match_req(&["1", "9"], false))
            .await
            .unwrap();
        assert!(
            tiers_for(&resp, "w1", "1").is_empty(),
            "stale block from the previous incarnation must be wiped, got {:?}",
            resp.matches
        );
        assert_eq!(
            tiers_for(&resp, "w1", "9"),
            vec![hbm()],
            "new block from the restarted incarnation must be indexed"
        );
    }
);

itest!(
    // P1 regression: a worker that expired out of `match` (its liveness TTL
    // lapsed) and then slow-restarts with a NEW incarnation must still be reset.
    // The durable meta (and thus the previous incarnation) must survive the TTL
    // so the restart is detected and the stale placement is wiped — otherwise the
    // dead incarnation's blocks would resurrect on the heartbeat that revives it.
    expired_worker_new_incarnation_still_wipes_stale_state,
    b,
    {
        let b = b.with_worker_ttl(Some(std::time::Duration::from_secs(1)));

        // Incarnation "A" reports h1.
        b.apply_external_kv_batch(apply_req_inc(
            "w1",
            "10.0.0.1:9000",
            "A",
            5,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
        assert_eq!(
            tiers_for(
                &b.match_external_kv(match_req(&["1"], false)).await.unwrap(),
                "w1",
                "1"
            ),
            vec![hbm()],
        );

        // The worker goes silent long enough for its liveness key to expire.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        assert!(
            b.match_external_kv(match_req(&["1"], false))
                .await
                .unwrap()
                .matches
                .is_empty(),
            "expired worker must be dropped from match",
        );

        // It slow-restarts as incarnation "B" (seq reset to 1) reporting h9. Even
        // though the liveness key had expired, the durable incarnation "A" is
        // still recorded, so the restart is detected: h1 is wiped, seq accepted.
        let after = b
            .apply_external_kv_batch(apply_req_inc(
                "w1",
                "10.0.0.1:9000",
                "B",
                1,
                vec![action(ExternalKvActionType::ActionReport, hbm(), &["9"])],
            ))
            .await
            .unwrap();
        assert!(
            !after.duplicate,
            "restarted worker's reset seq must be accepted"
        );

        let resp = b
            .match_external_kv(match_req(&["1", "9"], false))
            .await
            .unwrap();
        assert!(
            tiers_for(&resp, "w1", "1").is_empty(),
            "stale block from expired-then-restarted incarnation must be wiped, got {:?}",
            resp.matches
        );
        assert_eq!(tiers_for(&resp, "w1", "9"), vec![hbm()]);
    }
);

// --- T5 regression: degraded-start / lazy-connect semantics -----------------

/// A deferred backend (KV_INDEXER_REDIS_REQUIRED=0 path) does not connect at
/// construction; it connects lazily on first use and serves correctly once the
/// store is reachable. Requires a live store, so it skips when unset.
#[tokio::test]
async fn deferred_backend_connects_lazily_and_serves() {
    let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") else {
        eprintln!("skipping deferred_backend_connects_lazily_and_serves: set KV_INDEXER_REDIS_URL");
        return;
    };
    let ns = format!("itest:deferred_serves:{}", nanos());
    // Construction never touches the network.
    let b = RedisKvIndexerBackend::connect_single_deferred(&url, ns);
    // First use lazily establishes the connection and succeeds.
    b.apply_external_kv_batch(apply_req(
        "wd",
        "10.9.9.9:9000",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &["100", "101"],
        )],
    ))
    .await
    .expect("lazy connect + apply should succeed against a live store");
    let resp = b
        .match_external_kv(match_req(&["100", "101"], false))
        .await
        .unwrap();
    assert!(
        resp.matches.iter().any(|m| m.worker_id == "wd"),
        "reported hashes must be matchable after a lazy connect"
    );
}

/// A deferred backend pointed at an unreachable Redis must fail requests with an
/// error within a bounded time (connect timeout), never hang. No store needed.
#[tokio::test]
async fn deferred_backend_unreachable_errors_within_bound() {
    let b = RedisKvIndexerBackend::connect_single_deferred(
        "redis://127.0.0.1:6399",
        "itest:deferred_dead",
    );
    let started = std::time::Instant::now();
    let res = b.match_external_kv(match_req(&["x"], false)).await;
    let elapsed = started.elapsed();
    assert!(
        res.is_err(),
        "match against an unreachable redis must error, got {res:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(15),
        "request must be bounded by the connect timeout, took {elapsed:?}"
    );
}
