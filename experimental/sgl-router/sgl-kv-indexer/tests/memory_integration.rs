// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the process-local in-memory backend.

#[path = "common/kv.rs"]
mod test_kv;

use std::sync::Arc;

use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse,
    WorkerCacheSpec,
};
use sgl_kv_indexer::{
    InMemoryKvIndexerBackend, KvIndexerBackend, WorkerPrefixInput, COMPONENT_FULL, COMPONENT_SWA,
};
use test_kv::{action, apply_request as apply_req, component_report, dram, hbm};
use tonic::Status;

fn backend() -> InMemoryKvIndexerBackend {
    InMemoryKvIndexerBackend::new()
}

fn match_req(hs: &[i64], count_as_hit: bool) -> MatchExternalKvRequest {
    MatchExternalKvRequest {
        hashes: hs.to_vec(),
        count_as_hit,
    }
}

/// Returns the tiers a worker holds a hash at, per the match response.
fn tiers_for(resp: &MatchExternalKvResponse, worker: &str, hash: i64) -> Vec<i32> {
    let mut tiers = Vec::new();
    for m in &resp.matches {
        if m.worker_id != worker {
            continue;
        }
        for th in &m.hashes_by_tier {
            if th.hashes.contains(&hash) {
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
            let $b = backend();
            $body
        }
    };
}

itest!(report_then_match_returns_worker_and_address, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "10.0.0.1:9000",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1, 2])],
    ))
    .await
    .unwrap();

    let resp = b
        .match_external_kv(match_req(&[1, 2, 3], false))
        .await
        .unwrap();
    assert_eq!(resp.matches.len(), 1);
    let m = &resp.matches[0];
    assert_eq!(m.worker_id, "w1");
    assert_eq!(m.address, "10.0.0.1:9000");
    assert_eq!(tiers_for(&resp, "w1", 1), vec![hbm()]);
    assert_eq!(tiers_for(&resp, "w1", 2), vec![hbm()]);
    assert!(tiers_for(&resp, "w1", 3).is_empty());
});

itest!(large_request_preserves_complete_ordered_results, b, {
    // Exercise a large write and read while preserving complete ordered results.
    let expected_hashes: Vec<i64> = (0..300).collect();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &expected_hashes,
        )],
    ))
    .await
    .unwrap();

    let resp = b
        .match_external_kv(match_req(&expected_hashes, false))
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
    assert_eq!(tier.hashes, expected_hashes);
});

itest!(duplicate_report_is_idempotent, b, {
    for _ in 0..3 {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .unwrap();
    }
    let resp = b.match_external_kv(match_req(&[1], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", 1), vec![hbm()]);
});

itest!(identical_batch_replay_is_idempotent, b, {
    // Stores, removes, then stores the same hash again; the net state is
    // "stored". Re-delivering the identical batch must not change it.
    let batch = apply_req(
        "w1",
        "a",
        7,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &[9]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &[9]),
            action(ExternalKvActionType::ActionReport, hbm(), &[9]),
        ],
    );
    b.apply_external_kv_batch(batch.clone()).await.unwrap();
    let first = b.match_external_kv(match_req(&[9], false)).await.unwrap();
    b.apply_external_kv_batch(batch).await.unwrap();
    let second = b.match_external_kv(match_req(&[9], false)).await.unwrap();
    assert_eq!(tiers_for(&first, "w1", 9), vec![hbm()]);
    assert_eq!(tiers_for(&second, "w1", 9), vec![hbm()]);
});

itest!(recomputed_full_node_restores_hbm_placement, b, {
    // HiRadixCache lifecycle for an exact-match recomputation:
    // BlockStored(GPU) -> BlockRemoved(GPU) -> BlockStored(GPU).
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &[1])],
    ))
    .await
    .unwrap();
    let evicted = b.match_external_kv(match_req(&[1], false)).await.unwrap();
    assert!(tiers_for(&evicted, "w1", 1).is_empty());

    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        3,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
    ))
    .await
    .unwrap();
    let restored = b.match_external_kv(match_req(&[1], false)).await.unwrap();
    assert_eq!(tiers_for(&restored, "w1", 1), vec![hbm()]);
});

itest!(recomputed_split_reports_only_materialized_hashes, b, {
    // An evicted [prefix -> old suffix] is partially recomputed as
    // [prefix -> new suffix]. The old suffix must remain absent.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1, 2])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &[1, 2])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        3,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &[1]),
            action(ExternalKvActionType::ActionReport, hbm(), &[3]),
        ],
    ))
    .await
    .unwrap();

    let result = b
        .match_external_kv(match_req(&[1, 2, 3], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&result, "w1", 1), vec![hbm()]);
    assert!(tiers_for(&result, "w1", 2).is_empty());
    assert_eq!(tiers_for(&result, "w1", 3), vec![hbm()]);
});

itest!(recomputed_batch_replay_keeps_cpu_copy, b, {
    // Re-materializing on GPU must not revoke the existing host backup, and
    // re-delivering the same batch must leave both tiers unchanged.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, dram(), &[1])],
    ))
    .await
    .unwrap();
    let recomputed = apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
    );
    b.apply_external_kv_batch(recomputed.clone()).await.unwrap();
    b.apply_external_kv_batch(recomputed).await.unwrap();

    let result = b.match_external_kv(match_req(&[1], false)).await.unwrap();
    assert_eq!(tiers_for(&result, "w1", 1), vec![hbm(), dram()]);
});

itest!(revoke_partial_tier_keeps_other_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &[1]),
            action(ExternalKvActionType::ActionReport, dram(), &[1]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &[1]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&[1], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", 1), vec![dram()]);
});

itest!(revoke_missing_hash_is_idempotent, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &[404])],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&[404], false)).await.unwrap();
    assert!(resp.matches.is_empty());
});

itest!(multi_worker_multi_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a1",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w2",
        "a2",
        1,
        vec![action(ExternalKvActionType::ActionReport, dram(), &[1])],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&[1], false)).await.unwrap();
    assert_eq!(resp.matches.len(), 2);
    assert_eq!(tiers_for(&resp, "w1", 1), vec![hbm()]);
    assert_eq!(tiers_for(&resp, "w2", 1), vec![dram()]);
});

itest!(clear_all_at_tier_removes_only_that_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &[1, 2, 3]),
            action(ExternalKvActionType::ActionReport, dram(), &[1]),
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
        .match_external_kv(match_req(&[1, 2, 3], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&resp, "w1", 1), vec![dram()]);
    assert!(tiers_for(&resp, "w1", 2).is_empty());
    assert!(tiers_for(&resp, "w1", 3).is_empty());
});

itest!(
    count_as_hit_only_counts_matched_and_replay_does_not_double,
    b,
    {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .unwrap();

        // Diagnostic match (count_as_hit=false) must not count.
        b.match_external_kv(match_req(&[1, 2], false))
            .await
            .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1, 2] })
            .await
            .unwrap();
        assert!(counts.entries.is_empty());

        // Counting match: only the matched hash "1" is counted, "2" (a miss) is not.
        b.match_external_kv(match_req(&[1, 2], true)).await.unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1, 2] })
            .await
            .unwrap();
        assert_eq!(counts.entries.len(), 1);
        assert_eq!(counts.entries[0].hash, 1);
        assert_eq!(counts.entries[0].hit_count_total, 1);

        // Replaying the apply batch must not touch hit counts.
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1] })
            .await
            .unwrap();
        assert_eq!(counts.entries[0].hit_count_total, 1);
    }
);

itest!(full_revoke_drops_hit_key, b, {
    // Report a block, count a hit (creates the co-located :h key), then fully
    // revoke it. The hit key must go with the placement, or a
    // matched-then-evicted block leaks its counter forever.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
    ))
    .await
    .unwrap();

    // Counting match creates the hit key with c=1.
    b.match_external_kv(match_req(&[1], true)).await.unwrap();
    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1] })
        .await
        .unwrap();
    assert_eq!(counts.entries.len(), 1);
    assert_eq!(counts.entries[0].hit_count_total, 1);

    // Fully revoke the block: placement empties, so the hit key must go too.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &[1])],
    ))
    .await
    .unwrap();

    let resp = b.match_external_kv(match_req(&[1], false)).await.unwrap();
    assert!(resp.matches.is_empty());

    // Hit key is gone too: a leaked :h would still report a count here.
    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1] })
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
            action(ExternalKvActionType::ActionReport, hbm(), &[1]),
            action(ExternalKvActionType::ActionReport, dram(), &[1]),
        ],
    ))
    .await
    .unwrap();

    b.match_external_kv(match_req(&[1], true)).await.unwrap();

    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &[1])],
    ))
    .await
    .unwrap();

    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1] })
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
            action(ExternalKvActionType::ActionRevoke, hbm(), &[5]),
            action(ExternalKvActionType::ActionReport, hbm(), &[5]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&[5], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", 5), vec![hbm()]);

    // report-then-revoke on the same hash must net to "absent".
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &[6]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &[6]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&[6], false)).await.unwrap();
    assert!(tiers_for(&resp, "w1", 6).is_empty());
});

// --- prefix query: backend override vs. the trait's default implementation ---
//
// The trait default is the written semantics and the backend override is a read
// optimization, so they must agree field-for-field on the parts that ARE the
// contract (per-worker prefix set and best_prefix_blocks). `blocks_read` is
// observability and legitimately differs, so it is not compared.

/// Delegates every RPC to an in-memory backend EXCEPT `match_external_kv_prefix`,
/// which it leaves to the trait default — giving a reference answer computed from
/// the same state the optimized path reads.
struct DefaultViaMemory(Arc<InMemoryKvIndexerBackend>);

#[tonic::async_trait]
impl KvIndexerBackend for DefaultViaMemory {
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

    // Delegate the component-aware read to the same backend so the trait
    // default computes over the same placement and specs the fast path sees.
    async fn collect_worker_prefix_inputs(
        &self,
        hashes: &[i64],
    ) -> Result<Vec<WorkerPrefixInput>, Status> {
        self.0.collect_worker_prefix_inputs(hashes).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.0.get_external_kv_hit_counts(request).await
    }
}

fn shared_state_pair() -> (Arc<InMemoryKvIndexerBackend>, DefaultViaMemory) {
    let backend = Arc::new(InMemoryKvIndexerBackend::new());
    let reference = DefaultViaMemory(Arc::clone(&backend));
    (backend, reference)
}

fn prefix_req(hs: &[i64]) -> MatchExternalKvPrefixRequest {
    MatchExternalKvPrefixRequest {
        hashes: hs.to_vec(),
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

fn report(worker: &str, addr: &str, seq: u64, hs: &[i64]) -> ApplyExternalKvBatchRequest {
    apply_req(
        worker,
        addr,
        seq,
        vec![action(ExternalKvActionType::ActionReport, hbm(), hs)],
    )
}

#[tokio::test]
async fn prefix_fast_path_matches_default_impl() {
    let (fast, reference) = shared_state_pair();

    // Nested prefixes (hole-free), a diverging branch, and a hole.
    fast.apply_external_kv_batch(report("w-long", "10.0.0.1:1", 1, &[1, 2, 3, 4]))
        .await
        .unwrap();
    fast.apply_external_kv_batch(report("w-short", "10.0.0.2:1", 1, &[1, 2]))
        .await
        .unwrap();
    // w-hole holds 1, 3, 4 but not 2: strict prefix must be 1.
    fast.apply_external_kv_batch(report("w-hole", "10.0.0.3:1", 1, &[1, 3, 4]))
        .await
        .unwrap();
    // w-noaddr is unroutable and must be excluded by both paths.
    fast.apply_external_kv_batch(report("w-noaddr", "", 1, &[1, 2]))
        .await
        .unwrap();

    let query = [1, 2, 3, 4];
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
    let b = backend();
    // No worker holds the first queried block; the scan stops after one read.
    b.apply_external_kv_batch(report("w1", "10.0.0.1:1", 1, &[2, 3]))
        .await
        .unwrap();
    let resp = b
        .match_external_kv_prefix(prefix_req(&[1, 2, 3]))
        .await
        .unwrap();
    assert!(resp.matches.is_empty());
    assert_eq!(resp.best_prefix_blocks, 0);
    assert_eq!(resp.blocks_read, 1);
}

#[tokio::test]
async fn prefix_max_blocks_caps_the_scan() {
    let b = backend();
    b.apply_external_kv_batch(report("w1", "10.0.0.1:1", 1, &[1, 2, 3, 4]))
        .await
        .unwrap();
    let resp = b
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2, 3, 4],
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

// --- component-aware placement & prefix -------------------------------------

/// A hybrid-SWA spec: full servable from HBM+DRAM, swa a 100-token trailing
/// window servable from HBM.
fn swa_spec() -> WorkerCacheSpec {
    WorkerCacheSpec {
        version: 1,
        components: COMPONENT_FULL | COMPONENT_SWA,
        swa_window_tokens: 100,
        full_tier_mask: (1 << hbm()) | (1 << dram()),
        swa_tier_mask: 1 << hbm(),
        mamba_tier_mask: 0,
    }
}

fn apply_with_spec(
    worker: &str,
    addr: &str,
    seq: u64,
    spec: WorkerCacheSpec,
    actions: Vec<sgl_kv_indexer::pb::ExternalKvAction>,
) -> ApplyExternalKvBatchRequest {
    let mut req = apply_req(worker, addr, seq, actions);
    req.cache_spec = Some(spec);
    req
}

#[tokio::test]
async fn component_prefix_matches_default_impl() {
    let (fast, reference) = shared_state_pair();

    // Four full blocks (50 tokens each); swa present on all but the 4th, so the
    // largest boundary with an unbroken 100-token swa window is 3.
    let report = component_report(
        hbm(),
        &[1, 2, 3, 4],
        &[
            COMPONENT_FULL | COMPONENT_SWA,
            COMPONENT_FULL | COMPONENT_SWA,
            COMPONENT_FULL | COMPONENT_SWA,
            COMPONENT_FULL,
        ],
        &[50, 50, 50, 50],
    );
    fast.apply_external_kv_batch(apply_with_spec(
        "w-swa",
        "10.0.0.1:1",
        1,
        swa_spec(),
        vec![report],
    ))
    .await
    .unwrap();

    let query = [1, 2, 3, 4];
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
    assert_eq!(prefix_pairs(&fast_resp), vec![("w-swa".to_string(), 3)]);
}

#[tokio::test]
async fn partial_eviction_replace_shrinks_component_set() {
    let b = backend();
    // Store full+swa, then restate to full-only (partial swa eviction) via a
    // REPLACE snapshot for the same (hash, tier). No BlockRemoved is involved.
    b.apply_external_kv_batch(apply_with_spec(
        "w1",
        "10.0.0.1:1",
        1,
        swa_spec(),
        vec![component_report(
            hbm(),
            &[1, 2],
            &[
                COMPONENT_FULL | COMPONENT_SWA,
                COMPONENT_FULL | COMPONENT_SWA,
            ],
            &[80, 80],
        )],
    ))
    .await
    .unwrap();
    // Both blocks reusable (window 100 met by 2x80 tokens; head rule also holds).
    let before = b
        .match_external_kv_prefix(prefix_req(&[1, 2]))
        .await
        .unwrap();
    assert_eq!(before.best_prefix_blocks, 2);

    // Restate the second block to full only: swa gone there.
    b.apply_external_kv_batch(apply_with_spec(
        "w1",
        "10.0.0.1:1",
        2,
        swa_spec(),
        vec![component_report(hbm(), &[2], &[COMPONENT_FULL], &[80])],
    ))
    .await
    .unwrap();
    // That block has no swa now, and its trailing window (only 80 < 100) is not
    // headed, so the largest valid boundary drops to 1.
    let after = b
        .match_external_kv_prefix(prefix_req(&[1, 2]))
        .await
        .unwrap();
    assert_eq!(after.best_prefix_blocks, 1);

    let snapshot = b
        .match_external_kv(match_req(&[1, 2], false))
        .await
        .unwrap();
    let tier = &snapshot.matches[0].hashes_by_tier[0];
    assert_eq!(tier.hashes, vec![1, 2]);
    assert_eq!(
        tier.component_masks,
        vec![COMPONENT_FULL | COMPONENT_SWA, COMPONENT_FULL]
    );
    assert_eq!(tier.block_sizes, vec![80, 80]);
}

#[tokio::test]
async fn component_aware_worker_without_spec_is_excluded() {
    let b = backend();
    // Report component-aware placement but never send a spec: the worker cannot
    // be interpreted and must be excluded (NoSignal-safe), never over-reported.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "10.0.0.1:1",
        1,
        vec![component_report(
            hbm(),
            &[1, 2],
            &[COMPONENT_FULL, COMPONENT_FULL],
            &[16, 16],
        )],
    ))
    .await
    .unwrap();
    let resp = b
        .match_external_kv_prefix(prefix_req(&[1, 2]))
        .await
        .unwrap();
    assert!(resp.matches.is_empty());
    assert_eq!(resp.best_prefix_blocks, 0);
}

#[tokio::test]
async fn duplicate_hash_in_one_report_keeps_last_snapshot() {
    let b = backend();
    // A single REPORT action naming the same hash twice (a coalesced
    // store+restate): the LAST snapshot must win deterministically, never a race.
    b.apply_external_kv_batch(apply_with_spec(
        "w1",
        "10.0.0.1:1",
        1,
        swa_spec(),
        vec![component_report(
            hbm(),
            &[1, 1],
            &[COMPONENT_FULL | COMPONENT_SWA, COMPONENT_FULL],
            &[80, 80],
        )],
    ))
    .await
    .unwrap();
    // The hash ends as full-only (last snapshot); with swa required and a lone 80-token
    // block that is not a full head window, the boundary requiring swa fails,
    // so no reusable prefix.
    let resp = b.match_external_kv_prefix(prefix_req(&[1])).await.unwrap();
    assert_eq!(resp.best_prefix_blocks, 0);
}

#[tokio::test]
async fn absent_spec_batch_clears_stored_spec() {
    let b = backend();
    // First a component-aware batch establishes a spec + a reusable block.
    b.apply_external_kv_batch(apply_with_spec(
        "w1",
        "10.0.0.1:1",
        1,
        swa_spec(),
        vec![component_report(
            hbm(),
            &[1],
            &[COMPONENT_FULL | COMPONENT_SWA],
            &[200],
        )],
    ))
    .await
    .unwrap();
    assert_eq!(
        b.match_external_kv_prefix(prefix_req(&[1]))
            .await
            .unwrap()
            .best_prefix_blocks,
        1
    );

    // A later batch with NO spec (worker reverted to legacy) must clear the old
    // spec. The still-component-aware placement can then no longer be interpreted
    // (component data but no spec) -> fail closed, never scored on stale rules.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "10.0.0.1:1",
        2,
        vec![action(ExternalKvActionType::ActionReport, dram(), &[2])],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv_prefix(prefix_req(&[1])).await.unwrap();
    assert!(resp.matches.is_empty());
    assert_eq!(resp.best_prefix_blocks, 0);
}
