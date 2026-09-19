// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Two REAL scoring policies whose terms DISAGREE, composed and routed. The
//! in-crate fusion tests sum `ByIndex` stubs that rank the fleet the SAME way,
//! so their `select()` half lands on ws[2] whichever term you read; this one's
//! half discriminates. (Their `scores()` half does catch a dropped term —
//! verified by mutation, so this file does not claim otherwise.)
//!
//! NOT pinned here: how `load_based` scales load — W2's min-max scale-free
//! defect is unruled, and both candidate curves put the busiest worker at 0.0
//! and the idlest at 1.0, so every assertion below holds either way.

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::engine_load::{EngineLoadSnapshot, EngineWorkerLoad};
use sgl_router::policies::kv_events::{
    compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId,
};
use sgl_router::policies::load_based::LoadBasedPolicy;
use sgl_router::policies::scoring::{
    prefix_cache::PrefixCachePolicy, FusedScorePolicy, ScorePolicy,
};
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::workers::Worker;
use std::{collections::HashMap, sync::Arc, time::Instant};

const BLOCK: usize = 4;

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: id.into(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    }))
}

#[test]
fn the_weight_override_steers_a_two_term_fusion_past_either_term_alone() {
    let ids: Vec<u32> = (0..(BLOCK as u32 * 4)).collect();
    let tree = Arc::new(HashTree::new());
    tree.insert(
        &KvWorkerId::new("hot".into(), 0),
        None,
        &compute_block_hashes(&ids, BLOCK),
    );
    let oracle = BlockSizeOracle::new();
    oracle.try_set(BLOCK as u32).expect("fresh oracle");

    // "hot" holds the whole prompt AND is the busiest: the two terms disagree.
    let ws = vec![worker("hot"), worker("cold")];
    let _held: Vec<_> = (0..3).map(|_| ws[0].load_guard()).collect();

    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
    let cache = || PrefixCachePolicy::new(Arc::clone(&tree), Arc::clone(&oracle), 1.0);

    // Vacuity guard: if the terms agreed, no weight could change the answer and
    // everything below would pass against a composer that read only one of them.
    assert_eq!(cache().select(&ws, &ctx).unwrap().id, ws[0].id, "cache→hot");
    assert_eq!(
        LoadBasedPolicy::new().select(&ws, &ctx).unwrap().id,
        ws[1].id,
        "load→cold"
    );

    // Same two terms, same fleet, same request — only the override differs.
    for (load_weight, want) in [(0.25, &ws[0]), (4.0, &ws[1])] {
        let fused = FusedScorePolicy::new(vec![
            (Arc::new(cache()) as Arc<dyn Policy>, None),
            (Arc::new(LoadBasedPolicy::new()), Some(load_weight)),
        ])
        .expect("both terms are fusable");
        let got = fused.select(&ws, &ctx).expect("non-empty fleet");
        assert_eq!(got.id, want.id, "--fuse load_based={load_weight}");
    }
}

#[test]
fn fused_load_based_term_uses_the_request_snapshot() {
    let ws = vec![worker("w0"), worker("w1")];
    // Local counters changed after the request snapshot and prefer w0.
    let _after_snapshot: Vec<_> = (0..10).map(|_| ws[1].load_guard()).collect();
    let snapshot = EngineLoadSnapshot::from_workers(
        29,
        HashMap::from([
            (
                ws[0].url.clone(),
                EngineWorkerLoad {
                    num_running_reqs: 50,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 0,
                    captured_at: Instant::now(),
                },
            ),
            (
                ws[1].url.clone(),
                EngineWorkerLoad {
                    num_running_reqs: 1,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 0,
                    captured_at: Instant::now(),
                },
            ),
        ]),
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
    let fused = FusedScorePolicy::new(vec![(Arc::new(LoadBasedPolicy::new()), None)])
        .expect("load-based is fusable");

    assert_eq!(
        fused.select(&ws, &ctx).expect("must route").id,
        ws[1].id,
        "fused score must pass the request snapshot to the load-based term"
    );
}

#[test]
fn score_policy_forwards_the_request_snapshot_to_load_based() {
    let ws = vec![worker("w0"), worker("w1")];
    let _after_snapshot: Vec<_> = (0..10).map(|_| ws[1].load_guard()).collect();
    let snapshot = EngineLoadSnapshot::from_workers(
        31,
        HashMap::from([
            (
                ws[0].url.clone(),
                EngineWorkerLoad {
                    num_running_reqs: 50,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 0,
                    captured_at: Instant::now(),
                },
            ),
            (
                ws[1].url.clone(),
                EngineWorkerLoad {
                    num_running_reqs: 1,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 0,
                    captured_at: Instant::now(),
                },
            ),
        ]),
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
    let score = ScorePolicy::new(Arc::new(LoadBasedPolicy::new()));

    assert_eq!(
        score.select(&ws, &ctx).expect("must route").id,
        ws[1].id,
        "ScorePolicy must preserve the load-based snapshot contract"
    );
}
