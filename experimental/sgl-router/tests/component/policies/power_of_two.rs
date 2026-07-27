// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::load_monitor::AggregateLoad;
use sgl_router::policies::power_of_two::PowerOfTwoChoicesPolicy;
use sgl_router::policies::{Policy, PolicyCandidate, SelectionContext};
use sgl_router::workers::Worker;
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}"),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

/// Converts worker-local fixture counters into engine-load candidates.
fn candidates(workers: &[Arc<Worker>]) -> Vec<PolicyCandidate> {
    workers
        .iter()
        .map(|worker| {
            let load = worker.active_load() as u64;
            PolicyCandidate {
                worker: Arc::clone(worker),
                load: Some(AggregateLoad {
                    total_requests: load,
                    num_total_tokens: load,
                    ..AggregateLoad::default()
                }),
            }
        })
        .collect()
}

#[test]
fn selects_lower_load() {
    let a = worker("a");
    let b = worker("b");
    a.active_requests.store(10, Ordering::Relaxed);
    b.active_requests.store(2, Ordering::Relaxed);
    let p = PowerOfTwoChoicesPolicy::new();
    let ws = vec![a.clone(), b.clone()];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    let chosen = p.select(&candidates(&ws), &ctx).unwrap();
    assert_eq!(chosen.id.0, "b");
}

#[test]
fn distribution_skews_to_lower_load() {
    // With 3 workers and one heavily loaded, the loaded one should win
    // significantly less than 1/3 of selections.
    let workers = vec![worker("a"), worker("b"), worker("c")];
    workers[2].active_requests.store(100, Ordering::Relaxed); // c is loaded

    let p = PowerOfTwoChoicesPolicy::new();
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    let mut counts = std::collections::HashMap::new();
    for _ in 0..1000 {
        let w = p.select(&candidates(&workers), &ctx).unwrap();
        *counts.entry(w.id.0.clone()).or_insert(0) += 1;
    }
    let c_picks = *counts.get("c").unwrap_or(&0);
    assert!(
        c_picks < 200,
        "loaded worker should be picked < 20% of the time, got {c_picks}"
    );
}

#[test]
fn empty_returns_none() {
    let p = PowerOfTwoChoicesPolicy::new();
    let ws: Vec<Arc<Worker>> = vec![];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    assert!(p.select(&candidates(&ws), &ctx).is_none());
}

#[test]
fn single_worker_returns_it() {
    let p = PowerOfTwoChoicesPolicy::new();
    let ws = vec![worker("only")];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    assert_eq!(p.select(&candidates(&ws), &ctx).unwrap().id.0, "only");
}

#[test]
fn all_workers_reachable() {
    // Ensure all workers are selectable under equal load to prevent an
    // off-by-one error from skipping any worker during sampling
    let workers = vec![
        worker("a"),
        worker("b"),
        worker("c"),
        worker("d"),
        worker("e"),
    ];
    let p = PowerOfTwoChoicesPolicy::new();
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..1000 {
        let w = p.select(&candidates(&workers), &ctx).unwrap();
        seen.insert(w.id.0.clone());
    }
    assert_eq!(
        seen.len(),
        workers.len(),
        "every worker should be reachable, saw {seen:?}"
    );
}

/// Prefill power-of-two scoring uses total tokens rather than request count.
#[test]
fn prefill_compares_total_tokens() {
    let left = worker("left");
    let right = worker("right");
    left.set_mode(WorkerMode::Prefill);
    right.set_mode(WorkerMode::Prefill);
    let candidates = vec![
        PolicyCandidate {
            worker: left,
            load: Some(AggregateLoad {
                total_requests: 1,
                num_total_tokens: 100,
                ..AggregateLoad::default()
            }),
        },
        PolicyCandidate {
            worker: Arc::clone(&right),
            load: Some(AggregateLoad {
                total_requests: 10,
                num_total_tokens: 5,
                ..AggregateLoad::default()
            }),
        },
    ];
    let policy = PowerOfTwoChoicesPolicy::new();
    let model = ModelId("m".into());
    let context = SelectionContext::new(&model, None);
    assert_eq!(policy.select(&candidates, &context).unwrap().id, right.id);
}
