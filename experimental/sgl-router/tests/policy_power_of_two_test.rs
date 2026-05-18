// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::power_of_two::PowerOfTwoChoicesPolicy;
use sgl_router::policies::{Policy, SelectionContext};
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
    let chosen = p.select(&ws, &ctx).unwrap();
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
        let w = p.select(&workers, &ctx).unwrap();
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
    assert!(p.select(&ws, &ctx).is_none());
}

#[test]
fn single_worker_returns_it() {
    let p = PowerOfTwoChoicesPolicy::new();
    let ws = vec![worker("only")];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    assert_eq!(p.select(&ws, &ctx).unwrap().id.0, "only");
}
