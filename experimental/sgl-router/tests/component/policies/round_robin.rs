// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::round_robin::RoundRobinPolicy;
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::workers::Worker;
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
fn cycles_through_workers() {
    let p = RoundRobinPolicy::new();
    let ws = vec![worker("a"), worker("b"), worker("c")];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    let picks: Vec<_> = (0..6)
        .filter_map(|_| p.select(&ws, &ctx))
        .map(|w| w.id.0.clone())
        .collect();
    assert_eq!(picks, vec!["a", "b", "c", "a", "b", "c"]);
}

#[test]
fn empty_pool_returns_none() {
    let p = RoundRobinPolicy::new();
    let ws: Vec<Arc<Worker>> = vec![];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    assert!(p.select(&ws, &ctx).is_none());
}

#[test]
fn distribution_across_100_calls() {
    let p = RoundRobinPolicy::new();
    let ws = vec![worker("a"), worker("b"), worker("c")];
    let model_id = ModelId("m".into());
    let ctx = SelectionContext::new(&model_id, None);
    let mut counts = std::collections::HashMap::new();
    for _ in 0..99 {
        let w = p.select(&ws, &ctx).unwrap();
        *counts.entry(w.id.0.clone()).or_insert(0) += 1;
    }
    assert_eq!(counts["a"], 33);
    assert_eq!(counts["b"], 33);
    assert_eq!(counts["c"], 33);
}
