// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::Arc;

/// Deterministic load-based policy.
///
/// Chooses the candidate with the lowest current `Worker::active_load`.
/// Ties follow the candidate slice order, which is registry-dependent.
#[derive(Debug, Default)]
pub struct LoadBasedPolicy;

impl LoadBasedPolicy {
    pub fn new() -> Self {
        Self
    }

    pub fn pick_min_load(workers: &[Arc<Worker>]) -> Option<Arc<Worker>> {
        workers
            .iter()
            .min_by_key(|w| w.active_load())
            .map(Arc::clone)
    }
}

impl Policy for LoadBasedPolicy {
    fn select(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        Self::pick_min_load(workers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
            min_priority: None,
        }))
    }

    #[test]
    fn empty_returns_none() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert!(policy.select(&[], &ctx).is_none());
    }

    #[test]
    fn picks_lowest_active_load() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _g0 = w0.load_guard();
        assert_eq!(
            policy.select(&[w0, Arc::clone(&w1)], &ctx).unwrap().id,
            w1.id
        );
    }
}
