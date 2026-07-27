// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, PolicyCandidate, SelectionContext};
use crate::workers::Worker;
use rand::seq::SliceRandom;
use std::sync::Arc;

/// Engine-reported least-load policy.
///
/// Chooses the candidate with the lowest reported `running + waiting` count.
/// Ties are randomized to avoid stable registry-order concentration.
#[derive(Debug, Default)]
pub struct LoadBasedPolicy;

impl LoadBasedPolicy {
    /// Constructs a stateless load-based policy.
    pub fn new() -> Self {
        Self
    }

    /// Selects a minimum-request candidate with random tie-breaking.
    pub fn pick_min_load(candidates: &[PolicyCandidate]) -> Option<Arc<Worker>> {
        let minimum = candidates
            .iter()
            .filter_map(|candidate| candidate.load.as_ref().map(|load| load.total_requests))
            .min()?;
        candidates
            .iter()
            .filter(|candidate| {
                candidate.load.as_ref().map(|load| load.total_requests) == Some(minimum)
            })
            .collect::<Vec<_>>()
            .choose(&mut rand::thread_rng())
            .map(|candidate| Arc::clone(&candidate.worker))
    }
}

impl Policy for LoadBasedPolicy {
    /// Selects the least-loaded worker using engine-reported request counts.
    fn select(
        &self,
        candidates: &[PolicyCandidate],
        _ctx: &SelectionContext<'_>,
    ) -> Option<Arc<Worker>> {
        Self::pick_min_load(candidates)
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
        }))
    }

    #[test]
    fn empty_returns_none() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert!(policy
            .select(&crate::policies::test_policy_candidates(&[]), &ctx)
            .is_none());
    }

    #[test]
    fn picks_lowest_active_load() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _g0 = w0.load_guard();
        let candidates = crate::policies::test_policy_candidates(&[w0, Arc::clone(&w1)]);
        assert_eq!(policy.select(&candidates, &ctx).unwrap().id, w1.id);
    }
}
