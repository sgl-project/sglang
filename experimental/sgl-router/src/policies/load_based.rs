// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::load_scoring::EngineLoadScorer;
use crate::policies::{Policy, PolicyCandidate, SelectionContext};
use crate::workers::Worker;
use rand::Rng;
use std::sync::Arc;

/// Engine-pressure-weighted routing policy.
///
/// A candidate's routing weight is `1 - pressure`, where pressure combines
/// request saturation, uncached-token queue delay, throughput, and hottest-rank
/// KV usage. Higher-pressure engines therefore receive proportionally fewer new
/// requests instead of losing every decision because of one small load delta.
#[derive(Debug, Default)]
pub struct LoadBasedPolicy;

impl LoadBasedPolicy {
    pub fn new() -> Self {
        Self
    }

    /// Selects a candidate proportionally to its engine-pressure weight.
    ///
    /// Candidates without an Engine report retain the legacy least-active
    /// behavior. Production LoadMonitor snapshots are all-reported or filtered,
    /// but this fallback also keeps direct policy callers backward compatible.
    pub fn pick_weighted(candidates: &[PolicyCandidate]) -> Option<Arc<Worker>> {
        if candidates.iter().all(|candidate| candidate.load.is_none()) {
            return candidates
                .iter()
                .min_by_key(|candidate| candidate.worker.active_load())
                .map(|candidate| Arc::clone(&candidate.worker));
        }
        Self::pick_weighted_with(candidates, &mut rand::thread_rng())
    }

    fn pick_weighted_with<R: Rng + ?Sized>(
        candidates: &[PolicyCandidate],
        rng: &mut R,
    ) -> Option<Arc<Worker>> {
        let total_weight: f64 = candidates
            .iter()
            .map(candidate_weight)
            .filter(|weight| weight.is_finite() && *weight > 0.0)
            .sum();
        if !total_weight.is_finite() || total_weight <= f64::EPSILON {
            return None;
        }

        let mut draw = rng.gen_range(0.0..total_weight);
        let mut last_positive = None;
        for candidate in candidates {
            let weight = candidate_weight(candidate);
            if !weight.is_finite() || weight <= 0.0 {
                continue;
            }
            last_positive = Some(candidate);
            if draw < weight {
                return Some(Arc::clone(&candidate.worker));
            }
            draw -= weight;
        }

        // Floating-point subtraction can leave `draw` a few ulps above zero.
        last_positive.map(|candidate| Arc::clone(&candidate.worker))
    }
}

fn candidate_weight(candidate: &PolicyCandidate) -> f64 {
    candidate
        .load
        .as_ref()
        .map(EngineLoadScorer::routing_weight)
        .unwrap_or(1.0)
}

impl Policy for LoadBasedPolicy {
    fn select(
        &self,
        candidates: &[PolicyCandidate],
        _ctx: &SelectionContext<'_>,
    ) -> Option<Arc<Worker>> {
        Self::pick_weighted(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::load_monitor::AggregateLoad;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    fn candidate(worker: Arc<Worker>, pressure: Option<f64>) -> PolicyCandidate {
        PolicyCandidate {
            worker,
            load: pressure.map(|max_rank_token_usage| AggregateLoad {
                max_running_requests: 100,
                available_slots: 100,
                max_rank_token_usage,
                ..AggregateLoad::default()
            }),
        }
    }

    #[test]
    fn empty_returns_none() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert!(policy.select(&[], &ctx).is_none());
    }

    #[test]
    fn missing_metrics_preserve_least_active_routing() {
        let busy = worker("busy");
        let idle = worker("idle");
        let _busy_guard = busy.load_guard();
        let candidates = vec![candidate(busy, None), candidate(Arc::clone(&idle), None)];

        assert_eq!(
            LoadBasedPolicy::pick_weighted(&candidates).unwrap().id,
            idle.id
        );
    }

    #[test]
    fn pressure_reduces_observed_selection_frequency() {
        let cool = worker("cool");
        let hot = worker("hot");
        let candidates = vec![
            candidate(Arc::clone(&cool), Some(0.1)),
            candidate(Arc::clone(&hot), Some(0.9)),
        ];
        let mut rng = StdRng::seed_from_u64(7);
        let mut cool_count = 0;
        let mut hot_count = 0;
        for _ in 0..10_000 {
            match LoadBasedPolicy::pick_weighted_with(&candidates, &mut rng)
                .unwrap()
                .id
                .0
                .as_str()
            {
                "cool" => cool_count += 1,
                "hot" => hot_count += 1,
                other => panic!("unexpected worker {other}"),
            }
        }
        assert!(
            cool_count > hot_count * 7,
            "cool={cool_count}, hot={hot_count}"
        );
    }

    #[test]
    fn fully_pressured_candidate_is_never_selected() {
        let full = worker("full");
        let available = worker("available");
        let candidates = vec![
            candidate(full, Some(1.0)),
            candidate(Arc::clone(&available), Some(0.0)),
        ];
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..100 {
            assert_eq!(
                LoadBasedPolicy::pick_weighted_with(&candidates, &mut rng)
                    .unwrap()
                    .id,
                available.id
            );
        }
    }
}
