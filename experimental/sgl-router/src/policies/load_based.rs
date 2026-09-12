// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::admission::FreshLoadLookup;
use crate::policies::scoring::ScoringPolicy;
use crate::policies::SelectionContext;
use crate::workers::Worker;
use std::sync::Arc;

/// Prefers the least-loaded candidate; `select()` is the blanket impl's.
#[derive(Debug, Default)]
pub struct LoadBasedPolicy;

impl LoadBasedPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl ScoringPolicy for LoadBasedPolicy {
    fn needs_load_snapshot(&self) -> bool {
        true
    }

    /// `1.0` for the least loaded down to `0.0` for the most, min-max scaled to
    /// the CURRENT fleet -- relative, not absolute, so it cannot saturate:
    /// `1 - load/256` reads a busy fleet as all-`0.0`, tied inside
    /// `TIE_EPSILON`, so the term dies exactly when load matters most.
    ///
    /// Purely a preference: "everybody is busy" is not a reason to refuse to
    /// route, so this term never constrains. Capacity is `--filter`'s job.
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let lookup = FreshLoadLookup::new(ctx.load_snapshot(), workers.iter());
        let loads: Vec<usize> = workers.iter().map(|w| lookup.score_load(w)).collect();
        let lo = loads.iter().min().copied().unwrap_or(0);
        let span = (loads.iter().max().copied().unwrap_or(0) - lo) as f32;
        // `max(1.0)` is exact: a zero span means every `l - lo` is zero too.
        let score = |l: usize| 1.0 - (l - lo) as f32 / span.max(1.0);
        loads.into_iter().map(score).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::engine_load::{EngineLoadSnapshot, EngineWorkerLoad};
    use crate::policies::scoring::argmax::TIE_EPSILON;
    use crate::policies::Policy;
    use std::collections::HashMap;
    use std::time::Instant;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    #[test] // upstream's, retargeted from `pick_min_load` onto blanket `select`
    fn empty_returns_none() {
        let m = ModelId("tiny".into());
        let ctx = SelectionContext::new(&m, None);
        assert!(LoadBasedPolicy::new().select(&[], &ctx).is_none());
    }

    /// `select()` alone CANNOT detect a broken score: ARGMAX breaks a tie on
    /// load, so a constant `scores()` still lands on the minimum and that arm
    /// passes for the wrong reason. Ranking is therefore asserted on the vector
    /// itself, strictly outside `TIE_EPSILON` so a saturating curve cannot hide
    /// in the tie band -- what `300,900` is for. NaN needs its own arm because
    /// no ORDERING sees it: it makes every comparison false, which on `0,0` is
    /// the expected answer. Upstream's `picks_lowest_active_load` goes under
    /// rule 4 -- the unique-minimum 2-worker case, which `0,1` subsumes.
    #[test]
    fn scores_rank_strictly_by_load_and_the_choice_lands_on_the_minimum() {
        let model = ModelId("tiny".into());
        let (ctx, p) = (SelectionContext::new(&model, None), LoadBasedPolicy::new());
        for spec in ["0,1", "2,1,0", "1,0,1", "0,0", "5,2,9,2", "300,900"] {
            let loads: Vec<usize> = spec.split(',').map(|s| s.parse().unwrap()).collect();
            let ws: Vec<Arc<Worker>> = (0..loads.len()).map(|i| worker(&format!("w{i}"))).collect();
            let _held: Vec<_> = (ws.iter().zip(&loads))
                .flat_map(|(w, n)| (0..*n).map(move |_| w.load_guard()))
                .collect();
            let scores = p.scores(&ws, &ctx);
            for (i, j) in (0..loads.len()).flat_map(|i| (0..loads.len()).map(move |j| (i, j))) {
                let ok = (scores[i] > scores[j] + TIE_EPSILON, scores[i].is_nan());
                assert_eq!(ok, (loads[i] < loads[j], false), "{spec} scored {scores:?}");
            }
            let got = p.select(&ws, &ctx).expect("non-empty").active_load();
            assert_eq!(got, *loads.iter().min().expect("non-empty"), "{spec}");
        }
    }

    #[test]
    fn request_snapshot_overrides_later_router_active_load() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        // After the request snapshot, local counters say w0 is lighter.
        // The policy must still preserve the frozen Engine Load ordering.
        let _after_snapshot: Vec<_> = (0..10).map(|_| w1.load_guard()).collect();
        let snapshot = EngineLoadSnapshot::from_workers(
            23,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 50,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at: Instant::now(),
                    },
                ),
                (
                    w1.url.clone(),
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
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "load-based scoring must use the request snapshot before local active-load"
        );
    }

    #[test]
    fn recent_dispatches_after_snapshot_change_load_based_choice() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let captured_at = Instant::now();
        let snapshot = EngineLoadSnapshot::from_workers(
            37,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 0,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
                (
                    w1.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
            ]),
        );
        let _after_snapshot = [w0.timestamped_load_guard(), w0.timestamped_load_guard()];
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "dispatches newer than Engine Load must correct its queue depth"
        );
    }

    #[test]
    fn dispatches_before_snapshot_are_not_double_counted() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _before_snapshot = [w0.timestamped_load_guard(), w0.timestamped_load_guard()];
        std::thread::sleep(std::time::Duration::from_millis(5));
        let captured_at = Instant::now();
        let snapshot = EngineLoadSnapshot::from_workers(
            41,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 0,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
                (
                    w1.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
            ]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w0.id,
            "slots already covered by the snapshot must not be added again"
        );
    }

    #[test]
    fn incomplete_snapshot_uses_frozen_local_active_fallback() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _local_load = [w0.load_guard(), w0.load_guard()];
        let snapshot = EngineLoadSnapshot::from_workers(
            43,
            HashMap::from([(
                w0.url.clone(),
                EngineWorkerLoad {
                    num_running_reqs: 0,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 0,
                    captured_at: Instant::now(),
                },
            )]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "a partial Engine Load set must not mix engine and local gauges"
        );
    }
}
