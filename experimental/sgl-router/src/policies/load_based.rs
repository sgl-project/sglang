// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

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
    /// `1.0` for the least loaded down to `0.0` for the most, min-max scaled to
    /// the CURRENT fleet -- relative, not absolute, so it cannot saturate:
    /// `1 - load/256` reads a busy fleet as all-`0.0`, tied inside
    /// `TIE_EPSILON`, so the term dies exactly when load matters most.
    ///
    /// Purely a preference: "everybody is busy" is not a reason to refuse to
    /// route, so this term never constrains. Capacity is `--filter`'s job.
    fn scores(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Vec<f32> {
        let loads: Vec<usize> = workers.iter().map(|w| w.active_load()).collect();
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
    use crate::policies::scoring::argmax::TIE_EPSILON;
    use crate::policies::Policy;

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
}
