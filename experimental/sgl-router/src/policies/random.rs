// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{scoring::ScoringPolicy, SelectionContext};
use crate::workers::Worker;
use rand::Rng;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl ScoringPolicy for RandomPolicy {
    /// Argmax of n iid uniforms IS a uniform choice: exactly the old `choose`.
    /// Never constrains: a coin toss is not an eligibility rule.
    fn scores(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..workers.len()).map(|_| rng.gen()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
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

    /// Distributional: `select()` is not pure. Marginals alone are satisfied by
    /// a ROTATION -- what a constant `scores()` becomes under ARGMAX's rotating
    /// tiebreak -- so REPEATS share the band: P(pick==prev) is 1/N iid, 0 rotating.
    #[test]
    fn picks_uniformly_over_20k_draws_and_repeats_at_the_iid_rate() {
        const MEAN: f64 = 5_000.0; // 20_000 draws over 4 workers; repeats too
        const BAND: f64 = 5.0 * 61.237_244; // 5 sigma, sqrt(20_000 / 4 * 3 / 4)
        let (policy, model) = (RandomPolicy::new(), ModelId("tiny".into()));
        let ctx = SelectionContext::new(&model, None);
        let ws: Vec<Arc<Worker>> = (0..4).map(|i| worker(&format!("w{i}"))).collect();
        assert!(policy.select(&[], &ctx).is_none(), "empty fleet");
        let (mut counts, mut repeats, mut prev) = ([0usize; 4], 0usize, None);
        for _ in 0..20_000 {
            let got = policy.select(&ws, &ctx).expect("non-empty fleet");
            let i = ws.iter().position(|w| w.id == got.id).expect("a candidate");
            counts[i] += 1;
            repeats += usize::from(prev.replace(i) == Some(i));
        }
        for n in counts.iter().chain([&repeats]) {
            let dev = (*n as f64 - MEAN).abs(); // `> 0` is no-starvation
            assert!(*n > 0 && dev < BAND, "{counts:?} {repeats}");
        }
    }
}
