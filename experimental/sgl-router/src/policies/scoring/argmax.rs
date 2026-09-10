// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Selects one worker from per-worker scores.

use crate::workers::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Scores within this distance of the best are tied.
pub const TIE_EPSILON: f32 = 1e-6;

pub trait Selector: Send + Sync + std::fmt::Debug {
    /// Index into `workers` of the chosen candidate, or `None` when there is
    /// nothing to choose from. `scores[i]` belongs to `workers[i]`.
    fn pick(&self, workers: &[Arc<Worker>], scores: &[f32]) -> Option<usize>;
}

/// Highest score wins; ties choose the least-loaded candidate and rotate.
#[derive(Debug, Default)]
pub struct Argmax {
    rotor: AtomicUsize,
}

/// The default selector, shared by every scoring policy that does not override
/// [`super::ScoringPolicy::selector`].
pub static ARGMAX: Argmax = Argmax {
    rotor: AtomicUsize::new(0),
};

impl Selector for Argmax {
    fn pick(&self, workers: &[Arc<Worker>], scores: &[f32]) -> Option<usize> {
        if workers.is_empty() {
            return None;
        }
        let n = workers.len().min(scores.len());
        let best = (0..n)
            .map(|i| scores[i])
            .filter(|s| !s.is_nan())
            .fold(None::<f32>, |acc, s| Some(acc.map_or(s, |b| b.max(s))));
        let mut band: Vec<usize> = match best {
            Some(b) => (0..n)
                .filter(|&i| !scores[i].is_nan() && scores[i] >= b - TIE_EPSILON)
                .collect(),
            None => Vec::new(),
        };
        if band.is_empty() {
            tracing::debug!(
                n_workers = workers.len(),
                n_scores = scores.len(),
                "no usable score; falling back to load + rotation",
            );
            band = (0..workers.len()).collect();
        }
        let min_load = band.iter().map(|&i| workers[i].active_load()).min()?;
        let tied: Vec<usize> = band
            .into_iter()
            .filter(|&i| workers[i].active_load() == min_load)
            .collect();
        let k = self.rotor.fetch_add(1, Ordering::Relaxed) % tied.len();
        Some(tied[k])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use std::collections::HashSet;

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
    fn score_wins_unless_the_gap_is_inside_the_tie_band() {
        let ws = vec![worker("a"), worker("b")];
        let sel = Argmax::default();
        let _loaded = ws[1].load_guard();

        assert_eq!(sel.pick(&ws, &[1.0, 1.0 - 1e-3]), Some(0), "clear winner");
        let tie = [1.0 - 5e-7, 1.0];
        assert_eq!(sel.pick(&ws, &tie), Some(0), "tie -> less load");
        assert_eq!(sel.pick(&[], &[]), None, "nothing to choose from");
    }

    #[test]
    fn nan_never_wins_from_either_position() {
        let ws = vec![worker("a"), worker("b")];
        let sel = Argmax::default();
        assert_eq!(sel.pick(&ws, &[f32::NAN, 0.0]), Some(1));
        assert_eq!(sel.pick(&ws, &[0.0, f32::NAN]), Some(0));
        assert!(sel.pick(&ws, &[f32::NAN, f32::NAN]).is_some());
    }

    #[test]
    fn a_total_tie_rotates_over_every_candidate() {
        let ws = vec![worker("a"), worker("b"), worker("c")];
        let sel = Argmax::default();
        let picks: HashSet<usize> = (0..3).filter_map(|_| sel.pick(&ws, &[1.0; 3])).collect();
        assert_eq!(
            picks.len(),
            3,
            "three tied picks must cover all three: {picks:?}"
        );
    }
}
