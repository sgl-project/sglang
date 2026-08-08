// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! How a vector of per-worker scores becomes one chosen worker.
//!
//! Kept separate from the score itself so a composer can keep its own
//! [`Selector`] while a bare scoring policy inherits [`ARGMAX`].

use crate::workers::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Scores within this much of the best are treated as equal.
///
/// An f32 sum is order-dependent: fusing the same terms in a different order
/// moves a score by ~1e-7, which would otherwise make the winner depend on the
/// order terms were listed on the command line. The band resolves those on
/// load instead.
pub const TIE_EPSILON: f32 = 1e-6;

pub trait Selector: Send + Sync + std::fmt::Debug {
    /// Index into `workers` of the chosen candidate, or `None` when there is
    /// nothing to choose from. `scores[i]` belongs to `workers[i]`.
    fn pick(&self, workers: &[Arc<Worker>], scores: &[f32]) -> Option<usize>;
}

/// Highest score wins; a tie inside [`TIE_EPSILON`] goes to the least-loaded
/// candidate, and a load tie rotates.
///
/// Rotation is why this holds state instead of being a unit struct: it is what
/// stops a fleet of equally-good, equally-idle workers from all being handed
/// the same request. Round-robin's behaviour therefore already lives here,
/// which is why `round_robin` itself does not need to become a scoring policy.
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
        // Excess scores are ignored and missing ones are absent from the band;
        // a policy returning the wrong arity degrades, it does not index OOB.
        let n = workers.len().min(scores.len());
        // NaN is filtered, never compared: every ordered comparison against NaN
        // is false, so a NaN left in the running silently wins or loses
        // depending on which side of the comparison the iteration puts it.
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
            // Every score was NaN or absent. Nobody has a defensible
            // preference, so rank the full set on load rather than dropping the
            // request -- but say so, because a policy that always lands here
            // has silently stopped contributing to routing.
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

    /// The two halves of the tie band. The loads are NOT identical -- `b` is
    /// the loaded one -- and `a` wins BOTH times, which is what makes the
    /// second assert discriminating: the first pick leaves the rotor at 1, so
    /// dropping the load tiebreak rotates to `b` and fails. With the loaded
    /// worker at index 0 it did not fail, because rotation and least-load both
    /// answered `b` and the assert could not tell the two apart.
    #[test]
    fn score_wins_unless_the_gap_is_inside_the_tie_band() {
        let ws = vec![worker("a"), worker("b")];
        let sel = Argmax::default();
        let _loaded = ws[1].load_guard();

        assert_eq!(sel.pick(&ws, &[1.0, 1.0 - 1e-3]), Some(0), "clear winner");
        // `b` scores HIGHER here, but inside the band, so load decides.
        let tie = [1.0 - 5e-7, 1.0];
        assert_eq!(sel.pick(&ws, &tie), Some(0), "tie -> less load");
        assert_eq!(sel.pick(&[], &[]), None, "nothing to choose from");
    }

    /// In EITHER position: a comparison-based max keeps whichever operand the
    /// fold happened to see first when one side is NaN.
    #[test]
    fn nan_never_wins_from_either_position() {
        let ws = vec![worker("a"), worker("b")];
        let sel = Argmax::default();
        assert_eq!(sel.pick(&ws, &[f32::NAN, 0.0]), Some(1));
        assert_eq!(sel.pick(&ws, &[0.0, f32::NAN]), Some(0));
        // All-NaN still routes rather than dropping the request.
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
