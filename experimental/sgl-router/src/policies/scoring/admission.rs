// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Capacity as a hard constraint.
//!
//! The distinction this file exists to make: `load_based` says a busy worker is
//! a worse choice, and admission says an over-capacity worker is not a choice.
//! A weight cannot express the second — under enough pressure the whole fleet
//! scores badly and the least-bad worker still wins, which is exactly the
//! request you did not want to send.

use super::{EligibilityFilter, OnEmpty};
use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::Arc;

/// Rejects any worker already carrying `max_in_flight` router-dispatched
/// requests.
///
/// Router-local in-flight count and not an engine-reported queue depth: it is
/// the one signal that needs no worker cooperation and cannot go stale. It is
/// also per-router, so with several router replicas the effective cap is this
/// times the replica count — worth saying out loud, because the flag reads like
/// a global.
#[derive(Debug)]
pub struct Overloaded {
    max_in_flight: usize,
}

impl Overloaded {
    pub fn new(max_in_flight: usize) -> Self {
        Self { max_in_flight }
    }
}

impl EligibilityFilter for Overloaded {
    fn keep(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Vec<bool> {
        (workers.iter())
            .map(|w| w.active_load() < self.max_in_flight)
            .collect()
    }

    /// [`OnEmpty::Hold`], the whole point: when every worker is over its cap
    /// the answer is "not here", not "the least over-capacity one". This is
    /// also what makes the rule safe to list in any position — a `Hold` filter
    /// never yields to a lower-priority one.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Hold
    }
}

/// Usable as `--policy overloaded` too, which is only meaningful as a smoke
/// test: with nothing to rank by, the selector's load tiebreak decides.
impl Policy for Overloaded {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let eligible: Vec<Arc<Worker>> = (workers.iter())
            .zip(self.keep(workers, ctx))
            .filter(|(_, ok)| *ok)
            .map(|(w, _)| Arc::clone(w))
            .collect();
        eligible
            .iter()
            .min_by_key(|w| w.active_load())
            .map(Arc::clone)
    }

    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::scoring::{admit, refs};

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    /// The cap is a strict ceiling, and the two neighbours of it are what pin
    /// that: `at` sits exactly on the cap and must be OUT, `under` one below
    /// and must be IN. An off-by-one either way flips one of them.
    #[test]
    fn the_cap_is_a_strict_ceiling() {
        let ws = vec![worker("idle"), worker("under"), worker("at")];
        let _under: Vec<_> = (0..2).map(|_| ws[1].load_guard()).collect();
        let _at: Vec<_> = (0..3).map(|_| ws[2].load_guard()).collect();

        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert_eq!(
            Overloaded::new(3).keep(&ws, &ctx),
            vec![true, true, false],
            "load 3 against a cap of 3 is over",
        );
    }

    /// Everyone over the cap must NOT route. Paired with a control that does
    /// route, because an `admit` that always answered `None` would satisfy the
    /// first assert on its own.
    #[test]
    fn a_full_fleet_refuses_rather_than_picking_the_least_bad() {
        let ws = vec![worker("a"), worker("b")];
        let _a: Vec<_> = (0..5).map(|_| ws[0].load_guard()).collect();
        let _b: Vec<_> = (0..9).map(|_| ws[1].load_guard()).collect();

        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let full: Vec<Box<dyn EligibilityFilter>> = vec![Box::new(Overloaded::new(4))];
        assert!(
            admit(refs(&full), &ws, &ctx).is_none(),
            "both over the cap, and the filter Holds",
        );

        // `a` is under a cap of 6, so the same code path does route -- and to
        // `a`, not to the emptier-looking answer a broken filter would give.
        let some: Vec<Box<dyn EligibilityFilter>> = vec![Box::new(Overloaded::new(6))];
        let out = admit(refs(&some), &ws, &ctx).expect("a is under the cap");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].url, ws[0].url);
    }

    /// `Hold` must not be talked out of it by a later, lower-priority filter
    /// that would have admitted somebody. This is the property that makes the
    /// flag order-insensitive, which is the only reason it is safe to ship.
    #[test]
    fn hold_does_not_yield_to_a_later_filter() {
        #[derive(Debug)]
        struct AdmitAll;
        impl EligibilityFilter for AdmitAll {
            fn keep(&self, ws: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<bool> {
                vec![true; ws.len()]
            }
        }
        let ws = vec![worker("a")];
        let _busy: Vec<_> = (0..9).map(|_| ws[0].load_guard()).collect();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let chain: Vec<Box<dyn EligibilityFilter>> =
            vec![Box::new(Overloaded::new(2)), Box::new(AdmitAll)];
        assert!(admit(refs(&chain), &ws, &ctx).is_none());
    }
}
