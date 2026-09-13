// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-local in-flight admission control.

use super::{EligibilityFilter, OnEmpty};
use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::Arc;

/// Rejects workers at the router-local in-flight limit.
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

    /// Do not route to an over-capacity worker.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Hold
    }
}

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

    #[test]
    fn the_cap_is_a_strict_ceiling() {
        assert!(!Overloaded::new(3).needs_load_snapshot());
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

        let some: Vec<Box<dyn EligibilityFilter>> = vec![Box::new(Overloaded::new(6))];
        let out = admit(refs(&some), &ws, &ctx).expect("a is under the cap");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].url, ws[0].url);
    }

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
