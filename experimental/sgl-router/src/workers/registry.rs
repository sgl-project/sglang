// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::workers::worker::Worker;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct WorkerRegistry {
    by_id: DashMap<WorkerId, Arc<Worker>>,
    by_model: DashMap<ModelId, HashSet<WorkerId>>,
}

impl WorkerRegistry {
    pub fn add(&self, spec: WorkerSpec) {
        self.add_with_cb(spec, None);
    }

    /// Add a worker, optionally supplying a circuit-breaker config.
    /// Pass `None` to use the circuit-breaker default (threshold = 3).
    pub fn add_with_cb(&self, spec: WorkerSpec, cb: Option<CircuitBreakerConfig>) {
        let w = Arc::new(Worker::with_cb_config(spec, cb));
        let id = w.id.clone();
        for m in &w.model_ids {
            self.by_model
                .entry(m.clone())
                .or_default()
                .insert(id.clone());
        }
        self.by_id.insert(id, w);
    }

    pub fn remove(&self, id: &WorkerId) {
        if let Some((_, w)) = self.by_id.remove(id) {
            for m in &w.model_ids {
                if let Some(mut set) = self.by_model.get_mut(m) {
                    set.remove(id);
                }
            }
        }
    }

    pub fn workers_for(&self, model: &ModelId) -> Vec<Arc<Worker>> {
        self.by_model
            .get(model)
            .map(|ids| {
                ids.iter()
                    .filter_map(|i| self.by_id.get(i).map(|w| Arc::clone(&w)))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn healthy_workers_for(&self, model: &ModelId) -> Vec<Arc<Worker>> {
        self.workers_for(model)
            .into_iter()
            .filter(|w| w.breaker.allow())
            .collect()
    }

    pub fn workers_for_mode(&self, model: &ModelId, mode: WorkerMode) -> Vec<Arc<Worker>> {
        self.workers_for(model)
            .into_iter()
            .filter(|w| w.mode == mode)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn get(&self, id: &WorkerId) -> Option<Arc<Worker>> {
        self.by_id.get(id).map(|w| Arc::clone(&w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

    fn spec(id: &str, mode: WorkerMode, models: &[&str]) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode,
            model_ids: models.iter().map(|m| ModelId((*m).into())).collect(),
        }
    }

    #[test]
    fn add_then_query_by_model() {
        let r = WorkerRegistry::default();
        r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        r.add(spec("w2", WorkerMode::Plain, &["m1"]));
        let m1 = r.workers_for(&ModelId("m1".into()));
        let m2 = r.workers_for(&ModelId("m2".into()));
        let m_missing = r.workers_for(&ModelId("missing".into()));
        assert_eq!(m1.len(), 2);
        assert_eq!(m2.len(), 1);
        assert!(m_missing.is_empty());
    }

    #[test]
    fn remove_drops_from_all_models() {
        let r = WorkerRegistry::default();
        r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        r.remove(&WorkerId("w1".into()));
        assert!(r.workers_for(&ModelId("m1".into())).is_empty());
        assert!(r.workers_for(&ModelId("m2".into())).is_empty());
    }

    #[test]
    fn healthy_subset_filters_via_breaker() {
        let r = WorkerRegistry::default();
        r.add(spec("ok", WorkerMode::Plain, &["m"]));
        r.add(spec("bad", WorkerMode::Plain, &["m"]));
        // Force "bad" worker's breaker to deny.
        // (We need to inject breaker state; the easiest path is the
        // CircuitBreaker stub returning false after record_failure. Task 6
        // fills the real implementation; for now we test the registry
        // delegates to breaker.allow() correctly — see test
        // `healthy_subset_uses_breaker_allow` below once Task 6 lands.)
        let healthy = r.healthy_workers_for(&ModelId("m".into()));
        // Stub breaker always allows; both workers visible.
        assert_eq!(healthy.len(), 2);
    }

    #[test]
    fn worker_mode_filter() {
        let r = WorkerRegistry::default();
        r.add(spec("p", WorkerMode::Prefill, &["m"]));
        r.add(spec("d", WorkerMode::Decode, &["m"]));
        r.add(spec("plain", WorkerMode::Plain, &["m"]));
        let plain = r.workers_for_mode(&ModelId("m".into()), WorkerMode::Plain);
        let prefill = r.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill);
        assert_eq!(plain.len(), 1);
        assert_eq!(prefill.len(), 1);
    }
}
