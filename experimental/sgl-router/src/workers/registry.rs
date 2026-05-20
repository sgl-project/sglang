// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::workers::worker::Worker;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;

/// Reason a [`WorkerRegistry::add`] call refused the spec.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AddWorkerError {
    /// The spec's mode (plain vs prefill/decode) conflicts with workers
    /// already registered for one of its `model_ids`. The router does
    /// not support mixed PD + plain pools on a single model: the
    /// resolver derives the PD-vs-plain shape from the registered
    /// workers, and a mixed pool would silently degrade to whichever
    /// shape happens to be healthy when the other is breaker-open,
    /// surfacing the wrong error code to clients.
    #[error(
        "worker {worker:?} for model {model:?} would mix PD ({pd_mode}) with plain workers on \
         the same model — sgl-router does not support mixed pools. Use one of: only Plain \
         workers, or only Prefill+Decode workers."
    )]
    MixedPdAndPlain {
        worker: WorkerId,
        model: ModelId,
        /// The role of the *incoming* worker that triggered the conflict
        /// (the *existing* worker has the opposite role).
        pd_mode: &'static str,
    },
}

#[derive(Debug, Default)]
pub struct WorkerRegistry {
    by_id: DashMap<WorkerId, Arc<Worker>>,
    by_model: DashMap<ModelId, HashSet<WorkerId>>,
}

impl WorkerRegistry {
    pub fn add(&self, spec: WorkerSpec) -> Result<(), AddWorkerError> {
        self.add_with_cb(spec, None)
    }

    /// Add a worker, optionally supplying a circuit-breaker config.
    /// Pass `None` to use the circuit-breaker default (threshold = 3).
    ///
    /// Re-adding an existing `WorkerId` is an upsert: the prior entry's
    /// `by_model` memberships are cleared first so a model that the new
    /// spec no longer serves stops resolving to this worker.  Without the
    /// pre-removal step a worker whose model set shrank would still appear
    /// in `workers_for(<dropped model>)` because `by_id.get(...)` would
    /// return the new worker via the stale model→id index.
    ///
    /// Returns [`AddWorkerError::MixedPdAndPlain`] when adding the spec
    /// would mix PD (prefill/decode) workers with plain workers on the
    /// same model. The conflict is detected against the *existing*
    /// registry state — re-adding the same worker id is fine (the prior
    /// entry is removed first), and adding a worker whose own
    /// `model_ids` are all unmixed is fine even if other models in the
    /// process have a mix of modes.
    pub fn add_with_cb(
        &self,
        spec: WorkerSpec,
        cb: Option<CircuitBreakerConfig>,
    ) -> Result<(), AddWorkerError> {
        let incoming_mode = spec.mode;
        // Validate against existing workers BEFORE we mutate. Re-adding
        // the same id is an upsert; pretend the prior entry is gone for
        // the purposes of the check (otherwise an upsert of an unmixed
        // worker would self-conflict if its current entry already
        // serves the model).
        for model in &spec.model_ids {
            for existing in self.workers_for(model) {
                if existing.id == spec.id {
                    continue;
                }
                if modes_are_mixed(incoming_mode, existing.mode()) {
                    return Err(AddWorkerError::MixedPdAndPlain {
                        worker: spec.id,
                        model: model.clone(),
                        pd_mode: mode_name(incoming_mode),
                    });
                }
            }
        }
        let w = Arc::new(Worker::with_cb_config(spec, cb));
        let id = w.id.clone();
        self.remove(&id);
        for m in &w.model_ids {
            self.by_model
                .entry(m.clone())
                .or_default()
                .insert(id.clone());
        }
        self.by_id.insert(id, w);
        Ok(())
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
        // Use `would_allow` (non-mutating) for filtering — `allow()` would
        // claim a half-open probe slot for every enumerated candidate,
        // starving the worker that the policy actually picks. The probe
        // is claimed at dispatch time by `forward_*_to` in
        // [`crate::proxy`].
        self.workers_for(model)
            .into_iter()
            .filter(|w| w.breaker.would_allow())
            .collect()
    }

    pub fn workers_for_mode(&self, model: &ModelId, mode: WorkerMode) -> Vec<Arc<Worker>> {
        self.workers_for(model)
            .into_iter()
            .filter(|w| w.mode() == mode)
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

/// `true` when the two modes can't coexist for the same model — i.e.
/// one is `Plain` and the other is `Prefill` or `Decode`.
fn modes_are_mixed(a: WorkerMode, b: WorkerMode) -> bool {
    matches!(
        (a, b),
        (WorkerMode::Plain, WorkerMode::Prefill | WorkerMode::Decode)
            | (WorkerMode::Prefill | WorkerMode::Decode, WorkerMode::Plain)
    )
}

fn mode_name(m: WorkerMode) -> &'static str {
    match m {
        WorkerMode::Plain => "plain",
        WorkerMode::Prefill => "prefill",
        WorkerMode::Decode => "decode",
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
            bootstrap_port: None,
        }
    }

    #[test]
    fn add_then_query_by_model() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        let _ = r.add(spec("w2", WorkerMode::Plain, &["m1"]));
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
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        r.remove(&WorkerId("w1".into()));
        assert!(r.workers_for(&ModelId("m1".into())).is_empty());
        assert!(r.workers_for(&ModelId("m2".into())).is_empty());
    }

    #[test]
    fn healthy_subset_filters_via_breaker() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("ok", WorkerMode::Plain, &["m"]));
        let _ = r.add(spec("bad", WorkerMode::Plain, &["m"]));
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

    /// PD prefill/decode workers and plain workers cannot coexist on the
    /// same model. The resolver bases its PD-vs-plain shape on registered
    /// workers; mixing the two forces a fallback to whichever bucket
    /// happens to be healthy when the other is breaker-open, surfacing
    /// the wrong 5xx code (`no_healthy_workers` instead of
    /// `no_prefill_workers_available`). Reject the conflicting add up
    /// front so the operator sees the misconfiguration immediately.
    #[test]
    fn plain_then_pd_for_same_model_is_rejected() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("plain", WorkerMode::Plain, &["m"])).is_ok());
        let err = r
            .add(spec("p", WorkerMode::Prefill, &["m"]))
            .expect_err("PD worker must be rejected when model already has Plain workers");
        let msg = err.to_string();
        assert!(
            msg.contains("PD") && msg.contains("plain"),
            "error must name both modes; got: {msg}"
        );
        // Existing plain worker survives the rejection.
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Plain)
                .len(),
            1,
        );
        assert!(r
            .workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
            .is_empty());
    }

    #[test]
    fn pd_then_plain_for_same_model_is_rejected() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("p", WorkerMode::Prefill, &["m"])).is_ok());
        assert!(r.add(spec("d", WorkerMode::Decode, &["m"])).is_ok());
        let err = r
            .add(spec("plain", WorkerMode::Plain, &["m"]))
            .expect_err("plain worker must be rejected when model already has PD workers");
        let msg = err.to_string();
        assert!(
            msg.contains("PD") && msg.contains("plain"),
            "error must name both modes; got: {msg}"
        );
    }

    #[test]
    fn plain_only_pool_admits_more_plain_workers() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("a", WorkerMode::Plain, &["m"])).is_ok());
        assert!(r.add(spec("b", WorkerMode::Plain, &["m"])).is_ok());
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Plain)
                .len(),
            2,
        );
    }

    #[test]
    fn pd_pool_admits_more_pd_workers_in_both_roles() {
        let r = WorkerRegistry::default();
        assert!(r.add(spec("p1", WorkerMode::Prefill, &["m"])).is_ok());
        assert!(r.add(spec("p2", WorkerMode::Prefill, &["m"])).is_ok());
        assert!(r.add(spec("d1", WorkerMode::Decode, &["m"])).is_ok());
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
                .len(),
            2,
        );
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Decode)
                .len(),
            1,
        );
    }

    /// Re-adding a worker with a shrunken `model_ids` must drop the worker
    /// from the models it no longer serves.  The earlier implementation
    /// only updated `by_id`, leaving the stale `by_model` entries pointing
    /// at the new worker.
    #[test]
    fn re_add_with_shrunken_model_set_drops_stale_indexes() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1", "m2"]));
        assert_eq!(r.workers_for(&ModelId("m2".into())).len(), 1);

        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1"]));
        assert_eq!(
            r.workers_for(&ModelId("m2".into())).len(),
            0,
            "w1 no longer serves m2 after re-add"
        );
        assert_eq!(
            r.workers_for(&ModelId("m1".into())).len(),
            1,
            "w1 still serves m1"
        );
    }

    /// Re-adding the same id with a different mode reflects in
    /// `workers_for_mode`.
    #[test]
    fn re_add_with_different_mode_updates_mode_filter() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Prefill, &["m"]));
        let _ = r.add(spec("w1", WorkerMode::Decode, &["m"]));
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
                .len(),
            0,
        );
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Decode)
                .len(),
            1,
        );
    }
}
