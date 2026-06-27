// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::workers::worker::Worker;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

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
    /// Serializes the validate→insert section of `add_with_cb` so the
    /// `MixedPdAndPlain` check is atomic with the subsequent write. Two
    /// concurrent registrations from `manager::register_one` for the
    /// same model with conflicting modes could otherwise both observe
    /// an empty pool and both insert, leaving the registry in a mixed
    /// state — the exact corruption the check is meant to prevent.
    /// Reads (`workers_for`, `get`, `len`, …) stay lock-free against
    /// the underlying DashMaps; only writes through `add_with_cb` /
    /// `remove` take this lock so contention is bounded by registry
    /// mutation rate (worker-discovery events), not request rate.
    write: Mutex<()>,
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
    ///
    /// On rejection the registry is **not** mutated. If the rejected
    /// spec carries an id that already has an entry, the prior entry
    /// stays put — it's the caller's responsibility to decide whether
    /// to evict it (and, importantly, to also clean up sidecar state
    /// in `KvEventIndex` / `ActiveLoadRegistry` if so). Doing that
    /// cleanup here would leak orphan state into those sidecars when
    /// a caller actually wanted to keep the prior entry.
    pub fn add_with_cb(
        &self,
        spec: WorkerSpec,
        cb: Option<CircuitBreakerConfig>,
    ) -> Result<(), AddWorkerError> {
        let incoming_mode = spec.mode;
        // Hold the write lock for the entire validate→insert sequence.
        // Without it, two concurrent callers for conflicting modes on
        // the same model can both see an empty pool and both proceed
        // to insert, producing the mixed PD+plain state the check
        // exists to prevent.
        //
        // Mutex poisoning here means a previous writer panicked while
        // holding the lock — and since the critical section spans
        // `remove_locked` + several `by_model` updates + the final
        // `by_id.insert`, a panic mid-section can leave the registry
        // with a partial entry across the two DashMaps. Recovering via
        // `PoisonError::into_inner` would silently continue against
        // that half-written state; propagating the panic instead
        // surfaces the corruption to `manager::register_one`'s task
        // and ultimately trips `supervise_critical_tasks → mark_unready`
        // so the pod stops taking traffic. That's the right outcome.
        let _guard = self.write.lock().unwrap();
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
        self.remove_locked(&id);
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
        // Mirror `add_with_cb`'s write-lock acquisition so removals
        // don't race with concurrent adds (a stale `workers_for` snapshot
        // could otherwise let an add succeed against a peer that's
        // about to be removed, or vice versa).
        let _guard = self.write.lock().unwrap();
        self.remove_locked(id);
    }

    /// Internal removal that assumes the write lock is already held.
    /// Use this from any path that has acquired `self.write`.
    fn remove_locked(&self, id: &WorkerId) {
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

    /// Snapshot of every registered worker, across all models and modes,
    /// regardless of breaker state. Order is unspecified (iterates the
    /// underlying `DashMap`).
    ///
    /// Used by fleet-wide admin fan-out (e.g. `/flush_cache`) that targets
    /// every worker the router knows about rather than one model's pool, and
    /// by the `/metrics` scrape path to render per-worker gauges
    /// (`sgl_router_worker_health`, `_cb_state`, `_inflight_requests`) plus
    /// the pool-size gauge. The metrics path samples this fresh on each
    /// scrape rather than pushing, so a removed worker stops appearing
    /// immediately.
    pub fn all(&self) -> Vec<Arc<Worker>> {
        self.by_id.iter().map(|e| Arc::clone(e.value())).collect()
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
            min_priority: None,
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
    fn all_returns_every_worker_across_models_and_modes() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("w1", WorkerMode::Plain, &["m1"]));
        let _ = r.add(spec("p", WorkerMode::Prefill, &["m2"]));
        let _ = r.add(spec("d", WorkerMode::Decode, &["m2"]));
        let mut ids: Vec<String> = r.all().into_iter().map(|w| w.id.0.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["d", "p", "w1"]);
    }

    #[test]
    fn all_is_empty_for_fresh_registry() {
        assert!(WorkerRegistry::default().all().is_empty());
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
    fn all_lists_multi_model_worker_once() {
        let r = WorkerRegistry::default();
        // "a" serves two models; `all` must still list it once,
        // unlike a per-model enumeration which would double-count.
        let _ = r.add(spec("a", WorkerMode::Plain, &["m1", "m2"]));
        let _ = r.add(spec("b", WorkerMode::Plain, &["m1"]));
        let mut urls: Vec<String> = r.all().iter().map(|w| w.url.clone()).collect();
        urls.sort();
        assert_eq!(urls, vec!["http://a:30000", "http://b:30000"]);
    }

    /// `healthy_workers_for` must drop workers whose breaker is Open.
    /// An earlier version of this test asserted `healthy.len() == 2`
    /// against two workers with untouched breakers — i.e., it pinned
    /// only the no-op case (both Closed) and would have passed even if
    /// `healthy_workers_for` ignored the breaker entirely and was a
    /// thin alias for `workers_for`. Tripping one breaker and asserting
    /// the surviving set excludes it is the actual contract.
    #[test]
    fn healthy_subset_filters_via_breaker() {
        use crate::health::circuit_breaker::CircuitBreakerConfig;
        use std::num::NonZeroU32;
        use std::time::Duration;

        let r = WorkerRegistry::default();
        let _ = r.add_with_cb(spec("ok", WorkerMode::Plain, &["m"]), None);
        // Give "bad" a threshold=1 breaker so a single record_failure
        // flips it to Open.
        let _ = r.add_with_cb(
            spec("bad", WorkerMode::Plain, &["m"]),
            Some(CircuitBreakerConfig {
                threshold: NonZeroU32::new(1).unwrap(),
                cool_down: Duration::from_secs(30),
            }),
        );
        let bad = r.get(&WorkerId("bad".into())).expect("bad worker present");
        bad.breaker.record_failure();
        assert!(
            !bad.breaker.would_allow(),
            "sanity: threshold=1 + one failure must Open the breaker",
        );

        let healthy = r.healthy_workers_for(&ModelId("m".into()));
        assert_eq!(
            healthy.len(),
            1,
            "only the worker with a non-Open breaker should survive",
        );
        assert_eq!(healthy[0].id, WorkerId("ok".into()));
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

    /// On a rejected upsert with `MixedPdAndPlain`, the registry is
    /// **not** mutated — the prior entry for the rejected id stays
    /// put. Eviction (with the matching `KvEventIndex` /
    /// `ActiveLoadRegistry` cleanup) is the manager's responsibility;
    /// doing it here would leak orphan state in those sidecars.
    #[test]
    fn upsert_rejected_with_mixed_modes_leaves_registry_unchanged() {
        let r = WorkerRegistry::default();
        // Healthy PD pool on model m.
        let _ = r.add(spec("p", WorkerMode::Prefill, &["m"]));
        let _ = r.add(spec("d", WorkerMode::Decode, &["m"]));
        // Re-add "p" with Plain mode — discovery has reported a role flip.
        // The decode worker "d" is still on m, so validation rejects.
        let err = r
            .add(spec("p", WorkerMode::Plain, &["m"]))
            .expect_err("plain upsert must be rejected when peer decode worker remains");
        assert!(err.to_string().contains("plain"), "got: {err}");
        // Prior "p" entry survives (still Prefill). The registry
        // deliberately does NOT auto-evict on rejection — eviction
        // (and the matching sidecar cleanup) is the caller's call.
        let p = r
            .get(&WorkerId("p".into()))
            .expect("prior entry must remain — caller owns the cleanup");
        assert_eq!(p.mode(), WorkerMode::Prefill);
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
                .len(),
            1,
        );
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Decode)
                .len(),
            1,
        );
    }

    /// A *new* (not-yet-registered) worker rejected with `MixedPdAndPlain`
    /// must not affect the pool. Combined with the upsert test above,
    /// this pins that rejection never mutates registry state on its own.
    #[test]
    fn rejected_new_add_leaves_pool_untouched() {
        let r = WorkerRegistry::default();
        let _ = r.add(spec("plain", WorkerMode::Plain, &["m"]));
        let err = r
            .add(spec("p", WorkerMode::Prefill, &["m"]))
            .expect_err("PD worker must be rejected against existing plain pool");
        assert!(err.to_string().contains("plain"), "got: {err}");
        assert_eq!(
            r.workers_for_mode(&ModelId("m".into()), WorkerMode::Plain)
                .len(),
            1,
        );
        assert!(r.get(&WorkerId("p".into())).is_none());
    }

    /// Concurrent registrations from `manager::register_one` race against
    /// each other: each spawned task calls `add_with_cb` in parallel, and
    /// the validate-then-insert sequence inside that method is **not**
    /// atomic. Two threads adding workers of conflicting modes for the
    /// same model can both pass the existing-workers check (each sees an
    /// empty pool) and both proceed to insert, leaving the registry in a
    /// mixed PD+plain state — exactly the corruption the
    /// `MixedPdAndPlain` check is supposed to prevent.
    ///
    /// Invariant we pin: for every model, the resulting pool must be
    /// EITHER all-Plain OR all-PD, never a mix. We don't care which
    /// "winner" mode is selected — the racing manager already serialises
    /// per-WorkerId so it's the cross-id case that needs atomicity here.
    #[test]
    fn concurrent_conflicting_modes_never_produce_mixed_pool() {
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::thread;

        // All threads target one shared model so every `add_with_cb`
        // racer contends on the same `workers_for("m")` slot — that's
        // what makes the read-validate-write window of one thread
        // overlap with another's mutate. An earlier variant spread the
        // load across 4 models and did not reliably reproduce the bug
        // (per-slot contention was diluted to ~N/4 threads). 200
        // iterations × 16 threads triggers the race within the first
        // few iterations on the author's machine; post-fix the
        // invariant must hold across every iteration.
        const N_THREADS: usize = 16;
        const ITER: usize = 200;

        for iter in 0..ITER {
            let r = Arc::new(WorkerRegistry::default());
            let barrier = Arc::new(Barrier::new(N_THREADS));
            let mut handles = Vec::with_capacity(N_THREADS);
            for t in 0..N_THREADS {
                let r = Arc::clone(&r);
                let barrier = Arc::clone(&barrier);
                // Half the threads register Plain workers, half register
                // Prefill, all on the same model. With a non-atomic
                // validate→write inside `add_with_cb`, a Plain and a
                // Prefill thread both see an empty pool and both
                // succeed.
                let mode = if t % 2 == 0 {
                    WorkerMode::Plain
                } else {
                    WorkerMode::Prefill
                };
                let id = format!("iter{iter}-t{t}");
                handles.push(thread::spawn(move || {
                    barrier.wait();
                    let _ = r.add(spec(&id, mode, &["m"]));
                }));
            }
            for h in handles {
                h.join().unwrap();
            }

            // Invariant check: model is single-mode.
            let model = ModelId("m".into());
            let plain = r.workers_for_mode(&model, WorkerMode::Plain).len();
            let prefill = r.workers_for_mode(&model, WorkerMode::Prefill).len();
            let decode = r.workers_for_mode(&model, WorkerMode::Decode).len();
            let pd = prefill + decode;
            assert!(
                plain == 0 || pd == 0,
                "iter {iter}: registry holds a mixed pool — \
                 plain={plain}, prefill={prefill}, decode={decode}. \
                 The MixedPdAndPlain check in `add_with_cb` is not atomic \
                 across concurrent callers.",
            );
            // Sanity: the first thread to take the lock must succeed
            // (no peer exists yet). Defends against a degenerate "fix"
            // that satisfies the single-mode invariant by silently
            // rejecting every add.
            assert!(
                plain + pd >= 1,
                "iter {iter}: no workers were registered — \
                 the lock or mixed-mode check is starving every caller.",
            );
        }
    }
}
