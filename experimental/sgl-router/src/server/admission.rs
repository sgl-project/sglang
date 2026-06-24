// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Request-count admission control for the router.
//!
//! Without this gate the router forwards every request to the selected worker
//! the instant it arrives, so a saturated backend silently absorbs an unbounded
//! pile of in-flight requests at its connection front door — latency that the
//! client pays but the backend's own metrics never see. [`AdmissionQueue`]
//! bounds that: each worker admits at most `max_concurrent_per_worker`
//! in-flight requests, and when every candidate is full the request is parked
//! in a wait queue until a slot frees. The queue itself is bounded — once
//! `max_queued` requests are already waiting, further arrivals are shed with a
//! 503 so the router never queues without limit.
//!
//! Admission is opt-in via [`AdmissionConfig`]: [`AdmissionConfig::Disabled`]
//! (the default) makes the gate a pass-through and the router behaves exactly
//! as before.
//!
//! **Fairness:** wakeups are unordered. A freed slot wakes every parked waiter
//! and they race for it; losers re-park. There is no FIFO guarantee, so under
//! sustained saturation an individual request can be repeatedly out-raced. The
//! bounded queue caps how many wait, not how long any one waits;
//! `sgl_router_admission_wait_seconds` makes the wait distribution observable.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Notify, Semaphore};

use crate::config::AdmissionConfig;
use crate::policies::{Policy, SelectionContext};
use crate::server::error::ApiError;
use crate::server::metrics::MetricsRegistry;
use crate::workers::{LoadGuard, Worker};

/// Bounded, count-based admission gate shared on the request hot path.
#[derive(Debug)]
pub struct AdmissionQueue {
    /// Per-worker in-flight cap. `None` disables admission control entirely:
    /// `acquire` dispatches immediately with an unconditional load guard,
    /// matching the pre-admission behaviour.
    max_concurrent_per_worker: Option<usize>,
    /// Bounds how many requests may park waiting for a slot. `None` leaves the
    /// wait queue unbounded (park indefinitely, never shed). `Some(n)` sheds
    /// the request with 503 once `n` are already parked.
    queue_slots: Option<Semaphore>,
    /// Woken whenever a slot frees so parked requests re-attempt admission.
    notify: Notify,
    metrics: Arc<MetricsRegistry>,
}

impl AdmissionQueue {
    pub fn new(config: AdmissionConfig, metrics: Arc<MetricsRegistry>) -> Self {
        let (max_concurrent_per_worker, max_queued) = match config {
            AdmissionConfig::Disabled => (None, None),
            AdmissionConfig::Enabled {
                max_concurrent_per_worker,
                max_queued_requests,
            } => (Some(max_concurrent_per_worker.get()), max_queued_requests),
        };
        Self {
            max_concurrent_per_worker,
            queue_slots: max_queued.map(Semaphore::new),
            notify: Notify::new(),
            metrics,
        }
    }

    /// Signal that a worker slot has freed so parked requests wake and
    /// re-attempt admission. Wakes every parked waiter; each re-checks and only
    /// those that can claim a slot proceed, the rest re-park (see the fairness
    /// note in the module docs). A no-op when nothing is parked.
    pub fn notify_slot_freed(&self) {
        self.notify.notify_waiters();
    }

    /// Select a worker and claim an in-flight slot, parking until one frees if
    /// every candidate is at capacity. Returns [`ApiError::ServiceOverloaded`]
    /// (503) when the wait queue is already full, and
    /// [`ApiError::PolicySelectionFailed`] when the policy declines (the full
    /// candidate set when disabled, or the slot-having subset when enabled).
    ///
    /// The returned [`AdmissionGuard`] holds the claimed slot and wakes parked
    /// requests when it drops; the caller keeps it alive for the request's
    /// lifetime (stream end / client disconnect / error).
    pub async fn acquire(
        self: &Arc<Self>,
        candidates: &[Arc<Worker>],
        policy: &dyn Policy,
        ctx: &SelectionContext<'_>,
        model: &str,
    ) -> Result<(Arc<Worker>, AdmissionGuard), ApiError> {
        let Some(cap) = self.max_concurrent_per_worker else {
            // Disabled: dispatch immediately with an unconditional guard and no
            // wake obligation (nothing ever parks).
            let worker =
                policy
                    .select(candidates, ctx)
                    .ok_or_else(|| ApiError::PolicySelectionFailed {
                        model: model.to_string(),
                    })?;
            let guard = AdmissionGuard {
                load_guard: Some(worker.load_guard()),
                admission: None,
            };
            return Ok((worker, guard));
        };

        // Fast path: a slot is free right now.
        if let Some((worker, load_guard)) = self.try_claim(candidates, policy, ctx, cap, model)? {
            return Ok((worker, self.admitted_guard(load_guard)));
        }

        // All candidates full. Take a wait-queue slot, or shed with 503 if the
        // queue is already at its depth cap.
        let _queue_permit = match &self.queue_slots {
            Some(sem) => match sem.try_acquire() {
                Ok(permit) => Some(permit),
                Err(_) => {
                    self.metrics.record_backpressure_rejected(model);
                    return Err(ApiError::ServiceOverloaded {
                        model: model.to_string(),
                    });
                }
            },
            None => None,
        };
        let _parked = self.enter_queue();
        let parked_since = Instant::now();

        loop {
            // Register the waiter BEFORE re-checking, so a `notify_slot_freed`
            // racing between the check and the await is not lost.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            match self.try_claim(candidates, policy, ctx, cap, model) {
                Ok(Some((worker, load_guard))) => {
                    self.metrics
                        .observe_admission_wait(model, parked_since.elapsed().as_secs_f64());
                    return Ok((worker, self.admitted_guard(load_guard)));
                }
                Ok(None) => {}
                Err(e) => {
                    // The policy declined a now-non-empty slot-having set after
                    // the request had already parked (e.g. a sticky/cache-aware
                    // key whose only available worker is ineligible). Log it so a
                    // non-retryable error returned *after* a wait isn't silent to
                    // operators.
                    tracing::debug!(model = %model, "parked request declined by policy");
                    return Err(e);
                }
            }
            notified.await;
        }
    }

    /// Wrap a claimed [`LoadGuard`] in an [`AdmissionGuard`] that wakes parked
    /// requests when the slot frees.
    fn admitted_guard(self: &Arc<Self>, load_guard: LoadGuard) -> AdmissionGuard {
        AdmissionGuard {
            load_guard: Some(load_guard),
            admission: Some(Arc::clone(self)),
        }
    }

    /// One admission attempt. Returns `Ok(Some(..))` on a claimed slot,
    /// `Ok(None)` when every candidate is full (caller should park), and `Err`
    /// only when the policy declines a non-empty set of slot-having workers.
    fn try_claim(
        &self,
        candidates: &[Arc<Worker>],
        policy: &dyn Policy,
        ctx: &SelectionContext<'_>,
        cap: usize,
        model: &str,
    ) -> Result<Option<(Arc<Worker>, LoadGuard)>, ApiError> {
        // Each `continue` re-runs the filter; it is only reached when a
        // competing claim consumed the slot we picked, i.e. global forward
        // progress happened. With a finite candidate set and finite `cap` the
        // loop is bounded by the number of concurrent claimers, so it cannot
        // spin forever.
        loop {
            // Only offer the policy workers that currently have a free slot, so
            // a cache-affinity or sticky decision never pins onto a worker
            // that is already at capacity while another has room.
            let available: Vec<Arc<Worker>> = candidates
                .iter()
                .filter(|w| w.active_load() < cap)
                .cloned()
                .collect();
            if available.is_empty() {
                return Ok(None);
            }
            let worker =
                policy
                    .select(&available, ctx)
                    .ok_or_else(|| ApiError::PolicySelectionFailed {
                        model: model.to_string(),
                    })?;
            match worker.try_load_guard(cap) {
                Some(guard) => return Ok(Some((worker, guard))),
                // Raced: the chosen worker filled between the filter and the
                // claim. Recompute the available set and try again.
                None => continue,
            }
        }
    }

    fn enter_queue(&self) -> QueuedGuard<'_> {
        self.metrics.add_queued_requests(1);
        QueuedGuard { queue: self }
    }
}

/// Holds a claimed per-worker in-flight slot for the request's lifetime and,
/// when it drops, wakes admission-parked requests so they re-attempt.
///
/// The wake ordering lives in one place — this `Drop`: the inner [`LoadGuard`]
/// is dropped (decrementing the worker's in-flight count) BEFORE
/// `notify_slot_freed`, so a woken waiter re-checking capacity sees the freed
/// slot. `admission` is `None` on the disabled pass-through path, where nothing
/// ever parks and no wake is needed.
#[derive(Debug)]
pub struct AdmissionGuard {
    load_guard: Option<LoadGuard>,
    admission: Option<Arc<AdmissionQueue>>,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        // Decrement the slot first, THEN wake — a woken waiter must observe the
        // freed slot. Taking the `Option` forces the `LoadGuard` drop here,
        // before the notify, rather than at end-of-struct-drop.
        drop(self.load_guard.take());
        if let Some(admission) = &self.admission {
            admission.notify_slot_freed();
        }
    }
}

/// Tracks one parked request: decrements the queued counter (and the gauge) on
/// drop, whether the request was ultimately admitted or abandoned (the handler
/// future dropped on client disconnect). Requests shed with 503 return before
/// entering the queue, so they never construct this guard.
struct QueuedGuard<'a> {
    queue: &'a AdmissionQueue,
}

impl Drop for QueuedGuard<'_> {
    fn drop(&mut self) {
        self.queue.metrics.add_queued_requests(-1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use std::num::NonZeroUsize;
    use std::time::Duration;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    fn metrics() -> Arc<MetricsRegistry> {
        MetricsRegistry::new()
    }

    /// An `Enabled` admission config with the given per-worker cap.
    fn enabled(cap: usize, max_queued: Option<usize>) -> AdmissionConfig {
        AdmissionConfig::Enabled {
            max_concurrent_per_worker: NonZeroUsize::new(cap).unwrap(),
            max_queued_requests: max_queued,
        }
    }

    fn queue(config: AdmissionConfig) -> Arc<AdmissionQueue> {
        Arc::new(AdmissionQueue::new(config, metrics()))
    }

    /// Picks the first candidate offered — deterministic for tests.
    #[derive(Debug)]
    struct PickFirst;
    impl Policy for PickFirst {
        fn select(
            &self,
            workers: &[Arc<Worker>],
            _ctx: &SelectionContext<'_>,
        ) -> Option<Arc<Worker>> {
            workers.first().cloned()
        }
    }

    /// Never selects — models a policy that declines.
    #[derive(Debug)]
    struct PickNone;
    impl Policy for PickNone {
        fn select(
            &self,
            _workers: &[Arc<Worker>],
            _ctx: &SelectionContext<'_>,
        ) -> Option<Arc<Worker>> {
            None
        }
    }

    #[tokio::test]
    async fn disabled_gate_admits_even_when_worker_is_loaded() {
        let q = queue(AdmissionConfig::Disabled);
        let w = worker("w");
        // Pile on existing load; the disabled gate must not care.
        let _h1 = w.load_guard();
        let _h2 = w.load_guard();
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);

        let (chosen, _g) = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect("disabled gate must always admit");
        assert_eq!(chosen.id, w.id);
        assert_eq!(w.active_load(), 3);
    }

    #[tokio::test]
    async fn fast_path_admits_below_cap() {
        let q = queue(enabled(2, None));
        let w = worker("w");
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);

        let (chosen, _g) = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect("under cap must admit");
        assert_eq!(chosen.id, w.id);
        assert_eq!(w.active_load(), 1);
    }

    #[tokio::test]
    async fn skips_full_worker_for_one_with_a_free_slot() {
        let q = queue(enabled(1, None));
        let full = worker("full");
        let free = worker("free");
        let _hold = full.load_guard(); // full is at cap=1
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);

        // PickFirst would choose `full` first, but it is filtered out as
        // having no free slot, so `free` is admitted instead.
        let (chosen, _g) = q
            .acquire(
                &[Arc::clone(&full), Arc::clone(&free)],
                &PickFirst,
                &ctx,
                "m",
            )
            .await
            .expect("must admit the worker with a free slot");
        assert_eq!(chosen.id, free.id);
    }

    #[tokio::test]
    async fn rejects_with_503_when_queue_full() {
        // Depth cap of 0: never park, shed immediately once the worker is full.
        let m = metrics();
        let q = Arc::new(AdmissionQueue::new(enabled(1, Some(0)), Arc::clone(&m)));
        let w = worker("w");
        let _hold = w.load_guard(); // w at cap=1
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);

        let err = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect_err("full worker + full queue must be shed");
        assert!(
            matches!(err, ApiError::ServiceOverloaded { .. }),
            "expected ServiceOverloaded, got {err:?}",
        );
        let out = m.render();
        assert!(
            out.contains(r#"sgl_router_backpressure_rejected_total{model_id="m"} 1"#),
            "rejection must be counted: {out}",
        );
        // A shed request never enters the queue, so the gauge stays at 0.
        assert!(out.contains("sgl_router_queued_requests 0\n"), "{out}");
    }

    #[tokio::test]
    async fn policy_declining_nonempty_set_is_policy_selection_failed() {
        let q = queue(enabled(2, None));
        let w = worker("w");
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);

        let err = q
            .acquire(&[Arc::clone(&w)], &PickNone, &ctx, "m")
            .await
            .expect_err("declining policy must error");
        assert!(
            matches!(err, ApiError::PolicySelectionFailed { .. }),
            "expected PolicySelectionFailed, got {err:?}",
        );
    }

    #[tokio::test]
    async fn parks_until_slot_frees_then_admits() {
        let q = queue(enabled(1, None));
        let w = worker("w");
        let held = w.load_guard(); // w at cap=1 -> next acquire parks
        assert_eq!(w.active_load(), 1);

        let q2 = Arc::clone(&q);
        let cands = vec![Arc::clone(&w)];
        let handle = tokio::spawn(async move {
            let model = ModelId("m".into());
            let ctx = SelectionContext::new(&model, None);
            q2.acquire(&cands, &PickFirst, &ctx, "m")
                .await
                .map(|(w, _g)| w.id.clone())
        });

        // Let the task reach the parked state.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !handle.is_finished(),
            "acquire should park while worker is full"
        );

        // Free the slot and wake the waiter.
        drop(held);
        q.notify_slot_freed();

        let chosen = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("parked acquire must resolve once a slot frees")
            .expect("task panicked")
            .expect("acquire should succeed after slot frees");
        assert_eq!(chosen, w.id);
    }

    #[tokio::test]
    async fn unbounded_queue_parks_many_then_admits_each_as_slots_free() {
        // cap=1 + unbounded queue: many waiters all park (none shed); freeing
        // the slot cascades admissions one at a time as each admitted request's
        // guard drops and wakes the next.
        let m = metrics();
        let q = Arc::new(AdmissionQueue::new(enabled(1, None), Arc::clone(&m)));
        let w = worker("w");
        let held = w.load_guard(); // occupies the only slot

        let mut handles = Vec::new();
        for _ in 0..3 {
            let q2 = Arc::clone(&q);
            let cands = vec![Arc::clone(&w)];
            handles.push(tokio::spawn(async move {
                let model = ModelId("m".into());
                let ctx = SelectionContext::new(&model, None);
                // Drop the guard immediately on admission so the slot frees and
                // the next waiter can proceed.
                q2.acquire(&cands, &PickFirst, &ctx, "m").await.map(|_| ())
            }));
        }

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            handles.iter().all(|h| !h.is_finished()),
            "all three should park while the only slot is held"
        );
        // All three are parked; none was shed (unbounded queue).
        assert!(
            m.render().contains("sgl_router_queued_requests 3\n"),
            "{}",
            m.render(),
        );

        // Release the slot once; admissions cascade via each guard's drop.
        drop(held);
        q.notify_slot_freed();

        for h in handles {
            tokio::time::timeout(Duration::from_secs(2), h)
                .await
                .expect("each parked waiter must eventually admit")
                .expect("task panicked")
                .expect("acquire should succeed");
        }
        // Queue fully drained.
        assert!(
            m.render().contains("sgl_router_queued_requests 0\n"),
            "{}",
            m.render(),
        );
    }

    #[tokio::test]
    async fn abandoned_parked_request_releases_its_queue_slot() {
        // Depth cap 1: A parks (taking the one queue slot), B is shed. Aborting
        // A (client disconnect) must release the slot — observable via the
        // gauge returning to 0 — so a later C can park instead of being shed.
        let m = metrics();
        let q = Arc::new(AdmissionQueue::new(enabled(1, Some(1)), Arc::clone(&m)));
        let w = worker("w");
        let _held = w.load_guard(); // worker full for the whole test
        let cands = vec![Arc::clone(&w)];

        let qa = Arc::clone(&q);
        let ca = cands.clone();
        let a = tokio::spawn(async move {
            let model = ModelId("m".into());
            let ctx = SelectionContext::new(&model, None);
            qa.acquire(&ca, &PickFirst, &ctx, "m").await.map(|_| ())
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !a.is_finished(),
            "A should be parked holding the queue slot"
        );
        assert!(
            m.render().contains("sgl_router_queued_requests 1\n"),
            "A parked -> gauge should read 1: {}",
            m.render(),
        );

        // B finds the queue full -> shed.
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let b = q.acquire(&cands, &PickFirst, &ctx, "m").await;
        assert!(
            matches!(b, Err(ApiError::ServiceOverloaded { .. })),
            "B must be shed while the single queue slot is taken: {b:?}",
        );

        // A disconnects: aborting drops its `acquire` future, which releases the
        // queue permit and (via QueuedGuard::drop) returns the gauge to 0.
        a.abort();
        let _ = a.await;
        assert!(
            m.render().contains("sgl_router_queued_requests 0\n"),
            "A's abort must release its queue slot (gauge back to 0): {}",
            m.render(),
        );

        // C can now park (queue slot freed) rather than being shed.
        let qc = Arc::clone(&q);
        let cc = cands.clone();
        let c = tokio::spawn(async move {
            let model = ModelId("m".into());
            let ctx = SelectionContext::new(&model, None);
            qc.acquire(&cc, &PickFirst, &ctx, "m").await.map(|_| ())
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !c.is_finished(),
            "C should park (slot was freed by A's abort), not shed"
        );
        assert!(
            m.render().contains("sgl_router_queued_requests 1\n"),
            "C parked -> gauge should read 1 again: {}",
            m.render(),
        );
        c.abort();
        let _ = c.await;
    }
}
