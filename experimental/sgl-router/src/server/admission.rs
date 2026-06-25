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
//! **Fairness:** parked requests are admitted in strict FIFO order. When a
//! request completes, its freed per-worker slot is handed *directly* to the
//! most-senior parked waiter eligible for that worker (the first one whose
//! candidate set contains it) rather than decremented and re-contended. Because
//! the slot is transferred — the worker's in-flight count never dips during the
//! hand-off — a freshly arriving request that races for the same slot finds the
//! worker still at capacity and parks *behind* the existing waiters instead of
//! jumping ahead of them. A waiter is overtaken only by a later arrival that is
//! ineligible for the freed worker (a different candidate pool), never by one
//! competing for the same slot. `sgl_router_admission_wait_seconds` makes the
//! resulting wait distribution observable.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};

use crate::config::AdmissionConfig;
use crate::discovery::WorkerId;
use crate::policies::{Policy, SelectionContext};
use crate::server::error::ApiError;
use crate::server::metrics::MetricsRegistry;
use crate::workers::{LoadGuard, Worker};

/// A slot handed to a parked waiter: the worker it was claimed on plus the live
/// [`AdmissionGuard`] holding it. Sent over the waiter's oneshot on hand-off.
type Grant = (Arc<Worker>, AdmissionGuard);

/// One parked request. Shared (`Arc`) between the wait queue and the parking
/// `acquire` frame's [`WaiterTicket`]. The hand-off path pops it from the queue
/// and sends the freed slot through `grant_tx`; the `acquire` frame awaits the
/// matching receiver.
#[derive(Debug)]
struct Waiter {
    /// Monotonic id, used to find-and-remove this waiter from the queue on
    /// cancellation without relying on `Arc` pointer identity.
    id: u64,
    /// Ids of the workers this request may run on. A freed slot is handed to a
    /// waiter only when its worker is in this set, so a request is never woken
    /// for a pool it cannot use.
    candidate_ids: Vec<WorkerId>,
    /// Sender for the granted slot. `take`n exactly once — by the hand-off that
    /// delivers a slot, or by [`WaiterTicket::drop`] on cancellation, whichever
    /// happens first — so a slot is never both delivered and abandoned.
    grant_tx: Mutex<Option<oneshot::Sender<Grant>>>,
}

/// Bounded, count-based admission gate shared on the request hot path.
#[derive(Debug)]
pub struct AdmissionQueue {
    /// Per-worker in-flight cap. `None` disables admission control entirely:
    /// `acquire` dispatches immediately with an unconditional load guard,
    /// matching the pre-admission behaviour.
    max_concurrent_per_worker: Option<usize>,
    /// Bounds how many requests may park waiting for a slot. `None` leaves the
    /// wait queue unbounded (park indefinitely, never shed). `Some(sem)` sheds
    /// the request with 503 once all `sem` permits are taken.
    queue_slots: Option<Arc<Semaphore>>,
    /// Parked waiters in arrival (FIFO) order. Guarded by a sync mutex held only
    /// for short, await-free critical sections: the enqueue decision and the
    /// hand-off scan. Both the parking path's under-lock claim and the
    /// hand-off's decrement happen here, which is what makes "a freed slot is
    /// never lost between a waiter parking and a slot freeing" hold.
    waiters: Mutex<VecDeque<Arc<Waiter>>>,
    /// Source of monotonic [`Waiter::id`]s.
    next_waiter_id: AtomicU64,
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
            queue_slots: max_queued.map(|n| Arc::new(Semaphore::new(n))),
            waiters: Mutex::new(VecDeque::new()),
            next_waiter_id: AtomicU64::new(0),
            metrics,
        }
    }

    /// Select a worker and claim an in-flight slot, parking until one frees if
    /// every candidate is at capacity. Returns [`ApiError::ServiceOverloaded`]
    /// (503, retryable) when the request cannot be admitted — primarily when the
    /// wait queue is already at its depth cap — and [`ApiError::PolicySelectionFailed`]
    /// when the policy declines (the full candidate set when disabled, or the
    /// slot-having subset when enabled).
    ///
    /// The returned [`AdmissionGuard`] holds the claimed slot and, when it
    /// drops, hands the slot to the next FIFO waiter (or frees it); the caller
    /// keeps it alive for the request's lifetime (stream end / client
    /// disconnect / error).
    pub async fn acquire(
        self: &Arc<Self>,
        candidates: &[Arc<Worker>],
        policy: &dyn Policy,
        ctx: &SelectionContext<'_>,
        model: &str,
    ) -> Result<(Arc<Worker>, AdmissionGuard), ApiError> {
        let Some(cap) = self.max_concurrent_per_worker else {
            // Disabled: dispatch immediately with an unconditional guard and no
            // hand-off obligation (nothing ever parks).
            let worker =
                policy
                    .select(candidates, ctx)
                    .ok_or_else(|| ApiError::PolicySelectionFailed {
                        model: model.to_string(),
                    })?;
            let guard = AdmissionGuard::pass_through(worker.load_guard());
            return Ok((worker, guard));
        };

        // Fast path: claim a free slot lock-free. This can only succeed on a
        // genuinely free slot, which — because a freed slot is handed directly
        // to a waiter while the count stays at `cap` — means no waiter was
        // queued for that worker. So taking it here never jumps the FIFO queue.
        if let Some((worker, load_guard)) = self.try_claim(candidates, policy, ctx, cap, model)? {
            return Ok((worker.clone(), self.handed_off_guard(worker, load_guard)));
        }

        self.park(candidates, policy, ctx, cap, model).await
    }

    /// Slow path: every candidate was full on the lock-free attempt. Re-check
    /// under the lock (a slot may have freed), then either claim it, shed with
    /// 503, or park in FIFO order until a slot is handed to us.
    async fn park(
        self: &Arc<Self>,
        candidates: &[Arc<Worker>],
        policy: &dyn Policy,
        ctx: &SelectionContext<'_>,
        cap: usize,
        model: &str,
    ) -> Result<(Arc<Worker>, AdmissionGuard), ApiError> {
        let parked_since = Instant::now();

        // Enqueue under the lock. The `_ticket` returned alongside the receiver
        // owns the queue-depth gauge, the wait-queue permit, and removal of this
        // waiter from the queue — all released when it drops (on admission or
        // cancellation). It must outlive the `.await` below, so bind it.
        let (rx, _ticket) = {
            let mut waiters = self.waiters.lock();

            // A slot may have freed between the lock-free attempt and acquiring
            // this lock. Re-checking here, under the same lock the hand-off uses
            // to decrement, serializes the two: a slot freed before we parked is
            // claimed here, never lost.
            if let Some((worker, load_guard)) =
                self.try_claim(candidates, policy, ctx, cap, model)?
            {
                return Ok((worker.clone(), self.handed_off_guard(worker, load_guard)));
            }

            // Still full: reserve a wait-queue slot, or shed if the queue is at
            // its depth cap.
            let permit = match &self.queue_slots {
                Some(sem) => match Arc::clone(sem).try_acquire_owned() {
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

            let (tx, rx) = oneshot::channel();
            let waiter = Arc::new(Waiter {
                id: self.next_waiter_id.fetch_add(1, Ordering::Relaxed),
                candidate_ids: candidates.iter().map(|w| w.id.clone()).collect(),
                grant_tx: Mutex::new(Some(tx)),
            });
            waiters.push_back(Arc::clone(&waiter));
            self.metrics.add_queued_requests(1);

            let ticket = WaiterTicket {
                admission: Arc::clone(self),
                waiter,
                _permit: permit,
            };
            (rx, ticket)
        };

        match rx.await {
            Ok((worker, guard)) => {
                self.metrics
                    .observe_admission_wait(model, parked_since.elapsed().as_secs_f64());
                Ok((worker, guard))
            }
            // The sender was dropped without delivering a slot. In normal
            // operation the sender is held alive by `_ticket` until either the
            // hand-off sends through it or the ticket itself cancels — so this is
            // only reachable if the queue is being torn down (or an internal
            // invariant broke). Log it: a 503 here is NOT ordinary backpressure
            // (no `record_backpressure_rejected`), and silently folding it into
            // the shed path would hide a real bug behind a routine status.
            Err(_) => {
                tracing::error!(
                    model = %model,
                    "admission grant sender dropped without delivering a slot; \
                     treating as overloaded (queue teardown or invariant violation)"
                );
                Err(ApiError::ServiceOverloaded {
                    model: model.to_string(),
                })
            }
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

    /// Hand a freed `worker` slot (held by `load_guard`) to the most-senior
    /// parked waiter eligible for it, or release the slot if none want it.
    ///
    /// Called from [`AdmissionGuard::drop`]. The pop-or-decrement decision is
    /// made under the lock so it is atomic with the parking path's under-lock
    /// claim — a slot freed while a waiter is queued is always handed off, and a
    /// slot freed with no eligible waiter is decremented while a fresh parker
    /// would still see it as taken (and so be handed it on the next free) or see
    /// it freed (and claim it directly).
    fn release_slot(self: &Arc<Self>, worker: Arc<Worker>, mut load_guard: LoadGuard) {
        loop {
            let waiter = {
                let mut waiters = self.waiters.lock();
                match waiters
                    .iter()
                    .position(|w| w.candidate_ids.contains(&worker.id))
                {
                    Some(pos) => waiters.remove(pos),
                    None => {
                        // No taker: free the slot (decrement) while holding the
                        // lock, so the decision stays atomic with parkers.
                        drop(load_guard);
                        return;
                    }
                }
            };
            // `remove(pos)` cannot return `None` for an in-range index, but match
            // defensively rather than unwrap.
            let Some(waiter) = waiter else {
                drop(load_guard);
                return;
            };

            // Deliver outside the lock: the receiver may already be gone, in
            // which case the guard we hand over drops and its `Drop` re-enters
            // `release_slot` — so the lock must not be held here.
            let Some(tx) = waiter.grant_tx.lock().take() else {
                // The waiter was cancelled between our pop and now; the slot is
                // still ours, so try the next eligible waiter.
                continue;
            };
            let guard = self.handed_off_guard(Arc::clone(&worker), load_guard);
            match tx.send((Arc::clone(&worker), guard)) {
                Ok(()) => return,
                Err((_worker, returned)) => {
                    // Receiver gone (client disconnected before admission).
                    // Reclaim the slot from the undelivered guard and hand it to
                    // the next waiter. Disarming first makes the returned guard's
                    // `Drop` a no-op, so we loop here instead of recursing.
                    load_guard = returned.disarm();
                }
            }
        }
    }

    /// Wrap a claimed [`LoadGuard`] in an [`AdmissionGuard`] that, on drop,
    /// routes the freed slot through FIFO hand-off for `worker`.
    fn handed_off_guard(
        self: &Arc<Self>,
        worker: Arc<Worker>,
        load_guard: LoadGuard,
    ) -> AdmissionGuard {
        AdmissionGuard::Armed(ArmedGuard {
            load_guard: Some(load_guard),
            worker,
            admission: Arc::clone(self),
        })
    }

    /// Number of parked waiters. Test-only; production reads the
    /// `sgl_router_queued_requests` gauge instead.
    #[cfg(test)]
    fn queued_len(&self) -> usize {
        self.waiters.lock().len()
    }
}

/// Holds a claimed per-worker in-flight slot for the request's lifetime. The
/// three variants make the only legal states exact, so there is no "armed but
/// slotless" combination to guard against:
///
/// * [`PassThrough`](Self::PassThrough) — disabled gate: holds a load slot for
///   the request but has no wait queue to hand back to, so it just decrements on
///   drop.
/// * [`Armed`](Self::Armed) — enabled gate: on drop, the slot is routed through
///   FIFO hand-off (see [`ArmedGuard`]) rather than decremented.
/// * [`Spent`](Self::Spent) — holds nothing; the state left behind after
///   [`disarm`](Self::disarm) moves the slot out for re-hand-off. Drop is a
///   no-op.
///
/// The enum itself carries no `Drop`; the hand-off behaviour lives on the
/// [`ArmedGuard`] payload, so the disabled and spent states drop trivially.
#[must_use = "an AdmissionGuard must be held for the request's lifetime; \
              dropping it releases the worker slot (and hands it to the next waiter)"]
#[derive(Debug)]
pub enum AdmissionGuard {
    /// The slot has been moved out (reclaimed by [`disarm`](Self::disarm) for
    /// re-hand-off). Drop is a no-op.
    Spent,
    /// Disabled gate: a plain load slot that decrements on drop.
    PassThrough(LoadGuard),
    /// Enabled gate: the claimed slot plus its FIFO hand-off target.
    Armed(ArmedGuard),
}

impl AdmissionGuard {
    /// Guard for the disabled gate: holds the load slot for the request but has
    /// no wait queue to hand back to, so it just decrements on drop.
    fn pass_through(load_guard: LoadGuard) -> Self {
        Self::PassThrough(load_guard)
    }

    /// Move the load slot out of an [`Armed`](Self::Armed) guard, leaving the
    /// guard inert so its `Drop` is a no-op. The slot's in-flight count is
    /// preserved (the [`LoadGuard`] is moved, not dropped), so
    /// [`AdmissionQueue::release_slot`] can re-hand it to the next waiter without
    /// re-entering hand-off. Only ever called on a freshly-built `Armed` guard
    /// returned from a failed send, so the other arms are unreachable.
    fn disarm(self) -> LoadGuard {
        match self {
            // `take` leaves the `ArmedGuard` with no slot, so when it drops at
            // the end of this arm its `Drop` skips the hand-off.
            Self::Armed(mut armed) => armed
                .load_guard
                .take()
                .expect("armed guard reclaimed twice"),
            _ => unreachable!("disarm called on a non-armed AdmissionGuard"),
        }
    }
}

/// Payload of an [`AdmissionGuard::Armed`]: the claimed slot plus the worker and
/// queue needed to hand it off. The hand-off obligation lives here, in `Drop`,
/// so it is impossible to forget and runs exactly once whether the request ends
/// by stream completion, client disconnect, or error.
///
/// `load_guard` is an `Option` only so [`AdmissionGuard::disarm`] can move the
/// slot out for re-hand-off, leaving this guard inert; it is `Some` in every
/// other state.
#[derive(Debug)]
pub struct ArmedGuard {
    load_guard: Option<LoadGuard>,
    worker: Arc<Worker>,
    admission: Arc<AdmissionQueue>,
}

impl Drop for ArmedGuard {
    fn drop(&mut self) {
        // The slot is *moved* into `release_slot` (which keeps the worker's
        // in-flight count unchanged while it transfers, or decrements if no
        // waiter wants it) — never dropped here. `None` means `disarm` already
        // took it.
        if let Some(load_guard) = self.load_guard.take() {
            self.admission
                .release_slot(Arc::clone(&self.worker), load_guard);
        }
    }
}

/// Tracks one parked request's wait-queue accounting: the depth gauge, the
/// bounded-queue permit, and removal from the waiter list. Dropping it — on
/// admission or on the handler future being dropped (client disconnect) —
/// releases all three exactly once.
///
/// Taking `grant_tx` here ensures that if cancellation races an in-flight
/// hand-off, the hand-off finds the sender gone and routes the slot to the next
/// waiter instead of delivering it into a dropped receiver.
struct WaiterTicket {
    admission: Arc<AdmissionQueue>,
    waiter: Arc<Waiter>,
    _permit: Option<OwnedSemaphorePermit>,
}

impl Drop for WaiterTicket {
    fn drop(&mut self) {
        {
            let mut waiters = self.admission.waiters.lock();
            if let Some(pos) = waiters.iter().position(|w| w.id == self.waiter.id) {
                waiters.remove(pos);
            }
        }
        // Claim the sender so a concurrent hand-off returns the slot to the
        // queue rather than delivering into our dropped receiver. No-op if the
        // hand-off already delivered (normal admission).
        let _ = self.waiter.grant_tx.lock().take();
        self.admission.metrics.add_queued_requests(-1);
        // `_permit` drops here, returning the wait-queue slot.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use std::num::NonZeroUsize;
    use std::sync::Mutex as StdMutex;
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

    /// Poll until the wait queue holds exactly `n` parked waiters, or panic.
    /// Lets a test deterministically confirm waiters have parked (in a known
    /// order) before it releases a slot.
    async fn wait_until_queued(q: &Arc<AdmissionQueue>, n: usize) {
        for _ in 0..200 {
            if q.queued_len() == n {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!(
            "queue never reached depth {n} (stuck at {})",
            q.queued_len()
        );
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

        // First acquire claims the only slot, via the gate (so dropping its
        // guard triggers the hand-off path, not a bare decrement).
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let (_w0, held) = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect("first acquire claims the slot");
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

        wait_until_queued(&q, 1).await;
        assert!(
            !handle.is_finished(),
            "acquire should park while worker is full"
        );

        // Drop the first guard: its slot is handed to the parked waiter.
        drop(held);

        let chosen = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("parked acquire must resolve once a slot frees")
            .expect("task panicked")
            .expect("acquire should succeed after slot frees");
        assert_eq!(chosen, w.id);
    }

    #[tokio::test]
    async fn admits_parked_waiters_in_fifo_order() {
        // cap=1: one slot, three waiters parked in order A, B, C. They must be
        // admitted A, B, C regardless of scheduler wake order.
        let q = queue(enabled(1, None));
        let w = worker("w");

        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let (_w0, held) = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect("seed acquire claims the slot");

        let order: Arc<StdMutex<Vec<u32>>> = Arc::new(StdMutex::new(Vec::new()));
        let mut handles = Vec::new();
        // Park A, then B, then C — each confirmed parked before the next spawns,
        // so the queue order is deterministic.
        for label in 0..3u32 {
            let q2 = Arc::clone(&q);
            let cands = vec![Arc::clone(&w)];
            let order2 = Arc::clone(&order);
            handles.push(tokio::spawn(async move {
                let model = ModelId("m".into());
                let ctx = SelectionContext::new(&model, None);
                let (_w, guard) = q2.acquire(&cands, &PickFirst, &ctx, "m").await.unwrap();
                order2.lock().unwrap().push(label);
                // Drop the slot so the next FIFO waiter is handed it.
                drop(guard);
            }));
            wait_until_queued(&q, (label + 1) as usize).await;
        }

        // Release the seed slot; admissions cascade in FIFO order.
        drop(held);
        for h in handles {
            tokio::time::timeout(Duration::from_secs(2), h)
                .await
                .expect("each waiter must admit")
                .expect("task panicked");
        }

        assert_eq!(
            *order.lock().unwrap(),
            vec![0, 1, 2],
            "waiters must be admitted in arrival order"
        );
        assert_eq!(q.queued_len(), 0, "queue should fully drain");
    }

    #[tokio::test]
    async fn newcomer_parks_behind_existing_waiter() {
        // A parks first; a later arrival N must not jump it. Freeing exactly one
        // slot admits A, not N.
        let q = queue(enabled(1, None));
        let w = worker("w");

        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let (_w0, held) = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect("seed acquire claims the slot");

        let order: Arc<StdMutex<Vec<&'static str>>> = Arc::new(StdMutex::new(Vec::new()));

        let spawn_waiter = |name: &'static str| {
            let q2 = Arc::clone(&q);
            let cands = vec![Arc::clone(&w)];
            let order2 = Arc::clone(&order);
            tokio::spawn(async move {
                let model = ModelId("m".into());
                let ctx = SelectionContext::new(&model, None);
                let (_w, guard) = q2.acquire(&cands, &PickFirst, &ctx, "m").await.unwrap();
                order2.lock().unwrap().push(name);
                drop(guard);
            })
        };

        let a = spawn_waiter("A");
        wait_until_queued(&q, 1).await;
        let n = spawn_waiter("N");
        wait_until_queued(&q, 2).await;

        drop(held);
        for h in [a, n] {
            tokio::time::timeout(Duration::from_secs(2), h)
                .await
                .expect("waiter must admit")
                .expect("task panicked");
        }
        assert_eq!(
            *order.lock().unwrap(),
            vec!["A", "N"],
            "the earlier waiter must be admitted before the later arrival"
        );
    }

    #[tokio::test]
    async fn unbounded_queue_parks_many_then_admits_each_as_slots_free() {
        // cap=1 + unbounded queue: many waiters all park (none shed); freeing
        // the slot cascades admissions one at a time as each admitted request's
        // guard drops and hands the slot to the next.
        let m = metrics();
        let q = Arc::new(AdmissionQueue::new(enabled(1, None), Arc::clone(&m)));
        let w = worker("w");

        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let (_w0, held) = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect("seed acquire claims the slot");

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

        wait_until_queued(&q, 3).await;
        // All three are parked; none was shed (unbounded queue).
        assert!(
            m.render().contains("sgl_router_queued_requests 3\n"),
            "{}",
            m.render(),
        );

        // Release the slot once; admissions cascade via each guard's drop.
        drop(held);

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
        assert_eq!(q.queued_len(), 0);
    }

    #[tokio::test]
    async fn abandoned_parked_request_releases_its_queue_slot() {
        // Depth cap 1: A parks (taking the one queue slot), B is shed. Aborting
        // A (client disconnect) must release the slot — observable via the
        // gauge returning to 0 — so a later C can park instead of being shed.
        let m = metrics();
        let q = Arc::new(AdmissionQueue::new(enabled(1, Some(1)), Arc::clone(&m)));
        let w = worker("w");

        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let (_w0, _held) = q
            .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
            .await
            .expect("seed acquire claims the slot"); // worker full for the whole test
        let cands = vec![Arc::clone(&w)];

        let qa = Arc::clone(&q);
        let ca = cands.clone();
        let a = tokio::spawn(async move {
            let model = ModelId("m".into());
            let ctx = SelectionContext::new(&model, None);
            qa.acquire(&ca, &PickFirst, &ctx, "m").await.map(|_| ())
        });
        wait_until_queued(&q, 1).await;
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
        // queue permit and (via WaiterTicket::drop) returns the gauge to 0.
        a.abort();
        let _ = a.await;
        wait_until_queued(&q, 0).await;
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
        wait_until_queued(&q, 1).await;
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

    #[tokio::test]
    async fn freed_worker_slot_skips_ineligible_senior_waiter() {
        // Two single-slot worker pools. A parks first but is eligible only for
        // `wy`; B parks later, eligible only for `wx`. Freeing `wx` must admit B
        // (the only eligible waiter) even though A is more senior — a freed slot
        // is handed by FIFO *within* the eligible set, never to a waiter that
        // cannot use the worker. Freeing `wy` then admits A.
        let q = queue(enabled(1, None));
        let wx = worker("wx");
        let wy = worker("wy");

        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let (_x, held_x) = q
            .acquire(&[Arc::clone(&wx)], &PickFirst, &ctx, "m")
            .await
            .expect("wx seed");
        let (_y, held_y) = q
            .acquire(&[Arc::clone(&wy)], &PickFirst, &ctx, "m")
            .await
            .expect("wy seed");

        let order: Arc<StdMutex<Vec<&'static str>>> = Arc::new(StdMutex::new(Vec::new()));
        let spawn_waiter = |name: &'static str, cands: Vec<Arc<Worker>>| {
            let q2 = Arc::clone(&q);
            let order2 = Arc::clone(&order);
            tokio::spawn(async move {
                let model = ModelId("m".into());
                let ctx = SelectionContext::new(&model, None);
                let (chosen, guard) = q2.acquire(&cands, &PickFirst, &ctx, "m").await.unwrap();
                order2.lock().unwrap().push(name);
                (chosen.id.clone(), guard)
            })
        };

        // A: eligible only for wy (senior). B: eligible only for wx (junior).
        let a = spawn_waiter("A", vec![Arc::clone(&wy)]);
        wait_until_queued(&q, 1).await;
        let b = spawn_waiter("B", vec![Arc::clone(&wx)]);
        wait_until_queued(&q, 2).await;

        // Free wx: only B is eligible -> B admits despite A being senior.
        drop(held_x);
        let (b_worker, _b_guard) = tokio::time::timeout(Duration::from_secs(2), b)
            .await
            .expect("B must admit when its eligible worker frees")
            .expect("B task panicked");
        assert_eq!(b_worker, wx.id, "B must be admitted on wx");
        assert_eq!(order.lock().unwrap().as_slice(), &["B"]);
        wait_until_queued(&q, 1).await; // only A still parked

        // Free wy: now A (the only remaining, eligible) admits.
        drop(held_y);
        let (a_worker, _a_guard) = tokio::time::timeout(Duration::from_secs(2), a)
            .await
            .expect("A must admit when wy frees")
            .expect("A task panicked");
        assert_eq!(a_worker, wy.id, "A must be admitted on wy");
        assert_eq!(order.lock().unwrap().as_slice(), &["B", "A"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_acquires_never_exceed_cap_and_drain_to_zero() {
        // The whole reason the gate exists: a hard per-worker bound under load.
        // Hammer 3 workers (cap=2) with 200 concurrent acquires through the full
        // fast-path / park / hand-off / cancel-free machinery and assert no
        // worker ever exceeds cap, then everything drains to zero with no leak.
        let cap = 2;
        let q = queue(enabled(cap, None));
        let workers: Vec<Arc<Worker>> = (0..3).map(|i| worker(&format!("w{i}"))).collect();

        let mut handles = Vec::new();
        for _ in 0..200 {
            let q2 = Arc::clone(&q);
            let cands = workers.clone();
            handles.push(tokio::spawn(async move {
                let model = ModelId("m".into());
                let ctx = SelectionContext::new(&model, None);
                let (chosen, guard) = q2.acquire(&cands, &PickFirst, &ctx, "m").await.unwrap();
                // The hard bound must hold for every worker at all times.
                assert!(
                    chosen.active_load() <= cap,
                    "worker {:?} exceeded cap: {} > {cap}",
                    chosen.id,
                    chosen.active_load(),
                );
                // Hold the slot briefly so claims genuinely overlap.
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                drop(guard);
            }));
        }
        for h in handles {
            tokio::time::timeout(Duration::from_secs(10), h)
                .await
                .expect("every acquire must complete")
                .expect("task panicked");
        }

        for w in &workers {
            assert_eq!(w.active_load(), 0, "worker {:?} leaked a slot", w.id);
        }
        assert_eq!(q.queued_len(), 0, "wait queue must drain to empty");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn granted_slot_to_cancelled_receiver_is_rehanded_to_next_waiter() {
        // The riskiest path: a hand-off races a waiter's cancellation. A parks,
        // then B; we abort A at the same instant we free the only slot, so the
        // slot may be handed to A's already-dropped receiver. However that race
        // resolves (A's ticket wins `grant_tx`, or the send fails, or A receives
        // then its task is aborted and the grant's guard re-hands), the slot must
        // never leak: B must always admit and the worker must drain to zero.
        // Looped to exercise both interleavings.
        for iter in 0..40 {
            let q = queue(enabled(1, None));
            let w = worker("w");
            let model = ModelId("m".into());
            let ctx = SelectionContext::new(&model, None);
            let (_seed, held) = q
                .acquire(&[Arc::clone(&w)], &PickFirst, &ctx, "m")
                .await
                .expect("seed acquire");

            let qa = Arc::clone(&q);
            let ca = vec![Arc::clone(&w)];
            let a = tokio::spawn(async move {
                let model = ModelId("m".into());
                let ctx = SelectionContext::new(&model, None);
                let _g = qa.acquire(&ca, &PickFirst, &ctx, "m").await;
                // Hold briefly if admitted, to widen the race window.
                tokio::task::yield_now().await;
            });
            wait_until_queued(&q, 1).await;

            let qb = Arc::clone(&q);
            let cb = vec![Arc::clone(&w)];
            let b = tokio::spawn(async move {
                let model = ModelId("m".into());
                let ctx = SelectionContext::new(&model, None);
                // Drop the guard immediately on admission so the slot frees.
                let _ = qb.acquire(&cb, &PickFirst, &ctx, "m").await;
            });
            wait_until_queued(&q, 2).await;

            // Race: cancel A as the slot frees.
            a.abort();
            drop(held);

            // B must admit no matter how the race fell out.
            tokio::time::timeout(Duration::from_secs(2), b)
                .await
                .unwrap_or_else(|_| panic!("iter {iter}: B never admitted (slot leaked)"))
                .expect("B task panicked");
            let _ = a.await;

            // No leak: B has dropped its guard, the worker is idle, queue empty.
            for _ in 0..200 {
                if w.active_load() == 0 && q.queued_len() == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            assert_eq!(w.active_load(), 0, "iter {iter}: worker leaked a slot");
            assert_eq!(q.queued_len(), 0, "iter {iter}: queue did not drain");
        }
    }
}
