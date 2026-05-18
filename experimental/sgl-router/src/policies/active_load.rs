// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-worker active-load tracking with RAII guards and a stale-request
//! janitor.
//!
//! The cache-aware-zmq policy ([`super::cache_aware_zmq`]) needs to combine
//! the M3 hash tree's overlap score with a per-worker load signal. The
//! per-worker `Worker::active_requests` counter (from M2) tracks one axis —
//! number of in-flight HTTP requests — and is already drop-safe through
//! [`crate::workers::LoadGuard`].
//!
//! M4 adds two things on top of that:
//!
//! 1. **Per-request bookkeeping** keyed on a `RequestId` so a background
//!    janitor can sweep requests that outlive the configured
//!    `stale_request_timeout` and decrement the counters they were holding.
//!    Without this, a request whose `LoadGuard` is leaked (proxy task
//!    panics before the future drops, server hits a panic-catching
//!    middleware, etc.) would inflate a worker's load forever.
//! 2. **Two-axis tracking** so PD-disaggregation can score prefill (token
//!    count) separately from decode (block count). The two counters share
//!    the same registry shape; we expose them as a single
//!    [`ActiveLoadGuard`] holding both so the proxy's hot path mints one
//!    guard per request rather than two.
//!
//! # Drop semantics
//!
//! Guards decrement on drop AND remove themselves from the request tracker
//! so the janitor never double-decrements. The implementation uses
//! `Option<RegistryHandle>` inside the guard: the janitor's `expire_now`
//! path takes the handle (rendering subsequent drop a no-op for that
//! request), while normal RAII drop also takes the handle (rendering
//! subsequent janitor sweep a no-op). Either path may run first — the
//! other becomes a no-op. Rust's affine type system makes a literal
//! double-drop of the same guard value unreachable.
//!
//! # Clock injection
//!
//! [`ActiveLoadRegistry::new`] is generic over the clock so tests can drive
//! the janitor deterministically. Production wires a `SystemTimeClock`;
//! tests use a `MockClock`. The `Instant`-based timestamp on registration
//! is sufficient for the timeout comparison (monotonic), so the clock
//! abstraction is just two methods: `now()` and an associated `Instant`
//! type whose `duration_since(other)` returns the wall-clock delta.

use crate::discovery::WorkerId;
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Unique identifier for an in-flight request. Minted by
/// [`ActiveLoadRegistry::register`] and carried inside [`ActiveLoadGuard`]
/// so the janitor can address one request at a time.
#[derive(Clone, Eq, Hash, PartialEq, Debug)]
pub struct RequestId(pub Uuid);

impl RequestId {
    pub fn new_v4() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Per-worker counters: one for prefill (token) load, one for decode (block)
/// load. The two axes are tracked separately so cache-aware-zmq can score
/// prefill candidates by token load and decode candidates by block load
/// without each axis spamming through the other's counter.
///
/// Production tracks **active requests** as the unit (count of in-flight
/// requests pinning the worker), not raw token / block counts — until M5
/// wires real prompt-token / completion-block accounting into the proxy
/// path. The two axes will become meaningful once the proxy starts passing
/// `prompt_tokens` and `output_blocks` through to `register`.
#[derive(Debug, Default)]
struct WorkerCounters {
    prefill_load: AtomicUsize,
    decode_load: AtomicUsize,
}

/// Per-request bookkeeping the janitor consults to find expired requests.
#[derive(Debug)]
struct RequestEntry {
    worker: WorkerId,
    prefill_load: usize,
    decode_load: usize,
    registered_at: Instant,
}

/// Clock abstraction so tests can drive the janitor deterministically.
///
/// We only need `now()`; ordering is via `Instant::duration_since` which
/// already exists on the std type. Production implementers return the
/// monotonic system instant; tests return whatever value `MockClock` is set
/// to via `set_now`.
pub trait Clock: Send + Sync + std::fmt::Debug {
    fn now(&self) -> Instant;
}

/// Monotonic system clock used in production.
#[derive(Debug, Default, Clone)]
pub struct SystemTimeClock;

impl Clock for SystemTimeClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Test-only clock that returns a caller-controlled instant.
///
/// Wrapped in `parking_lot::Mutex` because tests cross await points; the
/// type is `Send + Sync` so it can be stored behind `Arc<dyn Clock>`.
#[derive(Debug)]
pub struct MockClock {
    now: parking_lot::Mutex<Instant>,
}

impl MockClock {
    pub fn new(start: Instant) -> Self {
        Self {
            now: parking_lot::Mutex::new(start),
        }
    }

    /// Advance the clock by `delta`. Returns the new `now`.
    pub fn advance(&self, delta: Duration) -> Instant {
        let mut guard = self.now.lock();
        *guard += delta;
        *guard
    }
}

impl Clock for MockClock {
    fn now(&self) -> Instant {
        *self.now.lock()
    }
}

/// Registry of in-flight requests + per-worker active-load counters.
///
/// Constructed once per `AppContext`; the cache-aware-zmq policy reads
/// per-worker `prefill_load` / `decode_load` from here when scoring
/// candidates, and the proxy holds an [`ActiveLoadGuard`] per request so
/// counters decrement on drop. A background task periodically calls
/// [`Self::sweep_stale`] to evict requests that outlived
/// `stale_request_timeout`.
#[derive(Debug)]
pub struct ActiveLoadRegistry {
    workers: DashMap<WorkerId, Arc<WorkerCounters>>,
    requests: DashMap<RequestId, RequestEntry>,
    clock: Arc<dyn Clock>,
    stale_request_timeout: Duration,
}

impl ActiveLoadRegistry {
    /// Construct an [`ActiveLoadRegistry`] wrapped in an [`Arc`].
    ///
    /// The registry is always shared (proxy + janitor + selector all hold
    /// the same instance), so the public constructor mints the `Arc`
    /// directly to remove an easy footgun where callers forget to wrap
    /// it. Tests that need the inner type for direct field access also
    /// receive `Arc<Self>`.
    pub fn new(clock: Arc<dyn Clock>, stale_request_timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            workers: DashMap::new(),
            requests: DashMap::new(),
            clock,
            stale_request_timeout,
        })
    }

    /// Default-config registry: monotonic system clock + 5-minute stale
    /// timeout. Convenience constructor for production callers; tests use
    /// [`Self::new`] with a `MockClock`.
    ///
    /// 5 minutes is comfortable above 99p generation tail latency while
    /// bounding leak-induced load inflation.
    pub fn with_defaults() -> Arc<Self> {
        Self::new(
            Arc::new(SystemTimeClock) as Arc<dyn Clock>,
            Duration::from_secs(5 * 60),
        )
    }

    /// Register a new in-flight request and return a guard that holds the
    /// active-load counters up. The guard's drop / explicit complete path
    /// decrements the counters and removes the request entry.
    pub fn register(
        self: &Arc<Self>,
        worker: WorkerId,
        prefill_load: usize,
        decode_load: usize,
    ) -> ActiveLoadGuard {
        let request_id = RequestId::new_v4();
        let counters = self
            .workers
            .entry(worker.clone())
            .or_insert_with(|| Arc::new(WorkerCounters::default()))
            .value()
            .clone();
        counters
            .prefill_load
            .fetch_add(prefill_load, Ordering::Relaxed);
        counters
            .decode_load
            .fetch_add(decode_load, Ordering::Relaxed);
        self.requests.insert(
            request_id.clone(),
            RequestEntry {
                worker: worker.clone(),
                prefill_load,
                decode_load,
                registered_at: self.clock.now(),
            },
        );
        ActiveLoadGuard {
            registry: Some(Arc::clone(self)),
            request_id: Some(request_id),
            worker,
        }
    }

    /// Drop a worker's per-worker counters entry. Called from
    /// [`crate::workers::manager`] on `DiscoveryEvent::Removed` so the
    /// `WorkerCounters` slot for a now-gone worker does not leak.
    ///
    /// Guards still alive for that worker remain valid; their drop tries
    /// `workers.get(&entry.worker)` which returns `None`, and the
    /// per-request `requests` entry is still removed cleanly. A subsequent
    /// `register` for the same `WorkerId` reinitializes the slot to 0 —
    /// the in-flight guards' loads are NOT re-added, by design (the
    /// worker is gone, those loads no longer mean anything).
    pub fn forget_worker(&self, id: &WorkerId) {
        self.workers.remove(id);
    }

    /// Returns `true` if the registry currently has a per-worker
    /// counters entry for `id`. Cheap; intended for tests + diagnostics.
    pub fn is_known(&self, id: &WorkerId) -> bool {
        self.workers.contains_key(id)
    }

    /// Current prefill load (sum across in-flight requests) for a worker.
    pub fn prefill_load(&self, worker: &WorkerId) -> usize {
        self.workers
            .get(worker)
            .map(|c| c.prefill_load.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Current decode load (sum across in-flight requests) for a worker.
    pub fn decode_load(&self, worker: &WorkerId) -> usize {
        self.workers
            .get(worker)
            .map(|c| c.decode_load.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Number of in-flight requests tracked (cheap; useful for tests +
    /// metrics).
    pub fn inflight_count(&self) -> usize {
        self.requests.len()
    }

    /// Sweep entries whose `registered_at + stale_request_timeout` is in
    /// the past. Returns the number of entries expired.
    ///
    /// Decrements both axes' worker counters for each expired entry. Safe
    /// to call concurrently with `register` and with guard drops — each
    /// `remove` operation is atomic and the per-worker counters are
    /// `AtomicUsize` so partial visibility cannot under-decrement.
    pub fn sweep_stale(&self) -> usize {
        let now = self.clock.now();
        let mut expired_ids: Vec<RequestId> = Vec::new();
        for entry in self.requests.iter() {
            if now.duration_since(entry.value().registered_at) >= self.stale_request_timeout {
                expired_ids.push(entry.key().clone());
            }
        }
        let mut count = 0;
        for id in expired_ids {
            // Use `remove`: if the guard's drop concurrently removed the
            // entry between our scan and this point, the second remove
            // returns `None` and we skip (no double-decrement).
            if let Some((_, entry)) = self.requests.remove(&id) {
                if let Some(counters) = self.workers.get(&entry.worker) {
                    counters
                        .prefill_load
                        .fetch_sub(entry.prefill_load, Ordering::Relaxed);
                    counters
                        .decode_load
                        .fetch_sub(entry.decode_load, Ordering::Relaxed);
                }
                count += 1;
                tracing::warn!(
                    request_id = %id,
                    worker = %entry.worker,
                    prefill_load = entry.prefill_load,
                    decode_load = entry.decode_load,
                    "stale request swept by active-load janitor",
                );
            }
        }
        count
    }
}

/// Spawn a background janitor task that periodically calls
/// [`ActiveLoadRegistry::sweep_stale`].
///
/// Returns a [`JanitorHandle`] that owns the join handle and a cancellation
/// token. Dropping the handle cancels the task; calling
/// [`JanitorHandle::shutdown`] cancels and awaits the join.
///
/// `interval` is the wall-clock cadence of the sweep. A sensible default
/// is half the configured `stale_request_timeout` so an expired entry is
/// reaped within 1.5× the timeout in the worst case. Pass a fresh
/// `Arc<ActiveLoadRegistry>` (cloned from the shared one held in
/// `AppContext`).
pub fn spawn_janitor(registry: Arc<ActiveLoadRegistry>, interval: Duration) -> JanitorHandle {
    let cancel = tokio_util::sync::CancellationToken::new();
    let cancel_for_task = cancel.clone();
    let join = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                biased;
                _ = cancel_for_task.cancelled() => {
                    tracing::debug!("active-load janitor: shutdown requested");
                    return;
                }
                _ = ticker.tick() => {
                    let n = registry.sweep_stale();
                    if n > 0 {
                        tracing::info!(
                            swept = n,
                            "active-load janitor: removed stale requests",
                        );
                    }
                }
            }
        }
    });
    JanitorHandle {
        cancel,
        join: Some(join),
    }
}

/// Owner handle for the background janitor task. Dropping the handle
/// cancels the task; calling [`Self::shutdown`] cancels AND awaits join,
/// giving callers a clean shutdown path.
#[must_use = "JanitorHandle owns the background task; dropping it cancels the janitor"]
pub struct JanitorHandle {
    cancel: tokio_util::sync::CancellationToken,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl JanitorHandle {
    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        if let Some(j) = self.join.take() {
            // 2 s ceiling guards against a runtime-teardown hang; the
            // janitor exits within one tick of `cancelled()`.
            let _ = tokio::time::timeout(Duration::from_secs(2), j).await;
        }
    }
}

impl Drop for JanitorHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// RAII guard returned by [`ActiveLoadRegistry::register`].
///
/// `#[must_use]`: a statement-form `registry.register(...)` would drop the
/// guard on the same line and decrement the counter before the request
/// actually executed, defeating the purpose. The compile-time warning
/// catches that misuse.
#[must_use = "ActiveLoadGuard must be held for the request's lifetime; dropping it immediately decrements counters"]
#[derive(Debug)]
pub struct ActiveLoadGuard {
    registry: Option<Arc<ActiveLoadRegistry>>,
    /// `None` after the janitor expired this request — drop becomes a
    /// no-op in that case. The guard keeps only the `RequestId`; the
    /// per-axis amounts live in the registry's `RequestEntry` so drop
    /// and the janitor consult the same source of truth.
    request_id: Option<RequestId>,
    worker: WorkerId,
}

impl ActiveLoadGuard {
    /// Read-only accessor (mainly for tests + diagnostic logging).
    pub fn worker(&self) -> &WorkerId {
        &self.worker
    }
}

impl Drop for ActiveLoadGuard {
    fn drop(&mut self) {
        // If the janitor already expired this request (or `expire_now` was
        // called explicitly), `request_id` is `None` and we skip — the
        // janitor already decremented the counters.
        let (Some(registry), Some(id)) = (self.registry.take(), self.request_id.take()) else {
            return;
        };
        // `remove` returns `Some` exactly once; if the janitor races us
        // and wins, we skip the decrement here.
        if let Some((_, entry)) = registry.requests.remove(&id) {
            if let Some(counters) = registry.workers.get(&entry.worker) {
                counters
                    .prefill_load
                    .fetch_sub(entry.prefill_load, Ordering::Relaxed);
                counters
                    .decode_load
                    .fetch_sub(entry.decode_load, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn registry_with_mock_clock(timeout: Duration) -> (Arc<ActiveLoadRegistry>, Arc<MockClock>) {
        let clock = Arc::new(MockClock::new(Instant::now()));
        let registry = ActiveLoadRegistry::new(Arc::clone(&clock) as Arc<dyn Clock>, timeout);
        (registry, clock)
    }

    #[test]
    fn single_worker_increment_decrement_round_trip() {
        let (registry, _) = registry_with_mock_clock(Duration::from_secs(60));
        let w = WorkerId("w0".into());
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
        let g = registry.register(w.clone(), 100, 5);
        assert_eq!(registry.prefill_load(&w), 100);
        assert_eq!(registry.decode_load(&w), 5);
        assert_eq!(registry.inflight_count(), 1);
        drop(g);
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
        assert_eq!(registry.inflight_count(), 0);
    }

    #[test]
    fn two_concurrent_guards_increment_to_2_then_drop_to_0() {
        let (registry, _) = registry_with_mock_clock(Duration::from_secs(60));
        let w = WorkerId("w0".into());
        let g1 = registry.register(w.clone(), 10, 1);
        let g2 = registry.register(w.clone(), 20, 2);
        assert_eq!(registry.prefill_load(&w), 30);
        assert_eq!(registry.decode_load(&w), 3);
        assert_eq!(registry.inflight_count(), 2);
        drop(g1);
        assert_eq!(registry.prefill_load(&w), 20);
        assert_eq!(registry.decode_load(&w), 2);
        drop(g2);
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
    }

    #[test]
    fn guard_decrements_on_implicit_drop_via_scope_exit() {
        let (registry, _) = registry_with_mock_clock(Duration::from_secs(60));
        let w = WorkerId("w0".into());
        {
            let _g = registry.register(w.clone(), 7, 1);
            assert_eq!(registry.prefill_load(&w), 7);
        }
        assert_eq!(registry.prefill_load(&w), 0);
    }

    #[test]
    fn distinct_workers_are_isolated() {
        let (registry, _) = registry_with_mock_clock(Duration::from_secs(60));
        let w0 = WorkerId("w0".into());
        let w1 = WorkerId("w1".into());
        let _g0 = registry.register(w0.clone(), 5, 0);
        let _g1 = registry.register(w1.clone(), 11, 0);
        assert_eq!(registry.prefill_load(&w0), 5);
        assert_eq!(registry.prefill_load(&w1), 11);
    }

    /// Gap closer #2: double-drop safety.
    ///
    /// Rust's affine type system makes a literal double-drop of the same
    /// `ActiveLoadGuard` value impossible — the compiler rejects
    /// `drop(g); drop(g);`. The interesting property is that the
    /// registry's own bookkeeping never under-decrements, even if the
    /// janitor and a guard's drop race. We assert that by simulating the
    /// race: the janitor wins (entry removed via `sweep_stale`), then the
    /// guard's drop runs — must be a no-op.
    #[test]
    fn janitor_then_guard_drop_does_not_underflow() {
        let (registry, clock) = registry_with_mock_clock(Duration::from_secs(1));
        let w = WorkerId("w0".into());
        let g = registry.register(w.clone(), 50, 5);
        clock.advance(Duration::from_secs(2));
        let n = registry.sweep_stale();
        assert_eq!(n, 1);
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
        // Janitor already removed entry; guard's drop must not under-flow.
        drop(g);
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
        assert_eq!(registry.inflight_count(), 0);
    }

    /// Gap closer #4: stale-request janitor expiry zeroes counters.
    #[test]
    fn janitor_expires_stale_requests() {
        let (registry, clock) = registry_with_mock_clock(Duration::from_secs(5));
        let w = WorkerId("w0".into());
        let _g = registry.register(w.clone(), 100, 4);
        assert_eq!(registry.prefill_load(&w), 100);
        // Just below the threshold — no expiry.
        clock.advance(Duration::from_secs(4));
        assert_eq!(registry.sweep_stale(), 0);
        assert_eq!(registry.prefill_load(&w), 100);
        // Past the threshold — expires.
        clock.advance(Duration::from_secs(2));
        assert_eq!(registry.sweep_stale(), 1);
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
    }

    #[test]
    fn janitor_is_idempotent_on_double_run() {
        let (registry, clock) = registry_with_mock_clock(Duration::from_secs(1));
        let w = WorkerId("w0".into());
        let _g = registry.register(w.clone(), 7, 0);
        clock.advance(Duration::from_secs(2));
        assert_eq!(registry.sweep_stale(), 1);
        // Second run finds nothing to do.
        assert_eq!(registry.sweep_stale(), 0);
        assert_eq!(registry.prefill_load(&w), 0);
    }

    #[test]
    fn janitor_leaves_fresh_requests_alone() {
        let (registry, clock) = registry_with_mock_clock(Duration::from_secs(60));
        let w = WorkerId("w0".into());
        let _g = registry.register(w.clone(), 50, 0);
        clock.advance(Duration::from_secs(1));
        assert_eq!(registry.sweep_stale(), 0);
        assert_eq!(registry.prefill_load(&w), 50);
    }

    /// Spawned janitor sweeps stale entries on its periodic tick. Uses
    /// real (short) sleeps so that the tokio interval timer fires; the
    /// registry's clock is the real `SystemTimeClock` so both views of
    /// "now" advance together. 200 ms total wait is comfortably above
    /// the 30 ms timeout we configure.
    #[tokio::test]
    async fn spawn_janitor_sweeps_stale_entries() {
        let clock: Arc<dyn Clock> = Arc::new(SystemTimeClock);
        let registry = ActiveLoadRegistry::new(clock, Duration::from_millis(30));
        let w = WorkerId("w0".into());
        let _g = registry.register(w.clone(), 50, 2);
        assert_eq!(registry.inflight_count(), 1);

        let handle = spawn_janitor(Arc::clone(&registry), Duration::from_millis(20));
        // Wait long enough for at least one sweep to find the entry
        // past the 30 ms timeout. 200 ms allows ~9 ticks of slack.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(registry.inflight_count(), 0, "janitor should have swept");
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
        handle.shutdown().await;
    }

    /// Shutdown is idempotent — calling `shutdown` once must cleanly
    /// terminate the janitor without hanging.
    #[tokio::test]
    async fn spawn_janitor_shutdown_is_clean() {
        let clock: Arc<dyn Clock> = Arc::new(SystemTimeClock);
        let registry = ActiveLoadRegistry::new(clock, Duration::from_secs(60));
        let handle = spawn_janitor(Arc::clone(&registry), Duration::from_millis(100));
        // Verify shutdown completes within a generous bound.
        let r = tokio::time::timeout(Duration::from_secs(2), handle.shutdown()).await;
        assert!(r.is_ok(), "janitor shutdown timed out");
    }

    /// Task B: `forget_worker` drops the per-worker counters entry so a
    /// disappeared worker does not leak a `WorkerCounters` slot.
    /// Existing guards still drop cleanly (no underflow / panic).
    #[test]
    fn forget_worker_drops_counters_entry() {
        let (registry, _) = registry_with_mock_clock(Duration::from_secs(60));
        let w = WorkerId("w0".into());
        let g = registry.register(w.clone(), 7, 2);
        assert!(registry.is_known(&w), "worker is known after register");
        assert_eq!(registry.prefill_load(&w), 7);

        registry.forget_worker(&w);
        assert!(
            !registry.is_known(&w),
            "worker counters entry must be removed after forget_worker",
        );
        // Per-worker counters are gone, so the load query reads 0.
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);

        // The guard still has a live request entry pointing at the
        // now-forgotten worker. Drop must NOT panic; the registry's
        // worker map being empty for this id is treated as the
        // "already-cleaned-up" terminal state.
        drop(g);
        assert_eq!(
            registry.inflight_count(),
            0,
            "guard's drop must still tear down the request entry",
        );
    }

    /// Task B: forgetting an unknown worker is a no-op (idempotent).
    /// The manager calls `forget_worker` unconditionally on `Removed`,
    /// so a double-Removed event or a Removed for a never-seen worker
    /// must not panic.
    #[test]
    fn forget_unknown_worker_is_noop() {
        let (registry, _) = registry_with_mock_clock(Duration::from_secs(60));
        registry.forget_worker(&WorkerId("never-registered".into()));
        // No assertion beyond "did not panic"; the body of the test
        // exercises the contract.
    }

    /// Concurrent stress: many guards on the same worker should leave the
    /// counter back at zero once all guards drop.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_register_drop_returns_to_zero() {
        let (registry, _) = registry_with_mock_clock(Duration::from_secs(60));
        let w = WorkerId("w0".into());
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..100 {
            let r = Arc::clone(&registry);
            let wid = w.clone();
            set.spawn(async move {
                for _ in 0..100 {
                    let _g = r.register(wid.clone(), 1, 1);
                    // Yield occasionally so tasks interleave.
                    tokio::task::yield_now().await;
                }
            });
        }
        while set.join_next().await.is_some() {}
        assert_eq!(registry.prefill_load(&w), 0);
        assert_eq!(registry.decode_load(&w), 0);
        assert_eq!(registry.inflight_count(), 0);
    }
}
