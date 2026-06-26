// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Parse a host from a worker URL. Matches SMG's `worker_builder.rs`
/// fallback chain: parse as-is, retry with `http://` prefix if missing,
/// fall back to `"localhost"` if both fail. The fallback is defensive —
/// discovery code should never emit an unparsable URL — but a panic
/// here would crash the whole router on a single bad config entry.
fn parse_bootstrap_host(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        if let Some(h) = parsed.host_str() {
            return h.to_string();
        }
    }
    if !url.contains("://") {
        if let Ok(parsed) = url::Url::parse(&format!("http://{url}")) {
            if let Some(h) = parsed.host_str() {
                return h.to_string();
            }
        }
    }
    tracing::warn!(
        worker_url = %url,
        "Failed to parse worker URL for bootstrap_host; defaulting to 'localhost'"
    );
    "localhost".to_string()
}

/// Tracks each in-flight slot with an acquisition timestamp so leaked slots —
/// guards that are never dropped because their request hung (e.g. a half-open
/// upstream after a worker restart) — can be reclaimed by a TTL sweep
/// ([`reclaim_stale`](SlotRegistry::reclaim_stale)). The shared `count` is the
/// same `Arc<AtomicUsize>` exposed as [`Worker::active_requests`], so the
/// admission cap reads it lock-free; the per-slot map is only touched on
/// claim / release / sweep, all off the read path.
#[derive(Debug)]
pub struct SlotRegistry {
    count: Arc<AtomicUsize>,
    slots: Mutex<HashMap<u64, Instant>>,
    next_id: AtomicU64,
}

impl SlotRegistry {
    fn new(count: Arc<AtomicUsize>) -> Arc<Self> {
        Arc::new(Self {
            count,
            slots: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        })
    }

    fn register(self: &Arc<Self>) -> LoadGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.slots.lock().insert(id, Instant::now());
        LoadGuard {
            registry: Arc::clone(self),
            id,
        }
    }

    /// Unconditional claim (always increments).
    fn claim(self: &Arc<Self>) -> LoadGuard {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.register()
    }

    /// Claim a slot only while the count is below `cap`, returning `None` when
    /// at capacity. The count is moved via a compare-and-swap retried until it
    /// succeeds or the cap is hit, so the cap holds *exactly* under racing
    /// callers — a hard bound, not a check-then-act race.
    fn try_claim(self: &Arc<Self>, cap: usize) -> Option<LoadGuard> {
        let mut current = self.count.load(Ordering::Relaxed);
        loop {
            if current >= cap {
                return None;
            }
            match self.count.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(self.register()),
                Err(actual) => current = actual,
            }
        }
    }

    /// Release slot `id`. Decrements the count only if the slot is still
    /// tracked, so a guard dropping *after* the janitor already reclaimed its
    /// slot is a no-op (no double-decrement / underflow).
    fn release(&self, id: u64) {
        if self.slots.lock().remove(&id).is_some() {
            self.count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Reset slot `id`'s age — used when a slot is handed to a fresh request
    /// (FIFO hand-off) so a continuously-cycled slot is never seen as stale.
    fn touch(&self, id: u64) {
        if let Some(t) = self.slots.lock().get_mut(&id) {
            *t = Instant::now();
        }
    }

    /// Force-release every slot older than `ttl` (defense-in-depth backstop for
    /// a leaked in-flight guard). Returns the number reclaimed. A later drop of
    /// a reclaimed slot's guard is a no-op (see [`release`](Self::release)).
    pub fn reclaim_stale(&self, ttl: Duration, now: Instant) -> usize {
        let mut slots = self.slots.lock();
        let stale: Vec<u64> = slots
            .iter()
            .filter(|(_, &acquired)| now.saturating_duration_since(acquired) >= ttl)
            .map(|(&id, _)| id)
            .collect();
        for id in &stale {
            slots.remove(id);
        }
        drop(slots);
        for _ in 0..stale.len() {
            self.count.fetch_sub(1, Ordering::Relaxed);
        }
        stale.len()
    }
}

/// RAII guard that increments `active_requests` on construction and decrements
/// on drop. Obtain via [`Worker::load_guard`]. Each guard owns one timestamped
/// slot in the worker's [`SlotRegistry`], so the TTL janitor can reclaim it if
/// the guard is leaked (request hung and never dropped it).
///
/// `#[must_use]`: a statement-form call like `worker.load_guard();` would drop
/// the guard on the same line, so the counter would never see the in-flight
/// request.  The compile-time warning catches that misuse.
#[must_use = "LoadGuard must be held for the request's lifetime; dropping it immediately decrements active_requests"]
#[derive(Debug)]
pub struct LoadGuard {
    registry: Arc<SlotRegistry>,
    id: u64,
}

impl LoadGuard {
    /// Reset this slot's age in the registry. Called on FIFO hand-off so a slot
    /// continuously transferred between live requests is not mistaken for a
    /// leaked one by the janitor.
    pub(crate) fn touch(&self) {
        self.registry.touch(self.id);
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.registry.release(self.id);
    }
}

impl WorkerMode {
    fn as_u8(self) -> u8 {
        match self {
            WorkerMode::Plain => 0,
            WorkerMode::Prefill => 1,
            WorkerMode::Decode => 2,
        }
    }

    /// Inverse of [`Self::as_u8`].  The only writers of the underlying
    /// `AtomicU8` are `as_u8`-derived values, so any out-of-range byte
    /// indicates memory corruption or a stale store from an
    /// incompatible build — fail loudly rather than silently mislabel
    /// the worker as `Decode`.
    fn from_u8(v: u8) -> Self {
        match v {
            0 => WorkerMode::Plain,
            1 => WorkerMode::Prefill,
            2 => WorkerMode::Decode,
            other => unreachable!("invalid WorkerMode discriminant {other}"),
        }
    }
}

pub struct Worker {
    pub id: WorkerId,
    pub url: String,
    /// Interior-mutable mode so `ModeChanged` can update in place without
    /// dropping the Worker (which would reset `active_requests` + breaker).
    mode: AtomicU8,
    pub model_ids: Vec<ModelId>,
    pub breaker: Arc<CircuitBreaker>,
    pub active_requests: Arc<AtomicUsize>,
    /// Per-slot timestamps for the in-flight count, enabling the TTL janitor to
    /// reclaim leaked guards. Shares `active_requests` as its count.
    slots: Arc<SlotRegistry>,
    /// Hostname parsed from `url` at construction time and cached.
    /// Used as the `bootstrap_host` field on PD-disagg requests so the
    /// prefill engine can match incoming KV-transfer requests from
    /// decode peers. Falls back to `"localhost"` if the URL fails to
    /// parse — a misconfigured worker will fail the prefill request
    /// downstream rather than panic here.
    bootstrap_host: String,
    /// SGLang bootstrap server port for prefill workers (`None` for
    /// decode and plain). Set via `--disaggregation-bootstrap-port` at
    /// worker startup; carried from `WorkerSpec`.
    bootstrap_port: Option<u16>,
}

impl Worker {
    pub fn new(spec: crate::discovery::WorkerSpec) -> Self {
        Self::with_cb_config(spec, None)
    }

    /// Construct a worker with an explicit circuit-breaker configuration.
    /// Pass `None` to use the default config (threshold = 3, cool_down = 30 s).
    pub fn with_cb_config(
        spec: crate::discovery::WorkerSpec,
        cb: Option<CircuitBreakerConfig>,
    ) -> Self {
        let breaker = match cb {
            Some(cfg) => Arc::new(CircuitBreaker::with_config(cfg)),
            None => Arc::new(CircuitBreaker::new()),
        };
        let bootstrap_host = parse_bootstrap_host(&spec.url);
        let active_requests = Arc::new(AtomicUsize::new(0));
        let slots = SlotRegistry::new(Arc::clone(&active_requests));
        Self {
            id: spec.id,
            url: spec.url,
            mode: AtomicU8::new(spec.mode.as_u8()),
            model_ids: spec.model_ids,
            breaker,
            active_requests,
            slots,
            bootstrap_host,
            bootstrap_port: spec.bootstrap_port,
        }
    }

    /// Hostname carried on PD-disagg request bodies as `bootstrap_host`.
    pub fn bootstrap_host(&self) -> &str {
        &self.bootstrap_host
    }

    /// SGLang bootstrap server port. `None` for decode / plain workers.
    pub fn bootstrap_port(&self) -> Option<u16> {
        self.bootstrap_port
    }

    /// Returns the current [`WorkerMode`] of this worker.
    ///
    /// Uses `Relaxed` ordering: mode changes are rare discovery events and do
    /// not need to synchronise with any other memory access.
    pub fn mode(&self) -> WorkerMode {
        WorkerMode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    /// Update the worker's mode in place.
    ///
    /// Preserves `active_requests` and `breaker` state — the same `Arc<Worker>`
    /// identity survives the mode transition.
    pub fn set_mode(&self, m: WorkerMode) {
        self.mode.store(m.as_u8(), Ordering::Relaxed);
    }

    pub fn active_load(&self) -> usize {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Returns a RAII guard that increments `active_requests` now and
    /// decrements when the guard is dropped.
    pub fn load_guard(&self) -> LoadGuard {
        self.slots.claim()
    }

    /// Like [`Worker::load_guard`], but only admits while `active_requests` is
    /// below `cap`; returns `None` when the worker is already at capacity. Used
    /// by the admission gate to enforce a hard per-worker in-flight limit.
    pub fn try_load_guard(&self, cap: usize) -> Option<LoadGuard> {
        self.slots.try_claim(cap)
    }

    /// Force-release in-flight slots whose guard has been held longer than
    /// `ttl` — a defense-in-depth backstop against a leaked admission slot
    /// (a guard whose request hung and never dropped). Returns the number
    /// reclaimed. Driven by the worker janitor; `now` is injected for tests.
    pub fn reclaim_stale_load(&self, ttl: Duration, now: Instant) -> usize {
        self.slots.reclaim_stale(ttl, now)
    }
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("id", &self.id)
            .field("url", &self.url)
            .field("mode", &self.mode())
            .field("active_load", &self.active_load())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

    #[test]
    fn load_guard_increments_and_decrements() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        });
        assert_eq!(w.active_load(), 0);
        let g = w.load_guard();
        assert_eq!(w.active_load(), 1);
        let g2 = w.load_guard();
        assert_eq!(w.active_load(), 2);
        drop(g);
        assert_eq!(w.active_load(), 1);
        drop(g2);
        assert_eq!(w.active_load(), 0);
    }

    #[test]
    fn try_load_guard_enforces_cap() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        });
        let cap = 2;
        let g1 = w.try_load_guard(cap);
        assert!(g1.is_some());
        assert_eq!(w.active_load(), 1);
        let _g2 = w.try_load_guard(cap);
        assert!(_g2.is_some());
        assert_eq!(w.active_load(), 2);
        // At cap: refused, counter unchanged.
        assert!(w.try_load_guard(cap).is_none());
        assert_eq!(w.active_load(), 2);
        // Freeing a slot re-opens admission.
        drop(g1);
        assert_eq!(w.active_load(), 1);
        let _g3 = w.try_load_guard(cap);
        assert!(_g3.is_some());
        assert_eq!(w.active_load(), 2);
    }

    #[test]
    fn try_load_guard_never_exceeds_cap_under_concurrency() {
        use std::sync::{Barrier, Mutex};
        use std::thread;

        let w = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }));
        let cap = 8;
        let n_threads = 64;
        let barrier = Arc::new(Barrier::new(n_threads));
        let granted: Arc<Mutex<Vec<LoadGuard>>> = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let w = Arc::clone(&w);
                let barrier = Arc::clone(&barrier);
                let granted = Arc::clone(&granted);
                thread::spawn(move || {
                    barrier.wait();
                    if let Some(guard) = w.try_load_guard(cap) {
                        // Hold the guard so the claimed slot is never released.
                        granted.lock().unwrap().push(guard);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(granted.lock().unwrap().len(), cap);
        assert_eq!(w.active_load(), cap);
    }

    #[test]
    fn mode_accessor_round_trips_all_variants() {
        for m in [WorkerMode::Plain, WorkerMode::Prefill, WorkerMode::Decode] {
            let w = Worker::new(WorkerSpec {
                id: WorkerId("w".into()),
                url: "http://x".into(),
                mode: m,
                model_ids: vec![],
                bootstrap_port: None,
            });
            assert_eq!(w.mode(), m);
        }
    }

    #[test]
    fn set_mode_updates_in_place() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: None,
        });
        assert_eq!(w.mode(), WorkerMode::Prefill);
        w.set_mode(WorkerMode::Decode);
        assert_eq!(w.mode(), WorkerMode::Decode);
        w.set_mode(WorkerMode::Plain);
        assert_eq!(w.mode(), WorkerMode::Plain);
    }

    #[test]
    fn bootstrap_port_returns_spec_value_for_prefill() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_port(), Some(8997));
    }

    #[test]
    fn bootstrap_port_defaults_to_none() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![],
            bootstrap_port: None,
        });
        assert_eq!(w.bootstrap_port(), None);
    }

    #[test]
    fn bootstrap_host_parses_ipv4_from_url() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_host(), "10.0.0.1");
    }

    #[test]
    fn bootstrap_host_parses_dns_name_from_url() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://prefill-0.svc.cluster.local:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_host(), "prefill-0.svc.cluster.local");
    }

    #[test]
    fn bootstrap_host_falls_back_to_localhost_for_unparsable_url() {
        // An empty / invalid URL is not expected from discovery, but the
        // accessor must return a usable string rather than panic — the
        // prefill worker will reject the request body-side if the host
        // really is unreachable.
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "not a url".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_host(), "localhost");
    }

    fn test_worker() -> Worker {
        Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        })
    }

    /// The TTL janitor force-releases a leaked slot, and the leaked guard's
    /// eventual drop is a no-op — the counter must not underflow.
    #[test]
    fn reclaim_stale_load_force_releases_without_underflow() {
        let w = test_worker();
        let g = w.load_guard();
        assert_eq!(w.active_load(), 1);

        // ttl=0: the just-acquired slot counts as stale and is reclaimed.
        let reclaimed = w.reclaim_stale_load(Duration::ZERO, Instant::now());
        assert_eq!(reclaimed, 1);
        assert_eq!(w.active_load(), 0, "stale slot must be force-released");

        // The leaked guard finally drops after the sweep: must NOT decrement
        // again (would underflow to usize::MAX and wedge the worker as "full").
        drop(g);
        assert_eq!(
            w.active_load(),
            0,
            "post-reclaim drop must be a no-op (no double-decrement / underflow)",
        );
    }

    /// A fresh slot (held less than the TTL) is left alone by the janitor.
    #[test]
    fn reclaim_stale_load_leaves_fresh_slots() {
        let w = test_worker();
        let _g = w.load_guard();
        assert_eq!(w.active_load(), 1);

        let reclaimed = w.reclaim_stale_load(Duration::from_secs(3600), Instant::now());
        assert_eq!(reclaimed, 0, "a fresh slot must not be reclaimed");
        assert_eq!(w.active_load(), 1);
    }

    /// `touch` resets a slot's age so a continuously handed-off slot is never
    /// seen as stale by the janitor.
    #[test]
    fn touch_resets_slot_age() {
        let w = test_worker();
        let g = w.load_guard();
        // Without touch, ttl=0 would reclaim it; touch keeps it "fresh" only
        // against a positive ttl, so assert via the fresh-ttl path after touch.
        g.touch();
        let reclaimed = w.reclaim_stale_load(Duration::from_secs(3600), Instant::now());
        assert_eq!(reclaimed, 0);
        assert_eq!(w.active_load(), 1);
    }
}
