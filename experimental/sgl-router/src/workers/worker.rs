// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

/// Tracks each in-flight slot with an acquisition timestamp so a routing
/// policy can ask how many slots were claimed recently
/// ([`count_acquired_since`](SlotRegistry::count_acquired_since)). The
/// registry is separate from [`Worker::active_requests`]: ordinary load
/// tracking stays lock-free, while policies that correct an engine snapshot
/// explicitly opt into timestamp tracking.
#[derive(Debug)]
pub struct SlotRegistry {
    slots: Mutex<HashMap<u64, Instant>>,
    next_id: AtomicU64,
}

impl SlotRegistry {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            slots: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        })
    }

    /// Records one timestamped slot and returns its identity.
    fn claim(&self) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.slots.lock().unwrap().insert(id, Instant::now());
        id
    }

    /// Releases timestamped slot `id`.
    fn release(&self, id: u64) {
        self.slots.lock().unwrap().remove(&id);
    }

    /// Count of currently-claimed slots acquired at or after `since`. Used to
    /// bound how many of this worker's in-flight requests are dispatches the
    /// engine hasn't reported back on yet (see
    /// `crate::policies::cache_aware_zmq::WorkerLoads::load_of`), rather than
    /// adding the full in-flight count — which would also include long-held
    /// slots from slow-draining streaming responses (see
    /// `crate::proxy::Proxy::forward_streaming_to`'s `stream_guards` doc)
    /// that the engine's own last report likely already accounts for.
    pub fn count_acquired_since(&self, since: Instant) -> usize {
        self.slots
            .lock()
            .unwrap()
            .values()
            .filter(|&&t| t >= since)
            .count()
    }
}

/// RAII guard that increments `active_requests` on construction and decrements
/// on drop. Obtain via [`Worker::load_guard`]. Policies that need to correct
/// an engine snapshot use the crate-private timestamped variant.
///
/// `#[must_use]`: a statement-form call like `worker.load_guard();` would
/// drop the guard on the same line, so the counter would never see the
/// in-flight request.  The compile-time warning catches that misuse.
#[must_use = "LoadGuard must be held for the request's lifetime; dropping it immediately decrements active_requests"]
pub struct LoadGuard {
    active_requests: Arc<AtomicUsize>,
    tracked_slot: Option<(Arc<SlotRegistry>, u64)>,
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        if let Some((registry, id)) = &self.tracked_slot {
            registry.release(*id);
        }
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
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
    /// Timestamped ledger for requests whose policy reads Engine Load;
    /// answers [`Worker::slots_acquired_since`].
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
        let slots = SlotRegistry::new();
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

    /// Number of this worker's currently in-flight requests dispatched at or
    /// after `since`. See [`SlotRegistry::count_acquired_since`].
    pub fn slots_acquired_since(&self, since: Instant) -> usize {
        self.slots.count_acquired_since(since)
    }

    /// Returns a RAII guard that increments `active_requests` now and
    /// decrements when the guard is dropped.
    pub fn load_guard(&self) -> LoadGuard {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        LoadGuard {
            active_requests: Arc::clone(&self.active_requests),
            tracked_slot: None,
        }
    }

    /// Returns a load guard that also records when the request was dispatched.
    pub(crate) fn timestamped_load_guard(&self) -> LoadGuard {
        let slot_id = self.slots.claim();
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        LoadGuard {
            active_requests: Arc::clone(&self.active_requests),
            tracked_slot: Some((Arc::clone(&self.slots), slot_id)),
        }
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
    use std::time::Duration;

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
    fn plain_load_guard_does_not_track_a_timestamped_slot() {
        let w = test_worker();
        let cutoff = Instant::now() - Duration::from_secs(1);
        let guard = w.load_guard();

        assert_eq!(w.active_load(), 1);
        assert_eq!(w.slots_acquired_since(cutoff), 0);

        drop(guard);
        assert_eq!(w.active_load(), 0);
    }

    #[test]
    fn timestamped_load_guard_tracks_and_releases_its_slot() {
        let w = test_worker();
        let cutoff = Instant::now() - Duration::from_secs(1);
        let guard = w.timestamped_load_guard();

        assert_eq!(w.active_load(), 1);
        assert_eq!(w.slots_acquired_since(cutoff), 1);

        drop(guard);
        assert_eq!(w.active_load(), 0);
        assert_eq!(w.slots_acquired_since(cutoff), 0);
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

    #[test]
    fn slots_acquired_since_excludes_earlier_slots() {
        let w = test_worker();
        let _g_old = w.timestamped_load_guard();
        // A real (small) sleep, not a synthetic `Instant` offset: the slot's
        // acquisition time is captured internally by `claim()`, not
        // injectable, so the ordering guarantee has to come from wall-clock
        // separation wide enough to beat any platform's monotonic-clock
        // resolution.
        std::thread::sleep(Duration::from_millis(5));
        let cutoff = Instant::now();
        let _g_new1 = w.timestamped_load_guard();
        let _g_new2 = w.timestamped_load_guard();
        assert_eq!(w.active_load(), 3);
        assert_eq!(
            w.slots_acquired_since(cutoff),
            2,
            "only slots claimed at/after cutoff should count"
        );
    }

    #[test]
    fn slots_acquired_since_counts_all_slots_for_a_cutoff_before_every_claim() {
        let w = test_worker();
        let long_ago = Instant::now() - Duration::from_secs(3600);
        let _g1 = w.timestamped_load_guard();
        let _g2 = w.timestamped_load_guard();
        assert_eq!(w.slots_acquired_since(long_ago), 2);
    }

    #[test]
    fn slots_acquired_since_is_zero_for_a_cutoff_after_every_claim() {
        let w = test_worker();
        let _g = w.timestamped_load_guard();
        std::thread::sleep(Duration::from_millis(5));
        let cutoff = Instant::now();
        assert_eq!(w.slots_acquired_since(cutoff), 0);
    }
}
