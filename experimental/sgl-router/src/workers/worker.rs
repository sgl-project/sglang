// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

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

/// RAII guard that increments `active_requests` on construction and
/// decrements on drop.  Obtain via [`Worker::load_guard`].
///
/// `#[must_use]`: a statement-form call like `worker.load_guard();` would
/// drop the guard on the same line, so the counter would never see the
/// in-flight request.  The compile-time warning catches that misuse.
#[must_use = "LoadGuard must be held for the request's lifetime; dropping it immediately decrements active_requests"]
pub struct LoadGuard {
    counter: Arc<AtomicUsize>,
}

impl LoadGuard {
    pub(crate) fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
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
        Self {
            id: spec.id,
            url: spec.url,
            mode: AtomicU8::new(spec.mode.as_u8()),
            model_ids: spec.model_ids,
            breaker,
            active_requests: Arc::new(AtomicUsize::new(0)),
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
        LoadGuard::new(self.active_requests.clone())
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
}
