// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

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
        Self {
            id: spec.id,
            url: spec.url,
            mode: AtomicU8::new(spec.mode.as_u8()),
            model_ids: spec.model_ids,
            breaker,
            active_requests: Arc::new(AtomicUsize::new(0)),
        }
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
        });
        assert_eq!(w.mode(), WorkerMode::Prefill);
        w.set_mode(WorkerMode::Decode);
        assert_eq!(w.mode(), WorkerMode::Decode);
        w.set_mode(WorkerMode::Plain);
        assert_eq!(w.mode(), WorkerMode::Plain);
    }
}
