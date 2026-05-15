// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::health::circuit_breaker::CircuitBreaker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// RAII guard that increments `active_requests` on construction and
/// decrements on drop.  Obtain via [`Worker::load_guard`].
pub struct LoadGuard {
    counter: Arc<AtomicUsize>,
}

impl LoadGuard {
    pub fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct Worker {
    pub id: WorkerId,
    pub url: String,
    pub mode: WorkerMode,
    pub model_ids: Vec<ModelId>,
    pub breaker: Arc<CircuitBreaker>,
    pub active_requests: Arc<AtomicUsize>,
}

impl Worker {
    pub fn new(spec: crate::discovery::WorkerSpec) -> Self {
        Self {
            id: spec.id,
            url: spec.url,
            mode: spec.mode,
            model_ids: spec.model_ids,
            breaker: Arc::new(CircuitBreaker::new()),
            active_requests: Arc::new(AtomicUsize::new(0)),
        }
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
            .field("mode", &self.mode)
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
}
