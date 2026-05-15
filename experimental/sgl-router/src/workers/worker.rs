// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::health::circuit_breaker::CircuitBreaker;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

pub struct Worker {
    pub id: WorkerId,
    pub url: String,
    pub mode: WorkerMode,
    pub model_ids: Vec<ModelId>,
    pub breaker: Arc<CircuitBreaker>,
    pub active_requests: AtomicUsize,
}

impl Worker {
    pub fn new(spec: crate::discovery::WorkerSpec) -> Self {
        Self {
            id: spec.id,
            url: spec.url,
            mode: spec.mode,
            model_ids: spec.model_ids,
            breaker: Arc::new(CircuitBreaker::new()),
            active_requests: AtomicUsize::new(0),
        }
    }

    pub fn active_load(&self) -> usize {
        self.active_requests.load(std::sync::atomic::Ordering::Relaxed)
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
