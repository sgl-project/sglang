// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Policy for RoundRobinPolicy {
    fn select(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }
        let i = self.counter.fetch_add(1, Ordering::Relaxed) % workers.len();
        Some(workers[i].clone())
    }
}
