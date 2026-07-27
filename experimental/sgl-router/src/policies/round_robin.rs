// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, PolicyCandidate, SelectionContext};
use crate::workers::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    /// Constructs a round-robin policy with a zero selection counter.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Policy for RoundRobinPolicy {
    /// Selects the next candidate in cyclic order.
    fn select(
        &self,
        candidates: &[PolicyCandidate],
        _ctx: &SelectionContext<'_>,
    ) -> Option<Arc<Worker>> {
        if candidates.is_empty() {
            return None;
        }
        let i = self.counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
        Some(Arc::clone(&candidates[i].worker))
    }
}
