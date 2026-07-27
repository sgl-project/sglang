// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::WorkerMode;
use crate::policies::{Policy, PolicyCandidate, SelectionContext};
use crate::workers::Worker;
use rand::Rng;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct PowerOfTwoChoicesPolicy;

impl PowerOfTwoChoicesPolicy {
    /// Constructs a stateless power-of-two policy.
    pub fn new() -> Self {
        Self
    }

    /// Returns the scoring load for a policy candidate.
    ///
    /// Prefill workers use total tokens; regular and decode workers use total
    /// requests. Missing load is not schedulable and returns `None`.
    fn score(candidate: &PolicyCandidate) -> Option<u64> {
        let load = candidate.load.as_ref()?;
        Some(if candidate.worker.mode() == WorkerMode::Prefill {
            load.num_total_tokens
        } else {
            load.total_requests
        })
    }
}

impl Policy for PowerOfTwoChoicesPolicy {
    /// Samples two distinct candidates and returns the lower reported score.
    fn select(
        &self,
        candidates: &[PolicyCandidate],
        _ctx: &SelectionContext<'_>,
    ) -> Option<Arc<Worker>> {
        match candidates.len() {
            0 => None,
            1 => Self::score(&candidates[0]).map(|_| Arc::clone(&candidates[0].worker)),
            len => {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len - 1);
                if j >= i {
                    j += 1;
                }
                let left = Self::score(&candidates[i])?;
                let right = Self::score(&candidates[j])?;
                let chosen = if left <= right {
                    &candidates[i]
                } else {
                    &candidates[j]
                };
                Some(Arc::clone(&chosen.worker))
            }
        }
    }
}
