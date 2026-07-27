// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, PolicyCandidate, SelectionContext};
use crate::workers::Worker;
use rand::seq::SliceRandom;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    /// Constructs a stateless random policy.
    pub fn new() -> Self {
        Self
    }
}

impl Policy for RandomPolicy {
    /// Selects a uniformly random candidate.
    fn select(
        &self,
        candidates: &[PolicyCandidate],
        _ctx: &SelectionContext<'_>,
    ) -> Option<Arc<Worker>> {
        candidates
            .choose(&mut rand::thread_rng())
            .map(|candidate| Arc::clone(&candidate.worker))
    }
}
