// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use rand::Rng;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct PowerOfTwoChoicesPolicy;

impl PowerOfTwoChoicesPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Policy for PowerOfTwoChoicesPolicy {
    fn select(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        match workers.len() {
            0 => None,
            1 => Some(workers[0].clone()),
            n => {
                // Select two random workers - use offset to guarantee different selection in O(1)
                // (`IteratorRandom::choose_multiple` is reservoir sampling and walks every worker).
                let mut rng = rand::thread_rng();
                let idx1 = rng.gen_range(0..n);
                // Pick idx2 from remaining indices: offset by 1 + random from (len-1) to guarantee different
                let idx2 = (idx1 + 1 + rng.gen_range(0..n - 1)) % n;

                let worker1 = &workers[idx1];
                let worker2 = &workers[idx2];

                // Select worker with lower load
                if worker1.active_load() <= worker2.active_load() {
                    Some(worker1.clone())
                } else {
                    Some(worker2.clone())
                }
            }
        }
    }
}
