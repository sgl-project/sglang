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
            len => {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len - 1);
                if j >= i {
                    j += 1;
                }
                Some(std::cmp::min_by_key(&workers[i], &workers[j], |w| w.active_load()).clone())
            }
        }
    }
}
