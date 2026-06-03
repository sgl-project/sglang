// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use rand::seq::IteratorRandom;
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
            _ => {
                let mut rng = rand::thread_rng();
                let mut chosen = workers.iter().choose_multiple(&mut rng, 2);
                chosen.sort_by_key(|w| w.active_load());
                Some(chosen[0].clone())
            }
        }
    }
}
