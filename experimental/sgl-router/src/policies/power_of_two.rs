// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::admission::compare_prefill_pressure;
use crate::policies::{Policy, ProposalKind, SelectionContext, SelectionProposal};
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
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
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
                Some(select_lower_pressure(&workers[i], &workers[j], ctx))
            }
        }
    }

    /// Returns the primary and backup from one sample.
    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match workers.len() {
            0 => None,
            1 => Some(
                SelectionProposal::primary(workers[0].clone()).with_kind(ProposalKind::PowerOfTwo),
            ),
            len => {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len - 1);
                if j >= i {
                    j += 1;
                }
                let (primary, backup) = ordered_pair(&workers[i], &workers[j], ctx);
                Some(SelectionProposal::with_backup(primary, backup))
            }
        }
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }
}

fn select_lower_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    ctx: &SelectionContext<'_>,
) -> Arc<Worker> {
    ordered_pair(left, right, ctx).0
}

fn ordered_pair(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    ctx: &SelectionContext<'_>,
) -> (Arc<Worker>, Arc<Worker>) {
    if compare_prefill_pressure(left, right, ctx.load_snapshot()).is_gt() {
        (Arc::clone(right), Arc::clone(left))
    } else {
        (Arc::clone(left), Arc::clone(right))
    }
}
