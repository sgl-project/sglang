// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{ready, Admission, PickRequest, PickResult, Policy};
use crate::workers::Worker;
use futures::future::BoxFuture;
use rand::seq::SliceRandom;
use std::sync::Arc;

/// A uniformly random admitted candidate.
#[derive(Debug)]
pub struct RandomPolicy {
    admission: Admission,
}

impl RandomPolicy {
    pub fn new(admission: Admission) -> Self {
        Self { admission }
    }
}

impl Policy for RandomPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult> {
        let ctx = request.admission();
        ready(self.admission.select(engines, &ctx, |admitted| {
            admitted.choose(&mut rand::thread_rng()).cloned()
        }))
    }
}
