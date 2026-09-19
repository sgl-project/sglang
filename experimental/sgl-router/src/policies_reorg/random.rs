// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::future::BoxFuture;
use rand::seq::SliceRandom;

use crate::workers::Worker;

use super::admission::Admission;
use super::{ready, Pick, PickError, PickRequest, Policy};

/// A uniformly random admitted candidate.
#[derive(Debug, Default)]
pub struct RandomPolicy {
    pub admission: Admission,
}

impl Policy for RandomPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        ready(
            self.admission
                .select(engines, request, "random", |admitted| {
                    admitted.choose(&mut rand::thread_rng()).cloned()
                }),
        )
    }
}
