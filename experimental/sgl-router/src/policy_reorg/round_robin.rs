// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{ready, Admission, PickRequest, PickResult, Policy};
use crate::workers::Worker;
use futures::future::BoxFuture;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Rotates over the admitted candidates; the cursor belongs to this instance.
#[derive(Debug)]
pub struct RoundRobinPolicy {
    admission: Admission,
    cursor: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new(admission: Admission) -> Self {
        Self {
            admission,
            cursor: AtomicUsize::new(0),
        }
    }
}

impl Policy for RoundRobinPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult> {
        let ctx = request.admission();
        ready(self.admission.select(engines, &ctx, |admitted| {
            let i = self.cursor.fetch_add(1, Ordering::Relaxed) % admitted.len();
            Some(Arc::clone(&admitted[i]))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{pick, worker};
    use super::*;

    #[tokio::test]
    async fn cycles_through_workers() {
        let policy = RoundRobinPolicy::new(Admission::allow_all());
        let fleet: Vec<_> = ["a", "b", "c"].map(worker).into();
        let mut picks = Vec::new();
        for _ in 0..6 {
            picks.push(pick(&policy, &fleet).await.unwrap().engine.id.0.clone());
        }
        assert_eq!(picks, ["a", "b", "c", "a", "b", "c"]);
    }
}
