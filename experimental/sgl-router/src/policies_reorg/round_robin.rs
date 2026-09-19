// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::workers::Worker;

use super::admission::Admission;
use super::{ready, Pick, PickError, PickRequest, Policy};

/// Rotates over the admitted candidates; the cursor belongs to this instance.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    pub admission: Admission,
    cursor: AtomicUsize,
}

impl Policy for RoundRobinPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        ready(
            self.admission
                .select(engines, request, "round_robin", |admitted| {
                    let i = self.cursor.fetch_add(1, Ordering::Relaxed) % admitted.len();
                    Some(admitted[i].clone())
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{pick, worker};
    use super::*;

    #[tokio::test]
    async fn cycles_through_engines() {
        let policy = RoundRobinPolicy::default();
        let fleet: Vec<_> = ["a", "b", "c"].map(worker).into();
        let mut picks = Vec::new();
        for _ in 0..4 {
            picks.push(pick(&policy, &fleet).await.unwrap().engine.id.0.clone());
        }
        assert_eq!(picks, ["a", "b", "c", "a"]);
    }
}
