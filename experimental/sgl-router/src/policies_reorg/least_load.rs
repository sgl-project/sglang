// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::policies::admission::FreshLoadLookup;
use crate::workers::Worker;

use super::admission::Admission;
use super::{ready, Pick, PickError, PickRequest, Policy};

/// The least loaded candidate: reported queue plus dispatches since the report
/// when every candidate has a fresh sample, else router-local in-flight.
/// Ties rotate.
#[derive(Debug, Default)]
pub struct LeastLoadPolicy {
    pub admission: Admission,
    rotor: AtomicUsize,
}

impl Policy for LeastLoadPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        ready(
            self.admission
                .select(engines, request, "least_load", |admitted| {
                    let loads = FreshLoadLookup::new(Some(request.load.snapshot()), admitted);
                    let keys: Vec<_> = admitted
                        .iter()
                        .map(|engine| (loads.score_load(engine), engine.active_load()))
                        .collect();
                    let best = keys.iter().min()?;
                    let tied: Vec<_> = (0..keys.len()).filter(|&i| keys[i] == *best).collect();
                    let i = self.rotor.fetch_add(1, Ordering::Relaxed) % tied.len();
                    Some(admitted[tied[i]].clone())
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{pick, worker};
    use super::*;

    #[tokio::test]
    async fn least_local_load_wins_and_ties_rotate() {
        let policy = LeastLoadPolicy::default();
        let (a, b, c) = (worker("a"), worker("b"), worker("c"));
        a.active_requests.store(3, Ordering::Relaxed);
        assert_eq!(
            pick(&policy, &[a, b.clone()]).await.unwrap().engine.id.0,
            "b"
        );
        let fleet = [b, c];
        let first = pick(&policy, &fleet).await.unwrap().engine.id.clone();
        let second = pick(&policy, &fleet).await.unwrap().engine.id.clone();
        assert_ne!(first, second);
    }
}
