// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The least-loaded candidate: engine-reported queue depth plus dispatches
//! since the report when every candidate has a fresh sample, else router-local
//! in-flight. Ties go to the least busy locally, then rotate.

use super::{ready, Admission, PickRequest, PickResult, Policy};
use crate::policies::state::engine_load::FreshLoadLookup;
use crate::workers::Worker;
use futures::future::BoxFuture;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub struct LeastLoadPolicy {
    admission: Admission,
    rotor: AtomicUsize,
}

impl LeastLoadPolicy {
    pub fn new(admission: Admission) -> Self {
        Self {
            admission,
            rotor: AtomicUsize::new(0),
        }
    }
}

impl Policy for LeastLoadPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult> {
        let ctx = request.admission();
        ready(self.admission.select(engines, &ctx, |admitted| {
            let loads = FreshLoadLookup::new(Some(request.load.snapshot()), admitted.iter());
            let keys: Vec<(usize, usize)> = admitted
                .iter()
                .map(|engine| (loads.score_load(engine), engine.active_load()))
                .collect();
            let best = keys.iter().min()?;
            let tied: Vec<usize> = (0..keys.len()).filter(|&i| keys[i] == *best).collect();
            let k = self.rotor.fetch_add(1, Ordering::Relaxed) % tied.len();
            Some(Arc::clone(&admitted[tied[k]]))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{id, pick, worker};
    use super::*;

    #[tokio::test]
    async fn least_local_load_wins_and_ties_rotate() {
        let policy = LeastLoadPolicy::new(Admission::allow_all());
        let (a, b, c) = (worker("a"), worker("b"), worker("c"));
        a.active_requests.store(3, Ordering::Relaxed);
        assert_eq!(id(&pick(&policy, &[a.clone(), b.clone()]).await), "b");

        let fleet = [b, c];
        let first = pick(&policy, &fleet).await.unwrap().engine.id.0.clone();
        let second = pick(&policy, &fleet).await.unwrap().engine.id.0.clone();
        assert_ne!(first, second);
    }
}
