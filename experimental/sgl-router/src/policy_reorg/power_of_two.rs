// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Two distinct random candidates; the one under less pressure wins. Serves
//! prefill and decode, comparing by the request's stage.

use super::{ready, Admission, PickRequest, PickResult, Policy, RoutingStage};
use crate::policies::state::engine_load::{
    compare_decode_pressure, compare_prefill_pressure, EngineLoadSnapshot,
};
use crate::workers::Worker;
use futures::future::BoxFuture;
use rand::Rng;
use std::sync::Arc;

#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    admission: Admission,
}

impl PowerOfTwoPolicy {
    pub fn new(admission: Admission) -> Self {
        Self { admission }
    }
}

impl Policy for PowerOfTwoPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult> {
        let ctx = request.admission();
        ready(self.admission.select(engines, &ctx, |admitted| {
            let (left, right) = sample_pair(admitted, &mut rand::thread_rng())?;
            Some(lower_pressure(left, right, request.stage, request.load.snapshot()).clone())
        }))
    }
}

/// Two distinct candidates when there are at least two; one pairs with itself.
pub(crate) fn sample_pair<'e>(
    engines: &'e [Arc<Worker>],
    rng: &mut impl Rng,
) -> Option<(&'e Arc<Worker>, &'e Arc<Worker>)> {
    let len = engines.len();
    let i = rng.gen_range(0..len.max(1));
    let left = engines.get(i)?;
    if len == 1 {
        return Some((left, left));
    }
    let mut j = rng.gen_range(0..len - 1);
    if j >= i {
        j += 1;
    }
    Some((left, &engines[j]))
}

pub(crate) fn lower_pressure<'e>(
    left: &'e Arc<Worker>,
    right: &'e Arc<Worker>,
    stage: RoutingStage,
    snapshot: &EngineLoadSnapshot,
) -> &'e Arc<Worker> {
    let compare = match stage {
        RoutingStage::Decode => compare_decode_pressure,
        RoutingStage::Plain | RoutingStage::Prefill => compare_prefill_pressure,
    };
    if compare(left, right, Some(snapshot)).is_gt() {
        right
    } else {
        left
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{id, pick, worker};
    use super::super::PickError;
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::Ordering;

    fn policy() -> PowerOfTwoPolicy {
        PowerOfTwoPolicy::new(Admission::allow_all())
    }

    #[tokio::test]
    async fn selects_lower_load_and_reaches_every_worker() {
        let (a, b) = (worker("a"), worker("b"));
        a.active_requests.store(10, Ordering::Relaxed);
        b.active_requests.store(2, Ordering::Relaxed);
        assert_eq!(id(&pick(&policy(), &[a, b]).await), "b");

        let fleet: Vec<_> = ["a", "b", "c", "d", "e"].map(worker).into();
        let mut seen = HashSet::new();
        for _ in 0..1000 {
            seen.insert(pick(&policy(), &fleet).await.unwrap().engine.id.0.clone());
        }
        assert_eq!(seen.len(), fleet.len(), "{seen:?}");
    }

    #[tokio::test]
    async fn empty_and_single_candidate_edges() {
        assert_eq!(
            pick(&policy(), &[]).await.unwrap_err(),
            PickError::NoCandidates
        );
        assert_eq!(id(&pick(&policy(), &[worker("only")]).await), "only");
    }
}
