// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::future::BoxFuture;
use rand::Rng;

use crate::workers::Worker;

use super::admission::Admission;
use super::{ready, Pick, PickError, PickRequest, Policy};

/// Two distinct random candidates; the one under lower stage load wins.
#[derive(Debug, Default)]
pub struct PowerOfTwoPolicy {
    pub admission: Admission,
}

impl Policy for PowerOfTwoPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        ready(
            self.admission
                .select(engines, request, "power_of_two", |admitted| {
                    choose(admitted, request).cloned()
                }),
        )
    }
}

/// One power-of-two choice; shared by policies that fall back to it.
pub(crate) fn choose<'e>(
    engines: &'e [Arc<Worker>],
    request: &PickRequest<'_>,
) -> Option<&'e Arc<Worker>> {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..engines.len().max(1));
    let left = engines.get(i)?;
    if engines.len() == 1 {
        return Some(left);
    }
    let j = (i + rng.gen_range(1..engines.len())) % engines.len();
    Some(
        request
            .load
            .lower_pressure(left, &engines[j], request.stage),
    )
}

#[cfg(test)]
mod tests {
    use super::super::testing::{pick, worker};
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn lower_load_wins_and_every_engine_is_reachable() {
        let policy = PowerOfTwoPolicy::default();
        let (a, b) = (worker("a"), worker("b"));
        a.active_requests.store(10, Ordering::Relaxed);
        assert_eq!(pick(&policy, &[a, b]).await.unwrap().engine.id.0, "b");
        let fleet: Vec<_> = ["a", "b", "c", "d", "e"].map(worker).into();
        let mut seen = HashSet::new();
        for _ in 0..500 {
            seen.insert(pick(&policy, &fleet).await.unwrap().engine.id.clone());
        }
        assert_eq!(seen.len(), fleet.len());
    }
}
