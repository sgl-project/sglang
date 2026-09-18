// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Pins a routing key to one engine; keyless requests and first touches go
//! through the configured fallback policy.

use super::{Admission, Pick, PickError, PickMode, PickRequest, PickResult, Policy};
use crate::policies::state::AffinityStore;
use crate::server::metrics::{MetricsRegistry, StickyOutcome};
use crate::workers::Worker;
use futures::future::BoxFuture;
use std::sync::Arc;

#[derive(Debug)]
pub struct StickyPolicy {
    admission: Admission,
    store: Arc<AffinityStore>,
    fallback: Arc<dyn Policy>,
    metrics: Arc<MetricsRegistry>,
}

impl StickyPolicy {
    pub fn new(
        admission: Admission,
        store: Arc<AffinityStore>,
        fallback: Arc<dyn Policy>,
        metrics: Arc<MetricsRegistry>,
    ) -> Self {
        Self {
            admission,
            store,
            fallback,
            metrics,
        }
    }

    async fn delegate(&self, engines: &[Arc<Worker>], request: &PickRequest<'_>) -> PickResult {
        if request.mode == PickMode::HitRequired {
            return Err(PickError::NoCandidates);
        }
        let admitted = self.admission.admit(engines, &request.admission())?;
        self.fallback.pick(&admitted, request).await
    }
}

impl Policy for StickyPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult> {
        Box::pin(async move {
            let Some(key) = request.affinity_key("sticky", request.routing_key) else {
                self.metrics.record_sticky(StickyOutcome::NoRoutingKey);
                return self.delegate(engines, request).await;
            };
            if let Some(bound) = self.store.bound(&key, engines) {
                match self.admission.check(bound, &request.admission()) {
                    Ok(()) => {
                        self.metrics.record_sticky(StickyOutcome::Hit);
                        return Ok(Pick {
                            engine: Arc::clone(bound),
                            reason: "sticky_hit",
                        });
                    }
                    Err(reason) if request.mode == PickMode::HitRequired => {
                        return Err(PickError::rejected_all(bound, reason));
                    }
                    Err(_) => return self.delegate(engines, request).await,
                }
            }
            if request.mode == PickMode::HitRequired {
                return Err(PickError::NoCandidates);
            }
            let remap = self.store.contains(&key);
            let admitted = self.admission.admit(engines, &request.admission())?;
            let mut pick = self.fallback.pick(&admitted, request).await?;
            pick.engine = Arc::clone(self.store.bind(key, &pick.engine, &admitted));
            self.metrics.record_sticky(if remap {
                StickyOutcome::Remap
            } else {
                StickyOutcome::Assigned
            });
            Ok(pick)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::round_robin::RoundRobinPolicy;
    use super::super::testing::{worker, Request};
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn a_key_sticks_until_its_engine_leaves_and_racing_first_touches_converge() {
        let policy = Arc::new(StickyPolicy::new(
            Admission::allow_all(),
            AffinityStore::new(Duration::from_secs(60)),
            Arc::new(RoundRobinPolicy::new(Admission::allow_all())),
            MetricsRegistry::new(),
        ));
        let fleet: Arc<Vec<_>> = Arc::new(["a", "b", "c"].map(worker).into());
        let picks = (0..16).map(|_| {
            let (policy, fleet) = (Arc::clone(&policy), Arc::clone(&fleet));
            tokio::spawn(async move { Request::routing_key("k").pick(&*policy, &fleet).await })
        });
        let mut pinned = None;
        for pick in picks {
            let engine = pick.await.unwrap().unwrap().engine;
            assert_eq!(pinned.get_or_insert(engine.id.clone()), &engine.id);
        }
        let pinned = pinned.unwrap();
        let rest: Vec<_> = fleet.iter().filter(|e| e.id != pinned).cloned().collect();
        let remapped = Request::routing_key("k")
            .pick(&*policy, &rest)
            .await
            .unwrap();
        assert!(remapped.engine.id != pinned && policy.store.len() == 1);
    }
}
