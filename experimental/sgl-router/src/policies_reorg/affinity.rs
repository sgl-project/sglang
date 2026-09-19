// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::server::metrics::{MetricsRegistry, StickyOutcome};
use crate::state::AffinityStore;
use crate::workers::Worker;

use super::admission::{Admission, Decision};
use super::{Pick, PickError, PickMode, PickRequest, Policy, Rejection};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityKind {
    Session,
    Sticky,
}

impl AffinityKind {
    fn name(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Sticky => "sticky",
        }
    }

    fn hit(self) -> &'static str {
        match self {
            Self::Session => "session_primary",
            Self::Sticky => "sticky_hit",
        }
    }
}

/// Reuses an admitted binding for the request's key; a new or unusable key
/// binds the fallback's choice. Sticky and session differ only in the key.
#[derive(Debug)]
pub struct AffinityPolicy {
    pub kind: AffinityKind,
    pub admission: Admission,
    pub store: Arc<AffinityStore>,
    /// Keys span buckets instead of being scoped to the current one.
    pub global: bool,
    pub fallback: Arc<dyn Policy>,
    pub metrics: Arc<MetricsRegistry>,
}

impl AffinityPolicy {
    fn record(&self, outcome: StickyOutcome) {
        if self.kind == AffinityKind::Sticky {
            self.metrics.record_sticky(outcome);
        }
    }

    async fn pick_async(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Result<Pick, PickError> {
        let hit_required = request.mode == PickMode::HitRequired;
        let value = match self.kind {
            AffinityKind::Session => request.session_key,
            AffinityKind::Sticky => request.routing_key,
        };
        if value.is_none() && !hit_required {
            self.record(StickyOutcome::NoRoutingKey);
        }
        let key = value
            .filter(|_| request.affinity_enabled)
            .map(|value| request.affinity_key(self.kind.name(), self.global, value));
        let Some(key) = key else {
            return match hit_required {
                true => Err(PickError::NoCandidates),
                false => self.delegate(engines, request).await,
            };
        };
        if let Some(bound) = self.store.bound(&key, engines) {
            return match self.admission.check.check(bound, request)? {
                Decision::Allow => {
                    self.record(StickyOutcome::Hit);
                    Ok(Pick {
                        engine: bound.clone(),
                        reason: self.kind.hit(),
                    })
                }
                Decision::Reject(reason) if hit_required => {
                    Err(PickError::AdmissionRejected(Rejection {
                        engine: bound.id.clone(),
                        reason,
                    }))
                }
                Decision::Reject(_) => self.delegate(engines, request).await,
            };
        }
        if hit_required {
            // A binding outside the candidates is reported so the resolver can preserve it.
            return Err(match self.store.binding(&key) {
                Some(engine) => PickError::AdmissionRejected(Rejection {
                    engine,
                    reason: "not_a_candidate".into(),
                }),
                None => PickError::NoCandidates,
            });
        }
        let admitted = self.admission.admit(engines, request)?;
        let pick = self.fallback.pick(&admitted, request).await?;
        let remap = self.store.contains(&key);
        let engine = self.store.bind(key, &pick.engine, &admitted).clone();
        let pick = self.admission.verify(
            Pick {
                engine,
                reason: if remap { "remap" } else { "assigned" },
            },
            request,
        )?;
        self.record(if remap {
            StickyOutcome::Remap
        } else {
            StickyOutcome::Assigned
        });
        Ok(pick)
    }

    async fn delegate(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Result<Pick, PickError> {
        self.admission
            .delegate(self.fallback.as_ref(), engines, request)
            .await
    }
}

impl Policy for AffinityPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(self.pick_async(engines, request))
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        Some(self.fallback.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::super::admission::InFlightLimit;
    use super::super::round_robin::RoundRobinPolicy;
    use super::super::testing::{pick, pick_as, worker};
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    fn policy(admission: Admission) -> AffinityPolicy {
        AffinityPolicy {
            kind: AffinityKind::Session,
            admission,
            store: AffinityStore::new(Duration::from_secs(60)),
            global: false,
            fallback: Arc::new(RoundRobinPolicy::default()),
            metrics: MetricsRegistry::new(),
        }
    }

    #[tokio::test]
    async fn a_key_binds_once_and_keyless_requests_never_bind() {
        let policy = policy(Admission::default());
        let fleet: Vec<_> = ["a", "b"].map(worker).into();
        assert_eq!(pick(&policy, &fleet).await.unwrap().reason, "round_robin");
        assert!(policy.store.is_empty());
        let first = pick_as(&policy, &fleet, Some("s"), PickMode::Normal)
            .await
            .unwrap();
        assert_eq!(first.reason, "assigned");
        let again = pick_as(&policy, &fleet, Some("s"), PickMode::Normal)
            .await
            .unwrap();
        assert!(again.engine.id == first.engine.id && again.reason == "session_primary");
    }

    #[tokio::test]
    async fn hit_required_never_binds_and_a_rejected_binding_falls_back() {
        let policy = policy(Admission::before(InFlightLimit(1)));
        let fleet: Vec<_> = ["a", "b"].map(worker).into();
        let miss = pick_as(&policy, &fleet, Some("s"), PickMode::HitRequired).await;
        assert!(matches!(miss, Err(PickError::NoCandidates)) && policy.store.is_empty());
        let bound = pick_as(&policy, &fleet, Some("s"), PickMode::Normal)
            .await
            .unwrap()
            .engine;
        bound.active_requests.store(1, Ordering::Relaxed);
        let fallback = pick_as(&policy, &fleet, Some("s"), PickMode::Normal)
            .await
            .unwrap();
        assert!(fallback.engine.id != bound.id && fallback.reason == "round_robin");
        assert!(matches!(
            pick_as(&policy, &fleet, Some("s"), PickMode::HitRequired).await,
            Err(PickError::AdmissionRejected(_))
        ));
        assert_eq!(policy.store.len(), 1);
    }
}
