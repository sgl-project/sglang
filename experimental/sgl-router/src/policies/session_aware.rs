// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Reuses the engine a session is bound to; a new session binds to a
//! power-of-two choice. Admission never rewrites an existing binding.

use super::{
    power_of_two, ready, Admission, AdmissionReason, EngineRejection, Pick, PickError, PickMode,
    PickRequest, PickResult, Policy,
};
use crate::policies::state::AffinityStore;
use crate::workers::Worker;
use futures::future::BoxFuture;
use std::sync::Arc;

#[derive(Debug)]
pub struct SessionAwarePolicy {
    admission: Admission,
    store: Arc<AffinityStore>,
}

impl SessionAwarePolicy {
    pub fn new(admission: Admission, store: Arc<AffinityStore>) -> Self {
        Self { admission, store }
    }

    fn pick_sync(&self, engines: &[Arc<Worker>], request: &PickRequest<'_>) -> PickResult {
        let ctx = request.admission();
        let key = request.affinity_key("session", request.session_id);
        let Some(key) = key else {
            let reason = if request.affinity_enabled {
                "no_session"
            } else {
                "range_fallback"
            };
            return self.fallback(engines, request, reason);
        };
        if let Some(bound) = self.store.bound(&key, engines) {
            return match self.admission.check(bound, &ctx) {
                Ok(()) => Ok(Pick {
                    engine: Arc::clone(bound),
                    reason: "session_primary",
                }),
                Err(reason) if request.mode == PickMode::HitRequired => {
                    Err(PickError::rejected_all(bound, reason))
                }
                Err(_) => self.fallback(engines, request, "session_admission_fallback"),
            };
        }
        if request.mode == PickMode::HitRequired {
            // A binding whose engine is not here is reported so the resolver can preserve it.
            return Err(match self.store.binding(&key) {
                Some(engine) => PickError::AdmissionRejected(EngineRejection {
                    engine,
                    reason: AdmissionReason::NotACandidate,
                }),
                None => PickError::NoCandidates,
            });
        }
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        let admitted = self.admission.admit(engines, &ctx)?;
        let chosen = power_of_two::choose(&admitted, request).ok_or(PickError::NoCandidates)?;
        let engine = Arc::clone(self.store.bind(key, chosen, &admitted));
        Ok(Pick {
            engine,
            reason: "assigned",
        })
    }

    fn fallback(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
        reason: &'static str,
    ) -> PickResult {
        if request.mode == PickMode::HitRequired {
            return Err(PickError::NoCandidates);
        }
        let ctx = request.admission();
        let mut pick = self.admission.select(engines, &ctx, |admitted| {
            power_of_two::choose(admitted, request).cloned()
        })?;
        pick.reason = reason;
        Ok(pick)
    }
}

impl Policy for SessionAwarePolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult> {
        ready(self.pick_sync(engines, request))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{id, worker, Request};
    use super::super::InFlightLimitAdmission;
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    fn policy(admission: Admission) -> SessionAwarePolicy {
        SessionAwarePolicy::new(admission, AffinityStore::new(Duration::from_secs(60)))
    }

    #[tokio::test]
    async fn a_session_binds_once_and_keyless_requests_never_bind() {
        let policy = policy(Admission::allow_all());
        let fleet: Vec<_> = ["a", "b", "c"].map(worker).into();
        let keyless = Request::default().pick(&policy, &fleet).await.unwrap();
        assert!(policy.store.is_empty() && keyless.reason == "no_session");

        let first = Request::session("s1").pick(&policy, &fleet).await.unwrap();
        assert_eq!(first.reason, "assigned");
        for _ in 0..5 {
            let again = Request::session("s1").pick(&policy, &fleet).await.unwrap();
            assert!(again.engine.id == first.engine.id && again.reason == "session_primary");
        }
    }

    #[tokio::test]
    async fn hit_required_never_binds_and_a_rejected_binding_is_kept() {
        let policy = policy(Admission::before(InFlightLimitAdmission {
            max_in_flight: 1,
        }));
        let fleet: Vec<_> = ["a", "b"].map(worker).into();
        let miss = Request::session("s")
            .hit_required()
            .pick(&policy, &fleet)
            .await;
        assert!(miss.unwrap_err() == PickError::NoCandidates && policy.store.is_empty());

        let bound = Request::session("s")
            .pick(&policy, &fleet)
            .await
            .unwrap()
            .engine;
        bound.active_requests.store(1, Ordering::Relaxed);
        let fallback = Request::session("s").pick(&policy, &fleet).await.unwrap();
        assert!(fallback.engine.id != bound.id && fallback.reason == "session_admission_fallback");
        assert!(matches!(
            Request::session("s")
                .hit_required()
                .pick(&policy, &fleet)
                .await,
            Err(PickError::NoAdmissibleEngine(_))
        ));
        bound.active_requests.store(0, Ordering::Relaxed);
        assert_eq!(
            id(&Request::session("s").pick(&policy, &fleet).await),
            bound.id.0
        );
    }
}
