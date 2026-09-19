// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::discovery::WorkerId;

use crate::workers::Worker;

use super::{Pick, PickError, PickRequest, Policy, Rejection};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

impl Decision {
    fn from(allowed: bool, reason: &str) -> Self {
        if allowed {
            Self::Allow
        } else {
            Self::Reject(reason.into())
        }
    }
}

pub trait EngineAdmission: Send + Sync + Debug {
    fn check(&self, engine: &Worker, request: &PickRequest<'_>) -> Result<Decision, PickError>;
}

#[derive(Debug)]
pub struct AllowAll;

impl EngineAdmission for AllowAll {
    fn check(&self, _: &Worker, _: &PickRequest<'_>) -> Result<Decision, PickError> {
        Ok(Decision::Allow)
    }
}

/// Engine-reported running and KV capacity; admits without a fresh native sample.
#[derive(Debug)]
pub struct Capacity;

impl EngineAdmission for Capacity {
    fn check(&self, engine: &Worker, request: &PickRequest<'_>) -> Result<Decision, PickError> {
        let fits = request
            .load
            .snapshot()
            .fresh_native_cache_load_for_url(&engine.url)
            .is_none_or(|load| {
                load.num_running_reqs < load.max_running_requests
                    && load.num_total_tokens.saturating_add(request.kv_tokens())
                        <= load.max_total_num_tokens
            });
        Ok(Decision::from(fits, "kv_capacity"))
    }
}

/// Waiting uncached tokens plus this request's input must fit the engine's
/// bucket budget; engines without a budget and engines without a fresh
/// native sample are admitted.
#[derive(Debug)]
pub struct PendingPrefill(pub HashMap<WorkerId, u64>);

impl EngineAdmission for PendingPrefill {
    fn check(&self, engine: &Worker, request: &PickRequest<'_>) -> Result<Decision, PickError> {
        let Some(budget) = self.0.get(&engine.id) else {
            return Ok(Decision::Allow);
        };
        let fits = request
            .load
            .snapshot()
            .fresh_native_cache_load_for_url(&engine.url)
            .is_none_or(|load| {
                load.num_waiting_uncached_tokens
                    .saturating_add(request.input_tokens)
                    <= *budget
            });
        Ok(Decision::from(fits, "pending_prefill_budget"))
    }
}

/// Router-local in-flight requests must stay below the limit.
#[derive(Debug)]
pub struct InFlightLimit(pub usize);

impl EngineAdmission for InFlightLimit {
    fn check(&self, engine: &Worker, _: &PickRequest<'_>) -> Result<Decision, PickError> {
        Ok(Decision::from(
            engine.active_load() < self.0,
            "in_flight_limit",
        ))
    }
}

/// Engine-reported waiting requests must stay below the limit; admits without a sample.
#[derive(Debug)]
pub struct QueueLimit(pub u64);

impl EngineAdmission for QueueLimit {
    fn check(&self, engine: &Worker, request: &PickRequest<'_>) -> Result<Decision, PickError> {
        let below = request
            .load
            .snapshot()
            .fresh_load_for_url(&engine.url)
            .is_none_or(|load| load.num_waiting_reqs < self.0);
        Ok(Decision::from(below, "queue_limit"))
    }
}

/// Every check must admit; the first rejection is the reason.
#[derive(Debug)]
pub struct AllOf(pub Vec<Arc<dyn EngineAdmission>>);

impl EngineAdmission for AllOf {
    fn check(&self, engine: &Worker, request: &PickRequest<'_>) -> Result<Decision, PickError> {
        for check in &self.0 {
            if let rejected @ Decision::Reject(_) = check.check(engine, request)? {
                return Ok(rejected);
            }
        }
        Ok(Decision::Allow)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Placement {
    #[default]
    BeforeSelection,
    AfterSelection,
}

#[derive(Debug, Clone)]
pub struct Admission {
    pub check: Arc<dyn EngineAdmission>,
    pub placement: Placement,
}

impl Default for Admission {
    fn default() -> Self {
        Self::before(AllowAll)
    }
}

impl Admission {
    pub fn before(check: impl EngineAdmission + 'static) -> Self {
        Self {
            check: Arc::new(check),
            placement: Placement::BeforeSelection,
        }
    }

    pub fn after(check: impl EngineAdmission + 'static) -> Self {
        Self {
            check: Arc::new(check),
            placement: Placement::AfterSelection,
        }
    }

    /// Runs `choose` at the configured placement and returns an admitted pick.
    pub fn select(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
        reason: &'static str,
        choose: impl FnOnce(&[Arc<Worker>]) -> Option<Arc<Worker>>,
    ) -> Result<Pick, PickError> {
        let admitted = self.admit(engines, request)?;
        let engine = choose(&admitted).ok_or(PickError::NoCandidates)?;
        self.verify(Pick { engine, reason }, request)
    }

    /// Lets `fallback` choose among the admitted candidates at this placement.
    pub async fn delegate(
        &self,
        fallback: &dyn Policy,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Result<Pick, PickError> {
        let admitted = self.admit(engines, request)?;
        let pick = fallback.pick(&admitted, request).await?;
        self.verify(pick, request)
    }

    /// Candidates a policy may choose from; a no-op under `AfterSelection`.
    pub fn admit(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Result<Vec<Arc<Worker>>, PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        if self.placement == Placement::AfterSelection {
            return Ok(engines.to_vec());
        }
        let mut admitted = Vec::new();
        let mut rejected = Vec::new();
        for engine in engines {
            match self.check.check(engine, request)? {
                Decision::Allow => admitted.push(engine.clone()),
                Decision::Reject(reason) => rejected.push(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }),
            }
        }
        if admitted.is_empty() {
            Err(PickError::NoAdmissibleEngine(rejected))
        } else {
            Ok(admitted)
        }
    }

    /// Checks the chosen engine under `AfterSelection`; never picks a replacement.
    pub fn verify(&self, pick: Pick, request: &PickRequest<'_>) -> Result<Pick, PickError> {
        if self.placement == Placement::AfterSelection {
            if let Decision::Reject(reason) = self.check.check(&pick.engine, request)? {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: pick.engine.id.clone(),
                    reason,
                }));
            }
        }
        Ok(pick)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{pick_with, worker};
    use super::*;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn checks_compose_and_report_the_first_rejection() {
        let (busy, idle) = (worker("busy"), worker("idle"));
        busy.active_requests.store(2, Ordering::Relaxed);
        let admission = Admission::before(AllOf(vec![
            Arc::new(Capacity),
            Arc::new(InFlightLimit(2)),
            Arc::new(QueueLimit(1)),
        ]));
        let pick = pick_with(&admission, &[busy.clone(), idle.clone()]).await;
        assert_eq!(pick.unwrap().engine.id.0, "idle");
        idle.active_requests.store(2, Ordering::Relaxed);
        let Err(PickError::NoAdmissibleEngine(rejections)) =
            pick_with(&admission, &[busy, idle]).await
        else {
            panic!("expected exhaustion")
        };
        assert!(rejections.iter().all(|r| r.reason == "in_flight_limit"));
    }
}
