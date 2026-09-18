// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Whether an engine may accept the request. Admission only checks; ranking,
//! replacement and fallback belong to the policy that attaches it.

use super::{EngineRejection, Pick, PickError};
use crate::config::{BucketConfig, BucketStage};
use crate::policies::state::engine_load::{has_kv_capacity, LoadView};
use crate::workers::Worker;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionReason {
    KvCapacity,
    PendingPrefillBudget,
    InFlightLimit,
    QueueLimit,
}

impl AdmissionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KvCapacity => "kv_capacity",
            Self::PendingPrefillBudget => "pending_prefill_budget",
            Self::InFlightLimit => "in_flight_limit",
            Self::QueueLimit => "queue_limit",
        }
    }
}

/// `Ok(())` admits; `Err` carries the reason the engine was rejected.
pub type AdmissionDecision = Result<(), AdmissionReason>;

pub struct AdmissionContext<'a> {
    pub load: &'a LoadView<'a>,
    /// KV the request will hold: the input for prefill, the projected peak for decode.
    pub kv_tokens: u64,
    /// Prefill work the engine has not cached; the full input when unknown.
    pub uncached_tokens: u64,
}

pub trait EngineAdmission: Send + Sync + fmt::Debug {
    fn check(&self, engine: &Worker, ctx: &AdmissionContext<'_>) -> AdmissionDecision;
}

#[derive(Debug)]
pub struct AllowAll;

impl EngineAdmission for AllowAll {
    fn check(&self, _engine: &Worker, _ctx: &AdmissionContext<'_>) -> AdmissionDecision {
        Ok(())
    }
}

/// Engine-reported running and KV capacity; admits without a fresh native sample.
#[derive(Debug)]
pub struct CapacityAdmission;

impl EngineAdmission for CapacityAdmission {
    fn check(&self, engine: &Worker, ctx: &AdmissionContext<'_>) -> AdmissionDecision {
        let load = ctx
            .load
            .snapshot()
            .fresh_native_cache_load_for_url(&engine.url);
        has_kv_capacity(load, ctx.kv_tokens)
            .then_some(())
            .ok_or(AdmissionReason::KvCapacity)
    }
}

/// Waiting uncached tokens plus this request's uncached work must fit the
/// engine's bucket budget; admits without a fresh native sample.
#[derive(Debug)]
pub struct PendingPrefillAdmission {
    budget_by_engine: HashMap<String, u64>,
}

impl PendingPrefillAdmission {
    pub fn new(budget_by_engine: HashMap<String, u64>) -> Self {
        Self { budget_by_engine }
    }

    /// `None` when no prefill bucket configures a budget.
    pub fn from_buckets(config: &BucketConfig) -> Option<Self> {
        let budget_by_engine: HashMap<String, u64> = config
            .buckets
            .iter()
            .filter(|spec| spec.stage == BucketStage::Prefill)
            .filter_map(|spec| Some((spec, spec.max_pending_prefill_tokens?)))
            .flat_map(|(spec, limit)| spec.worker_ids.iter().map(move |id| (id.clone(), limit)))
            .collect();
        (!budget_by_engine.is_empty()).then(|| Self::new(budget_by_engine))
    }
}

impl EngineAdmission for PendingPrefillAdmission {
    fn check(&self, engine: &Worker, ctx: &AdmissionContext<'_>) -> AdmissionDecision {
        let Some(limit) = self.budget_by_engine.get(&engine.id.0) else {
            return Ok(());
        };
        let Some(load) = ctx
            .load
            .snapshot()
            .fresh_native_cache_load_for_url(&engine.url)
        else {
            return Ok(());
        };
        (load
            .num_waiting_uncached_tokens
            .saturating_add(ctx.uncached_tokens)
            <= *limit)
            .then_some(())
            .ok_or(AdmissionReason::PendingPrefillBudget)
    }
}

/// Router-local in-flight requests must stay below the limit.
#[derive(Debug)]
pub struct InFlightLimitAdmission {
    pub max_in_flight: usize,
}

impl EngineAdmission for InFlightLimitAdmission {
    fn check(&self, engine: &Worker, _ctx: &AdmissionContext<'_>) -> AdmissionDecision {
        (engine.active_load() < self.max_in_flight)
            .then_some(())
            .ok_or(AdmissionReason::InFlightLimit)
    }
}

/// Engine-reported waiting requests must stay below the limit; admits without a sample.
#[derive(Debug)]
pub struct QueueLimitAdmission {
    pub limit: u64,
}

impl EngineAdmission for QueueLimitAdmission {
    fn check(&self, engine: &Worker, ctx: &AdmissionContext<'_>) -> AdmissionDecision {
        ctx.load
            .snapshot()
            .fresh_load_for_url(&engine.url)
            .is_none_or(|load| load.num_waiting_reqs < self.limit)
            .then_some(())
            .ok_or(AdmissionReason::QueueLimit)
    }
}

/// Every check must admit; the first rejection is the reason.
#[derive(Debug)]
pub struct AllOfAdmission(pub Vec<Box<dyn EngineAdmission>>);

impl EngineAdmission for AllOfAdmission {
    fn check(&self, engine: &Worker, ctx: &AdmissionContext<'_>) -> AdmissionDecision {
        self.0.iter().try_for_each(|check| check.check(engine, ctx))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionPlacement {
    BeforeSelection,
    AfterSelection,
}

/// A policy's attached admission and where it runs relative to selection.
#[derive(Debug)]
pub struct Admission {
    check: Box<dyn EngineAdmission>,
    placement: AdmissionPlacement,
}

impl Admission {
    pub fn allow_all() -> Self {
        Self::before(AllowAll)
    }

    pub fn before(check: impl EngineAdmission + 'static) -> Self {
        Self {
            check: Box::new(check),
            placement: AdmissionPlacement::BeforeSelection,
        }
    }

    pub fn after(check: impl EngineAdmission + 'static) -> Self {
        Self {
            check: Box::new(check),
            placement: AdmissionPlacement::AfterSelection,
        }
    }

    pub fn check(&self, engine: &Worker, ctx: &AdmissionContext<'_>) -> AdmissionDecision {
        self.check.check(engine, ctx)
    }

    /// Runs `select` at the configured placement: over the admitted engines
    /// before selection, or checking its choice afterwards.
    pub fn select(
        &self,
        engines: &[Arc<Worker>],
        ctx: &AdmissionContext<'_>,
        select: impl FnOnce(&[Arc<Worker>]) -> Option<Arc<Worker>>,
    ) -> Result<Pick, PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        let engine = match self.placement {
            AdmissionPlacement::BeforeSelection => {
                select(&self.admit(engines, ctx)?).ok_or(PickError::NoCandidates)?
            }
            AdmissionPlacement::AfterSelection => {
                let selected = select(engines).ok_or(PickError::NoCandidates)?;
                self.check(&selected, ctx)
                    .map_err(|reason| PickError::AdmissionRejected(rejection(&selected, reason)))?;
                selected
            }
        };
        Ok(Pick {
            engine,
            reason: "primary",
        })
    }

    /// The admitted engines in input order, or every rejection when none survive.
    pub fn admit<'e>(
        &self,
        engines: &'e [Arc<Worker>],
        ctx: &AdmissionContext<'_>,
    ) -> Result<Cow<'e, [Arc<Worker>]>, PickError> {
        let decisions: Vec<AdmissionDecision> = engines
            .iter()
            .map(|engine| self.check(engine, ctx))
            .collect();
        if decisions.iter().all(Result::is_ok) {
            return Ok(Cow::Borrowed(engines));
        }
        let mut admitted = Vec::new();
        let mut rejections = Vec::new();
        for (engine, decision) in engines.iter().zip(decisions) {
            match decision {
                Ok(()) => admitted.push(Arc::clone(engine)),
                Err(reason) => rejections.push(rejection(engine, reason)),
            }
        }
        if admitted.is_empty() {
            return Err(PickError::NoAdmissibleEngine(rejections));
        }
        Ok(Cow::Owned(admitted))
    }
}

fn rejection(engine: &Worker, reason: AdmissionReason) -> EngineRejection {
    EngineRejection {
        engine: engine.id.clone(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::worker;
    use super::*;
    use crate::policies::state::engine_load::EngineLoadTable;
    use std::sync::atomic::Ordering;

    fn ctx<'a>(load: &'a LoadView<'a>) -> AdmissionContext<'a> {
        AdmissionContext {
            load,
            kv_tokens: 1,
            uncached_tokens: 1,
        }
    }

    #[test]
    fn before_selection_hides_rejected_engines_and_reports_them_when_none_survive() {
        let (busy, idle) = (worker("busy"), worker("idle"));
        busy.active_requests.store(2, Ordering::Relaxed);
        let admission = Admission::before(InFlightLimitAdmission { max_in_flight: 2 });
        let table = EngineLoadTable::new();
        let load = LoadView::new(&table);
        let pick = admission
            .select(&[busy.clone(), idle.clone()], &ctx(&load), |admitted| {
                assert_eq!(admitted.len(), 1);
                admitted.first().cloned()
            })
            .unwrap();
        assert_eq!(pick.engine.id.0, "idle");

        idle.active_requests.store(2, Ordering::Relaxed);
        let err = admission
            .select(&[busy, idle], &ctx(&load), |admitted| {
                admitted.first().cloned()
            })
            .unwrap_err();
        assert!(matches!(err, PickError::NoAdmissibleEngine(rejections) if rejections.len() == 2));
    }

    #[test]
    fn after_selection_rejects_the_chosen_engine_without_retrying() {
        let (busy, idle) = (worker("busy"), worker("idle"));
        busy.active_requests.store(2, Ordering::Relaxed);
        let admission = Admission::after(InFlightLimitAdmission { max_in_flight: 2 });
        let table = EngineLoadTable::new();
        let load = LoadView::new(&table);
        let err = admission
            .select(&[busy.clone(), idle], &ctx(&load), |all| {
                all.first().cloned()
            })
            .unwrap_err();
        assert_eq!(
            err,
            PickError::AdmissionRejected(rejection(&busy, AdmissionReason::InFlightLimit))
        );
    }

    #[test]
    fn empty_input_is_no_candidates_and_allow_all_borrows() {
        let table = EngineLoadTable::new();
        let load = LoadView::new(&table);
        let admission = Admission::allow_all();
        assert_eq!(
            admission.select(&[], &ctx(&load), |_| None).unwrap_err(),
            PickError::NoCandidates
        );
        let engines = [worker("a")];
        assert!(matches!(
            admission.admit(&engines, &ctx(&load)),
            Ok(Cow::Borrowed(_))
        ));
    }
}
