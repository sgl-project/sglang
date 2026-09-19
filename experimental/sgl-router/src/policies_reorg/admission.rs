// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;
use std::sync::Arc;

use crate::state::load_monitor::engine_load::{EngineWorkerLoad, NativeCacheWorkerLoad};
use crate::workers::Worker;

use super::{PickError, PickRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

/// The selected engine's reports from the snapshot used by selection.
/// Missing, stale, or incomplete reports remain `None`, never zero load.
#[derive(Debug, Default, Clone, Copy)]
pub struct AdmissionLoad<'a> {
    pub reported: Option<&'a EngineWorkerLoad>,
    pub native: Option<&'a NativeCacheWorkerLoad>,
}

/// Checks one engine using the load observations retained by selection.
/// Each check defines its missing-data behavior and owns any other state handles it needs.
/// Each policy decides when to check an engine and how to handle rejection.
pub trait EngineAdmission: Send + Sync + Debug {
    fn check(
        &self,
        engine: &Worker,
        request: &PickRequest<'_>,
        load: AdmissionLoad<'_>,
    ) -> Result<Decision, PickError>;
}

#[derive(Debug)]
pub struct AllowAll;

impl EngineAdmission for AllowAll {
    fn check(
        &self,
        _: &Worker,
        _: &PickRequest<'_>,
        _: AdmissionLoad<'_>,
    ) -> Result<Decision, PickError> {
        Ok(Decision::Allow)
    }
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

/// Engine-reported running and KV capacity; admits without a fresh native sample.
#[derive(Debug)]
pub struct Capacity;

impl EngineAdmission for Capacity {
    fn check(
        &self,
        _: &Worker,
        request: &PickRequest<'_>,
        load: AdmissionLoad<'_>,
    ) -> Result<Decision, PickError> {
        let fits = load.native.is_none_or(|load| {
            load.num_running_reqs < load.max_running_requests
                && load
                    .num_total_tokens
                    .checked_add(request.kv_tokens())
                    .is_some_and(|projected| projected <= load.max_total_num_tokens)
        });
        Ok(Decision::from(fits, "kv_capacity"))
    }
}

/// Router-local in-flight requests must stay below the limit.
#[derive(Debug)]
pub struct InFlightLimit(pub usize);

impl EngineAdmission for InFlightLimit {
    fn check(
        &self,
        engine: &Worker,
        _: &PickRequest<'_>,
        _: AdmissionLoad<'_>,
    ) -> Result<Decision, PickError> {
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
    fn check(
        &self,
        _: &Worker,
        _: &PickRequest<'_>,
        load: AdmissionLoad<'_>,
    ) -> Result<Decision, PickError> {
        let below = load
            .reported
            .is_none_or(|load| load.num_waiting_reqs < self.0);
        Ok(Decision::from(below, "queue_limit"))
    }
}

/// Every check must admit; the first rejection or error stops evaluation.
#[derive(Debug)]
pub struct AllOf(pub Vec<Arc<dyn EngineAdmission>>);

impl EngineAdmission for AllOf {
    fn check(
        &self,
        engine: &Worker,
        request: &PickRequest<'_>,
        load: AdmissionLoad<'_>,
    ) -> Result<Decision, PickError> {
        for check in &self.0 {
            if let rejected @ Decision::Reject(_) = check.check(engine, request, load)? {
                return Ok(rejected);
            }
        }
        Ok(Decision::Allow)
    }
}
