// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::state::load_monitor::engine_load::EngineLoadSnapshot;
use crate::workers::Worker;

use super::{PickError, PickRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

/// Measurements for the selected engine, retained from selection's snapshot.
/// Missing, stale, or incomplete measurements remain unknown, never zero.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionState {
    pub running_requests: Option<u64>,
    /// Total KV footprint, not just the occupied KV tokens in basic reports.
    pub kv_tokens: Option<u64>,
}

impl AdmissionState {
    pub fn from_snapshot(snapshot: &EngineLoadSnapshot, engine: &Worker) -> Self {
        Self {
            running_requests: snapshot
                .fresh_load_for_url(&engine.url)
                .map(|load| load.num_running_reqs),
            kv_tokens: snapshot
                .fresh_native_cache_load_for_url(&engine.url)
                .map(|load| load.num_total_tokens),
        }
    }
}

/// Checks one engine using the measurements retained by selection.
/// `None` means no usable measurement, never zero load. Each check defines
/// its missing-data behavior and owns any other state handles it needs.
/// Each policy decides when to check an engine and how to handle rejection.
pub trait EngineAdmission: Send + Sync + Debug {
    fn check(
        &self,
        engine: &Worker,
        request: &PickRequest<'_>,
        state: AdmissionState,
    ) -> Result<Decision, PickError>;
}

/// Each admission name carries only the limits used by that rule.
/// Limits are absolute, per-engine caps (aggregated across its DP ranks).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdmissionConfig {
    AllowAll {},
    RunningPlusKvCapacity {
        max_running_requests: u64,
        max_kv_tokens: u64,
    },
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self::AllowAll {}
    }
}

impl EngineAdmission for AdmissionConfig {
    fn check(
        &self,
        _: &Worker,
        request: &PickRequest<'_>,
        state: AdmissionState,
    ) -> Result<Decision, PickError> {
        let Self::RunningPlusKvCapacity {
            max_running_requests,
            max_kv_tokens,
        } = self
        else {
            return Ok(Decision::Allow);
        };

        // Include this request. Comparing before addition avoids running-count
        // overflow, while checked_add rejects an overflowing KV projection.
        if state
            .running_requests
            .is_some_and(|running| running >= *max_running_requests)
        {
            return Ok(Decision::Reject("running_capacity".into()));
        }
        let request_tokens = request.expected_peak_tokens.unwrap_or(request.input_tokens);
        if state.kv_tokens.is_some_and(|tokens| {
            tokens
                .checked_add(request_tokens)
                .is_none_or(|projected| projected > *max_kv_tokens)
        }) {
            return Ok(Decision::Reject("kv_capacity".into()));
        }
        // Preserve fail-open behavior for unavailable measurements. A known
        // measurement is still checked when the other one is unavailable.
        Ok(Decision::Allow)
    }
}
