// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-engine admission: a set of caps compared against the selected engine's
//! current measurements. Request size is a bucket concern, not an admission one.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;

use super::PickError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

/// One engine's measurements at pick time. Reported values are `None` without a
/// fresh, complete report, never zero. In-flight requests are counted by this
/// router and always known.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineMetrics {
    pub running_requests: Option<u64>,
    pub running_capacity: Option<u64>,
    pub waiting_requests: Option<u64>,
    pub kv_tokens: Option<u64>,
    pub kv_capacity: Option<u64>,
    pub pending_prefill_tokens: Option<u64>,
    pub inflight_requests: u64,
}

impl EngineMetrics {
    /// Read `engine` from the load snapshot selection already captured.
    pub fn observe(engine: &Worker, load: &EngineReportedLoadSnapshot) -> Self {
        let basic = load.fresh_load_for_url(&engine.url);
        let native = load.fresh_native_cache_load_for_url(&engine.url);
        let capacity = |max: u64| (max > 0).then_some(max);
        Self {
            // Native first, so the count covers the ranks `running_capacity` sums.
            running_requests: native
                .map(|load| load.num_running_reqs)
                .or(basic.map(|load| load.num_running_reqs)),
            running_capacity: native.and_then(|load| capacity(load.max_running_requests)),
            waiting_requests: basic.map(|load| load.num_waiting_reqs),
            kv_tokens: native.map(|load| load.num_total_tokens),
            kv_capacity: native.and_then(|load| capacity(load.max_total_num_tokens)),
            pending_prefill_tokens: native.map(|load| load.num_waiting_uncached_tokens),
            inflight_requests: engine.router_inflight_load() as u64,
        }
    }
}

/// Checks one selected engine. Policies decide when to check and how to handle
/// rejection; admission never selects replacements or reserves capacity.
pub trait EngineAdmission: Send + Sync + Debug {
    fn check(&self, engine: &Worker, metrics: &EngineMetrics) -> Result<Decision, PickError>;
}

/// Per-engine caps; an unset limit is not checked and unknown metrics fail open.
/// Each admits while the engine's load is below it: counts directly, usages as
/// shares, in (0, 1], of the engine's reported capacity. Checks observe load;
/// they do not reserve capacity, and the request itself may take an engine past
/// a cap.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdmissionLimits {
    pub max_running_usage: Option<f64>,
    pub max_kv_usage: Option<f64>,
    pub max_waiting_requests: Option<u64>,
    pub max_pending_prefill_tokens: Option<u64>,
    pub max_inflight_requests: Option<u64>,
}

impl AdmissionLimits {
    /// Rejects usages outside (0, 1] and zero counts, which would admit nothing.
    pub fn validate(&self) -> Result<(), PickError> {
        let share = |usage: Option<f64>| usage.is_none_or(|u| u > 0.0 && u <= 1.0);
        let count = |cap: Option<u64>| cap != Some(0);
        if !(share(self.max_running_usage) && share(self.max_kv_usage)) {
            return Err(PickError::InvalidConfiguration(
                "admission usages must be in (0, 1]".into(),
            ));
        }
        if !(count(self.max_waiting_requests)
            && count(self.max_pending_prefill_tokens)
            && count(self.max_inflight_requests))
        {
            return Err(PickError::InvalidConfiguration(
                "admission counts must be positive".into(),
            ));
        }
        Ok(())
    }

    /// These limits, with each unset one taken from `defaults`.
    pub fn or(&self, defaults: &Self) -> Self {
        Self {
            max_running_usage: self.max_running_usage.or(defaults.max_running_usage),
            max_kv_usage: self.max_kv_usage.or(defaults.max_kv_usage),
            max_waiting_requests: self.max_waiting_requests.or(defaults.max_waiting_requests),
            max_pending_prefill_tokens: self
                .max_pending_prefill_tokens
                .or(defaults.max_pending_prefill_tokens),
            max_inflight_requests: self
                .max_inflight_requests
                .or(defaults.max_inflight_requests),
        }
    }
}

impl EngineAdmission for AdmissionLimits {
    fn check(&self, _: &Worker, engine: &EngineMetrics) -> Result<Decision, PickError> {
        let counts = [
            (
                "max_waiting_requests",
                self.max_waiting_requests,
                engine.waiting_requests,
            ),
            (
                "max_pending_prefill_tokens",
                self.max_pending_prefill_tokens,
                engine.pending_prefill_tokens,
            ),
            (
                "max_inflight_requests",
                self.max_inflight_requests,
                Some(engine.inflight_requests),
            ),
        ];
        for (name, max, current) in counts {
            if let (Some(max), Some(current)) = (max, current) {
                if current >= max {
                    return Ok(Decision::Reject(name.into()));
                }
            }
        }
        let usages = [
            (
                "max_running_usage",
                self.max_running_usage,
                engine.running_requests,
                engine.running_capacity,
            ),
            (
                "max_kv_usage",
                self.max_kv_usage,
                engine.kv_tokens,
                engine.kv_capacity,
            ),
        ];
        for (name, share, used, capacity) in usages {
            if let (Some(share), Some(used), Some(capacity)) = (share, used, capacity) {
                // Dividing exact integers lands on the share itself at the
                // boundary; multiplying the share can round to either side.
                if used as f64 / capacity as f64 >= share {
                    return Ok(Decision::Reject(name.into()));
                }
            }
        }
        Ok(Decision::Allow)
    }
}
