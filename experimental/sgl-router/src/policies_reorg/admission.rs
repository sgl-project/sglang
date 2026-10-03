// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-engine admission: a set of caps compared against the selected engine's
//! current measurements; usage caps also count the request itself.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;

use super::{PickError, PickRequest, Stage};

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
    /// KV tokens this request adds: its input on prefill, its expected peak otherwise.
    pub request_tokens: u64,
}

impl EngineMetrics {
    /// Read `engine` from the load snapshot selection already captured.
    pub fn observe(
        engine: &Worker,
        load: &EngineReportedLoadSnapshot,
        request: &PickRequest<'_>,
    ) -> Self {
        let basic = load.fresh_load_for_url(&engine.url);
        let native = load.fresh_native_cache_load_for_url(&engine.url);
        let capacity = |max: u64| (max > 0).then_some(max);
        Self {
            running_requests: basic.map(|load| load.num_running_reqs),
            running_capacity: native.and_then(|load| capacity(load.max_running_requests)),
            waiting_requests: basic.map(|load| load.num_waiting_reqs),
            kv_tokens: native.map(|load| load.num_total_tokens),
            kv_capacity: native.and_then(|load| capacity(load.max_total_num_tokens)),
            pending_prefill_tokens: native.map(|load| load.num_waiting_uncached_tokens),
            inflight_requests: engine.router_inflight_load() as u64,
            request_tokens: match request.stage {
                Stage::Prefill => request.input_tokens,
                _ => request.expected_peak_tokens.unwrap_or(request.input_tokens),
            },
        }
    }
}

/// Checks one selected engine. Policies decide when to check and how to handle
/// rejection; admission never selects replacements or reserves capacity.
pub trait EngineAdmission: Send + Sync + Debug {
    fn check(&self, engine: &Worker, metrics: &EngineMetrics) -> Result<Decision, PickError>;
}

/// Per-engine caps; an unset limit is not checked and unknown metrics fail open.
/// Counts admit while below their cap. Usages are shares, in (0, 1], of the
/// engine's reported capacity that its load plus this request may fill.
/// Checks observe load; they do not reserve capacity.
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
    pub fn validate(&self) -> Result<(), PickError> {
        let share = |usage: Option<f64>| usage.is_none_or(|u| u > 0.0 && u <= 1.0);
        if share(self.max_running_usage) && share(self.max_kv_usage) {
            Ok(())
        } else {
            Err(PickError::InvalidConfiguration(
                "admission usages must be in (0, 1]".into(),
            ))
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
                engine.running_requests.map(|running| running + 1),
                engine.running_capacity,
            ),
            (
                "max_kv_usage",
                self.max_kv_usage,
                engine
                    .kv_tokens
                    .map(|kv| kv.saturating_add(engine.request_tokens)),
                engine.kv_capacity,
            ),
        ];
        for (name, share, used, capacity) in usages {
            if let (Some(share), Some(used), Some(capacity)) = (share, used, capacity) {
                if used as f64 > share * capacity as f64 {
                    return Ok(Decision::Reject(name.into()));
                }
            }
        }
        Ok(Decision::Allow)
    }
}
