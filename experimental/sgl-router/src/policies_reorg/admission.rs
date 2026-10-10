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
    pub waiting_requests: Option<u64>,
    pub kv_tokens: Option<u64>,
    pub pending_prefill_tokens: Option<u64>,
    pub inflight_requests: u64,
}

impl EngineMetrics {
    /// Read `engine` from the load snapshot selection already captured.
    pub fn observe(engine: &Worker, load: &EngineReportedLoadSnapshot) -> Self {
        let basic = load.fresh_load_for_url(&engine.url);
        let native = load.fresh_native_cache_load_for_url(&engine.url);
        Self {
            running_requests: basic.map(|load| load.num_running_reqs),
            waiting_requests: basic.map(|load| load.num_waiting_reqs),
            kv_tokens: native.map(|load| load.num_total_tokens),
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

/// Per-engine caps; an unset limit is not checked. A limit admits while the
/// metric is below it. Unknown engine metrics fail open. The default allows
/// everything. Checks observe load; they do not reserve capacity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdmissionLimits {
    pub max_running_requests: Option<u64>,
    pub max_waiting_requests: Option<u64>,
    pub max_kv_tokens: Option<u64>,
    pub max_pending_prefill_tokens: Option<u64>,
    pub max_inflight_requests: Option<u64>,
}

impl EngineAdmission for AdmissionLimits {
    fn check(&self, _: &Worker, engine: &EngineMetrics) -> Result<Decision, PickError> {
        let limits = [
            (
                "max_running_requests",
                self.max_running_requests,
                engine.running_requests,
            ),
            (
                "max_waiting_requests",
                self.max_waiting_requests,
                engine.waiting_requests,
            ),
            ("max_kv_tokens", self.max_kv_tokens, engine.kv_tokens),
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
        for (name, max, current) in limits {
            if let (Some(max), Some(current)) = (max, current) {
                if current >= max {
                    return Ok(Decision::Reject(name.into()));
                }
            }
        }
        Ok(Decision::Allow)
    }
}
