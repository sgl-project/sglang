// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{AffinityConfig, AffinityMode, BalancedBy};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;

use super::admission::EngineMetrics;
use super::{Pick, PickError};

pub(super) fn choose(
    config: &AffinityConfig,
    affinity: Option<Pick>,
    alternative: Result<Pick, PickError>,
    load: &EngineReportedLoadSnapshot,
) -> Result<Pick, PickError> {
    let Some(affinity) = affinity else {
        return alternative;
    };
    match alternative {
        Ok(pick) if prefer_alternative(config, &affinity.engine, &pick.engine, load) => Ok(pick),
        Ok(_)
        | Err(
            PickError::NoCandidates
            | PickError::AdmissionRejected(_)
            | PickError::NoAdmissibleEngine(_),
        ) => Ok(affinity),
        Err(error) => Err(error),
    }
}

fn prefer_alternative(
    config: &AffinityConfig,
    affinity: &Worker,
    alternative: &Worker,
    load: &EngineReportedLoadSnapshot,
) -> bool {
    if config.mode != AffinityMode::Balanced {
        return false;
    }
    let metric = |engine| {
        let metrics = EngineMetrics::observe(engine, load);
        match config.balanced_by {
            BalancedBy::PendingPrefillTokens => metrics.pending_prefill_tokens,
            BalancedBy::RunningRequests => metrics.running_requests,
        }
    };
    let (Some(a), Some(b)) = (metric(affinity), metric(alternative)) else {
        return false;
    };
    a.saturating_sub(b) > config.load_gap() && a as f64 > b as f64 * config.load_factor
}
