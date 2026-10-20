// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Decode affinity to confirmed device/host prefixes within an eligible group.

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;

use crate::config::{AffinityConfig, AffinityMode, BalancedBy};
use crate::state::kv_events::RadixTreePrefixProvider;
use crate::state::load_monitor::engine_ranking::CandidateLoads;
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
use crate::workers::Worker;

use super::admission::{partition_admitted, AdmissionLimits, EngineAdmission};
use super::affinity;
use super::cache_aware::validate_config;
use super::power_of_two::PowerOfTwoPolicy;
use super::{Pick, PickError, PickRequest, Policy, Stage};

#[derive(Debug)]
pub struct DecodeCacheAwarePolicy {
    provider: RadixTreePrefixProvider,
    engine_load: Arc<EngineReportedLoadTable>,
    config: AffinityConfig,
    pub admission: Arc<dyn EngineAdmission>,
}

impl DecodeCacheAwarePolicy {
    pub fn new(
        provider: RadixTreePrefixProvider,
        engine_load: Arc<EngineReportedLoadTable>,
        config: AffinityConfig,
    ) -> Result<Self, PickError> {
        validate_config(&config)?;
        if !matches!(config.mode, AffinityMode::Prefer | AffinityMode::Balanced)
            || (config.mode == AffinityMode::Balanced
                && config.balanced_by != BalancedBy::RunningRequests)
        {
            return Err(PickError::InvalidConfiguration(
                "decode cache affinity requires prefer or balanced mode and balances running_requests, not prefill_tokens".into(),
            ));
        }
        Ok(Self {
            provider,
            engine_load,
            config,
            admission: Arc::new(AdmissionLimits::default()),
        })
    }
}

impl Policy for DecodeCacheAwarePolicy {
    fn supports(&self, stage: Stage) -> bool {
        stage == Stage::Decode
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            if request.stage != Stage::Decode {
                return Err(PickError::InvalidConfiguration(
                    "decode cache affinity requires a decode group".into(),
                ));
            }
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            let prefixes = request
                .token_ids
                .map(|tokens| {
                    self.provider
                        .decode_prefixes(tokens, request.cache_namespace)
                })
                .unwrap_or_default();
            let load = self.engine_load.capture_snapshot(Instant::now());
            // Admission precedes affinity: a cached prefix cannot bypass a cap.
            let (admitted, rejections) =
                partition_admitted(self.admission.as_ref(), engines, &load)?;
            if admitted.is_empty() {
                return Err(PickError::NoAdmissibleEngine(rejections));
            }
            let candidates: Vec<_> = admitted
                .iter()
                .filter_map(|engine| {
                    // A URL-level hit cannot choose the matching DP rank at dispatch.
                    if engine.dp_ranks() != 1 {
                        return None;
                    }
                    let &(resident, device) = prefixes.get(&engine.url)?;
                    let resident = resident.min(request.input_tokens);
                    let ratio = resident as f64 / request.input_tokens.max(1) as f64;
                    (resident > 0
                        && self
                            .config
                            .cache_affinity_min_matched_tokens
                            .is_none_or(|min| resident >= min)
                        && self
                            .config
                            .cache_affinity_min_match_ratio
                            .is_none_or(|min| ratio >= min))
                    .then_some((engine, resident, device.min(resident)))
                })
                .collect();
            let loads = CandidateLoads::new(Some(&load), candidates.iter().map(|c| c.0));
            // Avoid the most transfer first; prefer in-place reuse at equal depth.
            let best = candidates.iter().min_by(|left, right| {
                right
                    .1
                    .cmp(&left.1)
                    .then_with(|| right.2.cmp(&left.2))
                    .then_with(|| loads.compare_decode_engines(left.0, right.0))
                    .then_with(|| left.0.id.0.cmp(&right.0.id.0))
            });
            let affinity = best.map(|(engine, _, _)| Pick {
                engine: Arc::clone(engine),
                reason: "decode_cache_candidate",
            });
            if self.config.mode != AffinityMode::Balanced {
                if let Some(pick) = affinity {
                    return Ok(pick);
                }
            }
            let alternatives: Vec<_> = admitted
                .into_iter()
                .filter(|engine| {
                    affinity
                        .as_ref()
                        .is_none_or(|pick| pick.engine.id != engine.id)
                })
                .collect();
            let mut fallback = PowerOfTwoPolicy::new(Arc::clone(&self.engine_load));
            fallback.admission = Arc::clone(&self.admission);
            let alternative = fallback.pick(&alternatives, request).await.map(|mut pick| {
                pick.reason = if affinity.is_some() {
                    "decode_affinity_load"
                } else {
                    "no_decode_cache_candidate"
                };
                pick
            });
            affinity::choose(&self.config, affinity, alternative, &load, |_| 0)
        })
    }
}
