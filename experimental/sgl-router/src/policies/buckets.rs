// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds ordered candidate domains from request shape, SLO profile, and rank.

use crate::config::{BucketConfig, BucketSpec, BucketStage, SloBucketPolicy};
use crate::policies::admission::CandidateDomain;
use crate::policies::CacheCandidate;
use crate::workers::Worker;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Request fields used for bucket selection.
#[derive(Debug, Clone, Copy)]
pub struct BucketRequest {
    pub input_tokens: u64,
    pub expected_peak_sequence_tokens: Option<u64>,
    pub ttft_slo_ms: Option<u64>,
    pub tps_slo: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct BucketSelector {
    config: Option<BucketConfig>,
    /// Precomputed only for buckets above the measured scan/hash crossover.
    member_ids: HashMap<String, HashSet<String>>,
}

/// Measured crossover: SipHash costs more than a few short string comparisons.
const MEMBER_SCAN_MAX: usize = 4;

impl BucketSelector {
    pub fn new(config: Option<BucketConfig>) -> Self {
        let member_ids = config
            .as_ref()
            .map(|config| {
                config
                    .buckets
                    .iter()
                    .filter(|spec| spec.worker_ids.len() > MEMBER_SCAN_MAX)
                    .map(|spec| {
                        (
                            spec.id.clone(),
                            spec.worker_ids.iter().cloned().collect::<HashSet<_>>(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self { config, member_ids }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    pub fn prefill_domains(
        &self,
        workers: &[Arc<Worker>],
        request: BucketRequest,
    ) -> Vec<CandidateDomain> {
        let Some(config) = &self.config else {
            return vec![CandidateDomain::global_prefill(workers)];
        };
        self.ordered_specs(
            BucketStage::Prefill,
            config.ttft_slo_policy,
            |spec| prefill_compatible(spec, request.input_tokens),
            |spec| ttft_eligible(spec, request.ttft_slo_ms),
        )
        .into_iter()
        .filter_map(|spec| {
            let members = self.members(workers, spec);
            (!members.is_empty()).then(|| {
                CandidateDomain::bucket_prefill(
                    spec.id.clone(),
                    members,
                    spec.max_pending_prefill_tokens,
                )
            })
        })
        .collect()
    }

    pub fn decode_domains(
        &self,
        workers: &[Arc<Worker>],
        request: BucketRequest,
    ) -> Vec<CandidateDomain> {
        let Some(config) = &self.config else {
            return vec![CandidateDomain::global_decode(workers)];
        };
        // Keep a global decode domain when no decode bucket is configured.
        if !config
            .buckets
            .iter()
            .any(|spec| spec.stage == BucketStage::Decode)
        {
            return vec![CandidateDomain::global_decode(workers)];
        }
        self.ordered_specs(
            BucketStage::Decode,
            config.tps_slo_policy,
            |spec| {
                decode_compatible(
                    spec,
                    request.input_tokens,
                    request.expected_peak_sequence_tokens,
                )
            },
            |spec| tps_eligible(spec, request.tps_slo),
        )
        .into_iter()
        .filter_map(|spec| {
            let members = self.members(workers, spec);
            (!members.is_empty()).then(|| CandidateDomain::bucket_decode(spec.id.clone(), members))
        })
        .collect()
    }

    /// Maps global Indexer candidates to prefill buckets using `E` as the workload.
    pub fn bind_prefill_cache_candidate(
        &self,
        mut candidate: CacheCandidate,
        request: BucketRequest,
    ) -> Option<CacheCandidate> {
        let Some(config) = &self.config else {
            candidate.candidate_range_id = "global".to_string();
            candidate.max_pending_prefill_tokens = None;
            return Some(candidate);
        };
        let spec = config.buckets.iter().find(|spec| {
            spec.stage == BucketStage::Prefill
                && self.contains(spec, &candidate.worker.id.0)
                && within(
                    candidate.uncached_tokens,
                    spec.min_extend_tokens,
                    spec.max_extend_tokens,
                )
                && spec
                    .max_context_tokens
                    .is_none_or(|max_context| request.input_tokens <= max_context)
                && (config.ttft_slo_policy != SloBucketPolicy::SloFirst
                    || ttft_eligible(spec, request.ttft_slo_ms))
        })?;
        candidate.candidate_range_id = spec.id.clone();
        candidate.max_pending_prefill_tokens = spec.max_pending_prefill_tokens;
        Some(candidate)
    }

    /// Finds the prefill bucket containing a global session primary.
    pub fn prefill_affinity_domain(
        &self,
        workers: &[Arc<Worker>],
        primary: &Arc<Worker>,
        request: BucketRequest,
    ) -> Option<CandidateDomain> {
        let config = self.config.as_ref()?;
        let spec = config.buckets.iter().find(|spec| {
            spec.stage == BucketStage::Prefill
                && self.contains(spec, &primary.id.0)
                && spec
                    .max_context_tokens
                    .is_none_or(|max_context| request.input_tokens <= max_context)
                && (config.ttft_slo_policy != SloBucketPolicy::SloFirst
                    || ttft_eligible(spec, request.ttft_slo_ms))
        })?;
        let members = self.members(workers, spec);
        members
            .iter()
            .any(|worker| worker.id == primary.id)
            .then(|| {
                CandidateDomain::bucket_prefill(
                    spec.id.clone(),
                    members,
                    spec.max_pending_prefill_tokens,
                )
            })
    }

    fn contains(&self, spec: &BucketSpec, worker_id: &str) -> bool {
        if spec.worker_ids.len() <= MEMBER_SCAN_MAX {
            return spec.worker_ids.iter().any(|id| id == worker_id);
        }
        self.member_ids
            .get(&spec.id)
            .is_some_and(|ids| ids.contains(worker_id))
    }

    fn members(&self, workers: &[Arc<Worker>], spec: &BucketSpec) -> Vec<Arc<Worker>> {
        if spec.worker_ids.len() <= MEMBER_SCAN_MAX {
            return workers
                .iter()
                .filter(|worker| spec.worker_ids.iter().any(|id| id == &worker.id.0))
                .cloned()
                .collect();
        }
        let ids = self
            .member_ids
            .get(&spec.id)
            .expect("large bucket member index is built with the config");
        workers
            .iter()
            .filter(|worker| ids.contains(&worker.id.0))
            .cloned()
            .collect()
    }

    fn ordered_specs(
        &self,
        stage: BucketStage,
        slo_policy: SloBucketPolicy,
        compatible: impl Fn(&BucketSpec) -> bool,
        slo_eligible: impl Fn(&BucketSpec) -> bool,
    ) -> Vec<&BucketSpec> {
        let Some(config) = &self.config else {
            return Vec::new();
        };
        let mut compatible_specs: Vec<&BucketSpec> = config
            .buckets
            .iter()
            .filter(|spec| spec.stage == stage && compatible(spec))
            .collect();
        compatible_specs.sort_by(|left, right| {
            left.rank
                .cmp(&right.rank)
                .then_with(|| left.id.cmp(&right.id))
        });
        if slo_policy == SloBucketPolicy::Disabled {
            return compatible_specs;
        }

        let mut eligible = Vec::new();
        let mut degraded = Vec::new();
        for spec in compatible_specs {
            if slo_eligible(spec) {
                eligible.push(spec);
            } else {
                degraded.push(spec);
            }
        }
        match slo_policy {
            SloBucketPolicy::Disabled => unreachable!("handled before SLO partitioning"),
            SloBucketPolicy::SloFirst => {
                eligible.extend(degraded);
                eligible
            }
            SloBucketPolicy::BestEffort => {
                // Best effort prefers a bucket without an SLO tier.
                degraded.extend(eligible);
                degraded
            }
        }
    }
}

fn prefill_compatible(spec: &BucketSpec, input_tokens: u64) -> bool {
    // With no cache hit, E equals L.
    within(input_tokens, spec.min_extend_tokens, spec.max_extend_tokens)
        && spec
            .max_context_tokens
            .is_none_or(|max_context| input_tokens <= max_context)
}

fn decode_compatible(
    spec: &BucketSpec,
    input_tokens: u64,
    expected_peak_sequence_tokens: Option<u64>,
) -> bool {
    let Some(expected_peak_sequence_tokens) = expected_peak_sequence_tokens else {
        // Unknown output length can only use a catch-all decode bucket.
        return spec.min_sequence_tokens.is_none()
            && spec.max_sequence_tokens.is_none()
            && spec
                .max_context_tokens
                .is_none_or(|max_context| input_tokens <= max_context);
    };
    within(
        expected_peak_sequence_tokens,
        spec.min_sequence_tokens,
        spec.max_sequence_tokens,
    ) && spec
        .max_context_tokens
        .is_none_or(|max_context| expected_peak_sequence_tokens <= max_context)
}

fn within(value: u64, min: Option<u64>, max: Option<u64>) -> bool {
    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
}

fn ttft_eligible(spec: &BucketSpec, request_ttft_slo_ms: Option<u64>) -> bool {
    let Some(request_ttft_slo_ms) = request_ttft_slo_ms else {
        return true;
    };
    spec.ttft_p95_at_capacity_ms
        .is_some_and(|p95| p95 <= request_ttft_slo_ms)
}

fn tps_eligible(spec: &BucketSpec, request_tps_slo: Option<f64>) -> bool {
    let Some(request_tps_slo) = request_tps_slo else {
        return true;
    };
    spec.tps_p05_at_capacity
        .is_some_and(|p05| p05 >= request_tps_slo)
}
