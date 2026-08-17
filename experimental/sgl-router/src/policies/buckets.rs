// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! 根据请求形状、SLO profile 和 rank 生成有序 CandidateDomain。

use crate::config::{BucketConfig, BucketSpec, BucketStage, SloBucketPolicy};
use crate::policies::admission::CandidateDomain;
use crate::policies::CacheCandidate;
use crate::workers::Worker;
use std::collections::HashSet;
use std::sync::Arc;

/// Bucket 选择使用的请求字段。
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
    /// Precomputed once and aligned by index with `config.buckets`.
    member_ids: Vec<MemberIndex>,
}

/// Worker-ID lookup with the same semantics as scanning `spec.worker_ids`:
/// exact string matching without normalization, case folding, or trimming.
/// Duplicate IDs do not change the result, and an empty list matches no worker.
#[derive(Debug, Clone)]
enum MemberIndex {
    /// Scan `spec.worker_ids` directly without storing another copy.
    Scan,
    Set(HashSet<String>),
}

/// Measured crossover: SipHash costs more than a few short string comparisons.
const MEMBER_SCAN_MAX: usize = 4;

impl MemberIndex {
    fn new(worker_ids: &[String]) -> Self {
        if worker_ids.len() <= MEMBER_SCAN_MAX {
            Self::Scan
        } else {
            Self::Set(worker_ids.iter().cloned().collect())
        }
    }

    fn contains(&self, worker_ids: &[String], worker_id: &str) -> bool {
        match self {
            Self::Scan => worker_ids.iter().any(|id| id == worker_id),
            Self::Set(ids) => ids.contains(worker_id),
        }
    }
}

impl BucketSelector {
    pub fn new(config: Option<BucketConfig>) -> Self {
        let member_ids: Vec<MemberIndex> = config
            .as_ref()
            .map(|config| {
                config
                    .buckets
                    .iter()
                    .map(|spec| MemberIndex::new(&spec.worker_ids))
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
        .filter_map(|(spec, member_ids)| {
            let members = members(workers, &spec.worker_ids, member_ids);
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
        // 未配置 Decode Bucket 时保持全局 Decode domain。
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
        .filter_map(|(spec, member_ids)| {
            let members = members(workers, &spec.worker_ids, member_ids);
            (!members.is_empty()).then(|| CandidateDomain::bucket_decode(spec.id.clone(), members))
        })
        .collect()
    }

    /// 将全局 Indexer 候选绑定到其 Prefill Bucket，使用 `E` 检查工作量。
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
        let spec =
            config
                .buckets
                .iter()
                .zip(self.member_ids.iter())
                .find_map(|(spec, member_ids)| {
                    (spec.stage == BucketStage::Prefill
                        && member_ids.contains(&spec.worker_ids, &candidate.worker.id.0)
                        && within(
                            candidate.uncached_tokens,
                            spec.min_extend_tokens,
                            spec.max_extend_tokens,
                        )
                        && spec
                            .max_context_tokens
                            .is_none_or(|max_context| request.input_tokens <= max_context)
                        && (config.ttft_slo_policy != SloBucketPolicy::SloFirst
                            || ttft_eligible(spec, request.ttft_slo_ms)))
                    .then_some(spec)
                })?;
        candidate.candidate_range_id = spec.id.clone();
        candidate.max_pending_prefill_tokens = spec.max_pending_prefill_tokens;
        Some(candidate)
    }

    /// 查找全局 Session primary 自己所属的 Prefill Bucket。
    pub fn prefill_affinity_domain(
        &self,
        workers: &[Arc<Worker>],
        primary: &Arc<Worker>,
        request: BucketRequest,
    ) -> Option<CandidateDomain> {
        let config = self.config.as_ref()?;
        let (spec, member_ids) =
            config
                .buckets
                .iter()
                .zip(self.member_ids.iter())
                .find(|(spec, member_ids)| {
                    spec.stage == BucketStage::Prefill
                        && member_ids.contains(&spec.worker_ids, &primary.id.0)
                        && spec
                            .max_context_tokens
                            .is_none_or(|max_context| request.input_tokens <= max_context)
                        && (config.ttft_slo_policy != SloBucketPolicy::SloFirst
                            || ttft_eligible(spec, request.ttft_slo_ms))
                })?;
        let members = members(workers, &spec.worker_ids, member_ids);
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

    fn ordered_specs(
        &self,
        stage: BucketStage,
        slo_policy: SloBucketPolicy,
        compatible: impl Fn(&BucketSpec) -> bool,
        slo_eligible: impl Fn(&BucketSpec) -> bool,
    ) -> Vec<(&BucketSpec, &MemberIndex)> {
        let Some(config) = &self.config else {
            return Vec::new();
        };
        let mut compatible_specs: Vec<(&BucketSpec, &MemberIndex)> = config
            .buckets
            .iter()
            .zip(self.member_ids.iter())
            .filter(|(spec, _)| spec.stage == stage && compatible(spec))
            .collect();
        compatible_specs.sort_by(|(left, _), (right, _)| {
            left.rank
                .cmp(&right.rank)
                .then_with(|| left.id.cmp(&right.id))
        });
        if slo_policy == SloBucketPolicy::Disabled {
            return compatible_specs;
        }

        let mut eligible = Vec::new();
        let mut degraded = Vec::new();
        for (spec, member_ids) in compatible_specs {
            if slo_eligible(spec) {
                eligible.push((spec, member_ids));
            } else {
                degraded.push((spec, member_ids));
            }
        }
        match slo_policy {
            SloBucketPolicy::Disabled => unreachable!("handled before SLO partitioning"),
            SloBucketPolicy::SloFirst => {
                eligible.extend(degraded);
                eligible
            }
            SloBucketPolicy::BestEffort => {
                // Best effort 优先使用不占用 SLO tier 的 bucket。
                degraded.extend(eligible);
                degraded
            }
        }
    }
}

fn members(
    workers: &[Arc<Worker>],
    worker_ids: &[String],
    member_ids: &MemberIndex,
) -> Vec<Arc<Worker>> {
    match member_ids {
        MemberIndex::Scan => workers
            .iter()
            .filter(|worker| worker_ids.iter().any(|id| id == &worker.id.0))
            .cloned()
            .collect(),
        MemberIndex::Set(ids) => workers
            .iter()
            .filter(|worker| ids.contains(&worker.id.0))
            .cloned()
            .collect(),
    }
}

fn prefill_compatible(spec: &BucketSpec, input_tokens: u64) -> bool {
    // No-hit 时 E=L。
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
        // 未知输出长度时只允许 catch-all Decode Bucket。
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
