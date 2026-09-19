// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds the prefill and decode pools from `ModelConfig`: one policy
//! instance per bucket, with the admission the legacy path applied implicitly.

use std::sync::Arc;

use anyhow::{bail, Result};

use crate::buckets_reorg::{Bucket, Pool, Pools, SloPreference, TokenLimits};
use crate::config::{
    BucketSpec, BucketStage, DecodePolicyKind, FilterKind, ModelConfig, PolicyKind, SloBucketPolicy,
};
use crate::discovery::WorkerId;

use super::admission::{
    Admission, AllOf, Capacity, EngineAdmission, InFlightLimit, PendingPrefill,
};
use super::{least_load, power_of_two, random, round_robin, Policy};

pub fn build_pools(model: &ModelConfig) -> Result<Pools> {
    let DecodePolicyKind::PowerOfTwo = model.decode_policy else {
        bail!("--decode-policy {:?} is not supported", model.decode_policy);
    };
    let config = model.bucket_config.as_ref();
    let pool = |stage: BucketStage, default: PolicyKind, slo: SloBucketPolicy| -> Result<Pool> {
        let buckets = config
            .into_iter()
            .flat_map(|config| &config.buckets)
            .filter(|spec| spec.stage == stage)
            .map(|spec| bucket(spec, model, default))
            .collect::<Result<Vec<_>>>()?;
        if buckets.is_empty() {
            let admission = admission(default, model, stage, None);
            return Ok(Pool::implicit(build_policy(default, admission)?));
        }
        Ok(Pool {
            buckets,
            slo: match slo {
                SloBucketPolicy::Disabled => SloPreference::Disabled,
                SloBucketPolicy::SloFirst => SloPreference::SloFirst,
                SloBucketPolicy::BestEffort => SloPreference::BestEffort,
            },
        })
    };
    Ok(Pools {
        prefill: pool(
            BucketStage::Prefill,
            model.policy,
            config.map_or(SloBucketPolicy::Disabled, |c| c.ttft_slo_policy),
        )?,
        decode: pool(
            BucketStage::Decode,
            PolicyKind::PowerOfTwo,
            config.map_or(SloBucketPolicy::Disabled, |c| c.tps_slo_policy),
        )?,
    })
}

fn bucket(spec: &BucketSpec, model: &ModelConfig, default: PolicyKind) -> Result<Bucket> {
    let kind = spec.policy.unwrap_or(default);
    let (min, max) = match spec.stage {
        BucketStage::Prefill => (spec.min_extend_tokens, spec.max_extend_tokens),
        BucketStage::Decode => (spec.min_sequence_tokens, spec.max_sequence_tokens),
    };
    let admission = admission(kind, model, spec.stage, spec.max_pending_prefill_tokens);
    Ok(Bucket {
        id: spec.id.clone(),
        rank: spec.rank,
        worker_ids: Some(spec.worker_ids.iter().cloned().map(WorkerId).collect()),
        limits: TokenLimits {
            min,
            max,
            context: spec.max_context_tokens,
        },
        ttft_ms: spec.ttft_p95_at_capacity_ms,
        tokens_per_second: spec.tps_p05_at_capacity,
        policy: build_policy(kind, admission)?,
    })
}

pub fn build_policy(kind: PolicyKind, admission: Admission) -> Result<Arc<dyn Policy>> {
    Ok(match kind {
        PolicyKind::RoundRobin => Arc::new(round_robin::RoundRobinPolicy::new(admission)),
        PolicyKind::Random => Arc::new(random::RandomPolicy { admission }),
        PolicyKind::PowerOfTwo => Arc::new(power_of_two::PowerOfTwoPolicy { admission }),
        PolicyKind::LoadBased => Arc::new(least_load::LeastLoadPolicy::new(admission)),
        other => bail!("--policy {other} is not supported by policies_reorg yet"),
    })
}

/// The checks the legacy path applies implicitly for `kind`, kept explicit.
fn admission(
    kind: PolicyKind,
    model: &ModelConfig,
    stage: BucketStage,
    pending_prefill_budget: Option<u64>,
) -> Admission {
    let mut checks: Vec<Arc<dyn EngineAdmission>> = Vec::new();
    if matches!(
        kind,
        PolicyKind::PowerOfTwo | PolicyKind::SessionAware | PolicyKind::CacheAware
    ) {
        checks.push(Arc::new(Capacity));
        if let (BucketStage::Prefill, Some(budget)) = (stage, pending_prefill_budget) {
            checks.push(Arc::new(PendingPrefill(budget)));
        }
    }
    let overloaded = model
        .eligibility
        .as_ref()
        .filter(|eligibility| eligibility.filters.contains(&FilterKind::Overloaded))
        .and_then(|eligibility| eligibility.max_in_flight);
    if let Some(max_in_flight) = overloaded {
        checks.push(Arc::new(InFlightLimit(max_in_flight)));
    }
    match checks.len() {
        0 => Admission::default(),
        1 => Admission {
            check: checks.pop().unwrap(),
            placement: Default::default(),
        },
        _ => Admission::before(AllOf(checks)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BucketConfig, SamplingOverrides};

    fn model(policy: PolicyKind, buckets: Vec<BucketSpec>) -> ModelConfig {
        ModelConfig {
            id: "m".into(),
            tokenizer_path: "m".into(),
            policy,
            decode_policy: DecodePolicyKind::PowerOfTwo,
            bucket_config: (!buckets.is_empty()).then_some(BucketConfig {
                buckets,
                ttft_slo_policy: SloBucketPolicy::SloFirst,
                tps_slo_policy: SloBucketPolicy::Disabled,
            }),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: SamplingOverrides::default(),
        }
    }

    fn spec(id: &str, stage: BucketStage, policy: Option<PolicyKind>) -> BucketSpec {
        BucketSpec {
            id: id.into(),
            stage,
            rank: 0,
            worker_ids: vec![id.into()],
            min_extend_tokens: None,
            max_extend_tokens: Some(4096),
            min_sequence_tokens: None,
            max_sequence_tokens: None,
            max_context_tokens: None,
            ttft_p95_at_capacity_ms: None,
            tps_p05_at_capacity: None,
            max_pending_prefill_tokens: Some(1024),
            policy,
        }
    }

    #[test]
    fn buckets_attach_their_own_policy_and_unbucketed_stages_are_implicit() {
        let pools = build_pools(&model(
            PolicyKind::RoundRobin,
            vec![
                spec("a", BucketStage::Prefill, None),
                spec("b", BucketStage::Prefill, Some(PolicyKind::PowerOfTwo)),
            ],
        ))
        .unwrap();
        let kinds: Vec<_> = pools
            .prefill
            .buckets
            .iter()
            .map(|b| format!("{:?}", b.policy))
            .collect();
        assert!(
            kinds[0].starts_with("RoundRobinPolicy") && kinds[1].starts_with("PowerOfTwoPolicy")
        );
        assert!(kinds[1].contains("PendingPrefill(1024)") && !kinds[0].contains("Capacity"));
        assert_eq!(pools.prefill.slo, SloPreference::SloFirst);
        assert_eq!(pools.prefill.buckets[0].limits.max, Some(4096));
        assert!(pools.decode.buckets[0].worker_ids.is_none());
        assert!(build_pools(&model(PolicyKind::FusedScore, vec![])).is_err());
    }
}
