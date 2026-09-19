// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds the prefill and decode pools from `ModelConfig`: one policy
//! instance per bucket, with the admission the legacy path applied implicitly.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use crate::buckets_reorg::{Bucket, Pool, Pools, SloPreference, TokenLimits};
use crate::config::{
    BucketSpec, BucketStage, DecodePolicyKind, FilterKind, ModelConfig, PolicyKind,
    SessionAffinityMode, SloBucketPolicy, StickyFallbackKind,
};
use crate::discovery::WorkerId;
use crate::state::{AffinityStore, PrefixSource};

use super::admission::{
    Admission, AllOf, Capacity, EngineAdmission, InFlightLimit, PendingPrefill,
};
use super::affinity::{AffinityKind, AffinityPolicy};
use super::cache_aware::CacheAwarePolicy;
use super::{least_load, power_of_two, random, round_robin, Policy};

/// Shared services policies hold handles to.
pub struct Dependencies {
    pub affinity: Arc<AffinityStore>,
    pub prefix: Option<Arc<PrefixSource>>,
}

pub fn build_pools(model: &ModelConfig, deps: &Dependencies) -> Result<Pools> {
    let DecodePolicyKind::PowerOfTwo = model.decode_policy else {
        bail!("--decode-policy {:?} is not supported", model.decode_policy);
    };
    let config = model.bucket_config.as_ref();
    let pool = |stage: BucketStage, default: PolicyKind, slo: SloBucketPolicy| -> Result<Pool> {
        let buckets = config
            .into_iter()
            .flat_map(|config| &config.buckets)
            .filter(|spec| spec.stage == stage)
            .map(|spec| bucket(spec, model, default, deps))
            .collect::<Result<Vec<_>>>()?;
        if buckets.is_empty() {
            let admission = admission(default, model, stage, None);
            return Ok(Pool::implicit(build_policy(
                default, admission, model, deps,
            )?));
        }
        Ok(Pool {
            buckets,
            slo: match slo {
                SloBucketPolicy::Disabled => SloPreference::Disabled,
                SloBucketPolicy::SloFirst => SloPreference::SloFirst,
                SloBucketPolicy::BestEffort => SloPreference::BestEffort,
            },
            ..Default::default()
        })
    };
    let mut prefill = pool(
        BucketStage::Prefill,
        model.policy,
        config.map_or(SloBucketPolicy::Disabled, |c| c.ttft_slo_policy),
    )?;
    let session_mode = session_mode(model);
    let global = matches!(model.policy, PolicyKind::Sticky | PolicyKind::CacheAware)
        || (model.policy == PolicyKind::SessionAware
            && session_mode != SessionAffinityMode::Bucket);
    if global && config.is_some() {
        let admission = admission(model.policy, model, BucketStage::Prefill, None);
        prefill.affinity = Some(build_policy(model.policy, admission, model, deps)?);
        prefill.preserve_global_binding = session_mode == SessionAffinityMode::GlobalPreserve;
    }
    Ok(Pools {
        prefill,
        decode: pool(
            BucketStage::Decode,
            PolicyKind::PowerOfTwo,
            config.map_or(SloBucketPolicy::Disabled, |c| c.tps_slo_policy),
        )?,
    })
}

fn bucket(
    spec: &BucketSpec,
    model: &ModelConfig,
    default: PolicyKind,
    deps: &Dependencies,
) -> Result<Bucket> {
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
        policy: build_policy(kind, admission, model, deps)?,
    })
}

pub fn build_policy(
    kind: PolicyKind,
    admission: Admission,
    model: &ModelConfig,
    deps: &Dependencies,
) -> Result<Arc<dyn Policy>> {
    let affinity = |kind, global, fallback| {
        Arc::new(AffinityPolicy {
            kind,
            admission: admission.clone(),
            store: deps.affinity.clone(),
            global,
            fallback,
        })
    };
    Ok(match kind {
        PolicyKind::RoundRobin => Arc::new(round_robin::RoundRobinPolicy::new(admission)),
        PolicyKind::Random => Arc::new(random::RandomPolicy { admission }),
        PolicyKind::PowerOfTwo => Arc::new(power_of_two::PowerOfTwoPolicy { admission }),
        PolicyKind::LoadBased => Arc::new(least_load::LeastLoadPolicy::new(admission)),
        PolicyKind::SessionAware => affinity(
            AffinityKind::Session,
            session_mode(model) != SessionAffinityMode::Bucket,
            Arc::new(power_of_two::PowerOfTwoPolicy::default()),
        ),
        PolicyKind::CacheAware => Arc::new(CacheAwarePolicy {
            source: deps
                .prefix
                .clone()
                .ok_or_else(|| anyhow!("--policy cache_aware needs a prefix source"))?,
            admission,
            config: model.affinity.clone().unwrap_or_default(),
        }),
        PolicyKind::Sticky => {
            let fallback = match model.sticky.as_ref().map(|s| s.fallback_policy) {
                None | Some(StickyFallbackKind::RoundRobin) => PolicyKind::RoundRobin,
                Some(StickyFallbackKind::Random) => PolicyKind::Random,
                Some(StickyFallbackKind::PowerOfTwo) => PolicyKind::PowerOfTwo,
                Some(StickyFallbackKind::LoadBased) => PolicyKind::LoadBased,
            };
            let fallback = build_policy(fallback, Admission::default(), model, deps)?;
            affinity(AffinityKind::Sticky, true, fallback)
        }
        other => bail!("--policy {other} is not supported by policies_reorg yet"),
    })
}

fn session_mode(model: &ModelConfig) -> SessionAffinityMode {
    model
        .affinity
        .as_ref()
        .map_or(SessionAffinityMode::Bucket, |affinity| {
            affinity.session_affinity_mode
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
    use crate::config::{AffinityConfig, BucketConfig, SamplingOverrides};
    use std::time::Duration;

    fn deps() -> Dependencies {
        Dependencies {
            affinity: AffinityStore::new(Duration::from_secs(60)),
            prefix: None,
        }
    }

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
        let pools = build_pools(
            &model(
                PolicyKind::RoundRobin,
                vec![
                    spec("a", BucketStage::Prefill, None),
                    spec("b", BucketStage::Prefill, Some(PolicyKind::PowerOfTwo)),
                ],
            ),
            &deps(),
        )
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
        assert!(pools.prefill.affinity.is_none());
        assert_eq!(pools.prefill.buckets[0].limits.max, Some(4096));
        assert!(pools.decode.buckets[0].worker_ids.is_none());
        assert!(build_pools(&model(PolicyKind::FusedScore, vec![]), &deps()).is_err());
    }

    #[test]
    fn global_session_mode_enables_the_affinity_group() {
        let mut model = model(
            PolicyKind::SessionAware,
            vec![spec("a", BucketStage::Prefill, None)],
        );
        model.affinity = Some(AffinityConfig {
            session_affinity_mode: SessionAffinityMode::GlobalPreserve,
            ..AffinityConfig::default()
        });
        let pools = build_pools(&model, &deps()).unwrap();
        assert!(pools.prefill.affinity.is_some() && pools.prefill.preserve_global_binding);
        assert!(format!("{:?}", pools.prefill.buckets[0].policy).contains("global: true"));
    }
}
