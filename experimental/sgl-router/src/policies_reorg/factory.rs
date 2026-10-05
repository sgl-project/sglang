// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::{sync::Arc, time::Duration};

use anyhow::{bail, ensure, Result};
use serde::Deserialize;

use crate::buckets_reorg::{
    Bucket, BucketGroups, BucketResolver, EngineGroup, SloPreference, TokenLimits,
};
use crate::config::{
    AffinityConfig, AffinityMode, BalancedBy, DecodePolicyKind, FilterKind, ModelConfig,
    PolicyKind, SessionAffinityMode,
};
use crate::discovery::WorkerId;
use crate::state::kv_events::RadixTreePrefixProvider;
use crate::state::{
    kv_events::KvEventIndex, load_monitor::router_inflight_load::JanitorHandle, AffinityStore,
};

use super::{
    admission::AdmissionLimits,
    cache_aware::{CacheAwarePolicy, CacheSource},
    power_of_two::PowerOfTwoPolicy,
    session_aware::SessionAwarePolicy,
    Policy, Stage,
};

/// Reorg `--bucket-config`: plain or P/D buckets whose groups each own their
/// membership, policy and admission.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BucketsConfig {
    pub buckets: Vec<BucketSpec>,
    pub ttft_slo: SloPreference,
    pub tps_slo: SloPreference,
}

/// Set either `plain` or both `prefill` and `decode`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BucketSpec {
    pub id: String,
    pub rank: u32,
    pub min_input_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_context_tokens: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub plain: Option<GroupSpec>,
    pub prefill: Option<GroupSpec>,
    pub decode: Option<GroupSpec>,
}

/// Omitted fields mean every engine of the role and `--policy` (power-of-two on
/// decode); admission limits left unset take `--max-in-flight` / `--max-kv-usage`,
/// and affinity settings left unset take `--affinity-*`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroupSpec {
    pub worker_ids: Option<HashSet<WorkerId>>,
    /// Any listed Kubernetes Service, as namespace/name; exclusive with worker_ids.
    pub worker_services: Option<HashSet<String>>,
    pub policy: Option<PolicyKind>,
    pub admission: Option<AdmissionLimits>,
    pub affinity: Option<AffinitySpec>,
}

/// Per-group `--affinity-mode`, `--affinity-balanced-by` and `--affinity-load-*`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AffinitySpec {
    pub mode: Option<AffinityMode>,
    pub balanced_by: Option<BalancedBy>,
    pub load_factor: Option<f64>,
    pub load_gap: Option<u64>,
}

impl AffinitySpec {
    /// `defaults` with each set field replaced, validated like the CLI flags.
    fn or(&self, defaults: &AffinityConfig) -> Result<AffinityConfig> {
        let config = AffinityConfig {
            mode: self.mode.unwrap_or(defaults.mode),
            balanced_by: self.balanced_by.unwrap_or(defaults.balanced_by),
            load_factor: self.load_factor.unwrap_or(defaults.load_factor),
            load_gap: self.load_gap.or(defaults.load_gap),
            ..defaults.clone()
        };
        ensure!(
            matches!(config.mode, AffinityMode::Prefer | AffinityMode::Balanced),
            "group affinity mode must be prefer or balanced"
        );
        ensure!(
            config.mode == AffinityMode::Balanced
                || (self.balanced_by.is_none()
                    && self.load_factor.is_none()
                    && self.load_gap.is_none()),
            "group affinity balanced_by and load_* require mode balanced"
        );
        ensure!(
            config.load_factor.is_finite() && config.load_factor >= 1.0,
            "group affinity load_factor must be finite and at least 1"
        );
        Ok(config)
    }
}

impl BucketsConfig {
    /// Discovery decides which bucket has candidates. Plain ranks first so plain
    /// deployments do not scan for prefill engines on every request.
    fn default_buckets() -> Self {
        let all = || Some(GroupSpec::default());
        Self {
            buckets: vec![
                BucketSpec {
                    id: "plain".into(),
                    plain: all(),
                    ..Default::default()
                },
                BucketSpec {
                    id: "pd".into(),
                    rank: 1,
                    prefill: all(),
                    decode: all(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }
}

pub fn validate(model: &ModelConfig) -> Result<()> {
    ensure!(
        matches!(
            model.policy,
            PolicyKind::PowerOfTwo | PolicyKind::CacheAware | PolicyKind::SessionAware
        ),
        "reorg routing supports power_of_two, cache_aware, and session_aware"
    );
    ensure!(
        model.bucket_config.is_none(),
        "reorg routing reads reorg_buckets, not the legacy bucket_config"
    );
    ensure!(
        model.decode_policy == DecodePolicyKind::PowerOfTwo,
        "reorg routing requires --decode-policy power_of_two"
    );
    if let Some(filters) = &model.eligibility {
        ensure!(
            filters
                .filters
                .iter()
                .all(|filter| *filter == FilterKind::Overloaded),
            "reorg routing only supports --filter overloaded"
        );
    }
    if let Some(affinity) = &model.affinity {
        ensure!(
            !affinity.stable_pair && affinity.session_affinity_mode == SessionAffinityMode::Bucket,
            "reorg routing uses bucket-scoped sessions without --stable-pair"
        );
        ensure!(
            affinity.min_load_choices == 2,
            "reorg cache fallback requires --min-load-choices 2"
        );
    }
    Ok(())
}

/// Build `model.reorg_buckets`, or the default plain and P/D buckets. Callers
/// run [`validate`] first; `Cli::into_config` does so before startup reaches this point.
pub fn build_resolver(
    model: &ModelConfig,
    state: &KvEventIndex,
    external_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
) -> Result<(BucketResolver, Option<JanitorHandle>)> {
    let mut groups = Groups {
        model,
        state,
        external_index,
        affinity: model.affinity.clone().unwrap_or_default(),
        store: None,
        cleanup: None,
        source: None,
    };
    let default_buckets;
    let config = match &model.reorg_buckets {
        Some(config) => config,
        None => {
            default_buckets = BucketsConfig::default_buckets();
            &default_buckets
        }
    };
    ensure!(!config.buckets.is_empty(), "--bucket-config has no buckets");
    let mut ids = HashSet::new();
    let mut buckets = Vec::new();
    for spec in &config.buckets {
        let id = &spec.id;
        ensure!(
            !id.is_empty() && ids.insert(id),
            "bucket id {id:?} must be non-empty and unique"
        );
        ensure!(
            spec.min_input_tokens
                .zip(spec.max_input_tokens)
                .is_none_or(|(min, max)| min <= max)
                && spec.max_context_tokens != Some(0)
                && (spec.min_input_tokens)
                    .zip(spec.max_context_tokens)
                    .is_none_or(|(min, context)| min <= context)
                && spec.ttft_ms != Some(0)
                && spec
                    .tokens_per_second
                    .is_none_or(|t| t.is_finite() && t > 0.0),
            "bucket {id:?} needs min input tokens <= max input and context tokens, \
             and positive capacity and SLO estimates"
        );
        let bucket_groups = match (&spec.plain, &spec.prefill, &spec.decode) {
            (Some(plain), None, None) => {
                BucketGroups::Plain(groups.build(id, Stage::Plain, plain)?)
            }
            (None, Some(prefill), Some(decode)) => BucketGroups::Pd {
                prefill: groups.build(id, Stage::Prefill, prefill)?,
                decode: groups.build(id, Stage::Decode, decode)?,
            },
            _ => bail!("bucket {id:?} needs either plain or both prefill and decode"),
        };
        let mut bucket = Bucket::new(id.clone(), bucket_groups);
        bucket.rank = spec.rank;
        bucket.limits = TokenLimits {
            min: spec.min_input_tokens,
            max: spec.max_input_tokens,
        };
        bucket.max_context_tokens = spec.max_context_tokens;
        bucket.ttft_ms = spec.ttft_ms;
        bucket.tokens_per_second = spec.tokens_per_second;
        buckets.push(bucket);
    }
    let mut resolver = BucketResolver::new(buckets)?;
    resolver.ttft_slo = config.ttft_slo;
    resolver.tps_slo = config.tps_slo;
    Ok((resolver, groups.cleanup))
}

/// Builds group policies; session stores and cache sources are shared by all groups.
struct Groups<'a> {
    model: &'a ModelConfig,
    state: &'a KvEventIndex,
    external_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
    affinity: AffinityConfig,
    store: Option<Arc<AffinityStore>>,
    cleanup: Option<JanitorHandle>,
    source: Option<Arc<CacheSource>>,
}

impl Groups<'_> {
    fn build(&mut self, bucket: &str, stage: Stage, spec: &GroupSpec) -> Result<EngineGroup> {
        let default = match stage {
            Stage::Decode => PolicyKind::PowerOfTwo,
            _ => self.model.policy,
        };
        let kind = spec.policy.unwrap_or(default);
        // Only --policy has its affinity, cache and tokenizer settings resolved.
        ensure!(
            kind == PolicyKind::PowerOfTwo || kind == self.model.policy,
            "bucket groups use power_of_two or --policy, not {kind:?}"
        );
        ensure!(
            spec.worker_ids.as_ref().is_none_or(|ids| {
                !ids.is_empty() && ids.iter().all(|id| !id.0.trim().is_empty())
            }),
            "bucket {bucket:?} {stage:?} worker_ids and their values must not be empty; omit it for every engine"
        );
        ensure!(
            spec.worker_ids.is_none() || spec.worker_services.is_none(),
            "bucket {bucket:?} {stage:?} must use either worker_ids or worker_services"
        );
        ensure!(
            spec.worker_services.as_ref().is_none_or(|services| {
                !services.is_empty() && services.iter().all(|service| {
                    service.split_once('/').is_some_and(|(namespace, name)| {
                        !namespace.trim().is_empty() && !name.trim().is_empty()
                            && !name.contains('/') && !service.chars().any(char::is_whitespace)
                    })
                })
            }),
            "bucket {bucket:?} {stage:?} worker_services must be a nonempty list of namespace/service names"
        );
        let defaults = &self.model.reorg_admission;
        let admission = spec
            .admission
            .as_ref()
            .map_or(defaults.clone(), |a| a.or(defaults));
        admission.validate()?;
        let admission = Arc::new(admission);
        ensure!(
            spec.affinity.is_none() || kind != PolicyKind::PowerOfTwo,
            "bucket {bucket:?} {stage:?} sets affinity on a power_of_two group"
        );
        let affinity = match &spec.affinity {
            Some(spec) => spec.or(&self.affinity)?,
            None => self.affinity.clone(),
        };
        let load = self.state.engine_reported_load();
        let policy: Arc<dyn Policy> = match kind {
            PolicyKind::PowerOfTwo => {
                let mut policy = PowerOfTwoPolicy::new(load);
                policy.admission = admission;
                Arc::new(policy)
            }
            PolicyKind::SessionAware => {
                let mut policy = SessionAwarePolicy::new(self.store(), load);
                policy.admission = admission;
                policy.config = affinity;
                Arc::new(policy)
            }
            PolicyKind::CacheAware => {
                let mut policy = CacheAwarePolicy::new(self.source(), load, affinity)?;
                policy.admission = admission;
                Arc::new(policy)
            }
            other => bail!("reorg routing does not implement --policy {other:?}"),
        };
        Ok(EngineGroup {
            worker_ids: spec.worker_ids.clone(),
            worker_services: spec.worker_services.clone(),
            policy,
        })
    }

    fn store(&mut self) -> Arc<AffinityStore> {
        let affinity = &self.affinity;
        let cleanup = &mut self.cleanup;
        Arc::clone(self.store.get_or_insert_with(|| {
            let store = AffinityStore::new(Duration::from_secs(affinity.session_idle_secs));
            *cleanup =
                store.spawn_sweeper(Duration::from_secs(affinity.session_eviction_interval_secs));
            store
        }))
    }

    fn source(&mut self) -> Arc<CacheSource> {
        let (state, index) = (self.state, self.external_index.clone());
        Arc::clone(self.source.get_or_insert_with(|| {
            Arc::new(match index {
                Some(index) => CacheSource::Remote {
                    index,
                    block_size: state.block_size_oracle(),
                },
                None => CacheSource::Local(RadixTreePrefixProvider::new(
                    state.tree(),
                    state.block_size_oracle(),
                )),
            })
        }))
    }
}
