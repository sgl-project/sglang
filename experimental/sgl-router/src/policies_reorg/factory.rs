// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;

use crate::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup, TokenLimits};
use crate::config::{
    reorg::{GroupConfig, GroupsConfig, PolicyKind, ReorgConfig},
    AffinityConfig,
};
use crate::discovery::WorkerId;
use crate::policies::prefix_provider::RadixTreePrefixProvider;
use crate::state::load_monitor::router_inflight_load::JanitorHandle;
use crate::state::{kv_events::KvEventIndex, AffinityStore};

use super::{
    cache_aware::{CacheAwarePolicy, CacheSource},
    power_of_two::PowerOfTwoPolicy,
    session_aware::SessionAwarePolicy,
    Policy,
};

/// Construct policies once, sharing the monitor's live load table and prefix source.
/// The caller must retain the sweeper until requests have drained.
pub fn build_resolver(
    config: &ReorgConfig,
    state: &KvEventIndex,
    external_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
) -> Result<(BucketResolver, Option<JanitorHandle>)> {
    config.validate()?;
    anyhow::ensure!(
        config.kv_indexer.is_some() == external_index.is_some(),
        "reorg KV indexer configuration and client must agree"
    );
    let source = Arc::new(match external_index {
        Some(index) => CacheSource::Remote {
            index,
            block_size: state.block_size_oracle(),
        },
        None => CacheSource::Local(RadixTreePrefixProvider::new(
            state.tree(),
            state.block_size_oracle(),
        )),
    });
    let store = AffinityStore::new(Duration::from_secs(config.session.idle_secs));
    let group = |spec: &GroupConfig| -> Result<EngineGroup> {
        let admission = Arc::new(spec.admission.clone());
        let policy: Arc<dyn Policy> = match spec.policy {
            PolicyKind::PowerOfTwo => {
                let mut policy = PowerOfTwoPolicy::new(state.engine_reported_load());
                policy.admission = admission;
                Arc::new(policy)
            }
            PolicyKind::SessionAware => {
                let mut policy =
                    SessionAwarePolicy::new(Arc::clone(&store), state.engine_reported_load());
                policy.admission = admission;
                Arc::new(policy)
            }
            PolicyKind::CacheAware => {
                let mut policy = CacheAwarePolicy::new(
                    Arc::clone(&source),
                    state.engine_reported_load(),
                    AffinityConfig::default(),
                )?;
                policy.admission = admission;
                Arc::new(policy)
            }
        };
        Ok(EngineGroup {
            worker_ids: spec
                .worker_ids
                .as_ref()
                .map(|ids| ids.iter().cloned().map(WorkerId).collect()),
            policy,
        })
    };
    let buckets = config
        .buckets
        .iter()
        .map(|spec| {
            Ok(Bucket {
                id: spec.id.clone(),
                rank: spec.rank,
                limits: TokenLimits {
                    min: spec.min_input_tokens,
                    max: spec.max_input_tokens,
                },
                max_context_tokens: spec.max_context_tokens,
                ttft_ms: spec.ttft_ms,
                tokens_per_second: spec.tokens_per_second,
                groups: match &spec.groups {
                    GroupsConfig::Plain { plain } => BucketGroups::Plain(group(plain)?),
                    GroupsConfig::Pd { prefill, decode } => BucketGroups::Pd {
                        prefill: group(prefill)?,
                        decode: group(decode)?,
                    },
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut resolver = BucketResolver::new(buckets)?;
    resolver.ttft_slo = config.ttft_slo;
    resolver.tps_slo = config.tps_slo;
    let sweeper = config
        .uses(PolicyKind::SessionAware)
        .then(|| store.spawn_sweeper(Duration::from_secs(config.session.eviction_interval_secs)))
        .flatten();
    Ok((resolver, sweeper))
}
