// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::{bail, ensure, Result};

use crate::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
use crate::config::{DecodePolicyKind, FilterKind, ModelConfig, PolicyKind, SessionAffinityMode};
use crate::policies::prefix_provider::RadixTreePrefixProvider;
use crate::state::{
    kv_events::KvEventIndex, load_monitor::router_inflight_load::JanitorHandle, AffinityStore,
};

use super::{
    admission::AdmissionLimits,
    cache_aware::{CacheAwarePolicy, CacheSource},
    power_of_two::PowerOfTwoPolicy,
    session_aware::SessionAwarePolicy,
    Policy,
};

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
        "legacy --bucket-config cannot define complete reorg buckets"
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

/// Build the default plain and PD buckets. Callers run [`validate`] first;
/// `Cli::into_config` does so before startup reaches this point.
pub fn build_resolver(
    model: &ModelConfig,
    state: &KvEventIndex,
    external_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
) -> Result<(BucketResolver, Option<JanitorHandle>)> {
    let admission = Arc::new(AdmissionLimits {
        max_inflight_requests: model
            .eligibility
            .as_ref()
            .and_then(|e| e.max_in_flight)
            .map(|n| n as u64),
        ..Default::default()
    });
    let mut decode = PowerOfTwoPolicy::new(state.engine_reported_load());
    decode.admission = admission.clone();
    let decode: Arc<dyn Policy> = Arc::new(decode);
    let affinity = model.affinity.clone().unwrap_or_default();
    let mut cleanup = None;
    let policy: Arc<dyn Policy> = match model.policy {
        PolicyKind::PowerOfTwo => decode.clone(),
        PolicyKind::SessionAware => {
            let store = AffinityStore::new(Duration::from_secs(affinity.session_idle_secs));
            cleanup =
                store.spawn_sweeper(Duration::from_secs(affinity.session_eviction_interval_secs));
            let mut policy = SessionAwarePolicy::new(store, state.engine_reported_load());
            policy.admission = admission;
            Arc::new(policy)
        }
        PolicyKind::CacheAware => {
            let source = match external_index {
                Some(index) => CacheSource::Remote {
                    index,
                    block_size: state.block_size_oracle(),
                },
                None => CacheSource::Local(RadixTreePrefixProvider::new(
                    state.tree(),
                    state.block_size_oracle(),
                )),
            };
            let mut policy =
                CacheAwarePolicy::new(Arc::new(source), state.engine_reported_load(), affinity)?;
            policy.admission = admission;
            Arc::new(policy)
        }
        other => bail!("reorg routing does not implement --policy {other:?}"),
    };
    // Discovery determines which serving mode has candidates. Rank the plain
    // bucket first so plain deployments do not scan for prefill engines on
    // every request; buckets otherwise tie and would sort by ID ("pd" first).
    let mut pd = Bucket::new(
        "pd",
        BucketGroups::Pd {
            prefill: EngineGroup::new(policy.clone()),
            decode: EngineGroup::new(decode),
        },
    );
    pd.rank = 1;
    let resolver = BucketResolver::new(vec![
        Bucket::new("plain", BucketGroups::Plain(EngineGroup::new(policy))),
        pd,
    ])?;
    Ok((resolver, cleanup))
}
