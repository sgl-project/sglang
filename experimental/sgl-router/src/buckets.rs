// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Orders the buckets a request may use and runs each bucket's attached
//! policy until one picks an engine. Cache and session preference live in the
//! policies; this layer only owns membership, ordering and fallback.

use crate::config::{
    BucketConfig, BucketSpec, BucketStage, ModelConfig, PolicyKind, SessionAffinityMode,
    SloBucketPolicy,
};
use crate::discovery::{ModelId, WorkerId};
use crate::policies::cache_aware::PrefixMemo;
use crate::policies::pools::{PdPoolResolver, PdResolveError};
use crate::policies::state::engine_load::{EngineLoadTable, LoadView};
use crate::policies::{
    build_decode_policy, build_policy, AffinityScope, BuildError, Pick, PickError, PickMode,
    PickRequest, Policy, PolicyDependencies, RoutingStage,
};
use crate::server::metrics::MetricsRegistry;
use crate::workers::Worker;
use std::collections::HashSet;
use std::sync::Arc;

/// Request facts shared by every stage of one HTTP request.
pub struct SelectionRequest<'a> {
    pub model: &'a ModelId,
    pub input_tokens: u64,
    pub max_output_tokens: Option<u64>,
    pub ttft_slo_ms: Option<u64>,
    pub tps_slo: Option<f64>,
    pub session_id: Option<&'a str>,
    pub routing_key: Option<&'a str>,
    pub tokens: Option<&'a [u32]>,
    pub prefix: &'a PrefixMemo,
}

#[derive(Debug)]
pub enum SelectError {
    Pool(PdResolveError),
    /// Every bucket was tried; the last admission failure, or `NoCandidates`.
    Exhausted(PickError),
    InvalidSignal(String),
    /// A policy returned an engine it was not given.
    OutOfSet(WorkerId),
}

/// A configured bucket with its attached policy.
struct Bucket {
    spec: BucketSpec,
    members: HashSet<String>,
    kind: PolicyKind,
    policy: Arc<dyn Policy>,
}

/// One entry of the ordered pass: a candidate set and the policy that picks from it.
pub struct ResolvedEngineGroup<'r> {
    pub bucket_id: &'r str,
    pub engines: Vec<Arc<Worker>>,
    pub mode: PickMode,
    kind: PolicyKind,
    policy: &'r dyn Policy,
}

pub struct BucketResolver {
    pools: PdPoolResolver,
    load: Arc<EngineLoadTable>,
    metrics: Arc<MetricsRegistry>,
    config: Option<BucketConfig>,
    buckets: Vec<Bucket>,
    kind: PolicyKind,
    model_policy: Arc<dyn Policy>,
    decode_policy: Arc<dyn Policy>,
    scope: AffinityScope,
    session_mode: SessionAffinityMode,
}

impl BucketResolver {
    pub fn from_config(
        model: &ModelConfig,
        pools: PdPoolResolver,
        load: Arc<EngineLoadTable>,
        deps: &PolicyDependencies,
    ) -> Result<Self, BuildError> {
        let buckets = model
            .bucket_config
            .iter()
            .flat_map(|config| &config.buckets)
            .map(|spec| {
                let kind = spec.policy.unwrap_or(model.policy);
                let policy = match spec.stage {
                    BucketStage::Decode if spec.policy.is_none() => {
                        build_decode_policy(model.decode_policy)
                    }
                    _ => build_policy(kind, model, deps)?,
                };
                Ok(Bucket {
                    members: spec.worker_ids.iter().cloned().collect(),
                    spec: spec.clone(),
                    kind,
                    policy,
                })
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let session_mode = model
            .affinity
            .as_ref()
            .map_or(SessionAffinityMode::Bucket, |affinity| {
                affinity.session_affinity_mode
            });
        let global_affinity = model.policy == PolicyKind::Sticky
            || (model.policy == PolicyKind::SessionAware
                && session_mode != SessionAffinityMode::Bucket);
        Ok(Self {
            pools,
            load,
            metrics: Arc::clone(&deps.metrics),
            config: model.bucket_config.clone(),
            buckets,
            kind: model.policy,
            model_policy: build_policy(model.policy, model, deps)?,
            decode_policy: build_decode_policy(model.decode_policy),
            scope: if global_affinity {
                AffinityScope::Global
            } else {
                AffinityScope::Bucket
            },
            session_mode,
        })
    }

    pub fn is_bucketed(&self) -> bool {
        self.config.is_some()
    }

    pub fn needs_request_tokens(&self) -> bool {
        self.kinds().any(|kind| kind == PolicyKind::CacheAware)
    }

    pub fn needs_dispatch_timestamps(&self) -> bool {
        self.kinds().any(|kind| kind == PolicyKind::LoadBased)
    }

    fn kinds(&self) -> impl Iterator<Item = PolicyKind> + '_ {
        std::iter::once(self.kind).chain(self.buckets.iter().map(|bucket| bucket.kind))
    }

    /// Resolves the stage's healthy pool and picks one engine from it.
    pub async fn pick(
        &self,
        stage: RoutingStage,
        request: &SelectionRequest<'_>,
    ) -> Result<Pick, SelectError> {
        let pool = match stage {
            RoutingStage::Decode => self.pools.decode_candidates(request.model),
            RoutingStage::Plain | RoutingStage::Prefill => {
                self.pools.prefill_candidates(request.model)
            }
        }
        .map_err(SelectError::Pool)?;
        self.pick_from(stage, &pool, request).await
    }

    /// The ordered pass over `ordered_groups`. Admission failures advance to
    /// the next group; under global-preserve they also stop later groups from
    /// looking up or creating session bindings.
    pub async fn pick_from(
        &self,
        stage: RoutingStage,
        pool: &[Arc<Worker>],
        request: &SelectionRequest<'_>,
    ) -> Result<Pick, SelectError> {
        let load = LoadView::new(&self.load);
        let mut affinity_enabled = true;
        let mut last = PickError::NoCandidates;
        for group in self.ordered_groups(stage, pool, request) {
            let pick_request = PickRequest {
                model: request.model,
                stage,
                bucket_id: group.bucket_id,
                scope: self.scope,
                mode: group.mode,
                input_tokens: request.input_tokens,
                expected_peak_sequence_tokens: (stage == RoutingStage::Decode)
                    .then(|| {
                        request
                            .max_output_tokens
                            .map(|out| request.input_tokens.saturating_add(out))
                    })
                    .flatten(),
                session_id: request.session_id,
                routing_key: request.routing_key,
                tokens: request.tokens,
                prefix: Some(request.prefix),
                affinity_enabled,
                load: &load,
            };
            match group.policy.pick(&group.engines, &pick_request).await {
                Ok(pick) => {
                    if !group
                        .engines
                        .iter()
                        .any(|engine| engine.id == pick.engine.id)
                    {
                        return Err(SelectError::OutOfSet(pick.engine.id.clone()));
                    }
                    if stage != RoutingStage::Decode {
                        self.metrics
                            .record_policy_decision(&group.kind.to_string(), pick.reason);
                    }
                    return Ok(pick);
                }
                Err(PickError::NoCandidates) => {}
                Err(PickError::InvalidSignal(reason)) => {
                    return Err(SelectError::InvalidSignal(reason))
                }
                Err(rejected) => {
                    if group.mode == PickMode::HitRequired
                        && self.session_mode == SessionAffinityMode::GlobalPreserve
                    {
                        affinity_enabled = false;
                    }
                    last = rejected;
                }
            }
        }
        Err(SelectError::Exhausted(last))
    }

    /// The affinity group (whole pool, per-engine bucket rules, hit required)
    /// when the stage policy keeps affinity across buckets, then the compatible
    /// size buckets by SLO preference, rank and id. Without bucket
    /// configuration, or without decode buckets for decode, one implicit group.
    pub fn ordered_groups<'r>(
        &'r self,
        stage: RoutingStage,
        pool: &[Arc<Worker>],
        request: &SelectionRequest<'_>,
    ) -> Vec<ResolvedEngineGroup<'r>> {
        let (bucket_stage, default_policy) = match stage {
            RoutingStage::Decode => (BucketStage::Decode, &self.decode_policy),
            RoutingStage::Plain | RoutingStage::Prefill => {
                (BucketStage::Prefill, &self.model_policy)
            }
        };
        let implicit = |engines: Vec<Arc<Worker>>| {
            vec![ResolvedEngineGroup {
                bucket_id: "global",
                engines,
                mode: PickMode::Normal,
                kind: self.kind,
                policy: default_policy.as_ref(),
            }]
        };
        let Some(config) = &self.config else {
            return implicit(pool.to_vec());
        };
        let staged: Vec<&Bucket> = self
            .buckets
            .iter()
            .filter(|bucket| bucket.spec.stage == bucket_stage)
            .collect();
        if staged.is_empty() {
            return implicit(pool.to_vec());
        }
        let peak = request
            .max_output_tokens
            .map(|out| request.input_tokens.saturating_add(out));
        let slo_policy = match bucket_stage {
            BucketStage::Prefill => config.ttft_slo_policy,
            BucketStage::Decode => config.tps_slo_policy,
        };
        let compatible = |spec: &BucketSpec| match bucket_stage {
            BucketStage::Prefill => prefill_compatible(spec, request.input_tokens),
            BucketStage::Decode => decode_compatible(spec, request.input_tokens, peak),
        };
        let eligible = |spec: &BucketSpec| match bucket_stage {
            BucketStage::Prefill => ttft_eligible(spec, request.ttft_slo_ms),
            BucketStage::Decode => tps_eligible(spec, request.tps_slo),
        };
        let mut groups = Vec::new();
        if stage != RoutingStage::Decode && self.affinity_first() {
            let engines: Vec<Arc<Worker>> = pool
                .iter()
                .filter(|engine| {
                    staged
                        .iter()
                        .find(|bucket| bucket.members.contains(&engine.id.0))
                        .is_none_or(|bucket| {
                            bucket
                                .spec
                                .max_context_tokens
                                .is_none_or(|max| request.input_tokens <= max)
                                && (slo_policy != SloBucketPolicy::SloFirst
                                    || eligible(&bucket.spec))
                        })
                })
                .cloned()
                .collect();
            groups.push(ResolvedEngineGroup {
                bucket_id: "affinity",
                engines,
                mode: PickMode::HitRequired,
                kind: self.kind,
                policy: self.model_policy.as_ref(),
            });
        }
        let mut ordered: Vec<&Bucket> = staged
            .into_iter()
            .filter(|bucket| compatible(&bucket.spec))
            .collect();
        ordered.sort_by(|left, right| {
            (left.spec.rank, &left.spec.id).cmp(&(right.spec.rank, &right.spec.id))
        });
        match slo_policy {
            SloBucketPolicy::Disabled => {}
            // SLO-first serves matching buckets first; best effort keeps them in reserve.
            SloBucketPolicy::SloFirst => ordered.sort_by_key(|bucket| !eligible(&bucket.spec)),
            SloBucketPolicy::BestEffort => ordered.sort_by_key(|bucket| eligible(&bucket.spec)),
        }
        groups.extend(ordered.into_iter().filter_map(|bucket| {
            let engines: Vec<Arc<Worker>> = pool
                .iter()
                .filter(|engine| bucket.members.contains(&engine.id.0))
                .cloned()
                .collect();
            (!engines.is_empty()).then(|| ResolvedEngineGroup {
                bucket_id: &bucket.spec.id,
                engines,
                mode: PickMode::Normal,
                kind: bucket.kind,
                policy: bucket.policy.as_ref(),
            })
        }));
        groups
    }

    fn affinity_first(&self) -> bool {
        self.kind == PolicyKind::CacheAware || self.scope == AffinityScope::Global
    }
}

fn prefill_compatible(spec: &BucketSpec, input_tokens: u64) -> bool {
    within(input_tokens, spec.min_extend_tokens, spec.max_extend_tokens)
        && spec
            .max_context_tokens
            .is_none_or(|max| input_tokens <= max)
}

fn decode_compatible(spec: &BucketSpec, input_tokens: u64, peak: Option<u64>) -> bool {
    let Some(peak) = peak else {
        // Unknown output length can only use a catch-all decode bucket.
        return spec.min_sequence_tokens.is_none()
            && spec.max_sequence_tokens.is_none()
            && spec
                .max_context_tokens
                .is_none_or(|max| input_tokens <= max);
    };
    within(peak, spec.min_sequence_tokens, spec.max_sequence_tokens)
        && spec.max_context_tokens.is_none_or(|max| peak <= max)
}

fn within(value: u64, min: Option<u64>, max: Option<u64>) -> bool {
    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
}

fn ttft_eligible(spec: &BucketSpec, slo_ms: Option<u64>) -> bool {
    slo_ms.is_none_or(|slo| spec.ttft_p95_at_capacity_ms.is_some_and(|p95| p95 <= slo))
}

fn tps_eligible(spec: &BucketSpec, slo: Option<f64>) -> bool {
    slo.is_none_or(|slo| spec.tps_p05_at_capacity.is_some_and(|p05| p05 >= slo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AffinityConfig, SamplingOverrides};
    use crate::policies::state::kv_events::{BlockSizeOracle, KvEventIndex};
    use crate::policies::state::AffinityStore;
    use crate::policies::testing::worker;
    use crate::policies::{ready, AdmissionReason, AffinityScope, EngineRejection, PickResult};
    use futures::future::BoxFuture;
    use std::time::Duration;

    fn bucket(id: &str, stage: BucketStage, rank: u32, worker_ids: &[&str]) -> BucketSpec {
        BucketSpec {
            id: id.into(),
            stage,
            rank,
            worker_ids: worker_ids.iter().map(|id| (*id).into()).collect(),
            min_extend_tokens: None,
            max_extend_tokens: None,
            min_sequence_tokens: None,
            max_sequence_tokens: None,
            max_context_tokens: None,
            ttft_p95_at_capacity_ms: None,
            tps_p05_at_capacity: None,
            max_pending_prefill_tokens: None,
            policy: None,
        }
    }

    fn resolver(
        policy: PolicyKind,
        buckets: Vec<BucketSpec>,
        slo: SloBucketPolicy,
    ) -> BucketResolver {
        let model = ModelConfig {
            id: "m".into(),
            tokenizer_path: "m".into(),
            policy,
            decode_policy: Default::default(),
            bucket_config: Some(BucketConfig {
                buckets,
                ttft_slo_policy: slo,
                tps_slo_policy: SloBucketPolicy::Disabled,
            }),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: Some(AffinityConfig {
                session_affinity_mode: SessionAffinityMode::GlobalPreserve,
                ..AffinityConfig::default()
            }),
            eligibility: None,
            sampling_overrides: SamplingOverrides::default(),
        };
        let deps = PolicyDependencies {
            metrics: MetricsRegistry::new(),
            affinity: AffinityStore::new(Duration::from_secs(60)),
            local_cache: Some(KvEventIndex::new()),
            remote_cache: None,
            block_size: BlockSizeOracle::new(),
        };
        BucketResolver::from_config(
            &model,
            PdPoolResolver::new(Arc::default()),
            EngineLoadTable::new(),
            &deps,
        )
        .unwrap()
    }

    fn request<'a>(
        model: &'a ModelId,
        prefix: &'a PrefixMemo,
        ttft_slo_ms: Option<u64>,
    ) -> SelectionRequest<'a> {
        SelectionRequest {
            model,
            input_tokens: 256,
            max_output_tokens: None,
            ttft_slo_ms,
            tps_slo: None,
            session_id: Some("s"),
            routing_key: None,
            tokens: None,
            prefix,
        }
    }

    fn ids<'a>(groups: &'a [ResolvedEngineGroup<'a>]) -> Vec<&'a str> {
        groups.iter().map(|group| group.bucket_id).collect()
    }

    #[tokio::test]
    async fn slo_first_serves_eligible_buckets_before_degrading_and_best_effort_reverses() {
        let mut cheap = bucket("cheap", BucketStage::Prefill, 10, &["cheap"]);
        cheap.ttft_p95_at_capacity_ms = Some(300);
        let mut fast = bucket("fast", BucketStage::Prefill, 20, &["fast"]);
        fast.ttft_p95_at_capacity_ms = Some(80);
        let pool = [worker("cheap"), worker("fast")];
        let (model, memo) = (ModelId("m".into()), PrefixMemo::new());
        let req = request(&model, &memo, Some(100));

        let slo_first = resolver(
            PolicyKind::PowerOfTwo,
            vec![cheap.clone(), fast.clone()],
            SloBucketPolicy::SloFirst,
        );
        assert_eq!(
            ids(&slo_first.ordered_groups(RoutingStage::Prefill, &pool, &req)),
            ["fast", "cheap"]
        );
        let best_effort = resolver(
            PolicyKind::PowerOfTwo,
            vec![cheap, fast],
            SloBucketPolicy::BestEffort,
        );
        assert_eq!(
            ids(&best_effort.ordered_groups(RoutingStage::Prefill, &pool, &req)),
            ["cheap", "fast"]
        );
    }

    #[tokio::test]
    async fn the_affinity_group_ignores_extend_ranges_but_keeps_context_and_slo_rules() {
        let mut short = bucket("p-short", BucketStage::Prefill, 10, &["short"]);
        short.max_extend_tokens = Some(64);
        short.max_context_tokens = Some(128);
        let mut long = bucket("p-long", BucketStage::Prefill, 20, &["long"]);
        long.min_extend_tokens = Some(65);
        let pool = [worker("short"), worker("long"), worker("unbucketed")];
        let (model, memo) = (ModelId("m".into()), PrefixMemo::new());
        let resolver = resolver(
            PolicyKind::SessionAware,
            vec![short, long],
            SloBucketPolicy::Disabled,
        );

        let groups =
            resolver.ordered_groups(RoutingStage::Prefill, &pool, &request(&model, &memo, None));
        assert_eq!(ids(&groups), ["affinity", "p-long"]);
        assert!(groups[0].mode == PickMode::HitRequired && resolver.scope == AffinityScope::Global);
        let affinity: Vec<&str> = groups[0].engines.iter().map(|e| e.id.0.as_str()).collect();
        // `long` is in range; `short` fails its own context cap, not its extend range.
        assert_eq!(affinity, ["long", "unbucketed"]);
    }

    #[derive(Debug)]
    struct Scripted(Vec<PickResult>, std::sync::Mutex<usize>);

    impl Policy for Scripted {
        fn pick<'a>(
            &'a self,
            _: &'a [Arc<Worker>],
            _: &'a PickRequest<'a>,
        ) -> BoxFuture<'a, PickResult> {
            let mut i = self.1.lock().unwrap();
            let result = self.0[*i].clone();
            *i += 1;
            ready(result)
        }
    }

    #[tokio::test]
    async fn admission_failures_advance_and_an_out_of_set_pick_is_an_error() {
        let pool = [worker("a"), worker("b")];
        let (model, memo) = (ModelId("m".into()), PrefixMemo::new());
        let mut resolver = resolver(
            PolicyKind::PowerOfTwo,
            vec![
                bucket("one", BucketStage::Prefill, 10, &["a"]),
                bucket("two", BucketStage::Prefill, 20, &["b"]),
            ],
            SloBucketPolicy::Disabled,
        );
        let rejected = EngineRejection {
            engine: WorkerId("a".into()),
            reason: AdmissionReason::KvCapacity,
        };
        resolver.buckets[0].policy = Arc::new(Scripted(
            vec![Err(PickError::NoAdmissibleEngine(vec![rejected.clone()]))],
            Default::default(),
        ));
        resolver.buckets[1].policy = Arc::new(Scripted(
            vec![
                Ok(Pick {
                    engine: worker("stranger"),
                    reason: "primary",
                }),
                Err(PickError::NoCandidates),
            ],
            Default::default(),
        ));

        let req = request(&model, &memo, None);
        assert!(
            matches!(resolver.pick_from(RoutingStage::Prefill, &pool, &req).await, Err(SelectError::OutOfSet(id)) if id.0 == "stranger")
        );
        resolver.buckets[0].policy = Arc::new(Scripted(
            vec![Err(PickError::NoAdmissibleEngine(vec![rejected]))],
            Default::default(),
        ));
        assert!(matches!(
            resolver.pick_from(RoutingStage::Prefill, &pool, &req).await,
            Err(SelectError::Exhausted(PickError::NoAdmissibleEngine(_)))
        ));
    }
}
