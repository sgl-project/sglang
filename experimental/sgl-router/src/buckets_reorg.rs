// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use crate::discovery::WorkerId;
use crate::policies_reorg::{Pick, PickError, PickMode, PickRequest, Policy, Stage};
use crate::workers::{Worker, WorkerRegistry};

/// Resolver input: the policy-facing request plus the SLO targets used for ordering.
#[derive(Debug, Clone, Copy)]
pub struct SelectionRequest<'a> {
    pub pick: PickRequest<'a>,
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
}

impl<'a> SelectionRequest<'a> {
    pub fn new(pick: PickRequest<'a>) -> Self {
        Self {
            pick,
            ttft_ms: None,
            tokens_per_second: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenLimits {
    pub min: Option<u64>,
    pub max: Option<u64>,
    pub context: Option<u64>,
}

impl TokenLimits {
    fn fits(&self, request: &PickRequest<'_>) -> bool {
        let tokens = match request.stage {
            Stage::Plain | Stage::Prefill => Some(request.input_tokens),
            Stage::Decode => request.expected_peak_tokens,
        };
        self.context
            .is_none_or(|max| tokens.unwrap_or(request.input_tokens) <= max)
            && match tokens {
                Some(tokens) => {
                    self.min.is_none_or(|min| tokens >= min)
                        && self.max.is_none_or(|max| tokens <= max)
                }
                None => self.min.is_none() && self.max.is_none(),
            }
    }
}

#[derive(Debug)]
pub struct Bucket {
    pub id: String,
    pub rank: u32,
    /// `None` means every engine in the pool.
    pub worker_ids: Option<HashSet<WorkerId>>,
    /// Input range for plain/prefill; peak sequence range for decode.
    pub limits: TokenLimits,
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub policy: Arc<dyn Policy>,
}

impl Bucket {
    fn contains(&self, engine: &Worker) -> bool {
        self.worker_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&engine.id))
    }

    fn matches_slo(&self, request: &SelectionRequest<'_>) -> bool {
        match request.pick.stage {
            Stage::Plain | Stage::Prefill => request
                .ttft_ms
                .is_none_or(|target| self.ttft_ms.is_some_and(|estimate| estimate <= target)),
            Stage::Decode => request.tokens_per_second.is_none_or(|target| {
                self.tokens_per_second
                    .is_some_and(|estimate| estimate >= target)
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SloPreference {
    #[default]
    Disabled,
    SloFirst,
    BestEffort,
}

#[derive(Debug, Default)]
pub struct Pool {
    pub buckets: Vec<Bucket>,
    pub slo: SloPreference,
    /// Probes the whole pool for an existing binding before the size buckets.
    pub affinity: Option<Arc<dyn Policy>>,
    /// Global-preserve: a stranded binding stops later buckets from binding.
    pub preserve_global_binding: bool,
}

impl Pool {
    /// One catch-all bucket for a pool without configured buckets.
    pub fn implicit(policy: Arc<dyn Policy>) -> Self {
        Self {
            buckets: vec![Bucket {
                id: "default".into(),
                rank: 0,
                worker_ids: None,
                limits: TokenLimits::default(),
                ttft_ms: None,
                tokens_per_second: None,
                policy,
            }],
            ..Default::default()
        }
    }
}

/// The prefill pool also serves plain requests; a plain deployment never asks for decode.
#[derive(Debug)]
pub struct Pools {
    pub prefill: Pool,
    pub decode: Pool,
}

impl Pools {
    pub fn plain(pool: Pool) -> Self {
        Self {
            prefill: pool,
            decode: Pool::default(),
        }
    }

    fn for_stage(&self, stage: Stage) -> &Pool {
        match stage {
            Stage::Plain | Stage::Prefill => &self.prefill,
            Stage::Decode => &self.decode,
        }
    }
}

#[derive(Debug)]
pub struct ResolvedGroup<'a> {
    pub bucket: &'a str,
    pub mode: PickMode,
    pub engines: Vec<Arc<Worker>>,
    pub policy: &'a dyn Policy,
}

pub struct BucketResolver {
    pub workers: Arc<WorkerRegistry>,
    pub pools: Pools,
    pub fallback_on_rejection: bool,
}

impl BucketResolver {
    pub fn new(workers: Arc<WorkerRegistry>, pools: Pools) -> Self {
        Self {
            workers,
            pools,
            fallback_on_rejection: true,
        }
    }

    pub fn ordered_groups(
        &self,
        request: &SelectionRequest<'_>,
    ) -> Result<Vec<ResolvedGroup<'_>>, PickError> {
        let pick = &request.pick;
        if pick
            .expected_peak_tokens
            .is_some_and(|tokens| tokens < pick.input_tokens)
            || request
                .tokens_per_second
                .is_some_and(|tps| !tps.is_finite() || tps <= 0.0)
        {
            return Err(PickError::InvalidSignal(
                "invalid request size or SLO".into(),
            ));
        }
        let pool = self.pools.for_stage(pick.stage);
        let mut engines: Vec<_> = self
            .workers
            .healthy_workers_for(pick.model)
            .into_iter()
            .filter(|engine| engine.mode() == pick.stage)
            .collect();
        // Stable order so cursor-based policies see a consistent candidate list.
        engines.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        let mut groups = Vec::new();
        if let Some(policy) = &pool.affinity {
            // Size ranges do not exclude a binding holder; its own context limit and SLO rule do.
            let members: Vec<_> = engines
                .iter()
                .filter(|engine| {
                    pool.buckets
                        .iter()
                        .find(|bucket| bucket.contains(engine))
                        .is_none_or(|bucket| {
                            bucket
                                .limits
                                .context
                                .is_none_or(|max| pick.input_tokens <= max)
                                && (pool.slo != SloPreference::SloFirst
                                    || bucket.matches_slo(request))
                        })
                })
                .cloned()
                .collect();
            if !members.is_empty() {
                groups.push(ResolvedGroup {
                    bucket: "affinity",
                    mode: PickMode::HitRequired,
                    engines: members,
                    policy: policy.as_ref(),
                });
            }
        }
        let mut buckets: Vec<_> = pool
            .buckets
            .iter()
            .filter(|bucket| bucket.limits.fits(pick))
            .collect();
        buckets.sort_by_key(|bucket| {
            let demoted = match pool.slo {
                SloPreference::Disabled => false,
                SloPreference::SloFirst => !bucket.matches_slo(request),
                SloPreference::BestEffort => bucket.matches_slo(request),
            };
            (demoted, bucket.rank, &bucket.id)
        });
        groups.extend(buckets.into_iter().filter_map(|bucket| {
            let members: Vec<_> = engines
                .iter()
                .filter(|engine| bucket.contains(engine))
                .cloned()
                .collect();
            (!members.is_empty()).then_some(ResolvedGroup {
                bucket: &bucket.id,
                mode: PickMode::Normal,
                engines: members,
                policy: bucket.policy.as_ref(),
            })
        }));
        Ok(groups)
    }

    pub async fn pick(&self, request: &SelectionRequest<'_>) -> Result<Pick, PickError> {
        let pool = self.pools.for_stage(request.pick.stage);
        let mut affinity_enabled = true;
        let mut rejections = Vec::new();
        for group in self.ordered_groups(request)? {
            let scoped = PickRequest {
                bucket: group.bucket,
                mode: group.mode,
                affinity_enabled,
                ..request.pick
            };
            match group.policy.pick(&group.engines, &scoped).await {
                Ok(pick) => {
                    if !group
                        .engines
                        .iter()
                        .any(|engine| Arc::ptr_eq(engine, &pick.engine))
                    {
                        return Err(PickError::OutsideCandidates(pick.engine.id.clone()));
                    }
                    return Ok(pick);
                }
                Err(PickError::NoCandidates) => continue,
                Err(PickError::NoAdmissibleEngine(reasons)) => rejections.extend(reasons),
                Err(PickError::AdmissionRejected(reason)) => rejections.push(reason),
                Err(error) => return Err(error),
            }
            // Only an admission rejection reaches here.
            if !self.fallback_on_rejection {
                return Err(PickError::NoAdmissibleEngine(rejections));
            }
            if group.mode == PickMode::HitRequired && pool.preserve_global_binding {
                affinity_enabled = false;
            }
        }
        if rejections.is_empty() {
            Err(PickError::NoCandidates)
        } else {
            Err(PickError::NoAdmissibleEngine(rejections))
        }
    }
}
