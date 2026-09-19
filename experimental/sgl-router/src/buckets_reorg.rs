// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Buckets define service constraints; their role groups own membership and policy.
//! Each stage selects independently and may choose a different bucket.

use std::collections::HashSet;
use std::sync::Arc;

use crate::discovery::WorkerId;
use crate::policies_reorg::{Pick, PickError, PickRequest, Policy, Stage};
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
}

impl TokenLimits {
    fn fits(&self, request: &PickRequest<'_>) -> bool {
        let tokens = match request.stage {
            Stage::Plain | Stage::Prefill => Some(request.input_tokens),
            Stage::Decode => request.expected_peak_tokens,
        };
        match tokens {
            Some(tokens) => {
                self.min.is_none_or(|min| tokens >= min) && self.max.is_none_or(|max| tokens <= max)
            }
            None => self.min.is_none() && self.max.is_none(),
        }
    }
}

#[derive(Debug)]
pub struct EngineGroup {
    /// Lower ranks are tried first within this role's SLO preference tier.
    pub rank: u32,
    /// `None` includes all registered engines matching the request's model and role.
    pub worker_ids: Option<HashSet<WorkerId>>,
    /// Input range for plain/prefill; peak sequence range for decode.
    pub limits: TokenLimits,
    pub policy: Arc<dyn Policy>,
}

impl EngineGroup {
    /// A catch-all group with the supplied policy.
    pub fn new(policy: Arc<dyn Policy>) -> Self {
        Self {
            rank: 0,
            worker_ids: None,
            limits: TokenLimits::default(),
            policy,
        }
    }

    fn contains(&self, engine: &Worker) -> bool {
        self.worker_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&engine.id))
    }
}

#[derive(Debug, Default)]
pub struct Bucket {
    pub id: String,
    /// Shared capacity: input length for plain/prefill, peak sequence for decode
    /// (or input length when the output budget is unknown).
    pub max_context_tokens: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub plain: Option<EngineGroup>,
    pub prefill: Option<EngineGroup>,
    pub decode: Option<EngineGroup>,
}

impl Bucket {
    fn group(&self, stage: Stage) -> Option<&EngineGroup> {
        match stage {
            Stage::Plain => self.plain.as_ref(),
            Stage::Prefill => self.prefill.as_ref(),
            Stage::Decode => self.decode.as_ref(),
        }
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

#[derive(Debug)]
pub struct ResolvedGroup<'a> {
    pub bucket: &'a str,
    pub engines: Vec<Arc<Worker>>,
    pub policy: &'a dyn Policy,
}

pub struct BucketResolver {
    pub workers: Arc<WorkerRegistry>,
    pub buckets: Vec<Bucket>,
    /// TTFT ordering for plain and prefill selection.
    pub prefill_slo: SloPreference,
    pub decode_slo: SloPreference,
    pub fallback_on_rejection: bool,
}

impl BucketResolver {
    pub fn new(workers: Arc<WorkerRegistry>, buckets: Vec<Bucket>) -> Self {
        Self {
            workers,
            buckets,
            prefill_slo: SloPreference::Disabled,
            decode_slo: SloPreference::Disabled,
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
        let (context_tokens, slo) = match pick.stage {
            Stage::Plain | Stage::Prefill => (pick.input_tokens, self.prefill_slo),
            Stage::Decode => (
                pick.expected_peak_tokens.unwrap_or(pick.input_tokens),
                self.decode_slo,
            ),
        };
        let mut engines: Vec<_> = self
            .workers
            .healthy_workers_for(pick.model)
            .into_iter()
            .filter(|engine| engine.mode() == pick.stage)
            .collect();
        // Stable order so cursor-based policies see a consistent candidate list.
        engines.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        let mut groups: Vec<_> = self
            .buckets
            .iter()
            .filter(|bucket| {
                bucket
                    .max_context_tokens
                    .is_none_or(|max| context_tokens <= max)
            })
            .filter_map(|bucket| bucket.group(pick.stage).map(|group| (bucket, group)))
            .filter(|(_, group)| group.limits.fits(pick))
            .collect();
        groups.sort_by_key(|(bucket, group)| {
            let demoted = match slo {
                SloPreference::Disabled => false,
                SloPreference::SloFirst => !bucket.matches_slo(request),
                SloPreference::BestEffort => bucket.matches_slo(request),
            };
            (demoted, group.rank, &bucket.id)
        });
        Ok(groups
            .into_iter()
            .filter_map(|(bucket, group)| {
                let members: Vec<_> = engines
                    .iter()
                    .filter(|engine| group.contains(engine))
                    .cloned()
                    .collect();
                (!members.is_empty()).then_some(ResolvedGroup {
                    bucket: &bucket.id,
                    engines: members,
                    policy: group.policy.as_ref(),
                })
            })
            .collect())
    }

    pub async fn pick(&self, request: &SelectionRequest<'_>) -> Result<Pick, PickError> {
        let mut rejections = Vec::new();
        for group in self.ordered_groups(request)? {
            let scoped = PickRequest {
                bucket: group.bucket,
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
                Err(error) if !self.fallback_on_rejection => return Err(error),
                Err(PickError::NoAdmissibleEngine(reasons)) => rejections.extend(reasons),
                Err(PickError::AdmissionRejected(reason)) => rejections.push(reason),
                Err(error) => return Err(error),
            }
        }
        if rejections.is_empty() {
            Err(PickError::NoCandidates)
        } else {
            Err(PickError::NoAdmissibleEngine(rejections))
        }
    }
}
