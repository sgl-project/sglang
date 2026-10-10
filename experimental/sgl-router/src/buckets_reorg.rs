// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Order length-compatible buckets by optional SLO preferences, then capacity and rank.
//!
//! ```text
//! BucketResolver (one model's buckets)
//!   -> Bucket (token limits, context capacity, rank, SLO estimates)
//!        -> Plain: one EngineGroup
//!        -> PD: prefill + decode EngineGroups
//!             -> each EngineGroup: worker membership + its own Policy
//! ```
//!
//! - [`BucketResolver::resolve`] returns length-compatible buckets in preference order.
//! - [`EngineGroup::pick`] filters live workers by model, health, stage, and membership,
//!   then calls [`Policy::pick`] and validates the returned engine.
//!
//! The handler tries buckets in order, advancing on missing candidates or admission
//! rejection. Both P/D picks must succeed in the same bucket before dispatch.
//! [`WorkerRegistry`] owns live workers; groups reference their IDs. Policies own
//! their load/KV/affinity dependencies and pass selected observations to admission.

use std::collections::HashSet;
use std::sync::Arc;

use crate::discovery::{ModelId, WorkerId};
use crate::policies_reorg::{Pick, PickError, PickRequest, Policy, Stage};
use crate::workers::WorkerRegistry;

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenLimits {
    pub min: Option<u64>,
    pub max: Option<u64>,
}

impl TokenLimits {
    fn fits(&self, tokens: u64) -> bool {
        self.min.is_none_or(|min| tokens >= min) && self.max.is_none_or(|max| tokens <= max)
    }
}

#[derive(Debug)]
pub struct EngineGroup {
    /// `None` includes all registered engines matching the request's model and role.
    pub worker_ids: Option<HashSet<WorkerId>>,
    pub policy: Arc<dyn Policy>,
}

impl EngineGroup {
    pub fn new(policy: Arc<dyn Policy>) -> Self {
        Self {
            worker_ids: None,
            policy,
        }
    }

    /// Resolve live members and invoke this group's policy. Never changes buckets.
    pub async fn pick(
        &self,
        workers: &WorkerRegistry,
        request: &PickRequest<'_>,
    ) -> Result<Pick, PickError> {
        let mut engines: Vec<_> = workers
            .healthy_workers_for(request.model)
            .into_iter()
            .filter(|engine| engine.mode() == request.stage)
            .filter(|engine| {
                self.worker_ids
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&engine.id))
            })
            .collect();
        // Stable order so cursor-based policies see a consistent candidate list.
        engines.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        let pick = self.policy.pick(&engines, request).await?;
        if !engines
            .iter()
            .any(|engine| Arc::ptr_eq(engine, &pick.engine))
        {
            return Err(PickError::OutsideCandidates(pick.engine.id.clone()));
        }
        Ok(pick)
    }
}

/// A bucket serves a request on one plain engine or on its own P/D groups.
#[derive(Debug)]
pub enum BucketGroups {
    Plain(EngineGroup),
    Pd {
        prefill: EngineGroup,
        decode: EngineGroup,
    },
}

/// Prepared request facts shared by all bucket attempts. The bucket supplies
/// its ID and each group's stage when calling policies.
#[derive(Debug)]
pub struct BucketRequest<'a> {
    pub model: &'a ModelId,
    pub input_tokens: u64,
    pub expected_peak_tokens: Option<u64>,
    pub prefix: Option<&'a crate::policies_reorg::cache_aware::PrefixMemo>,
    pub token_ids: Option<&'a [u32]>,
    pub session_key: Option<&'a str>,
    pub routing_key: Option<&'a str>,
}

/// A complete selection from one bucket. For plain serving, `prefill` is the
/// plain engine and `decode` is absent; PD supplies both picks.
#[derive(Debug)]
pub struct BucketPick {
    pub prefill: Pick,
    pub decode: Option<Pick>,
}

#[derive(Debug)]
pub struct Bucket {
    pub id: String,
    /// Break ties within an SLO tier between equally sized buckets; lower ranks win.
    pub rank: u32,
    /// Inclusive input-token range used to choose the bucket.
    pub limits: TokenLimits,
    /// Full sequence capacity, checked against the expected peak when known.
    pub max_context_tokens: Option<u64>,
    /// Optional service estimates used only for bucket ordering.
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub groups: BucketGroups,
}

impl Bucket {
    pub fn new(id: impl Into<String>, groups: BucketGroups) -> Self {
        Self {
            id: id.into(),
            rank: 0,
            limits: TokenLimits::default(),
            max_context_tokens: None,
            ttft_ms: None,
            tokens_per_second: None,
            groups,
        }
    }

    /// Reject a policy installed on a stage it cannot serve before any request reaches it.
    pub fn validate(&self) -> Result<(), PickError> {
        let groups: &[(Stage, &EngineGroup)] = match &self.groups {
            BucketGroups::Plain(group) => &[(Stage::Plain, group)],
            BucketGroups::Pd { prefill, decode } => {
                &[(Stage::Prefill, prefill), (Stage::Decode, decode)]
            }
        };
        for (stage, group) in groups {
            if !group.policy.supports(*stage) {
                return Err(PickError::InvalidConfiguration(format!(
                    "bucket {} installs a policy that cannot serve the {stage:?} stage",
                    self.id
                )));
            }
        }
        Ok(())
    }

    /// Select this bucket's plain engine or complete P/D pair, without dispatching.
    /// A failed group reports its stage; the caller may then try another bucket.
    pub async fn pick_engines(
        &self,
        workers: &WorkerRegistry,
        request: &BucketRequest<'_>,
    ) -> Result<BucketPick, (Stage, PickError)> {
        let (prefill, decode) = match &self.groups {
            BucketGroups::Plain(group) => (
                self.pick_from_group(group, Stage::Plain, workers, request)
                    .await?,
                None,
            ),
            BucketGroups::Pd { prefill, decode } => {
                let prefill = self
                    .pick_from_group(prefill, Stage::Prefill, workers, request)
                    .await?;
                let decode = self
                    .pick_from_group(decode, Stage::Decode, workers, request)
                    .await?;
                (prefill, Some(decode))
            }
        };
        Ok(BucketPick { prefill, decode })
    }

    /// Scope the request to this bucket and role, then ask the group for one engine.
    async fn pick_from_group(
        &self,
        group: &EngineGroup,
        stage: Stage,
        workers: &WorkerRegistry,
        request: &BucketRequest<'_>,
    ) -> Result<Pick, (Stage, PickError)> {
        let request = PickRequest {
            model: request.model,
            stage,
            bucket: &self.id,
            input_tokens: request.input_tokens,
            expected_peak_tokens: request.expected_peak_tokens,
            prefix: request.prefix,
            token_ids: request.token_ids,
            session_key: request.session_key,
            routing_key: request.routing_key,
        };
        group
            .pick(workers, &request)
            .await
            .map_err(|error| (stage, error))
    }

    fn fits(&self, input_tokens: u64, expected_peak_tokens: Option<u64>) -> bool {
        self.limits.fits(input_tokens)
            && self
                .max_context_tokens
                .is_none_or(|max| expected_peak_tokens.unwrap_or(input_tokens) <= max)
    }

    fn input_capacity(&self) -> u64 {
        self.limits
            .max
            .unwrap_or(u64::MAX)
            .min(self.max_context_tokens.unwrap_or(u64::MAX))
    }
}

/// Soft preference; nonpreferred buckets remain available for fallback.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SloPreference {
    #[default]
    Disabled,
    SloFirst,
    BestEffort,
}

impl SloPreference {
    fn penalty(self, matches: Option<bool>) -> u8 {
        match (self, matches) {
            (Self::SloFirst, Some(false)) | (Self::BestEffort, Some(true)) => 1,
            _ => 0,
        }
    }
}

/// Model-specific bucket configuration. Selection does not inspect engine state.
#[derive(Debug, Default)]
pub struct BucketResolver {
    pub buckets: Vec<Bucket>,
    pub ttft_slo: SloPreference,
    pub tps_slo: SloPreference,
}

impl BucketResolver {
    /// Fails if any bucket installs a policy on a stage it cannot serve.
    pub fn new(buckets: Vec<Bucket>) -> Result<Self, PickError> {
        for bucket in &buckets {
            bucket.validate()?;
        }
        Ok(Self {
            buckets,
            ..Self::default()
        })
    }

    /// Return all length-compatible buckets, ordered by unmet SLO preferences,
    /// then input capacity, rank, and ID. Both preferences have equal weight.
    /// The caller tries their groups in order until a complete engine selection succeeds.
    pub fn resolve(
        &self,
        input_tokens: u64,
        expected_peak_tokens: Option<u64>,
        ttft_ms: Option<u64>,
        tokens_per_second: Option<f64>,
    ) -> Result<Vec<&Bucket>, PickError> {
        if expected_peak_tokens.is_some_and(|tokens| tokens < input_tokens) {
            return Err(PickError::InvalidSignal(
                "expected peak tokens are below input length".into(),
            ));
        }
        if self.ttft_slo != SloPreference::Disabled && ttft_ms == Some(0) {
            return Err(PickError::InvalidSignal(
                "requested TTFT must be positive".into(),
            ));
        }
        if self.tps_slo != SloPreference::Disabled
            && tokens_per_second.is_some_and(|tps| !tps.is_finite() || tps <= 0.0)
        {
            return Err(PickError::InvalidSignal(
                "requested tokens per second must be finite and positive".into(),
            ));
        }
        let mut buckets: Vec<_> = self
            .buckets
            .iter()
            .filter(|bucket| bucket.fits(input_tokens, expected_peak_tokens))
            .collect();
        buckets.sort_by_key(|bucket| {
            let ttft_matches = ttft_ms.map(|target| {
                bucket
                    .ttft_ms
                    .is_some_and(|estimate| estimate > 0 && estimate <= target)
            });
            let tps_matches = tokens_per_second.map(|target| {
                bucket.tokens_per_second.is_some_and(|estimate| {
                    estimate.is_finite() && estimate > 0.0 && estimate >= target
                })
            });
            let penalty = self.ttft_slo.penalty(ttft_matches) + self.tps_slo.penalty(tps_matches);
            (penalty, bucket.input_capacity(), bucket.rank, &bucket.id)
        });
        Ok(buckets)
    }
}
