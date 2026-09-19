// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Order buckets by request length; each bucket owns the groups used to pick engines.
//!
//! ```text
//! BucketResolver (one model's buckets)
//!   -> Bucket (token limits, context capacity, rank)
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
//! their load/KV/affinity dependencies and share observations within each attempt.

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
    /// Break ties between equally sized buckets; lower ranks win.
    pub rank: u32,
    /// Inclusive input-token range used to choose the bucket.
    pub limits: TokenLimits,
    /// Full sequence capacity, checked against the expected peak when known.
    pub max_context_tokens: Option<u64>,
    pub groups: BucketGroups,
}

impl Bucket {
    pub fn new(id: impl Into<String>, groups: BucketGroups) -> Self {
        Self {
            id: id.into(),
            rank: 0,
            limits: TokenLimits::default(),
            max_context_tokens: None,
            groups,
        }
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

/// Model-specific bucket configuration. Selection does not inspect engine state.
#[derive(Debug, Default)]
pub struct BucketResolver {
    pub buckets: Vec<Bucket>,
}

impl BucketResolver {
    pub fn new(buckets: Vec<Bucket>) -> Self {
        Self { buckets }
    }

    /// Return all length-compatible buckets, ordered by input capacity, rank, and ID.
    /// The caller tries their groups in order until a complete engine selection succeeds.
    pub fn resolve(
        &self,
        input_tokens: u64,
        expected_peak_tokens: Option<u64>,
    ) -> Result<Vec<&Bucket>, PickError> {
        if expected_peak_tokens.is_some_and(|tokens| tokens < input_tokens) {
            return Err(PickError::InvalidSignal(
                "expected peak tokens are below input length".into(),
            ));
        }
        let mut buckets: Vec<_> = self
            .buckets
            .iter()
            .filter(|bucket| bucket.fits(input_tokens, expected_peak_tokens))
            .collect();
        buckets.sort_by_key(|bucket| (bucket.input_capacity(), bucket.rank, &bucket.id));
        Ok(buckets)
    }
}
