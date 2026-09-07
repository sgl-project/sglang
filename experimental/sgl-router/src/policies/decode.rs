// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Decode policy extension point, independent of prefill affinity.

use crate::config::DecodePolicyKind;
use crate::policies::admission::{
    compare_decode_pressure, resolve_decode, CandidateDomain, DecisionReason, FinalDecision,
    RoutingStage,
};
use crate::policies::engine_load::EngineLoadSnapshot;
use crate::policies::registry::select_decode_with_affinity;
use crate::policies::{ProposalKind, SelectionProposal};
use rand::Rng;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct DecodeSelectionContext<'a> {
    load_snapshot: Option<&'a EngineLoadSnapshot>,
    prefill_url: Option<&'a str>,
}

impl<'a> DecodeSelectionContext<'a> {
    pub fn new() -> Self {
        Self {
            load_snapshot: None,
            prefill_url: None,
        }
    }

    /// Engine load snapshot captured at request ingress.
    pub fn with_load_snapshot(mut self, load_snapshot: &'a EngineLoadSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    pub fn load_snapshot(&self) -> Option<&EngineLoadSnapshot> {
        self.load_snapshot
    }

    /// Prefill URL used by `legacy_host_affinity`.
    pub fn with_prefill_url(mut self, prefill_url: &'a str) -> Self {
        self.prefill_url = Some(prefill_url);
        self
    }

    pub fn prefill_url(&self) -> Option<&str> {
        self.prefill_url
    }
}

pub trait DecodePolicy: Send + Sync + std::fmt::Debug {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal>;
}

/// Resolves decode admission and degrades to Power-of-Two when capacity is exhausted.
pub fn resolve_decode_with_capacity_fallback(
    domain: &CandidateDomain,
    proposal: &SelectionProposal,
    request_kv_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> Option<FinalDecision> {
    if let Some(decision) = resolve_decode(domain, proposal, request_kv_tokens, snapshot) {
        return Some(decision);
    }
    if domain.stage != RoutingStage::Decode
        || !domain
            .workers
            .iter()
            .any(|worker| worker.id == proposal.primary.id)
    {
        return None;
    }

    let fallback = DecodePowerOfTwoPolicy::new().propose(
        domain,
        &DecodeSelectionContext::new().with_load_snapshot(snapshot),
    )?;
    Some(FinalDecision {
        selected: fallback.primary,
        primary: Arc::clone(&proposal.primary),
        backup: proposal
            .backup
            .as_ref()
            .filter(|backup| domain.workers.iter().any(|worker| worker.id == backup.id))
            .cloned(),
        reason: DecisionReason::CapacityFallbackPowerOfTwo,
        candidate_range_id: domain.id.clone(),
        load_snapshot_version: snapshot.version,
    })
}

/// Samples two workers from a decode domain and orders them by decode pressure.
#[derive(Debug, Default)]
pub struct DecodePowerOfTwoPolicy;

impl DecodePowerOfTwoPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl DecodePolicy for DecodePowerOfTwoPolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match domain.workers.len() {
            0 => None,
            1 => Some(
                SelectionProposal::primary(Arc::clone(&domain.workers[0]))
                    .with_kind(ProposalKind::PowerOfTwo),
            ),
            len => {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len - 1);
                if j >= i {
                    j += 1;
                }
                let left = &domain.workers[i];
                let right = &domain.workers[j];
                let (primary, backup) =
                    if compare_decode_pressure(left, right, ctx.load_snapshot()).is_gt() {
                        (Arc::clone(right), Arc::clone(left))
                    } else {
                        (Arc::clone(left), Arc::clone(right))
                    };
                Some(
                    SelectionProposal::with_backup(primary, backup)
                        .with_kind(ProposalKind::PowerOfTwo),
                )
            }
        }
    }
}

/// Compatibility policy for legacy same-host PD decode selection.
#[derive(Debug, Default)]
pub struct LegacyHostAffinityDecodePolicy;

impl DecodePolicy for LegacyHostAffinityDecodePolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        let prefill_url = ctx.prefill_url()?;
        select_decode_with_affinity(prefill_url, &domain.workers).map(SelectionProposal::primary)
    }
}

/// Builds a decode policy scoped to one role.
pub fn build_decode_policy(kind: DecodePolicyKind) -> Box<dyn DecodePolicy> {
    match kind {
        DecodePolicyKind::PowerOfTwo => Box::new(DecodePowerOfTwoPolicy::new()),
        DecodePolicyKind::LegacyHostAffinity => Box::new(LegacyHostAffinityDecodePolicy),
    }
}
