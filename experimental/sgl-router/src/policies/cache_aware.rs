// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds bounded cache-aware candidates from ingress Indexer results.

use crate::config::AffinityConfig;
use crate::policies::admission::FreshLoadLookup;
use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
use crate::policies::{
    CacheCandidate, CacheCandidateProposal, Policy, PrefillProposal, ProposalKind,
    SelectionContext, SelectionProposal,
};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: AffinityConfig,
}

impl CacheAwarePolicy {
    pub fn new(config: AffinityConfig) -> Self {
        Self { config }
    }

    fn cache_candidate_proposal(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<CacheCandidateProposal> {
        let input_tokens = ctx.input_tokens()?;
        let signal = ctx.external_prefix()?;
        let sgl_kv_indexer::PrefixOutcome::Matched { matches, .. } = &signal.outcome else {
            return None;
        };
        if signal.query_blocks == 0 || workers.is_empty() {
            return None;
        }

        // The #33370 indexer contract routes on the worker address (matched
        // byte-for-byte against registered worker URLs); worker_id is for
        // logs only.
        let by_url: HashMap<&str, &Arc<Worker>> = workers
            .iter()
            .map(|worker| (worker.url.as_str(), worker))
            .collect();
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for entry in matches {
            let Some(worker) = by_url.get(entry.address.as_str()) else {
                continue;
            };
            if entry.matched_prefix_blocks == 0 || !seen.insert(worker.id.clone()) {
                continue;
            }
            let matched_prefix_tokens = estimate_matched_prefix_tokens(
                input_tokens,
                signal.query_blocks,
                entry.matched_prefix_blocks,
            );
            if !self.passes_cache_gate(input_tokens, matched_prefix_tokens) {
                continue;
            }
            candidates.push(CacheCandidate {
                worker: Arc::clone(worker),
                matched_prefix_tokens,
                uncached_tokens: input_tokens.saturating_sub(matched_prefix_tokens),
                candidate_range_id: ctx.candidate_range_id().to_string(),
                max_pending_prefill_tokens: None,
            });
        }

        let limit = self.candidate_limit(workers.len());
        if limit == 0 {
            return None;
        }
        let loads = FreshLoadLookup::new(
            ctx.load_snapshot(),
            candidates.iter().map(|candidate| &candidate.worker),
        );
        if candidates.len() > limit {
            candidates.select_nth_unstable_by(limit, |left, right| {
                compare_candidate_seed(left, right, &loads)
            });
            candidates.truncate(limit);
        }
        candidates.sort_by(|left, right| compare_candidate_seed(left, right, &loads));
        if candidates.is_empty() {
            return None;
        }
        Some(CacheCandidateProposal {
            candidates,
            cache_switch_margin_tokens: self.config.cache_switch_margin_tokens,
        })
    }

    fn passes_cache_gate(&self, input_tokens: u64, matched_prefix_tokens: u64) -> bool {
        self.config
            .cache_affinity_min_matched_tokens
            .is_none_or(|minimum| matched_prefix_tokens >= minimum)
            && self
                .config
                .cache_affinity_min_match_ratio
                .is_none_or(|minimum| {
                    input_tokens > 0
                        && matched_prefix_tokens as f64 / input_tokens as f64 >= minimum
                })
    }

    fn candidate_limit(&self, worker_count: usize) -> usize {
        let proportional = (self.config.cache_candidate_ratio.clamp(0.0, 1.0) * worker_count as f64)
            .ceil() as usize;
        worker_count
            .min(self.config.cache_candidate_max_workers)
            .min(self.config.cache_candidate_min_workers.max(proportional))
    }
}

fn compare_candidate_seed(
    left: &CacheCandidate,
    right: &CacheCandidate,
    loads: &FreshLoadLookup<'_>,
) -> Ordering {
    right
        .matched_prefix_tokens
        .cmp(&left.matched_prefix_tokens)
        .then_with(|| loads.compare_prefill_pressure(&left.worker, &right.worker))
        .then_with(|| left.worker.id.0.cmp(&right.worker.id.0))
}

impl Policy for CacheAwarePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        self.propose(workers, ctx).map(|proposal| proposal.primary)
    }

    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match self.propose_prefill(workers, ctx)? {
            PrefillProposal::Pair(proposal) => Some(proposal),
            PrefillProposal::CacheCandidates(proposal) => {
                let candidate = proposal.candidates.into_iter().next()?;
                Some(
                    SelectionProposal::primary(candidate.worker)
                        .with_kind(ProposalKind::CacheAffinity),
                )
            }
        }
    }

    fn propose_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        if ctx.affinity_lookup_enabled() {
            if let Some(proposal) = self.cache_candidate_proposal(workers, ctx) {
                return Some(PrefillProposal::CacheCandidates(proposal));
            }
        }
        PowerOfTwoChoicesPolicy::new()
            .propose(workers, ctx)
            .map(PrefillProposal::Pair)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }
}

fn estimate_matched_prefix_tokens(
    input_tokens: u64,
    query_blocks: usize,
    matched_prefix_blocks: u32,
) -> u64 {
    let query_blocks = u64::try_from(query_blocks).unwrap_or(u64::MAX).max(1);
    let matched_prefix_blocks = u64::from(matched_prefix_blocks).min(query_blocks);
    input_tokens.saturating_mul(matched_prefix_blocks) / query_blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_token_estimate_caps_untrusted_block_count() {
        assert_eq!(estimate_matched_prefix_tokens(80, 8, 99), 80);
    }
}
