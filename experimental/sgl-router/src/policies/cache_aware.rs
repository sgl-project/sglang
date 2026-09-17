// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds bounded cache-aware candidates from ingress Indexer results.

use crate::config::AffinityConfig;
use crate::policies::admission::FreshLoadLookup;
use crate::policies::balancing::PowerOfTwoChoicesPolicy;
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
            let matched_prefix_blocks =
                cap_matched_prefix_blocks(signal.query_blocks, entry.matched_prefix_blocks);
            let matched_prefix_tokens = estimate_matched_prefix_tokens(
                input_tokens,
                signal.query_blocks,
                matched_prefix_blocks,
            );
            if !self.passes_cache_gate(input_tokens, matched_prefix_tokens) {
                continue;
            }
            candidates.push(CacheCandidate {
                worker: Arc::clone(worker),
                matched_prefix_tokens,
                uncached_tokens: input_tokens.saturating_sub(matched_prefix_tokens),
                matched_prefix_blocks,
                candidate_range_id: ctx.candidate_range_id().to_string(),
                max_pending_prefill_tokens: None,
            });
        }

        if let Some((selector, request)) = ctx.prefill_cache_bucket() {
            candidates = candidates
                .into_iter()
                .filter_map(|candidate| {
                    selector.prepare_prefill_cache_candidate(candidate, request)
                })
                .collect();
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
            enable_pressure_guard: self.config.pressure_guard,
            pressure_abs_threshold_tokens: self.config.pressure_abs_threshold_tokens,
            pressure_abs_threshold_ms: self.config.pressure_abs_threshold_ms,
            pressure_rel_threshold: self.config.pressure_rel_threshold,
            worker_queue_limit: self.config.worker_queue_limit,
            saturation_queue_floor: self.config.saturation_queue_floor,
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

/// Caps an indexer-supplied matched-block count at the blocks the query
/// actually asked about: a query cannot match more blocks than it contains.
/// Both the token estimate and the diverted-overlap histogram read the capped
/// value, so the clamp lives here rather than at each use.
fn cap_matched_prefix_blocks(query_blocks: usize, matched_prefix_blocks: u32) -> u32 {
    matched_prefix_blocks.min(u32::try_from(query_blocks).unwrap_or(u32::MAX))
}

fn estimate_matched_prefix_tokens(
    input_tokens: u64,
    query_blocks: usize,
    matched_prefix_blocks: u32,
) -> u64 {
    let matched_prefix_blocks = u64::from(cap_matched_prefix_blocks(
        query_blocks,
        matched_prefix_blocks,
    ));
    let query_blocks = u64::try_from(query_blocks).unwrap_or(u64::MAX).max(1);
    input_tokens.saturating_mul(matched_prefix_blocks) / query_blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_token_estimate_caps_untrusted_block_count() {
        assert_eq!(estimate_matched_prefix_tokens(80, 8, 99), 80);
    }

    #[test]
    fn matched_block_cap_is_shared_by_the_estimate_and_the_candidate() {
        // One clamp, two readers: the histogram must never see a block count
        // the token estimate would have thrown away.
        assert_eq!(cap_matched_prefix_blocks(8, 99), 8);
        assert_eq!(cap_matched_prefix_blocks(8, 3), 3);
        assert_eq!(cap_matched_prefix_blocks(0, 3), 0);
    }
}

#[cfg(test)]
mod proposal_tests {
    use crate::config::AffinityConfig;
    use crate::kv_events::PrefixSignal;
    use crate::policies::admission::resolve_cache_candidates;
    use crate::policies::admission::DecisionReason;
    use crate::policies::cache_aware::CacheAwarePolicy;
    use crate::policies::test_support::cache_candidate;
    use crate::policies::test_support::snapshot;
    use crate::policies::test_support::worker;
    use crate::policies::test_support::TestEngineLoad;
    use crate::policies::*;
    #[test]
    fn cache_candidate_proposal_carries_target_specific_work() {
        let hot = worker("hot");
        let proposal = CacheCandidateProposal {
            candidates: vec![CacheCandidate {
                worker: Arc::clone(&hot),
                matched_prefix_tokens: 75,
                uncached_tokens: 25,
                matched_prefix_blocks: 3,
                candidate_range_id: "global".into(),
                max_pending_prefill_tokens: None,
            }],
            cache_switch_margin_tokens: 8,
            ..Default::default()
        };

        assert_eq!(proposal.candidates[0].worker.id, hot.id);
        assert_eq!(proposal.candidates[0].matched_prefix_tokens, 75);
        assert_eq!(proposal.candidates[0].uncached_tokens, 25);
    }

    #[test]
    fn cache_affinity_uses_longest_routable_prefix_holder() {
        let model = ModelId("model".into());
        let hot = worker("hot");
        let other = worker("other");
        let workers = vec![Arc::clone(&hot), Arc::clone(&other)];
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 8,
                        worker_id: "gone".into(),
                        address: "http://gone:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 6,
                        worker_id: "hot".into(),
                        address: "http://hot:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 4,
                        worker_id: "other".into(),
                        address: "http://other:30000".into(),
                    },
                ],
                best_prefix_blocks: 8,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_request_tokens(Some(&[1, 2, 3, 4, 5, 6, 7, 8]))
            .with_input_tokens(8_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig::default());

        let proposal = policy
            .propose(&workers, &ctx)
            .expect("a routable indexer hit must propose a worker");

        assert_eq!(proposal.kind, ProposalKind::CacheAffinity);
        assert_eq!(proposal.primary.id, hot.id);
    }

    #[test]
    fn cache_candidates_keep_bounded_target_specific_uncached_work() {
        let model = ModelId("model".into());
        let hot = worker("hot");
        let warm = worker("warm");
        let workers = vec![Arc::clone(&hot), Arc::clone(&warm)];
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 8,
                        worker_id: "gone".into(),
                        address: "http://gone:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 6,
                        worker_id: "hot".into(),
                        address: "http://hot:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 4,
                        worker_id: "warm".into(),
                        address: "http://warm:30000".into(),
                    },
                ],
                best_prefix_blocks: 8,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(8_000)
            .with_external_prefix(Some(&signal));
        let config = AffinityConfig {
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 2,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::new(config);

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("routable matches must produce cache candidates")
        else {
            panic!("cache hits must not be collapsed to a primary/backup pair");
        };

        assert_eq!(proposal.candidates.len(), 2);
        assert_eq!(proposal.candidates[0].worker.id, hot.id);
        assert_eq!(proposal.candidates[0].matched_prefix_tokens, 6_000);
        assert_eq!(proposal.candidates[0].uncached_tokens, 2_000);
        assert_eq!(proposal.candidates[1].worker.id, warm.id);
        assert_eq!(proposal.candidates[1].matched_prefix_tokens, 4_000);
        assert_eq!(proposal.candidates[1].uncached_tokens, 4_000);
    }

    #[test]
    fn cache_candidate_bound_keeps_the_best_k_from_a_large_match_set() {
        let model = ModelId("model".into());
        let workers: Vec<Arc<Worker>> = (0..64)
            .map(|index| worker(&format!("w{index:02}")))
            .collect();
        let matches = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| sgl_kv_indexer::PrefixMatch {
                matched_prefix_blocks: (index + 1) as u32,
                worker_id: worker.id.0.clone(),
                address: worker.url.clone(),
            })
            .collect();
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches,
                best_prefix_blocks: workers.len() as u32,
            },
            query_blocks: 64,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(64_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 4,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 4,
            ..Default::default()
        });

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("the bounded best candidates must survive")
        else {
            panic!("cache hits must retain candidate-set semantics");
        };

        assert_eq!(proposal.candidates.len(), 4);
        assert_eq!(
            proposal
                .candidates
                .iter()
                .map(|candidate| candidate.matched_prefix_tokens)
                .collect::<Vec<_>>(),
            vec![64_000, 63_000, 62_000, 61_000]
        );
    }

    #[test]
    fn equal_cache_hits_bound_by_the_captured_local_load_before_worker_id() {
        let model = ModelId("model".into());
        let workers: Vec<Arc<Worker>> = (0..8)
            .map(|index| {
                let worker = worker(&format!("w{index}"));
                worker
                    .active_requests
                    .store(8 - index, std::sync::atomic::Ordering::Relaxed);
                worker
            })
            .collect();
        let matches = workers
            .iter()
            .map(|worker| sgl_kv_indexer::PrefixMatch {
                matched_prefix_blocks: 4,
                worker_id: worker.id.0.clone(),
                address: worker.url.clone(),
            })
            .collect();
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches,
                best_prefix_blocks: 4,
            },
            query_blocks: 4,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(4_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig {
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 2,
            ..Default::default()
        });

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("equal hits must retain the least-loaded replicas")
        else {
            panic!("cache hits must retain candidate-set semantics");
        };

        assert_eq!(
            proposal
                .candidates
                .iter()
                .map(|candidate| candidate.worker.id.0.as_str())
                .collect::<Vec<_>>(),
            vec!["w7", "w6"]
        );
    }

    #[test]
    fn cache_candidate_gates_are_configurable_lower_bounds_with_and_semantics() {
        let model = ModelId("model".into());
        let half = worker("half");
        let below_ratio = worker("below-ratio");
        let workers = vec![Arc::clone(&half), Arc::clone(&below_ratio)];
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 4,
                        worker_id: "half".into(),
                        address: "http://half:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 3,
                        worker_id: "below-ratio".into(),
                        address: "http://below-ratio:30000".into(),
                    },
                ],
                best_prefix_blocks: 4,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(80)
            .with_external_prefix(Some(&signal));
        let config = AffinityConfig {
            cache_affinity_min_matched_tokens: Some(30),
            cache_affinity_min_match_ratio: Some(0.5),
            cache_candidate_min_workers: 8,
            cache_candidate_max_workers: 8,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::new(config);

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("one candidate satisfies both lower bounds")
        else {
            panic!("the admitted cache candidate must retain H/E");
        };

        assert_eq!(proposal.candidates.len(), 1);
        assert_eq!(proposal.candidates[0].worker.id, half.id);
    }

    #[test]
    fn default_cache_gate_rejects_a_prefix_below_the_absolute_floor() {
        let model = ModelId("model".into());
        let weak = worker("weak");
        let workers = vec![Arc::clone(&weak)];
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![sgl_kv_indexer::PrefixMatch {
                    matched_prefix_blocks: 3,
                    worker_id: "weak".into(),
                    address: "http://weak:30000".into(),
                }],
                best_prefix_blocks: 3,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(80)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig::default());

        let proposal = policy
            .propose_prefill(&workers, &ctx)
            .expect("a weak hit must degrade to no-hit P2, not fail selection");

        assert!(
            matches!(proposal, PrefillProposal::Pair(_)),
            "the default gate must keep a tiny hit from forcing cache affinity"
        );
    }

    #[test]
    fn default_cache_gate_accepts_the_indexer_scan_cap_for_a_long_prompt() {
        let model = ModelId("model".into());
        let holder = worker("holder");
        let workers = vec![Arc::clone(&holder)];
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![sgl_kv_indexer::PrefixMatch {
                    matched_prefix_blocks: 2_048,
                    worker_id: "holder".into(),
                    address: "http://holder:30000".into(),
                }],
                best_prefix_blocks: 2,
            },
            query_blocks: 4_125,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(4_125)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig::default());

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("the default absolute gate must accept a 2048-token lower bound")
        else {
            panic!("a server-truncated long-prefix hit must not degrade to P2");
        };

        assert_eq!(proposal.candidates[0].worker.id, holder.id);
        assert_eq!(proposal.candidates[0].matched_prefix_tokens, 2_048);
        assert_eq!(proposal.candidates[0].uncached_tokens, 2_077);
    }

    #[test]
    fn cache_affinity_without_signal_degrades_to_a_plain_p2_proposal() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = CacheAwarePolicy::new(AffinityConfig::default());
        let ctx = SelectionContext::new(&model, None);

        let proposal = policy
            .propose(&workers, &ctx)
            .expect("cache miss must still route through P2");

        assert_eq!(proposal.kind, ProposalKind::PowerOfTwo);
        assert!(proposal.backup.is_some());
    }

    #[test]
    fn cache_tournament_skips_capacity_exhausted_matches_and_returns_no_backup() {
        let full = worker("full");
        let winner = worker("winner");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&full, 90, 10, None),
                cache_candidate(&winner, 70, 30, None),
            ],
            cache_switch_margin_tokens: 16,
            ..Default::default()
        };
        let loads = snapshot(&[
            (
                &full,
                TestEngineLoad {
                    num_running_reqs: 8,
                    num_tokens: 9_950,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &winner,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads, &[])
            .decision
            .expect("a later admitted cache match must survive");

        assert_eq!(decision.selected.id, winner.id);
        assert_eq!(decision.primary.id, winner.id);
        assert!(decision.backup.is_none());
        assert_eq!(decision.reason, DecisionReason::CacheCandidate);
    }

    #[test]
    fn cache_tournament_compares_every_admitted_challenger_before_finalizing() {
        let first = worker("first");
        let second = worker("second");
        let final_winner = worker("final-winner");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&first, 40, 60, None),
                cache_candidate(&second, 60, 40, None),
                cache_candidate(&final_winner, 80, 20, None),
            ],
            cache_switch_margin_tokens: 0,
            ..Default::default()
        };
        let loads = snapshot(&[
            (
                &first,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &second,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &final_winner,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads, &[])
            .decision
            .expect("all admitted candidates must participate in the tournament");

        assert_eq!(decision.selected.id, final_winner.id);
        assert_eq!(decision.primary.id, final_winner.id);
        assert!(decision.backup.is_none());
    }

    #[test]
    fn cache_tournament_uses_uncached_work_for_pending_but_full_input_for_kv() {
        let candidate = worker("candidate");
        let proposal = CacheCandidateProposal {
            candidates: vec![cache_candidate(&candidate, 80, 20, Some(30))],
            cache_switch_margin_tokens: 16,
            ..Default::default()
        };
        let pending_allows = snapshot(&[(
            &candidate,
            TestEngineLoad {
                num_waiting_reqs: 5,
                max_total_num_tokens: 1_000,
                ..TestEngineLoad::default()
            },
        )]);
        assert!(
            resolve_cache_candidates(&proposal, 100, &pending_allows, &[])
                .decision
                .is_some(),
            "pending admission must project E=20, not L=100"
        );

        let kv_rejects = snapshot(&[(
            &candidate,
            TestEngineLoad {
                num_tokens: 30,
                num_waiting_reqs: 5,
                max_total_num_tokens: 100,
                ..TestEngineLoad::default()
            },
        )]);
        assert!(
            resolve_cache_candidates(&proposal, 100, &kv_rejects, &[])
                .decision
                .is_none(),
            "KV safety must conservatively project the complete input L=100"
        );
    }

    #[test]
    fn cache_tournament_keeps_cache_gain_when_legacy_token_guard_is_unavailable() {
        let congested = worker("congested");
        let idle = worker("idle");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&congested, 90, 10, None),
                cache_candidate(&idle, 80, 20, None),
            ],
            cache_switch_margin_tokens: 32,
            ..Default::default()
        };
        let loads = snapshot(&[
            (
                &congested,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &idle,
                TestEngineLoad {
                    num_waiting_reqs: 10,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads, &[])
            .decision
            .unwrap();
        assert_eq!(decision.selected.id, congested.id);
    }

    #[test]
    fn cache_tournament_keeps_a_material_cache_gain_despite_pressure() {
        let hot = worker("hot");
        let idle = worker("idle");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&hot, 90, 10, None),
                cache_candidate(&idle, 20, 80, None),
            ],
            cache_switch_margin_tokens: 32,
            ..Default::default()
        };
        let loads = snapshot(&[
            (
                &hot,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &idle,
                TestEngineLoad {
                    num_waiting_reqs: 10,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads, &[])
            .decision
            .unwrap();
        assert_eq!(
            decision.selected.id, hot.id,
            "pressure may break a near tie, but must not erase a material cache-work gain"
        );
    }

    #[test]
    fn cache_tournament_uses_work_order_when_legacy_token_guard_is_unavailable() {
        let best_work = worker("best-work");
        let near_tie = worker("near-tie");
        let beyond_margin = worker("beyond-margin");
        let proposal = CacheCandidateProposal {
            // The policy supplies candidates in increasing E order. Each
            // adjacent pair is a near tie, but the last candidate is more
            // than one configured margin away from the global work minimum.
            candidates: vec![
                cache_candidate(&best_work, 100, 0, None),
                cache_candidate(&near_tie, 80, 20, None),
                cache_candidate(&beyond_margin, 60, 40, None),
            ],
            cache_switch_margin_tokens: 32,
            ..Default::default()
        };
        let loads = snapshot(&[
            (
                &best_work,
                TestEngineLoad {
                    num_waiting_reqs: 10_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &near_tie,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &beyond_margin,
                TestEngineLoad {
                    num_waiting_reqs: 0,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads, &[])
            .decision
            .unwrap();
        assert_eq!(
            decision.selected.id, best_work.id,
            "without a unit-compatible token-pressure signal, cache work remains authoritative"
        );
    }
}
