// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefill worker selection: the ladder that turns a policy proposal into a
//! committed worker.
//!
//! WHY this is a module rather than a block inside the chat handler: the ladder
//! has several rungs — cache-candidate resolution, the global session-affinity
//! probe, per-domain admission, the capacity fallback — and each rung fails into
//! the next. Rungs get added over time, and the assertions worth writing are
//! almost always about the ladder as a whole ("a saturated fleet still routes",
//! "a sampled choice never lands on a rejected worker"), not about one rung in
//! isolation. Written as closures inside an HTTP handler those assertions can
//! only be expressed as end-to-end HTTP tests; written here they are unit tests.
//!
//! The module owns the decision and reports why; it does not own the HTTP
//! response. Mapping a failed selection onto a status code stays in the route.

use std::sync::Arc;

use crate::config::{DecodePolicyKind, PolicyKind, SessionAffinityMode};
use crate::discovery::ModelId;
use crate::policies::admission::{
    resolve_cache_candidates, resolve_decode, resolve_prefill, resolve_prefill_admitted,
    CandidateDomain, CandidateRange, DecisionReason,
};
use crate::policies::buckets::{BucketRequest, BucketSelector};
use crate::policies::decode::{
    build_decode_policy, resolve_decode_with_capacity_fallback, DecodeSelectionContext,
};
use crate::policies::engine_load::EngineLoadSnapshot;
use crate::policies::{
    ExternalPrefixSignal, Policy, PrefillProposal, ProposalKind, SelectionContext,
};
use crate::server::metrics::{MetricsRegistry, PolicySelectionFailureReason};
use crate::workers::Worker;

/// Everything one prefill selection reads. Collaborators first, then the
/// per-request facts.
pub(crate) struct PrefillSelectionInputs<'a> {
    pub policy: &'a Arc<dyn Policy>,
    pub policy_kind: PolicyKind,
    pub bucket_selector: &'a BucketSelector,
    pub metrics: &'a MetricsRegistry,
    pub model_id: &'a ModelId,
    /// Model name, for log lines only.
    pub model_str: &'a str,
    pub body: Option<&'a [u8]>,
    pub routing_key: Option<&'a str>,
    pub session_id: Option<&'a str>,
    pub request_input_tokens: u64,
    pub request_tokens: Option<&'a [u32]>,
    pub external_prefix: Option<&'a ExternalPrefixSignal>,
    /// Required whenever `policy.uses_shared_prefill_admission()`; the
    /// per-domain rung panics without it. `Policy::needs_load_snapshot`
    /// defaults to `uses_shared_prefill_admission`, which is what keeps the
    /// two in step for the ingress caller.
    pub load_snapshot: Option<&'a EngineLoadSnapshot>,
    pub workers: &'a [Arc<Worker>],
    pub bucket_request: BucketRequest,
    /// The configured mode. Without Bucket partitioning all modes reduce to
    /// the single global domain, and the ladder applies that reduction itself.
    pub session_affinity_mode: SessionAffinityMode,
}

/// The outcome of one selection: the worker if the ladder found one, and the
/// reason the last rung to record one gave. `failure_reason` is meaningful
/// only when `selected` is `None`.
pub(crate) struct PrefillSelection {
    pub selected: Option<Arc<Worker>>,
    pub failure_reason: PolicySelectionFailureReason,
}

/// Runs the prefill selection ladder.
pub(crate) fn select_prefill_worker(inputs: &PrefillSelectionInputs<'_>) -> PrefillSelection {
    let mut selector = Selector {
        inputs,
        failure_reason: PolicySelectionFailureReason::ProposalEmpty,
    };
    let selected = selector.run();
    PrefillSelection {
        selected,
        failure_reason: selector.failure_reason,
    }
}

/// Carries the failure reason across rungs. Each rung that gives up overwrites
/// it, so the reported reason is the one the last rung to record any gave —
/// a rung that returns `None` without recording leaves the previous reason
/// standing.
struct Selector<'a> {
    inputs: &'a PrefillSelectionInputs<'a>,
    failure_reason: PolicySelectionFailureReason,
}

impl<'a> Selector<'a> {
    fn run(&mut self) -> Option<Arc<Worker>> {
        let inputs = self.inputs;
        // Without Bucket partitioning all modes reduce to the single global
        // domain, so reduce once here rather than trusting every caller to.
        let session_affinity_mode = if inputs.bucket_selector.is_enabled() {
            inputs.session_affinity_mode
        } else {
            SessionAffinityMode::Bucket
        };
        let use_global_affinity_probe = inputs.bucket_selector.is_enabled()
            && inputs.policy.is_bucket_affinity_policy()
            && session_affinity_mode != SessionAffinityMode::Bucket;

        // Cache-Aware resolves one bounded global candidate set and returns a final winner.
        let cache_winner = self.cache_winner();

        let global_affinity_probe = use_global_affinity_probe
            .then(|| {
                let snapshot = inputs.load_snapshot?;
                let global_range = CandidateRange::global(inputs.workers);
                let probe_ctx = self
                    .base_context(global_range.id)
                    .with_load_snapshot(snapshot)
                    .without_affinity_assignment();
                inputs.policy.propose(global_range.workers, &probe_ctx)
            })
            .flatten();
        // A new or stale session may create its first assignment in the target Bucket.
        let global_affinity_missed = global_affinity_probe
            .as_ref()
            .is_some_and(|proposal| !matches!(proposal.kind, ProposalKind::SessionAffinity));
        let global_affinity_worker = global_affinity_probe
            .and_then(|proposal| {
                matches!(proposal.kind, ProposalKind::SessionAffinity).then_some(proposal.primary)
            })
            .and_then(|primary| {
                inputs.bucket_selector.prefill_affinity_domain(
                    inputs.workers,
                    &primary,
                    inputs.bucket_request,
                )
            })
            // Rebuild the backup inside the primary's own Bucket.
            .and_then(|domain| self.select_in_domain(&domain, true, false, false));

        cache_winner.or_else(|| {
            // Materialize normal domains only when Cache-Aware has no winner.
            let prefill_domains = inputs
                .bucket_selector
                .prefill_domains(inputs.workers, inputs.bucket_request);
            if inputs.policy_kind == PolicyKind::CacheAware {
                // Cache miss or failure retries ordered domains with ordinary P2.
                return self.select_domains(&prefill_domains, false, false);
            }
            global_affinity_worker.or_else(|| match session_affinity_mode {
                SessionAffinityMode::GlobalPreserve if global_affinity_missed => {
                    self.select_domains(&prefill_domains, true, true)
                }
                SessionAffinityMode::GlobalPreserve => {
                    self.select_domains(&prefill_domains, false, false)
                }
                SessionAffinityMode::Bucket | SessionAffinityMode::GlobalRebind => {
                    self.select_domains(&prefill_domains, true, true)
                }
            })
        })
    }

    /// The per-request `SelectionContext` every rung starts from.
    fn base_context(&self, candidate_range_id: &'a str) -> SelectionContext<'a> {
        let inputs = self.inputs;
        SelectionContext::with_routing_key(inputs.model_id, inputs.body, inputs.routing_key)
            .with_session_id(inputs.session_id)
            .with_candidate_range_id(candidate_range_id)
            .with_input_tokens(inputs.request_input_tokens)
            .with_request_tokens(inputs.request_tokens)
            .with_external_prefix(inputs.external_prefix)
    }

    fn cache_winner(&mut self) -> Option<Arc<Worker>> {
        let inputs = self.inputs;
        if inputs.policy_kind != PolicyKind::CacheAware {
            return None;
        }
        let snapshot = inputs.load_snapshot?;
        let global_range = CandidateRange::global(inputs.workers);
        let cache_ctx = self
            .base_context(global_range.id)
            .with_load_snapshot(snapshot)
            .with_prefill_cache_bucket(inputs.bucket_selector, inputs.bucket_request);
        let PrefillProposal::CacheCandidates(proposal) = inputs
            .policy
            .propose_prefill(global_range.workers, &cache_ctx)?
        else {
            return None;
        };
        let bounded_candidate_count = proposal.candidates.len();
        let cache_decision =
            resolve_cache_candidates(&proposal, inputs.request_input_tokens, snapshot);
        inputs
            .metrics
            .record_cache_admission_evaluations(cache_decision.admission_evaluated_candidates);
        inputs
            .metrics
            .record_cache_admission_rejections(cache_decision.admission_rejected_candidates);
        inputs.metrics.record_cache_pressure_guard(
            cache_decision.pressure_guard_compared_pairs,
            cache_decision.pressure_guard_overrides,
        );
        inputs
            .metrics
            .record_cache_monitor_decision(cache_decision.prefill_pressure_source);
        let Some(decision) = cache_decision.decision else {
            self.failure_reason = PolicySelectionFailureReason::CacheCandidatesExhausted;
            return None;
        };
        let selected_candidate = proposal
            .candidates
            .iter()
            .find(|candidate| candidate.worker.id == decision.selected.id)?;
        tracing::debug!(
            model = %inputs.model_str,
            policy = ?ProposalKind::CacheAffinity,
            range = %decision.candidate_range_id,
            selected = %decision.selected.url,
            cache_candidates = bounded_candidate_count,
            input_tokens = inputs.request_input_tokens,
            matched_prefix_tokens = selected_candidate.matched_prefix_tokens,
            uncached_tokens = selected_candidate.uncached_tokens,
            reason = ?decision.reason,
            load_snapshot_version = decision.load_snapshot_version,
            prefill_pressure_source = cache_decision.prefill_pressure_source,
            "cache candidate winner",
        );
        inputs
            .metrics
            .record_policy_decision("cache_aware", "cache_candidate");
        Some(decision.selected)
    }

    /// Ordered domains, first without the capacity fallback and then with it —
    /// a later domain that admits normally beats an earlier one that only the
    /// capacity fallback could serve.
    fn select_domains(
        &mut self,
        domains: &[CandidateDomain],
        affinity_lookup_enabled: bool,
        affinity_assignment_enabled: bool,
    ) -> Option<Arc<Worker>> {
        domains
            .iter()
            .find_map(|domain| {
                self.select_in_domain(
                    domain,
                    affinity_lookup_enabled,
                    affinity_assignment_enabled,
                    false,
                )
            })
            .or_else(|| {
                domains.iter().find_map(|domain| {
                    self.select_in_domain(
                        domain,
                        affinity_lookup_enabled,
                        affinity_assignment_enabled,
                        true,
                    )
                })
            })
    }

    fn select_in_domain(
        &mut self,
        domain: &CandidateDomain,
        affinity_lookup_enabled: bool,
        affinity_assignment_enabled: bool,
        allow_capacity_fallback: bool,
    ) -> Option<Arc<Worker>> {
        let inputs = self.inputs;
        let candidate_range = domain.prefill_range()?;
        let mut selection_ctx = self.base_context(candidate_range.id);
        if let Some(snapshot) = inputs.load_snapshot {
            selection_ctx = selection_ctx.with_load_snapshot(snapshot);
        }
        let selection_ctx = if !affinity_lookup_enabled {
            selection_ctx.without_affinity_lookup()
        } else if !affinity_assignment_enabled {
            selection_ctx.without_affinity_assignment()
        } else {
            selection_ctx
        };
        let Some(PrefillProposal::Pair(proposal)) = inputs
            .policy
            .propose_prefill(candidate_range.workers, &selection_ctx)
        else {
            // Domain retries are ordinary pair proposals.
            return None;
        };
        if inputs.policy.uses_shared_prefill_admission() {
            let snapshot = inputs
                .load_snapshot
                .expect("shared prefill admission requires a load snapshot");
            let decision = if allow_capacity_fallback {
                resolve_prefill(
                    &candidate_range,
                    &proposal,
                    inputs.request_input_tokens,
                    snapshot,
                )
            } else {
                resolve_prefill_admitted(
                    &candidate_range,
                    &proposal,
                    inputs.request_input_tokens,
                    snapshot,
                )
            };
            let Some(decision) = decision else {
                self.failure_reason = PolicySelectionFailureReason::PrefillAdmissionExhausted;
                return None;
            };
            let reason = prefill_policy_reason(
                inputs.policy_kind,
                proposal.kind,
                decision.reason,
                inputs.session_id.is_some_and(|value| !value.is_empty()),
                affinity_lookup_enabled,
            );
            inputs.policy.commit_prefill_selection(
                &selection_ctx,
                proposal.kind,
                &decision.selected,
            );
            inputs
                .metrics
                .record_policy_decision(&inputs.policy_kind.to_string(), reason);
            tracing::debug!(
                model = %inputs.model_str,
                policy = ?proposal.kind,
                range = %decision.candidate_range_id,
                primary = %decision.primary.url,
                backup = ?decision.backup.as_ref().map(|worker| worker.url.as_str()),
                selected = %decision.selected.url,
                reason = ?decision.reason,
                load_snapshot_version = decision.load_snapshot_version,
                "prefill policy decision",
            );
            Some(decision.selected)
        } else {
            tracing::debug!(
                model = %inputs.model_str,
                policy = ?proposal.kind,
                range = %candidate_range.id,
                selected = %proposal.primary.url,
                "prefill policy decision without shared admission",
            );
            Some(proposal.primary)
        }
    }
}

/// Return the low-cardinality reason for the final Prefill decision.
fn prefill_policy_reason(
    policy: PolicyKind,
    proposal: ProposalKind,
    decision: DecisionReason,
    has_session_id: bool,
    affinity_lookup_enabled: bool,
) -> &'static str {
    match policy {
        PolicyKind::SessionAware => match (proposal, decision) {
            (ProposalKind::SessionAffinity, DecisionReason::Primary) => "session_primary",
            (ProposalKind::SessionAffinity, DecisionReason::BackupPrimaryAdmission) => {
                "session_admission_backup"
            }
            (ProposalKind::SessionAffinity, DecisionReason::BackupPressureGuard) => {
                "session_pressure_backup"
            }
            (ProposalKind::SessionAffinity, DecisionReason::RangeFallback) => {
                "session_range_fallback"
            }
            (_, DecisionReason::CapacityFallbackPowerOfTwo) => "capacity_fallback_power_of_two",
            (_, DecisionReason::RangeFallback) => "range_fallback",
            (_, _) if !affinity_lookup_enabled => "range_fallback",
            (_, _) if !has_session_id => "no_session",
            (ProposalKind::PowerOfTwo, _) => "assigned",
            _ => "primary",
        },
        PolicyKind::CacheAware => match (proposal, decision) {
            (_, DecisionReason::CacheCandidate)
            | (ProposalKind::CacheAffinity, DecisionReason::Primary) => "cache_candidate",
            (_, DecisionReason::Primary) => "no_cache_candidate",
            (_, DecisionReason::BackupPrimaryAdmission) => "no_cache_candidate_admission_backup",
            (_, DecisionReason::BackupPressureGuard) => "no_cache_candidate_pressure_backup",
            (_, DecisionReason::RangeFallback) => "no_cache_candidate_range_fallback",
            (_, DecisionReason::CapacityFallbackPowerOfTwo) => {
                "no_cache_candidate_capacity_fallback_power_of_two"
            }
        },
        _ => match decision {
            DecisionReason::Primary => "primary",
            DecisionReason::CacheCandidate => "cache_candidate",
            DecisionReason::BackupPrimaryAdmission => "admission_backup",
            DecisionReason::BackupPressureGuard => "pressure_backup",
            DecisionReason::RangeFallback => "range_fallback",
            DecisionReason::CapacityFallbackPowerOfTwo => "capacity_fallback_power_of_two",
        },
    }
}

/// Everything one decode-peer selection reads. Collaborators first, then the
/// per-request facts.
pub(crate) struct DecodeSelectionInputs<'a> {
    pub decode_policy_kind: DecodePolicyKind,
    pub bucket_selector: &'a BucketSelector,
    /// Model name, for log lines only.
    pub model_str: &'a str,
    /// URL of the committed prefill worker; `legacy_host_affinity` pairs the
    /// decode peer against it.
    pub prefill_url: &'a str,
    pub decode_workers: &'a [Arc<Worker>],
    pub request_input_tokens: u64,
    pub requested_max_output_tokens: Option<u64>,
    pub ttft_slo_ms: Option<u64>,
    pub tps_slo: Option<f64>,
    pub load_snapshot: Option<&'a EngineLoadSnapshot>,
}

/// Runs the decode selection ladder.
///
/// Two rungs, and the order is the point: every Bucket domain is tried on the
/// strict rung before any domain is retried with the capacity fallback, so a
/// domain that fits the request is never passed over in favour of an earlier
/// domain that only fits it by relaxing capacity.
///
/// `None` means no domain yielded a peer; the route maps that onto its own
/// status code, as it does for prefill.
pub(crate) fn select_decode_peer(inputs: &DecodeSelectionInputs<'_>) -> Option<Arc<Worker>> {
    let request_kv_tokens = projected_decode_kv_tokens(
        inputs.request_input_tokens,
        inputs.requested_max_output_tokens,
    );
    // Only an explicit output budget justifies reserving peak sequence room;
    // without one the projection degenerates to the input length and would
    // bucket every request as if it decoded nothing.
    let expected_peak_sequence_tokens = inputs
        .requested_max_output_tokens
        .map(|_| request_kv_tokens);
    let decode_domains = inputs.bucket_selector.decode_domains(
        inputs.decode_workers,
        BucketRequest {
            input_tokens: inputs.request_input_tokens,
            expected_peak_sequence_tokens,
            ttft_slo_ms: inputs.ttft_slo_ms,
            tps_slo: inputs.tps_slo,
        },
    );
    let decode_policy = build_decode_policy(inputs.decode_policy_kind);
    let select_in_domain = |decode_domain: &CandidateDomain, allow_capacity_fallback: bool| {
        let snapshot = inputs.load_snapshot?;
        let decode_ctx = DecodeSelectionContext::new()
            .with_load_snapshot(snapshot)
            .with_prefill_url(inputs.prefill_url);
        let decode_proposal = decode_policy.propose(decode_domain, &decode_ctx)?;
        let decode_decision = if allow_capacity_fallback {
            resolve_decode_with_capacity_fallback(
                decode_domain,
                &decode_proposal,
                request_kv_tokens,
                snapshot,
            )
        } else {
            resolve_decode(decode_domain, &decode_proposal, request_kv_tokens, snapshot)
        }?;
        tracing::debug!(
            model = %inputs.model_str,
            policy = ?inputs.decode_policy_kind,
            range = %decode_decision.candidate_range_id,
            primary = %decode_decision.primary.url,
            backup = ?decode_decision.backup.as_ref().map(|worker| worker.url.as_str()),
            selected = %decode_decision.selected.url,
            reason = ?decode_decision.reason,
            load_snapshot_version = decode_decision.load_snapshot_version,
            "decode policy decision",
        );
        Some(decode_decision.selected)
    };
    decode_domains
        .iter()
        .find_map(|domain| select_in_domain(domain, false))
        .or_else(|| {
            decode_domains
                .iter()
                .find_map(|domain| select_in_domain(domain, true))
        })
}

/// Project the peak sequence length without integer wraparound.
fn projected_decode_kv_tokens(input_tokens: u64, max_output_tokens: Option<u64>) -> u64 {
    max_output_tokens.map_or(input_tokens, |output_tokens| {
        input_tokens.saturating_add(output_tokens)
    })
}

#[cfg(test)]
mod tests {
    use super::{prefill_policy_reason, projected_decode_kv_tokens};
    use crate::config::PolicyKind;
    use crate::policies::admission::DecisionReason;
    use crate::policies::ProposalKind;

    #[test]
    fn decode_kv_projection_includes_the_explicit_output_budget() {
        assert_eq!(projected_decode_kv_tokens(1_024, Some(512)), 1_536);
        assert_eq!(projected_decode_kv_tokens(1_024, None), 1_024);
        assert_eq!(projected_decode_kv_tokens(u64::MAX - 1, Some(8)), u64::MAX);
    }

    #[test]
    fn session_reason_distinguishes_hit_assignment_and_keyless_fallback() {
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::SessionAffinity,
                DecisionReason::Primary,
                true,
                true,
            ),
            "session_primary"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::PowerOfTwo,
                DecisionReason::Primary,
                true,
                true,
            ),
            "assigned"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::PowerOfTwo,
                DecisionReason::Primary,
                false,
                true,
            ),
            "no_session"
        );
    }

    #[test]
    fn session_reason_preserves_admission_and_pressure_escapes() {
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::SessionAffinity,
                DecisionReason::BackupPrimaryAdmission,
                true,
                true,
            ),
            "session_admission_backup"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::SessionAffinity,
                DecisionReason::BackupPressureGuard,
                true,
                true,
            ),
            "session_pressure_backup"
        );
    }

    #[test]
    fn cache_no_winner_p2_is_distinct_from_cache_candidate() {
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::CacheAware,
                ProposalKind::PowerOfTwo,
                DecisionReason::Primary,
                false,
                false,
            ),
            "no_cache_candidate"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::CacheAware,
                ProposalKind::CacheAffinity,
                DecisionReason::Primary,
                false,
                true,
            ),
            "cache_candidate"
        );
    }
}
