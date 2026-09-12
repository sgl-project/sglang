// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Worker selection: the two ladders that turn a policy proposal into a
//! committed worker — [`select_prefill_worker`] and [`select_decode_peer`].
//!
//! WHY this is a module rather than a block inside the chat handler: each
//! ladder has several rungs — cache-candidate resolution, the global
//! session-affinity probe, per-domain admission, the capacity fallback — and
//! each rung fails into the next. Rungs get added over time, and the
//! assertions worth writing are almost always about a ladder as a whole ("a
//! saturated fleet still routes", "a sampled choice never lands on a rejected
//! worker"), not about one rung in isolation. Written as closures inside an
//! HTTP handler those assertions can only be expressed as end-to-end HTTP
//! tests; written here they are unit tests.
//!
//! Both ladders live here rather than one per module because `CandidateDomain`
//! already carries `stage: RoutingStage`: prefill and decode are two
//! configurations of one idea, so a rung added to one and not the other has to
//! be visible on one screen.
//!
//! The module owns the decision and reports why; it does not own the HTTP
//! response. Mapping a failed selection onto a status code stays in the route.

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
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
use crate::server::metrics::{CacheAwareDecision, MetricsRegistry, PolicySelectionFailureReason};
use crate::workers::Worker;

/// Everything one prefill selection reads. Collaborators first, then the
/// per-request facts.
pub(crate) struct PrefillSelectionInputs<'a> {
    pub policy: &'a dyn Policy,
    pub policy_kind: PolicyKind,
    pub bucket_selector: &'a BucketSelector,
    pub metrics: &'a MetricsRegistry,
    /// Names the model in the selection context and in the log lines.
    pub model_id: &'a ModelId,
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
    pub ttft_slo_ms: Option<u64>,
    pub tps_slo: Option<f64>,
    /// The configured mode. Without Bucket partitioning all modes reduce to
    /// the single global domain, and the ladder applies that reduction itself.
    pub session_affinity_mode: SessionAffinityMode,
    /// `--worker-queue-limit`. `None` disables the queue gate entirely.
    pub worker_queue_limit: Option<u64>,
    /// `--min-load-choices`: sample size for the min-load capacity fallback.
    pub min_load_choices: usize,
}

/// The queue-gate blind warn is sampled: it fires on a per-request path, and
/// the condition (gate configured, zero fresh engine load samples fleet-wide)
/// is steady-state, so 1-in-64 is plenty to surface it without log flooding.
const QUEUE_GATE_BLIND_LOG_SAMPLE: u64 = 64;
static QUEUE_GATE_BLIND_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Maps the queue-gate audit of a Cache-Aware selection that produced no
/// winner onto its decision label. Pure so both boundaries are pinned by
/// unit tests rather than inferred from the ladder that calls it:
///
/// - `admission_evaluated == 0` is what makes the GATE, rather than KV
///   capacity, the reason the candidate set came back empty. Without it a
///   capacity exhaustion books as a gate diversion whenever one owner
///   happens to be queueing.
/// - `fleet_all_queued` separates a real diversion (somewhere unqueued to go)
///   from fleet saturation (nowhere to go, and the prefix was kept).
fn cache_aware_fallback_decision(
    queue_gate_rejected: u64,
    admission_evaluated: u64,
    fleet_all_queued: bool,
) -> CacheAwareDecision {
    if queue_gate_rejected == 0 || admission_evaluated > 0 {
        // Capacity, not the gate: either nothing was gated out, or owners
        // survived the gate and then failed capacity admission.
        return CacheAwareDecision::CacheMiss;
    }
    if fleet_all_queued {
        CacheAwareDecision::AllQueued
    } else {
        CacheAwareDecision::CacheWorkerQueued
    }
}

/// Runs the prefill selection ladder. `Err` carries the reason the last rung
/// to record one gave up with; the route maps it onto a status code.
pub(crate) fn select_prefill_worker(
    inputs: &PrefillSelectionInputs<'_>,
) -> Result<Arc<Worker>, PolicySelectionFailureReason> {
    let mut selector = Selector {
        inputs,
        // Prefill reserves no peak sequence room: the decode peer, not the
        // prefill worker, holds the KV for the tokens still to be generated.
        bucket_request: BucketRequest {
            input_tokens: inputs.request_input_tokens,
            expected_peak_sequence_tokens: None,
            ttft_slo_ms: inputs.ttft_slo_ms,
            tps_slo: inputs.tps_slo,
        },
        failure_reason: PolicySelectionFailureReason::ProposalEmpty,
        cache_gate_audit: None,
    };
    let selected = selector.run();
    selected.ok_or(selector.failure_reason)
}

/// Carries the per-request bucket request and the failure reason across rungs.
/// Each rung that gives up overwrites the reason, so the reported one is what
/// the last rung to record any gave — a rung that returns `None` without
/// recording leaves the previous reason standing.
struct Selector<'a> {
    inputs: &'a PrefillSelectionInputs<'a>,
    bucket_request: BucketRequest,
    failure_reason: PolicySelectionFailureReason,
    /// Queue-gate audit of a Cache-Aware resolution that produced no winner:
    /// (gate-rejected candidates, candidates that reached capacity admission,
    /// fleet saturation, deepest rejected prefix). `None` when there was no
    /// resolution at all — no load snapshot, or no candidate proposal — which
    /// reads as a plain miss because nothing was gated out.
    cache_gate_audit: Option<(u64, u64, bool, u32)>,
}

impl<'a> Selector<'a> {
    fn run(&mut self) -> Option<Arc<Worker>> {
        let inputs = self.inputs;
        let bucket_request = self.bucket_request;
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
        let cache_winner_hit = cache_winner.is_some();

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
                    bucket_request,
                )
            })
            // Rebuild the backup inside the primary's own Bucket.
            .and_then(|domain| self.select_in_domain(&domain, true, false, false));

        let selected = cache_winner.or_else(|| {
            // Materializing the normal domains clones the member list of every
            // Bucket, so build them only on the rung that actually reads them.
            let prefill_domains = || {
                inputs
                    .bucket_selector
                    .prefill_domains(inputs.workers, bucket_request)
            };
            if inputs.policy_kind == PolicyKind::CacheAware {
                // Cache miss or failure retries ordered domains with ordinary P2.
                return self.select_domains(&prefill_domains(), false, false);
            }
            if let Some(worker) = global_affinity_worker {
                return Some(worker);
            }
            match session_affinity_mode {
                SessionAffinityMode::GlobalPreserve if global_affinity_missed => {
                    self.select_domains(&prefill_domains(), true, true)
                }
                SessionAffinityMode::GlobalPreserve => {
                    self.select_domains(&prefill_domains(), false, false)
                }
                SessionAffinityMode::Bucket | SessionAffinityMode::GlobalRebind => {
                    self.select_domains(&prefill_domains(), true, true)
                }
            }
        });
        if inputs.policy_kind == PolicyKind::CacheAware && !cache_winner_hit {
            let (rejected, evaluated, fleet_all_queued, blocks) =
                self.cache_gate_audit.unwrap_or((0, 0, false, 0));
            let decision = cache_aware_fallback_decision(rejected, evaluated, fleet_all_queued);
            if matches!(decision, CacheAwareDecision::CacheWorkerQueued) {
                // A real diversion: an unqueued destination existed and the
                // gate gave up `blocks` of matched prefix to reach it. The
                // histogram is the evidence for whether the gate is trading
                // large cached prefixes for short waits.
                inputs
                    .metrics
                    .observe_diverted_overlap_blocks(&inputs.model_id.0, u64::from(blocks));
            }
            inputs
                .metrics
                .record_cache_aware_decision(&inputs.model_id.0, decision);
        }
        selected
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
        let bucket_request = self.bucket_request;
        if inputs.policy_kind != PolicyKind::CacheAware {
            return None;
        }
        let snapshot = inputs.load_snapshot?;
        let global_range = CandidateRange::global(inputs.workers);
        let cache_ctx = self
            .base_context(global_range.id)
            .with_load_snapshot(snapshot)
            .with_prefill_cache_bucket(inputs.bucket_selector, bucket_request);
        let PrefillProposal::CacheCandidates(proposal) = inputs
            .policy
            .propose_prefill(global_range.workers, &cache_ctx)?
        else {
            return None;
        };
        let bounded_candidate_count = proposal.candidates.len();
        // The queue gate reads the engine-published load sample and fails open
        // per worker. When NO worker has a fresh sample the gate is inert
        // fleet-wide and nothing would say so: `cache_worker_queued` sitting at
        // 0 is indistinguishable from a healthy fleet. Warn (sampled) — a fleet
        // that never advertised a load port must not silently disable the gate.
        if inputs.worker_queue_limit.is_some()
            && !inputs.workers.is_empty()
            && inputs
                .workers
                .iter()
                .all(|worker| snapshot.fresh_load_for_url(&worker.url).is_none())
            && QUEUE_GATE_BLIND_LOG_COUNTER
                .fetch_add(1, AtomicOrdering::Relaxed)
                .is_multiple_of(QUEUE_GATE_BLIND_LOG_SAMPLE)
        {
            tracing::warn!(
                model = %inputs.model_id,
                worker_queue_limit = inputs.worker_queue_limit,
                workers = inputs.workers.len(),
                "--worker-queue-limit is set but no worker has a fresh engine load \
                 sample, so the queue gate is inert. Check that engines advertise a \
                 load port and publish LoadStat",
            );
        }
        let cache_decision = resolve_cache_candidates(
            &proposal,
            inputs.request_input_tokens,
            snapshot,
            inputs.workers,
        );
        self.cache_gate_audit = Some((
            cache_decision.queue_gate_rejected_candidates,
            cache_decision.admission_evaluated_candidates,
            cache_decision.fleet_all_queued,
            cache_decision.queue_gate_best_rejected_blocks,
        ));
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
            model = %inputs.model_id,
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
        inputs.metrics.record_cache_aware_decision(
            &inputs.model_id.0,
            if cache_decision.queue_gate_fell_back {
                // The gate removed every owner and nowhere in the fleet is
                // unqueued, so the prefix was kept rather than traded for a
                // wait that cannot be dodged. Booked as saturation, never as
                // a plain hit.
                CacheAwareDecision::AllQueued
            } else {
                CacheAwareDecision::CacheHit
            },
        );
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
                    inputs.worker_queue_limit,
                    inputs.min_load_choices,
                )
            } else {
                resolve_prefill_admitted(
                    &candidate_range,
                    &proposal,
                    inputs.request_input_tokens,
                    snapshot,
                    inputs.worker_queue_limit,
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
                model = %inputs.model_id,
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
                model = %inputs.model_id,
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
    /// Names the model in the log lines.
    pub model_id: &'a ModelId,
    /// URL of the committed prefill worker; `legacy_host_affinity` pairs the
    /// decode peer against it.
    pub prefill_url: &'a str,
    pub decode_workers: &'a [Arc<Worker>],
    pub request_input_tokens: u64,
    pub requested_max_output_tokens: Option<u64>,
    pub ttft_slo_ms: Option<u64>,
    pub tps_slo: Option<f64>,
    /// Required: every rung resolves its proposal against the snapshot, so
    /// without one the ladder reports no peer at all rather than picking one
    /// blind.
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
    // Every rung resolves against the snapshot, so without one no domain can
    // yield a peer and there is nothing worth building.
    let snapshot = inputs.load_snapshot?;
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
    let decode_ctx = DecodeSelectionContext::new()
        .with_load_snapshot(snapshot)
        .with_prefill_url(inputs.prefill_url);
    let select_in_domain = |decode_domain: &CandidateDomain, allow_capacity_fallback: bool| {
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
            model = %inputs.model_id,
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
    use super::{
        cache_aware_fallback_decision, prefill_policy_reason, projected_decode_kv_tokens,
        select_decode_peer, select_prefill_worker, DecodeSelectionInputs, PrefillSelectionInputs,
    };
    use crate::config::{DecodePolicyKind, PolicyKind, SessionAffinityMode};
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::admission::{resolve_prefill_admitted, CandidateRange, DecisionReason};
    use crate::policies::buckets::BucketSelector;
    use crate::policies::engine_load::{EngineLoadSnapshot, NativeCacheWorkerLoad};
    use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
    use crate::policies::{Policy, ProposalKind, SelectionProposal};
    use crate::server::metrics::{
        CacheAwareDecision, MetricsRegistry, PolicySelectionFailureReason,
    };
    use crate::workers::Worker;
    use std::sync::Arc;
    use std::time::Instant;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: None,
        }))
    }

    /// `(worker, tokens already held, published KV capacity)`.
    fn snapshot(entries: &[(&Arc<Worker>, u64, u64)]) -> EngineLoadSnapshot {
        EngineLoadSnapshot::from_native_cache_workers(
            7,
            entries
                .iter()
                .map(|(worker, used, capacity)| {
                    (
                        worker.url.clone(),
                        NativeCacheWorkerLoad {
                            num_running_reqs: 0,
                            num_waiting_reqs: 0,
                            num_waiting_uncached_tokens: 0,
                            num_used_tokens: *used,
                            num_total_tokens: *used,
                            max_total_num_tokens: *capacity,
                            max_running_requests: 64,
                            prefill_throughput_tokens_per_s: None,
                            estimated_prefill_queue_ms: None,
                            captured_at: Instant::now(),
                        },
                    )
                })
                .collect(),
        )
    }

    /// Power-of-two prefill, Bucket partitioning off, no session affinity —
    /// the single global domain, which is what isolates the ladder's rungs.
    #[allow(clippy::too_many_arguments)]
    fn prefill_inputs<'a>(
        policy: &'a dyn Policy,
        bucket_selector: &'a BucketSelector,
        metrics: &'a MetricsRegistry,
        model_id: &'a ModelId,
        workers: &'a [Arc<Worker>],
        load_snapshot: Option<&'a EngineLoadSnapshot>,
        request_input_tokens: u64,
    ) -> PrefillSelectionInputs<'a> {
        PrefillSelectionInputs {
            policy,
            policy_kind: PolicyKind::PowerOfTwo,
            bucket_selector,
            metrics,
            model_id,
            body: None,
            routing_key: None,
            session_id: None,
            request_input_tokens,
            request_tokens: None,
            external_prefix: None,
            load_snapshot,
            workers,
            worker_queue_limit: None,
            min_load_choices: 2,
            ttft_slo_ms: None,
            tps_slo: None,
            session_affinity_mode: SessionAffinityMode::Bucket,
        }
    }

    fn decode_inputs<'a>(
        bucket_selector: &'a BucketSelector,
        model_id: &'a ModelId,
        decode_workers: &'a [Arc<Worker>],
        load_snapshot: Option<&'a EngineLoadSnapshot>,
        request_input_tokens: u64,
    ) -> DecodeSelectionInputs<'a> {
        DecodeSelectionInputs {
            decode_policy_kind: DecodePolicyKind::PowerOfTwo,
            bucket_selector,
            model_id,
            prefill_url: "http://prefill:30000",
            decode_workers,
            request_input_tokens,
            requested_max_output_tokens: None,
            ttft_slo_ms: None,
            tps_slo: None,
            load_snapshot,
        }
    }

    #[test]
    fn an_empty_fleet_reports_the_proposal_empty_failure() {
        let policy = PowerOfTwoChoicesPolicy::new();
        let buckets = BucketSelector::new(None);
        let metrics = MetricsRegistry::new();
        let model = ModelId("model".into());
        let workers: Vec<Arc<Worker>> = Vec::new();
        let loads = snapshot(&[]);

        let outcome = select_prefill_worker(&prefill_inputs(
            &policy,
            &buckets,
            &metrics,
            &model,
            &workers,
            Some(&loads),
            64,
        ));

        assert!(matches!(
            outcome,
            Err(PolicySelectionFailureReason::ProposalEmpty)
        ));
    }

    #[test]
    fn a_saturated_fleet_still_routes_through_the_capacity_fallback() {
        let full = worker("full");
        let also_full = worker("also-full");
        let workers = vec![Arc::clone(&full), Arc::clone(&also_full)];
        let loads = snapshot(&[(&full, 100, 100), (&also_full, 100, 100)]);

        // Without this the test would pass even if the strict rung had served
        // the request, and would prove nothing about the fallback rung.
        let range = CandidateRange::global(&workers);
        assert!(
            resolve_prefill_admitted(
                &range,
                &SelectionProposal::with_backup(Arc::clone(&full), Arc::clone(&also_full)),
                64,
                &loads,
                None,
            )
            .is_none(),
            "fixture must saturate every worker so the strict rung admits none",
        );

        let policy = PowerOfTwoChoicesPolicy::new();
        let buckets = BucketSelector::new(None);
        let metrics = MetricsRegistry::new();
        let model = ModelId("model".into());

        let selected = select_prefill_worker(&prefill_inputs(
            &policy,
            &buckets,
            &metrics,
            &model,
            &workers,
            Some(&loads),
            64,
        ))
        .expect("the capacity fallback must still place the request");
        assert!(selected.id == full.id || selected.id == also_full.id);
    }

    #[test]
    fn a_sampled_choice_never_lands_on_a_rejected_worker() {
        let full = worker("full");
        let roomy = worker("roomy");
        let workers = vec![Arc::clone(&full), Arc::clone(&roomy)];
        let loads = snapshot(&[(&full, 100, 100), (&roomy, 0, 100_000)]);
        let policy = PowerOfTwoChoicesPolicy::new();
        let buckets = BucketSelector::new(None);
        let metrics = MetricsRegistry::new();
        let model = ModelId("model".into());

        // Power-of-two samples its pair at random, so one pass proves nothing
        // about which rung answered.
        for _ in 0..32 {
            let selected = select_prefill_worker(&prefill_inputs(
                &policy,
                &buckets,
                &metrics,
                &model,
                &workers,
                Some(&loads),
                64,
            ))
            .expect("one worker has room");
            assert_eq!(selected.id, roomy.id);
        }
    }

    #[test]
    fn the_decode_ladder_reports_no_peer_without_a_load_snapshot() {
        let peer = worker("decode");
        let workers = vec![Arc::clone(&peer)];
        let buckets = BucketSelector::new(None);
        let model = ModelId("model".into());

        assert!(select_decode_peer(&decode_inputs(&buckets, &model, &workers, None, 64)).is_none());
        assert!(
            select_decode_peer(&decode_inputs(
                &buckets,
                &model,
                &workers,
                Some(&snapshot(&[(&peer, 0, 100_000)])),
                64,
            ))
            .is_some(),
            "the same fleet routes as soon as a snapshot is available",
        );
    }

    #[test]
    fn a_saturated_decode_fleet_still_routes_through_the_capacity_fallback() {
        let full = worker("decode-full");
        let also_full = worker("decode-also-full");
        let workers = vec![Arc::clone(&full), Arc::clone(&also_full)];
        let loads = snapshot(&[(&full, 100, 100), (&also_full, 100, 100)]);
        let buckets = BucketSelector::new(None);
        let model = ModelId("model".into());

        let selected =
            select_decode_peer(&decode_inputs(&buckets, &model, &workers, Some(&loads), 64))
                .expect("the capacity fallback must still place the decode peer");
        assert!(selected.id == full.id || selected.id == also_full.id);
    }

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

    #[test]
    fn cache_aware_fallback_decision_needs_the_gate_to_have_emptied_the_set() {
        // The trap this pins: one owner queueing while the others exhaust KV
        // capacity is a CAPACITY problem, not a gate diversion. Only a gate
        // that removed every owner leaves zero candidates evaluated.
        assert!(matches!(
            cache_aware_fallback_decision(1, 3, false),
            CacheAwareDecision::CacheMiss
        ));
        assert!(matches!(
            cache_aware_fallback_decision(1, 3, true),
            CacheAwareDecision::CacheMiss
        ));
        // Nothing gated out at all: a plain miss.
        assert!(matches!(
            cache_aware_fallback_decision(0, 0, false),
            CacheAwareDecision::CacheMiss
        ));
        assert!(matches!(
            cache_aware_fallback_decision(0, 4, true),
            CacheAwareDecision::CacheMiss
        ));
    }

    #[test]
    fn cache_aware_fallback_decision_separates_diversion_from_saturation() {
        // Gate removed every owner and somewhere unqueued exists: a real
        // diversion off the prefix.
        assert!(matches!(
            cache_aware_fallback_decision(2, 0, false),
            CacheAwareDecision::CacheWorkerQueued
        ));
        // Gate removed every owner and nowhere is unqueued: saturation.
        assert!(matches!(
            cache_aware_fallback_decision(2, 0, true),
            CacheAwareDecision::AllQueued
        ));
    }
}
