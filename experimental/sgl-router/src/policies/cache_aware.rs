// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware routing: candidate preparation, queue and capacity evaluation,
//! cache/pressure tournament, saturation recovery, and decision reporting.

use crate::config::AffinityConfig;
use crate::discovery::ModelId;
use crate::policies::admission::{
    fleet_is_all_queued, has_kv_capacity, materially_more_pressured, queue_gate_admits,
    DecisionReason, FinalDecision, FreshLoadLookup,
};
use crate::policies::balancing::PowerOfTwoChoicesPolicy;
use crate::policies::{
    Policy, PrefillEvaluation, ProposalKind, SelectionContext, SelectionProposal,
};
use crate::server::metrics::{CacheAwareDecision, MetricsRegistry};
use crate::workers::engine_reports::EngineSnapshot;
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Cache-Aware prefill candidate where `E = L - H`.
#[derive(Clone)]
pub struct CacheCandidate {
    pub worker: Arc<Worker>,
    pub matched_prefix_tokens: u64,
    pub uncached_tokens: u64,
    /// Matched prefix length in blocks, as reported by the prefix signal.
    /// Selection reads `matched_prefix_tokens`; the block count exists for
    /// observability (the diverted-overlap histogram reads against the
    /// tree/indexer block domain).
    pub matched_prefix_blocks: u32,
    /// Domain containing this candidate.
    pub candidate_range_id: String,
    /// Optional pending prefill limit checked against `E`.
    pub max_pending_prefill_tokens: Option<u64>,
}

/// Bounded cache observations, before queue and capacity evaluation.
#[derive(Clone, Default)]
pub struct CacheCandidateProposal {
    pub candidates: Vec<CacheCandidate>,
}

/// Cache-aware evaluation, including the audit needed if the workflow falls back.
#[derive(Clone)]
pub struct CacheSelection {
    pub candidates: Vec<CacheCandidate>,
    pub resolution: CacheCandidateResolution,
}

/// Cache-Aware selection audit data. These fields do not affect selection.
#[derive(Clone)]
pub struct CacheCandidateResolution {
    pub decision: Option<FinalDecision>,
    pub prefill_pressure_source: &'static str,
    /// Candidates actually put through KV-capacity / pending-prefill
    /// admission. Queue-gate rejections are excluded: the gate runs first and
    /// they are never evaluated, so counting them here would silently deflate
    /// the rejected/evaluated ratio whenever the gate is armed.
    pub admission_evaluated_candidates: u64,
    /// Candidates rejected by KV-capacity / pending-prefill admission. Does
    /// not include queue-gate rejections — those are counted separately so a
    /// busy fleet never reads as a capacity problem.
    pub admission_rejected_candidates: u64,
    /// Candidates rejected by the queue gate: their engine queue is at or
    /// over `--worker-queue-limit`. The gate runs BEFORE hard admission, so
    /// these candidates were never evaluated against capacity and are
    /// disjoint from `admission_rejected_candidates` by construction.
    pub queue_gate_rejected_candidates: u64,
    /// Deepest matched prefix, in blocks, among the queue-gate-rejected
    /// candidates — the locality a diversion gave up. 0 when the gate
    /// rejected nothing.
    pub queue_gate_best_rejected_blocks: u32,
    /// True when the gate removed EVERY candidate and nowhere in the fleet is
    /// unqueued, so the winner came from the ungated candidate set instead.
    /// The mirror of `range_fallback`'s second tier: diversion buys nothing
    /// here, so discarding the prefix would be pure loss.
    pub queue_gate_fell_back: bool,
    /// True when no worker in the fleet has room before the gate. Computed
    /// once here so the route handler does not re-derive the gate over the
    /// fleet; always false when the gate is disabled.
    pub fleet_all_queued: bool,
    pub pressure_guard_compared_pairs: u64,
    pub pressure_guard_overrides: u64,
}

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
        Some(CacheCandidateProposal { candidates })
    }

    /// Resolves prepared candidates with this policy's queue, capacity, and pressure rules.
    pub fn evaluate_candidates(
        &self,
        proposal: CacheCandidateProposal,
        request_input_tokens: u64,
        snapshot: &EngineSnapshot,
        fleet: &[Arc<Worker>],
    ) -> CacheSelection {
        let resolution = resolve_cache_candidates(
            &proposal.candidates,
            &self.config,
            request_input_tokens,
            snapshot,
            fleet,
        );
        CacheSelection {
            candidates: proposal.candidates,
            resolution,
        }
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
        .then_with(|| loads.compare_prefill_pressure_then_id(&left.worker, &right.worker))
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
        if ctx.affinity_lookup_enabled() {
            if let Some(proposal) = self.cache_candidate_proposal(workers, ctx) {
                let candidate = proposal.candidates.into_iter().next()?;
                return Some(
                    SelectionProposal::primary(candidate.worker)
                        .with_kind(ProposalKind::CacheAffinity),
                );
            }
        }
        PowerOfTwoChoicesPolicy::new().propose(workers, ctx)
    }

    fn evaluate_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillEvaluation> {
        if ctx.affinity_lookup_enabled() {
            if let Some(proposal) = self.cache_candidate_proposal(workers, ctx) {
                let empty = EngineSnapshot::default();
                let fleet = ctx.routable_fleet().unwrap_or(workers);
                let snapshot = ctx.load_snapshot().unwrap_or(&empty);
                // The queue gate reads the engine-published load sample and fails open
                // per worker. When NO worker has a fresh sample the gate is inert
                // fleet-wide and nothing would say so: `cache_worker_queued` sitting at
                // 0 is indistinguishable from a healthy fleet. Warn (sampled) — a fleet
                // that never advertised a load port must not silently disable the gate.
                if self.config.worker_queue_limit.is_some()
                    && !fleet.is_empty()
                    && fleet
                        .iter()
                        .all(|worker| snapshot.fresh_load_for_url(&worker.url).is_none())
                    && QUEUE_GATE_BLIND_LOG_COUNTER
                        .fetch_add(1, AtomicOrdering::Relaxed)
                        .is_multiple_of(QUEUE_GATE_BLIND_LOG_SAMPLE)
                {
                    tracing::warn!(
                        model = %ctx.model(),
                        worker_queue_limit = self.config.worker_queue_limit,
                        workers = fleet.len(),
                        "--worker-queue-limit is set but no worker has a fresh engine load \
                         sample, so the queue gate is inert. Check that engines advertise a \
                         load port and publish LoadStat",
                    );
                }
                let selection =
                    self.evaluate_candidates(proposal, ctx.input_tokens()?, snapshot, fleet);
                if let Some(decision) = &selection.resolution.decision {
                    if decision.reason == DecisionReason::SaturationPin
                        && SATURATION_PIN_LOG_COUNTER
                            .fetch_add(1, AtomicOrdering::Relaxed)
                            .is_multiple_of(SATURATION_PIN_LOG_SAMPLE)
                    {
                        tracing::info!(
                            model = %&ctx.model().0,
                            worker = %decision.selected.url,
                            saturation_queue_floor = self.config.saturation_queue_floor,
                            worker_queue_limit = self.config.worker_queue_limit,
                            "fleet saturated, keeping affinity with a queueing prefix owner \
                             instead of diverting",
                        );
                    }
                }
                return Some(PrefillEvaluation::Cache(selection));
            }
        }
        PowerOfTwoChoicesPolicy::new()
            .propose(workers, ctx)
            .map(PrefillEvaluation::Pair)
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

/// Resolve against the healthy prefill fleet, not the router-wide engine table.
/// Queue saturation must exclude other models and decode-only workers.
fn resolve_cache_candidates(
    candidates: &[CacheCandidate],
    config: &AffinityConfig,
    request_input_tokens: u64,
    snapshot: &EngineSnapshot,
    fleet: &[Arc<Worker>],
) -> CacheCandidateResolution {
    let queue_limit = config.worker_queue_limit;
    let fleet_all_queued = fleet_is_all_queued(snapshot, fleet, queue_limit);
    // Queue rejections and capacity rejections have disjoint audit counters.
    let mut evaluated: Vec<&CacheCandidate> = Vec::with_capacity(candidates.len());
    // Kept, not just counted: the saturation pin ranks these by pressure when
    // it fires. Stays unallocated while the gate is disabled, because nothing
    // is ever rejected then.
    let mut queue_gate_rejected: Vec<&CacheCandidate> = Vec::new();
    let mut queue_gate_best_rejected_blocks = 0u32;
    for candidate in candidates {
        if queue_gate_admits(snapshot, &candidate.worker, queue_limit) {
            evaluated.push(candidate);
        } else {
            queue_gate_rejected.push(candidate);
            queue_gate_best_rejected_blocks =
                queue_gate_best_rejected_blocks.max(candidate.matched_prefix_blocks);
        }
    }
    let queue_gate_rejected_candidates = queue_gate_rejected.len() as u64;
    // Without an explicit saturation floor, restore all owners only when every
    // fleet worker is provably queued. A configured floor uses pressure-only pinning.
    let queue_gate_fell_back = evaluated.is_empty()
        && queue_gate_rejected_candidates > 0
        && fleet_all_queued
        && config.saturation_queue_floor.is_none();
    if queue_gate_fell_back {
        evaluated.extend(candidates.iter());
    }
    // Built over the candidates that actually reach admission: a gated-out
    // candidate missing native monitor data would otherwise break the
    // lookup's full-coverage check and silently downgrade every pressure
    // comparison (and `prefill_pressure_source`) to router-local.
    let loads = FreshLoadLookup::new(
        Some(snapshot),
        evaluated.iter().copied().map(|candidate| &candidate.worker),
    );
    let admission_evaluated_candidates = evaluated.len() as u64;
    let admitted: Vec<&CacheCandidate> = evaluated
        .into_iter()
        .filter(|candidate| is_cache_candidate_admitted(candidate, request_input_tokens, &loads))
        .collect();
    let admission_rejected_candidates =
        admission_evaluated_candidates.saturating_sub(admitted.len() as u64);
    let Some(work_floor) = admitted
        .iter()
        .copied()
        .min_by_key(|candidate| candidate.uncached_tokens)
    else {
        let pinned = saturation_pin(
            &queue_gate_rejected,
            config.saturation_queue_floor,
            request_input_tokens,
            snapshot,
            fleet,
        );
        return CacheCandidateResolution {
            decision: pinned.map(|pinned| FinalDecision {
                selected: Arc::clone(&pinned.worker),
                primary: Arc::clone(&pinned.worker),
                backup: None,
                reason: DecisionReason::SaturationPin,
                candidate_range_id: pinned.candidate_range_id.clone(),
                load_snapshot_version: snapshot.version,
            }),
            prefill_pressure_source: loads.prefill_pressure_source(),
            admission_evaluated_candidates,
            admission_rejected_candidates,
            queue_gate_rejected_candidates,
            queue_gate_best_rejected_blocks,
            queue_gate_fell_back,
            fleet_all_queued,
            pressure_guard_compared_pairs: 0,
            pressure_guard_overrides: 0,
        };
    };
    let (winner, pressure_guard_compared_pairs, pressure_guard_overrides) =
        cache_tournament(work_floor, admitted, config, &loads);
    CacheCandidateResolution {
        decision: Some(FinalDecision {
            selected: Arc::clone(&winner.worker),
            primary: Arc::clone(&winner.worker),
            backup: None,
            reason: DecisionReason::CacheCandidate,
            candidate_range_id: winner.candidate_range_id.clone(),
            load_snapshot_version: snapshot.version,
        }),
        prefill_pressure_source: loads.prefill_pressure_source(),
        admission_evaluated_candidates,
        admission_rejected_candidates,
        queue_gate_rejected_candidates,
        queue_gate_best_rejected_blocks,
        queue_gate_fell_back,
        fleet_all_queued,
        pressure_guard_compared_pairs,
        pressure_guard_overrides,
    }
}

/// Saturation relaxes the queue gate, never the capacity checks.
fn saturation_pin<'a>(
    rejected: &[&'a CacheCandidate],
    floor: Option<u64>,
    input_tokens: u64,
    snapshot: &EngineSnapshot,
    fleet: &[Arc<Worker>],
) -> Option<&'a CacheCandidate> {
    let floor = floor?;
    if rejected.is_empty()
        || snapshot.any_fresh_queue_below(fleet.iter().map(|worker| worker.url.as_str()), floor)
    {
        return None;
    }
    // Compare the rejected owners using their own coverage, not the empty admitted pool.
    let loads = FreshLoadLookup::new(
        Some(snapshot),
        rejected.iter().map(|candidate| &candidate.worker),
    );
    rejected
        .iter()
        .copied()
        .filter(|candidate| is_cache_candidate_admitted(candidate, input_tokens, &loads))
        .min_by(|left, right| loads.compare_prefill_pressure_then_id(&left.worker, &right.worker))
}

/// Preserve candidate order: threshold-based pressure overrides are not transitive.
fn cache_tournament<'a>(
    work_floor: &'a CacheCandidate,
    admitted: Vec<&'a CacheCandidate>,
    config: &AffinityConfig,
    loads: &FreshLoadLookup<'_>,
) -> (&'a CacheCandidate, u64, u64) {
    let near_tie_ceiling = work_floor
        .uncached_tokens
        .saturating_add(config.cache_switch_margin_tokens);
    let mut winner = work_floor;
    let mut pressure_guard_compared_pairs = 0;
    let mut pressure_guard_overrides = 0;
    for candidate in admitted {
        if candidate.worker.id == winner.worker.id || candidate.uncached_tokens > near_tie_ceiling {
            continue;
        }
        let baseline = compare_cache_candidates(winner, candidate, config, loads, false);
        let ordering =
            if config.pressure_guard && cache_pressure_guard_comparable(winner, candidate, loads) {
                pressure_guard_compared_pairs += 1;
                let guarded = compare_cache_candidates(winner, candidate, config, loads, true);
                if guarded != baseline {
                    pressure_guard_overrides += 1;
                }
                guarded
            } else {
                baseline
            };
        if ordering.is_gt() {
            winner = candidate;
        }
    }
    (
        winner,
        pressure_guard_compared_pairs,
        pressure_guard_overrides,
    )
}

fn is_cache_candidate_admitted(
    candidate: &CacheCandidate,
    request_input_tokens: u64,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    let Some(load) = loads.get(&candidate.worker.id) else {
        return true;
    };
    has_kv_capacity(Some(load), request_input_tokens)
        && candidate.max_pending_prefill_tokens.is_none_or(|limit| {
            load.num_waiting_uncached_tokens
                .saturating_add(candidate.uncached_tokens)
                <= limit
        })
}

fn compare_cache_candidates(
    left: &CacheCandidate,
    right: &CacheCandidate,
    config: &AffinityConfig,
    loads: &FreshLoadLookup<'_>,
    enable_pressure_guard: bool,
) -> Ordering {
    let work_delta = left.uncached_tokens.abs_diff(right.uncached_tokens);
    if work_delta > config.cache_switch_margin_tokens {
        return left
            .uncached_tokens
            .cmp(&right.uncached_tokens)
            .then_with(|| loads.compare_prefill_pressure_then_id(&left.worker, &right.worker));
    }
    if enable_pressure_guard {
        if materially_more_pressured(
            &left.worker,
            &right.worker,
            config.pressure_abs_threshold_tokens,
            config.pressure_abs_threshold_ms,
            config.pressure_rel_threshold,
            loads,
        ) {
            return Ordering::Greater;
        }
        if materially_more_pressured(
            &right.worker,
            &left.worker,
            config.pressure_abs_threshold_tokens,
            config.pressure_abs_threshold_ms,
            config.pressure_rel_threshold,
            loads,
        ) {
            return Ordering::Less;
        }
    }
    left.uncached_tokens
        .cmp(&right.uncached_tokens)
        .then_with(|| loads.compare_prefill_pressure_then_id(&left.worker, &right.worker))
}

fn cache_pressure_guard_comparable(
    left: &CacheCandidate,
    right: &CacheCandidate,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    loads.comparable_get(&left.worker.id).is_some()
        && loads.comparable_get(&right.worker.id).is_some()
}

/// The queue-gate blind warn is sampled: it fires on a per-request path, and
/// the condition (gate configured, zero fresh engine load samples fleet-wide)
/// is steady-state, so 1-in-64 is plenty to surface it without log flooding.
const QUEUE_GATE_BLIND_LOG_SAMPLE: u64 = 64;
static QUEUE_GATE_BLIND_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The saturation-pin info log is sampled for the same reason as the
/// queue-gate blind warn: it fires on a per-request path and the condition
/// (a saturated fleet) persists for many requests, so 1-in-64 surfaces it
/// without log flooding.
const SATURATION_PIN_LOG_SAMPLE: u64 = 64;
static SATURATION_PIN_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Classify saturation before capacity exhaustion so a full fleet remains visible.
fn cache_aware_fallback_decision(
    queue_gate_rejected: u64,
    admission_evaluated: u64,
    fleet_all_queued: bool,
) -> CacheAwareDecision {
    if queue_gate_rejected == 0 {
        return CacheAwareDecision::CacheMiss;
    }
    if fleet_all_queued {
        return CacheAwareDecision::AllQueued;
    }
    if admission_evaluated > 0 {
        // Capacity, not the gate: owners survived the gate and then failed
        // capacity admission.
        return CacheAwareDecision::CacheMiss;
    }
    CacheAwareDecision::CacheWorkerQueued
}

impl CacheSelection {
    pub(crate) fn record_selection(
        &self,
        metrics: &MetricsRegistry,
        model_id: &ModelId,
        input_tokens: u64,
    ) -> Option<Arc<Worker>> {
        let cache_decision = &self.resolution;
        metrics.record_cache_admission_evaluations(cache_decision.admission_evaluated_candidates);
        metrics.record_cache_admission_rejections(cache_decision.admission_rejected_candidates);
        metrics.record_cache_pressure_guard(
            cache_decision.pressure_guard_compared_pairs,
            cache_decision.pressure_guard_overrides,
        );
        metrics.record_cache_monitor_decision(cache_decision.prefill_pressure_source);
        let Some(decision) = &cache_decision.decision else {
            return None;
        };
        let selected_candidate = self
            .candidates
            .iter()
            .find(|candidate| candidate.worker.id == decision.selected.id)?;
        tracing::debug!(
            model = %model_id,
            policy = ?ProposalKind::CacheAffinity,
            range = %decision.candidate_range_id,
            selected = %decision.selected.url,
            cache_candidates = self.candidates.len(),
            input_tokens = input_tokens,
            matched_prefix_tokens = selected_candidate.matched_prefix_tokens,
            uncached_tokens = selected_candidate.uncached_tokens,
            reason = ?decision.reason,
            load_snapshot_version = decision.load_snapshot_version,
            prefill_pressure_source = cache_decision.prefill_pressure_source,
            "cache candidate winner",
        );
        metrics.record_policy_decision(
            "cache_aware",
            if decision.reason == DecisionReason::SaturationPin {
                "saturation_pin"
            } else {
                "cache_candidate"
            },
        );
        if decision.reason == DecisionReason::SaturationPin {
            // The pin books the saturation label because it always means
            // affinity was kept under a queueing fleet. It does not retire
            // the off-owner draw in `run`: when every gate-rejected owner
            // also fails capacity admission the pin yields no decision, and
            // the fallback records the same label from an off-owner landing.
            metrics.record_cache_aware_decision(&model_id.0, CacheAwareDecision::AllQueued);
        } else {
            metrics.record_cache_aware_decision(
                &model_id.0,
                if cache_decision.queue_gate_fell_back {
                    // The gate removed every owner and nowhere in the fleet is
                    // unqueued, so the prefix was kept rather than traded for a
                    // wait that cannot be dodged. Booked as saturation, never
                    // as a plain hit.
                    CacheAwareDecision::AllQueued
                } else {
                    CacheAwareDecision::CacheHit
                },
            );
        }
        Some(Arc::clone(&decision.selected))
    }
}

/// Cache audit carried until the workflow knows whether fallback succeeded.
#[derive(Default)]
pub(crate) struct CacheFallbackAudit {
    rejected: u64,
    evaluated: u64,
    fleet_all_queued: bool,
    best_rejected_blocks: u32,
}

impl From<&CacheCandidateResolution> for CacheFallbackAudit {
    fn from(resolution: &CacheCandidateResolution) -> Self {
        Self {
            rejected: resolution.queue_gate_rejected_candidates,
            evaluated: resolution.admission_evaluated_candidates,
            fleet_all_queued: resolution.fleet_all_queued,
            best_rejected_blocks: resolution.queue_gate_best_rejected_blocks,
        }
    }
}

impl CacheFallbackAudit {
    pub(crate) fn record_fallback(&self, metrics: &MetricsRegistry, model: &ModelId) {
        let decision =
            cache_aware_fallback_decision(self.rejected, self.evaluated, self.fleet_all_queued);
        if matches!(decision, CacheAwareDecision::CacheWorkerQueued) {
            metrics.observe_diverted_overlap_blocks(&model.0, u64::from(self.best_rejected_blocks));
        }
        metrics.record_cache_aware_decision(&model.0, decision);
    }
}

#[cfg(test)]
mod test_support {
    use super::*;

    pub(super) struct CacheTestCase {
        pub candidates: Vec<CacheCandidate>,
        pub config: AffinityConfig,
    }

    // Original resolver tests used zeroed proposal settings, independently of CLI defaults.
    pub(super) fn test_config() -> AffinityConfig {
        AffinityConfig {
            cache_switch_margin_tokens: 0,
            pressure_guard: false,
            pressure_abs_threshold_tokens: 0,
            pressure_abs_threshold_ms: None,
            pressure_rel_threshold: 0.0,
            worker_queue_limit: None,
            saturation_queue_floor: None,
            ..AffinityConfig::default()
        }
    }

    pub(super) fn resolve_cache_candidates(
        case: &CacheTestCase,
        tokens: u64,
        snapshot: &EngineSnapshot,
        fleet: &[Arc<Worker>],
    ) -> CacheCandidateResolution {
        CacheAwarePolicy::new(case.config.clone())
            .evaluate_candidates(
                CacheCandidateProposal {
                    candidates: case.candidates.clone(),
                },
                tokens,
                snapshot,
                fleet,
            )
            .resolution
    }
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
    use super::test_support::{resolve_cache_candidates, test_config, CacheTestCase};

    use super::*;
    use crate::config::AffinityConfig;
    use crate::kv_events::PrefixSignal;
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
        let proposal = CacheTestCase {
            candidates: vec![CacheCandidate {
                worker: Arc::clone(&hot),
                matched_prefix_tokens: 75,
                uncached_tokens: 25,
                matched_prefix_blocks: 3,
                candidate_range_id: "global".into(),
                max_pending_prefill_tokens: None,
            }],
            config: AffinityConfig {
                cache_switch_margin_tokens: 8,
                ..test_config()
            },
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

        let PrefillEvaluation::Cache(proposal) = policy
            .evaluate_prefill(&workers, &ctx)
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

        let PrefillEvaluation::Cache(proposal) = policy
            .evaluate_prefill(&workers, &ctx)
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

        let PrefillEvaluation::Cache(proposal) = policy
            .evaluate_prefill(&workers, &ctx)
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

        let PrefillEvaluation::Cache(proposal) = policy
            .evaluate_prefill(&workers, &ctx)
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
            .evaluate_prefill(&workers, &ctx)
            .expect("a weak hit must degrade to no-hit P2, not fail selection");

        assert!(
            matches!(proposal, PrefillEvaluation::Pair(_)),
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

        let PrefillEvaluation::Cache(proposal) = policy
            .evaluate_prefill(&workers, &ctx)
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
        let proposal = CacheTestCase {
            candidates: vec![
                cache_candidate(&full, 90, 10, None),
                cache_candidate(&winner, 70, 30, None),
            ],
            config: AffinityConfig {
                cache_switch_margin_tokens: 16,
                ..test_config()
            },
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
        let proposal = CacheTestCase {
            candidates: vec![
                cache_candidate(&first, 40, 60, None),
                cache_candidate(&second, 60, 40, None),
                cache_candidate(&final_winner, 80, 20, None),
            ],
            config: AffinityConfig {
                cache_switch_margin_tokens: 0,
                ..test_config()
            },
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
        let proposal = CacheTestCase {
            candidates: vec![cache_candidate(&candidate, 80, 20, Some(30))],
            config: AffinityConfig {
                cache_switch_margin_tokens: 16,
                ..test_config()
            },
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
        let proposal = CacheTestCase {
            candidates: vec![
                cache_candidate(&congested, 90, 10, None),
                cache_candidate(&idle, 80, 20, None),
            ],
            config: AffinityConfig {
                cache_switch_margin_tokens: 32,
                ..test_config()
            },
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
        let proposal = CacheTestCase {
            candidates: vec![
                cache_candidate(&hot, 90, 10, None),
                cache_candidate(&idle, 20, 80, None),
            ],
            config: AffinityConfig {
                cache_switch_margin_tokens: 32,
                ..test_config()
            },
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
        let proposal = CacheTestCase {
            // The policy supplies candidates in increasing E order. Each
            // adjacent pair is a near tie,but the last candidate is more
            // than one configured margin away from the global work minimum.
            candidates: vec![
                cache_candidate(&best_work, 100, 0, None),
                cache_candidate(&near_tie, 80, 20, None),
                cache_candidate(&beyond_margin, 60, 40, None),
            ],
            config: AffinityConfig {
                cache_switch_margin_tokens: 32,
                ..test_config()
            },
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

#[cfg(test)]
mod resolution_tests {
    use super::test_support::{resolve_cache_candidates, test_config, CacheTestCase};
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::workers::engine_reports::NativeCacheWorkerLoad;
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

    fn snapshot(entries: &[(&Arc<Worker>, u64, u64, u64, u64)]) -> EngineSnapshot {
        EngineSnapshot::from_native_cache_workers(
            7,
            entries
                .iter()
                .map(|(worker, running, waiting, used, capacity)| {
                    (
                        worker.url.clone(),
                        NativeCacheWorkerLoad {
                            num_running_reqs: *running,
                            num_waiting_reqs: *waiting,
                            num_waiting_uncached_tokens: *waiting,
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

    #[test]
    fn complete_monitor_pressure_guard_overrides_a_near_cache_gain() {
        let congested = worker("congested");
        let idle = worker("idle");
        let proposal = CacheTestCase {
            candidates: vec![
                CacheCandidate {
                    worker: Arc::clone(&congested),
                    matched_prefix_tokens: 90,
                    uncached_tokens: 10,
                    matched_prefix_blocks: 9,
                    candidate_range_id: "global".into(),
                    max_pending_prefill_tokens: None,
                },
                CacheCandidate {
                    worker: Arc::clone(&idle),
                    matched_prefix_tokens: 80,
                    uncached_tokens: 20,
                    matched_prefix_blocks: 8,
                    candidate_range_id: "global".into(),
                    max_pending_prefill_tokens: None,
                },
            ],
            config: AffinityConfig {
                cache_switch_margin_tokens: 32,
                pressure_guard: true,
                pressure_abs_threshold_tokens: 100,
                pressure_abs_threshold_ms: None,
                pressure_rel_threshold: 1.5,
                worker_queue_limit: None,
                saturation_queue_floor: None,
                ..test_config()
            },
        };
        let loads = snapshot(&[
            (&congested, 1, 1_000, 10, 10_000),
            (&idle, 1, 10, 10, 10_000),
        ]);

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &[]);
        assert_eq!(
            resolution
                .decision
                .expect("the idle candidate remains admitted")
                .selected
                .id,
            idle.id
        );
        assert_eq!(resolution.prefill_pressure_source, "native_queue_tokens");
        assert_eq!(resolution.admission_rejected_candidates, 0);
        assert_eq!(resolution.pressure_guard_compared_pairs, 1);
        assert_eq!(resolution.pressure_guard_overrides, 1);
    }

    fn candidate(
        worker: &Arc<Worker>,
        uncached_tokens: u64,
        matched_blocks: u32,
    ) -> CacheCandidate {
        CacheCandidate {
            worker: Arc::clone(worker),
            matched_prefix_tokens: 100 - uncached_tokens,
            uncached_tokens,
            matched_prefix_blocks: matched_blocks,
            candidate_range_id: "global".into(),
            max_pending_prefill_tokens: None,
        }
    }

    fn queue_gate_proposal(candidates: Vec<CacheCandidate>, limit: Option<u64>) -> CacheTestCase {
        CacheTestCase {
            candidates,
            config: AffinityConfig {
                worker_queue_limit: limit,
                ..test_config()
            },
        }
    }

    fn saturation_proposal(
        candidates: Vec<CacheCandidate>,
        limit: Option<u64>,
        floor: Option<u64>,
    ) -> CacheTestCase {
        CacheTestCase {
            candidates,
            config: AffinityConfig {
                worker_queue_limit: limit,
                saturation_queue_floor: floor,
                ..test_config()
            },
        }
    }

    #[test]
    fn queue_gate_diverts_off_an_owner_over_the_limit() {
        let owner = worker("owner");
        let other = worker("other");
        let proposal = queue_gate_proposal(
            vec![
                // The owner holds the deeper prefix and would always win
                // without the gate.
                candidate(&owner, 10, 9),
                candidate(&other, 60, 4),
            ],
            Some(4),
        );
        // The owner is at the limit (4 waiting >= 4); the other is idle.
        let loads = snapshot(&[(&owner, 1, 4, 10, 10_000), (&other, 1, 0, 10, 10_000)]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&other)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert_eq!(
            resolution
                .decision
                .expect("the unqueued candidate remains admitted")
                .selected
                .id,
            other.id
        );
        assert_eq!(resolution.queue_gate_rejected_candidates, 1);
        assert_eq!(resolution.queue_gate_best_rejected_blocks, 9);
        // A gate rejection must not read as a capacity rejection.
        assert_eq!(resolution.admission_rejected_candidates, 0);
    }

    #[test]
    fn queue_gate_disabled_keeps_the_deepest_owner() {
        let owner = worker("owner");
        let other = worker("other");
        let proposal = queue_gate_proposal(
            vec![candidate(&owner, 10, 9), candidate(&other, 60, 4)],
            None,
        );
        let loads = snapshot(&[(&owner, 1, 40, 10, 10_000), (&other, 1, 0, 10, 10_000)]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&other)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert_eq!(
            resolution
                .decision
                .expect("no gate, owner wins")
                .selected
                .id,
            owner.id
        );
        assert_eq!(resolution.queue_gate_rejected_candidates, 0);
    }

    #[test]
    fn queue_gate_missing_snapshot_admits() {
        let unknown = worker("unknown");
        let proposal = queue_gate_proposal(vec![candidate(&unknown, 10, 9)], Some(4));
        // The worker never published a load sample: the gate fails open.
        let loads = snapshot(&[]);
        let fleet = vec![Arc::clone(&unknown)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert_eq!(
            resolution
                .decision
                .expect("an unknown queue admits")
                .selected
                .id,
            unknown.id
        );
        assert_eq!(resolution.queue_gate_rejected_candidates, 0);
    }

    #[test]
    fn queue_gate_exhausted_candidates_returns_no_decision_with_audit() {
        // The only owner is queueing but a non-owner is idle, so diversion
        // still buys something: the request must leave the prefix.
        let owner = worker("owner");
        let idle_stranger = worker("idle_stranger");
        let proposal = queue_gate_proposal(vec![candidate(&owner, 10, 9)], Some(4));
        let loads = snapshot(&[
            (&owner, 1, 9, 10, 10_000),
            (&idle_stranger, 0, 0, 10, 10_000),
        ]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&idle_stranger)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert!(resolution.decision.is_none());
        assert_eq!(resolution.queue_gate_rejected_candidates, 1);
        assert_eq!(resolution.queue_gate_best_rejected_blocks, 9);
        assert_eq!(resolution.admission_rejected_candidates, 0);
        assert_eq!(resolution.admission_evaluated_candidates, 0);
        assert!(!resolution.fleet_all_queued);
        assert!(!resolution.queue_gate_fell_back);
    }

    #[test]
    fn queue_gate_admits_one_below_the_limit() {
        // Pins the admit side of the `<` boundary: `limit - 1` waiting must
        // still win on affinity, or the gate fires a request early.
        let owner = worker("owner");
        let other = worker("other");
        let proposal = queue_gate_proposal(
            vec![candidate(&owner, 10, 9), candidate(&other, 60, 4)],
            Some(4),
        );
        let loads = snapshot(&[(&owner, 1, 3, 10, 10_000), (&other, 1, 0, 10, 10_000)]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&other)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert_eq!(
            resolution
                .decision
                .expect("one below the limit is not queueing")
                .selected
                .id,
            owner.id
        );
        assert_eq!(resolution.queue_gate_rejected_candidates, 0);
        assert_eq!(resolution.admission_evaluated_candidates, 2);
    }

    #[test]
    fn queue_gate_keeps_the_prefix_when_the_whole_fleet_is_queueing() {
        // Every owner is over the limit AND so is every other worker, so a
        // diversion could not dodge a wait. Discarding the prefix would be
        // pure loss: the ungated tier re-admits the owners.
        let owner = worker("owner");
        let shallow_owner = worker("shallow_owner");
        let proposal = queue_gate_proposal(
            vec![candidate(&owner, 10, 9), candidate(&shallow_owner, 60, 4)],
            Some(4),
        );
        let loads = snapshot(&[
            (&owner, 1, 9, 10, 10_000),
            (&shallow_owner, 1, 5, 10, 10_000),
        ]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&shallow_owner)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert_eq!(
            resolution
                .decision
                .expect("an all-queueing fleet must keep its prefix")
                .selected
                .id,
            owner.id,
            "the deepest prefix must win once diversion buys nothing"
        );
        assert!(resolution.queue_gate_fell_back);
        assert!(resolution.fleet_all_queued);
        assert_eq!(resolution.queue_gate_rejected_candidates, 2);
        assert_eq!(
            resolution.admission_evaluated_candidates, 2,
            "the ungated tier puts every candidate through capacity admission"
        );
    }

    #[test]
    fn queue_gate_saturation_survives_a_capacity_exhausted_re_admission() {
        // The audit tuple the decision label is read from, in the one case
        // that used to lose the saturation signal: the fleet is queueing, the
        // ungated tier re-admits the owners, and they then fail hard
        // admission, so there is no winner. `fleet_all_queued` must still be
        // set on the way out, because the label is keyed on the fleet being
        // saturated and not on where the request finally landed.
        let owner = worker("owner");
        let shallow_owner = worker("shallow_owner");
        let proposal = queue_gate_proposal(
            vec![candidate(&owner, 10, 9), candidate(&shallow_owner, 60, 4)],
            Some(4),
        );
        // Queueing AND out of KV: used == capacity on both.
        let loads = snapshot(&[
            (&owner, 1, 9, 10_000, 10_000),
            (&shallow_owner, 1, 5, 10_000, 10_000),
        ]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&shallow_owner)];

        let resolution = resolve_cache_candidates(&proposal, 100_000, &loads, &fleet);

        assert!(
            resolution.decision.is_none(),
            "a capacity-exhausted fleet cannot produce a winner"
        );
        assert!(resolution.fleet_all_queued);
        assert!(resolution.queue_gate_fell_back);
        assert_eq!(resolution.queue_gate_rejected_candidates, 2);
        assert_eq!(
            resolution.admission_evaluated_candidates, 2,
            "re-admission is what makes a zero-evaluated saturated audit unreachable"
        );
    }

    #[test]
    fn queue_gate_fleet_saturation_needs_a_fresh_sample_everywhere() {
        // A worker with no fresh sample has an unknown queue, not a proven
        // full one, so the fleet is not saturated and the gate keeps diverting.
        let owner = worker("owner");
        let silent = worker("silent");
        let proposal = queue_gate_proposal(vec![candidate(&owner, 10, 9)], Some(4));
        let loads = snapshot(&[(&owner, 1, 9, 10, 10_000)]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&silent)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert!(resolution.decision.is_none());
        assert!(!resolution.fleet_all_queued);
        assert!(!resolution.queue_gate_fell_back);
    }

    #[test]
    fn saturation_pin_keeps_affinity_with_the_least_pressured_owner() {
        let calm_owner = worker("calm_owner");
        let busy_owner = worker("busy_owner");
        // A fleet worker that is NOT a cache candidate: the saturation check
        // reads fleet-wide fresh samples, not just the candidate set. It is
        // over the floor, so it does not break the saturation claim.
        let fleet_only = worker("fleet_only");
        let proposal = saturation_proposal(
            vec![
                // The busy owner holds the deeper prefix and would win
                // without the gate; both owners are over the limit.
                candidate(&busy_owner, 10, 9),
                candidate(&calm_owner, 60, 4),
            ],
            Some(4),
            Some(2),
        );
        let loads = snapshot(&[
            (&busy_owner, 1, 9, 10, 10_000),
            (&calm_owner, 1, 5, 10, 10_000),
            (&fleet_only, 1, 3, 10, 10_000),
        ]);
        let fleet = vec![
            Arc::clone(&busy_owner),
            Arc::clone(&calm_owner),
            Arc::clone(&fleet_only),
        ];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        let decision = resolution
            .decision
            .expect("nothing reads below the floor: pin to a prefix owner");
        assert_eq!(decision.reason, DecisionReason::SaturationPin);
        assert_eq!(
            decision.selected.id, calm_owner.id,
            "saturation suspends the gate, not the tiebreak"
        );
        assert_eq!(decision.primary.id, calm_owner.id);
        assert!(decision.backup.is_none());
        assert_eq!(decision.load_snapshot_version, loads.version);
        assert_eq!(resolution.queue_gate_rejected_candidates, 2);
        assert_eq!(resolution.queue_gate_best_rejected_blocks, 9);
        assert_eq!(resolution.admission_rejected_candidates, 0);
    }

    #[test]
    fn no_saturation_floor_preserves_queue_gate_exhaustion() {
        let owner = worker("owner");
        let other = worker("other");
        let proposal = saturation_proposal(
            vec![candidate(&owner, 10, 9), candidate(&other, 60, 4)],
            Some(4),
            None,
        );
        let loads = snapshot(&[(&owner, 1, 9, 10, 10_000), (&other, 1, 5, 10, 10_000)]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&other)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        // With no floor the queue gate's own second tier applies: every owner
        // is over the limit and nowhere in the fleet is unqueued, so the
        // prefix is kept rather than traded for a wait that cannot be dodged.
        // The pin is what a floor buys; without one this is the behaviour.
        assert!(resolution.queue_gate_fell_back);
        assert_eq!(
            resolution.decision.as_ref().map(|d| d.reason),
            Some(DecisionReason::CacheCandidate)
        );
        assert_eq!(resolution.queue_gate_rejected_candidates, 2);
        assert_eq!(resolution.queue_gate_best_rejected_blocks, 9);
        assert_eq!(resolution.admission_rejected_candidates, 0);
    }

    #[test]
    fn saturation_pin_yields_to_a_provably_idle_fleet_worker() {
        let owner = worker("owner");
        // Not a candidate — but a fresh reading below the floor anywhere in
        // the fleet means diverting can pay, so the pin must not fire.
        let idle_elsewhere = worker("idle_elsewhere");
        let proposal = saturation_proposal(vec![candidate(&owner, 10, 9)], Some(4), Some(2));
        let loads = snapshot(&[
            (&owner, 1, 9, 10, 10_000),
            (&idle_elsewhere, 1, 0, 10, 10_000),
        ]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&idle_elsewhere)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        assert!(resolution.decision.is_none());
        assert_eq!(resolution.queue_gate_rejected_candidates, 1);
        assert_eq!(resolution.queue_gate_best_rejected_blocks, 9);
    }

    #[test]
    fn saturation_pin_treats_an_unknown_queue_as_not_idle() {
        let owner = worker("owner");
        // In the fleet, routable, and never published a sample.
        let unsampled = worker("unsampled");
        let proposal = saturation_proposal(vec![candidate(&owner, 10, 9)], Some(4), Some(2));
        // The snapshot holds only the over-limit owner. `unsampled` is a
        // real destination whose queue is unknown, not proof of a better
        // one — opposite of the gate's fail-open.
        let loads = snapshot(&[(&owner, 1, 9, 10, 10_000)]);
        let fleet = vec![Arc::clone(&owner), Arc::clone(&unsampled)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        let decision = resolution
            .decision
            .expect("an unknown queue must not read as below the floor");
        assert_eq!(decision.reason, DecisionReason::SaturationPin);
        assert_eq!(decision.selected.id, owner.id);
    }

    #[test]
    fn saturation_pin_ignores_idle_workers_outside_the_routable_fleet() {
        let owner = worker("owner");
        // Present in the router-wide load table but not routable for this
        // request: a PD decode peer, another model's worker, or a worker the
        // registry no longer reports healthy. Decode peers idle near zero
        // waiting, so scanning the whole table would veto the pin on every
        // PD deployment.
        let off_fleet_idle = worker("off_fleet_idle");
        let proposal = saturation_proposal(vec![candidate(&owner, 10, 9)], Some(4), Some(2));
        let loads = snapshot(&[
            (&owner, 1, 9, 10, 10_000),
            (&off_fleet_idle, 1, 0, 10, 10_000),
        ]);
        let fleet = vec![Arc::clone(&owner)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        let decision = resolution
            .decision
            .expect("an unroutable idle worker is not a destination a diversion could reach");
        assert_eq!(decision.reason, DecisionReason::SaturationPin);
        assert_eq!(decision.selected.id, owner.id);
    }

    #[test]
    fn saturation_pin_skips_a_capacity_exhausted_owner() {
        let full = worker("full");
        let admitted_owner = worker("admitted_owner");
        let proposal = saturation_proposal(
            vec![candidate(&full, 10, 9), candidate(&admitted_owner, 60, 4)],
            Some(4),
            Some(2),
        );
        // Both owners are over the queue limit, and `full` is the pressure
        // minimum — but it is also KV-exhausted (used + request exceeds
        // capacity), so it cannot take the request even pinned.
        let loads = snapshot(&[
            (&full, 1, 5, 10_000, 10_000),
            (&admitted_owner, 1, 9, 10, 10_000),
        ]);
        let fleet = vec![Arc::clone(&full), Arc::clone(&admitted_owner)];

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &fleet);

        let decision = resolution
            .decision
            .expect("the capacity-admitted owner can be pinned");
        assert_eq!(decision.reason, DecisionReason::SaturationPin);
        assert_eq!(decision.selected.id, admitted_owner.id);
        // `full` is booked under the gate, not capacity — the pin's own
        // capacity filter must not pollute the audit counters.
        assert_eq!(resolution.queue_gate_rejected_candidates, 2);
        assert_eq!(resolution.admission_rejected_candidates, 0);
    }
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    #[test]
    fn cache_aware_fallback_decision_needs_the_gate_to_have_emptied_the_set() {
        // The trap this pins: on an UNSATURATED fleet, one owner queueing
        // while the others exhaust KV capacity is a CAPACITY problem, not a
        // gate diversion. Only a gate that removed every owner leaves zero
        // candidates evaluated.
        assert!(matches!(
            cache_aware_fallback_decision(1, 3, false),
            CacheAwareDecision::CacheMiss
        ));
        // Nothing gated out at all: a plain miss, saturated or not.
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
        // diversion off the prefix. `resolve_cache_candidates` leaves
        // `evaluated` at zero here because its second tier does not fire on
        // an unsaturated fleet.
        assert!(matches!(
            cache_aware_fallback_decision(2, 0, false),
            CacheAwareDecision::CacheWorkerQueued
        ));
        // Saturation, in the shape the resolver actually produces: the
        // second tier re-admitted the gated-out owners, so `evaluated` is
        // NON-zero, and they then failed hard admission. Booking the
        // capacity outcome here would drop the saturation signal exactly
        // where it matters — hence saturation is asked first. Pinning
        // `(2, 0, true)` instead would assert a state the resolver cannot
        // reach: re-admission and the `AllQueued` precondition are the same
        // condition, so an empty `evaluated` never survives it.
        assert!(matches!(
            cache_aware_fallback_decision(2, 2, true),
            CacheAwareDecision::AllQueued
        ));
    }
}
