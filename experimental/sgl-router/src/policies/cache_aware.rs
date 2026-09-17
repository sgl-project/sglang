// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware candidate preparation, queue/capacity evaluation, and reporting.

use crate::config::AffinityConfig;
use crate::discovery::ModelId;
use crate::policies::admission::{
    fleet_is_all_queued, has_kv_capacity, materially_more_pressured, queue_gate_admits,
    DecisionReason, FinalDecision, FreshLoadLookup,
};
use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
use crate::policies::{Policy, PrefillProposal, ProposalKind, SelectionContext, SelectionProposal};
use crate::server::metrics::{CacheAwareDecision, MetricsRegistry};
use crate::workers::engine_load_reports::EngineLoadSnapshot;
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

/// Bounded set of Cache-Aware candidates.
#[derive(Clone, Default)]
pub struct CacheCandidateProposal {
    pub candidates: Vec<CacheCandidate>,
    pub cache_switch_margin_tokens: u64,
    pub enable_pressure_guard: bool,
    pub pressure_abs_threshold_tokens: u64,
    pub pressure_abs_threshold_ms: Option<f64>,
    pub pressure_rel_threshold: f64,
    /// Queue gate: a candidate whose engine reports at least this many
    /// waiting requests cannot win on cache affinity. `None` disables the
    /// gate. See [`crate::config::AffinityConfig::worker_queue_limit`].
    pub worker_queue_limit: Option<u64>,
    /// Saturation pin: when no candidate survives the gate and hard
    /// admission, at least one was queue-gate-rejected, and no worker in
    /// the routable fleet has a fresh queue reading strictly below this
    /// floor, the request pins to the least-pressured rejected prefix
    /// owner instead of diverting — the diversion cannot dodge a wait and
    /// would forfeit the matched prefix. `None` disables the pin. See
    /// [`crate::config::AffinityConfig::saturation_queue_floor`].
    pub saturation_queue_floor: Option<u64>,
}

/// Cache-Aware selection audit data. These fields do not affect selection.
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

/// Evaluated cache candidates and the audit used if routing falls back.
pub(crate) struct CacheSelection {
    candidates: Vec<CacheCandidate>,
    pub(crate) resolution: CacheCandidateResolution,
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

/// Selects a worker from bounded cache candidates and records guard coverage.
/// `fleet` is the worker set this request could actually be routed to (the
/// model's healthy prefill pool). Only the saturation pin reads it, and it
/// must be the routable fleet rather than the router-wide load table: that
/// table also holds decode peers and other models' workers, none of which a
/// diversion could reach.
pub fn resolve_cache_candidates(
    proposal: &CacheCandidateProposal,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
    fleet: &[Arc<Worker>],
) -> CacheCandidateResolution {
    let queue_limit = proposal.worker_queue_limit;
    let fleet_all_queued = fleet_is_all_queued(snapshot, fleet, queue_limit);
    // The queue gate runs before hard admission so a busy worker never
    // pollutes the capacity-rejection counters, and so the diverted-overlap
    // audit below sees exactly the candidates the gate removed. One pass:
    // the gate answer per candidate is what splits the set, so asking twice
    // would re-read the snapshot for every candidate on every request.
    let mut evaluated: Vec<&CacheCandidate> = Vec::with_capacity(proposal.candidates.len());
    // Kept, not just counted: the saturation pin ranks these by pressure when
    // it fires. Stays unallocated while the gate is disabled, because nothing
    // is ever rejected then.
    let mut queue_gate_rejected: Vec<&CacheCandidate> = Vec::new();
    let mut queue_gate_best_rejected_blocks = 0u32;
    for candidate in &proposal.candidates {
        if queue_gate_admits(snapshot, &candidate.worker, queue_limit) {
            evaluated.push(candidate);
        } else {
            queue_gate_rejected.push(candidate);
            queue_gate_best_rejected_blocks =
                queue_gate_best_rejected_blocks.max(candidate.matched_prefix_blocks);
        }
    }
    let queue_gate_rejected_candidates = queue_gate_rejected.len() as u64;
    // Second tier, mirroring `range_fallback`: when the gate removed every
    // candidate AND nowhere in the fleet is unqueued, diversion cannot dodge
    // a wait, so returning no decision would trade the whole prefix for
    // nothing. Re-admit the ungated set. While an unqueued worker still
    // exists the gate keeps its teeth and the request leaves the prefix.
    //
    // A configured saturation floor supersedes this tier rather than stacking
    // with it: the floor names a weaker, tunable saturation condition and
    // pins with a pressure-only ranking, where re-admission would re-rank by
    // uncached work first. Both keep the prefix; only one may decide which
    // owner, so the explicit knob wins and this tier covers the unset case.
    let queue_gate_fell_back = evaluated.is_empty()
        && queue_gate_rejected_candidates > 0
        && fleet_all_queued
        && proposal.saturation_queue_floor.is_none();
    if queue_gate_fell_back {
        evaluated.extend(proposal.candidates.iter());
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
        // Saturation pin: no candidate survived the gate and hard
        // admission (and at least one was gate-rejected), but diverting
        // only pays when a meaningfully idle destination exists. With a
        // floor configured and no fresh queue reading below it, the request
        // would wait wherever it lands — so waiting at a prefix owner
        // dominates: same wait, prefill from cache instead of a full cold
        // prefill that evicts other prefixes and manufactures the next
        // round of misses. Saturation suspends the gate, not the tiebreak:
        // pin to the least-pressured rejected owner, skipping any that also
        // fail hard admission (a capacity-exhausted owner cannot take the
        // request).
        let pinned = if queue_gate_rejected.is_empty() {
            None
        } else {
            proposal.saturation_queue_floor.and_then(|floor| {
                if snapshot
                    .any_fresh_queue_below(fleet.iter().map(|worker| worker.url.as_str()), floor)
                {
                    return None;
                }
                // Ranked over its own lookup, not `loads`: `loads` covers the
                // gate-ADMITTED set, which is empty precisely when the pin
                // fires. An empty lookup reports no engine coverage, so every
                // comparison would fall back to router-local load — tie at
                // zero for every owner, decided by worker id. The pin ranks
                // the rejected owners, so it must see the rejected owners.
                let pin_loads = FreshLoadLookup::new(
                    Some(snapshot),
                    queue_gate_rejected
                        .iter()
                        .map(|candidate| &candidate.worker),
                );
                queue_gate_rejected
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        is_cache_candidate_admitted(candidate, request_input_tokens, &pin_loads)
                    })
                    .min_by(|left, right| {
                        pin_loads
                            .compare_prefill_pressure(&left.worker, &right.worker)
                            .then_with(|| left.worker.id.0.cmp(&right.worker.id.0))
                    })
            })
        };
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
    let near_tie_ceiling = work_floor
        .uncached_tokens
        .saturating_add(proposal.cache_switch_margin_tokens);
    let mut winner = work_floor;
    let mut pressure_guard_compared_pairs = 0;
    let mut pressure_guard_overrides = 0;
    for candidate in admitted {
        if candidate.worker.id == winner.worker.id || candidate.uncached_tokens > near_tie_ceiling {
            continue;
        }
        let baseline = compare_cache_candidates(winner, candidate, proposal, &loads, false);
        let ordering = if proposal.enable_pressure_guard
            && cache_pressure_guard_comparable(winner, candidate, &loads)
        {
            pressure_guard_compared_pairs += 1;
            let guarded = compare_cache_candidates(winner, candidate, proposal, &loads, true);
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
    proposal: &CacheCandidateProposal,
    loads: &FreshLoadLookup<'_>,
    enable_pressure_guard: bool,
) -> Ordering {
    let work_delta = left.uncached_tokens.abs_diff(right.uncached_tokens);
    if work_delta > proposal.cache_switch_margin_tokens {
        return left
            .uncached_tokens
            .cmp(&right.uncached_tokens)
            .then_with(|| loads.compare_prefill_pressure(&left.worker, &right.worker))
            .then_with(|| left.worker.id.0.cmp(&right.worker.id.0));
    }
    if enable_pressure_guard {
        if materially_more_pressured(
            &left.worker,
            &right.worker,
            proposal.pressure_abs_threshold_tokens,
            proposal.pressure_abs_threshold_ms,
            proposal.pressure_rel_threshold,
            loads,
        ) {
            return Ordering::Greater;
        }
        if materially_more_pressured(
            &right.worker,
            &left.worker,
            proposal.pressure_abs_threshold_tokens,
            proposal.pressure_abs_threshold_ms,
            proposal.pressure_rel_threshold,
            loads,
        ) {
            return Ordering::Less;
        }
    }
    left.uncached_tokens
        .cmp(&right.uncached_tokens)
        .then_with(|| loads.compare_prefill_pressure(&left.worker, &right.worker))
        .then_with(|| left.worker.id.0.cmp(&right.worker.id.0))
}

fn cache_pressure_guard_comparable(
    left: &CacheCandidate,
    right: &CacheCandidate,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    loads.comparable_get(&left.worker.id).is_some()
        && loads.comparable_get(&right.worker.id).is_some()
}

impl CacheCandidateProposal {
    pub(crate) fn evaluate(
        self,
        input_tokens: u64,
        snapshot: &EngineLoadSnapshot,
        fleet: &[Arc<Worker>],
        model: &ModelId,
    ) -> CacheSelection {
        // The queue gate reads the engine-published load sample and fails open
        // per worker. When NO worker has a fresh sample the gate is inert
        // fleet-wide and nothing would say so: `cache_worker_queued` sitting at
        // 0 is indistinguishable from a healthy fleet. Warn (sampled) — a fleet
        // that never advertised a load port must not silently disable the gate.
        if self.worker_queue_limit.is_some()
            && !fleet.is_empty()
            && fleet
                .iter()
                .all(|worker| snapshot.fresh_load_for_url(&worker.url).is_none())
            && QUEUE_GATE_BLIND_LOG_COUNTER
                .fetch_add(1, AtomicOrdering::Relaxed)
                .is_multiple_of(QUEUE_GATE_BLIND_LOG_SAMPLE)
        {
            tracing::warn!(
                model = %model,
                worker_queue_limit = self.worker_queue_limit,
                workers = fleet.len(),
                "--worker-queue-limit is set but no worker has a fresh engine load \
                 sample, so the queue gate is inert. Check that engines advertise a \
                 load port and publish LoadStat",
            );
        }
        let resolution = resolve_cache_candidates(&self, input_tokens, snapshot, fleet);
        let selection = CacheSelection {
            candidates: self.candidates,
            resolution,
        };
        if let Some(decision) = &selection.resolution.decision {
            if decision.reason == DecisionReason::SaturationPin
                && SATURATION_PIN_LOG_COUNTER
                    .fetch_add(1, AtomicOrdering::Relaxed)
                    .is_multiple_of(SATURATION_PIN_LOG_SAMPLE)
            {
                tracing::info!(
                    model = %&model.0,
                    worker = %decision.selected.url,
                    saturation_queue_floor = self.saturation_queue_floor,
                    worker_queue_limit = self.worker_queue_limit,
                    "fleet saturated, keeping affinity with a queueing prefix owner \
                     instead of diverting",
                );
            }
        }
        selection
    }
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
pub(crate) fn cache_aware_fallback_decision(
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
