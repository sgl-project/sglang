// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared capacity admission and pressure guards for prefill and decode.
//!
//! Native Cache-Aware uses monitor-backed admission only when every expected
//! DP rank has a fresh, complete #34608 ZMQ sample. Otherwise it falls back to
//! Router-local load.
//!
//! The optional queue gate (`--worker-queue-limit`) is the one criterion here
//! that reads the basic #34608 sample (`num_waiting_reqs`) rather than the
//! native monitor fields: a worker already making requests wait cannot win on
//! cache affinity. It fails open on a missing sample — see
//! [`queue_gate_admits`]. The optional saturation floor
//! (`--saturation-queue-floor`) cancels a diversion that has no payoff: when
//! no candidate survives the gate and hard admission, at least one was
//! gate-rejected, and no worker in the routable fleet reads below the floor,
//! the request pins to the least-pressured prefix owner instead of
//! cold-prefilling on a non-owner.

use crate::policies::engine_load::{EngineLoadSnapshot, NativeCacheWorkerLoad};
use crate::policies::power_of_two::select_with_snapshot;
use crate::policies::{CacheCandidate, CacheCandidateProposal, GuardHints, SelectionProposal};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// A prefill candidate domain and its optional queue budget.
///
/// `max_pending_prefill_tokens` is enforced only when the native monitor
/// provides `num_waiting_uncached_tokens`.
pub struct CandidateRange<'a> {
    pub id: &'a str,
    pub workers: &'a [Arc<Worker>],
    pub max_pending_prefill_tokens: Option<u64>,
}

impl<'a> CandidateRange<'a> {
    pub fn global(workers: &'a [Arc<Worker>]) -> Self {
        Self {
            id: "global",
            workers,
            max_pending_prefill_tokens: None,
        }
    }
}

/// Role-specific candidate domains resolved before policy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStage {
    Prefill,
    Decode,
}

#[derive(Clone)]
pub struct CandidateDomain {
    pub id: String,
    pub stage: RoutingStage,
    pub workers: Vec<Arc<Worker>>,
    pub max_pending_prefill_tokens: Option<u64>,
}

impl CandidateDomain {
    pub fn global_prefill(workers: &[Arc<Worker>]) -> Self {
        Self {
            id: "global".to_string(),
            stage: RoutingStage::Prefill,
            workers: workers.to_vec(),
            max_pending_prefill_tokens: None,
        }
    }

    pub fn global_decode(workers: &[Arc<Worker>]) -> Self {
        Self {
            id: "global".to_string(),
            stage: RoutingStage::Decode,
            workers: workers.to_vec(),
            max_pending_prefill_tokens: None,
        }
    }

    pub fn bucket_prefill(
        id: impl Into<String>,
        workers: Vec<Arc<Worker>>,
        max_pending_prefill_tokens: Option<u64>,
    ) -> Self {
        Self {
            id: id.into(),
            stage: RoutingStage::Prefill,
            workers,
            max_pending_prefill_tokens,
        }
    }

    pub fn bucket_decode(id: impl Into<String>, workers: Vec<Arc<Worker>>) -> Self {
        Self {
            id: id.into(),
            stage: RoutingStage::Decode,
            workers,
            max_pending_prefill_tokens: None,
        }
    }

    pub fn prefill_range(&self) -> Option<CandidateRange<'_>> {
        (self.stage == RoutingStage::Prefill).then(|| CandidateRange {
            id: self.id.as_str(),
            workers: &self.workers,
            max_pending_prefill_tokens: self.max_pending_prefill_tokens,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    Primary,
    CacheCandidate,
    BackupPrimaryAdmission,
    BackupPressureGuard,
    RangeFallback,
    CapacityFallbackPowerOfTwo,
    /// Fleet saturated: no cache candidate survived the queue gate and
    /// capacity admission, at least one was rejected by the gate, and no
    /// fleet worker has a fresh queue reading below the saturation floor —
    /// so the request pinned to a prefix owner instead of diverting.
    SaturationPin,
}

#[derive(Clone)]
pub struct FinalDecision {
    pub selected: Arc<Worker>,
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub reason: DecisionReason,
    pub candidate_range_id: String,
    pub load_snapshot_version: u64,
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
    pub pressure_guard_compared_pairs: u64,
    pub pressure_guard_overrides: u64,
}

/// The queue gate, in one place. Both subtle decisions live here: the
/// boundary is `<` (a worker AT the limit is already making this request
/// wait), and an unknown queue ADMITS. The gate reads the basic #34608
/// sample because `num_waiting_reqs` is what the request cares about, and
/// the router-side in-flight counter cannot separate a running request from
/// a waiting one — so there is no honest substitute, and the gate fails open
/// rather than comparing the limit against a different quantity.
pub(crate) fn queue_gate_admits(
    snapshot: &EngineLoadSnapshot,
    worker: &Worker,
    limit: Option<u64>,
) -> bool {
    let Some(limit) = limit else {
        return true;
    };
    snapshot
        .fresh_load_for_url(&worker.url)
        .is_none_or(|load| load.num_waiting_reqs < limit)
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
    let loads = FreshLoadLookup::new(
        Some(snapshot),
        proposal
            .candidates
            .iter()
            .map(|candidate| &candidate.worker),
    );
    // The queue gate runs before hard admission so a busy worker never
    // pollutes the capacity-rejection counters, and so the diverted-overlap
    // audit below sees exactly the candidates the gate removed. One pass:
    // the gate answer per candidate is what splits the set, so asking twice
    // would re-read the snapshot for every candidate on every request.
    let (queue_gate_admitted, queue_gate_rejected): (Vec<&CacheCandidate>, Vec<&CacheCandidate>) =
        proposal.candidates.iter().partition(|candidate| {
            queue_gate_admits(snapshot, &candidate.worker, proposal.worker_queue_limit)
        });
    let queue_gate_rejected_candidates = queue_gate_rejected.len() as u64;
    let queue_gate_best_rejected_blocks = queue_gate_rejected
        .iter()
        .map(|candidate| candidate.matched_prefix_blocks)
        .max()
        .unwrap_or(0);
    let admission_evaluated_candidates = queue_gate_admitted.len() as u64;
    let admitted: Vec<&CacheCandidate> = queue_gate_admitted
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
                queue_gate_rejected
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        is_cache_candidate_admitted(candidate, request_input_tokens, &loads)
                    })
                    .min_by(|left, right| {
                        loads
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
        pressure_guard_compared_pairs,
        pressure_guard_overrides,
    }
}

pub fn resolve_prefill(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
    queue_limit: Option<u64>,
) -> Option<FinalDecision> {
    resolve_prefill_admitted(range, proposal, request_input_tokens, snapshot, queue_limit).or_else(
        || {
            if !contains_worker(range, &proposal.primary) {
                return None;
            }
            let backup = proposal
                .backup
                .as_ref()
                .filter(|worker| contains_worker(range, worker))
                .cloned();
            let legal = legal_prefill_candidates(range, proposal);
            let selected = select_with_snapshot(&legal, Some(snapshot))?;
            Some(FinalDecision {
                selected,
                primary: Arc::clone(&proposal.primary),
                backup,
                reason: DecisionReason::CapacityFallbackPowerOfTwo,
                candidate_range_id: range.id.to_string(),
                load_snapshot_version: snapshot.version,
            })
        },
    )
}

/// Resolves prefill admission without overcommitting a full candidate range.
pub fn resolve_prefill_admitted(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
    queue_limit: Option<u64>,
) -> Option<FinalDecision> {
    if !contains_worker(range, &proposal.primary) {
        return None;
    }
    let backup = proposal
        .backup
        .as_ref()
        .filter(|worker| contains_worker(range, worker))
        .cloned();
    // The queue gate demotes an admitted primary or backup exactly as it
    // demotes a cache candidate: a worker already making requests wait must
    // not win on proposal position alone, or the fallback the gate diverted
    // to hands the request straight back to it. Demotion is not rejection —
    // `range_fallback` below is two-tier, and its second tier returns the
    // least-pressured admitted worker (the demoted one included) when every
    // admitted worker is queueing, so an all-queueing fleet still routes.
    let primary_admitted = is_proposal_worker_eligible(proposal, &proposal.primary)
        && is_prefill_admitted(range, &proposal.primary, request_input_tokens, snapshot)
        && queue_gate_admits(snapshot, &proposal.primary, queue_limit);
    let backup_admitted = backup.as_ref().is_some_and(|worker| {
        is_proposal_worker_eligible(proposal, worker)
            && is_prefill_admitted(range, worker, request_input_tokens, snapshot)
            && queue_gate_admits(snapshot, worker, queue_limit)
    });

    let (selected, reason) = match (primary_admitted, backup.as_ref(), backup_admitted) {
        (true, Some(backup), true) => {
            if pressure_guard_prefers_backup(
                &proposal.primary,
                backup,
                &proposal.guard_hints,
                snapshot,
            ) {
                (Arc::clone(backup), DecisionReason::BackupPressureGuard)
            } else {
                (Arc::clone(&proposal.primary), DecisionReason::Primary)
            }
        }
        (true, _, _) => (Arc::clone(&proposal.primary), DecisionReason::Primary),
        (false, Some(backup), true) => (Arc::clone(backup), DecisionReason::BackupPrimaryAdmission),
        _ => {
            let legal = legal_prefill_candidates(range, proposal);
            range_fallback(range, &legal, request_input_tokens, snapshot, queue_limit)?
        }
    };
    Some(FinalDecision {
        selected,
        primary: Arc::clone(&proposal.primary),
        backup,
        reason,
        candidate_range_id: range.id.to_string(),
        load_snapshot_version: snapshot.version,
    })
}

pub fn resolve_decode(
    domain: &CandidateDomain,
    proposal: &SelectionProposal,
    request_kv_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> Option<FinalDecision> {
    if domain.stage != RoutingStage::Decode || !contains_domain_worker(domain, &proposal.primary) {
        return None;
    }
    let backup = proposal
        .backup
        .as_ref()
        .filter(|worker| contains_domain_worker(domain, worker))
        .cloned();
    let primary_admitted = is_decode_admitted(&proposal.primary, request_kv_tokens, snapshot);
    let backup_admitted = backup
        .as_ref()
        .is_some_and(|worker| is_decode_admitted(worker, request_kv_tokens, snapshot));
    let (selected, reason) = match (primary_admitted, backup.as_ref(), backup_admitted) {
        (true, Some(backup), true) => {
            if compare_decode_pressure(&proposal.primary, backup, Some(snapshot)).is_gt() {
                (Arc::clone(backup), DecisionReason::BackupPressureGuard)
            } else {
                (Arc::clone(&proposal.primary), DecisionReason::Primary)
            }
        }
        (true, _, _) => (Arc::clone(&proposal.primary), DecisionReason::Primary),
        (false, Some(backup), true) => (Arc::clone(backup), DecisionReason::BackupPrimaryAdmission),
        _ => decode_domain_fallback(domain, request_kv_tokens, snapshot)?,
    };
    Some(FinalDecision {
        selected,
        primary: Arc::clone(&proposal.primary),
        backup,
        reason,
        candidate_range_id: domain.id.clone(),
        load_snapshot_version: snapshot.version,
    })
}

fn contains_worker(range: &CandidateRange<'_>, candidate: &Arc<Worker>) -> bool {
    range.workers.iter().any(|worker| worker.id == candidate.id)
}

fn contains_domain_worker(domain: &CandidateDomain, candidate: &Arc<Worker>) -> bool {
    domain
        .workers
        .iter()
        .any(|worker| worker.id == candidate.id)
}

fn is_proposal_worker_eligible(proposal: &SelectionProposal, candidate: &Arc<Worker>) -> bool {
    proposal
        .eligible_workers
        .as_ref()
        .is_none_or(|workers| workers.iter().any(|worker| worker.id == candidate.id))
}

/// Applies snapshot-backed capacity admission when native monitor data is complete.
/// Workers without monitor data remain eligible and use Router-local ordering.
fn has_kv_capacity(load: Option<&NativeCacheWorkerLoad>, requested_tokens: u64) -> bool {
    let Some(load) = load else {
        return true;
    };
    load.num_running_reqs.saturating_add(1) <= load.max_running_requests
        && load.num_total_tokens.saturating_add(requested_tokens) <= load.max_total_num_tokens
}

fn is_prefill_admitted(
    range: &CandidateRange<'_>,
    worker: &Arc<Worker>,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> bool {
    let load = snapshot.fresh_native_cache_load_for_url(&worker.url);
    has_kv_capacity(load, request_input_tokens)
        && range.max_pending_prefill_tokens.is_none_or(|limit| {
            load.is_none_or(|load| {
                load.num_waiting_uncached_tokens
                    .saturating_add(request_input_tokens)
                    <= limit
            })
        })
}

fn is_decode_admitted(
    worker: &Arc<Worker>,
    request_kv_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> bool {
    has_kv_capacity(
        snapshot.fresh_native_cache_load_for_url(&worker.url),
        request_kv_tokens,
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

fn materially_more_pressured(
    candidate: &Arc<Worker>,
    other: &Arc<Worker>,
    absolute_threshold_tokens: u64,
    absolute_threshold_ms: Option<f64>,
    relative_threshold: f64,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    let (Some(candidate_load), Some(other_load)) = (
        loads.comparable_get(&candidate.id),
        loads.comparable_get(&other.id),
    ) else {
        return false;
    };
    if let Some(absolute_threshold_ms) = absolute_threshold_ms.filter(|_| {
        candidate_load.estimated_prefill_queue_ms.is_some()
            && other_load.estimated_prefill_queue_ms.is_some()
    }) {
        let candidate_pressure = candidate_load
            .estimated_prefill_queue_ms
            .expect("availability was checked");
        let other_pressure = other_load
            .estimated_prefill_queue_ms
            .expect("availability was checked");
        return candidate_pressure - other_pressure > absolute_threshold_ms
            && candidate_pressure > other_pressure * relative_threshold;
    }
    candidate_load
        .num_waiting_uncached_tokens
        .saturating_sub(other_load.num_waiting_uncached_tokens)
        > absolute_threshold_tokens
        && candidate_load.num_waiting_uncached_tokens as f64
            > other_load.num_waiting_uncached_tokens as f64 * relative_threshold
}

/// Constant-time request view over one captured load snapshot.
///
/// External values are compared only when every candidate is present. Mixed
/// candidate sets use Router-local active load to preserve ordering.
pub(crate) struct FreshLoadLookup<'a> {
    by_worker_id: HashMap<String, &'a NativeCacheWorkerLoad>,
    basic_by_worker_id: HashMap<String, &'a crate::policies::engine_load::EngineWorkerLoad>,
    local_active_by_worker_id: HashMap<String, usize>,
    compare_engine: bool,
    compare_basic_engine: bool,
}

impl<'a> FreshLoadLookup<'a> {
    pub(crate) fn new<'w>(
        snapshot: Option<&'a EngineLoadSnapshot>,
        workers: impl IntoIterator<Item = &'w Arc<Worker>>,
    ) -> Self {
        let workers: Vec<&Arc<Worker>> = workers.into_iter().collect();
        let local_active_by_worker_id: HashMap<String, usize> = workers
            .iter()
            .map(|worker| (worker.id.0.clone(), worker.active_load()))
            .collect();
        let by_worker_id = snapshot
            .into_iter()
            .flat_map(|snapshot| {
                workers.iter().filter_map(move |worker| {
                    snapshot
                        .fresh_native_cache_load_for_url(&worker.url)
                        .map(|load| (worker.id.0.clone(), load))
                })
            })
            .collect::<HashMap<_, _>>();
        let basic_by_worker_id = snapshot
            .into_iter()
            .flat_map(|snapshot| {
                workers.iter().filter_map(move |worker| {
                    snapshot
                        .fresh_load_for_url(&worker.url)
                        .map(|load| (worker.id.0.clone(), load))
                })
            })
            .collect::<HashMap<_, _>>();
        let compare_engine = !local_active_by_worker_id.is_empty()
            && by_worker_id.len() == local_active_by_worker_id.len();
        let compare_basic_engine = !local_active_by_worker_id.is_empty()
            && basic_by_worker_id.len() == local_active_by_worker_id.len();
        Self {
            by_worker_id,
            basic_by_worker_id,
            local_active_by_worker_id,
            compare_engine,
            compare_basic_engine,
        }
    }

    pub(crate) fn get(
        &self,
        worker_id: &crate::discovery::WorkerId,
    ) -> Option<&'a NativeCacheWorkerLoad> {
        self.by_worker_id.get(worker_id.0.as_str()).copied()
    }

    fn comparable_get(
        &self,
        worker_id: &crate::discovery::WorkerId,
    ) -> Option<&'a NativeCacheWorkerLoad> {
        self.compare_engine.then(|| self.get(worker_id)).flatten()
    }

    fn pressure_key(&self, worker: &Arc<Worker>) -> PressureKey<'a> {
        PressureKey {
            load: self.comparable_get(&worker.id),
            local_active: self
                .local_active_by_worker_id
                .get(worker.id.0.as_str())
                .copied()
                .unwrap_or(usize::MAX),
        }
    }

    fn compare_prefill_keys(&self, left: &PressureKey<'a>, right: &PressureKey<'a>) -> Ordering {
        match (left.load, right.load) {
            (Some(left_load), Some(right_load)) => compare_prefill_load(left_load, right_load)
                .then_with(|| left.local_active.cmp(&right.local_active)),
            _ => left.local_active.cmp(&right.local_active),
        }
    }

    fn compare_decode_keys(&self, left: &PressureKey<'a>, right: &PressureKey<'a>) -> Ordering {
        match (left.load, right.load) {
            (Some(left_load), Some(right_load)) => compare_decode_load(left_load, right_load)
                .then_with(|| left.local_active.cmp(&right.local_active)),
            _ => left.local_active.cmp(&right.local_active),
        }
    }

    pub(crate) fn compare_prefill_pressure(
        &self,
        left: &Arc<Worker>,
        right: &Arc<Worker>,
    ) -> Ordering {
        self.compare_prefill_keys(&self.pressure_key(left), &self.pressure_key(right))
    }

    pub(crate) fn prefill_pressure_source(&self) -> &'static str {
        if self.compare_engine
            && self
                .by_worker_id
                .values()
                .all(|load| load.estimated_prefill_queue_ms.is_some())
        {
            "estimated_prefill_queue_ms"
        } else if self.compare_engine {
            "native_queue_tokens"
        } else {
            "router_local"
        }
    }

    /// Returns a queue depth consistent with admission for this request.
    ///
    /// A fully covered candidate set uses `waiting + running`; otherwise the
    /// whole set uses Router-local active load. Dispatches after the snapshot
    /// are added to the reported value.
    pub(crate) fn score_load(&self, worker: &Arc<Worker>) -> usize {
        self.compare_basic_engine
            .then(|| self.basic_by_worker_id.get(worker.id.0.as_str()).copied())
            .flatten()
            .map(|load| {
                let recent_dispatches = worker
                    .slots_acquired_since(load.captured_at)
                    .try_into()
                    .unwrap_or(u64::MAX);
                load.num_waiting_reqs
                    .saturating_add(load.num_running_reqs)
                    .saturating_add(recent_dispatches)
                    .try_into()
                    .unwrap_or(usize::MAX)
            })
            .unwrap_or_else(|| {
                self.local_active_by_worker_id
                    .get(worker.id.0.as_str())
                    .copied()
                    .unwrap_or(usize::MAX)
            })
    }
    fn min_by_pressure_key(
        &self,
        candidates: Vec<Arc<Worker>>,
        compare: impl Fn(&Self, &PressureKey<'a>, &PressureKey<'a>) -> Ordering,
    ) -> Option<Arc<Worker>> {
        let mut candidates = candidates.into_iter();
        let mut best = candidates.next()?;
        let mut best_key = self.pressure_key(&best);
        for candidate in candidates {
            let key = self.pressure_key(&candidate);
            if compare(self, &key, &best_key).is_lt() {
                best = candidate;
                best_key = key;
            }
        }
        Some(best)
    }
}

struct PressureKey<'a> {
    load: Option<&'a NativeCacheWorkerLoad>,
    local_active: usize,
}

fn range_fallback(
    range: &CandidateRange<'_>,
    legal: &[Arc<Worker>],
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
    queue_limit: Option<u64>,
) -> Option<(Arc<Worker>, DecisionReason)> {
    let admitted = legal
        .iter()
        .filter(|worker| contains_worker(range, worker))
        .filter(|worker| is_prefill_admitted(range, worker, request_input_tokens, snapshot))
        .cloned()
        .collect::<Vec<_>>();
    let loads = FreshLoadLookup::new(Some(snapshot), admitted.iter());
    // Two-tier under the queue gate: least-pressured worker that is not
    // queueing, and only when every admitted worker is queueing,
    // least-pressured overall. Both tiers are load-bearing. Pressure ranks by
    // queue tokens / depth while the gate reads queue length, and those
    // disagree exactly where the gate earns its keep — a shallow worker with
    // a backlog is the fleet minimum BY PRESSURE, so a single-tier fallback
    // hands the request straight back to the cache home the gate just
    // rejected. The second tier is what keeps an all-queueing fleet routable
    // instead of failing every request.
    let unqueued = admitted
        .iter()
        .filter(|worker| queue_gate_admits(snapshot, worker, queue_limit))
        .cloned()
        .collect::<Vec<_>>();
    let pool = if unqueued.is_empty() {
        admitted
    } else {
        unqueued
    };
    loads
        .min_by_pressure_key(pool, FreshLoadLookup::compare_prefill_keys)
        .map(|worker| (worker, DecisionReason::RangeFallback))
}

fn legal_prefill_candidates(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
) -> Vec<Arc<Worker>> {
    proposal
        .eligible_workers
        .as_deref()
        .unwrap_or(range.workers)
        .iter()
        .filter(|worker| contains_worker(range, worker))
        .cloned()
        .collect()
}

fn decode_domain_fallback(
    domain: &CandidateDomain,
    request_kv_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> Option<(Arc<Worker>, DecisionReason)> {
    let admitted = domain
        .workers
        .iter()
        .filter(|worker| is_decode_admitted(worker, request_kv_tokens, snapshot))
        .cloned()
        .collect::<Vec<_>>();
    let loads = FreshLoadLookup::new(Some(snapshot), admitted.iter());
    loads
        .min_by_pressure_key(admitted, FreshLoadLookup::compare_decode_keys)
        .map(|worker| (worker, DecisionReason::RangeFallback))
}

/// Compares prefill pressure by queue time when available, then by the V3 load tuple.
pub(crate) fn compare_prefill_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&EngineLoadSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_native_cache_load_for_url(&left.url)?,
            snapshot.fresh_native_cache_load_for_url(&right.url)?,
        ))
    }) {
        Some((left_load, right_load)) => compare_prefill_load(left_load, right_load)
            .then_with(|| left.active_load().cmp(&right.active_load())),
        None => left.active_load().cmp(&right.active_load()),
    }
}

fn prefill_pressure_key(load: &NativeCacheWorkerLoad) -> (u64, u64, u64) {
    (
        load.num_waiting_uncached_tokens,
        load.num_waiting_reqs,
        load.num_running_reqs,
    )
}

fn compare_prefill_load(left: &NativeCacheWorkerLoad, right: &NativeCacheWorkerLoad) -> Ordering {
    match (
        left.estimated_prefill_queue_ms,
        right.estimated_prefill_queue_ms,
    ) {
        (Some(left_ms), Some(right_ms)) => left_ms
            .total_cmp(&right_ms)
            .then_with(|| prefill_pressure_key(left).cmp(&prefill_pressure_key(right))),
        _ => prefill_pressure_key(left).cmp(&prefill_pressure_key(right)),
    }
}

/// Compares decode pressure from LoadStat without treating unknown capacity as zero.
pub(crate) fn compare_decode_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&EngineLoadSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_native_cache_load_for_url(&left.url)?,
            snapshot.fresh_native_cache_load_for_url(&right.url)?,
        ))
    }) {
        Some((left_load, right_load)) => compare_decode_load(left_load, right_load)
            .then_with(|| left.active_load().cmp(&right.active_load())),
        None => left.active_load().cmp(&right.active_load()),
    }
}

fn compare_decode_load(left: &NativeCacheWorkerLoad, right: &NativeCacheWorkerLoad) -> Ordering {
    let kv_usage = match (left.max_total_num_tokens, right.max_total_num_tokens) {
        (left_cap, right_cap) if left_cap > 0 && right_cap > 0 => u128::from(left.num_used_tokens)
            .saturating_mul(u128::from(right_cap))
            .cmp(&u128::from(right.num_used_tokens).saturating_mul(u128::from(left_cap))),
        _ => Ordering::Equal,
    };
    left.num_waiting_reqs
        .cmp(&right.num_waiting_reqs)
        .then_with(|| left.num_running_reqs.cmp(&right.num_running_reqs))
        .then(kv_usage)
        .then_with(|| left.num_used_tokens.cmp(&right.num_used_tokens))
}

fn pressure_guard_prefers_backup(
    primary: &Arc<Worker>,
    backup: &Arc<Worker>,
    hints: &GuardHints,
    snapshot: &EngineLoadSnapshot,
) -> bool {
    if !hints.enable_pressure_guard {
        return false;
    }
    let (Some(primary_load), Some(backup_load)) = (
        snapshot.fresh_native_cache_load_for_url(&primary.url),
        snapshot.fresh_native_cache_load_for_url(&backup.url),
    ) else {
        return false;
    };
    if let Some(absolute_threshold_ms) = hints.pressure_abs_threshold_ms.filter(|_| {
        primary_load.estimated_prefill_queue_ms.is_some()
            && backup_load.estimated_prefill_queue_ms.is_some()
    }) {
        let primary_ms = primary_load
            .estimated_prefill_queue_ms
            .expect("availability was checked");
        let backup_ms = backup_load
            .estimated_prefill_queue_ms
            .expect("availability was checked");
        return primary_ms - backup_ms > absolute_threshold_ms
            && primary_ms > backup_ms * hints.pressure_rel_threshold;
    }
    primary_load
        .num_waiting_uncached_tokens
        .saturating_sub(backup_load.num_waiting_uncached_tokens)
        > hints.pressure_abs_threshold_tokens
        && primary_load.num_waiting_uncached_tokens as f64
            > backup_load.num_waiting_uncached_tokens as f64 * hints.pressure_rel_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
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

    fn snapshot(entries: &[(&Arc<Worker>, u64, u64, u64, u64)]) -> EngineLoadSnapshot {
        EngineLoadSnapshot::from_native_cache_workers(
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
    fn capacity_rejects_only_when_the_published_capacity_is_exceeded() {
        let full = worker("full");
        let unknown = worker("unknown");
        let workers = vec![Arc::clone(&full), Arc::clone(&unknown)];
        let range = CandidateRange::global(&workers);
        let loads = snapshot(&[(&full, 0, 0, 90, 100), (&unknown, 0, 0, 0, 1_000)]);

        assert!(resolve_prefill(
            &range,
            &SelectionProposal::primary(Arc::clone(&full)),
            20,
            &loads,
            None,
        )
        .is_some());
        assert_eq!(
            resolve_prefill(&range, &SelectionProposal::primary(full), 20, &loads, None)
                .expect("fallback selects the admitted worker")
                .selected
                .id,
            unknown.id
        );
    }

    #[test]
    fn all_capacity_rejected_falls_back_to_power_of_two_within_eligible_domain() {
        let primary = worker("primary");
        let backup = worker("backup");
        let filtered = worker("filtered");
        let workers = vec![
            Arc::clone(&primary),
            Arc::clone(&backup),
            Arc::clone(&filtered),
        ];
        let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup))
            .with_eligible_workers(vec![Arc::clone(&primary), Arc::clone(&backup)]);
        let loads = snapshot(&[
            (&primary, 0, 0, 100, 100),
            (&backup, 0, 0, 100, 100),
            (&filtered, 0, 0, 0, 100),
        ]);

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            None,
        )
        .expect("capacity exhaustion must degrade within the legal domain");

        assert!(matches!(
            decision.selected.id.0.as_str(),
            "primary" | "backup"
        ));
        assert_eq!(decision.reason, DecisionReason::CapacityFallbackPowerOfTwo);
    }

    #[test]
    fn capacity_fallback_uses_the_explicit_snapshot_for_power_of_two() {
        let primary = worker("primary");
        let backup = worker("backup");
        let workers = vec![Arc::clone(&primary), Arc::clone(&backup)];
        let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));
        let explicit = snapshot(&[(&primary, 0, 0, 100, 100), (&backup, 0, 10, 100, 100)]);
        let opposite = snapshot(&[(&primary, 0, 10, 100, 100), (&backup, 0, 0, 100, 100)]);
        let opposite_decision = select_with_snapshot(&workers, Some(&opposite))
            .expect("the opposite snapshot has the same legal workers");
        assert_eq!(opposite_decision.id, backup.id);

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &explicit,
            None,
        )
        .expect("capacity exhaustion must degrade to Power-of-Two");

        assert_eq!(decision.selected.id, primary.id);
        assert_eq!(decision.load_snapshot_version, explicit.version);
    }

    #[test]
    fn prefill_pressure_uses_waiting_then_running_requests() {
        let busy = worker("busy");
        let idle = worker("idle");
        let loads = snapshot(&[(&busy, 1, 8, 10, 100), (&idle, 9, 2, 90, 100)]);
        assert!(compare_prefill_pressure(&busy, &idle, Some(&loads)).is_gt());
    }

    #[test]
    fn missing_snapshot_uses_local_active_load() {
        let left = worker("left");
        let right = worker("right");
        let _guard = left.load_guard();
        assert!(compare_prefill_pressure(&left, &right, None).is_gt());
    }

    #[test]
    fn complete_monitor_pressure_guard_overrides_a_near_cache_gain() {
        let congested = worker("congested");
        let idle = worker("idle");
        let proposal = CacheCandidateProposal {
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
            cache_switch_margin_tokens: 32,
            enable_pressure_guard: true,
            pressure_abs_threshold_tokens: 100,
            pressure_abs_threshold_ms: None,
            pressure_rel_threshold: 1.5,
            worker_queue_limit: None,
            saturation_queue_floor: None,
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

    fn queue_gate_proposal(
        candidates: Vec<CacheCandidate>,
        limit: Option<u64>,
    ) -> CacheCandidateProposal {
        CacheCandidateProposal {
            candidates,
            worker_queue_limit: limit,
            ..Default::default()
        }
    }

    fn saturation_proposal(
        candidates: Vec<CacheCandidate>,
        limit: Option<u64>,
        floor: Option<u64>,
    ) -> CacheCandidateProposal {
        CacheCandidateProposal {
            candidates,
            worker_queue_limit: limit,
            saturation_queue_floor: floor,
            ..Default::default()
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

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &[]);

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

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &[]);

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

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &[]);

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
        let owner = worker("owner");
        let proposal = queue_gate_proposal(vec![candidate(&owner, 10, 9)], Some(4));
        let loads = snapshot(&[(&owner, 1, 9, 10, 10_000)]);

        let resolution = resolve_cache_candidates(&proposal, 100, &loads, &[]);

        assert!(resolution.decision.is_none());
        assert_eq!(resolution.queue_gate_rejected_candidates, 1);
        assert_eq!(resolution.queue_gate_best_rejected_blocks, 9);
        assert_eq!(resolution.admission_rejected_candidates, 0);
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

        assert!(resolution.decision.is_none());
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

    #[test]
    fn range_fallback_prefers_an_unqueued_worker_over_a_shallower_queueing_one() {
        // The primary fails KV-capacity admission so selection reaches the
        // range fallback. The queueing worker is the fallback minimum BY
        // PRESSURE (1 waiting uncached token), so a single-tier fallback
        // would pick it — handing the request straight back to a worker the
        // gate just rejected.
        let primary = worker("primary");
        let shallow_queued = worker("shallow_queued");
        let busy_unqueued = worker("busy_unqueued");
        let workers = vec![
            Arc::clone(&primary),
            Arc::clone(&shallow_queued),
            Arc::clone(&busy_unqueued),
        ];
        let proposal = SelectionProposal::primary(Arc::clone(&primary));
        let load = |waiting: u64, waiting_uncached: u64, total: u64, max_total: u64| {
            NativeCacheWorkerLoad {
                num_running_reqs: 0,
                num_waiting_reqs: waiting,
                num_waiting_uncached_tokens: waiting_uncached,
                num_used_tokens: total,
                num_total_tokens: total,
                max_total_num_tokens: max_total,
                max_running_requests: 64,
                prefill_throughput_tokens_per_s: None,
                estimated_prefill_queue_ms: None,
                captured_at: Instant::now(),
            }
        };
        // Queue depth and pressure disagree by construction: shallow_queued
        // waits 5 (over the limit) behind 1 uncached token, busy_unqueued
        // waits 3 (under the limit) behind 1000 uncached tokens. The primary
        // is KV-full, so it is not admitted at all.
        let loads = EngineLoadSnapshot::from_native_cache_workers(
            7,
            [
                (primary.url.clone(), load(0, 0, 10_000, 10_000)),
                (shallow_queued.url.clone(), load(5, 1, 10, 10_000)),
                (busy_unqueued.url.clone(), load(3, 1_000, 10, 10_000)),
            ]
            .into_iter()
            .collect(),
        );

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            Some(4),
        )
        .expect("an admitted worker exists");

        assert_eq!(
            decision.selected.id, busy_unqueued.id,
            "the fallback must prefer the unqueued worker even though the \
             queueing one is the pressure minimum"
        );
        assert_eq!(decision.reason, DecisionReason::RangeFallback);
    }

    #[test]
    fn queue_gate_demotes_a_queueing_primary_that_capacity_would_admit() {
        // The whole point of the gate is that the cache-affinity fallback
        // must not hand the request back to a queueing worker. The primary
        // here has plenty of KV capacity, so without the gate the
        // `(true, _, _)` arm returns it unconditionally and the gate is
        // bypassed on the single most common path.
        let queued_primary = worker("queued_primary");
        let unqueued = worker("unqueued");
        let workers = vec![Arc::clone(&queued_primary), Arc::clone(&unqueued)];
        let proposal = SelectionProposal::primary(Arc::clone(&queued_primary));
        let loads = snapshot(&[
            (&queued_primary, 0, 6, 10, 10_000),
            (&unqueued, 0, 0, 10, 10_000),
        ]);

        let ungated = resolve_prefill_admitted(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            None,
        )
        .expect("without a limit the primary is admitted");
        assert_eq!(ungated.selected.id, queued_primary.id);
        assert_eq!(ungated.reason, DecisionReason::Primary);

        let gated = resolve_prefill_admitted(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            Some(4),
        )
        .expect("the unqueued worker takes over");
        assert_eq!(gated.selected.id, unqueued.id);
        assert_eq!(gated.reason, DecisionReason::RangeFallback);
    }

    #[test]
    fn queue_gate_still_routes_when_the_only_admitted_worker_is_queueing() {
        // Demotion must never become rejection: with every admitted worker
        // over the limit, the second fallback tier keeps the request routable.
        let only = worker("only");
        let workers = vec![Arc::clone(&only)];
        let proposal = SelectionProposal::primary(Arc::clone(&only));
        let loads = snapshot(&[(&only, 0, 9, 10, 10_000)]);

        let decision = resolve_prefill_admitted(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            Some(4),
        )
        .expect("an all-queueing fleet must still route");

        assert_eq!(decision.selected.id, only.id);
        assert_eq!(decision.reason, DecisionReason::RangeFallback);
    }

    #[test]
    fn range_fallback_keeps_an_all_queueing_fleet_routable() {
        let primary = worker("primary");
        let left = worker("left");
        let right = worker("right");
        let workers = vec![Arc::clone(&primary), Arc::clone(&left), Arc::clone(&right)];
        let proposal = SelectionProposal::primary(Arc::clone(&primary));
        // The primary is KV-full; both fallback workers are over the limit.
        // The second tier takes the least-pressured one instead of failing
        // the request.
        let loads = snapshot(&[
            (&primary, 0, 0, 10_000, 10_000),
            (&left, 0, 9, 10, 10_000),
            (&right, 0, 5, 10, 10_000),
        ]);

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            Some(4),
        )
        .expect("an all-queueing fleet must still route");

        assert_eq!(decision.selected.id, right.id);
        assert_eq!(decision.reason, DecisionReason::RangeFallback);
    }
}
