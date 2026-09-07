// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared capacity admission and pressure guards for prefill and decode.
//!
//! Native Cache-Aware uses monitor-backed admission only when every expected
//! DP rank has a fresh, complete #34608 ZMQ sample. Otherwise it falls back to
//! Router-local load.

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
    pub admission_evaluated_candidates: u64,
    pub admission_rejected_candidates: u64,
    pub pressure_guard_compared_pairs: u64,
    pub pressure_guard_overrides: u64,
}

/// Selects a worker from bounded cache candidates and records guard coverage.
pub fn resolve_cache_candidates(
    proposal: &CacheCandidateProposal,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> CacheCandidateResolution {
    let loads = FreshLoadLookup::new(
        Some(snapshot),
        proposal
            .candidates
            .iter()
            .map(|candidate| &candidate.worker),
    );
    let admitted: Vec<&CacheCandidate> = proposal
        .candidates
        .iter()
        .filter(|candidate| is_cache_candidate_admitted(candidate, request_input_tokens, &loads))
        .collect();
    let admission_rejected_candidates =
        proposal.candidates.len().saturating_sub(admitted.len()) as u64;
    let Some(work_floor) = admitted
        .iter()
        .copied()
        .min_by_key(|candidate| candidate.uncached_tokens)
    else {
        return CacheCandidateResolution {
            decision: None,
            prefill_pressure_source: loads.prefill_pressure_source(),
            admission_evaluated_candidates: proposal.candidates.len() as u64,
            admission_rejected_candidates,
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
        admission_evaluated_candidates: proposal.candidates.len() as u64,
        admission_rejected_candidates,
        pressure_guard_compared_pairs,
        pressure_guard_overrides,
    }
}

pub fn resolve_prefill(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> Option<FinalDecision> {
    resolve_prefill_admitted(range, proposal, request_input_tokens, snapshot).or_else(|| {
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
    })
}

/// Resolves prefill admission without overcommitting a full candidate range.
pub fn resolve_prefill_admitted(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> Option<FinalDecision> {
    if !contains_worker(range, &proposal.primary) {
        return None;
    }
    let backup = proposal
        .backup
        .as_ref()
        .filter(|worker| contains_worker(range, worker))
        .cloned();
    let primary_admitted = is_proposal_worker_eligible(proposal, &proposal.primary)
        && is_prefill_admitted(range, &proposal.primary, request_input_tokens, snapshot);
    let backup_admitted = backup.as_ref().is_some_and(|worker| {
        is_proposal_worker_eligible(proposal, worker)
            && is_prefill_admitted(range, worker, request_input_tokens, snapshot)
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
            range_fallback(range, &legal, request_input_tokens, snapshot)?
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
) -> Option<(Arc<Worker>, DecisionReason)> {
    let admitted = legal
        .iter()
        .filter(|worker| contains_worker(range, worker))
        .filter(|worker| is_prefill_admitted(range, worker, request_input_tokens, snapshot))
        .cloned()
        .collect::<Vec<_>>();
    let loads = FreshLoadLookup::new(Some(snapshot), admitted.iter());
    loads
        .min_by_pressure_key(admitted, FreshLoadLookup::compare_prefill_keys)
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
            &loads
        )
        .is_some());
        assert_eq!(
            resolve_prefill(&range, &SelectionProposal::primary(full), 20, &loads)
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

        let decision = resolve_prefill(&CandidateRange::global(&workers), &proposal, 32, &loads)
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

        let decision = resolve_prefill(&CandidateRange::global(&workers), &proposal, 32, &explicit)
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
                    candidate_range_id: "global".into(),
                    max_pending_prefill_tokens: None,
                },
                CacheCandidate {
                    worker: Arc::clone(&idle),
                    matched_prefix_tokens: 80,
                    uncached_tokens: 20,
                    candidate_range_id: "global".into(),
                    max_pending_prefill_tokens: None,
                },
            ],
            cache_switch_margin_tokens: 32,
            enable_pressure_guard: true,
            pressure_abs_threshold_tokens: 100,
            pressure_abs_threshold_ms: None,
            pressure_rel_threshold: 1.5,
        };
        let loads = snapshot(&[
            (&congested, 1, 1_000, 10, 10_000),
            (&idle, 1, 10, 10, 10_000),
        ]);

        let resolution = resolve_cache_candidates(&proposal, 100, &loads);
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
}
