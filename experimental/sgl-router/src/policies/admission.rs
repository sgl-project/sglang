// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared eligibility, capacity checks, and comparable worker pressure.
//!
//! Missing engine measurements fail open for capacity checks. Pressure ranking
//! uses one common measurement source for the candidate set being compared.

use crate::policies::balancing::select_with_snapshot;
use crate::policies::{GuardHints, Policy, SelectionContext, SelectionProposal};
use crate::workers::engine_reports::{EngineSnapshot, EngineWorkerLoad, NativeCacheWorkerLoad};
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

/// The queue gate, in one place. Both subtle decisions live here: the
/// boundary is `<` (a worker AT the limit is already making this request
/// wait), and an unknown queue ADMITS. The gate reads
/// [`EngineWorkerLoad::num_waiting_reqs`] because that is what the request
/// cares about, and the router-side in-flight counter cannot separate a
/// running request from a waiting one — so there is no honest substitute,
/// and the gate fails open rather than comparing the limit against a
/// different quantity.
pub(crate) fn queue_gate_admits(
    snapshot: &EngineSnapshot,
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

/// True when the gate has provably nowhere unqueued to divert to: every
/// worker in `fleet` has a fresh sample at or over the limit. An unset limit,
/// an empty fleet, or a single worker with no fresh sample all make this
/// false — an unknown queue is not a proven full one.
pub(crate) fn fleet_is_all_queued(
    snapshot: &EngineSnapshot,
    fleet: &[Arc<Worker>],
    limit: Option<u64>,
) -> bool {
    limit.is_some()
        && !fleet.is_empty()
        && fleet
            .iter()
            .all(|worker| !queue_gate_admits(snapshot, worker, limit))
}

/// Whether to use P2 when no worker passes capacity admission in this domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityFallback {
    Allowed,
    Disabled,
}

pub fn resolve_prefill(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &EngineSnapshot,
    queue_limit: Option<u64>,
    fallback: CapacityFallback,
) -> Option<FinalDecision> {
    admit_prefill_pair(range, proposal, request_input_tokens, snapshot, queue_limit).or_else(|| {
        if fallback == CapacityFallback::Disabled {
            return None;
        }
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
fn admit_prefill_pair(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &EngineSnapshot,
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
    snapshot: &EngineSnapshot,
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
pub(crate) fn has_kv_capacity(load: Option<&NativeCacheWorkerLoad>, requested_tokens: u64) -> bool {
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
    snapshot: &EngineSnapshot,
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
    snapshot: &EngineSnapshot,
) -> bool {
    has_kv_capacity(
        snapshot.fresh_native_cache_load_for_url(&worker.url),
        request_kv_tokens,
    )
}

pub(crate) fn materially_more_pressured(
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
    basic_by_worker_id: HashMap<String, &'a EngineWorkerLoad>,
    local_active_by_worker_id: HashMap<String, usize>,
    compare_engine: bool,
    compare_basic_engine: bool,
}

impl<'a> FreshLoadLookup<'a> {
    pub(crate) fn new<'w>(
        snapshot: Option<&'a EngineSnapshot>,
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

    pub(crate) fn compare_prefill_pressure_then_id(
        &self,
        left: &Arc<Worker>,
        right: &Arc<Worker>,
    ) -> Ordering {
        self.compare_prefill_pressure(left, right)
            .then_with(|| left.id.0.cmp(&right.id.0))
    }

    pub(crate) fn get(
        &self,
        worker_id: &crate::discovery::WorkerId,
    ) -> Option<&'a NativeCacheWorkerLoad> {
        self.by_worker_id.get(worker_id.0.as_str()).copied()
    }

    pub(crate) fn comparable_get(
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
    snapshot: &EngineSnapshot,
    queue_limit: Option<u64>,
) -> Option<(Arc<Worker>, DecisionReason)> {
    let admitted = legal
        .iter()
        .filter(|worker| contains_worker(range, worker))
        .filter(|worker| is_prefill_admitted(range, worker, request_input_tokens, snapshot))
        .cloned()
        .collect::<Vec<_>>();
    // Two-tier under the queue gate: least-pressured worker that is not
    // queueing, and only when every admitted worker is queueing,
    // least-pressured overall. Both tiers are load-bearing. Pressure ranks by
    // queue tokens / depth while the gate reads queue length, and those
    // disagree exactly where the gate earns its keep — a shallow worker with
    // a backlog is the fleet minimum BY PRESSURE, so a single-tier fallback
    // hands the request straight back to the cache home the gate just
    // rejected. The second tier is what keeps an all-queueing fleet routable
    // instead of failing every request.
    let pool = match queue_limit {
        // Gate disabled: the tiers coincide, so filtering would only clone
        // the vector.
        None => admitted,
        Some(_) => {
            let unqueued = admitted
                .iter()
                .filter(|worker| queue_gate_admits(snapshot, worker, queue_limit))
                .cloned()
                .collect::<Vec<_>>();
            if unqueued.is_empty() {
                admitted
            } else {
                unqueued
            }
        }
    };
    // Scoped to the pool actually ranked: an admitted-but-gated worker
    // missing native monitor data would otherwise downgrade the comparison
    // for the whole unqueued tier to router-local.
    let loads = FreshLoadLookup::new(Some(snapshot), pool.iter());
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
    snapshot: &EngineSnapshot,
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
    snapshot: Option<&EngineSnapshot>,
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
    snapshot: Option<&EngineSnapshot>,
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
    snapshot: &EngineSnapshot,
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

/// What a filter means when it has rejected every worker it was shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnEmpty {
    /// Ignore this filter when it rejects every candidate.
    Abstain,
    /// Keep the rejection: no worker is admissible.
    Hold,
}

/// A hard constraint applied before scoring.
pub trait EligibilityFilter: Send + Sync + std::fmt::Debug {
    /// Returns one admission flag per worker; `true` keeps the candidate.
    fn keep(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<bool>;

    /// Whether this constraint reads `ctx.request_tokens()`.
    fn needs_tokens(&self) -> bool {
        false
    }

    /// Controls the result when this filter rejects every candidate.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Abstain
    }
}

/// Applies ordered filters. `None` means a holding filter rejected all candidates.
pub fn apply_filters<'f>(
    filters: impl IntoIterator<Item = &'f dyn EligibilityFilter>,
    workers: &[Arc<Worker>],
    ctx: &SelectionContext<'_>,
) -> Option<Vec<Arc<Worker>>> {
    let mut alive: Vec<Arc<Worker>> = workers.to_vec();

    for filter in filters {
        if alive.is_empty() {
            break;
        }
        let flags = filter.keep(&alive, ctx);
        let on_empty = filter.on_empty();
        if flags.len() != alive.len() {
            tracing::debug!(
                filter = ?filter,
                n_workers = alive.len(),
                n_flags = flags.len(),
                "eligibility filter returned the wrong arity",
            );
            if on_empty == OnEmpty::Hold {
                return None;
            }
        }
        let untouched = alive.len() == workers.len();

        let next: Vec<Arc<Worker>> = (alive.iter().enumerate())
            .filter(|(i, _)| flags.get(*i).copied().unwrap_or(true))
            .map(|(_, w)| Arc::clone(w))
            .collect();

        if next.is_empty() {
            match on_empty {
                OnEmpty::Hold => return None,
                OnEmpty::Abstain if untouched => {
                    tracing::debug!(
                        filter = ?filter,
                        n_workers = workers.len(),
                        "eligibility filter has no eligible workers; falling back to the full candidate set",
                    );
                    continue;
                }
                OnEmpty::Abstain => {
                    tracing::debug!(
                        filter = ?filter,
                        n_alive = alive.len(),
                        "eligibility filter conflicts with a higher-priority one; yielding",
                    );
                    continue;
                }
            }
        }
        alive = next;
    }

    Some(alive)
}

/// Rejects workers at the router-local in-flight limit.
#[derive(Debug)]
pub struct Overloaded {
    max_in_flight: usize,
}

impl Overloaded {
    pub fn new(max_in_flight: usize) -> Self {
        Self { max_in_flight }
    }
}

impl EligibilityFilter for Overloaded {
    fn keep(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Vec<bool> {
        (workers.iter())
            .map(|w| w.active_load() < self.max_in_flight)
            .collect()
    }

    /// Do not route to an over-capacity worker.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Hold
    }
}

impl Policy for Overloaded {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let eligible: Vec<Arc<Worker>> = (workers.iter())
            .zip(self.keep(workers, ctx))
            .filter(|(_, ok)| *ok)
            .map(|(w, _)| Arc::clone(w))
            .collect();
        eligible
            .iter()
            .min_by_key(|w| w.active_load())
            .map(Arc::clone)
    }

    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        Some(self)
    }
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
            CapacityFallback::Allowed
        )
        .is_some());
        assert_eq!(
            resolve_prefill(
                &range,
                &SelectionProposal::primary(full),
                20,
                &loads,
                None,
                CapacityFallback::Allowed
            )
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
            CapacityFallback::Allowed,
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
            CapacityFallback::Allowed,
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
        let loads = EngineSnapshot::from_native_cache_workers(
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
            CapacityFallback::Allowed,
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

        let ungated = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            None,
            CapacityFallback::Disabled,
        )
        .expect("without a limit the primary is admitted");
        assert_eq!(ungated.selected.id, queued_primary.id);
        assert_eq!(ungated.reason, DecisionReason::Primary);

        let gated = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            Some(4),
            CapacityFallback::Disabled,
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

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            Some(4),
            CapacityFallback::Disabled,
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
            CapacityFallback::Allowed,
        )
        .expect("an all-queueing fleet must still route");

        assert_eq!(decision.selected.id, right.id);
        assert_eq!(decision.reason, DecisionReason::RangeFallback);
    }
}

#[cfg(test)]
mod overloaded_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::scoring::refs;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    #[test]
    fn the_cap_is_a_strict_ceiling() {
        assert!(!Overloaded::new(3).needs_load_snapshot());
        let ws = vec![worker("idle"), worker("under"), worker("at")];
        let _under: Vec<_> = (0..2).map(|_| ws[1].load_guard()).collect();
        let _at: Vec<_> = (0..3).map(|_| ws[2].load_guard()).collect();

        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert_eq!(
            Overloaded::new(3).keep(&ws, &ctx),
            vec![true, true, false],
            "load 3 against a cap of 3 is over",
        );
    }

    #[test]
    fn a_full_fleet_refuses_rather_than_picking_the_least_bad() {
        let ws = vec![worker("a"), worker("b")];
        let _a: Vec<_> = (0..5).map(|_| ws[0].load_guard()).collect();
        let _b: Vec<_> = (0..9).map(|_| ws[1].load_guard()).collect();

        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let full: Vec<Box<dyn EligibilityFilter>> = vec![Box::new(Overloaded::new(4))];
        assert!(
            apply_filters(refs(&full), &ws, &ctx).is_none(),
            "both over the cap, and the filter Holds",
        );

        let some: Vec<Box<dyn EligibilityFilter>> = vec![Box::new(Overloaded::new(6))];
        let out = apply_filters(refs(&some), &ws, &ctx).expect("a is under the cap");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].url, ws[0].url);
    }

    #[test]
    fn hold_does_not_yield_to_a_later_filter() {
        #[derive(Debug)]
        struct AdmitAll;
        impl EligibilityFilter for AdmitAll {
            fn keep(&self, ws: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<bool> {
                vec![true; ws.len()]
            }
        }
        let ws = vec![worker("a")];
        let _busy: Vec<_> = (0..9).map(|_| ws[0].load_guard()).collect();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let chain: Vec<Box<dyn EligibilityFilter>> =
            vec![Box::new(Overloaded::new(2)), Box::new(AdmitAll)];
        assert!(apply_filters(refs(&chain), &ws, &ctx).is_none());
    }
}

#[cfg(test)]
mod proposal_tests {
    use super::CapacityFallback;
    use crate::policies::admission::resolve_prefill;
    use crate::policies::admission::CandidateRange;
    use crate::policies::admission::DecisionReason;
    use crate::policies::admission::FreshLoadLookup;
    use crate::policies::test_support::snapshot;
    use crate::policies::test_support::worker;
    use crate::policies::test_support::TestEngineLoad;
    use crate::policies::*;
    #[test]
    fn decode_pressure_tie_is_not_broken_by_worker_id() {
        let a = worker("a");
        let z = worker("z");
        assert_eq!(
            admission::compare_decode_pressure(&a, &z, None),
            std::cmp::Ordering::Equal,
            "P2 must preserve random sampling when observable pressure is equal"
        );
    }

    #[test]
    fn mixed_freshness_uses_one_captured_local_level_for_the_candidate_set() {
        let aggregate_idle = worker("aggregate-idle");
        let aggregate_busy = worker("aggregate-busy");
        let stale = worker("stale");
        aggregate_idle
            .active_requests
            .store(5, std::sync::atomic::Ordering::Relaxed);
        aggregate_busy
            .active_requests
            .store(1, std::sync::atomic::Ordering::Relaxed);
        let snapshot = snapshot(&[
            (
                &aggregate_idle,
                TestEngineLoad {
                    num_waiting_reqs: 0,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &aggregate_busy,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let lookup =
            FreshLoadLookup::new(Some(&snapshot), [&aggregate_idle, &aggregate_busy, &stale]);
        assert!(lookup.get(&aggregate_idle.id).is_some());
        assert!(lookup.get(&stale.id).is_none());
        assert_eq!(
            lookup.compare_prefill_pressure(&aggregate_idle, &aggregate_busy),
            std::cmp::Ordering::Greater,
            "one stale member makes the complete candidate set compare by the captured local level"
        );
    }

    #[test]
    fn admission_uses_admitted_backup_before_scanning_candidate_range() {
        let primary = worker("primary");
        let backup = worker("backup");
        let fallback = worker("fallback");
        let workers = vec![
            Arc::clone(&primary),
            Arc::clone(&backup),
            Arc::clone(&fallback),
        ];
        let snapshot = snapshot(&[
            (
                &primary,
                TestEngineLoad {
                    num_running_reqs: 4,
                    num_tokens: 990,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                TestEngineLoad {
                    num_tokens: 10,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &fallback,
                TestEngineLoad {
                    num_tokens: 10,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let range = CandidateRange::global(&workers);
        let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

        let decision = resolve_prefill(
            &range,
            &proposal,
            32,
            &snapshot,
            None,
            CapacityFallback::Allowed,
        )
        .expect("an admitted backup must be selected");

        assert_eq!(decision.selected.id, backup.id);
        assert_eq!(decision.reason, DecisionReason::BackupPrimaryAdmission);
    }

    #[test]
    fn missing_engine_snapshot_does_not_hard_reject_a_registry_healthy_primary() {
        let primary = worker("primary");
        let workers = vec![Arc::clone(&primary)];
        let snapshot = EngineSnapshot::default();

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &SelectionProposal::primary(Arc::clone(&primary)),
            1_000_000,
            &snapshot,
            None,
            CapacityFallback::Allowed,
        )
        .expect("disabled reporting must preserve the healthy registry candidate");

        assert_eq!(decision.selected.id, primary.id);
        assert_eq!(decision.reason, DecisionReason::Primary);
    }

    #[test]
    fn prefill_pair_keeps_primary_when_both_workers_fit_capacity() {
        let primary = worker("primary");
        let backup = worker("backup");
        let workers = vec![Arc::clone(&primary), Arc::clone(&backup)];
        let snapshot = snapshot(&[
            (
                &primary,
                TestEngineLoad {
                    num_waiting_reqs: 200,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                TestEngineLoad {
                    num_waiting_reqs: 20,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let proposal = SelectionProposal::with_backup(primary, backup);

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            80,
            &snapshot,
            None,
            CapacityFallback::Allowed,
        )
        .expect("both candidates fit capacity");

        assert_eq!(decision.reason, DecisionReason::Primary);
    }

    #[test]
    fn admission_scans_range_only_after_primary_and_backup_both_fail() {
        let primary = worker("primary");
        let backup = worker("backup");
        let fallback = worker("fallback");
        let workers = vec![
            Arc::clone(&primary),
            Arc::clone(&backup),
            Arc::clone(&fallback),
        ];
        let snapshot = snapshot(&[
            (
                &primary,
                TestEngineLoad {
                    num_running_reqs: 4,
                    num_tokens: 990,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                TestEngineLoad {
                    num_tokens: 990,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &fallback,
                TestEngineLoad {
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let proposal = SelectionProposal::with_backup(primary, backup);

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &snapshot,
            None,
            CapacityFallback::Allowed,
        )
        .expect("an admitted range fallback must be selected");

        assert_eq!(decision.selected.id, fallback.id);
        assert_eq!(decision.reason, DecisionReason::RangeFallback);
    }
}
