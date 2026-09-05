// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared admission and candidate comparison for Prefill and Decode.
//!
//! Decisions use only fields published in `LoadStat`.

use crate::policies::engine_load::{EngineLoadSnapshot, EngineWorkerLoad};
use crate::policies::power_of_two::select_with_snapshot;
use crate::policies::{CacheCandidate, CacheCandidateProposal, SelectionProposal};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// A Prefill candidate range and its optional pending-token budget.
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

/// A role-specific candidate domain resolved before policy evaluation.
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
    /// Both Decode candidates were admitted; the lower-pressure backup won.
    BackupLoadComparison,
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

/// Resolves a bounded cache-candidate set, using pressure to break near ties.
pub fn resolve_cache_candidates(
    proposal: &CacheCandidateProposal,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> Option<FinalDecision> {
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
    let work_floor = admitted
        .iter()
        .copied()
        .min_by_key(|candidate| candidate.uncached_tokens)?;
    let near_tie_ceiling = work_floor
        .uncached_tokens
        .saturating_add(proposal.cache_switch_margin_tokens);
    let mut winner = work_floor;
    for candidate in admitted {
        if candidate.uncached_tokens <= near_tie_ceiling
            && compare_cache_candidates(winner, candidate, &loads).is_gt()
        {
            winner = candidate;
        }
    }
    Some(FinalDecision {
        selected: Arc::clone(&winner.worker),
        primary: Arc::clone(&winner.worker),
        backup: None,
        reason: DecisionReason::CacheCandidate,
        candidate_range_id: winner.candidate_range_id.clone(),
        load_snapshot_version: snapshot.version,
    })
}

pub fn resolve_prefill(
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
        && is_prefill_admitted(&proposal.primary, request_input_tokens, snapshot);
    let backup_admitted = backup.as_ref().is_some_and(|worker| {
        is_proposal_worker_eligible(proposal, worker)
            && is_prefill_admitted(worker, request_input_tokens, snapshot)
    });

    let (selected, reason) = match (primary_admitted, backup.as_ref(), backup_admitted) {
        (true, _, _) => (Arc::clone(&proposal.primary), DecisionReason::Primary),
        (false, Some(backup), true) => (Arc::clone(backup), DecisionReason::BackupPrimaryAdmission),
        _ => {
            let legal = legal_prefill_candidates(range, proposal);
            range_fallback(&legal, request_input_tokens, snapshot).or_else(|| {
                select_with_snapshot(&legal, Some(snapshot))
                    .map(|worker| (worker, DecisionReason::CapacityFallbackPowerOfTwo))
            })?
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
                (Arc::clone(backup), DecisionReason::BackupLoadComparison)
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

/// A zero LoadStat capacity is unknown and does not reject a candidate.
fn has_kv_capacity(load: Option<&EngineWorkerLoad>, requested_tokens: u64) -> bool {
    let Some(load) = load else {
        return true;
    };
    load.max_total_num_tokens == 0
        || load.num_tokens.saturating_add(requested_tokens) <= load.max_total_num_tokens
}

fn is_prefill_admitted(
    worker: &Arc<Worker>,
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> bool {
    has_kv_capacity(
        snapshot.fresh_load_for_url(&worker.url),
        request_input_tokens,
    )
}

fn is_decode_admitted(
    worker: &Arc<Worker>,
    request_kv_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> bool {
    has_kv_capacity(snapshot.fresh_load_for_url(&worker.url), request_kv_tokens)
}

fn is_cache_candidate_admitted(
    candidate: &CacheCandidate,
    request_input_tokens: u64,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    has_kv_capacity(loads.get(&candidate.worker.id), request_input_tokens)
}

fn compare_cache_candidates(
    left: &CacheCandidate,
    right: &CacheCandidate,
    loads: &FreshLoadLookup<'_>,
) -> Ordering {
    left.uncached_tokens
        .cmp(&right.uncached_tokens)
        .then_with(|| loads.compare_prefill_pressure(&left.worker, &right.worker))
        .then_with(|| left.worker.id.0.cmp(&right.worker.id.0))
}

/// Per-request lookup that uses engine pressure only for a complete fresh set.
pub(crate) struct FreshLoadLookup<'a> {
    by_worker_id: HashMap<String, &'a EngineWorkerLoad>,
    local_active_by_worker_id: HashMap<String, usize>,
    compare_engine: bool,
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
                        .fresh_load_for_url(&worker.url)
                        .map(|load| (worker.id.0.clone(), load))
                })
            })
            .collect::<HashMap<_, _>>();
        let compare_engine = !local_active_by_worker_id.is_empty()
            && by_worker_id.len() == local_active_by_worker_id.len();
        Self {
            by_worker_id,
            local_active_by_worker_id,
            compare_engine,
        }
    }

    pub(crate) fn get(
        &self,
        worker_id: &crate::discovery::WorkerId,
    ) -> Option<&'a EngineWorkerLoad> {
        self.by_worker_id.get(worker_id.0.as_str()).copied()
    }

    fn comparable_get(
        &self,
        worker_id: &crate::discovery::WorkerId,
    ) -> Option<&'a EngineWorkerLoad> {
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
            (Some(left_load), Some(right_load)) => prefill_pressure_key(left_load)
                .cmp(&prefill_pressure_key(right_load))
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

    /// Returns corrected engine queue depth for a complete fresh set, otherwise
    /// local load.
    pub(crate) fn score_load(&self, worker: &Arc<Worker>) -> usize {
        self.comparable_get(&worker.id)
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
    load: Option<&'a EngineWorkerLoad>,
    local_active: usize,
}

fn range_fallback(
    legal: &[Arc<Worker>],
    request_input_tokens: u64,
    snapshot: &EngineLoadSnapshot,
) -> Option<(Arc<Worker>, DecisionReason)> {
    let admitted = legal
        .iter()
        .filter(|worker| is_prefill_admitted(worker, request_input_tokens, snapshot))
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

/// Compares Prefill pressure by waiting requests, running requests, and KV use.
pub(crate) fn compare_prefill_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&EngineLoadSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_load_for_url(&left.url)?,
            snapshot.fresh_load_for_url(&right.url)?,
        ))
    }) {
        Some((left_load, right_load)) => prefill_pressure_key(left_load)
            .cmp(&prefill_pressure_key(right_load))
            .then_with(|| left.active_load().cmp(&right.active_load())),
        None => left.active_load().cmp(&right.active_load()),
    }
}

fn prefill_pressure_key(load: &EngineWorkerLoad) -> (u64, u64, u64, u64) {
    (
        load.num_waiting_reqs,
        load.num_running_reqs,
        load.num_tokens,
        load.max_total_num_tokens,
    )
}

/// Compares Decode pressure using only LoadStat values.
pub(crate) fn compare_decode_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&EngineLoadSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_load_for_url(&left.url)?,
            snapshot.fresh_load_for_url(&right.url)?,
        ))
    }) {
        Some((left_load, right_load)) => compare_decode_load(left_load, right_load)
            .then_with(|| left.active_load().cmp(&right.active_load())),
        None => left.active_load().cmp(&right.active_load()),
    }
}

fn compare_decode_load(left: &EngineWorkerLoad, right: &EngineWorkerLoad) -> Ordering {
    let kv_usage = match (left.max_total_num_tokens, right.max_total_num_tokens) {
        (left_cap, right_cap) if left_cap > 0 && right_cap > 0 => u128::from(left.num_tokens)
            .saturating_mul(u128::from(right_cap))
            .cmp(&u128::from(right.num_tokens).saturating_mul(u128::from(left_cap))),
        _ => Ordering::Equal,
    };
    left.num_waiting_reqs
        .cmp(&right.num_waiting_reqs)
        .then_with(|| left.num_running_reqs.cmp(&right.num_running_reqs))
        .then(kv_usage)
        .then_with(|| left.num_tokens.cmp(&right.num_tokens))
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
        EngineLoadSnapshot::from_workers(
            7,
            entries
                .iter()
                .map(|(worker, running, waiting, used, capacity)| {
                    (
                        worker.url.clone(),
                        EngineWorkerLoad {
                            num_running_reqs: *running,
                            num_waiting_reqs: *waiting,
                            num_tokens: *used,
                            max_total_num_tokens: *capacity,
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
        let loads = snapshot(&[(&full, 0, 0, 90, 100), (&unknown, 0, 0, 90, 0)]);

        assert!(resolve_prefill(
            &range,
            &SelectionProposal::primary(Arc::clone(&full)),
            20,
            &loads
        )
        .is_some());
        assert_eq!(
            resolve_prefill(&range, &SelectionProposal::primary(full), 20, &loads)
                .expect("fallback selects unknown-capacity worker")
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
}
