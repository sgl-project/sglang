// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped view over one captured load snapshot: capacity checks and
//! prefill/decode pressure ordering with Router-local fallback.

use crate::policies::state::engine_load::{
    EngineLoadSnapshot, EngineLoadTable, EngineWorkerLoad, NativeCacheWorkerLoad,
};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// Applies snapshot-backed capacity admission when native monitor data is complete.
/// Workers without monitor data remain eligible and use Router-local ordering.
pub(crate) fn has_kv_capacity(load: Option<&NativeCacheWorkerLoad>, requested_tokens: u64) -> bool {
    let Some(load) = load else {
        return true;
    };
    load.num_running_reqs.saturating_add(1) <= load.max_running_requests
        && load.num_total_tokens.saturating_add(requested_tokens) <= load.max_total_num_tokens
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

    pub(crate) fn compare_prefill_keys(
        &self,
        left: &PressureKey<'a>,
        right: &PressureKey<'a>,
    ) -> Ordering {
        match (left.load, right.load) {
            (Some(left_load), Some(right_load)) => compare_prefill_load(left_load, right_load)
                .then_with(|| left.local_active.cmp(&right.local_active)),
            _ => left.local_active.cmp(&right.local_active),
        }
    }

    pub(crate) fn compare_decode_keys(
        &self,
        left: &PressureKey<'a>,
        right: &PressureKey<'a>,
    ) -> Ordering {
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
    pub(crate) fn min_by_pressure_key(
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

pub(crate) struct PressureKey<'a> {
    load: Option<&'a NativeCacheWorkerLoad>,
    local_active: usize,
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

/// One load snapshot per selection pass, captured on first use so policies
/// that never read load never pay for it.
pub struct LoadView<'a> {
    table: Option<&'a EngineLoadTable>,
    snapshot: OnceLock<EngineLoadSnapshot>,
}

impl<'a> LoadView<'a> {
    pub fn new(table: &'a EngineLoadTable) -> Self {
        Self {
            table: Some(table),
            snapshot: OnceLock::new(),
        }
    }

    /// A view over an already captured snapshot.
    pub fn from_snapshot(snapshot: EngineLoadSnapshot) -> Self {
        Self {
            table: None,
            snapshot: OnceLock::from(snapshot),
        }
    }

    pub fn snapshot(&self) -> &EngineLoadSnapshot {
        self.snapshot.get_or_init(|| {
            self.table
                .map(|table| table.capture_snapshot(Instant::now()))
                .unwrap_or_default()
        })
    }
}
