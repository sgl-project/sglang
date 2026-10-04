// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine pressure comparisons over one captured load snapshot.

use super::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedSchedulingLoad, EngineReportedWorkerLoad,
};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// Compares prefill pressure by queue time when available, then by the V3 load tuple.
pub(crate) fn compare_prefill_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&EngineReportedLoadSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_native_cache_load_for_url(&left.url)?,
            snapshot.fresh_native_cache_load_for_url(&right.url)?,
        ))
    }) {
        Some((left_load, right_load)) => {
            compare_prefill_load(left_load, right_load).then_with(|| {
                left.router_inflight_load()
                    .cmp(&right.router_inflight_load())
            })
        }
        None => left
            .router_inflight_load()
            .cmp(&right.router_inflight_load()),
    }
}

fn prefill_pressure_key(load: &EngineReportedSchedulingLoad) -> (u64, u64, u64) {
    (
        load.num_waiting_uncached_tokens,
        load.num_waiting_reqs,
        load.num_running_reqs,
    )
}

fn compare_prefill_load(
    left: &EngineReportedSchedulingLoad,
    right: &EngineReportedSchedulingLoad,
) -> Ordering {
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
    snapshot: Option<&EngineReportedLoadSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_native_cache_load_for_url(&left.url)?,
            snapshot.fresh_native_cache_load_for_url(&right.url)?,
        ))
    }) {
        Some((left_load, right_load)) => {
            compare_decode_load(left_load, right_load).then_with(|| {
                left.router_inflight_load()
                    .cmp(&right.router_inflight_load())
            })
        }
        None => left
            .router_inflight_load()
            .cmp(&right.router_inflight_load()),
    }
}

fn compare_decode_load(
    left: &EngineReportedSchedulingLoad,
    right: &EngineReportedSchedulingLoad,
) -> Ordering {
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

/// Constant-time request view over one captured load snapshot.
///
/// External values are compared only when every candidate is present. Mixed
/// candidate sets use Router-local active load to preserve ordering.
///
/// The key-level helpers (`comparable_get`, `compare_*_keys`,
/// `min_by_pressure_key`) are `pub(crate)` only for legacy admission.
pub(crate) struct CandidateLoads<'a> {
    by_worker_id: HashMap<String, &'a EngineReportedSchedulingLoad>,
    basic_by_worker_id: HashMap<String, &'a EngineReportedWorkerLoad>,
    local_active_by_worker_id: HashMap<String, usize>,
    compare_engine: bool,
    compare_basic_engine: bool,
}

impl<'a> CandidateLoads<'a> {
    pub(crate) fn new<'w>(
        snapshot: Option<&'a EngineReportedLoadSnapshot>,
        workers: impl IntoIterator<Item = &'w Arc<Worker>>,
    ) -> Self {
        let workers: Vec<&Arc<Worker>> = workers.into_iter().collect();
        let local_active_by_worker_id: HashMap<String, usize> = workers
            .iter()
            .map(|worker| (worker.id.0.clone(), worker.router_inflight_load()))
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
    ) -> Option<&'a EngineReportedSchedulingLoad> {
        self.by_worker_id.get(worker_id.0.as_str()).copied()
    }

    pub(crate) fn comparable_get(
        &self,
        worker_id: &crate::discovery::WorkerId,
    ) -> Option<&'a EngineReportedSchedulingLoad> {
        self.compare_engine.then(|| self.get(worker_id)).flatten()
    }

    fn pressure_key(&self, worker: &Arc<Worker>) -> LoadKey<'a> {
        LoadKey {
            load: self.comparable_get(&worker.id),
            local_active: self
                .local_active_by_worker_id
                .get(worker.id.0.as_str())
                .copied()
                .unwrap_or(usize::MAX),
        }
    }

    pub(crate) fn compare_prefill_keys(&self, left: &LoadKey<'a>, right: &LoadKey<'a>) -> Ordering {
        match (left.load, right.load) {
            (Some(left_load), Some(right_load)) => compare_prefill_load(left_load, right_load)
                .then_with(|| left.local_active.cmp(&right.local_active)),
            _ => left.local_active.cmp(&right.local_active),
        }
    }

    pub(crate) fn compare_decode_keys(&self, left: &LoadKey<'a>, right: &LoadKey<'a>) -> Ordering {
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
        compare: impl Fn(&Self, &LoadKey<'a>, &LoadKey<'a>) -> Ordering,
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

pub(crate) struct LoadKey<'a> {
    load: Option<&'a EngineReportedSchedulingLoad>,
    local_active: usize,
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
            ..Default::default()
        }))
    }

    /// `(worker, running, waiting)`; waiting doubles as uncached tokens.
    fn snapshot(entries: &[(&Arc<Worker>, u64, u64)]) -> EngineReportedLoadSnapshot {
        EngineReportedLoadSnapshot::from_native_cache_workers(
            7,
            entries
                .iter()
                .map(|(worker, running, waiting)| {
                    (
                        worker.url.clone(),
                        EngineReportedSchedulingLoad {
                            num_running_reqs: *running,
                            num_waiting_reqs: *waiting,
                            num_waiting_uncached_tokens: *waiting,
                            num_used_tokens: 0,
                            num_total_tokens: 0,
                            max_total_num_tokens: 100,
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
    fn prefill_pressure_uses_waiting_then_running_requests() {
        let busy = worker("busy");
        let idle = worker("idle");
        let loads = snapshot(&[(&busy, 1, 8), (&idle, 9, 2)]);
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
    fn decode_pressure_tie_is_not_broken_by_worker_id() {
        let (a, z) = (worker("a"), worker("z"));
        assert_eq!(compare_decode_pressure(&a, &z, None), Ordering::Equal);
    }

    #[test]
    fn one_stale_candidate_makes_the_set_compare_by_local_load() {
        let idle = worker("idle");
        let busy = worker("busy");
        let stale = worker("stale");
        let _idle_guards: Vec<_> = (0..5).map(|_| idle.load_guard()).collect();
        let _busy_guard = busy.load_guard();
        let snapshot = snapshot(&[(&idle, 0, 0), (&busy, 0, 1_000)]);

        let loads = CandidateLoads::new(Some(&snapshot), [&idle, &busy, &stale]);
        assert!(loads.get(&idle.id).is_some());
        assert!(loads.get(&stale.id).is_none());
        assert!(loads.compare_prefill_pressure(&idle, &busy).is_gt());
    }
}
