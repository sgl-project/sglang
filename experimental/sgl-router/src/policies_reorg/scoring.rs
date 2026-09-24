// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Ordered engine-pressure keys. Lower is better; these are not weighted sums.

use std::cmp::Ordering;
use std::sync::Arc;

use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;

/// Select once for the candidates being ranked, never separately for each pair.
#[derive(Clone, Copy)]
pub(super) enum LoadSource {
    RouterInflight,
    EngineReported,
    PrefillQueueTime,
}

pub(super) fn load_source<'a>(
    snapshot: &EngineReportedLoadSnapshot,
    engines: impl IntoIterator<Item = &'a Arc<Worker>>,
) -> LoadSource {
    let mut source = LoadSource::PrefillQueueTime;
    for engine in engines {
        let Some(load) = snapshot.fresh_native_cache_load_for_url(&engine.url) else {
            return LoadSource::RouterInflight;
        };
        if load.estimated_prefill_queue_ms.is_none() {
            source = LoadSource::EngineReported;
        }
    }
    source
}

/// A key captures router-local load once as well as the reported pressure.
/// Compare keys made for the same stage and load source.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct EngineScore {
    queue_ms: f64,
    // Prefill: (pending tokens, waiting, running). Decode: (waiting, running, 0).
    counts: (u64, u64, u64),
    // Decode only. Compare utilization exactly by cross multiplication.
    used_tokens: u64,
    capacity: u64,
    inflight: usize,
}

impl Ord for EngineScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.queue_ms
            .total_cmp(&other.queue_ms)
            .then_with(|| self.counts.cmp(&other.counts))
            .then_with(|| {
                (u128::from(self.used_tokens) * u128::from(other.capacity))
                    .cmp(&(u128::from(other.used_tokens) * u128::from(self.capacity)))
            })
            .then_with(|| self.used_tokens.cmp(&other.used_tokens))
            .then_with(|| self.inflight.cmp(&other.inflight))
    }
}

impl PartialOrd for EngineScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for EngineScore {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for EngineScore {}

pub(super) fn prefill_score(
    engine: &Worker,
    snapshot: &EngineReportedLoadSnapshot,
    source: LoadSource,
) -> EngineScore {
    let mut score = EngineScore {
        inflight: engine.router_inflight_load(),
        ..EngineScore::default()
    };
    if !matches!(source, LoadSource::RouterInflight) {
        let load = snapshot
            .fresh_native_cache_load_for_url(&engine.url)
            .expect("load source requires complete candidate reports");
        score.counts = (
            load.num_waiting_uncached_tokens,
            load.num_waiting_reqs,
            load.num_running_reqs,
        );
        if matches!(source, LoadSource::PrefillQueueTime) {
            score.queue_ms = load
                .estimated_prefill_queue_ms
                .expect("load source requires candidate queue-time estimates");
        }
    }
    score
}

pub(super) fn decode_score(
    engine: &Worker,
    snapshot: &EngineReportedLoadSnapshot,
    source: LoadSource,
) -> EngineScore {
    let mut score = EngineScore {
        inflight: engine.router_inflight_load(),
        ..EngineScore::default()
    };
    if !matches!(source, LoadSource::RouterInflight) {
        let load = snapshot
            .fresh_native_cache_load_for_url(&engine.url)
            .expect("load source requires complete candidate reports");
        score.counts = (load.num_waiting_reqs, load.num_running_reqs, 0);
        score.used_tokens = load.num_used_tokens;
        score.capacity = load.max_total_num_tokens;
    }
    score
}
