// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine-reported runtime load, fed by the load subscriber.
//!
//! Workers publish a [`LoadSnapshot`] gauge on their dedicated load socket
//! (see `python/sglang/srt/managers/load_snapshot.py`). The load subscriber
//! routes those into this table, keyed per `(worker_url, dp_rank)`; the
//! cache-aware-zmq policy sums each worker's ranks into a truthful load
//! signal, falling back to the router-side in-flight counter for any worker
//! whose picture is incomplete (cold start, a stale or silent rank, or a
//! worker that predates load publishing) — see [`EngineLoadTable::snapshot_fresh`].
//!
//! Load is a *gauge*, not a delta: last value wins, no sequence/replay
//! semantics. Entries older than [`EngineLoadTable::freshness`] are ignored.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::Deserialize;

/// Per-scheduler runtime load snapshot. Mirrors the Python `LoadSnapshot` in
/// `managers/load_snapshot.py` — the same struct that backs `/v1/loads` and
/// DP balancing, republished on the worker's dedicated load socket.
///
/// Wire shape is a msgpack **map** keyed by field name, declared
/// `omit_defaults=True` on the Python side, so any field may be absent and
/// every one defaults to zero here. The snapshot carries considerably more
/// (memory, speculative, LoRA, disaggregation and queue sub-structs) and serde
/// ignores what is not declared, which is what keeps this forward-compatible as
/// the engine grows. Of the four declared, routing reads only
/// `num_running_reqs` + `num_waiting_reqs`; the two token counts are decoded
/// but not yet consumed.
///
/// Because absent and zero are indistinguishable on the wire, a producer-side
/// field rename would report every worker as idle rather than failing — the
/// wire-shape test on the Python side is what guards that.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct LoadSnapshot {
    /// Requests currently running on the engine.
    pub num_running_reqs: u64,
    /// Requests queued waiting to run.
    pub num_waiting_reqs: u64,
    /// KV tokens currently in use.
    pub num_used_tokens: u64,
    /// KV-cache token capacity; 0 when unknown.
    pub max_total_num_tokens: u64,
}

/// Decode a single load frame's msgpack payload into a [`LoadSnapshot`].
pub fn decode_load_snapshot(payload: &[u8]) -> Result<LoadSnapshot, rmp_serde::decode::Error> {
    rmp_serde::from_slice(payload)
}

/// A per-rank load snapshot older than this is treated as stale, so a silent
/// or slow publisher degrades to the router-side load signal rather than
/// pinning a worker at its last reported value.
const DEFAULT_FRESHNESS: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
struct LoadEntry {
    load: LoadSnapshot,
    at: Instant,
}

/// Per-`(worker_url, dp_rank)` engine-reported load. Written by the load
/// subscriber pump, read by the cache-aware-zmq policy. Shared out of
/// [`super::kv_events::index::KvEventIndex`] the same way the hash tree is.
#[derive(Debug)]
pub struct EngineLoadTable {
    by_rank: DashMap<(String, u32), LoadEntry>,
    /// Worker URLs that advertised a load topic, mapped to how many ranks each
    /// is expected to publish. Serves two purposes: distinguishing "load-aware
    /// routing active" from "silently degraded to the in-flight counter"
    /// (see [`Self::expected_count`]), and giving [`Self::snapshot_fresh`] the
    /// denominator it needs to notice a rank that has never published at all.
    expected: DashMap<String, u32>,
    freshness: Duration,
}

impl EngineLoadTable {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashMap::new(),
            freshness: DEFAULT_FRESHNESS,
        })
    }

    #[cfg(test)]
    pub fn with_freshness(freshness: Duration) -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashMap::new(),
            freshness,
        })
    }

    /// Record the latest load for one `(worker_url, dp_rank)`.
    pub fn set(&self, url: &str, dp_rank: u32, load: LoadSnapshot, at: Instant) {
        self.by_rank
            .insert((url.to_string(), dp_rank), LoadEntry { load, at });
    }

    /// Mark a worker as expected to publish load on `dp_size` ranks (it
    /// advertised a load topic).
    pub fn mark_expected(&self, url: &str, dp_size: u32) {
        self.expected.insert(url.to_string(), dp_size);
    }

    /// Number of workers expected to publish load. Compared against the size
    /// of [`Self::snapshot_fresh`] to surface a dead/misconfigured publisher
    /// (expected > 0 but no fresh snapshots) in logs.
    pub fn expected_count(&self) -> usize {
        self.expected.len()
    }

    /// One pass over the table returning, per worker URL, the summed queue
    /// depth (`num_running_reqs + num_waiting_reqs`) across that worker's
    /// ranks — **but only for workers reporting a full, fresh complement of
    /// ranks**. Any worker with a stale rank, or with fewer entries than the
    /// rank count it advertised, is omitted so the caller falls back to its own
    /// load signal. Computed once per selection so per-worker lookups are O(1).
    ///
    /// WHY the completeness half matters as much as the freshness half: a
    /// partial sum under-reports depth, which makes a worker look idle and
    /// draws *more* traffic to it. Freshness alone cannot catch a rank that has
    /// never published — an entry that does not exist cannot be judged stale —
    /// so a rank whose publisher failed to bind at startup would otherwise
    /// bias selection toward the broken worker indefinitely.
    pub fn snapshot_fresh(&self, now: Instant) -> HashMap<String, usize> {
        // url -> (summed depth, all-ranks-fresh, ranks seen).
        let mut acc: HashMap<String, (usize, bool, u32)> = HashMap::new();
        for entry in self.by_rank.iter() {
            let fresh = now.duration_since(entry.value().at) <= self.freshness;
            let l = &entry.value().load;
            let depth = (l.num_running_reqs.saturating_add(l.num_waiting_reqs)) as usize;
            let slot = acc.entry(entry.key().0.clone()).or_insert((0, true, 0));
            slot.0 = slot.0.saturating_add(depth);
            slot.1 = slot.1 && fresh;
            slot.2 += 1;
        }
        acc.into_iter()
            .filter_map(|(url, (depth, all_fresh, seen))| {
                // A worker absent from `expected` published without advertising
                // a load topic; trust what it sent rather than inventing a
                // denominator for it.
                let complete = self
                    .expected
                    .get(&url)
                    .is_none_or(|expected| seen >= *expected);
                (all_fresh && complete).then_some((url, depth))
            })
            .collect()
    }

    /// Drop every rank entry (and the expected mark) for a worker. Called on
    /// worker removal so a re-added worker does not leave stale load behind.
    pub fn forget_worker(&self, url: &str) {
        self.by_rank.retain(|k, _| k.0 != url);
        self.expected.remove(url);
    }

    #[cfg(test)]
    pub fn entry_count(&self) -> usize {
        self.by_rank.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(running: u64, waiting: u64) -> LoadSnapshot {
        LoadSnapshot {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_used_tokens: 0,
            max_total_num_tokens: 0,
        }
    }

    #[test]
    fn sums_queue_depth_across_ranks() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.set("http://w:30000", 0, load(5, 1), now);
        t.set("http://w:30000", 1, load(3, 2), now);
        let fresh = t.snapshot_fresh(now);
        // (5+1) + (3+2) = 11
        assert_eq!(fresh.get("http://w:30000").copied(), Some(11));
    }

    #[test]
    fn stale_entries_are_dropped_from_snapshot() {
        let t = EngineLoadTable::with_freshness(Duration::from_millis(10));
        let old = Instant::now();
        t.set("http://w:30000", 0, load(9, 9), old);
        // A read far in the future sees the entry as stale -> worker absent.
        let later = old + Duration::from_secs(60);
        assert!(!t.snapshot_fresh(later).contains_key("http://w:30000"));
    }

    #[test]
    fn forget_worker_clears_all_ranks() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.set("http://w:30000", 0, load(1, 0), now);
        t.set("http://w:30000", 1, load(1, 0), now);
        t.set("http://other:30000", 0, load(1, 0), now);
        t.forget_worker("http://w:30000");
        assert_eq!(t.entry_count(), 1);
        assert!(!t.snapshot_fresh(now).contains_key("http://w:30000"));
        assert!(t.snapshot_fresh(now).contains_key("http://other:30000"));
    }

    /// A worker with any stale rank is omitted entirely (not summed over only
    /// its fresh ranks), so a partially-silent worker falls back to the
    /// router-side counter instead of looking misleadingly idle.
    #[test]
    fn partial_freshness_excludes_worker() {
        let t = EngineLoadTable::with_freshness(Duration::from_secs(5));
        let now = Instant::now();
        let stale = now - Duration::from_secs(3600);
        t.set("http://w:30000", 0, load(5, 1), now); // fresh
        t.set("http://w:30000", 1, load(9, 9), stale); // stale
        assert!(
            !t.snapshot_fresh(now).contains_key("http://w:30000"),
            "any stale rank must drop the whole worker from the snapshot"
        );
    }

    #[test]
    fn expected_count_tracks_marked_workers_and_forget() {
        let t = EngineLoadTable::new();
        assert_eq!(t.expected_count(), 0);
        t.mark_expected("http://w:30000", 1);
        t.mark_expected("http://w:30000", 1); // idempotent
        t.mark_expected("http://other:30000", 1);
        assert_eq!(t.expected_count(), 2);
        t.forget_worker("http://w:30000");
        assert_eq!(t.expected_count(), 1);
    }

    /// A rank that never published cannot be judged stale, so freshness alone
    /// would report a partial sum — making the worker look idle and drawing
    /// *more* traffic to it. This is the startup-bind-failure case.
    #[test]
    fn worker_missing_a_rank_entirely_is_excluded() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.mark_expected("http://w:30000", 4);
        for rank in 0..3 {
            t.set("http://w:30000", rank, load(5, 5), now);
        }

        assert!(
            !t.snapshot_fresh(now).contains_key("http://w:30000"),
            "3 of 4 ranks reporting must not be treated as a complete picture"
        );

        t.set("http://w:30000", 3, load(5, 5), now);
        assert_eq!(
            t.snapshot_fresh(now).get("http://w:30000").copied(),
            Some(40),
            "the full complement sums normally"
        );
    }

    /// A worker that never advertised a load topic has no expected rank count,
    /// so whatever it publishes is taken at face value rather than measured
    /// against an invented denominator.
    #[test]
    fn unmarked_worker_is_trusted_without_a_denominator() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.set("http://w:30000", 0, load(2, 3), now);
        assert_eq!(
            t.snapshot_fresh(now).get("http://w:30000").copied(),
            Some(5)
        );
    }
}
