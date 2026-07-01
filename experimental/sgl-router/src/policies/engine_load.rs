// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine-reported runtime load, fed by the load subscriber.
//!
//! Workers publish a [`LoadStat`] gauge on their dedicated load socket (see
//! `python/sglang/srt/managers/scheduler_components/load_publisher.py`). The
//! load subscriber routes those into this table, keyed per
//! `(worker_url, dp_rank)`; the
//! cache-aware-zmq policy reads the freshest aggregate per worker as a
//! truthful load signal, falling back to the router-side in-flight counter
//! when no fresh snapshot exists (cold start, stale publisher, or a worker
//! that predates load publishing).
//!
//! Load is a *gauge*, not a delta: last value wins, no sequence/replay
//! semantics. Entries older than [`EngineLoadTable::freshness`] are ignored.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::{DashMap, DashSet};
use serde::de::{self, Deserializer, IgnoredAny, SeqAccess, Visitor};
use serde::Deserialize;

/// Per-scheduler runtime load snapshot. Mirrors the Python `LoadStat` in
/// `managers/scheduler_components/load_publisher.py`, published on the
/// worker's dedicated load socket (separate from KV-cache events).
///
/// Wire shape (msgspec `tag=True` + `array_like`):
/// `["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens,
/// max_total_num_tokens, attn_dp_rank?]`. We read the four counts and ignore
/// any trailing fields (`attn_dp_rank` — the router keys load by the
/// subscriber's socket rank, not the payload).
#[derive(Debug, Clone, PartialEq)]
pub struct LoadStat {
    /// Requests currently running on the engine.
    pub num_running_reqs: u64,
    /// Requests queued waiting to run.
    pub num_waiting_reqs: u64,
    /// KV tokens currently in use.
    pub num_tokens: u64,
    /// KV-cache token capacity; 0 when unknown.
    pub max_total_num_tokens: u64,
}

impl<'de> Deserialize<'de> for LoadStat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LoadStatVisitor;

        impl<'de> Visitor<'de> for LoadStatVisitor {
            type Value = LoadStat;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a tagged msgpack array [\"LoadStat\", ...fields]")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<LoadStat, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let tag: String = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("event tag"))?;
                if tag != "LoadStat" {
                    return Err(de::Error::custom(format!(
                        "expected \"LoadStat\" tag, got {tag:?}"
                    )));
                }
                // Counts are always emitted (no Python defaults), but default
                // missing fields to 0 and drain trailing fields (attn_dp_rank,
                // future additions) for forward-compatibility.
                let num_running_reqs: u64 = seq.next_element()?.unwrap_or(0);
                let num_waiting_reqs: u64 = seq.next_element()?.unwrap_or(0);
                let num_tokens: u64 = seq.next_element()?.unwrap_or(0);
                let max_total_num_tokens: u64 = seq.next_element()?.unwrap_or(0);
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(LoadStat {
                    num_running_reqs,
                    num_waiting_reqs,
                    num_tokens,
                    max_total_num_tokens,
                })
            }
        }

        deserializer.deserialize_seq(LoadStatVisitor)
    }
}

/// Decode a single load frame's msgpack payload into a [`LoadStat`].
pub fn decode_load_stat(payload: &[u8]) -> Result<LoadStat, rmp_serde::decode::Error> {
    rmp_serde::from_slice(payload)
}

/// A per-rank load snapshot older than this is treated as stale, so a silent
/// or slow publisher degrades to the router-side load signal rather than
/// pinning a worker at its last reported value.
const DEFAULT_FRESHNESS: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
struct LoadEntry {
    load: LoadStat,
    at: Instant,
}

/// Per-`(worker_url, dp_rank)` engine-reported load. Written by the load
/// subscriber pump, read by the cache-aware-zmq policy. Shared out of
/// [`super::kv_events::index::KvEventIndex`] the same way the hash tree is.
#[derive(Debug)]
pub struct EngineLoadTable {
    by_rank: DashMap<(String, u32), LoadEntry>,
    /// Worker URLs that advertised a load topic and so are *expected* to
    /// publish load. Lets the router distinguish "load-aware routing active"
    /// from "silently degraded to the in-flight counter" (expected but no
    /// fresh snapshot) — see [`Self::expected_count`].
    expected: DashSet<String>,
    freshness: Duration,
}

impl EngineLoadTable {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashSet::new(),
            freshness: DEFAULT_FRESHNESS,
        })
    }

    #[cfg(test)]
    pub fn with_freshness(freshness: Duration) -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashSet::new(),
            freshness,
        })
    }

    /// Record the latest load for one `(worker_url, dp_rank)`.
    pub fn set(&self, url: &str, dp_rank: u32, load: LoadStat, at: Instant) {
        self.by_rank
            .insert((url.to_string(), dp_rank), LoadEntry { load, at });
    }

    /// Mark a worker as expected to publish load (it advertised a load topic).
    pub fn mark_expected(&self, url: &str) {
        self.expected.insert(url.to_string());
    }

    /// Number of workers expected to publish load. Compared against the size
    /// of [`Self::snapshot_fresh`] to surface a dead/misconfigured publisher
    /// (expected > 0 but no fresh snapshots) in logs.
    pub fn expected_count(&self) -> usize {
        self.expected.len()
    }

    /// One pass over the table returning, per worker URL, the summed queue
    /// depth (`num_running_reqs + num_waiting_reqs`) across that worker's
    /// ranks — **but only for workers whose every known rank is fresh**. A
    /// worker with any stale rank is omitted, so the caller falls back to its
    /// own load signal. (Summing only the fresh ranks would make a worker
    /// whose other ranks went silent look misleadingly idle and draw *more*
    /// traffic.) Computed once per selection so per-worker lookups are O(1).
    pub fn snapshot_fresh(&self, now: Instant) -> HashMap<String, usize> {
        // url -> (summed depth across all ranks, all-ranks-fresh).
        let mut acc: HashMap<String, (usize, bool)> = HashMap::new();
        for entry in self.by_rank.iter() {
            let fresh = now.duration_since(entry.value().at) <= self.freshness;
            let l = &entry.value().load;
            let depth = (l.num_running_reqs.saturating_add(l.num_waiting_reqs)) as usize;
            let slot = acc.entry(entry.key().0.clone()).or_insert((0, true));
            slot.0 = slot.0.saturating_add(depth);
            slot.1 = slot.1 && fresh;
        }
        acc.into_iter()
            .filter_map(|(url, (depth, all_fresh))| all_fresh.then_some((url, depth)))
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

    fn load(running: u64, waiting: u64) -> LoadStat {
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_tokens: 0,
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
        t.mark_expected("http://w:30000");
        t.mark_expected("http://w:30000"); // idempotent
        t.mark_expected("http://other:30000");
        assert_eq!(t.expected_count(), 2);
        t.forget_worker("http://w:30000");
        assert_eq!(t.expected_count(), 1);
    }
}
