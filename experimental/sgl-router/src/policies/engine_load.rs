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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// Aggregated, usable Engine load for one Worker at a fixed instant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineWorkerLoad {
    pub num_running_reqs: u64,
    pub num_waiting_reqs: u64,
    pub num_tokens: u64,
    pub max_total_num_tokens: u64,
    pub captured_at: Instant,
}

/// Immutable Engine-load view captured once at request ingress.
#[derive(Debug, Clone, Default)]
pub struct EngineLoadSnapshot {
    pub version: u64,
    workers: HashMap<String, EngineWorkerLoad>,
}

impl EngineLoadSnapshot {
    pub fn fresh_load_for_url(&self, worker_url: &str) -> Option<&EngineWorkerLoad> {
        self.workers.get(worker_url)
    }

    /// Builds a view from already validated Worker data for tests and offline checks.
    pub fn from_workers(version: u64, workers: HashMap<String, EngineWorkerLoad>) -> Self {
        Self { version, workers }
    }
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
                // The Python publisher always emits all four counts. Treat a
                // shortened frame as malformed rather than inventing zeros:
                // a partial gauge must fall back to router-local load, never
                // make a worker appear artificially idle.
                let num_running_reqs: u64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("num_running_reqs"))?;
                let num_waiting_reqs: u64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("num_waiting_reqs"))?;
                let num_tokens: u64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("num_tokens"))?;
                let max_total_num_tokens: u64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("max_total_num_tokens"))?;
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
    /// Per-rank publishers the worker advertised. A worker is usable only
    /// when every advertised rank has a fresh value; accepting a partial
    /// aggregate would make a silent rank look idle and attract traffic.
    expected: DashSet<(String, u32)>,
    freshness: Duration,
    version: AtomicU64,
}

impl EngineLoadTable {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashSet::new(),
            freshness: DEFAULT_FRESHNESS,
            version: AtomicU64::new(0),
        })
    }

    #[cfg(test)]
    pub fn with_freshness(freshness: Duration) -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashSet::new(),
            freshness,
            version: AtomicU64::new(0),
        })
    }

    /// Record the latest load for one `(worker_url, dp_rank)`.
    pub fn set(&self, url: &str, dp_rank: u32, load: LoadStat, at: Instant) {
        self.by_rank
            .insert((url.to_string(), dp_rank), LoadEntry { load, at });
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark one advertised scheduler rank as expected to publish load.
    pub fn mark_expected_rank(&self, url: &str, dp_rank: u32) {
        if self.expected.insert((url.to_string(), dp_rank)) {
            self.version.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of workers expected to publish load. Compared against the size
    /// of [`Self::snapshot_fresh`] to surface a dead/misconfigured publisher
    /// (expected > 0 but no fresh snapshots) in logs.
    pub fn expected_count(&self) -> usize {
        self.expected
            .iter()
            .map(|entry| entry.key().0.clone())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Shared accumulation pass behind [`Self::snapshot_fresh`] and
    /// [`Self::capture_snapshot`]. It produces the #34608 fields summed across
    /// ranks and the OLDEST snapshot timestamp — **but only for workers whose
    /// every advertised rank is present and fresh**. A missing or stale rank is
    /// omitted, so the caller falls back to its own load signal. (Summing
    /// only the fresh ranks would make a worker whose other ranks went silent
    /// look misleadingly idle and draw *more* traffic.) Callers that never
    /// registered expected ranks retain the legacy all-known-ranks rule.
    /// `snapshot_fresh` and any other consumer walking this same pass can
    /// never disagree with each other about which workers count as fresh.
    ///
    /// The oldest (not newest) rank's timestamp is deliberately what's kept
    /// alongside the depth: a caller using it as a "dispatches not yet
    /// reflected in this number" cutoff (see
    /// `crate::policies::cache_aware_zmq::WorkerLoads::load_of`) needs a
    /// bound that never treats an unreported dispatch as already-covered —
    /// the freshest rank's timestamp could do exactly that for whichever
    /// rank published less recently. This conservatism is one-sided, not
    /// free: for a multi-rank worker with skewed publish times, a dispatch
    /// that landed on (and was already reported by) the FRESHER rank can
    /// get re-added by the caller's cutoff-based correction anyway, since
    /// that correction has no way to attribute a dispatch to a specific
    /// rank. That's an accepted, bounded over-count (it biases the wrong
    /// direction relative to the under-count this method exists to avoid,
    /// not a correctness hole) rather than something this method can close
    /// on its own — closing it would require per-rank dispatch attribution,
    /// which the router-side slot tracking below doesn't have.
    fn fresh_worker_loads(&self, now: Instant) -> HashMap<String, EngineWorkerLoad> {
        // url -> rank -> (reported load, fresh, timestamp).
        let mut observed: HashMap<String, HashMap<u32, (LoadStat, bool, Instant)>> = HashMap::new();
        for entry in self.by_rank.iter() {
            let at = entry.value().at;
            let fresh = now.saturating_duration_since(at) <= self.freshness;
            observed
                .entry(entry.key().0.clone())
                .or_default()
                .insert(entry.key().1, (entry.value().load.clone(), fresh, at));
        }
        let mut expected: HashMap<String, HashSet<u32>> = HashMap::new();
        for entry in self.expected.iter() {
            expected
                .entry(entry.key().0.clone())
                .or_default()
                .insert(entry.key().1);
        }

        let workers: HashSet<String> = observed.keys().chain(expected.keys()).cloned().collect();
        workers
            .into_iter()
            .filter_map(|url| {
                let ranks = observed.get(&url)?;
                let required: Vec<u32> = match expected.get(&url) {
                    Some(expected_ranks) => expected_ranks.iter().copied().collect(),
                    None => ranks.keys().copied().collect(),
                };
                let mut num_running_reqs = 0u64;
                let mut num_waiting_reqs = 0u64;
                let mut num_tokens = 0u64;
                let mut max_total_num_tokens = 0u64;
                let mut oldest_at = None;
                for rank in required {
                    let (load, fresh, at) = ranks.get(&rank)?;
                    if !fresh {
                        return None;
                    }
                    num_running_reqs = num_running_reqs.saturating_add(load.num_running_reqs);
                    num_waiting_reqs = num_waiting_reqs.saturating_add(load.num_waiting_reqs);
                    num_tokens = num_tokens.saturating_add(load.num_tokens);
                    max_total_num_tokens =
                        max_total_num_tokens.saturating_add(load.max_total_num_tokens);
                    oldest_at = Some(oldest_at.map_or(*at, |oldest: Instant| oldest.min(*at)));
                }
                oldest_at.map(|captured_at| {
                    (
                        url,
                        EngineWorkerLoad {
                            num_running_reqs,
                            num_waiting_reqs,
                            num_tokens,
                            max_total_num_tokens,
                            captured_at,
                        },
                    )
                })
            })
            .collect()
    }

    /// Captures the immutable view consumed by all routing decisions in one request.
    pub fn capture_snapshot(&self, now: Instant) -> EngineLoadSnapshot {
        EngineLoadSnapshot {
            version: self.version.load(Ordering::Acquire),
            workers: self.fresh_worker_loads(now),
        }
    }

    pub(crate) fn fresh_worker_state(&self, now: Instant) -> HashMap<String, (usize, Instant)> {
        self.fresh_worker_loads(now)
            .into_iter()
            .map(|(url, load)| {
                (
                    url,
                    (
                        load.num_running_reqs
                            .saturating_add(load.num_waiting_reqs)
                            .try_into()
                            .unwrap_or(usize::MAX),
                        load.captured_at,
                    ),
                )
            })
            .collect()
    }

    /// Per worker URL, the summed queue depth (`num_running_reqs +
    /// num_waiting_reqs`) across that worker's ranks, for workers whose
    /// every advertised rank is fresh. Computed once per selection so per-worker
    /// lookups are O(1). See [`Self::fresh_worker_state`] for the freshness
    /// gate behind this.
    pub fn snapshot_fresh(&self, now: Instant) -> HashMap<String, usize> {
        self.fresh_worker_state(now)
            .into_iter()
            .map(|(url, (depth, _))| (url, depth))
            .collect()
    }

    /// Drop every rank entry (and the expected mark) for a worker. Called on
    /// worker removal so a re-added worker does not leave stale load behind.
    pub fn forget_worker(&self, url: &str) {
        self.by_rank.retain(|k, _| k.0 != url);
        self.expected.retain(|key| key.0 != url);
        self.version.fetch_add(1, Ordering::Relaxed);
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
    fn load_wire_rejects_wrong_tag_and_missing_counts() {
        let mut wrong_tag = Vec::new();
        rmp::encode::write_array_len(&mut wrong_tag, 5).unwrap();
        rmp::encode::write_str(&mut wrong_tag, "OtherStat").unwrap();
        for value in [1, 2, 3, 4] {
            rmp::encode::write_u64(&mut wrong_tag, value).unwrap();
        }
        assert!(decode_load_stat(&wrong_tag).is_err());

        let mut missing_count = Vec::new();
        rmp::encode::write_array_len(&mut missing_count, 4).unwrap();
        rmp::encode::write_str(&mut missing_count, "LoadStat").unwrap();
        for value in [1, 2, 3] {
            rmp::encode::write_u64(&mut missing_count, value).unwrap();
        }
        assert!(decode_load_stat(&missing_count).is_err());
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
    fn missing_expected_rank_excludes_worker() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.mark_expected_rank("http://w:30000", 0);
        t.mark_expected_rank("http://w:30000", 1);
        t.set("http://w:30000", 0, load(5, 1), now);
        assert!(
            !t.snapshot_fresh(now).contains_key("http://w:30000"),
            "an advertised rank without a reading must not produce a partial aggregate"
        );

        t.set("http://w:30000", 1, load(3, 2), now);
        assert_eq!(t.snapshot_fresh(now).get("http://w:30000"), Some(&11));
    }

    #[test]
    fn fresh_worker_state_picks_the_earliest_rank_timestamp() {
        let t = EngineLoadTable::new();
        let earlier = Instant::now() - Duration::from_secs(2);
        let later = earlier + Duration::from_secs(1);
        t.set("http://w:30000", 0, load(5, 1), later);
        t.set("http://w:30000", 1, load(3, 2), earlier);
        let now = later + Duration::from_millis(1);
        assert_eq!(
            t.fresh_worker_state(now).get("http://w:30000").copied(),
            Some((11, earlier)),
            "must expose the OLDEST rank's timestamp, not the newest"
        );
    }

    #[test]
    fn fresh_worker_state_agrees_with_snapshot_fresh_on_which_workers_are_present() {
        let t = EngineLoadTable::with_freshness(Duration::from_secs(5));
        let now = Instant::now();
        let stale = now - Duration::from_secs(3600);
        t.set("http://fresh:30000", 0, load(1, 0), now);
        t.set("http://mixed:30000", 0, load(1, 0), now);
        t.set("http://mixed:30000", 1, load(1, 0), stale);

        let depths = t.snapshot_fresh(now);
        let state = t.fresh_worker_state(now);
        assert!(depths.contains_key("http://fresh:30000"));
        assert!(state.contains_key("http://fresh:30000"));
        assert!(!depths.contains_key("http://mixed:30000"));
        assert!(!state.contains_key("http://mixed:30000"));
    }

    #[test]
    fn expected_count_tracks_marked_workers_and_forget() {
        let t = EngineLoadTable::new();
        assert_eq!(t.expected_count(), 0);
        t.mark_expected_rank("http://w:30000", 0);
        t.mark_expected_rank("http://w:30000", 1); // same worker
        t.mark_expected_rank("http://other:30000", 0);
        assert_eq!(t.expected_count(), 2);
        t.forget_worker("http://w:30000");
        assert_eq!(t.expected_count(), 1);
    }
}
