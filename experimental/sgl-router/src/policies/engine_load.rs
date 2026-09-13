// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine-reported runtime load, fed by the load subscriber.
//!
//! Workers publish a [`LoadStat`] gauge on their dedicated load socket (see
//! `python/sglang/srt/managers/scheduler_components/load_publisher.py`). The
//! load subscriber routes those into this table, keyed per
//! `(worker_url, dp_rank)`. Request handling captures the freshest complete
//! aggregate and falls back to Router-local load when it is unavailable.
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

/// Per-rank load fields consumed by native Cache-Aware.
///
/// Short frames cannot drive admission or pressure guards, so native
/// `cache_aware` falls back to Router-local load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCacheRankLoad {
    pub num_waiting_uncached_tokens: u64,
    pub num_total_tokens: u64,
    pub max_running_requests: u64,
    pub total_prefill_uncached_tokens: u64,
    pub total_prefill_busy_us: u64,
}

/// Per-scheduler runtime load snapshot. Mirrors the Python `LoadStat` in
/// `managers/scheduler_components/load_publisher.py`, published on the
/// worker's dedicated load socket (separate from KV-cache events).
///
/// The stable prefix remains `["LoadStat", running, waiting, used_tokens,
/// max_tokens, attn_dp_rank]`; V4 appends the V3 native Cache-Aware fields.
/// Older publishers therefore decode successfully with `native_cache=None`,
/// which deliberately excludes them from monitor-backed admission/guard.
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
    /// V3 native Cache-Aware semantics. `None` means the publisher is an old
    /// four-field #34608 producer or sent a truncated extension.
    pub native_cache: Option<NativeCacheRankLoad>,
}

/// Engine load for one worker captured at a fixed point in time.
///
/// The four #34608 fields are summed across DP ranks. `captured_at` retains
/// the oldest rank timestamp so later local dispatches can be added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineWorkerLoad {
    pub num_running_reqs: u64,
    pub num_waiting_reqs: u64,
    pub num_tokens: u64,
    pub max_total_num_tokens: u64,
    pub captured_at: Instant,
}

/// Complete ZMQ monitor aggregate used by native Cache-Aware.
///
/// Prefill throughput and queue time require two monotonic samples from every
/// DP rank. Initial samples and counter resets leave both values unavailable.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeCacheWorkerLoad {
    pub num_running_reqs: u64,
    pub num_waiting_reqs: u64,
    pub num_waiting_uncached_tokens: u64,
    pub num_used_tokens: u64,
    pub num_total_tokens: u64,
    pub max_total_num_tokens: u64,
    pub max_running_requests: u64,
    pub prefill_throughput_tokens_per_s: Option<f64>,
    pub estimated_prefill_queue_ms: Option<f64>,
    pub captured_at: Instant,
}

/// Immutable engine load view captured once at request ingress.
///
/// Keys are worker URLs used for dispatch. Missing, stale, or rank-incomplete
/// workers are omitted and must use Router-local active load.
#[derive(Debug, Clone, Default)]
pub struct EngineLoadSnapshot {
    pub version: u64,
    workers: HashMap<String, EngineWorkerLoad>,
    native_cache_workers: HashMap<String, NativeCacheWorkerLoad>,
}

impl EngineLoadSnapshot {
    pub fn fresh_load_for_url(&self, worker_url: &str) -> Option<&EngineWorkerLoad> {
        self.workers.get(worker_url)
    }

    /// Returns only complete, fresh native Cache-Aware monitor data.
    pub fn fresh_native_cache_load_for_url(
        &self,
        worker_url: &str,
    ) -> Option<&NativeCacheWorkerLoad> {
        self.native_cache_workers.get(worker_url)
    }

    /// Builds a view from worker data that already passed freshness and rank checks.
    /// Production requests should use [`EngineLoadTable::capture_snapshot`].
    pub fn from_workers(version: u64, workers: HashMap<String, EngineWorkerLoad>) -> Self {
        Self {
            version,
            workers,
            native_cache_workers: HashMap::new(),
        }
    }

    /// Builds a test snapshot from complete native monitor data.
    /// Production requests must use [`EngineLoadTable::capture_snapshot`].
    pub fn from_native_cache_workers(
        version: u64,
        workers: HashMap<String, NativeCacheWorkerLoad>,
    ) -> Self {
        let basic = workers
            .iter()
            .map(|(url, load)| {
                (
                    url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: load.num_running_reqs,
                        num_waiting_reqs: load.num_waiting_reqs,
                        num_tokens: load.num_used_tokens,
                        max_total_num_tokens: load.max_total_num_tokens,
                        captured_at: load.captured_at,
                    },
                )
            })
            .collect();
        Self {
            version,
            workers: basic,
            native_cache_workers: workers,
        }
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
                // `attn_dp_rank` is informational: the subscriber's socket
                // rank is authoritative for aggregation. Keep accepting null
                // and integer values from both old and new publishers.
                let _attn_dp_rank: Option<IgnoredAny> = seq.next_element()?;

                // The extension is deliberately all-or-nothing. A four-field
                // #34608 message remains valid for lightweight queue routing,
                // but a partial semantic tail is not valid monitor data.
                let native_cache = match seq.next_element::<u64>()? {
                    None => None,
                    Some(num_waiting_uncached_tokens) => {
                        let num_total_tokens = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::missing_field("num_total_tokens"))?;
                        let max_running_requests = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::missing_field("max_running_requests"))?;
                        let total_prefill_uncached_tokens =
                            seq.next_element()?.ok_or_else(|| {
                                de::Error::missing_field("total_prefill_uncached_tokens")
                            })?;
                        let total_prefill_busy_us = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::missing_field("total_prefill_busy_us"))?;
                        Some(NativeCacheRankLoad {
                            num_waiting_uncached_tokens,
                            num_total_tokens,
                            max_running_requests,
                            total_prefill_uncached_tokens,
                            total_prefill_busy_us,
                        })
                    }
                };
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(LoadStat {
                    num_running_reqs,
                    num_waiting_reqs,
                    num_tokens,
                    max_total_num_tokens,
                    native_cache,
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
    previous_native_cache: Option<NativeCacheRankLoad>,
    at: Instant,
}

type NativeRankObservation = (LoadStat, Option<NativeCacheRankLoad>, bool, Instant);
type NativeWorkerObservations = HashMap<u32, NativeRankObservation>;

/// Per-`(worker_url, dp_rank)` engine-reported load, written by the load
/// subscriber pump and captured once at request ingress.
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
        let key = (url.to_string(), dp_rank);
        let previous_native_cache = self
            .by_rank
            .get(&key)
            .and_then(|entry| entry.load.native_cache.clone());
        self.by_rank.insert(
            key,
            LoadEntry {
                load,
                previous_native_cache,
                at,
            },
        );
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark one advertised scheduler rank as expected to publish load.
    pub fn mark_expected_rank(&self, url: &str, dp_rank: u32) {
        if self.expected.insert((url.to_string(), dp_rank)) {
            self.version.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of workers expected to publish load.
    pub fn expected_count(&self) -> usize {
        self.expected
            .iter()
            .map(|entry| entry.key().0.clone())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Shared accumulation pass behind [`Self::capture_snapshot`]. It sums
    /// fields across ranks and keeps the oldest snapshot timestamp, but only for workers whose
    /// every advertised rank is present and fresh**. A missing or stale rank is
    /// omitted, so the caller falls back to its own load signal. (Summing
    /// only the fresh ranks would make a worker whose other ranks went silent
    /// look misleadingly idle and draw *more* traffic.) Callers that never
    /// registered expected ranks retain the all-known-ranks rule. The oldest
    /// timestamp represents the freshness of the complete aggregate.
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

    /// Aggregates complete native Cache-Aware monitor data.
    ///
    /// Every rank must be fresh, capacity-valid, and include the #34608
    /// extension. Otherwise the worker is omitted from monitor-backed guards.
    fn fresh_native_cache_worker_loads(
        &self,
        now: Instant,
    ) -> HashMap<String, NativeCacheWorkerLoad> {
        let mut observed: HashMap<String, NativeWorkerObservations> = HashMap::new();
        for entry in self.by_rank.iter() {
            let at = entry.value().at;
            let fresh = now.saturating_duration_since(at) <= self.freshness;
            observed.entry(entry.key().0.clone()).or_default().insert(
                entry.key().1,
                (
                    entry.value().load.clone(),
                    entry.value().previous_native_cache.clone(),
                    fresh,
                    at,
                ),
            );
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
                let mut num_waiting_uncached_tokens = 0u64;
                let mut num_used_tokens = 0u64;
                let mut num_total_tokens = 0u64;
                let mut max_total_num_tokens = 0u64;
                let mut max_running_requests = 0u64;
                let mut oldest_at = None;
                let mut prefill_throughput_tokens_per_s = 0.0f64;
                let mut complete_prefill_sample = !required.is_empty();

                for rank in required {
                    let (load, previous, fresh, at) = ranks.get(&rank)?;
                    let native = load.native_cache.as_ref()?;
                    if !fresh || load.max_total_num_tokens == 0 || native.max_running_requests == 0
                    {
                        return None;
                    }
                    num_running_reqs = num_running_reqs.saturating_add(load.num_running_reqs);
                    num_waiting_reqs = num_waiting_reqs.saturating_add(load.num_waiting_reqs);
                    num_waiting_uncached_tokens = num_waiting_uncached_tokens
                        .saturating_add(native.num_waiting_uncached_tokens);
                    num_used_tokens = num_used_tokens.saturating_add(load.num_tokens);
                    num_total_tokens = num_total_tokens.saturating_add(native.num_total_tokens);
                    max_total_num_tokens =
                        max_total_num_tokens.saturating_add(load.max_total_num_tokens);
                    max_running_requests =
                        max_running_requests.saturating_add(native.max_running_requests);
                    oldest_at = Some(oldest_at.map_or(*at, |oldest: Instant| oldest.min(*at)));

                    match previous {
                        Some(previous)
                            if native.total_prefill_uncached_tokens
                                > previous.total_prefill_uncached_tokens
                                && native.total_prefill_busy_us
                                    > previous.total_prefill_busy_us =>
                        {
                            let tokens = native.total_prefill_uncached_tokens
                                - previous.total_prefill_uncached_tokens;
                            let busy_us =
                                native.total_prefill_busy_us - previous.total_prefill_busy_us;
                            let rate = 1_000_000.0 * tokens as f64 / busy_us as f64;
                            if rate.is_finite() && rate > 0.0 {
                                prefill_throughput_tokens_per_s += rate;
                            } else {
                                complete_prefill_sample = false;
                            }
                        }
                        _ => complete_prefill_sample = false,
                    }
                }
                let prefill_throughput_tokens_per_s =
                    complete_prefill_sample.then_some(prefill_throughput_tokens_per_s);
                let estimated_prefill_queue_ms = prefill_throughput_tokens_per_s
                    .map(|rate| 1_000.0 * num_waiting_uncached_tokens as f64 / rate);
                oldest_at.map(|captured_at| {
                    (
                        url,
                        NativeCacheWorkerLoad {
                            num_running_reqs,
                            num_waiting_reqs,
                            num_waiting_uncached_tokens,
                            num_used_tokens,
                            num_total_tokens,
                            max_total_num_tokens,
                            max_running_requests,
                            prefill_throughput_tokens_per_s,
                            estimated_prefill_queue_ms,
                            captured_at,
                        },
                    )
                })
            })
            .collect()
    }

    /// Captures one immutable view for all routing decisions in a request.
    pub fn capture_snapshot(&self, now: Instant) -> EngineLoadSnapshot {
        EngineLoadSnapshot {
            version: self.version.load(Ordering::Acquire),
            workers: self.fresh_worker_loads(now),
            native_cache_workers: self.fresh_native_cache_worker_loads(now),
        }
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
            native_cache: None,
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
    fn load_wire_preserves_the_v3_native_cache_extension_and_accepts_old_short_frames() {
        let mut full = Vec::new();
        rmp::encode::write_array_len(&mut full, 11).unwrap();
        rmp::encode::write_str(&mut full, "LoadStat").unwrap();
        for value in [2, 3, 4, 100] {
            rmp::encode::write_u64(&mut full, value).unwrap();
        }
        rmp::encode::write_nil(&mut full).unwrap();
        for value in [500, 600, 32, 1_000, 2_000] {
            rmp::encode::write_u64(&mut full, value).unwrap();
        }
        let decoded = decode_load_stat(&full).expect("complete extended LoadStat decodes");
        assert_eq!(decoded.num_running_reqs, 2);
        assert_eq!(
            decoded
                .native_cache
                .expect("extension must be retained")
                .num_waiting_uncached_tokens,
            500
        );

        let mut old = Vec::new();
        rmp::encode::write_array_len(&mut old, 6).unwrap();
        rmp::encode::write_str(&mut old, "LoadStat").unwrap();
        for value in [2, 3, 4, 100] {
            rmp::encode::write_u64(&mut old, value).unwrap();
        }
        rmp::encode::write_nil(&mut old).unwrap();
        assert!(
            decode_load_stat(&old)
                .expect("old #34608 four-field frame remains decodable")
                .native_cache
                .is_none(),
            "short frames must never be promoted to complete native monitor data"
        );
    }

    #[test]
    fn sums_queue_depth_across_ranks() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.set("http://w:30000", 0, load(5, 1), now);
        t.set("http://w:30000", 1, load(3, 2), now);
        let fresh = t.capture_snapshot(now);
        // (5+1) + (3+2) = 11
        let load = fresh.fresh_load_for_url("http://w:30000").unwrap();
        assert_eq!(load.num_running_reqs + load.num_waiting_reqs, 11);
    }

    #[test]
    fn stale_entries_are_dropped_from_snapshot() {
        let t = EngineLoadTable::with_freshness(Duration::from_millis(10));
        let old = Instant::now();
        t.set("http://w:30000", 0, load(9, 9), old);
        // A read far in the future sees the entry as stale -> worker absent.
        let later = old + Duration::from_secs(60);
        assert!(t
            .capture_snapshot(later)
            .fresh_load_for_url("http://w:30000")
            .is_none());
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
        let snapshot = t.capture_snapshot(now);
        assert!(snapshot.fresh_load_for_url("http://w:30000").is_none());
        assert!(snapshot.fresh_load_for_url("http://other:30000").is_some());
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
            t.capture_snapshot(now)
                .fresh_load_for_url("http://w:30000")
                .is_none(),
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
            t.capture_snapshot(now)
                .fresh_load_for_url("http://w:30000")
                .is_none(),
            "an advertised rank without a reading must not produce a partial aggregate"
        );

        t.set("http://w:30000", 1, load(3, 2), now);
        let snapshot = t.capture_snapshot(now);
        let load = snapshot.fresh_load_for_url("http://w:30000").unwrap();
        assert_eq!(load.num_running_reqs + load.num_waiting_reqs, 11);
    }

    #[test]
    fn capture_snapshot_uses_the_earliest_rank_timestamp() {
        let t = EngineLoadTable::new();
        let earlier = Instant::now() - Duration::from_secs(2);
        let later = earlier + Duration::from_secs(1);
        t.set("http://w:30000", 0, load(5, 1), later);
        t.set("http://w:30000", 1, load(3, 2), earlier);
        let now = later + Duration::from_millis(1);
        let snapshot = t.capture_snapshot(now);
        let load = snapshot.fresh_load_for_url("http://w:30000").unwrap();
        assert_eq!(load.num_running_reqs + load.num_waiting_reqs, 11);
        assert_eq!(load.captured_at, earlier);
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

    #[test]
    fn complete_v3_semantic_samples_derive_prefill_queue_time() {
        let t = EngineLoadTable::new();
        let first = Instant::now();
        let second = first + Duration::from_secs(2);
        let mut old = load(2, 3);
        old.num_tokens = 16_000;
        old.max_total_num_tokens = 32_000;
        old.native_cache = Some(NativeCacheRankLoad {
            num_waiting_uncached_tokens: 1_000,
            num_total_tokens: 20_000,
            max_running_requests: 64,
            total_prefill_uncached_tokens: 10_000,
            total_prefill_busy_us: 2_000_000,
        });
        let mut new = old.clone();
        new.num_tokens = 20_000;
        let native = new
            .native_cache
            .as_mut()
            .expect("test sample has native-cache extension");
        native.num_waiting_uncached_tokens = 4_000;
        native.num_total_tokens = 24_000;
        native.total_prefill_uncached_tokens = 22_000;
        native.total_prefill_busy_us = 4_000_000;

        t.mark_expected_rank("http://w:30000", 0);
        t.set("http://w:30000", 0, old, first);
        t.set("http://w:30000", 0, new, second);

        let snapshot = t.capture_snapshot(second);
        let worker = snapshot
            .fresh_native_cache_load_for_url("http://w:30000")
            .expect("complete fresh rank must be usable");
        assert_eq!(worker.num_waiting_uncached_tokens, 4_000);
        assert_eq!(worker.num_total_tokens, 24_000);
        assert_eq!(worker.max_running_requests, 64);
        assert_eq!(worker.prefill_throughput_tokens_per_s, Some(6_000.0));
        assert_eq!(
            worker.estimated_prefill_queue_ms,
            Some(666.666_666_666_666_6)
        );
    }
}
