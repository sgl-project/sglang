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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::workers::Worker;

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

/// One worker's engine-reported load, summed across its dp ranks.
///
/// The two halves are stored, not the sum, so `waiting <= depth()` holds by
/// construction. Storing `running + waiting` and `waiting` instead would let a
/// caller build a value where the queue exceeds the total, and would discard
/// `running` — the quantity that answers whether an engine is near its
/// concurrency cap or queueing well below it, which is the observation this
/// whole load signal exists to expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkerDepth {
    running: usize,
    waiting: usize,
}

impl WorkerDepth {
    pub fn new(running: usize, waiting: usize) -> Self {
        Self { running, waiting }
    }

    /// Total outstanding work. Ranks who is least loaded.
    pub fn depth(&self) -> usize {
        self.running.saturating_add(self.waiting)
    }

    /// Requests admitted to the engine that have not started yet — the direct
    /// answer to "would a request sent here wait behind others", which
    /// [`Self::depth`] only approximates, and approximates badly: an engine can
    /// queue while running well below its concurrency cap, so equal depth does
    /// not imply equal queue.
    ///
    /// Summed across the worker's dp ranks while a request is dispatched to
    /// one of them, so on a multi-rank worker this reads roughly `dp_size`
    /// times the queue any single request would actually sit behind.
    pub fn waiting(&self) -> usize {
        self.waiting
    }

    /// Requests the engine is actively working on.
    pub fn running(&self) -> usize {
        self.running
    }

    fn add(&mut self, other: Self) {
        self.running = self.running.saturating_add(other.running);
        self.waiting = self.waiting.saturating_add(other.waiting);
    }
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
    /// Freshness window in milliseconds. Atomic so the Load Monitor can
    /// align it with `--load-monitor-stale-after-ms` after construction.
    freshness_ms: AtomicU64,
    /// Latest pull-mode status per worker URL; absent == last pull OK (or
    /// the worker is fed by the push socket only).
    pull_status: DashMap<String, PullStatus>,
}

/// Outcome of the latest pull-mode sample for a worker (the Load Monitor's
/// `GET /v1/loads`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullStatus {
    /// HTTP / transport / payload failure; the previous rows are kept for
    /// diagnostics but the worker reads as `Unreachable`.
    Unreachable(String),
    /// The engine answered but every rank reported zero token or request
    /// capacity (SGLang's early-boot snapshot); reads as `Stale`.
    ZeroCapacity,
}

/// Routing-visible freshness of a worker's load report, combining the
/// per-rank gauge age with the pull status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// No rank has ever reported.
    Missing,
    /// Every rank's latest gauge is younger than the freshness window.
    Fresh,
    /// Some rank's gauge is older than the window, or the engine reported
    /// zero capacity.
    Stale,
    /// The latest pull failed.
    Unreachable,
}

impl Freshness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Fresh => "fresh",
            Self::Stale => "stale",
            Self::Unreachable => "unreachable",
        }
    }
}

/// Endpoint-level aggregate over a worker's ranks (for diagnostics and
/// metrics; routing reads [`WorkerLoads`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerAggregate {
    pub num_running_requests: u64,
    pub num_waiting_requests: u64,
    pub num_used_tokens: u64,
    pub max_total_tokens: u64,
    pub ranks: usize,
}

impl WorkerAggregate {
    pub fn num_total_requests(&self) -> u64 {
        self.num_running_requests
            .saturating_add(self.num_waiting_requests)
    }

    pub fn free_tokens(&self) -> u64 {
        self.max_total_tokens.saturating_sub(self.num_used_tokens)
    }
}

fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

impl EngineLoadTable {
    pub fn new() -> Arc<Self> {
        Self::with_freshness(DEFAULT_FRESHNESS)
    }

    pub fn with_freshness(freshness: Duration) -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashSet::new(),
            freshness_ms: AtomicU64::new(duration_ms(freshness)),
            pull_status: DashMap::new(),
        })
    }

    /// Current freshness window.
    pub fn freshness(&self) -> Duration {
        Duration::from_millis(self.freshness_ms.load(Ordering::Relaxed))
    }

    /// Replace the freshness window (the Load Monitor aligns it with its
    /// `stale_after`, so push and pull reports age under one rule).
    pub fn set_freshness(&self, freshness: Duration) {
        self.freshness_ms
            .store(duration_ms(freshness), Ordering::Relaxed);
    }

    /// Record the outcome of a pull-mode sample. `None` clears a previous
    /// failure (the pull succeeded).
    pub fn set_pull_status(&self, url: &str, status: Option<PullStatus>) {
        match status {
            Some(s) => {
                self.pull_status.insert(url.to_string(), s);
            }
            None => {
                self.pull_status.remove(url);
            }
        }
    }

    pub fn pull_status(&self, url: &str) -> Option<PullStatus> {
        self.pull_status.get(url).map(|s| s.clone())
    }

    /// Freshness of one worker as routing sees it: a failed pull wins over
    /// the gauge age, zero capacity reads as stale, and a worker is fresh
    /// only when EVERY rank's latest gauge is inside the window.
    pub fn worker_freshness(&self, url: &str, now: Instant) -> Freshness {
        match self.pull_status.get(url).map(|s| s.clone()) {
            Some(PullStatus::Unreachable(_)) => return Freshness::Unreachable,
            Some(PullStatus::ZeroCapacity) => return Freshness::Stale,
            None => {}
        }
        let window = self.freshness();
        let mut seen = false;
        let mut all_fresh = true;
        for entry in self.by_rank.iter() {
            if entry.key().0 != url {
                continue;
            }
            seen = true;
            if now.saturating_duration_since(entry.value().at) > window {
                all_fresh = false;
            }
        }
        match (seen, all_fresh) {
            (false, _) => Freshness::Missing,
            (true, true) => Freshness::Fresh,
            (true, false) => Freshness::Stale,
        }
    }

    /// Sum of the latest gauges over a worker's ranks, with the oldest rank
    /// timestamp. `None` when no rank has reported.
    pub fn worker_aggregate(&self, url: &str) -> Option<(WorkerAggregate, Instant)> {
        let mut agg = WorkerAggregate::default();
        let mut oldest: Option<Instant> = None;
        for entry in self.by_rank.iter() {
            if entry.key().0 != url {
                continue;
            }
            let l = &entry.value().load;
            agg.num_running_requests = agg.num_running_requests.saturating_add(l.num_running_reqs);
            agg.num_waiting_requests = agg.num_waiting_requests.saturating_add(l.num_waiting_reqs);
            agg.num_used_tokens = agg.num_used_tokens.saturating_add(l.num_tokens);
            agg.max_total_tokens = agg.max_total_tokens.saturating_add(l.max_total_num_tokens);
            agg.ranks += 1;
            oldest = Some(oldest.map_or(entry.value().at, |o| o.min(entry.value().at)));
        }
        oldest.map(|at| (agg, at))
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

    /// Shared accumulation pass behind [`Self::snapshot_fresh`] (and used
    /// directly by `cache_aware_zmq::WorkerLoads`, which needs both halves):
    /// one walk of the table, per worker URL, producing the summed queue
    /// depth (`num_running_reqs + num_waiting_reqs`) across that worker's
    /// ranks and the OLDEST snapshot timestamp among them — **but only for
    /// workers whose every known rank is fresh**. A worker with any stale
    /// rank is omitted, so the caller falls back to its own load signal.
    /// (Summing only the fresh ranks would make a worker whose other ranks
    /// went silent look misleadingly idle and draw *more* traffic.)
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
    /// `depth` and `waiting` are kept apart rather than pre-summed because they
    /// answer different questions and the difference is load-bearing. `depth`
    /// ranks who is least loaded. `waiting` alone says whether a request sent
    /// here would sit behind others before its prefill starts — and on real
    /// traffic those diverge: engines have been observed queueing at 7-8
    /// running, far below `max_running_requests`, so a large `depth` is neither
    /// necessary nor sufficient for "this worker will make you wait".
    pub(crate) fn fresh_worker_state(
        &self,
        now: Instant,
    ) -> HashMap<String, (WorkerDepth, Instant)> {
        // A named accumulator, not a 4-tuple: the fields are read by name at
        // every update, so adding one cannot silently renumber the others.
        struct Acc {
            load: WorkerDepth,
            all_fresh: bool,
            oldest_at: Instant,
        }
        let mut acc: HashMap<String, Acc> = HashMap::new();
        for entry in self.by_rank.iter() {
            let at = entry.value().at;
            let fresh = now.saturating_duration_since(at) <= self.freshness();
            let l = &entry.value().load;
            // `try_from`, not `as`: the counts are engine-supplied and the
            // deserializer defaults missing fields, so they are not trusted
            // input. Two independent truncating casts could otherwise produce
            // `waiting > running + waiting` on a 32-bit target.
            let rank = WorkerDepth::new(
                usize::try_from(l.num_running_reqs).unwrap_or(usize::MAX),
                usize::try_from(l.num_waiting_reqs).unwrap_or(usize::MAX),
            );
            let slot = acc.entry(entry.key().0.clone()).or_insert(Acc {
                load: WorkerDepth::default(),
                all_fresh: true,
                oldest_at: at,
            });
            slot.load.add(rank);
            slot.all_fresh = slot.all_fresh && fresh;
            slot.oldest_at = slot.oldest_at.min(at);
        }
        acc.into_iter()
            .filter_map(|(url, a)| a.all_fresh.then_some((url, (a.load, a.oldest_at))))
            .collect()
    }

    /// Per worker URL, total depth across that worker's ranks, for workers
    /// whose every known rank is fresh.
    ///
    /// Test-only, and deliberately not public: it collapses the queue back into
    /// the depth sum, which is exactly the distinction the routing path depends
    /// on keeping. Production reads [`Self::fresh_worker_state`].
    #[cfg(test)]
    pub fn snapshot_fresh(&self, now: Instant) -> HashMap<String, usize> {
        self.fresh_worker_state(now)
            .into_iter()
            .map(|(url, (d, _))| (url, d.depth()))
            .collect()
    }

    /// Drop every rank entry (and the expected mark) for a worker. Called on
    /// worker removal so a re-added worker does not leave stale load behind.
    pub fn forget_worker(&self, url: &str) {
        self.by_rank.retain(|k, _| k.0 != url);
        self.expected.remove(url);
        self.pull_status.remove(url);
    }

    #[cfg(test)]
    pub fn entry_count(&self) -> usize {
        self.by_rank.len()
    }
}

/// One consistent per-selection view of engine-reported load: every
/// worker's fresh (`running + num_waiting`) plus its own dispatches acquired
/// since that snapshot's timestamp (see [`Self::load_of`]); a worker without
/// a fresh snapshot falls back to the router-side in-flight counter
/// (`Worker::active_load`). Holding the snapshot keeps every per-worker
/// `load_of` an O(1) map lookup. Shared by every load-aware policy
/// (cache-aware-zmq, power_of_two, load_based, the PD decode pick) so they
/// all score a worker the same way.
#[derive(Debug)]
pub struct WorkerLoads {
    /// url -> (engine-reported depth + queue, that snapshot's oldest-rank
    /// timestamp).
    fresh: HashMap<String, (WorkerDepth, Instant)>,
}

impl WorkerLoads {
    /// Build the per-selection snapshot from one `fresh_worker_state` pass.
    /// The single construction chokepoint guarantees every comparison in a
    /// given `select` sees one consistent view of load.
    pub fn from_engine(table: &EngineLoadTable, now: Instant) -> Self {
        Self {
            fresh: table.fresh_worker_state(now),
        }
    }

    /// A worker's current load: the engine-reported queue depth as of the
    /// last fresh snapshot, plus this worker's own dispatches made *since*
    /// that snapshot's timestamp — i.e. exactly the requests the engine
    /// hasn't had a chance to report back on yet. This is deliberately not
    /// the worker's full `active_load()`: that counter also includes
    /// long-held slots from slow-draining streaming responses that the
    /// engine's own last report has likely already accounted for.
    ///
    /// This correction is per-router-process: it only sees dispatches THIS
    /// router pod made. It closes the single-pod stale-gauge herd, but does
    /// not coordinate with other router replicas.
    pub fn load_of(&self, w: &Worker) -> usize {
        match self.fresh.get(w.url.as_str()) {
            Some(&(d, at)) => d.depth().saturating_add(w.slots_acquired_since(at)),
            None => w.active_load(),
        }
    }

    /// Whether `w` has a fresh engine snapshot in this view.
    pub fn is_fresh(&self, w: &Worker) -> bool {
        self.fresh.contains_key(w.url.as_str())
    }

    /// How many requests are queued on this worker's engine, or `None` when no
    /// fresh snapshot says.
    ///
    /// Deliberately not defaulted to 0 or to any router-side value. The
    /// router-side counter has no queue component at all — it cannot tell a
    /// dispatched-and-running request from a dispatched-and-waiting one — so
    /// there is no honest substitute, and inventing one would silently turn a
    /// queue gate into a comparison against a different quantity. `None` means
    /// "unknown", and callers gate open on it.
    ///
    /// Unlike [`Self::load_of`] this carries no since-snapshot correction: a
    /// dispatch this router just made is not known to be *queued* — the engine
    /// may well be running it — so adding it here would manufacture queue
    /// depth that may not exist.
    pub fn waiting_of(&self, w: &Worker) -> Option<usize> {
        self.fresh.get(w.url.as_str()).map(|(d, _)| d.waiting())
    }

    /// Number of workers whose load came from the engine (vs the router-side
    /// fallback). Used only to annotate the rebalance log.
    pub fn engine_worker_count(&self) -> usize {
        self.fresh.len()
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
    fn fresh_worker_state_picks_the_earliest_rank_timestamp() {
        let t = EngineLoadTable::new();
        let earlier = Instant::now() - Duration::from_secs(2);
        let later = earlier + Duration::from_secs(1);
        t.set("http://w:30000", 0, load(5, 1), later);
        t.set("http://w:30000", 1, load(3, 2), earlier);
        let now = later + Duration::from_millis(1);
        assert_eq!(
            t.fresh_worker_state(now).get("http://w:30000").copied(),
            Some((WorkerDepth::new(8, 3), earlier)),
            "must sum depth AND waiting across ranks, and expose the OLDEST rank's timestamp"
        );
    }

    /// `waiting` must survive the per-URL rank sum as its own quantity: the
    /// queue gate reads it directly, and collapsing it into `depth` is what the
    /// gate exists to stop doing.
    #[test]
    fn fresh_worker_state_keeps_waiting_separable_from_depth() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        // Same depth (13), very different queueing behaviour.
        t.set("http://queueing:30000", 0, load(8, 5), now);
        t.set("http://flowing:30000", 0, load(13, 0), now);
        let state = t.fresh_worker_state(now);
        assert_eq!(
            state.get("http://queueing:30000").map(|(d, _)| *d),
            Some(WorkerDepth::new(8, 5)),
        );
        assert_eq!(
            state.get("http://flowing:30000").map(|(d, _)| *d),
            Some(WorkerDepth::new(13, 0)),
            "equal depth must not imply equal queue"
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
        t.mark_expected("http://w:30000");
        t.mark_expected("http://w:30000"); // idempotent
        t.mark_expected("http://other:30000");
        assert_eq!(t.expected_count(), 2);
        t.forget_worker("http://w:30000");
        assert_eq!(t.expected_count(), 1);
    }
}
