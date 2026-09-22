// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Peer-snapshot bootstrap for the KV-event tree.
//!
//! A freshly started replica subscribes to each worker's KV topic mid-stream:
//! ZMQ SUB delivers deltas from whatever sequence the publisher has reached,
//! so every block already resident in the engine's radix cache is invisible to
//! that replica. Until traffic re-stores those blocks the replica routes
//! cache-blind, and — worse for the fleet — its own dispatches scatter
//! prefixes across workers that warm replicas were keeping consolidated.
//!
//! This module closes that window by pulling a tree snapshot from a warm
//! sibling replica over HTTP and grafting it beneath the live delta stream.
//!
//! # Ordering is the whole design
//!
//! The subscriber starts **first**, before any snapshot is fetched, so no
//! delta can slip through the gap between "snapshot taken on the peer" and
//! "subscription live here". While a rank is [`BootstrapState::Pending`] the
//! pump buffers its batches instead of applying them; once the snapshot is
//! grafted, the seeded cursor does two jobs: as a FILTER it discards the held
//! batches the snapshot already reflects, and as a WATERMARK it proves the
//! stream resumes at `peer_cursor + 1`. A hole there means a delta was lost, so
//! the rank bails to cold rather than splicing over it. Both jobs are why
//! [`PeerSnapshot`] carries the producer's cursors and not just tree state.
//!
//! The watermark check runs at graft time when batches are already held. When
//! none are — the common case, since the fetch usually beats the first event —
//! there is nothing to compare against yet, so the check is deferred to the first
//! batch that arrives. Reaching that state also means nothing arrived between
//! subscribing and grafting, and the subscriber is live before the fetch, so
//! silence is far more often "this rank published nothing" than "we lost a
//! delta". A bounded wait therefore ends in a QUESTION, not a verdict: the fleet
//! is asked whether the publisher moved past the watermark (any peer's cursor is
//! admissible — sequence numbers are the publisher's), and only a witness above
//! it discards the graft. See `SPLICE_PROOF_TIMEOUT` and `spawn_splice_probe`.
//!
//! # Failure is "run cold", never "run wrong" — with one bounded exception
//!
//! Every edge — unreachable peer, older peer without the endpoint, a peer that
//! is itself still cold, a block-size mismatch, a sequence gap — resolves to
//! [`BootstrapState::Failed`] for the affected rank, which means the replica
//! serves with today's behaviour (live deltas only). A stale or partial snapshot
//! is never grafted on a hope: a missed `BlockRemoved` would become a permanent
//! false cache hit, which is silent, whereas a cold rank is merely slow and is
//! visible in `sgl_router_kv_bootstrap_state`.
//!
//! Two paths do not fit that shape, and both are deliberate:
//!
//! * hold-back overflow discards the queued prefix, so the rank resumes from the
//!   overflow point rather than replaying it;
//! * a graft whose splice no batch and no peer can speak to is KEPT, tallied
//!   [`RankOutcome::WarmUnwitnessed`] rather than [`RankOutcome::Warm`].
//!   Discarding instead would cost every quiet fleet its warm tree on a timer,
//!   which is the regression this module exists to prevent; the residual risk is
//!   a delta dropped by a fresh, idle subscriber and never followed by another.
//!
//! # Trust boundary
//!
//! Snapshots arrive over the network, so the boundary is enforced by types
//! rather than by discipline. [`PeerSnapshot`] is the wire shape and carries
//! [`WireWorker`], not [`KvWorkerId`]; [`VettedSnapshot::from_wire`] is the only
//! bridge, and it resolves every wire identity against the locally discovered
//! worker set, so a peer cannot introduce a [`KvWorkerId`] the local registry
//! does not already know (which would otherwise let routing resolve to a
//! non-registered endpoint — see the provenance note on [`KvWorkerId`]).
//!
//! [`VettedSnapshot`]'s fields are private and
//! [`VettedSnapshot::graft_into`] is the only route to the tree's restore path
//! from outside this module, so structural validation — format, block size,
//! parent bounds — cannot be skipped by a caller that assembles nodes itself.
//! The tree re-checks shape on its own behalf regardless.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;

use parking_lot::Mutex;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::tree::{KvWorkerId, SnapshotNode};

/// Wire-format version. Bump on any incompatible change to
/// [`PeerSnapshot`]; a receiver rejects anything it does not recognise, so a
/// mixed-version fleet degrades to cold boots rather than corrupt trees.
pub const SNAPSHOT_FORMAT: u32 = 1;

/// Path served by the producer and fetched by the consumer.
pub const SNAPSHOT_PATH: &str = "/internal/kv_snapshot";

/// Ceiling on a decompressed peer snapshot, as a resource guard rather than a
/// format bound.
///
/// A legitimate body is bounded by the fleet's total KV capacity in blocks, which
/// is orders of magnitude below this; the ceiling exists because gzip lets a small
/// response demand an arbitrarily large allocation, and the producing route is
/// unauthenticated. Generous on purpose: rejecting a real snapshot costs a cold
/// boot, so this should only ever catch a body no honest peer would send.
const MAX_INFLATED_SNAPSHOT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Query parameter by which a consumer states how stale a cached snapshot it
/// will accept, in milliseconds. See [`PRODUCER_CACHE_TTL`].
pub const MAX_AGE_PARAM: &str = "max_age_ms";

/// Query parameter by which a consumer asks for the cursor table alone, with no
/// tree. See [`fetch_cursors`].
pub const CURSORS_ONLY_PARAM: &str = "cursors_only";

/// How long a producer may reuse an already-built snapshot for a request that
/// states no freshness requirement of its own.
///
/// # Why staleness is a correctness input, not a tuning knob
///
/// A cached snapshot was exported some time BEFORE the consumer that receives it
/// subscribed. Every event the publisher emitted in that window is in neither
/// place — not in the snapshot, not in the consumer's held queue — so the live
/// stream resumes above `cursor + 1`, the watermark reads a hole, and the graft
/// is discarded to [`RankOutcome::Gap`]. On a fleet whose ranks publish
/// continuously that window is the dominant remaining loss: a fixed TTL costs a
/// share of ranks to `Gap` even when no rank is abandoned or uncovered.
///
/// A shorter fixed TTL only makes that less likely. What removes it is letting
/// the CONSUMER state the requirement, since only the consumer knows when its
/// ranks began holding: [`MAX_AGE_PARAM`] carries "no older than this", the
/// producer rebuilds when its entry does not meet it, and a bootstrap fetch asks
/// for an export strictly after its own subscribe. Gap by staleness then cannot
/// happen rather than merely happening less.
///
/// So this constant is only the DEFAULT for requests that state no
/// requirement, sent by an older router image that does not send the parameter.
/// A splice probe is not such a caller: it asks for the cursor table alone
/// ([`CURSORS_ONLY_PARAM`]), which the producer reads live with no cache
/// involved. The default can be generous, which is what keeps a boot herd
/// sharing one walk.
pub const PRODUCER_CACHE_TTL: Duration = Duration::from_secs(2);

/// Default upper bound on one peer-snapshot fetch; see the
/// `bootstrap_fetch_timeout_cap_ms` field doc for the rationale, and
/// `default_bootstrap_fetch_timeout_cap_ms`, which must agree with this.
///
/// This bounds a transfer that is MAKING PROGRESS. A peer that stops sending is
/// caught far sooner by [`SNAPSHOT_FETCH_READ_TIMEOUT`], and one that never
/// answers by [`SNAPSHOT_FETCH_CONNECT_TIMEOUT`], so the old job of this
/// value — "a generous deadline cannot park the sweep on one hung peer" — now
/// belongs to those two. What is left for it to bound is a large but healthy
/// body, and sizing it for a hung peer instead is what makes a warm fleet
/// unbootstrappable: a snapshot on a large fleet runs to tens of megabytes
/// gzipped, hundreds inflated, and takes tens of seconds to serve. A value in
/// that same range times out every fetch, books every peer `Unreachable` and
/// abandons every rank — a failure that gets MORE likely the warmer (i.e. more
/// useful) the fleet's tree becomes.
pub const DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP: Duration = Duration::from_secs(120);

/// Bound on establishing the connection to a peer. A peer that is gone answers
/// this fast, so the sweep reaches its next candidate without spending the
/// per-fetch budget on a dead one.
pub const SNAPSHOT_FETCH_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Bound on the gap BETWEEN body chunks, not on the whole response.
///
/// This is what separates "hung" from "big". A total-request timeout cannot
/// tell them apart: it fires identically on a peer that sent nothing for 30s
/// and on one that streamed tens of megabytes perfectly for 31s. Bounding idle
/// time instead lets an arbitrarily large healthy snapshot complete while still
/// cutting a stalled peer loose in seconds.
pub const SNAPSHOT_FETCH_READ_TIMEOUT: Duration = Duration::from_secs(15);

/// Per-rank bootstrap outcome.
///
/// `Pending` is the only state in which the pump buffers rather than applies,
/// so every terminal transition must be reached on every path — a rank stuck
/// `Pending` would buffer forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapState {
    /// Subscribed; snapshot not yet applied. Batches are buffered.
    Pending,
    /// Snapshot grafted and cursor seeded.
    Recovered,
    /// No usable snapshot. The rank runs on live deltas only — today's
    /// behaviour, and not an error worth failing the process over.
    Failed,
}

impl BootstrapState {
    /// Numeric encoding for the `sgl_router_kv_bootstrap_state` gauge.
    pub fn as_metric(self) -> u64 {
        match self {
            Self::Pending => 0,
            Self::Recovered => 1,
            Self::Failed => 2,
        }
    }

    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Pending)
    }
}

/// How one snapshot FETCH from one peer turned out. Doubles as the `outcome`
/// label on `sgl_router_kv_peer_snapshot_total`.
///
/// Counted per peer attempt, never per rank — see [`RankOutcome`] for the
/// per-rank verdicts. Keeping the two in separate metrics is what makes either
/// one divisible by anything: a single accepted fetch can settle several ranks,
/// and a single rank can outlive many rejected fetches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotOutcome {
    /// Vetted, covers at least one rank we are bootstrapping, and was handed to
    /// the pump. Says nothing about whether any rank ended up warm — the splice
    /// check happens later, and reports via [`RankOutcome`].
    Accepted,
    /// Peer did not answer, or answered non-200 (including the 404 an older
    /// router image returns for an endpoint it does not serve).
    Unreachable,
    /// Peer answered but has no graftable state for us: still bootstrapping,
    /// settled with an empty tree, or holding a real tree that shares no
    /// carriers with our live workers (`NothingUsable`). The last case is a
    /// WARM peer — deliberately not what [`PeerSnapshot::is_cold`] means, so
    /// it cannot count toward the sweep's cold-fleet verdict even though it
    /// shares this metric label.
    ColdPeer,
    /// Peer answered with a snapshot we must not trust: unknown format, an
    /// invalid parent reference, or a block size that makes its hashes
    /// incomparable to ours.
    Rejected,
}

impl SnapshotOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Unreachable => "unreachable",
            Self::ColdPeer => "cold_peer",
            Self::Rejected => "rejected",
        }
    }
}

/// How one bounded peer sweep ended. Doubles as the `result` label on
/// `sgl_router_kv_bootstrap_sweep_total`; a closed, compiler-checked set like
/// [`SnapshotOutcome`]'s, since a bare string label would let any call site
/// mint new metric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepOutcome {
    Found,
    NoPeers,
    FleetCold,
    TimedOut,
}

impl SweepOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Found => "found",
            Self::NoPeers => "no_peers",
            Self::FleetCold => "fleet_cold",
            Self::TimedOut => "timed_out",
        }
    }
}

/// How one rank's bootstrap ended. Doubles as the `outcome` label on
/// `sgl_router_kv_bootstrap_rank_total`.
///
/// Recorded exactly once per rank per incarnation, at the moment the verdict
/// becomes final — which for a graft is when the splice is PROVEN, not when the
/// tree state lands. That lag is deliberate: a rank counted warm at graft time
/// and later demoted by the deferred watermark check would be counted twice,
/// under two labels, leaving "how many ranks actually ended warm" underivable.
/// [`BootstrapState`] still reports the live per-rank state without the lag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankOutcome {
    /// Snapshot grafted AND its continuity with the live stream proven.
    Warm,
    /// The live stream does not join up with the snapshot's watermark, so the
    /// grafted state was discarded.
    Gap,
    /// Grafted, kept, but never witnessed: no batch arrived to prove the splice
    /// and no peer produced a cursor table naming the rank. Distinct from
    /// `Warm` on purpose — the state is being trusted on the absence of
    /// contrary evidence rather than on proof. An operator seeing this label
    /// persistently has one of two problems: peers are unreachable, or the
    /// fleet is mid-rollout with every peer on an older image — an old
    /// producer's full export omits ranks whose blocks it no longer carries,
    /// so only a patched peer can witness them.
    WarmUnwitnessed,
    /// The accepted snapshot tracked no cursor for this rank, so there was no
    /// watermark to splice against.
    Uncovered,
    /// No peer supplied a usable snapshot: the bootstrap deadline expired,
    /// discovery confirmed there are no siblings, or every sibling proved it
    /// has no state to give (settled-empty or permanently incompatible).
    Abandoned,
    /// Too many batches piled up waiting for a snapshot; the held prefix was
    /// dropped, so no snapshot can be spliced across it.
    Overflow,
    /// The publisher restarted mid-bootstrap, renumbering the stream the
    /// snapshot's watermark refers to.
    PublisherReset,
    /// The tree refused the snapshot's structure at graft time.
    TreeRejected,
}

impl RankOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Gap => "gap",
            Self::WarmUnwitnessed => "warm_unwitnessed",
            Self::Uncovered => "uncovered",
            Self::Abandoned => "abandoned",
            Self::Overflow => "overflow",
            Self::PublisherReset => "publisher_reset",
            Self::TreeRejected => "tree_rejected",
        }
    }
}

/// Worker identity as it crosses the wire.
///
/// Deliberately NOT [`KvWorkerId`]: deserialising straight into that type
/// would mint routing identities from network input, which its provenance
/// contract forbids. [`VettedSnapshot::from_wire`] is the only bridge, and it
/// only ever hands back ids that came from the local live set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireWorker {
    pub url: String,
    pub dp_rank: u32,
}

/// A peer replica's view of the KV tree, plus the cursors needed to splice it
/// under a live delta stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerSnapshot {
    /// See [`SNAPSHOT_FORMAT`].
    pub format: u32,
    /// Producer's established block size. Block hashes are only comparable at
    /// the same page size, so a mismatch is fatal to the snapshot.
    pub block_size: u32,
    /// Producer's primary hashing mode (EAGLE-family workers hash token
    /// bigrams), derived from its live worker set at export time. Kept for
    /// older consumers that still vet on it; current consumers do not — see
    /// [`VettedSnapshot::from_wire`].
    pub is_bigram: bool,
    /// Whether the producer is a tree worth copying: its own bootstrap has
    /// settled AND it actually holds nodes.
    ///
    /// The second half matters during a rolling update. Settlement latches once
    /// the bootstrap deadline expires, so a replica that failed its own
    /// bootstrap is "settled" with an empty tree; without the node check two
    /// new replicas could bootstrap from each other and both inherit nothing.
    ///
    /// The primary defence is upstream of this flag: a replica that has not
    /// passed `/readyz` is not `ready` in its EndpointSlice, so peer discovery
    /// never offers it as a candidate. This field covers the propagation race
    /// where readiness and EndpointSlice state briefly disagree.
    pub producer_ready: bool,
    /// Worker table; node carrier lists index into this.
    pub workers: Vec<WireWorker>,
    /// `(worker-table index, last-applied seq)` on the producer at export
    /// time. The consumer seeds these so its pump can filter the deltas the
    /// snapshot already reflects.
    pub cursors: Vec<(u32, i64)>,
    /// Tree nodes in dependency order; see [`SnapshotNode`].
    pub nodes: Vec<SnapshotNode>,
}

impl PeerSnapshot {
    /// Whether this body is unusable as a graft source right now: the producer
    /// says it is not ready, or the tree is empty. The single definition of
    /// "cold" for vetting ([`VetError::ProducerCold`]).
    pub fn is_cold(&self) -> bool {
        !self.producer_ready || self.holds_no_state()
    }

    /// Whether this body proves the producer has nothing to hand over — an
    /// empty tree.
    ///
    /// Deliberately weaker than [`Self::is_cold`], and the predicate the peer
    /// sweep's cold-fleet verdict is built on. `producer_ready` conjoins "my
    /// own bootstrap has settled" with "my tree is non-empty", so a peer that
    /// is mid-bootstrap while already ingesting live events answers
    /// `producer_ready: false` with a NON-empty node list. Vetting must still
    /// refuse to graft from it, but the sweep must keep waiting on it:
    /// counting it as a cold witness lets a joining replica settle cold on the
    /// first pass while a sibling seconds from being a usable source is the
    /// only candidate in the fleet.
    pub fn holds_no_state(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Last-applied sequence this producer reports for a rank, addressed by wire
    /// identity rather than [`KvWorkerId`].
    ///
    /// Deliberately available without vetting: the caller wants one number to
    /// compare against a watermark, not tree state to graft. Sequence numbers are
    /// minted by the publisher, so any observer reporting seq N proves the
    /// publisher emitted N — which is what makes a peer's cursor usable as
    /// evidence about a rank's progress even from a peer too cold to bootstrap
    /// from.
    pub fn wire_cursor_for(&self, url: &str, dp_rank: u32) -> Option<i64> {
        let idx = self
            .workers
            .iter()
            .position(|w| w.url == url && w.dp_rank == dp_rank)? as u32;
        self.cursors
            .iter()
            .find(|(i, _)| *i == idx)
            .map(|(_, seq)| *seq)
    }
}

/// Rate-limit key for the snapshot-attempt log: one entry per peer per
/// verdict.
type PeerOutcomeKey = (String, &'static str);

/// Rate-limit state for one [`PeerOutcomeKey`]: attempt count plus the
/// last-seen detail, so a cause CHANGE under the same verdict is surfaced
/// rather than throttled away.
type AttemptLogState = (u64, String);

/// Tracks per-rank bootstrap progress and answers the one question `/readyz`
/// needs: has initial bootstrap settled?
///
/// # Why the answer latches
///
/// Once settled, [`BootstrapTracker::settled`] stays true forever. Without
/// that latch a steady-state scale-up would register a fresh `Pending` rank
/// and flip an already-serving replica back to 503 — turning a cache
/// optimisation into an availability incident. The gate exists only to hold
/// traffic off a *newly booted* replica.
#[derive(Debug)]
pub struct BootstrapTracker {
    states: Mutex<HashMap<KvWorkerId, BootstrapState>>,
    /// Set when the first worker is registered; the deadline runs from there
    /// rather than from process start, so slow worker discovery does not eat
    /// the bootstrap budget.
    deadline: Mutex<Option<Instant>>,
    timeout: Duration,
    /// Upper bound on one peer-snapshot fetch within `timeout`. Carried here
    /// so the sweep can size its HTTP client from the same configured object
    /// as the deadline that bounds it.
    fetch_cap: Duration,
    /// Whether peer bootstrap is configured at all (a `--kv-peer-selector` is
    /// set), as opposed to configured-and-already-finished.
    ///
    /// WHY this is not derivable from `settled()`: both a
    /// [`BootstrapTracker::disabled`] tracker and one that has completed its work
    /// report settled, but they must be treated oppositely at `add_worker`.
    /// Disabled means "never hold this rank's batches"; finished means "readiness
    /// is already green, but a newly discovered worker should still be warmed".
    /// Collapsing the two silently denies a late worker its snapshot and leaves
    /// its subtree cold — the tree ends up short by that worker's blocks with
    /// nothing in the metrics to say so.
    enabled: bool,
    latched: AtomicBool,
    /// Whether the one-shot re-arm at first worker discovery has happened.
    ///
    /// Keyed on this rather than on `states.is_empty()`: the map empties again
    /// whenever `forget` removes the last rank, so an engine flapping
    /// remove/re-add would re-arm on every cycle and hold the readiness gate for
    /// an unbounded multiple of the configured timeout.
    rearmed: AtomicBool,
    /// Per-rank incarnation number, bumped whenever a rank is registered afresh
    /// (i.e. after a `forget`, which is what `remove_worker` does).
    ///
    /// WHY: `spawn_bootstrap` runs detached for up to the whole bootstrap
    /// deadline. A worker removed and re-added inside that window is `Pending`
    /// again, so the in-flight task's snapshot would be grafted onto the NEW
    /// incarnation and seed a watermark from the OLD publisher's numbering —
    /// after which every batch from the fresh publisher (restarting at seq 1) is
    /// filtered as out-of-order. The rank then sits on stale tree state
    /// indefinitely while reporting `Recovered`: "run wrong", which this module
    /// otherwise rules out. A process-wide counter is NOT sufficient, because a
    /// concurrent `add_worker` for an unrelated worker would invalidate a valid
    /// task; the epoch has to be per rank.
    epochs: Mutex<HashMap<KvWorkerId, u64>>,
    epoch_seq: AtomicU64,
    /// Ranks already given one post-gap retry.
    ///
    /// A gap means the grafted state was discarded because the live stream did not
    /// join the snapshot's watermark — the most wasteful failure there is, since a
    /// snapshot WAS fetched and then thrown away. A fresher snapshot usually
    /// splices, so it is worth one more sweep. Capped at one per rank: a rank that
    /// keeps gapping would otherwise re-fetch a fleet-wide body indefinitely.
    gap_retried: Mutex<HashSet<KvWorkerId>>,
    /// Per-peer-attempt tallies keyed by [`SnapshotOutcome::as_label`].
    ///
    /// Kept here rather than pushed into the metrics registry because the
    /// bootstrap path has no metrics handle, and the metrics surface already
    /// samples live state at scrape time for the per-worker gauges. Labels are
    /// a fixed `&'static str` set, so cardinality is bounded.
    peer_outcomes: Mutex<HashMap<&'static str, u64>>,
    /// Per-rank tallies keyed by [`RankOutcome::as_label`]. Same storage
    /// reasoning as `peer_outcomes`; separate map because the two count
    /// different things and mixing them makes both undivisible.
    rank_outcomes: Mutex<HashMap<&'static str, u64>>,
    /// Per-(peer, outcome) fetch-attempt counts and last-seen detail, used
    /// only to rate-limit the attempt log in [`Self::record_peer_outcome`].
    peer_attempts: Mutex<HashMap<PeerOutcomeKey, AttemptLogState>>,
    /// Per-sweep-verdict tallies keyed by [`SweepOutcome::as_label`]. Rank and
    /// fetch outcomes alone cannot separate "settled cold as soon as every
    /// sibling proved empty" from "burned the whole deadline" — both end in
    /// `RankOutcome::Abandoned` — so the sweep's own verdict gets a counter.
    sweep_results: Mutex<HashMap<&'static str, u64>>,
}

impl BootstrapTracker {
    pub fn new(timeout: Duration) -> Self {
        Self::new_with_fetch_cap(timeout, DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP)
    }

    pub fn new_with_fetch_cap(timeout: Duration, fetch_cap: Duration) -> Self {
        Self {
            fetch_cap,
            states: Mutex::new(HashMap::new()),
            // Armed at construction, NOT lazily at first registration.
            //
            // WHY: the deadline is the only escape from the readiness gate, and
            // several paths reach "workers are registered but this tracker has no
            // ranks" — a worker that publishes no KV events, a `/server_info`
            // failure, a block-size mismatch. With a lazily-armed deadline those
            // leave `settled()` false forever, so `/readyz` never returns 200 and
            // the pod is bricked. Worse, it is self-sustaining fleet-wide: an
            // unready replica is absent from its own EndpointSlice, so every
            // replica would see an empty peer set. An always-armed clock cannot
            // deadlock. `register` re-arms once so the normal path still measures
            // from first worker discovery rather than from process start.
            deadline: Mutex::new(Some(Instant::now() + timeout)),
            timeout,
            enabled: true,
            latched: AtomicBool::new(false),
            rearmed: AtomicBool::new(false),
            epochs: Mutex::new(HashMap::new()),
            epoch_seq: AtomicU64::new(0),
            gap_retried: Mutex::new(HashSet::new()),
            peer_outcomes: Mutex::new(HashMap::new()),
            rank_outcomes: Mutex::new(HashMap::new()),
            peer_attempts: Mutex::new(HashMap::new()),
            sweep_results: Mutex::new(HashMap::new()),
        }
    }

    /// Tally one snapshot FETCH against one peer, and log it.
    pub fn record_peer_outcome(&self, outcome: SnapshotOutcome, peer: &str, detail: Option<&str>) {
        *self
            .peer_outcomes
            .lock()
            .entry(outcome.as_label())
            .or_insert(0) += 1;
        if outcome == SnapshotOutcome::Accepted {
            return;
        }
        let detail = detail.unwrap_or("");
        let (attempts, detail_changed) = {
            let mut counts = self.peer_attempts.lock();
            let entry = counts
                .entry((peer.to_string(), outcome.as_label()))
                .or_insert((0, String::new()));
            entry.0 += 1;
            let changed = entry.1 != detail;
            if changed {
                entry.1 = detail.to_string();
            }
            (entry.0, changed && entry.0 > 1)
        };
        // The first failure against a peer is signal; the thousandth repeat of
        // the same verdict is not — but a sweep kept alive by unreachable or
        // non-covering peers must still show progress at the default log level.
        // A detail CHANGE under the same verdict (connection-refused becoming
        // DNS, 404 becoming 503) is a new cause, not a repeat: surface it too.
        if attempts == 1 || attempts % 100 == 0 || detail_changed {
            info!(
                peer = %peer,
                outcome = outcome.as_label(),
                attempts,
                detail,
                "kv-bootstrap: snapshot attempt did not yield state",
            );
        } else {
            debug!(
                peer = %peer,
                outcome = outcome.as_label(),
                attempts,
                detail,
                "kv-bootstrap: snapshot attempt did not yield state",
            );
        }
    }

    /// Tally one rank's final bootstrap verdict. Called from the pump only, so
    /// the once-per-rank property is maintained by the state gates there.
    pub fn record_rank_outcome(&self, outcome: RankOutcome) {
        *self
            .rank_outcomes
            .lock()
            .entry(outcome.as_label())
            .or_insert(0) += 1;
    }

    /// Per-peer-attempt tallies for the metrics surface.
    pub fn peer_outcome_counts(&self) -> Vec<(&'static str, u64)> {
        self.peer_outcomes
            .lock()
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    /// Per-rank tallies for the metrics surface.
    pub fn rank_outcome_counts(&self) -> Vec<(&'static str, u64)> {
        self.rank_outcomes
            .lock()
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    /// Tally one peer sweep's terminal verdict. Called once per sweep, from
    /// the delivery point, so the counts sum to the number of sweeps run.
    ///
    /// `peers_tried` is the size of the candidate set the verdict was proven
    /// over. Nothing here uses it yet; it is recorded because a `TimedOut`
    /// over a non-empty candidate set means "siblings were there and we could
    /// not get their state", while the same verdict over an empty one means
    /// only that discovery had not caught up — a distinction a readiness gate
    /// cannot be built without.
    pub fn record_sweep_result(&self, result: SweepOutcome, peers_tried: usize) {
        let _ = peers_tried;
        *self
            .sweep_results
            .lock()
            .entry(result.as_label())
            .or_insert(0) += 1;
    }

    /// Per-sweep-verdict tallies for the metrics surface.
    pub fn sweep_result_counts(&self) -> Vec<(&'static str, u64)> {
        self.sweep_results
            .lock()
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    /// A tracker that is settled from the start, for the paths where peer
    /// bootstrap is disabled entirely.
    pub fn disabled() -> Self {
        let mut t = Self::new(Duration::ZERO);
        t.enabled = false;
        t.latched.store(true, Ordering::Relaxed);
        t
    }

    /// Whether peer bootstrap is configured. Unlike [`BootstrapTracker::settled`]
    /// this never flips: it answers "may ranks bootstrap at all?", not "is
    /// readiness still waiting?".
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Upper bound on one peer-snapshot fetch within [`Self::timeout`]; see
    /// [`super::index::snapshot_fetch_timeout`], which combines the two.
    pub fn fetch_cap(&self) -> Duration {
        self.fetch_cap
    }

    /// Ranks added after the tracker has latched are recorded (so metrics stay
    /// accurate) but cannot un-settle readiness.
    /// Register ranks as `Pending` and return the obligation set: each rank
    /// paired with the incarnation number a later control message must still
    /// match to be allowed to act on it.
    pub fn register(&self, ids: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let mut states = self.states.lock();
        // Re-arm exactly once, at first worker discovery, so the budget is not
        // spent by slow discovery. Strictly one-shot — see `rearmed`.
        if !self.rearmed.swap(true, Ordering::Relaxed) {
            *self.deadline.lock() = Some(Instant::now() + self.timeout);
            // Clearing the latch is not a readiness regression, it is what
            // makes the gate exist at all.
            //
            // `settled()` latches on the CONSTRUCTION-time deadline, and two
            // callers reach it before any rank is registered: a `/metrics`
            // scrape, and a sibling's `/internal/kv_snapshot` fetch (both read
            // it, and the second does so on the cheap cursors-only probe too).
            // On a fleet whose first `add_worker` lands after the budget has
            // already elapsed — a large fleet with a slow `/server_info`
            // fan-out — either one latches `settled()` with an empty rank map,
            // the re-arm above is then read through a short-circuit that never
            // consults it, and the readiness gate is dead for the life of the
            // process: `/readyz` goes green the moment one worker is
            // registered, while the very first sweep is still in flight.
            //
            // Safe because it is strictly one-shot and paired with the re-arm:
            // the escape hatch is still armed, so a replica whose ranks never
            // resolve still settles on the new deadline. It also cannot be a
            // scale-up un-readying a serving replica — this runs at the FIRST
            // registration, which is boot.
            self.latched.store(false, Ordering::Relaxed);
        }
        let mut epochs = self.epochs.lock();
        let mut obligations = Vec::with_capacity(ids.len());
        for id in ids {
            let fresh = !states.contains_key(id);
            states.entry(id.clone()).or_insert(BootstrapState::Pending);
            // A rank still present kept its incarnation; one that was forgotten
            // (remove_worker) gets a new one, which invalidates any bootstrap
            // task still in flight for the previous incarnation.
            let epoch = if fresh {
                let e = self.epoch_seq.fetch_add(1, Ordering::Relaxed) + 1;
                epochs.insert(id.clone(), e);
                e
            } else {
                // A registered rank always has an epoch (`register` and `forget`
                // write both maps together). Mint rather than default to 0 so a
                // future asymmetry cannot make two incarnations share a number.
                *epochs
                    .entry(id.clone())
                    .or_insert_with(|| self.epoch_seq.fetch_add(1, Ordering::Relaxed) + 1)
            };
            obligations.push((id.clone(), epoch));
        }
        obligations
    }

    /// Current incarnation of `id`, or `None` if it is not registered.
    pub fn epoch_of(&self, id: &KvWorkerId) -> Option<u64> {
        self.epochs.lock().get(id).copied()
    }

    /// Update a REGISTERED rank's state. A rank that is not registered is
    /// ignored.
    ///
    /// WHY non-creating: `forget` is the authority on membership, and it removes
    /// from both `states` and `epochs`. An inserting `set` could resurrect a
    /// state entry for a forgotten rank, leaving state without an epoch — after
    /// which `register`'s non-fresh branch mints epoch 0 (so two incarnations
    /// could share a number, defeating the incarnation gate) and its
    /// `or_insert(Pending)` leaves the zombie `Recovered`, so the new incarnation
    /// never buffers, never bootstraps, and the gauge reports `recovered` for a
    /// rank that never ran.
    pub fn set(&self, id: &KvWorkerId, state: BootstrapState) {
        if let Some(slot) = self.states.lock().get_mut(id) {
            *slot = state;
        }
    }

    pub fn forget(&self, ids: &[KvWorkerId]) {
        let mut states = self.states.lock();
        let mut epochs = self.epochs.lock();
        for id in ids {
            states.remove(id);
            epochs.remove(id);
        }
    }

    /// Move a gap-discarded rank back to `Pending` so it can be swept again,
    /// returning the obligation to queue.
    ///
    /// `None` when the retry must not happen: already retried once, the rank was
    /// forgotten or re-registered by a new incarnation, or it is not in the
    /// `Failed` state a gap leaves behind. The caller MUST queue a returned
    /// obligation — a `Pending` rank nobody owns holds its batches until the
    /// pump's per-rank cap overflows.
    ///
    /// Readiness is unaffected: `settled()` short-circuits on `latched`, so a rank
    /// returning to `Pending` cannot drag `/readyz` back to 503.
    pub fn retry_after_gap(&self, id: &KvWorkerId) -> Option<(KvWorkerId, u64)> {
        let mut states = self.states.lock();
        if states.get(id) != Some(&BootstrapState::Failed) {
            return None;
        }
        // Same lock order as `register` (states, then epochs).
        let epoch = *self.epochs.lock().get(id)?;
        if !self.gap_retried.lock().insert(id.clone()) {
            return None;
        }
        states.insert(id.clone(), BootstrapState::Pending);
        Some((id.clone(), epoch))
    }

    pub fn state_of(&self, id: &KvWorkerId) -> Option<BootstrapState> {
        self.states.lock().get(id).copied()
    }

    /// Snapshot of every tracked rank, for the metrics surface.
    pub fn states(&self) -> Vec<(KvWorkerId, BootstrapState)> {
        self.states
            .lock()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Whether every rank a `Pending` entry exists for has reached a terminal
    /// state, or the deadline has passed. Latches on first `true`.
    ///
    /// The deadline is always armed (see [`BootstrapTracker::new`]), so this
    /// cannot stay `false` forever regardless of what does or does not get
    /// registered.
    pub fn settled(&self) -> bool {
        if self.latched.load(Ordering::Relaxed) {
            return true;
        }
        let all_terminal = {
            let states = self.states.lock();
            !states.is_empty() && states.values().all(|s| s.is_terminal())
        };
        let expired = self.deadline.lock().is_some_and(|d| Instant::now() >= d);
        if all_terminal || expired {
            self.latched.store(true, Ordering::Relaxed);
            if expired && !all_terminal {
                warn!(
                    timeout_ms = self.timeout.as_millis(),
                    "kv-bootstrap: deadline elapsed with ranks still pending; \
                     serving with a partially warmed cache-aware tree",
                );
            }
            return true;
        }
        false
    }

    /// Time left before the deadline forces settlement, or `None` when no
    /// deadline is armed yet.
    pub fn time_remaining(&self) -> Option<Duration> {
        self.deadline
            .lock()
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
}

/// The set of sibling replicas this router may pull a snapshot from.
///
/// Written by peer discovery, read by the bootstrap task. Self is excluded by
/// the discovery layer; this type only stores what it is given.
#[derive(Debug, Default)]
pub struct PeerRegistry {
    peers: Mutex<Vec<String>>,
    /// Whether peer discovery has reported at least once, even with an empty
    /// result.
    ///
    /// WHY this matters: an empty peer set is ambiguous. It means either "the
    /// watch has not delivered yet" — routine, since worker discovery regularly
    /// wins the race against it — or "this replica genuinely has no siblings".
    /// Treating the first as the second makes a joining replica give up before
    /// its peers are even known and boot cold, which silently defeats the whole
    /// feature. Only after a sync is an empty set conclusive.
    synced: AtomicBool,
    /// Whether a non-empty peer set has ever been observed.
    ///
    /// The peer set legitimately dips to empty for an instant — an EndpointSlice
    /// repack deletes the last slice before its replacement arrives, and during a
    /// rolling update every sibling can be `notReady` at once. The bootstrap retry
    /// loop samples `known_to_have_no_peers()` many times per boot, so it has many
    /// chances to catch such a dip, and one hit is permanent for that boot. Once
    /// siblings have been seen, "empty" is treated as transient and the loop keeps
    /// waiting rather than concluding this replica is alone.
    ever_had_peers: AtomicBool,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace(&self, peers: Vec<String>) {
        self.synced.store(true, Ordering::Relaxed);
        if !peers.is_empty() {
            self.ever_had_peers.store(true, Ordering::Relaxed);
        }
        let mut guard = self.peers.lock();
        if *guard != peers {
            info!(count = peers.len(), peers = ?peers, "kv-bootstrap: peer set updated");
        }
        *guard = peers;
    }

    pub fn len(&self) -> usize {
        self.peers.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.lock().is_empty()
    }

    /// Whether peer discovery has reported at least once. An empty peer set is
    /// only conclusive once this is true.
    pub fn synced(&self) -> bool {
        self.synced.load(Ordering::Relaxed)
    }

    /// True when discovery has confirmed this replica has no siblings, so there
    /// is no point waiting for one.
    pub fn known_to_have_no_peers(&self) -> bool {
        self.synced() && self.is_empty() && !self.ever_had_peers.load(Ordering::Relaxed)
    }

    /// Candidate peers in shuffled order, so simultaneous boots spread their
    /// snapshot fetches instead of stampeding whichever peer sorts first.
    pub fn candidates(&self) -> Vec<String> {
        let mut peers = self.peers.lock().clone();
        peers.shuffle(&mut rand::thread_rng());
        peers
    }
}

/// The URL a snapshot fetch goes to, carrying the caller's freshness
/// requirement when it has one.
fn snapshot_url(peer_base_url: &str, max_age: Option<Duration>) -> String {
    let base = format!("{}{}", peer_base_url.trim_end_matches('/'), SNAPSHOT_PATH);
    match max_age {
        // Saturating rather than wrapping: an absurd duration must read as "any
        // age will do", never as a near-zero age that forces a needless rebuild.
        Some(d) => {
            let ms = u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
            format!("{base}?{MAX_AGE_PARAM}={ms}")
        }
        None => base,
    }
}

/// The URL a splice probe goes to. Carries no freshness requirement: the
/// producer reads its cursors live on this path, so there is no cached
/// generation to negotiate against.
///
/// `pub(crate)` so route tests can drive the wire through the same builder —
/// passing an empty base yields a path-relative URI usable in-process.
pub(crate) fn cursors_url(peer_base_url: &str) -> String {
    format!(
        "{}{}?{CURSORS_ONLY_PARAM}=true",
        peer_base_url.trim_end_matches('/'),
        SNAPSHOT_PATH,
    )
}

/// Fetch one peer's snapshot. `Ok(None)` means "peer reachable but has no
/// snapshot to give" (non-200, including an older image's 404).
///
/// `max_age` is the oldest export the caller can use — a correctness input for
/// a bootstrap, since only the consumer knows when its ranks began holding; see
/// [`PRODUCER_CACHE_TTL`]. `None` accepts whatever the peer has cached; no
/// production path asks for that — it is the behaviour of an older router
/// image, kept because a mixed-version fleet's old consumer asks exactly this
/// (and component tests exercise it). An older PRODUCER ignores the parameter
/// and answers from its own cache, so the fleet degrades to the pre-parameter
/// behaviour rather than failing.
/// What a snapshot fetch got back.
pub enum FetchAnswer {
    Body(PeerSnapshot),
    /// The peer answered non-success. The status distinguishes "an older
    /// router image that does not serve this route" (404) from a sick peer
    /// (5xx) — both retriable, but the first-occurrence log should name which.
    NoBody(reqwest::StatusCode),
}

pub async fn fetch_snapshot(
    http: &reqwest::Client,
    peer_base_url: &str,
    max_age: Option<Duration>,
) -> Result<FetchAnswer, anyhow::Error> {
    fetch_body(http, &snapshot_url(peer_base_url, max_age), peer_base_url).await
}

/// Fetch one peer's cursor table for a splice probe.
///
/// Asks the producer to omit the tree. The probe reads a single sequence number
/// through [`PeerSnapshot::wire_cursor_for`] and has no use for nodes, so on a
/// large fleet fetching a full export per unproven rank per peer is the dominant
/// cost of the whole bootstrap path — and it scales with the tree, which is the
/// one thing that grows without bound.
///
/// A producer that does not recognise the parameter ignores it and answers with
/// a full snapshot. That decodes to the same type, and its cursor table answers
/// the same witness question — though it is a SUBSET of the cursors-only one:
/// the full export filters its table to ranks still carrying tree nodes, while
/// the dedicated path reports every observed rank (see
/// `KvEventIndex::peer_cursors_body`). A mixed-version fleet therefore degrades
/// to the old transfer cost — and to observing only the witnesses an old
/// producer would have reported — rather than to a false or missing witness.
/// This is also why the shared decode still hands off to the blocking pool: on
/// that fallback the body really can be hundreds of megabytes.
///
/// Never gate the answer on `producer_ready`: in this body it means "worth
/// COPYING", and is gated on holding nodes, while the probe's question is "did
/// you OBSERVE this publisher". A peer with an empty tree can still report the
/// one cursor entry that proves advance, which is why the probe consults
/// [`PeerSnapshot::wire_cursor_for`] unconditionally.
pub async fn fetch_cursors(
    http: &reqwest::Client,
    peer_base_url: &str,
) -> Result<Option<PeerSnapshot>, anyhow::Error> {
    match fetch_body(http, &cursors_url(peer_base_url), peer_base_url).await {
        Ok(FetchAnswer::Body(snap)) => Ok(Some(snap)),
        Ok(FetchAnswer::NoBody(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Shared transport for both fetch shapes: send, branch on what the peer
/// actually encoded, and decode off the runtime.
///
/// `peer_base_url` is carried separately from `url` only so the log names the
/// peer rather than a URL with a query string on it.
///
/// # Compression
///
/// The snapshot body is the largest thing this router ever transfers — it carries
/// the whole tree, so it grows with the node count rather than with the fleet size
/// — and slow transfers are what starve the bootstrap deadline, so the request
/// asks for gzip and the producer compresses that route.
///
/// Negotiated by hand rather than via reqwest's `gzip` feature: that feature is
/// crate-wide, and with it compiled in `Accepts::default()` sets `gzip: true`, so
/// EVERY client in the process would start advertising gzip and auto-decoding
/// responses — including the proxy client carrying SSE on the hot path. Asking
/// on this one request keeps the behaviour change where it belongs.
async fn fetch_body(
    http: &reqwest::Client,
    url: &str,
    peer_base_url: &str,
) -> Result<FetchAnswer, anyhow::Error> {
    let resp = http
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    if !resp.status().is_success() {
        debug!(
            peer = %peer_base_url,
            status = %resp.status(),
            "kv-bootstrap: peer returned no usable body for the snapshot request",
        );
        return Ok(FetchAnswer::NoBody(resp.status()));
    }
    // Branch on what the peer actually sent, not on what we asked for: a router
    // image that predates route compression answers identity, and must keep
    // working. reqwest leaves both the header and the body untouched here because
    // its `gzip` feature is off.
    //
    // Exact-match on purpose. The only legitimate producer is another router of
    // this codebase, which sends the bare token, so a compound or non-canonical
    // coding means an intermediary rewrote the body — better surfaced as a decode
    // failure than guessed at.
    let gzipped = resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"gzip"));
    // Bounded BEFORE buffering, not only at the inflate below. The producing
    // route is unauthenticated, so an `identity` response can demand an
    // arbitrarily large allocation with none of gzip's amplification needed —
    // and an OOM here kills the process before it can log, which is the
    // failure the inflate ceiling was added to prevent. Counted across chunks
    // because `Content-Length` is absent on a chunked response.
    let body = read_bounded(resp, peer_base_url).await?;
    // Inflating and parsing a multi-megabyte tree is seconds of CPU with no await
    // point, and this runs on the runtime that also proxies requests — including
    // the SSE hot path. Hand it to the blocking pool instead of occupying a worker
    // for the whole decode. A cursors-only body does not need this, but the
    // fallback against an unpatched peer does, and one transport path is worth
    // more than saving a task hop on the cheap case.
    tokio::task::spawn_blocking(move || decode_snapshot(&body, gzipped))
        .await?
        .map(FetchAnswer::Body)
}

/// Buffer a response body, refusing one that grows past
/// [`MAX_INFLATED_SNAPSHOT_BYTES`].
///
/// The same ceiling as the inflate, because the resource being protected is
/// the same: a booting router's heap. A legitimate snapshot is bounded by the
/// fleet's KV capacity in blocks, orders of magnitude below this.
async fn read_bounded(
    resp: reqwest::Response,
    peer_base_url: &str,
) -> Result<bytes::Bytes, anyhow::Error> {
    if resp
        .content_length()
        .is_some_and(|n| n > MAX_INFLATED_SNAPSHOT_BYTES)
    {
        anyhow::bail!(
            "peer {peer_base_url} declares a snapshot past the \
             {MAX_INFLATED_SNAPSHOT_BYTES}-byte ceiling; refusing to buffer it"
        );
    }
    let mut resp = resp;
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = resp.chunk().await? {
        if buf.len() as u64 + chunk.len() as u64 > MAX_INFLATED_SNAPSHOT_BYTES {
            anyhow::bail!(
                "peer {peer_base_url} streamed past the \
                 {MAX_INFLATED_SNAPSHOT_BYTES}-byte ceiling; refusing to buffer it"
            );
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.freeze())
}

/// Inflate (when gzipped) and parse one snapshot body. Blocking and CPU-bound —
/// callers run it off the async runtime.
fn decode_snapshot(body: &[u8], gzipped: bool) -> Result<PeerSnapshot, anyhow::Error> {
    if !gzipped {
        return Ok(serde_json::from_slice(body)?);
    }
    // Bounded on purpose. The producing endpoint is unauthenticated (see the
    // `kv_snapshot` route), gzip amplifies by up to ~1000x, and an unbounded
    // `read_to_end` would let one small response OOM a booting router — the most
    // silent failure available, since the process dies before it can log. The
    // ceiling sits far above a maxed-out fleet's tree so it only ever rejects a
    // body no legitimate peer would send.
    let mut inflated = Vec::new();
    let read = GzDecoder::new(body)
        .take(MAX_INFLATED_SNAPSHOT_BYTES + 1)
        .read_to_end(&mut inflated)?;
    if read as u64 > MAX_INFLATED_SNAPSHOT_BYTES {
        anyhow::bail!(
            "peer snapshot inflates past the {MAX_INFLATED_SNAPSHOT_BYTES}-byte ceiling; \
             refusing to buffer it"
        );
    }
    // Parse from the slice rather than streaming the decoder into serde: for one
    // large document `from_slice` can borrow string data straight out of the
    // buffer where `from_reader` must copy.
    Ok(serde_json::from_slice(&inflated)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A caller with no freshness requirement must send a bare path, so an older
    /// router image sees the request it has always seen.
    #[test]
    fn snapshot_url_omits_the_parameter_when_any_age_will_do() {
        assert_eq!(
            snapshot_url("http://peer:3000", None),
            format!("http://peer:3000{SNAPSHOT_PATH}"),
        );
        // Trailing slash on the base must not produce a doubled separator.
        assert_eq!(
            snapshot_url("http://peer:3000/", None),
            format!("http://peer:3000{SNAPSHOT_PATH}"),
        );
    }

    /// The splice probe asks for the cursor table alone. The parameter is what
    /// makes the producer skip its tree; without this the change ships a dead
    /// producer branch.
    #[test]
    fn cursors_url_asks_for_the_cursor_table_alone() {
        assert_eq!(
            cursors_url("http://peer:3000"),
            format!("http://peer:3000{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
        // Trailing slash on the base must not produce a doubled separator.
        assert_eq!(
            cursors_url("http://peer:3000/"),
            format!("http://peer:3000{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
        // An empty base yields the path-relative URI, for in-process routers.
        assert_eq!(
            cursors_url(""),
            format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
    }

    #[test]
    fn snapshot_url_states_the_requirement_in_milliseconds() {
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::from_millis(1500))),
            format!("http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=1500"),
        );
        // Zero is the strictest request, not an absent one.
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::ZERO)),
            format!("http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=0"),
        );
    }

    /// Saturate rather than wrap. A wrapped duration would read as a near-zero
    /// age and force the peer into a fleet-wide rebuild it was never asked for.
    #[test]
    fn snapshot_url_saturates_an_absurd_age() {
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::MAX)),
            format!(
                "http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}={}",
                u64::MAX
            ),
        );
    }

    /// An empty peer set is only conclusive if siblings were never seen. The set
    /// dips to empty transiently during slice repacks and rolling updates, and
    /// the bootstrap retry loop samples it many times per boot.
    #[test]
    fn transient_empty_peer_set_is_not_conclusive() {
        let r = PeerRegistry::new();
        assert!(!r.known_to_have_no_peers(), "unsynced is never conclusive");

        r.replace(vec![]);
        assert!(
            r.known_to_have_no_peers(),
            "synced + never any peers ⇒ genuinely alone",
        );

        r.replace(vec!["http://sibling:8090".into()]);
        r.replace(vec![]);
        assert!(
            !r.known_to_have_no_peers(),
            "once siblings have been seen, an empty set must be read as transient",
        );
    }

    /// Re-registering after a `forget` must mint a NEW incarnation, so a
    /// bootstrap task still in flight for the old one is recognisably stale.
    #[test]
    fn reregistration_after_forget_mints_a_new_epoch() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let first = t.register(std::slice::from_ref(&a));
        let e1 = first[0].1;

        // Re-registering a still-present rank keeps its incarnation.
        let again = t.register(std::slice::from_ref(&a));
        assert_eq!(again[0].1, e1, "a live rank keeps its incarnation");

        t.forget(std::slice::from_ref(&a));
        assert_eq!(t.epoch_of(&a), None);
        let second = t.register(std::slice::from_ref(&a));
        assert_ne!(
            second[0].1, e1,
            "remove + re-add must invalidate the previous incarnation",
        );
    }

    #[test]
    fn tracker_settles_when_all_ranks_terminal() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let b = KvWorkerId::new("http://a".into(), 1);
        t.register(&[a.clone(), b.clone()]);
        assert!(!t.settled());

        t.set(&a, BootstrapState::Recovered);
        assert!(!t.settled(), "one rank still pending");

        t.set(&b, BootstrapState::Failed);
        assert!(t.settled(), "Failed is terminal — cold is a valid outcome");
    }

    #[test]
    fn fetch_cap_round_trips_through_the_tracker() {
        let t =
            BootstrapTracker::new_with_fetch_cap(Duration::from_secs(120), Duration::from_secs(90));
        assert_eq!(t.timeout(), Duration::from_secs(120));
        assert_eq!(t.fetch_cap(), Duration::from_secs(90));
        assert_eq!(
            BootstrapTracker::new(Duration::from_secs(120)).fetch_cap(),
            DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP,
            "a tracker built without a cap takes the default",
        );
    }

    #[test]
    fn tracker_settles_on_deadline_with_pending_ranks() {
        let t = BootstrapTracker::new(Duration::ZERO);
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.settled(), "zero timeout settles immediately");
    }

    /// An empty tracker is not settled *yet* — it may still be waiting on
    /// workers to be discovered — but its deadline is already armed, so it
    /// cannot wait forever.
    #[test]
    fn tracker_with_no_ranks_is_not_settled_before_deadline() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        assert!(!t.settled());
        assert!(
            t.time_remaining().is_some(),
            "the deadline must be armed at construction, not at first register",
        );
    }

    /// Regression: a tracker that is NEVER registered must still settle.
    ///
    /// Reachable whenever `/readyz`'s worker registry is non-empty but no rank
    /// entered bootstrap — engines not publishing KV events, `/server_info`
    /// failures, a block-size mismatch. Arming the deadline lazily made that
    /// state permanent, bricking the pod and, because an unready replica leaves
    /// its own EndpointSlice, deadlocking the whole fleet.
    #[test]
    fn tracker_never_registered_still_settles() {
        let t = BootstrapTracker::new(Duration::ZERO);
        assert!(
            t.settled(),
            "an unregistered tracker must not hold /readyz at 503 forever",
        );
    }

    /// Registration keeps the deadline armed.
    ///
    /// `register` also re-arms it so the budget runs from first worker discovery
    /// rather than process start, but that is an optimisation measured in wall
    /// clock; asserting on it would mean comparing two `Instant::now()` samples,
    /// which is exactly the kind of timing-dependent test that turns flaky under
    /// CI load. The property that matters — armed, therefore cannot hang — is
    /// what is asserted.
    #[test]
    fn registration_keeps_the_deadline_armed() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.time_remaining().is_some());
        assert!(!t.settled(), "a pending rank must still gate readiness");
    }

    #[test]
    fn disabled_tracker_is_settled_immediately() {
        assert!(BootstrapTracker::disabled().settled());
    }

    #[test]
    fn tracker_forget_removes_state() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        t.register(std::slice::from_ref(&a));
        assert_eq!(t.state_of(&a), Some(BootstrapState::Pending));
        t.forget(std::slice::from_ref(&a));
        assert_eq!(t.state_of(&a), None);
    }

    #[test]
    fn peer_registry_shuffles_without_losing_entries() {
        let r = PeerRegistry::new();
        assert!(r.is_empty());
        let peers: Vec<String> = (0..16).map(|i| format!("http://r{i}")).collect();
        r.replace(peers.clone());
        assert_eq!(r.len(), 16);

        let mut got = r.candidates();
        assert_eq!(got.len(), 16);
        got.sort();
        let mut want = peers;
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn a_synced_empty_set_with_no_history_means_no_siblings() {
        let reg = PeerRegistry::new();
        reg.replace(vec![]);
        assert!(reg.synced());
        assert!(reg.known_to_have_no_peers());
    }

    /// The gate has exactly one way to die silently, and this is it.
    ///
    /// `settled()` latches on the construction-time deadline, and a `/metrics`
    /// scrape or a sibling's snapshot fetch reaches it before the first worker
    /// is discovered. On a fleet where that discovery takes longer than the
    /// budget, the latch fires over an EMPTY rank map — after which `/readyz`
    /// goes green as soon as one worker registers, while the first sweep is
    /// still in flight. Nothing errors; the feature is simply off.
    #[test]
    fn a_premature_settle_does_not_disarm_the_gate() {
        let t = BootstrapTracker::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(t.settled(), "the construction deadline has expired");

        let rank = KvWorkerId::new("http://w1".into(), 0);
        t.register(std::slice::from_ref(&rank));
        assert!(
            !t.settled(),
            "the first registration re-arms the budget, so a Pending rank must \
             hold readiness again",
        );

        t.set(&rank, BootstrapState::Recovered);
        assert!(
            t.settled(),
            "and it settles normally once the rank resolves"
        );
    }

    /// A replica that registers nothing at all — every worker publishes no KV
    /// events, say — must still settle, or the pod is bricked and the whole
    /// fleet deadlocks behind empty EndpointSlices.
    #[test]
    fn a_replica_that_never_registers_still_settles() {
        let t = BootstrapTracker::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(t.settled());
    }
}
