// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle bundle for the KV-event index.
//!
//! Couples the three submodules that are independent in their own right but
//! always operate together in production:
//!
//! - [`HashTree`] — the cache-aware routing index keyed by SGLang block hash.
//! - [`EngineLoadTable`] — engine-reported per-worker load.
//! - Two [`KvEventSubscriberRegistry`]s — one per `(worker_url, dp_rank)` on
//!   the cache topic, one on the load topic.
//! - A pump task that drains [`WorkerEvent`]s and applies KV batches to the
//!   tree and `Load` snapshots to the engine-load table.
//!
//! `add_worker` / `remove_worker` are driven from the worker manager on every
//! `DiscoveryEvent::Added` / `DiscoveryEvent::Removed`.
//!
//! # Race avoidance
//!
//! The pump runs independently of the lifecycle calls, so an event can sit in
//! the mpsc buffer while `remove_worker` is in progress. To prevent stale
//! events from re-inserting tree state for a worker that was just torn down,
//! [`KvEventIndex`] maintains a `live_workers` set; entries are removed
//! **before** the subscriber tasks are joined, and the pump filters every
//! event through this set before mutating the tree.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::block_size_oracle::BlockSizeOracle;
use super::bootstrap::{
    fetch_cursors, fetch_snapshot, BootstrapState, BootstrapTracker, PeerRegistry, PeerSnapshot,
    RankOutcome, SnapshotOutcome, VettedSnapshot, WireWorker, SNAPSHOT_FORMAT,
};
use super::discovery::{fetch_event_config, EventConfig};
use super::subscriber::{KvEventSubscriberRegistry, SubKind, WorkerEvent};
use super::tree::{HashTree, KvWorkerId};
use super::wire::{KvCacheEvent, KvEventBatch};
use crate::policies::engine_load::EngineLoadTable;

/// Channel buffer between the subscriber registry and the pump task.
///
/// Bounded so a misbehaving publisher cannot exhaust memory.  Realistic
/// per-worker event rates are < 1 kHz; a 1024-deep buffer absorbs a
/// half-second burst at 2 kHz before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

/// Per-rank cap on batches held back while that rank is
/// [`BootstrapState::Pending`].
///
/// A rank that overflows this cannot be spliced (see `drain_pending`), so the
/// cap trades a small amount of memory for the chance to bootstrap at all.
/// Sized to match `EVENT_CHANNEL_BUFFER`: if the pump is that far behind, the
/// snapshot is not arriving in time anyway.
const PENDING_BATCH_LIMIT: usize = 1024;

/// Delay between peer-sweep attempts while no usable peer has been found.
///
/// Short relative to the bootstrap deadline that bounds the whole sweep, so a
/// peer becoming available is picked up promptly.
const PEER_RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// Depth of the obligation queue feeding the coordinator.
///
/// Sized well past the per-fleet worker count so `add_worker` never blocks on
/// the coordinator; the fallback for a full queue is a sweep of its own, which is
/// correct but re-downloads a body the in-flight sweep is already fetching.
const BOOTSTRAP_QUEUE_DEPTH: usize = 1024;

/// Passes to skip a peer after a retriable failure.
///
/// `NothingUsable` / `Unreachable` / no-coverage are all worth retrying — the peer
/// may discover our workers, or come back — but retrying on every 250ms pass means
/// re-downloading a multi-megabyte tree from a replica that is itself serving
/// traffic, up to ~120 times per booting replica. A few passes of cooldown keeps
/// the retry useful without the load.
const PEER_COOLDOWN_PASSES: u32 = 4;

/// How many snapshot fetches the per-request timeout should leave room for
/// within one bootstrap deadline.
///
/// A single fetch must not be allowed to consume the whole deadline. The sweep
/// walks candidates sequentially and awaits each fetch, so a per-request timeout
/// equal to the deadline means the first unresponsive peer starves every other
/// candidate — the outer deadline then cancels the sweep mid-fetch, so the
/// replica boots cold having tallied no peer outcome at all while the terminal
/// log still reports the full candidate list. Budgeting several attempts per
/// deadline is what makes the "try the next peer" loop reachable under a slow
/// producer.
const SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE: u32 = 4;

/// Preferred floor for the per-fetch timeout: the snapshot body is a
/// multi-megabyte tree, so too short a timeout would reject every peer instead.
///
/// Preferred, not absolute — see [`snapshot_fetch_timeout`], which lets the
/// deadline override it. An unconditional floor would hand a single fetch the
/// entire budget at the default deadline, reintroducing the starvation above.
///
/// The CLI's `--kv-bootstrap-fetch-timeout-cap-ms` minimum mirrors this
/// value; keep them in agreement if it moves.
const SNAPSHOT_FETCH_TIMEOUT_FLOOR: Duration = Duration::from_secs(5);

/// Per-request timeout for a peer snapshot fetch, a strict fraction of the
/// configured bootstrap budget so several peers can be tried within it.
///
/// `cap` (from `--kv-bootstrap-fetch-timeout-cap-ms`, default
/// [`super::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP`]) bounds the
/// derivation, so a deliberately generous deadline (see
/// `MAX_KV_BOOTSTRAP_TIMEOUT_MS`) still cannot let one hung peer park for
/// minutes; raise it when the snapshot body outgrows the default budget.
///
/// Derived from the CONFIGURED budget, not from the time a given sweep has left:
/// a later sweep running on a shrunken remainder still carries this value, so it
/// bounds the search less tightly than the first sweep. Recomputing per sweep
/// would need the client rebuilt per sweep, which costs a connection pool.
pub(crate) fn snapshot_fetch_timeout(deadline: Duration, cap: Duration) -> Duration {
    // A pre-settled (bootstrap-disabled) tracker reports a zero deadline. The
    // client is still built, and a zero reqwest timeout means "expire
    // immediately" rather than "no timeout", so fall back to the floor.
    if deadline.is_zero() {
        return SNAPSHOT_FETCH_TIMEOUT_FLOOR;
    }
    // The floor yields to the deadline rather than overriding it. The default
    // budget equals the floor, so an unconditional floor would make every
    // default-configured router spend its whole deadline on one peer. It also
    // yields to the cap, so a below-floor cap degrades to itself instead of
    // panicking `clamp`.
    let floor = SNAPSHOT_FETCH_TIMEOUT_FLOOR.min(deadline / 2).min(cap);
    (deadline / SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE).clamp(floor, cap)
}

/// How long a grafted rank may wait for its own live stream to prove the splice
/// before the fleet is asked instead.
///
/// A snapshot can be grafted before the rank's first live batch arrives, so the
/// watermark check is deferred to that batch. Nothing guarantees a batch ever
/// comes: an idle rank would otherwise serve grafted state forever with its
/// continuity unproven, and a `BlockRemoved` lost in the subscribe window would
/// survive as a permanent false cache hit that no later event corrects.
///
/// On expiry the rank is NOT discarded — see `spawn_splice_probe` for why
/// silence is not evidence — it is probed against the fleet's own cursors.
const SPLICE_PROOF_TIMEOUT: Duration = Duration::from_secs(30);

/// Consecutive unanswerable probes after which an unproven graft is kept and
/// tallied [`RankOutcome::WarmUnwitnessed`].
///
/// Without a stop the pump would re-probe an idle rank forever whenever the
/// fleet is unreachable — a single-replica deployment being the obvious case —
/// and the rank's verdict would never resolve in the metrics. Keeping rather
/// than discarding follows the same reasoning as the probe itself.
const MAX_UNKNOWN_PROBES: u32 = 3;

/// How often the pump checks for splice proofs that never arrived. Coarse: the
/// deadline it enforces is [`SPLICE_PROOF_TIMEOUT`], not this.
const SPLICE_PROOF_SWEEP_INTERVAL: Duration = Duration::from_secs(5);

/// A grafted rank whose continuity with the live stream is not yet proven.
///
/// Carries when the wait started so the wait can be bounded; see
/// [`SPLICE_PROOF_TIMEOUT`].
#[derive(Debug, Clone, Copy)]
struct PendingProof {
    /// Watermark the first arriving batch must not exceed by more than one.
    watermark: i64,
    /// When the graft happened, or when the last probe was launched.
    since: Instant,
    /// Consecutive probes that found no witness; see [`MAX_UNKNOWN_PROBES`].
    unknown_probes: u32,
    /// When the outstanding probe was launched, if one is.
    ///
    /// A fresh outstanding probe blocks relaunch; past the timeout it is
    /// presumed lost — its verdict may never land — and asked again. Strictly
    /// better than a bare bool latch, on which any lost verdict (a panicked
    /// probe task, say) would freeze the rank in Recovered forever with no
    /// [`RankOutcome`] recorded. Honest accounting, verified against the code
    /// in review: because `launch_probe` arms both clocks from one instant,
    /// the presumed-lost cadence today equals what `since` alone would give —
    /// the second field's payoff is making "never probed" and "verdict lost"
    /// distinguishable states, not a different retry schedule. Cleared, as
    /// `None`, by whichever verdict arrives.
    probe_launched: Option<Instant>,
}

impl PendingProof {
    /// Whether the sweep should launch a probe for this rank now: never while
    /// one is outstanding and fresh; yes once the last probe — or the graft
    /// itself, if none has run yet — is older than `timeout`. The pump's sweep
    /// filter uses this single definition so the guard cannot drift from its
    /// tests.
    fn due_for_probe(&self, timeout: Duration) -> bool {
        match self.probe_launched {
            Some(launched) => launched.elapsed() >= timeout,
            None => self.since.elapsed() >= timeout,
        }
    }

    /// Arm both clocks from one instant: re-start the retry spacing and mark
    /// the new probe outstanding. One method so the two writes cannot drift
    /// apart the way two statements at the call site gradually could.
    fn launch_probe(&mut self, now: Instant) {
        self.since = now;
        self.probe_launched = Some(now);
    }
}

/// Control-plane messages for the pump task.
///
/// Tree mutation MUST stay on the single writer (see the single-writer property
/// in [`super::tree`]), so the bootstrap task never touches the tree itself —
/// it fetches and vets a snapshot, then hands it to the pump through this
/// channel.
#[derive(Debug)]
enum PumpControl {
    /// Graft a vetted snapshot, seed cursors, then release each rank's held
    /// batches.
    ///
    /// `obligations` is the set this message discharges: every rank named here
    /// leaves `Pending` when this is handled, whether or not the snapshot covered
    /// it. Deriving the set from the snapshot instead would leave a rank the peer
    /// never mentioned buffering forever.
    ///
    /// Each entry carries the incarnation it was registered under, so a task
    /// still in flight for a worker that has since been removed and re-added
    /// cannot graft onto the new incarnation.
    ApplySnapshot {
        obligations: Vec<(KvWorkerId, u64)>,
        vetted: Box<VettedSnapshot>,
    },
    /// Stop holding batches for `ranks`: release what is buffered and mark
    /// them [`BootstrapState::Failed`]. Sent when no peer could supply a
    /// snapshot, or when the bootstrap deadline fires.
    AbandonBootstrap { obligations: Vec<(KvWorkerId, u64)> },
    /// A worker was removed: drop any pump-local state still keyed by these
    /// ranks. `remove_worker` runs on the worker-manager task and cannot touch
    /// the pump's `held` / `awaiting_splice_proof` maps itself, so without this
    /// they leak, and a re-added worker's fresh publisher would inherit the old
    /// incarnation's queue and watermark.
    ForgetRanks { ranks: Vec<KvWorkerId> },
    /// Result of asking the fleet whether a rank's publisher moved past the
    /// watermark of a snapshot whose splice was never proven locally.
    ///
    /// The probe runs off-pump because it does network I/O; the verdict comes
    /// back here so the tree write stays on the single writer.
    SpliceProbe {
        rank: KvWorkerId,
        epoch: u64,
        verdict: SpliceVerdict,
    },
}

/// What the fleet says about a publisher's progress past an unproven watermark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpliceVerdict {
    /// Some peer has applied a sequence ABOVE our watermark. Sequence numbers
    /// come from the publisher, so that batch was emitted — and we never saw it,
    /// which is exactly the hole the splice check exists to catch.
    Advanced,
    /// Peers answered and none is past the watermark, so there is nothing we
    /// could have missed: the grafted state is continuous with a stream that has
    /// simply been silent.
    NoAdvance,
    /// Nobody answered, so the question stays open and the wait is re-armed.
    Unknown,
}

/// Per-worker bookkeeping kept inside [`KvEventIndex`] so `remove_worker`
/// knows which DP ranks were actually subscribed (not the advertised
/// `dp_size`, which may overflow `u16` and skip ranks).
#[derive(Debug, Clone)]
struct WorkerEntry {
    /// DP ranks that were successfully spawned for this worker. Used by
    /// `remove_worker` to know which `(url, dp_rank)` cursors and tree
    /// states to clear.
    dp_ranks: Vec<u32>,
}

/// Bundle of `HashTree` + `KvEventSubscriberRegistry` + pump task.
///
/// Construct one instance per router process and hand it to the worker
/// manager as `Option<Arc<KvEventIndex>>` — `None` disables the cache-aware
/// routing path entirely.
pub struct KvEventIndex {
    tree: Arc<HashTree>,
    subscribers: Arc<KvEventSubscriberRegistry>,
    /// Second registry subscribing to the load topic (one per worker rank),
    /// feeding `LoadStat` snapshots into `engine_load`. Shares the pump
    /// channel with `subscribers`; keyed independently so KV and load
    /// subscribers for the same worker don't collide.
    load_subscribers: Arc<KvEventSubscriberRegistry>,
    /// Engine-reported per-worker load, written by the pump from
    /// `WorkerEvent::Load` and read by the cache-aware-zmq policy.
    engine_load: Arc<EngineLoadTable>,
    pump: Mutex<Option<JoinHandle<()>>>,
    pump_cancel: CancellationToken,
    workers: Mutex<HashMap<String, WorkerEntry>>,
    http: reqwest::Client,
    /// Set of currently-attached `(worker_url, dp_rank)` pairs. The pump
    /// drops any event whose `worker` is not in this set, so a batch
    /// queued by a subscriber that was torn down by `remove_worker` does
    /// not re-pollute the tree after `clear_worker` ran.
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    /// Per-`(worker_url, dp_rank)` last-applied sequence number. The
    /// subscriber forwards every batch with no de-dup; this map filters
    /// any batch whose `seq` is not strictly greater than the previously
    /// applied one. Cleared on `remove_worker` because a re-added worker
    /// may legitimately have a fresh publisher whose sequence numbers
    /// restart from 1.
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    /// Worker-sourced publish block size (`page_size * dcp_size`), shared
    /// with the cache-aware-zmq policy.
    /// `add_worker` calls `try_set(cfg.block_size)` so the first worker
    /// establishes the value; subsequent workers that disagree are
    /// rejected (logged + not subscribed). The policy reads it at routing
    /// time to size its `compute_block_hashes` call.
    block_size_oracle: Arc<BlockSizeOracle>,
    /// Per-rank bootstrap progress; also what `/readyz` consults.
    bootstrap: Arc<BootstrapTracker>,
    /// Sibling replicas a snapshot may be pulled from. Empty disables peer
    /// bootstrap without any other configuration.
    peers: Arc<PeerRegistry>,
    /// Control channel into the pump, so snapshot grafting happens on the
    /// single writer rather than in the bootstrap task.
    ctrl_tx: mpsc::Sender<PumpControl>,
    /// Most recently built snapshot and when it was built. Async mutex because
    /// it gates the build, and a waiter must yield its worker — see
    /// [`KvEventIndex::peer_snapshot_body`].
    snapshot_cache: AsyncMutex<Option<CachedSnapshot>>,
    /// Separate client for snapshot fetches.
    ///
    /// WHY not `http`: that client carries a 2s TOTAL-request timeout, which is
    /// right for `/server_info` introspection and far too short for a
    /// multi-megabyte tree body. Sharing it would make peer bootstrap silently
    /// fail for exactly the large trees worth bootstrapping, and report the
    /// failure as `unreachable`.
    ///
    /// Its timeout is a strict fraction of the bootstrap deadline, not the
    /// deadline itself — see [`snapshot_fetch_timeout`]. The sweep bounding the
    /// whole search is not enough on its own: the sweep awaits candidates in
    /// turn, so only a shorter per-request bound lets it reach a second one.
    snapshot_http: reqwest::Client,
    /// Obligations waiting for the coordinator to fold them into the sweep that
    /// is in flight, or to start one. See [`bootstrap_coordinator`].
    bootstrap_tx: mpsc::Sender<ObligationBatch>,
}

/// A built snapshot together with its already-encoded JSON body.
///
/// WHY the body is cached and not just the struct: the producer serves this to
/// every booting sibling, and `serde_json` on a multi-megabyte tree costs about
/// as much as the tree walk that produced it. Caching only the struct amortises
/// the walk across the TTL but re-encodes per request, so a boot herd pays the
/// encode N times for one identical document.
struct CachedSnapshot {
    /// When this snapshot's CONTENTS were sampled — the instant before the
    /// cursors were read — not when the build finished.
    ///
    /// The whole freshness contract hangs on this being the earlier of the two.
    /// A consumer asking for "no older than N" is asking what the snapshot
    /// covers, and it covers the publisher's stream as of the cursor read; a
    /// walk plus encode of a multi-megabyte tree takes long enough that stamping
    /// completion would claim coverage the document does not have, which is
    /// exactly the gap this parameter exists to close.
    ///
    /// One residual, unavoidable at this layer: a cursor reflects what this
    /// replica had APPLIED, so the stamp still overstates by the producer's own
    /// receive-to-apply latency. That is sub-millisecond against a window that
    /// used to be seconds.
    exported_at: Instant,
    snap: Arc<PeerSnapshot>,
    /// `Bytes` so handing it to a response body is a refcount bump rather than
    /// a copy of the whole tree.
    body: Bytes,
}

/// One batch of obligations handed to the coordinator.
///
/// Carries `holding_since` because the coordinator cannot otherwise know how
/// fresh a snapshot these ranks need: see [`PendingSweep::freshness_floor`].
struct ObligationBatch {
    obligations: Vec<(KvWorkerId, u64)>,
    /// When these ranks began holding batches — i.e. the instant after which a
    /// peer's export must have been taken for the graft to splice.
    holding_since: Instant,
    late_join: LateJoin,
}

/// What a batch accepts when it arrives while a sweep is already in flight.
///
/// The sweep asked for freshness on behalf of the ranks it started with, so a
/// batch that arrives afterwards may be delivered against an export predating
/// its own `holding_since` — which the pump then resolves [`RankOutcome::Gap`].
/// Whether that is acceptable depends on what the batch has left to spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LateJoin {
    /// Ride the in-flight sweep's snapshot regardless.
    ///
    /// For discovery this is the right trade and the reason the coordinator
    /// exists: a fleet's worth of workers lands DURING the first fetch by
    /// design, and making them wait for a sweep of their own would cost a second
    /// fleet-wide fetch on every boot. A rank that gaps this way still has its
    /// one retry.
    Permitted,
    /// Wait for a sweep that asks on this batch's behalf.
    ///
    /// For a gap retry, which is the only [`Refused`](LateJoin::Refused) caller,
    /// riding along is equivalent to dropping it: the retry exists because the
    /// last export did not splice, `gap_retried` caps it at one, and a snapshot
    /// taken before the rank resumed holding re-gaps by construction. Deferring
    /// costs one loop iteration, since the coordinator re-enters `take_pending`
    /// as soon as it has delivered.
    Refused,
}

/// Obligations taken off the queue for one sweep, with the freshness every one
/// of them requires.
struct PendingSweep {
    obligations: Vec<(KvWorkerId, u64)>,
    /// The NEWEST `holding_since` among the absorbed batches, because the
    /// sweep's one snapshot has to satisfy every rank in it: a rank that began
    /// holding at T needs an export after T, so the strictest requirement wins.
    ///
    /// An `Instant` rather than a max-age duration on purpose. The sweep retries
    /// for as long as the deadline allows, and each attempt re-derives
    /// `elapsed()` from this — so the age it asks for grows while the condition
    /// it encodes (`exported_at > floor`) stays exactly the same.
    freshness_floor: Instant,
    /// Batches that arrived too late for this sweep to have asked on their
    /// behalf and refused to ride along. Handed back for the next one.
    deferred: Vec<ObligationBatch>,
}

impl PendingSweep {
    /// Fold a batch in before the sweep starts, tightening the floor it will ask
    /// with.
    fn absorb(&mut self, batch: ObligationBatch) {
        self.freshness_floor = self.freshness_floor.max(batch.holding_since);
        self.obligations.extend(batch.obligations);
    }

    /// Fold a batch in after the floor has been frozen — i.e. once the sweep is
    /// running and its request has already stated what it wants.
    ///
    /// Deliberately does NOT tighten the floor: the fetch has already gone out
    /// under the old one, so raising it here would describe a guarantee this
    /// sweep never asked for.
    fn admit_late(&mut self, batch: ObligationBatch) {
        if batch.late_join == LateJoin::Refused && batch.holding_since > self.freshness_floor {
            self.deferred.push(batch);
        } else {
            self.obligations.extend(batch.obligations);
        }
    }
}

impl From<ObligationBatch> for PendingSweep {
    fn from(batch: ObligationBatch) -> Self {
        Self {
            obligations: batch.obligations,
            freshness_floor: batch.holding_since,
            deferred: Vec::new(),
        }
    }
}

/// Serialise bootstrap into one sweep at a time, and let every rank pending when
/// that sweep lands share its snapshot.
///
/// # Why this exists
///
/// A peer snapshot is fleet-wide: one body carries the whole worker table, every
/// cursor, and the whole tree, so a single fetch already contains everything
/// every pending rank needs. `add_worker` fires once per discovered engine, so
/// sweeping from there re-downloads that same document once per engine. The
/// document grows with the fleet while the fetch count grows with it too — on a
/// 168-engine fleet that measured as ~144 fetches of a 35 MB body per booting
/// replica, most of which timed out and left their ranks cold.
///
/// # Why the fetch is not delayed to collect a batch first
///
/// It does not need to be: the sweep is the collection window. The fetch of a
/// multi-megabyte body takes far longer than an EndpointSlice watch event takes
/// to deliver a fleet, so ranks discovered while it is in flight are merged into
/// the same delivery and ride the same snapshot — at no latency cost.
///
/// # Why that is safe
///
/// The invariant is a SEQUENCE condition, not a wall-clock one: a graft is sound
/// when the rank's live stream resumes at `peer_cursor + 1`. Vetting happens
/// after the body arrives, against the live-worker set at that moment, so a
/// worker discovered during the fetch is already live and already covered by the
/// snapshot. And the watermark check adjudicates every rank independently, so a
/// rank whose publisher did advance in between is discarded to
/// [`RankOutcome::Gap`] and runs cold — never spliced over a hole.
///
/// One narrow window remains by construction: a rank whose obligation arrives
/// after vetting but before the merge is delivered against a snapshot that
/// dropped it as unknown, which the pump reports as
/// [`RankOutcome::Uncovered`]. Also safe, also cold, and bounded by the gap
/// between those two instants.
///
/// # Freshness, and who a sweep speaks for
///
/// Sharing one snapshot only helps if that snapshot is fresh enough for the
/// ranks sharing it, and the ranks in a batch began holding at different
/// moments. [`PendingSweep::freshness_floor`] keeps the strictest of them and
/// every fetch attempt states it, so a peer serving from cache rebuilds rather
/// than handing back an export older than some rank in the batch.
///
/// That guarantee covers the ranks the sweep STARTED with. A rank merged in
/// afterwards was not represented in the request, so it may still be delivered
/// against an export predating it — safe (the watermark check discards it to
/// [`RankOutcome::Gap`]) but wasted. Discovery accepts that trade;
/// [`LateJoin::Refused`] batches do not and are deferred to the next sweep.
async fn bootstrap_coordinator(
    mut rx: mpsc::Receiver<ObligationBatch>,
    index: std::sync::Weak<KvEventIndex>,
    cancel: CancellationToken,
) {
    // Ranks already given a second sweep. A rank no peer covers must not buy a
    // fresh fleet-wide fetch on every pass — that is the amplification this
    // coordinator exists to remove — so each gets at most one retry.
    let mut requeued: HashSet<KvWorkerId> = HashSet::new();
    loop {
        // Block for the first obligation, then take whatever else is already
        // queued. No waiting: the fetch itself is the batching window.
        let Some(mut pending) = take_pending(&mut rx, &cancel).await else {
            return;
        };

        // Obligations already taken from the channel are dropped without a
        // `PumpControl` if this fails, which would normally strand their ranks in
        // `Pending`. Only reachable once the last `Arc<KvEventIndex>` is gone —
        // i.e. teardown, where the pump that would have received the message is
        // gone too and nothing is left to keep ready.
        let deps = {
            let Some(index) = index.upgrade() else { return };
            index.bootstrap_deps()
        };
        let deadline = deps.deadline();
        let ranks: Vec<KvWorkerId> = pending.obligations.iter().map(|(r, _)| r.clone()).collect();

        // ONE sweep, awaited here rather than spawned — that is what makes this
        // single-flight. Obligations discovered while it runs pile up in the
        // channel and are merged below, so they ride this same snapshot.
        let started = Instant::now();
        let result = sweep_until_deadline(&deps, &ranks, deadline, pending.freshness_floor).await;
        let joined = drain_ready(&mut rx, &mut pending);
        if joined > 0 || !pending.deferred.is_empty() {
            debug!(
                joined,
                deferred = pending.deferred.len(),
                total = pending.obligations.len(),
                sweep_ms = started.elapsed().as_millis(),
                "kv-bootstrap: ranks arriving during the sweep joined its snapshot, \
                 minus any that need a fresher one",
            );
        }

        // The sweep's coverage check (`covers_any`) only spoke for the ranks it
        // was given, so a rank merged afterwards may be absent from the accepted
        // peer's tree — the peer can itself be partially discovered and know one
        // of our workers but not another. Delivering such a rank into this
        // snapshot resolves it `Uncovered`, which is TERMINAL; before this
        // coordinator existed its own sweep would have kept looking for a peer
        // that did cover it. Give it that second look instead.
        if let SweepResult::Found(ref vetted) = result {
            let mut retry = Vec::new();
            let mut deliver = Vec::with_capacity(pending.obligations.len());
            for ob in pending.obligations.drain(..) {
                let covered = vetted.covers_any(std::slice::from_ref(&ob.0));
                if covered || !requeued.insert(ob.0.clone()) {
                    deliver.push(ob);
                } else {
                    retry.push(ob);
                }
            }
            pending.obligations = deliver;
            if !retry.is_empty() {
                // Carry the floor forward rather than re-stamping to now: these
                // ranks have been holding since it, so demanding an export newer
                // than they need would force the peer into avoidable rebuilds.
                pending.deferred.push(ObligationBatch {
                    obligations: retry,
                    holding_since: pending.freshness_floor,
                    late_join: LateJoin::Permitted,
                });
            }
        }

        for batch in std::mem::take(&mut pending.deferred) {
            // Re-queue via the index rather than a sender this task owns: a
            // long-lived clone here would hold the channel open forever and kill
            // the all-senders-dropped exit.
            match index.upgrade() {
                // A full queue must not drop the batch — an obligation nothing
                // owns strands its ranks in `Pending`, which holds `/readyz` and
                // eventually overflows the pump's hold-back. Sweeping un-batched
                // costs a fetch and keeps the freshness these ranks asked for,
                // where folding them into this delivery would spend it.
                Some(idx) => {
                    if let Err(e) = idx.bootstrap_tx.try_send(batch) {
                        warn!("kv-bootstrap: batch queue unavailable ({e}); sweeping un-batched");
                        idx.spawn_bootstrap(e.into_inner());
                    }
                }
                // Teardown: nothing is left to run another sweep, so deliver them
                // here rather than stranding them.
                None => pending.obligations.extend(batch.obligations),
            }
        }

        deliver_bootstrap(&deps, pending.obligations, result, deadline).await;
    }
}

/// Block for the first obligation, then take everything already queued behind it
/// without waiting.
///
/// `None` means stop — cancelled, or every sender dropped with nothing pending.
async fn take_pending(
    rx: &mut mpsc::Receiver<ObligationBatch>,
    cancel: &CancellationToken,
) -> Option<PendingSweep> {
    // Block until there is something to do, so an idle fleet costs nothing.
    let first = tokio::select! {
        _ = cancel.cancelled() => return None,
        first = rx.recv() => first?,
    };
    let mut pending = PendingSweep::from(first);
    // Pre-sweep, so these tighten the floor the sweep will ask with rather than
    // arriving after it has already asked.
    while let Ok(more) = rx.try_recv() {
        pending.absorb(more);
    }
    Some(pending)
}

/// Move every immediately-available obligation into `into`, returning how many
/// ranks were added. Never waits.
///
/// Called once the sweep has run, so batches are admitted under the frozen floor
/// — see [`PendingSweep::admit_late`]. Ranks it defers are not counted as joined.
fn drain_ready(rx: &mut mpsc::Receiver<ObligationBatch>, into: &mut PendingSweep) -> usize {
    let before = into.obligations.len();
    while let Ok(more) = rx.try_recv() {
        into.admit_late(more);
    }
    into.obligations.len() - before
}

/// Encode a snapshot for the wire.
///
/// `PeerSnapshot` is a plain `Serialize` struct with no non-string map keys, so
/// this cannot actually fail; an empty body on the impossible branch keeps the
/// caller total, and a consumer reads it as an unusable peer and moves on.
fn encode_snapshot(snap: &PeerSnapshot) -> Bytes {
    match serde_json::to_vec(snap) {
        Ok(v) => Bytes::from(v),
        Err(e) => {
            warn!(error = %e, "kv-bootstrap: snapshot serialisation failed");
            Bytes::new()
        }
    }
}

impl std::fmt::Debug for KvEventIndex {
    /// Terse by design: the interesting state (tree, cursors) is large and
    /// lock-guarded, and this exists only so [`crate::server::app_context`]
    /// can keep its `Debug` derive.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvEventIndex")
            .field("workers", &self.workers.lock().len())
            .field("tree_nodes", &self.tree.node_count())
            .field("peers", &self.peers.len())
            .finish_non_exhaustive()
    }
}

impl KvEventIndex {
    /// Build an empty index and spawn the pump task. Peer bootstrap is
    /// permanently disabled on this constructor (it uses a pre-settled tracker);
    /// use [`KvEventIndex::new_with_bootstrap`] to enable it.
    pub fn new() -> Arc<Self> {
        Self::new_with_http(
            reqwest::Client::builder()
                .timeout(Duration::from_secs(2))
                .build()
                .expect("default http client builds"),
        )
    }

    /// Constructor used by tests so they can supply a custom timeout.
    pub fn new_with_http(http: reqwest::Client) -> Arc<Self> {
        Self::new_with_http_and_oracle(http, BlockSizeOracle::new())
    }

    /// Constructor that lets the caller supply a pre-shared
    /// [`BlockSizeOracle`]. Production wires this from `AppContext` so
    /// the same oracle the index seeds is the one the cache-aware-zmq
    /// policy reads at routing time. Tests use this to pre-populate the
    /// oracle and exercise the mismatch-rejection path.
    pub fn new_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::build(
            http,
            block_size_oracle,
            Arc::new(BootstrapTracker::disabled()),
        )
    }

    /// Constructor that enables peer bootstrap with the supplied tracker.
    /// Production wires a tracker built from `--kv-bootstrap-timeout-ms`; the
    /// other constructors use a pre-settled tracker so existing callers keep
    /// today's apply-immediately behaviour.
    pub fn new_with_bootstrap(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        bootstrap: Arc<BootstrapTracker>,
    ) -> Arc<Self> {
        Self::build(http, block_size_oracle, bootstrap)
    }

    fn build(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        bootstrap: Arc<BootstrapTracker>,
    ) -> Arc<Self> {
        // A fraction of the deadline, not the deadline itself: see
        // `snapshot_fetch_timeout`. The sweep bounds the whole search, but only a
        // shorter per-request bound lets it reach a second candidate.
        let snapshot_http = reqwest::Client::builder()
            .timeout(snapshot_fetch_timeout(
                bootstrap.timeout(),
                bootstrap.fetch_cap(),
            ))
            .build()
            .unwrap_or_else(|_| http.clone());
        let tree = Arc::new(HashTree::new());
        let (tx, rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let (ctrl_tx, ctrl_rx) = mpsc::channel::<PumpControl>(16);
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(tx.clone()));
        let load_subscribers = Arc::new(KvEventSubscriberRegistry::with_kind(tx, SubKind::Load));
        let engine_load = EngineLoadTable::new();
        let cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>> = Arc::new(Mutex::new(HashMap::new()));
        let live_workers: Arc<Mutex<HashSet<KvWorkerId>>> = Arc::new(Mutex::new(HashSet::new()));
        let pump_cancel = CancellationToken::new();
        let peers = Arc::new(PeerRegistry::new());
        let (bootstrap_tx, bootstrap_rx) = mpsc::channel(BOOTSTRAP_QUEUE_DEPTH);
        let pump = tokio::spawn(pump_loop(
            PumpDeps {
                tree: tree.clone(),
                engine_load: engine_load.clone(),
                cursors: cursors.clone(),
                live_workers: live_workers.clone(),
                bootstrap: bootstrap.clone(),
                oracle: Arc::clone(&block_size_oracle),
                http: http.clone(),
                peers: Arc::clone(&peers),
                snapshot_http: snapshot_http.clone(),
                bootstrap_tx: bootstrap_tx.clone(),
                ctrl_tx: ctrl_tx.clone(),
            },
            pump_cancel.clone(),
            rx,
            ctrl_rx,
        ));
        let index = Arc::new(Self {
            tree,
            subscribers,
            load_subscribers,
            engine_load,
            pump: Mutex::new(Some(pump)),
            pump_cancel: pump_cancel.clone(),
            workers: Mutex::new(HashMap::new()),
            http,
            live_workers,
            cursors,
            block_size_oracle,
            bootstrap,
            peers,
            ctrl_tx,
            snapshot_cache: AsyncMutex::new(None),
            snapshot_http,
            bootstrap_tx,
        });
        // Gated on the SAME predicate as registration (`peer_bootstrap_enabled`),
        // not on `settled()`. If the two ever disagree, obligations get queued
        // with nothing draining them: their ranks stay `Pending` and hold batches
        // until the pump's per-rank cap overflows. An enabled tracker that is
        // already settled at construction — a zero timeout — is exactly that case.
        if index.bootstrap.enabled() {
            tokio::spawn(bootstrap_coordinator(
                bootstrap_rx,
                Arc::downgrade(&index),
                pump_cancel,
            ));
        }
        index
    }

    /// Build (or reuse) this replica's snapshot for a peer to bootstrap from.
    ///
    /// # The caller states how fresh it needs
    ///
    /// `max_age` is the oldest export this requester can use — see
    /// [`PRODUCER_CACHE_TTL`] for why a fixed TTL cannot answer this and the
    /// consumer can. A cached entry is reused only if it meets the requirement.
    ///
    /// # Single-flight, without spending threads on it
    ///
    /// The cache lock is held across the build on purpose: a simultaneous
    /// scale-up has every new replica asking at once, and serialising the
    /// builders means the fleet pays one walk per generation of requesters
    /// rather than one per requester. Waiters re-check against their OWN
    /// `max_age` after acquiring, so a build that started after a waiter began
    /// holding satisfies it and it returns without a second walk; one that
    /// started before does not, and that waiter builds. The herd therefore costs
    /// a walk per build-duration of arrival spread, not a walk per member.
    ///
    /// Both halves of that serialisation must yield, because this runs on the
    /// runtime that also proxies requests, and the requesters are a boot herd
    /// re-sweeping several times a second:
    ///
    /// * the gate is an async mutex, so waiters park the TASK rather than the
    ///   worker thread — a blocking mutex here lets N concurrent fetches pin N
    ///   workers doing nothing, and the walk is exactly when they pile up;
    /// * the walk itself is CPU-bound with no await point, so it goes to the
    ///   blocking pool rather than occupying a worker for its whole duration.
    ///
    /// Shard read locks are taken one at a time inside the walk, so routing can
    /// still match against the other shards throughout.
    /// Returns the snapshot already encoded, which is the only form anything
    /// needs: the HTTP route is the sole consumer.
    ///
    /// There is deliberately no struct-returning twin. `snapshot_entry` encodes
    /// as part of filling the cache, so an accessor that handed back only the
    /// `Arc<PeerSnapshot>` would still pay a multi-megabyte `serde_json` pass and
    /// a blocking-pool round trip, then discard the result — a trap for the next
    /// caller who just wants to read a field.
    pub async fn peer_snapshot_body(&self, max_age: Duration) -> Bytes {
        self.snapshot_entry(max_age).await.1
    }

    /// Build this replica's cursor table for a peer's splice probe.
    ///
    /// Deliberately NOT a smaller `peer_snapshot_body`. A probe asks one
    /// question — "is your cursor for this rank above my watermark?" — and the
    /// cursors live in their own map, which `snapshot_entry` already reads
    /// independently of, and before, the tree walk. Serving them therefore costs
    /// a read of that map, where answering the same question from a full export
    /// costs a walk, a multi-megabyte serialise, and a compress.
    ///
    /// Every observed rank is reported, including ones whose blocks this
    /// replica no longer holds (`BlockRemoved`-only history, `AllBlocksCleared`,
    /// eviction). The full export's cursor table is narrower — it is filtered
    /// to ranks that still carry a node in the tree, because a graft recipient
    /// needs a block source, not a witness. The witness question does not have
    /// that constraint: having SEEN the publisher's stream at seq N is evidence
    /// even after the blocks themselves are gone. So this table is a superset
    /// of the full export's, and callers comparing the two bodies must not
    /// expect set equality. (One consequence: witnesses the full export's
    /// table omits still answer here, so proving splices this way reports
    /// marginally more `RankOutcome::Gap`.)
    ///
    /// Takes no `max_age` and touches no cache: the cursors are read live, so the
    /// answer is strictly fresher than any freshness a caller could request.
    /// Reading live also removes the accuracy cost the cached path carried, where
    /// a cursor stale by up to the producer TTL could miss advancement that had
    /// just happened and report a splice as continuous when it was not. The body
    /// is one small entry per rank — nothing like the tree export the producer
    /// cache exists for — and the endpoint serves the internal fleet, so the
    /// uncached live read is the deliberate design rather than a missing
    /// throttle.
    ///
    /// `nodes` is always empty, which makes this body ungraftable by
    /// construction rather than by discipline: `VettedSnapshot::from_wire`
    /// rejects an empty node list as `VetError::ProducerCold`, so a bootstrap
    /// fetch can never be silently satisfied by a cursors-only answer.
    pub fn peer_cursors_body(&self) -> Bytes {
        let hash_config = self.block_size_oracle.hash_config();
        let (block_size, is_bigram) = hash_config.unwrap_or((0, false));
        // One pass under one short lock: worker `i` of the table and cursor
        // entry `i` are pushed from the same map entry, so the pairing
        // `cursors[i].0 == i` is structural rather than comment-enforced, and
        // the nothing-in-between ordering cannot drift the way two separately
        // collected vectors could. `apply_batch` holds this lock twice per
        // batch on the single-writer pump, so keep the hold itself cheap.
        let (workers, cursors) = {
            let guard = self.cursors.lock();
            let mut workers: Vec<WireWorker> = Vec::with_capacity(guard.len());
            let mut cursors: Vec<(u32, i64)> = Vec::with_capacity(guard.len());
            for (i, (w, seq)) in guard.iter().enumerate() {
                workers.push(WireWorker {
                    url: w.url.clone(),
                    dp_rank: w.dp_rank,
                });
                cursors.push((i as u32, *seq));
            }
            (workers, cursors)
        };
        let snap = PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size,
            is_bigram,
            // Same question `snapshot_entry` answers — "am I worth copying?" —
            // read off the tree directly because this path never exports a
            // `nodes` vec to measure instead.
            producer_ready: hash_config.is_some()
                && self.bootstrap.settled()
                && self.tree.node_count() > 0,
            workers,
            cursors,
            nodes: Vec::new(),
        };
        // Encoded inline, unlike the tree export: this body is one entry per
        // rank, so a blocking-pool hop would cost more than the serialise.
        encode_snapshot(&snap)
    }

    /// (test-only) Apply one stored block directly, bypassing the pump, so
    /// server/route tests get an index whose cursors-only and full exports
    /// actually differ — on an empty index the two bodies are indistinguishable,
    /// which is what makes tests built on one unable to fail when the
    /// cursors-only branch breaks. Production applies events only through the
    /// pump channel; this exists because seeding from outside `kv_events`
    /// cannot reach the internals, by design.
    #[cfg(test)]
    pub(crate) fn seed_stored_block_for_test(
        &self,
        worker: &KvWorkerId,
        seq: i64,
        block_hash: i64,
    ) {
        let block_size = self
            .block_size_oracle
            .hash_config()
            .map(|(size, _)| size)
            .unwrap_or(0);
        apply_batch(
            &self.tree,
            &self.cursors,
            worker,
            seq,
            &KvEventBatch {
                ts: 0.0,
                events: vec![KvCacheEvent::BlockStored(super::wire::BlockStored {
                    parent_block_hash: None,
                    block_hashes: vec![block_hash],
                    token_ids: vec![],
                    block_size,
                    lora_id: None,
                    medium: None,
                })],
                attn_dp_rank: None,
            },
        );
    }

    /// A snapshot that declares itself useless, for the paths that must answer
    /// without a tree. Consumers skip a `producer_ready: false` peer and retry.
    fn not_ready_snapshot(&self) -> PeerSnapshot {
        let (block_size, is_bigram) = self.block_size_oracle.hash_config().unwrap_or((0, false));
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size,
            is_bigram,
            producer_ready: false,
            workers: Vec::new(),
            cursors: Vec::new(),
            nodes: Vec::new(),
        }
    }

    async fn snapshot_entry(&self, max_age: Duration) -> (Arc<PeerSnapshot>, Bytes) {
        let mut cache = self.snapshot_cache.lock().await;
        if let Some(c) = cache.as_ref() {
            if c.exported_at.elapsed() < max_age {
                return (Arc::clone(&c.snap), c.body.clone());
            }
        }

        // Stamped BEFORE the cursors are read, so the entry never claims to
        // cover more of the publisher's stream than it does. See
        // `CachedSnapshot::exported_at`.
        let exported_at = Instant::now();

        // Read the cursors BEFORE walking the tree, never after.
        //
        // WHY the order is load-bearing: `export_snapshot` takes each shard's
        // read lock in turn, so it is not one instant, and `apply_batch` mutates
        // the tree before advancing the cursor. Reading cursors last can
        // therefore report a sequence whose effects the walk only partially
        // captured — and the consumer would then filter its own copy of that
        // batch as already-reflected, losing a `BlockRemoved` permanently.
        // Reading them first makes the watermark lag the tree instead: the
        // consumer replays deltas the snapshot already has, and insert/remove
        // are idempotent, so it converges.
        let cursor_by_worker: Vec<(KvWorkerId, i64)> = self
            .cursors
            .lock()
            .iter()
            .map(|(w, seq)| (w.clone(), *seq))
            .collect();
        let tree = Arc::clone(&self.tree);
        let walked = tokio::task::spawn_blocking(move || tree.export_snapshot()).await;
        let (worker_table, nodes) = match walked {
            Ok(v) => v,
            Err(e) => {
                // The walk panicked or the runtime is shutting down. Answer with
                // a snapshot that declares itself useless rather than a partial
                // tree, and do NOT cache it: the consumer skips a
                // `producer_ready: false` peer, so it retries elsewhere.
                warn!(error = %e, "kv-bootstrap: snapshot walk failed; reporting not-ready to peers");
                let snap = Arc::new(self.not_ready_snapshot());
                // Deliberately NOT cached, same as before: a consumer skips a
                // `producer_ready: false` peer, so the next request should get a
                // real attempt rather than a cached refusal. Encoding inline is
                // fine — this body is a handful of bytes.
                let body = encode_snapshot(&snap);
                return (snap, body);
            }
        };
        let index_of: HashMap<&KvWorkerId, u32> = worker_table
            .iter()
            .enumerate()
            .map(|(i, w)| (w, i as u32))
            .collect();
        let cursors: Vec<(u32, i64)> = cursor_by_worker
            .iter()
            .filter_map(|(w, seq)| index_of.get(w).map(|&i| (i, *seq)))
            .collect();
        let hash_config = self.block_size_oracle.hash_config();
        let (block_size, is_bigram) = hash_config.unwrap_or((0, false));
        let snap = Arc::new(PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size,
            is_bigram,
            // "Am I worth copying?", not merely "did I stop waiting?".
            //
            // `settled()` latches when the bootstrap deadline expires, so a
            // replica that failed its own bootstrap is settled while holding an
            // empty tree. Reporting ready on that basis would let two new
            // replicas in a rolling update bootstrap from each other and both
            // inherit nothing. Requiring a non-empty tree also correctly makes
            // the very first replica of a cold fleet a non-source until it has
            // learned something from live events.
            producer_ready: hash_config.is_some() && self.bootstrap.settled() && !nodes.is_empty(),
            workers: worker_table
                .iter()
                .map(|w| WireWorker {
                    url: w.url.clone(),
                    dp_rank: w.dp_rank,
                })
                .collect(),
            cursors,
            nodes,
        });
        // Encode on the blocking pool for the same reason the walk goes there:
        // serialising a multi-megabyte tree is CPU-bound with no await point,
        // and this runs on the runtime that is also proxying requests.
        let to_encode = Arc::clone(&snap);
        let body = match tokio::task::spawn_blocking(move || encode_snapshot(&to_encode)).await {
            Ok(b) => b,
            Err(e) => {
                // Runtime shutting down or the encode panicked. Answer this
                // caller without caching, so a later request retries.
                warn!(error = %e, "kv-bootstrap: snapshot encode failed");
                return (snap, Bytes::new());
            }
        };
        *cache = Some(CachedSnapshot {
            exported_at,
            snap: Arc::clone(&snap),
            body: body.clone(),
        });
        (snap, body)
    }

    /// Shared handle to the bootstrap tracker. `/readyz` reads it to decide
    /// whether initial bootstrap has settled; the metrics surface reads it for
    /// the per-rank state gauge.
    pub fn bootstrap(&self) -> Arc<BootstrapTracker> {
        Arc::clone(&self.bootstrap)
    }

    /// Shared handle to the peer registry, written by peer discovery.
    pub fn peers(&self) -> Arc<PeerRegistry> {
        Arc::clone(&self.peers)
    }

    /// Shared accessor for the per-process block-size oracle. The
    /// `CacheAwareZmqPolicy` (via [`crate::policies::factory`]) holds the
    /// same `Arc` so the value the index seeds is the value the policy
    /// hashes against.
    pub fn block_size_oracle(&self) -> Arc<BlockSizeOracle> {
        Arc::clone(&self.block_size_oracle)
    }

    /// Clone the underlying tree handle for cache-aware selection and
    /// metrics. The pump is the sole writer; callers should treat the
    /// returned handle as read-only.
    pub fn tree(&self) -> Arc<HashTree> {
        self.tree.clone()
    }

    /// Shared accessor for the engine-load table. The `CacheAwareZmqPolicy`
    /// (via [`crate::policies::factory`]) holds the same `Arc` and only reads
    /// it at selection time. Load *values* are written solely by the pump
    /// (from `LoadStat` events); `add_worker` / `remove_worker` here manage
    /// the expected set and per-worker eviction.
    pub fn engine_load(&self) -> Arc<EngineLoadTable> {
        Arc::clone(&self.engine_load)
    }

    /// Register a worker. If `preresolved` is `Some`, the caller has
    /// already fetched `/server_info` (worker manager path) and we skip
    /// the internal HTTP round-trip; otherwise (standalone callers,
    /// e.g. integration tests) we fall back to `fetch_event_config`.
    ///
    /// Opens one ZMQ SUB per advertised DP rank. If the worker is not
    /// publishing KV events (older SGLang, opt-out config), this is a
    /// logged no-op — the worker still routes via the non-cache-aware
    /// policies.
    pub async fn add_worker(&self, worker_url: &str, preresolved: Option<EventConfig>) {
        let cfg: EventConfig = match preresolved {
            Some(c) => c,
            None => match fetch_event_config(worker_url, &self.http).await {
                Ok(Some(c)) => c,
                Ok(None) => {
                    info!(
                        worker_url = %worker_url,
                        "kv-events: worker is not publishing; cache-aware routing disabled for this worker",
                    );
                    return;
                }
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        "kv-events: /server_info introspection failed; skipping subscriber",
                    );
                    return;
                }
            },
        };
        // Reconcile this worker's publish block size with the oracle BEFORE
        // any subscriber state is created. The first worker establishes
        // the value; later workers must agree. A mismatch means the
        // router and at least one engine would compute different block
        // hashes for the same prompt, silently destroying cache-aware
        // routing quality — reject loudly instead.
        if let Err(err) = self.block_size_oracle.try_set(cfg.block_size) {
            warn!(
                worker_url = %worker_url,
                established_block_size = err.established,
                worker_block_size = err.candidate,
                "kv-events: worker block size (page_size * dcp_size) disagrees with the \
                 established block_size; skipping worker — cache-aware routing requires \
                 every worker to publish at the same block size. Check --page-size and \
                 --dcp-size on this worker; the discovery log line for it reports both",
            );
            return;
        }
        // The worker's hashing mode is reported to the oracle below, at the
        // registration-commit point. EAGLE-family workers hash KV blocks over
        // token bigrams, so the policy must use the bigram hasher for its
        // query hashes to match the worker's stored hashes.
        info!(
            worker_url = %worker_url,
            dp_size = cfg.dp_size,
            port_base = cfg.port_base,
            block_size = cfg.block_size,
            is_bigram = cfg.is_bigram,
            "kv-events: subscribing",
        );
        // Compute the DP ranks that will actually be subscribed (skip
        // ranks whose port overflows u16; the subscriber will warn on
        // each skipped rank).
        let port_base_u32 = u32::from(cfg.port_base);
        let dp_ranks: Vec<u32> = (0..cfg.dp_size)
            .filter(|rank| (port_base_u32 + rank) <= u32::from(u16::MAX))
            .collect();
        if dp_ranks.is_empty() {
            warn!(
                worker_url = %worker_url,
                port_base = cfg.port_base,
                dp_size = cfg.dp_size,
                "kv-events: every advertised rank's port overflows u16; skipping worker",
            );
            return;
        }
        let ids: Vec<KvWorkerId> = dp_ranks
            .iter()
            .map(|&rank| KvWorkerId::new(worker_url.to_string(), rank))
            .collect();
        // Mark every rank live BEFORE the subscriber starts so any event
        // it queues is accepted by the pump.
        {
            let mut live = self.live_workers.lock();
            for id in &ids {
                live.insert(id.clone());
            }
        }
        // Register for bootstrap BEFORE the subscriber starts too, so the very
        // first batch is held back rather than applied ahead of the snapshot.
        // Ordering matters in the other direction as well: the subscription
        // must be live before the snapshot is fetched, so no delta can fall
        // into the gap between the peer's export and our first received batch.
        let bootstrap_obligations: Vec<(KvWorkerId, u64)> = if self.peer_bootstrap_enabled() {
            self.bootstrap.register(&ids)
        } else {
            Vec::new()
        };
        // Registration commits here: the worker's hashing mode joins the
        // oracle's live registry together with its worker entry, so every
        // early return above leaves the mode registry untouched (the block
        // size latch may already have been published by `try_set`).
        self.block_size_oracle
            .report_worker(worker_url, cfg.is_bigram);
        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerEntry {
                dp_ranks: dp_ranks.clone(),
            },
        );
        self.subscribers.add_worker(worker_url, &cfg).await;
        if !bootstrap_obligations.is_empty() {
            // Stamped HERE, after `subscribers.add_worker` — not at `register`.
            // It is the instant the subscriber went live that a peer's export has
            // to beat, and anything earlier would let the sweep accept a snapshot
            // taken during the subscribe window, which is precisely the hole the
            // watermark check would then reject as `Gap`.
            let batch = ObligationBatch {
                obligations: bootstrap_obligations,
                holding_since: Instant::now(),
                late_join: LateJoin::Permitted,
            };
            // Hand off rather than sweeping here, so 168 discovered workers
            // share one fleet-wide fetch instead of pulling the same body 168
            // times. This rank's subscriber is already live (above), so whichever
            // sweep picks the obligation up exports strictly after it.
            if let Err(e) = self.bootstrap_tx.try_send(batch) {
                // Queue full or coordinator gone. Fall back to sweeping
                // directly: an obligation that nothing owns would leave its rank
                // `Pending` with batches held until the queue overflows, so
                // dropping it is not an option.
                warn!("kv-bootstrap: batch queue unavailable ({e}); sweeping un-batched");
                self.spawn_bootstrap(e.into_inner());
            }
        }
        // Load subscribers (one per rank on the dedicated load port). A no-op
        // when the worker advertises no load port (older engine). Workers that
        // do advertise one are marked expected, so the router can tell a dead
        // load publisher apart from one that was never configured.
        if cfg.load_port_base.is_some() {
            self.engine_load.mark_expected(worker_url);
        }
        self.load_subscribers.add_worker(worker_url, &cfg).await;
    }

    /// Whether this worker's ranks should enter the bootstrap state machine.
    ///
    /// Keyed on whether peer bootstrap is CONFIGURED, deliberately not on whether
    /// it has settled. Those answer different questions, and conflating them cost
    /// real warm coverage: a worker discovered after the first graft resolved
    /// every pending rank found the tracker latched, was denied an obligation, and
    /// so never got its subtree — the replica served a tree short by that
    /// worker's blocks, with no `bootstrap_state` entry to say so. Readiness is
    /// safe either way, because `settled()` short-circuits on `latched` and a rank
    /// registered afterwards cannot drag `/readyz` back to 503.
    ///
    /// WHY this deliberately does NOT check for an empty peer set: registering is
    /// what arms the bootstrap deadline. Skipping registration because nobody is
    /// available to ask leaves the deadline unarmed, and an unarmed deadline
    /// means `settled()` can never become true — so `/readyz` would stay 503
    /// forever. That deadlocks a whole fleet, because an unready replica is
    /// absent from its own EndpointSlice, so *every* replica would see an empty
    /// peer set and none would ever register. Ranks are always registered; the
    /// no-peer case is resolved by `spawn_bootstrap`, which waits for discovery
    /// to confirm there are no siblings and only then abandons.
    fn peer_bootstrap_enabled(&self) -> bool {
        self.bootstrap.enabled()
    }

    /// Fetch a snapshot for `ranks` from the warmest peer that answers, and
    /// hand the result to the pump.
    ///
    /// Runs detached: `/readyz` is gated by the tracker, not by awaiting this,
    /// so a slow peer delays readiness only up to the bootstrap deadline. Every
    /// exit path sends exactly one [`PumpControl`] message, which is what
    /// guarantees the held-back batches are eventually released.
    fn spawn_bootstrap(&self, batch: ObligationBatch) {
        let deps = self.bootstrap_deps();
        tokio::spawn(async move {
            let ObligationBatch {
                obligations,
                holding_since,
                late_join: _,
            } = batch;
            let ranks: Vec<KvWorkerId> = obligations.iter().map(|(r, _)| r.clone()).collect();
            let deadline = deps.deadline();
            let result = sweep_until_deadline(&deps, &ranks, deadline, holding_since).await;
            deliver_bootstrap(&deps, obligations, result, deadline).await;
        });
    }

    /// Clone the handles a sweep needs, so it can run detached (or be awaited by
    /// the coordinator) without borrowing `self`.
    fn bootstrap_deps(&self) -> BootstrapDeps {
        BootstrapDeps {
            http: self.snapshot_http.clone(),
            peers: Arc::clone(&self.peers),
            bootstrap: Arc::clone(&self.bootstrap),
            live_workers: Arc::clone(&self.live_workers),
            oracle: Arc::clone(&self.block_size_oracle),
            ctrl_tx: self.ctrl_tx.clone(),
        }
    }

    /// Tear down a worker's subscribers and clear it from the tree.
    /// Idempotent: a remove for a worker that was never added is a no-op.
    ///
    /// The live-worker entries are dropped **before** the subscriber join,
    /// so any event still buffered in the mpsc by the time the pump
    /// reaches it is dropped instead of re-inserted into the tree.
    pub async fn remove_worker(&self, worker_url: &str) {
        let Some(entry) = self.workers.lock().remove(worker_url) else {
            return;
        };
        // The oracle vote drops WITH the entry, not at the end of teardown:
        // this function awaits below, and placing the forget after those
        // awaits would let a same-URL worker re-added in the window have its
        // fresh vote deleted when this task resumes.
        self.block_size_oracle.forget_worker(worker_url);
        let ids: Vec<KvWorkerId> = entry
            .dp_ranks
            .iter()
            .map(|&dp_rank| KvWorkerId {
                url: worker_url.to_string(),
                dp_rank,
            })
            .collect();
        // 1. Mark every rank dead. Any pump-queued events arriving after
        //    this point will be filtered.
        {
            let mut live = self.live_workers.lock();
            for id in &ids {
                live.remove(id);
            }
        }
        // 2. Cancel and join the per-rank subscriber tasks (KV + load). No
        //    further events for these ranks will be queued after this returns.
        self.subscribers.remove_worker(worker_url).await;
        self.load_subscribers.remove_worker(worker_url).await;
        // 3. Drop each rank's tree state and cursor, and the worker's engine
        //    load. Any event already in the mpsc buffer at this point will be
        //    filtered by the live-set check inside the pump.
        self.engine_load.forget_worker(worker_url);
        // Drop bootstrap state too, so a rank that never finished cannot hold
        // the readiness gate open after its worker is gone.
        self.bootstrap.forget(&ids);
        // Hand the whole teardown to the pump: `held`, `awaiting_splice_proof`,
        // the tree carriers, and the cursor. Doing the tree/cursor half here would
        // race a graft already past its gates (see the `ForgetRanks` arm).
        //
        // ORDER IS LOAD-BEARING: `bootstrap.forget` above must run BEFORE this
        // send. That is what makes an in-flight `ApplySnapshot` for these ranks a
        // no-op — `apply_snapshot` filters on `state_of == Pending`, which
        // `forget` has already cleared. Swapping the two reopens the hole.
        if self
            .ctrl_tx
            .send(PumpControl::ForgetRanks { ranks: ids.clone() })
            .await
            .is_err()
        {
            // Only reachable once the pump has exited, i.e. during shutdown. Clean
            // up inline so the state does not outlive the worker in that case;
            // there is no writer left to race.
            debug!("kv-events: pump is gone; tearing down worker state inline");
            let mut cursors = self.cursors.lock();
            for id in &ids {
                self.tree.clear_worker(id);
                cursors.remove(id);
            }
        }
    }

    /// Number of worker URLs the index is currently subscribed to. The
    /// count includes workers whose `/server_info` resolved but excludes
    /// any whose discovery returned `Ok(None)` (worker reachable but not
    /// publishing) or `Err` (transient discovery failure). Exposed for
    /// tests + future metrics; not part of the routing hot path.
    pub fn known_worker_count(&self) -> usize {
        self.workers.lock().len()
    }

    /// Shut down the pump task. Cancels the subscriber registry first so no
    /// further events are queued, then cancels the pump so any buffered
    /// events are discarded and the task exits promptly.
    pub async fn shutdown(&self) {
        self.subscribers.shutdown().await;
        self.load_subscribers.shutdown().await;
        self.pump_cancel.cancel();
        let handle = self.pump.lock().take();
        if let Some(h) = handle {
            // 2s ceiling guards against a pathological tokio runtime
            // teardown; under normal operation the pump exits within one
            // poll of `pump_cancel.cancelled()`.
            match tokio::time::timeout(Duration::from_secs(2), h).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!(error = %e, "kv-events pump task did not join cleanly"),
                Err(_) => warn!("kv-events pump task did not stop within 2s"),
            }
        }
    }
}

/// The handles a bounded peer sweep needs, cloned out of [`KvEventIndex`] so the
/// sweep can run detached — or be awaited by the coordinator — without borrowing
/// `self`.
#[derive(Clone)]
struct BootstrapDeps {
    http: reqwest::Client,
    peers: Arc<PeerRegistry>,
    bootstrap: Arc<BootstrapTracker>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    oracle: Arc<BlockSizeOracle>,
    ctrl_tx: mpsc::Sender<PumpControl>,
}

impl BootstrapDeps {
    /// Budget for one sweep.
    ///
    /// While readiness is still gated, this is whatever is LEFT of the tracker's
    /// single window — never a fresh one. That window is the `/readyz` gate, armed
    /// once at first worker discovery, and `--kv-bootstrap-timeout-ms` is
    /// documented as how long readiness may hold; re-arming per sweep would hold
    /// it for an unbounded multiple of the configured value, the same bug
    /// `BootstrapTracker::rearmed` exists to prevent for flapping workers.
    ///
    /// Once settled that constraint is gone, so a worker discovered afterwards
    /// gets a full window instead of the remainder — which is zero, because
    /// `time_remaining` saturates at the deadline. Without this, allowing late
    /// ranks to register would be pointless: every one of them would time out
    /// instantly and run cold, warming nothing.
    fn deadline(&self) -> Duration {
        if self.bootstrap.settled() {
            return self.bootstrap.timeout();
        }
        self.bootstrap
            .time_remaining()
            .unwrap_or(self.bootstrap.timeout())
    }
}

/// Outcome of one bounded peer sweep.
enum SweepResult {
    Found(VettedSnapshot),
    /// Discovery confirmed there are no siblings, so waiting cannot help.
    NoPeers,
    TimedOut {
        peers_tried: usize,
        last_reason: Option<String>,
    },
}

/// Sweep peers until one yields a usable snapshot, discovery proves there are
/// none, or the budget runs out.
///
/// Retries rather than sweeping once: worker discovery regularly completes before
/// the peer watch has delivered its first EndpointSlice list, so a single sweep
/// sees zero candidates and abandons — the joining replica then boots cold even
/// though warm siblings existed. The same loop covers a rolling update whose only
/// visible candidates are surge pods still finishing their own bootstrap.
///
/// `freshness_floor` is the instant every fetch demands the peer's export beat:
/// re-derived per attempt, so a sweep that runs for minutes keeps asking for the
/// same coverage rather than drifting into accepting older state.
async fn sweep_until_deadline(
    deps: &BootstrapDeps,
    ranks: &[KvWorkerId],
    deadline: Duration,
    freshness_floor: Instant,
) -> SweepResult {
    let ctx = SweepCtx {
        http: &deps.http,
        peers: &deps.peers,
        bootstrap: &deps.bootstrap,
        live_workers: &deps.live_workers,
        oracle: &deps.oracle,
        freshness_floor,
    };
    // Shared so the terminal log can name the last concrete reason rather than
    // only "no usable snapshot".
    let last_reason: Mutex<Option<String>> = Mutex::new(None);
    let attempt = async {
        let mut permanently_rejected: HashSet<String> = HashSet::new();
        let mut cooldown: HashMap<String, u32> = HashMap::new();
        loop {
            if let Some(vetted) = sweep_peers(
                &ctx,
                ranks,
                &mut permanently_rejected,
                &mut cooldown,
                &last_reason,
            )
            .await
            {
                return Some(vetted);
            }
            if deps.peers.known_to_have_no_peers() {
                debug!("kv-bootstrap: discovery confirmed no sibling replicas");
                return None;
            }
            tokio::time::sleep(PEER_RETRY_INTERVAL).await;
        }
    };

    // The deadline bounds the whole sweep, not each request, so a fleet of slow
    // peers cannot outlast the readiness gate.
    match tokio::time::timeout(deadline, attempt).await {
        Ok(Some(vetted)) => SweepResult::Found(vetted),
        Ok(None) => SweepResult::NoPeers,
        Err(_) => SweepResult::TimedOut {
            peers_tried: deps.peers.len(),
            last_reason: last_reason.lock().clone(),
        },
    }
}

/// Turn a sweep result into the single [`PumpControl`] message its obligations
/// are owed. Every exit path sends exactly one, which is what releases the ranks
/// from `Pending`.
async fn deliver_bootstrap(
    deps: &BootstrapDeps,
    obligations: Vec<(KvWorkerId, u64)>,
    result: SweepResult,
    deadline: Duration,
) {
    let n = obligations.len();
    let msg = match result {
        SweepResult::Found(vetted) => PumpControl::ApplySnapshot {
            obligations,
            vetted: Box::new(vetted),
        },
        SweepResult::NoPeers => {
            info!(
                ranks = n,
                "kv-bootstrap: no sibling replicas to bootstrap from; ranks will run cold",
            );
            PumpControl::AbandonBootstrap { obligations }
        }
        SweepResult::TimedOut {
            peers_tried,
            last_reason,
        } => {
            warn!(
                ranks = n,
                timeout_ms = deadline.as_millis(),
                peers_tried,
                last_reason = last_reason.as_deref().unwrap_or("none recorded"),
                "kv-bootstrap: no peer supplied a usable snapshot within the deadline; \
                 ranks will run cold",
            );
            PumpControl::AbandonBootstrap { obligations }
        }
    };
    if deps.ctrl_tx.send(msg).await.is_err() {
        warn!("kv-bootstrap: pump is gone; bootstrap result discarded");
    }
}

/// One pass over the candidate peers, returning the first snapshot that vets.
///
/// Kept separate from [`sweep_until_deadline`]'s retry loop so "which peer do we
/// take?" and "how long do we keep looking?" stay independently readable.
struct SweepCtx<'a> {
    http: &'a reqwest::Client,
    peers: &'a PeerRegistry,
    bootstrap: &'a BootstrapTracker,
    live_workers: &'a Mutex<HashSet<KvWorkerId>>,
    oracle: &'a BlockSizeOracle,
    /// See [`sweep_until_deadline`]. Held as the instant, converted to an age at
    /// each fetch.
    freshness_floor: Instant,
}

async fn sweep_peers(
    ctx: &SweepCtx<'_>,
    ranks: &[KvWorkerId],
    permanently_rejected: &mut HashSet<String>,
    cooldown: &mut HashMap<String, u32>,
    last_reason: &Mutex<Option<String>>,
) -> Option<VettedSnapshot> {
    let SweepCtx {
        http,
        peers,
        bootstrap,
        live_workers,
        oracle,
        freshness_floor,
    } = ctx;
    // The vet needs only the local block size: block hashes at different
    // page sizes can never share a tree. Hashing mode is deliberately not
    // vetted — see `VettedSnapshot::from_wire`.
    let local_block_size = oracle.get()?;
    for peer in peers.candidates() {
        // A format / block-size mismatch is a stable property of that
        // peer for the life of the process, so re-downloading its whole tree on
        // every 250ms pass cannot change the answer — it just loads a replica
        // that is itself booting.
        if permanently_rejected.contains(&peer) {
            continue;
        }
        // Retriable, but not on every single pass.
        if let Some(remaining) = cooldown.get_mut(&peer) {
            if *remaining > 0 {
                *remaining -= 1;
                continue;
            }
        }
        // Ask for an export that beats the floor. Derived per attempt, not once:
        // the condition is "newer than the floor", and only the age it
        // corresponds to moves as the sweep retries.
        let max_age = freshness_floor.elapsed();
        let snap = match fetch_snapshot(http, &peer, Some(max_age)).await {
            Ok(Some(s)) => s,
            Ok(None) => {
                bootstrap.record_peer_outcome(SnapshotOutcome::Unreachable, &peer, None);
                cooldown.insert(peer.clone(), PEER_COOLDOWN_PASSES);
                continue;
            }
            Err(e) => {
                // `{e:#}` for the anyhow chain: the bare Display prints only the
                // outermost message, dropping the reqwest/io cause that names what
                // actually went wrong.
                bootstrap.record_peer_outcome(
                    SnapshotOutcome::Unreachable,
                    &peer,
                    Some(&format!("{e:#}")),
                );
                // Cool this peer down like every other non-accept outcome. The
                // failures reaching here are mostly stable properties of that peer
                // for the life of the process — a transport it cannot complete, a
                // body that will not inflate, JSON that will not parse — so
                // re-fetching a multi-megabyte body from it on every 250ms pass is
                // pure load on a replica that is itself serving traffic, and it
                // starves candidates that might actually answer.
                cooldown.insert(peer.clone(), PEER_COOLDOWN_PASSES);
                continue;
            }
        };
        // Snapshot the live set at vet time so a peer cannot introduce a worker
        // this replica has not discovered.
        let live = live_workers.lock().clone();
        match VettedSnapshot::from_wire(snap, &live, Some(local_block_size)) {
            Ok(vetted) => {
                // Vetting only proves the snapshot is well formed and hash-
                // comparable. It can still know nothing about the ranks we are
                // bootstrapping, in which case accepting it would end the sweep
                // and leave those ranks cold.
                if !vetted.covers_any(ranks) {
                    debug!(
                        peer = %peer,
                        nodes = vetted.node_count(),
                        "kv-bootstrap: peer has no state for the ranks being bootstrapped; \
                         continuing to look",
                    );
                    cooldown.insert(peer.clone(), PEER_COOLDOWN_PASSES);
                    continue;
                }
                // How much of the copy can steer a selection: carried nodes
                // hash under a mode their carriers voted at fetch time;
                // structure nodes are match paths only. An `unvoted` carrier
                // means the vote left the registry between vet and accept
                // (a removal interleaved). See `VettedSnapshot`'s
                // `carrier_counts`.
                let (carried, structure) = vetted.carrier_counts();
                let (bigram, unigram, unvoted) =
                    vetted.workers().iter().fold((0, 0, 0), |(b, u, x), w| {
                        match oracle.vote_of(&w.url) {
                            Some(true) => (b + 1, u, x),
                            Some(false) => (b, u + 1, x),
                            None => (b, u, x + 1),
                        }
                    });
                info!(
                    peer = %peer,
                    nodes = vetted.node_count(),
                    carried_nodes = carried,
                    structure_nodes = structure,
                    workers = vetted.worker_count(),
                    dropped_workers = vetted.dropped_workers(),
                    carrier_votes = %format!("{bigram} bigram / {unigram} unigram / {unvoted} unvoted"),
                    "kv-bootstrap: snapshot accepted; handing to pump",
                );
                bootstrap.record_peer_outcome(SnapshotOutcome::Accepted, &peer, None);
                return Some(vetted);
            }
            Err(e) => {
                if e.outcome() == SnapshotOutcome::Rejected {
                    // Loud: a fleet-wide block-size or format disagreement means
                    // NO replica can ever bootstrap, and at debug level the only
                    // symptom is a generic deadline warning.
                    warn!(
                        peer = %peer,
                        error = %e,
                        "kv-bootstrap: peer snapshot is permanently incompatible; \
                         not retrying this peer",
                    );
                    permanently_rejected.insert(peer.clone());
                }
                // Scoped so no guard is ever held across an await.
                *last_reason.lock() = Some(format!("{peer}: {e}"));
                cooldown.insert(peer.clone(), PEER_COOLDOWN_PASSES);
                bootstrap.record_peer_outcome(e.outcome(), &peer, Some(&e.to_string()));
            }
        }
    }
    None
}

/// Shared state the pump task reads and writes. Bundled rather than passed as
/// eight positional arguments, which is both unreadable and easy to transpose.
struct PumpDeps {
    tree: Arc<HashTree>,
    engine_load: Arc<EngineLoadTable>,
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    bootstrap: Arc<BootstrapTracker>,
    /// Fleet-mode registry the pump refreshes after a publisher reset (see
    /// `spawn_mode_recheck`), and the `/server_info` client for that fetch —
    /// distinct from `snapshot_http`, which pulls peer snapshot bodies.
    /// Oracle writes deliberately bypass `ctrl_tx`: the oracle has its own
    /// concurrency semantics; the single-writer rule covers tree/cursor/load
    /// state only.
    oracle: Arc<BlockSizeOracle>,
    http: reqwest::Client,
    /// Peer set and client for the splice probe; see `spawn_splice_probe`. The
    /// pump does not fetch snapshots for bootstrap itself — only this one
    /// question, about state it already grafted.
    peers: Arc<PeerRegistry>,
    snapshot_http: reqwest::Client,
    /// Obligation queue, so a gap-discarded rank can be handed back for another
    /// sweep instead of staying cold with budget unspent.
    bootstrap_tx: mpsc::Sender<ObligationBatch>,
    /// Loopback into this pump's own control channel, so a probe answer arrives
    /// on the single writer like every other tree mutation.
    ctrl_tx: mpsc::Sender<PumpControl>,
}

/// Re-introspect one worker's hashing mode after its publisher restarted.
///
/// Runs detached: an HTTP round-trip must never stall the single writer. No
/// coalescing beyond the natural one-task-per-`PublisherReset` granularity —
/// a worker restart resets all its ranks, and a handful of concurrent
/// `/server_info` GETs is cheaper than bookkeeping to merge them.
///
/// The write is fenced the same way `PumpControl::ApplySnapshot` is: the
/// rank epoch is captured at spawn and rechecked before reporting, alongside
/// a live-set membership test. Without that gate a fetch that outlives
/// `remove_worker` (up to the full retry budget) would resurrect a vote for
/// a dead worker — with no future `forget_worker` to erase it — and a fetch
/// from an older incarnation could overwrite a re-added worker's fresh vote.
/// Trackers with no registered epochs (peer bootstrap disabled) fence on the
/// live set alone.
fn spawn_mode_recheck(
    http: &reqwest::Client,
    oracle: &Arc<BlockSizeOracle>,
    live_workers: &Arc<Mutex<HashSet<KvWorkerId>>>,
    bootstrap: &Arc<BootstrapTracker>,
    worker: &KvWorkerId,
) {
    let http = http.clone();
    let oracle = Arc::clone(oracle);
    let live_workers = Arc::clone(live_workers);
    let bootstrap = Arc::clone(bootstrap);
    let id = worker.clone();
    let url = worker.url.clone();
    let epoch = bootstrap.epoch_of(&id);
    tokio::spawn(async move {
        let cfg = match fetch_event_config(&url, &http).await {
            Ok(Some(cfg)) => cfg,
            Ok(None) => {
                // The worker answered: it no longer publishes KV events.
                // Keep the vote — removal is the manager's call.
                info!(
                    worker_url = %url,
                    "kv-events: worker stopped publishing KV events across its restart; \
                     keeping the vote until removal",
                );
                return;
            }
            Err(e) => {
                // Unreachable or a definitive HTTP failure: a fetch that
                // cannot answer must not re-shape the fleet.
                warn!(
                    worker_url = %url,
                    error = %e,
                    "kv-events: post-restart mode re-introspection failed; keeping the vote",
                );
                return;
            }
        };
        // Fence the write (see the fn doc): only land a vote that the
        // worker's CURRENT incarnation can still claim.
        if bootstrap.epoch_of(&id) != epoch {
            debug!(
                worker_url = %url,
                "kv-events: mode re-introspection skipped; the rank changed incarnation mid-fetch",
            );
            return;
        }
        if !live_workers.lock().iter().any(|r| r.url == url) {
            debug!(
                worker_url = %url,
                "kv-events: mode re-introspection skipped; the worker was removed mid-fetch",
            );
            return;
        }
        if let Some(established) = oracle.get() {
            if established != cfg.block_size {
                warn!(
                    worker_url = %url,
                    established_block_size = established,
                    worker_block_size = cfg.block_size,
                    "kv-events: restarted worker's block size (page_size * dcp_size) disagrees \
                     with the fleet; its entries will never match queries — restart it with \
                     --page-size / --dcp-size matching the fleet",
                );
            }
        }
        let previous = oracle.vote_of(&url);
        if previous != Some(cfg.is_bigram) {
            warn!(
                worker_url = %url,
                previous = ?previous,
                is_bigram = cfg.is_bigram,
                "kv-events: worker hashing mode changed across a publisher restart; \
                 updating its vote in the fleet view",
            );
        }
        oracle.report_worker(&url, cfg.is_bigram);
    });
}

/// Drain `WorkerEvent`s: apply KV `Batch`es to the tree and `Load` snapshots
/// to the engine-load table. Out-of-order (seq ≤ last_applied) and stale
/// (worker not in `live_workers`) KV batches are skipped; `Load` is a gauge
/// with no seq. `PublisherReset` events clear the cursor so a publisher
/// restarting from seq=1 (after sending END_SEQ) is not filtered.
///
/// Also the sole writer of tree state, including snapshot grafts arriving as
/// [`PumpControl`]; see the single-writer property in [`super::tree`].
async fn pump_loop(
    deps: PumpDeps,
    cancel: CancellationToken,
    mut rx: mpsc::Receiver<WorkerEvent>,
    mut ctrl_rx: mpsc::Receiver<PumpControl>,
) {
    let PumpDeps {
        tree,
        engine_load,
        cursors,
        live_workers,
        bootstrap,
        oracle,
        http,
        peers,
        snapshot_http,
        bootstrap_tx,
        ctrl_tx,
    } = deps;
    let pump_state = PumpState {
        tree: &tree,
        cursors: &cursors,
        bootstrap: &bootstrap,
        bootstrap_tx: &bootstrap_tx,
        live_workers: &live_workers,
    };

    // Batches held back while their rank is `Pending`. Pump-local: the pump is
    // the only task that touches it, so no lock is needed.
    let mut held: HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>> = HashMap::new();
    // Ranks grafted from a snapshot whose continuity with the live stream is
    // not yet provable, mapped to the watermark and when the wait started.
    //
    // WHY deferred: a snapshot can be grafted before the rank's first live
    // batch has even arrived, so there is nothing to compare the watermark
    // against yet. The check runs on whichever batch turns up first — held or
    // live — and the entry is consumed by that one check, or by the sweep below
    // if no batch ever arrives.
    let mut awaiting_splice_proof: HashMap<KvWorkerId, PendingProof> = HashMap::new();
    let mut proof_sweep = tokio::time::interval(SPLICE_PROOF_SWEEP_INTERVAL);
    proof_sweep.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Once the control channel closes its `recv()` resolves immediately and
    // forever, so it must be dropped from the select or the loop spins hot.
    let mut ctrl_open = true;

    loop {
        let ev = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                info!("kv-events pump: shutdown requested; exiting");
                return;
            }
            // Control first: releasing held batches unblocks readiness, and the
            // channel is near-empty in steady state.
            ctrl = ctrl_rx.recv(), if ctrl_open => {
                match ctrl {
                    Some(PumpControl::ApplySnapshot {
                        obligations,
                        vetted,
                    }) => {
                        apply_snapshot(
                            &pump_state,
                            &mut held,
                            &mut awaiting_splice_proof,
                            &obligations,
                            *vetted,
                        );
                    }
                    Some(PumpControl::ForgetRanks { ranks }) => {
                        for rank in ranks {
                            held.remove(&rank);
                            awaiting_splice_proof.remove(&rank);
                            // Tree and cursor teardown happens HERE, on the single
                            // writer, not in `remove_worker`.
                            //
                            // WHY it moved: `apply_snapshot` samples its gates once
                            // and then does O(nodes) of work (retain, prune,
                            // restore). With teardown running off-pump, a graft
                            // that passed its gates could finish AFTER
                            // `clear_worker` and re-insert carriers plus a stale
                            // cursor for a worker that is gone — and nothing would
                            // ever reclaim them. Routing both through this channel
                            // makes the two strictly ordered: a graft queued before
                            // this is applied and then undone here; one queued
                            // after is filtered by the epoch/live/state gates.
                            tree.clear_worker(&rank);
                            cursors.lock().remove(&rank);
                        }
                    }
                    Some(PumpControl::AbandonBootstrap { obligations }) => {
                        for (rank, epoch) in obligations {
                            if bootstrap.epoch_of(&rank) != Some(epoch) {
                                // Superseded incarnation; its state belongs to a
                                // different add_worker.
                                continue;
                            }
                            fail_rank(
                                &pump_state,
                                &mut held,
                                &rank,
                                false,
                                RankOutcome::Abandoned,
                            );
                        }
                    }
                    Some(PumpControl::SpliceProbe { rank, epoch, verdict }) => {
                        // Whatever else happens to this verdict, the probe that
                        // produced it is no longer outstanding. Clearing the
                        // marker HERE — before the gate below, not per-arm —
                        // keeps a verdict the gate drops (superseded epoch while
                        // the entry survives) from looking like a running probe
                        // until the presumed-lost timeout releases the rank.
                        // That drop does occur: remove_worker forgets the epoch
                        // BEFORE its ForgetRanks lands on this channel, so a
                        // verdict queued ahead of that ForgetRanks arrives with
                        // its epoch already gone and its entry still present.
                        // The entry cannot outlive the drop — the already-queued
                        // ForgetRanks removes it — but the marker must not
                        // linger even that long.
                        if let Some(proof) = awaiting_splice_proof.get_mut(&rank) {
                            proof.probe_launched = None;
                        }
                        // The rank may have proven itself, been forgotten, or been
                        // re-registered while the probe was in flight; the epoch
                        // and the map entry together say whether the answer is
                        // still about the state we asked on behalf of.
                        if bootstrap.epoch_of(&rank) != Some(epoch)
                            || !awaiting_splice_proof.contains_key(&rank)
                        {
                            debug!(
                                worker = ?rank,
                                epoch,
                                "kv-bootstrap: dropping a probe verdict that no longer \
                                 addresses live state",
                            );
                            continue;
                        }
                        match verdict {
                            SpliceVerdict::Advanced => {
                                awaiting_splice_proof.remove(&rank);
                                demote_unproven_rank(&pump_state, &rank, RankOutcome::Gap);
                                requeue_gapped_rank(&bootstrap, &bootstrap_tx, &rank);
                            }
                            SpliceVerdict::NoAdvance => {
                                awaiting_splice_proof.remove(&rank);
                                bootstrap.record_rank_outcome(RankOutcome::Warm);
                            }
                            // No witness. Keep the rank warm and ask again — but
                            // not forever: a fleet that never answers (a
                            // single-replica deployment, say) would otherwise
                            // leave the verdict unresolved and the probe looping.
                            SpliceVerdict::Unknown => {
                                if let Some(proof) = awaiting_splice_proof.get_mut(&rank) {
                                    proof.unknown_probes += 1;
                                    debug!(
                                        worker = ?rank,
                                        watermark = proof.watermark,
                                        probes = proof.unknown_probes,
                                        max_unknown_probes = MAX_UNKNOWN_PROBES,
                                        "kv-bootstrap: no witness answered this probe",
                                    );
                                    if proof.unknown_probes >= MAX_UNKNOWN_PROBES {
                                        info!(
                                            worker = ?rank,
                                            watermark = proof.watermark,
                                            probes = proof.unknown_probes,
                                            "kv-bootstrap: no peer could witness this rank's \
                                             progress; keeping the grafted state unproven",
                                        );
                                        awaiting_splice_proof.remove(&rank);
                                        bootstrap
                                            .record_rank_outcome(RankOutcome::WarmUnwitnessed);
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        debug!("kv-events pump: control channel closed");
                        ctrl_open = false;
                    }
                }
                continue;
            }
            // Ranks whose splice proof never arrived. Placed in the select rather
            // than keyed off event arrival BECAUSE the failure mode is the absence
            // of events: a rank that goes quiet right after a graft is exactly the
            // one that would otherwise never be checked.
            _ = proof_sweep.tick(), if !awaiting_splice_proof.is_empty() => {
                let expired: Vec<(KvWorkerId, i64)> = awaiting_splice_proof
                    .iter_mut()
                    .filter(|(_, p)| p.due_for_probe(SPLICE_PROOF_TIMEOUT))
                    .map(|(rank, p)| {
                        // Re-arm both clocks before probing so retries are
                        // spaced by the timeout, never by the sweep tick. A
                        // probe still running past that window is presumed
                        // lost — its verdict may never land — and asked again
                        // once per timeout.
                        p.launch_probe(Instant::now());
                        (rank.clone(), p.watermark)
                    })
                    .collect();
                for (rank, watermark) in expired {
                    // Do NOT discard on silence alone. Reaching the deferred path
                    // means nothing arrived between subscribing and grafting, and
                    // the subscriber is live before the snapshot is fetched — so
                    // silence is far more often "this rank published nothing" than
                    // "we lost a delta". Discarding on a timer would throw away a
                    // healthy warm tree on every quiet fleet, which is the exact
                    // regression this feature exists to prevent. Ask the fleet
                    // instead, and act only on positive evidence.
                    let Some(epoch) = bootstrap.epoch_of(&rank) else {
                        awaiting_splice_proof.remove(&rank);
                        continue;
                    };
                    spawn_splice_probe(
                        snapshot_http.clone(),
                        Arc::clone(&peers),
                        ctrl_tx.clone(),
                        rank,
                        watermark,
                        epoch,
                    );
                }
                continue;
            }
            recv = rx.recv() => match recv {
                Some(ev) => ev,
                None => {
                    warn!("kv-events pump: receiver closed unexpectedly; exiting");
                    return;
                }
            }
        };

        // Filter events from workers that are no longer attached. This is
        // load-bearing: `remove_worker` clears the live set BEFORE joining
        // the subscriber task, so any event still buffered when the pump
        // reaches it would otherwise re-pollute the tree.
        let worker = ev.worker();
        if !live_workers.lock().contains(worker) {
            debug!(
                worker = ?worker,
                "kv-events pump: dropping event from detached worker",
            );
            continue;
        }

        match ev {
            WorkerEvent::Load { worker, load } => {
                // Gauge: last value wins, no sequence/dedup. The live-worker
                // filter above already dropped load from detached workers.
                engine_load.set(&worker.url, worker.dp_rank, load, Instant::now());
            }
            WorkerEvent::PublisherReset { worker } => {
                // A fresh publisher restarts sequencing at 1, so any pending
                // splice proof is about a stream that no longer exists and
                // would misfire against the new numbering.
                awaiting_splice_proof.remove(&worker);
                // Same reasoning for a rank still awaiting its snapshot: its held
                // batches are pre-reset, and a post-reset seq of 1 can never
                // exceed the peer's watermark, so the gap check would pass and
                // the stale tree would be kept while every real delta was
                // filtered. Bail to cold and discard the dead prefix.
                if bootstrap.state_of(&worker) == Some(BootstrapState::Pending) {
                    warn!(
                        worker = ?worker,
                        "kv-bootstrap: publisher reset while awaiting a snapshot; \
                         abandoning bootstrap for this rank",
                    );
                    fail_rank(
                        &pump_state,
                        &mut held,
                        &worker,
                        true,
                        RankOutcome::PublisherReset,
                    );
                }
                if cursors.lock().remove(&worker).is_some() {
                    info!(
                        worker = ?worker,
                        "kv-events pump: publisher reset; cursor cleared",
                    );
                }
                // A reset also means the engine process restarted — possibly
                // with different speculative-decoding config. Discovery only
                // notices restarts that change its entry (pod replacement, a
                // readiness toggle, Removed→Added); a silent in-place restart
                // (same pod UID/IP, or static worker-urls) emits nothing, so
                // the pump re-introspects the vote itself. Note resets only
                // follow a graceful publisher shutdown; a crash restart
                // surfaces as a stream gap handled by the gap path instead.
                spawn_mode_recheck(&http, &oracle, &live_workers, &bootstrap, &worker);
            }
            WorkerEvent::Batch { worker, seq, batch } => {
                // A rank still awaiting its snapshot holds its batches: applying
                // them first would put live deltas *under* the snapshot, where a
                // stale `BlockStored` from the peer could resurrect a block this
                // rank has already evicted.
                if bootstrap.state_of(&worker) == Some(BootstrapState::Pending) {
                    let queue = held.entry(worker.clone()).or_default();
                    if queue.len() >= PENDING_BATCH_LIMIT {
                        // Dropping from the middle of the stream would leave a
                        // hole the snapshot cannot be spliced across, so give up
                        // on bootstrapping this rank and let it run live.
                        warn!(
                            worker = ?worker,
                            limit = PENDING_BATCH_LIMIT,
                            "kv-bootstrap: held-batch limit reached before a snapshot arrived; \
                             abandoning bootstrap for this rank",
                        );
                        fail_rank(&pump_state, &mut held, &worker, true, RankOutcome::Overflow);
                    } else {
                        queue.push_back((seq, batch));
                        continue;
                    }
                    // Falls through: the rank is no longer Pending, so this
                    // batch is applied directly below.
                }
                // First batch after a graft proves — or disproves — that the
                // snapshot joins up with this rank's live stream.
                if let Some(proof) = awaiting_splice_proof.remove(&worker) {
                    if seq > proof.watermark + 1 {
                        warn!(
                            worker = ?worker,
                            peer_cursor = proof.watermark,
                            first_live_seq = seq,
                            "kv-bootstrap: sequence gap between snapshot and live stream; \
                             discarding snapshot state for this rank to avoid stale cache entries",
                        );
                        bootstrap.record_rank_outcome(RankOutcome::Gap);
                        bootstrap.set(&worker, BootstrapState::Failed);
                        tree.clear_worker(&worker);
                        cursors.lock().remove(&worker);
                        requeue_gapped_rank(&bootstrap, &bootstrap_tx, &worker);
                    } else {
                        // The deferred check passed: this rank's grafted state is
                        // now proven continuous with its live stream, which is the
                        // point at which it counts as warm.
                        bootstrap.record_rank_outcome(RankOutcome::Warm);
                    }
                }
                apply_batch(&tree, &cursors, &worker, seq, &batch);
            }
        }
    }
}

/// Apply one batch, honouring the cursor's out-of-order filter.
///
/// This is the single place tree deltas are written, whether the batch came
/// straight off the wire or out of a bootstrap hold-back queue — which is what
/// makes cursor seeding sufficient to reconcile a snapshot with the live stream.
/// The pump-owned handles both graft helpers need. Bundled so their signatures
/// stay readable as the state machine's gates accumulate.
struct PumpState<'a> {
    tree: &'a HashTree,
    cursors: &'a Mutex<HashMap<KvWorkerId, i64>>,
    bootstrap: &'a BootstrapTracker,
    live_workers: &'a Mutex<HashSet<KvWorkerId>>,
    /// Obligation queue, so the graft path can hand a gapped rank back for
    /// another sweep. See [`requeue_gapped_rank`].
    bootstrap_tx: &'a mpsc::Sender<ObligationBatch>,
}

fn apply_batch(
    tree: &HashTree,
    cursors: &Mutex<HashMap<KvWorkerId, i64>>,
    worker: &KvWorkerId,
    seq: i64,
    batch: &KvEventBatch,
) {
    if let Some(p) = cursors.lock().get(worker).copied() {
        if seq <= p {
            debug!(
                worker = ?worker,
                seq,
                last_applied = p,
                "kv-events pump: out-of-order batch; skipping",
            );
            return;
        }
    }
    for event in &batch.events {
        match event {
            KvCacheEvent::BlockStored(b) => {
                tree.insert(worker, b.parent_block_hash, &b.block_hashes);
            }
            KvCacheEvent::BlockRemoved(b) => {
                tree.remove(worker, &b.block_hashes);
            }
            KvCacheEvent::AllBlocksCleared => {
                tree.clear_worker(worker);
            }
        }
    }
    cursors.lock().insert(worker.clone(), seq);
}

/// Give up on bootstrapping `rank`: release whatever it held and mark it
/// [`BootstrapState::Failed`].
///
/// `discard_held` distinguishes the two reasons for giving up. On overflow the
/// queue has already lost its head, so replaying it would apply a
/// discontiguous prefix — drop it and resume from the live stream. Otherwise
/// the queue is intact and replaying it reproduces exactly today's
/// live-deltas-only behaviour.
fn fail_rank(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
    discard_held: bool,
    outcome: RankOutcome,
) {
    let (tree, cursors, bootstrap) = (st.tree, st.cursors, st.bootstrap);
    // Only transition ranks that are actually mid-bootstrap; an
    // AbandonBootstrap racing a successful ApplySnapshot must not undo it.
    //
    // This gate is also what keeps the rank tally once-per-rank: a second
    // attempt to fail an already-resolved rank returns before recording.
    if bootstrap.state_of(rank) != Some(BootstrapState::Pending) {
        held.remove(rank);
        return;
    }
    bootstrap.set(rank, BootstrapState::Failed);
    bootstrap.record_rank_outcome(outcome);
    // Nothing the snapshot contributed can be trusted for this rank.
    tree.clear_worker(rank);
    cursors.lock().remove(rank);
    let queue = held.remove(rank).unwrap_or_default();
    if discard_held {
        return;
    }
    for (seq, batch) in queue {
        apply_batch(tree, cursors, rank, seq, &batch);
    }
}

/// Ask the fleet whether `rank`'s publisher has moved past `watermark`, and post
/// the verdict back to the pump.
///
/// Any peer's cursor is admissible evidence: sequence numbers are the
/// publisher's, so a peer reporting one above our watermark proves a batch we
/// never received was emitted. A peer too cold to bootstrap from is still a
/// valid witness, which is why this reads the wire cursor directly instead of
/// vetting.
///
/// One caveat, pre-existing and unchanged by this probe: if the publisher reset
/// and we missed the reset event, a renumbered stream can report a cursor below
/// the old watermark, reading as `NoAdvance`. The `PublisherReset` arm handles
/// the case where the event does arrive.
///
/// Asks the peer for its cursor table alone, not a snapshot. The question is
/// whether the publisher's sequence EVER passed the watermark, which one integer
/// per rank answers completely; fetching a tree to read it made the proof cost
/// scale with the tree, so a fleet large enough to need bootstrap was also the
/// fleet that could not afford to prove it. Reading cursors live rather than from
/// a cached export also removes the false-continuous risk the cached path
/// carried, where a cursor stale by up to the producer TTL could miss
/// advancement that had just happened.
fn spawn_splice_probe(
    http: reqwest::Client,
    peers: Arc<PeerRegistry>,
    ctrl_tx: mpsc::Sender<PumpControl>,
    rank: KvWorkerId,
    watermark: i64,
    epoch: u64,
) {
    tokio::spawn(async move {
        let mut answered = false;
        let mut verdict = SpliceVerdict::Unknown;
        for peer in peers.candidates() {
            match fetch_cursors(&http, &peer).await {
                Ok(Some(snap)) => {
                    // A decodable body is NOT a witness: only a peer whose
                    // cursor table NAMES this rank has ever observed its
                    // publisher. Latching `answered` on the fetch alone would
                    // manufacture NoAdvance out of ignorance — a peer that
                    // never saw this rank cannot say its publisher has not
                    // moved, yet the verdict would resolve Warm as if
                    // continuity had been proven.
                    if let Some(seq) = snap.wire_cursor_for(&rank.url, rank.dp_rank) {
                        answered = true;
                        if seq > watermark {
                            warn!(
                                worker = ?rank,
                                peer = %peer,
                                watermark,
                                peer_cursor = seq,
                                "kv-bootstrap: a peer is past our unproven watermark, so a \
                                 batch we never received was published; discarding grafted \
                                 state for this rank",
                            );
                            verdict = SpliceVerdict::Advanced;
                            break;
                        }
                    }
                }
                // Non-200 is already logged inside fetch_body; a read failure
                // gets one line here, so a peer serving corrupt bodies is not
                // indistinguishable from a peer that never saw the rank. Either
                // way this peer is not a witness for this pass.
                Ok(None) => continue,
                Err(e) => {
                    debug!(
                        worker = ?rank,
                        peer = %peer,
                        error = %format_args!("{e:#}"),
                        "kv-bootstrap: splice probe could not read this peer",
                    );
                    continue;
                }
            }
        }
        if answered && verdict != SpliceVerdict::Advanced {
            debug!(
                worker = ?rank,
                watermark,
                "kv-bootstrap: no peer is past the watermark; treating the silent stream \
                 as continuous",
            );
            verdict = SpliceVerdict::NoAdvance;
        }
        if let Err(e) = ctrl_tx
            .send(PumpControl::SpliceProbe {
                rank,
                epoch,
                verdict,
            })
            .await
        {
            // Only the pump's shutdown closes the receiver, and a gone pump has
            // no proof state left to care about — but say so for forensics.
            debug!(
                error = %e,
                "kv-bootstrap: control channel closed; dropping probe verdict",
            );
        }
    });
}

/// Drop the grafted state of a rank whose splice was never proven.
///
/// Not [`fail_rank`]: that one only acts on a rank still
/// [`BootstrapState::Pending`], and this rank is `Recovered` — it was grafted,
/// it is serving, and the wait for evidence has run out. There is no held queue
/// to replay either, because reaching the deferred path required an empty one.
/// Hand a gap-discarded rank back for one more sweep.
///
/// A gap is the costliest failure: a snapshot was fetched, grafted, then thrown
/// away because the live stream did not join its watermark. A fresher snapshot
/// usually splices, and on a large fleet these were the ONLY remaining loss once
/// amplification and timeouts were fixed — 199 of 1512 ranks in a measured
/// rollout. The tracker caps this at one retry per rank.
///
/// Stamped with NOW and marked [`LateJoin::Refused`], which together are what
/// stop the retry being spent on state that cannot splice: the rank resumed
/// holding at this instant, so the sweep that takes it must ask for an export
/// beating that, and it will not be folded into a sweep already in flight whose
/// request predates it. Without both, a retry landing mid-sweep is adjudicated
/// against a snapshot taken before the gap, re-gaps by construction, and burns
/// the one attempt `gap_retried` allows.
fn requeue_gapped_rank(
    bootstrap: &BootstrapTracker,
    tx: &mpsc::Sender<ObligationBatch>,
    rank: &KvWorkerId,
) {
    let Some(obligation) = bootstrap.retry_after_gap(rank) else {
        return;
    };
    let batch = ObligationBatch {
        obligations: vec![obligation],
        holding_since: Instant::now(),
        late_join: LateJoin::Refused,
    };
    if let Err(e) = tx.try_send(batch) {
        // The rank is `Pending` now and nobody owns it, which would hold its
        // batches until the per-rank cap overflows. Put it back.
        warn!("kv-bootstrap: could not re-queue gapped rank ({e}); leaving it cold");
        bootstrap.set(rank, BootstrapState::Failed);
        return;
    }
    debug!(worker = ?rank, "kv-bootstrap: gapped rank re-queued for another sweep");
}

fn demote_unproven_rank(st: &PumpState<'_>, rank: &KvWorkerId, outcome: RankOutcome) {
    // A rank that has since been forgotten or re-registered is not ours to
    // demote; `Recovered` is the only state this can legitimately act on.
    if st.bootstrap.state_of(rank) != Some(BootstrapState::Recovered) {
        return;
    }
    st.bootstrap.set(rank, BootstrapState::Failed);
    st.bootstrap.record_rank_outcome(outcome);
    st.tree.clear_worker(rank);
    st.cursors.lock().remove(rank);
}

/// Graft a vetted snapshot, seed cursors, then release held batches.
///
/// Runs on the pump so it is the sole tree writer for the duration.
fn apply_snapshot(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    awaiting_splice_proof: &mut HashMap<KvWorkerId, PendingProof>,
    obligations: &[(KvWorkerId, u64)],
    mut vetted: VettedSnapshot,
) {
    let (tree, cursors, bootstrap, live_workers) =
        (st.tree, st.cursors, st.bootstrap, st.live_workers);
    // Ranks that left `Pending` while the fetch was in flight are already
    // applying live deltas; grafting older state beneath them is exactly the
    // stale splice this design refuses.
    // Derived from the ranks this bootstrap owns, NOT from the snapshot's worker
    // table: a rank the peer never mentioned still has to leave `Pending`, and it
    // does so below via the missing-cursor path.
    //
    // Three gates, each excluding a different way this message can be stale:
    // wrong incarnation (worker removed and re-added while the fetch was in
    // flight), no longer live (removed outright — every other tree write in the
    // pump checks this, so a graft must too), and no longer Pending (something
    // already resolved it).
    let live = live_workers.lock().clone();
    let pending: HashSet<KvWorkerId> = obligations
        .iter()
        .filter(|(w, epoch)| {
            bootstrap.epoch_of(w) == Some(*epoch)
                && live.contains(w)
                && bootstrap.state_of(w) == Some(BootstrapState::Pending)
        })
        .map(|(w, _)| w.clone())
        .collect();
    vetted.retain_workers(&pending);

    let node_count = vetted.node_count();
    if pending.is_empty() {
        debug!("kv-bootstrap: snapshot has no still-pending ranks to graft; discarding");
        return;
    }
    info!(
        ranks = pending.len(),
        nodes = node_count,
        "kv-bootstrap: grafting peer snapshot",
    );
    if let Err(e) = vetted.graft_into(tree) {
        warn!(
            error = %e,
            nodes = node_count,
            "kv-bootstrap: snapshot rejected by the tree; affected ranks will run cold",
        );
        for rank in pending {
            fail_rank(st, held, &rank, false, RankOutcome::TreeRejected);
        }
        return;
    }

    for rank in pending {
        // No cursor means the peer was not tracking this rank, so its tree
        // slice has no watermark to splice against — the rank is cold.
        let Some(peer_cursor) = vetted.cursor_for(&rank) else {
            debug!(
                worker = ?rank,
                "kv-bootstrap: peer had no cursor for this rank; running cold",
            );
            fail_rank(st, held, &rank, false, RankOutcome::Uncovered);
            continue;
        };

        // Splice check: the stream must continue from the snapshot's watermark.
        // A hole means a delta was lost (ZMQ drops at its HWM), and a lost
        // `BlockRemoved` would leave a permanent false cache hit — so bail to
        // cold rather than graft over a gap.
        //
        // The evidence may not exist yet: nothing says a batch has arrived for
        // this rank by now. When the queue is empty the check is deferred to
        // whichever batch lands first (see `awaiting_splice_proof`).
        let queue = held.remove(&rank).unwrap_or_default();
        // Whether the splice is proven RIGHT NOW, which decides when this rank
        // may be tallied warm. A held batch is evidence; an empty queue is not.
        let mut proven = false;
        match queue.front().map(|(seq, _)| *seq) {
            Some(first_seq) if first_seq > peer_cursor + 1 => {
                warn!(
                    worker = ?rank,
                    peer_cursor,
                    first_held_seq = first_seq,
                    "kv-bootstrap: sequence gap between snapshot and live stream; \
                     running cold to avoid stale cache entries",
                );
                // Put the queue back so `fail_rank` replays it after clearing.
                held.insert(rank.clone(), queue);
                fail_rank(st, held, &rank, false, RankOutcome::Gap);
                requeue_gapped_rank(st.bootstrap, st.bootstrap_tx, &rank);
                continue;
            }
            Some(_) => proven = true,
            None => {
                awaiting_splice_proof.insert(
                    rank.clone(),
                    PendingProof {
                        watermark: peer_cursor,
                        since: Instant::now(),
                        unknown_probes: 0,
                        probe_launched: None,
                    },
                );
            }
        }

        // Seed the watermark, never backwards: a cursor already ahead of the
        // peer's means our live stream has outrun the snapshot.
        {
            let mut guard = cursors.lock();
            let entry = guard.entry(rank.clone()).or_insert(peer_cursor);
            *entry = (*entry).max(peer_cursor);
        }
        bootstrap.set(&rank, BootstrapState::Recovered);
        // The seeded cursor filters whatever the snapshot already reflects.
        for (seq, batch) in queue {
            apply_batch(tree, cursors, &rank, seq, &batch);
        }
        // Tallied only once the splice is proven. When it is not yet, the
        // deferred check (or its expiry) records the verdict instead, so each
        // rank contributes exactly one count.
        if proven {
            bootstrap.record_rank_outcome(RankOutcome::Warm);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::engine_load::LoadStat;
    use crate::policies::kv_events::bootstrap::{CURSORS_ONLY_PARAM, SNAPSHOT_PATH};
    use crate::policies::kv_events::wire::{BlockRemoved, BlockStored, KvEventBatch};

    fn worker_id(url: &str, rank: u32) -> KvWorkerId {
        KvWorkerId {
            url: url.into(),
            dp_rank: rank,
        }
    }

    fn batch(events: Vec<KvCacheEvent>) -> KvEventBatch {
        KvEventBatch {
            ts: 0.0,
            events,
            attn_dp_rank: None,
        }
    }

    // ---- bootstrap fan-in (take_pending / drain_ready) ----

    fn obligation(n: u32) -> ObligationBatch {
        discovery_batch(n, Instant::now())
    }

    fn discovery_batch(n: u32, holding_since: Instant) -> ObligationBatch {
        ObligationBatch {
            obligations: vec![(worker_id(&format!("http://w{n}:30000"), 0), n as u64)],
            holding_since,
            late_join: LateJoin::Permitted,
        }
    }

    /// A discovery burst is taken in one go, so one fleet-wide snapshot serves
    /// every rank instead of one fetch per worker.
    #[tokio::test]
    async fn take_pending_takes_the_whole_queued_burst() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        for n in 0..8 {
            tx.send(obligation(n)).await.unwrap();
        }
        let got = take_pending(&mut rx, &cancel).await.expect("obligations");
        assert_eq!(got.obligations.len(), 8, "all eight are taken together");
    }

    /// Absorbing a burst must adopt the STRICTEST freshness in it: the sweep
    /// fetches once for all of them, so a snapshot older than the last rank to
    /// start holding cannot splice for that rank.
    #[tokio::test]
    async fn take_pending_adopts_the_newest_holding_instant() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        let newest = base + Duration::from_millis(300);
        tx.send(discovery_batch(0, base)).await.unwrap();
        tx.send(discovery_batch(1, newest)).await.unwrap();
        tx.send(discovery_batch(2, base + Duration::from_millis(100)))
            .await
            .unwrap();

        let got = take_pending(&mut rx, &cancel).await.expect("obligations");
        assert_eq!(got.obligations.len(), 3);
        assert_eq!(
            got.freshness_floor, newest,
            "the floor must be the newest holding instant, not the first seen",
        );
    }

    /// And it must not wait for a quiet period to do it — the sweep is the
    /// batching window, so adding latency here would be pure cost.
    #[tokio::test]
    async fn take_pending_does_not_wait() {
        let (tx, mut rx) = mpsc::channel(8);
        let cancel = CancellationToken::new();
        tx.send(obligation(1)).await.unwrap();
        let started = Instant::now();
        let got = take_pending(&mut rx, &cancel).await.expect("obligations");
        let elapsed = started.elapsed();
        assert_eq!(got.obligations.len(), 1);
        assert!(
            elapsed < Duration::from_millis(50),
            "returned in {elapsed:?}; must not debounce",
        );
    }

    /// The load-bearing behaviour: ranks discovered WHILE a sweep is in flight
    /// are merged into that sweep's delivery, so they ride its snapshot rather
    /// than waiting for a second fetch.
    #[tokio::test]
    async fn drain_ready_merges_arrivals_from_during_the_sweep() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        tx.send(obligation(0)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");
        assert_eq!(pending.obligations.len(), 1);

        // Stand in for the fetch: more workers show up while it is running.
        for n in 1..6 {
            tx.send(obligation(n)).await.unwrap();
        }
        let joined = drain_ready(&mut rx, &mut pending);
        assert_eq!(joined, 5, "late arrivals are reported");
        assert_eq!(
            pending.obligations.len(),
            6,
            "and merged into the same delivery",
        );
        assert!(
            pending.deferred.is_empty(),
            "discovery rides along rather than waiting for a sweep of its own",
        );
    }

    /// A mid-sweep merge must NOT retroactively tighten the floor: the request
    /// has already gone out under the old one, so claiming otherwise would
    /// describe a guarantee this sweep never asked for.
    #[tokio::test]
    async fn drain_ready_does_not_move_the_floor_the_sweep_asked_with() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        tx.send(discovery_batch(0, base)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");

        tx.send(discovery_batch(1, base + Duration::from_secs(5)))
            .await
            .unwrap();
        drain_ready(&mut rx, &mut pending);
        assert_eq!(
            pending.freshness_floor, base,
            "the floor is frozen once the sweep is in flight",
        );
    }

    /// A gap retry landing mid-sweep must not be spent on that sweep's snapshot.
    /// It was fetched under a floor older than the retry, so it re-gaps by
    /// construction — and `gap_retried` allows no third attempt.
    #[tokio::test]
    async fn drain_ready_defers_a_gap_retry_the_sweep_cannot_speak_for() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        tx.send(discovery_batch(0, base)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");

        let rank = worker_id("http://gapped:30000", 0);
        tx.send(ObligationBatch {
            obligations: vec![(rank.clone(), 7)],
            holding_since: base + Duration::from_millis(200),
            late_join: LateJoin::Refused,
        })
        .await
        .unwrap();

        let joined = drain_ready(&mut rx, &mut pending);
        assert_eq!(joined, 0, "a deferred batch is not counted as joined");
        assert_eq!(
            pending.obligations.len(),
            1,
            "and is not delivered into this sweep",
        );
        assert_eq!(pending.deferred.len(), 1, "it is handed to the next one");
        assert_eq!(pending.deferred[0].obligations[0].0, rank);
    }

    /// The refusal is about freshness, not about being a retry: a retry the
    /// sweep's own floor already covers has nothing to gain from waiting.
    #[tokio::test]
    async fn drain_ready_admits_a_retry_the_floor_already_covers() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        tx.send(discovery_batch(0, base)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");

        tx.send(ObligationBatch {
            obligations: vec![(worker_id("http://early:30000", 0), 7)],
            holding_since: base - Duration::from_millis(200),
            late_join: LateJoin::Refused,
        })
        .await
        .unwrap();

        assert_eq!(drain_ready(&mut rx, &mut pending), 1);
        assert!(pending.deferred.is_empty());
    }

    /// One snapshot fetch must never be able to consume the whole bootstrap
    /// deadline: the sweep awaits each candidate in turn, so a per-request timeout
    /// equal to the deadline lets the first unresponsive peer starve every other
    /// candidate and the replica boots cold with warm peers still unqueried.
    ///
    /// The deadlines below span the CONFIGURED DEFAULT upward; the default is
    /// the one case where the floor, not the divisor, shapes the answer — the
    /// value every unconfigured router actually runs with.
    #[test]
    fn snapshot_fetch_timeout_is_a_strict_fraction_of_every_nonzero_deadline() {
        use crate::policies::kv_events::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP as CAP;
        let default_secs = crate::config::types::default_bootstrap_timeout_ms() / 1_000;
        assert_eq!(default_secs, 5, "test's premise: the default budget is 5s");

        for deadline_secs in [1, 2, default_secs, 6, 10, 20, 30, 60, 120, 600] {
            let deadline = Duration::from_secs(deadline_secs);
            let per_fetch = snapshot_fetch_timeout(deadline, CAP);
            assert!(
                per_fetch < deadline,
                "a single fetch may not consume the whole {deadline_secs}s deadline \
                 (got {per_fetch:?})",
            );
            assert!(
                deadline.as_secs_f64() / per_fetch.as_secs_f64() >= 2.0,
                "{deadline_secs}s must buy at least two attempts; one fetch got {per_fetch:?}",
            );
        }

        // Above the floor the divisor governs, so the budget buys the full
        // attempt count rather than merely two.
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(120), CAP)
                * SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE,
            Duration::from_secs(120),
        );
        // A pre-settled (bootstrap-disabled) tracker reports a zero deadline, where
        // a derived value would be zero — an immediate expiry, not "no timeout".
        assert_eq!(
            snapshot_fetch_timeout(Duration::ZERO, CAP),
            SNAPSHOT_FETCH_TIMEOUT_FLOOR,
        );
    }

    /// The configurable cap binds a deadline that would otherwise derive
    /// above it, yields to storage-heavy fleets that need more per fetch,
    /// and never panics when misconfigured below the floor.
    #[test]
    fn snapshot_fetch_timeout_honours_the_configured_cap() {
        // A 10-minute deadline would derive 150s per fetch; the default cap
        // holds it at 30s, and a raised cap takes over exactly at its value.
        use crate::policies::kv_events::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP as CAP;
        assert_eq!(snapshot_fetch_timeout(Duration::from_secs(600), CAP), CAP);
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(600), Duration::from_secs(90)),
            Duration::from_secs(90),
            "the raised cap must not itself be exceeded",
        );
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(600), Duration::from_secs(400)),
            Duration::from_secs(150),
            "above the derivation the cap stops binding",
        );
        // A cap below the floor degrades to the cap itself rather than
        // panicking `clamp` — the CLI rejects this, but the math stays safe
        // for any path that skips it.
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(120), Duration::from_secs(2)),
            Duration::from_secs(2),
        );
        // And the default constant must not drift from the config default.
        assert_eq!(
            CAP,
            Duration::from_millis(crate::config::types::default_bootstrap_fetch_timeout_cap_ms()),
        );
    }

    /// The producer reuses its cached export only for a caller whose freshness
    /// requirement it actually meets. This is the whole point of the parameter:
    /// under a fixed TTL the peer happily served an export taken before the
    /// consumer subscribed, and the graft then gapped on the watermark.
    #[tokio::test]
    async fn producer_reuses_its_export_only_when_it_meets_the_callers_max_age() {
        let index = KvEventIndex::new();
        let exported_at = || async {
            index
                .snapshot_cache
                .lock()
                .await
                .as_ref()
                .expect("an entry was cached")
                .exported_at
        };

        index.peer_snapshot_body(Duration::from_secs(60)).await;
        let first = exported_at().await;

        // A caller that can live with a minute-old tree gets the same one back.
        index.peer_snapshot_body(Duration::from_secs(60)).await;
        assert_eq!(first, exported_at().await, "a met requirement reuses");

        // A caller that began holding after that export cannot use it.
        tokio::time::sleep(Duration::from_millis(5)).await;
        index.peer_snapshot_body(Duration::ZERO).await;
        assert!(
            exported_at().await > first,
            "an unmet requirement must rebuild, not serve the stale export",
        );
    }

    /// The stamp has to be taken BEFORE the contents are sampled. Stamping
    /// completion would claim coverage the document does not have — exactly the
    /// hole the parameter closes — and the walk plus encode of a real tree is
    /// long enough for that to matter.
    #[tokio::test]
    async fn producer_stamps_the_export_before_it_samples() {
        let index = KvEventIndex::new();
        let before = Instant::now();
        index.peer_snapshot_body(Duration::ZERO).await;
        let after = Instant::now();
        let exported_at = index
            .snapshot_cache
            .lock()
            .await
            .as_ref()
            .expect("an entry was cached")
            .exported_at;
        assert!(
            exported_at >= before && exported_at <= after,
            "the stamp must sit inside the build, at its start",
        );
    }

    /// The whole point: a probe wants one integer per rank, so this body must
    /// carry the cursors and NOT the tree. Asserting `nodes` is empty is also
    /// what keeps the body ungraftable — `from_wire` rejects an empty node list.
    #[tokio::test]
    async fn cursors_only_body_carries_cursors_and_no_nodes() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.report_worker("http://w1:30000", false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        let w = worker_id("http://w1:30000", 0);
        apply_batch(
            &index.tree,
            &index.cursors,
            &w,
            42,
            &batch(vec![KvCacheEvent::BlockStored(BlockStored {
                block_hashes: vec![111],
                parent_block_hash: None,
                token_ids: vec![],
                block_size: 256,
                lora_id: None,
                medium: None,
            })]),
        );

        let body = index.peer_cursors_body();
        let snap: PeerSnapshot = serde_json::from_slice(&body).expect("valid JSON");

        assert!(snap.nodes.is_empty(), "the tree must not be exported");
        assert_eq!(
            snap.wire_cursor_for("http://w1:30000", 0),
            Some(42),
            "the probe's one question must be answerable from this body",
        );
        assert_eq!(snap.block_size, 256);
        assert!(!snap.is_bigram);
        assert!(
            snap.producer_ready,
            "a settled replica holding nodes is a valid witness",
        );
    }

    /// The wire stamp for older consumers must be the majority-derived
    /// primary: a bigram-majority fleet stamps `is_bigram: true`, so an old
    /// router build that still vets the stamp agrees with this replica's
    /// query hashing.
    #[tokio::test]
    async fn snapshot_stamps_the_majority_derived_hashing_mode() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.report_worker("http://w1:30000", true);
        oracle.report_worker("http://w2:30000", true);
        oracle.report_worker("http://w3:30000", false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));

        let body = index.peer_cursors_body();
        let snap: PeerSnapshot = serde_json::from_slice(&body).expect("valid JSON");
        assert!(
            snap.is_bigram,
            "the stamp must track the majority-derived primary (2 bigram vs 1 unigram)",
        );
    }

    /// A replica with an empty tree must not claim to be worth believing, for the
    /// same reason `snapshot_entry` checks it: a replica whose own bootstrap timed
    /// out is settled while holding nothing.
    #[tokio::test]
    async fn cursors_only_body_reports_not_ready_with_an_empty_tree() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.report_worker("http://w1:30000", false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));

        let snap: PeerSnapshot =
            serde_json::from_slice(&index.peer_cursors_body()).expect("valid JSON");
        assert!(!snap.producer_ready, "an empty tree is not a source");
    }

    /// The cost claim, asserted structurally rather than by timing: serving cursors
    /// must leave the snapshot cache untouched, because populating it is the
    /// expensive walk this path exists to avoid.
    #[tokio::test]
    async fn cursors_only_body_never_populates_the_snapshot_cache() {
        let index = KvEventIndex::new();
        index.peer_cursors_body();
        assert!(
            index.snapshot_cache.lock().await.is_none(),
            "cursors-only must not walk or cache the tree",
        );
    }

    /// A launched probe must block relaunch inside its timeout window —
    /// otherwise one slower than the timeout accumulates one duplicate per
    /// sweep, each re-asking every peer — but it must NOT block past it: a
    /// verdict that never lands would freeze the rank in Recovered forever.
    /// Both halves live in one predicate, [`PendingProof::due_for_probe`],
    /// which the pump's sweep filter shares, so this pins the production
    /// guard rather than a copy of it.
    #[test]
    fn a_launched_probe_blocks_relaunch_until_it_is_presumed_lost() {
        let past = Instant::now() - SPLICE_PROOF_TIMEOUT - Duration::from_secs(1);
        let mut proof = PendingProof {
            watermark: 10,
            since: Instant::now(),
            unknown_probes: 0,
            probe_launched: Some(Instant::now()),
        };
        assert!(
            !proof.due_for_probe(SPLICE_PROOF_TIMEOUT),
            "an outstanding probe still inside its timeout must not be relaunched",
        );

        // The defining state of the guard: graft-age EXPIRED, probe FRESH.
        // Production arms both clocks together (see `launch_probe`), so this
        // combination is not a reachable state — pinned because only the
        // Some-arm's precedence over `since` makes it hold, which is exactly
        // what a simplification back to a since-only predicate would lose.
        proof.since = past;
        assert!(
            !proof.due_for_probe(SPLICE_PROOF_TIMEOUT),
            "an expired graft with a fresh outstanding probe still waits",
        );

        proof.probe_launched = Some(past);
        assert!(
            proof.due_for_probe(SPLICE_PROOF_TIMEOUT),
            "a probe older than the timeout is presumed lost and asked again",
        );

        proof.probe_launched = None;
        assert!(
            proof.due_for_probe(SPLICE_PROOF_TIMEOUT),
            "once the verdict clears the probe, the graft's own age makes it due",
        );
    }

    /// The two producer answers are intentionally NOT set-equal. A rank that
    /// observed a publisher and then lost the blocks (cleared here) keeps its
    /// cursor in the cursors-only body — that observation is still a valid
    /// witness — while the full export drops it, because its cursor table only
    /// covers ranks carrying tree nodes. The full side uses a SECOND rank that
    /// still carries a block, so the export is a real one (not the not-ready
    /// body an empty tree would fetch) and the assertion genuinely pins the
    /// carrier filtering. See [`KvEventIndex::peer_cursors_body`].
    #[tokio::test]
    async fn the_cursors_only_table_keeps_witnesses_the_full_export_loses() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.report_worker("http://w1:30000", false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        let cleared = worker_id("http://w1:30000", 0);
        let holding = worker_id("http://w2:30000", 0);
        index.seed_stored_block_for_test(&holding, 7, 999);
        index.seed_stored_block_for_test(&cleared, 42, 111);
        // Observed-then-cleared: the cursor survives where the blocks do not.
        apply_batch(
            &index.tree,
            &index.cursors,
            &cleared,
            43,
            &batch(vec![KvCacheEvent::AllBlocksCleared]),
        );

        let thin: PeerSnapshot =
            serde_json::from_slice(&index.peer_cursors_body()).expect("valid JSON");
        assert_eq!(
            thin.wire_cursor_for("http://w1:30000", 0),
            Some(43),
            "a cleared rank is still a witness to its publisher's stream",
        );

        let full: PeerSnapshot =
            serde_json::from_slice(&index.peer_snapshot_body(Duration::ZERO).await)
                .expect("valid JSON");
        assert!(
            !full.nodes.is_empty(),
            "the held block keeps this a real export, not the not-ready body",
        );
        assert_eq!(
            full.wire_cursor_for("http://w1:30000", 0),
            None,
            "carriers only: the cleared rank's cursor is dropped from the export",
        );
        assert_eq!(
            full.wire_cursor_for("http://w2:30000", 0),
            Some(7),
            "the rank still carrying blocks keeps its cursor in the export",
        );
    }

    /// A witness body: no nodes, just a cursor table. Structurally it is what
    /// the cursors-only producer serves, which is all the probe ever reads.
    fn witness_snapshot(entries: &[(&str, u32, i64)]) -> PeerSnapshot {
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: false,
            producer_ready: false,
            workers: entries
                .iter()
                .map(|(url, dp_rank, _)| WireWorker {
                    url: (*url).to_string(),
                    dp_rank: *dp_rank,
                })
                .collect(),
            cursors: entries
                .iter()
                .enumerate()
                .map(|(i, (_, _, seq))| (i as u32, *seq))
                .collect(),
            nodes: Vec::new(),
        }
    }

    /// Spin an axum `/server_info` and return `(base_url, hits, release)`:
    /// `hits` ticks once per request, and `body["status"]` steers the reply —
    /// `503` answers 503 with an empty body (a retriable failure), `"hold"`
    /// parks the reply until `release.notify_waiters()`, anything else
    /// answers 200 with the body (steering key removed).
    async fn serve_server_info(
        mut body: serde_json::Value,
    ) -> (String, mpsc::Receiver<()>, Arc<tokio::sync::Notify>) {
        let (hits_tx, hits_rx) = mpsc::channel(8);
        let release = Arc::new(tokio::sync::Notify::new());
        let release_in_handler = Arc::clone(&release);
        let status = body
            .as_object_mut()
            .expect("server_info body is a JSON object")
            .remove("status")
            .unwrap_or(serde_json::json!(200));
        let hold = status == "hold";
        let retriable = status == 503;
        let app = axum::Router::new().route(
            "/server_info",
            axum::routing::get(move || {
                let hits_tx = hits_tx.clone();
                let release = Arc::clone(&release_in_handler);
                let body = body.clone();
                async move {
                    let _ = hits_tx.send(()).await;
                    if hold {
                        release.notified().await;
                    }
                    let status = if retriable {
                        axum::http::StatusCode::SERVICE_UNAVAILABLE
                    } else {
                        axum::http::StatusCode::OK
                    };
                    (status, axum::Json(body))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), hits_rx, release)
    }

    /// Serve one canned body on the real snapshot path, recording the query
    /// strings it is sent: what the probe puts on the wire is part of the
    /// contract, since the producer only skips its tree when actually ASKED.
    async fn serve_snapshot_recording_queries(
        snap: PeerSnapshot,
    ) -> (String, Arc<std::sync::Mutex<Vec<Option<String>>>>) {
        let queries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&queries);
        let app = axum::Router::new().route(
            SNAPSHOT_PATH,
            axum::routing::get(move |uri: axum::http::Uri| {
                let seen = Arc::clone(&seen);
                let snap = snap.clone();
                async move {
                    seen.lock()
                        .expect("queries lock")
                        .push(uri.query().map(str::to_string));
                    axum::Json(snap)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), queries)
    }

    /// Run one probe against a one-peer fleet serving `snap`, returning the
    /// verdict the probe posts to the pump-control channel — the same path a
    /// verdict takes in production. Asserts along the way that every request
    /// asked for the cursor table alone.
    async fn probe_once(snap: PeerSnapshot, probed: &KvWorkerId, watermark: i64) -> SpliceVerdict {
        let (base, queries) = serve_snapshot_recording_queries(snap).await;
        let peers = Arc::new(PeerRegistry::new());
        peers.replace(vec![base]);
        let (tx, mut rx) = mpsc::channel(1);
        spawn_splice_probe(
            reqwest::Client::new(),
            peers,
            tx,
            probed.clone(),
            watermark,
            0,
        );
        let outcome = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("a live one-peer fleet always gets an answer")
            .expect("the control channel outlives the probe");
        assert!(
            queries.lock().expect("queries lock").iter().all(|q| q
                .as_deref()
                .unwrap_or_default()
                .contains(CURSORS_ONLY_PARAM)),
            "every probe request must ask for the cursor table alone",
        );
        match outcome {
            PumpControl::SpliceProbe { verdict, .. } => verdict,
            other => panic!("expected a splice-probe verdict, got {other:?}"),
        }
    }

    /// Pin for the decodable-body-is-not-a-witness guard documented at
    /// [`spawn_splice_probe`]: a peer whose cursor table does not NAME the
    /// probed rank has never observed its publisher, and its body alone must
    /// not resolve the verdict to NoAdvance.
    #[tokio::test]
    async fn probe_ignores_a_body_that_does_not_name_the_rank() {
        let snap = witness_snapshot(&[("http://other:30000", 0, 999)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Unknown,
        );
    }

    #[tokio::test]
    async fn probe_reports_no_advance_when_the_best_witness_is_at_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 10)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::NoAdvance,
        );
    }

    #[tokio::test]
    async fn probe_reports_advance_when_a_witness_is_past_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 11)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Advanced,
        );
    }

    /// A worker discovered after readiness opened must still be allowed to warm.
    /// Gating registration on `settled()` denied it silently: the replica served a
    /// tree missing that worker's blocks with no `bootstrap_state` entry to show
    /// for it.
    #[test]
    fn enabled_survives_settling_so_late_workers_still_bootstrap() {
        let tracker = BootstrapTracker::new(Duration::from_millis(1));
        let id = worker_id("http://w1:30000", 0);
        tracker.register(std::slice::from_ref(&id));
        std::thread::sleep(Duration::from_millis(5));
        assert!(tracker.settled(), "deadline expiry settles readiness");
        assert!(
            tracker.enabled(),
            "settling must NOT disable bootstrap for workers found later",
        );
    }

    /// The disabled case must stay distinguishable from the finished case, or a
    /// router with no `--kv-peer-selector` would start holding batches.
    #[test]
    fn disabled_tracker_is_not_enabled() {
        let t = BootstrapTracker::disabled();
        assert!(t.settled());
        assert!(
            !t.enabled(),
            "no selector configured ⇒ never register ranks"
        );
    }

    /// The premise behind `BootstrapDeps::deadline`'s settled branch: the
    /// remaining window saturates at zero, so a late rank handed
    /// `time_remaining()` would get no budget at all and abandon instantly.
    #[test]
    fn time_remaining_saturates_to_zero_once_expired() {
        let tracker = BootstrapTracker::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            tracker.time_remaining(),
            Some(Duration::ZERO),
            "expired window must read as zero, not as None",
        );
        assert!(
            tracker.timeout() > Duration::ZERO,
            "so the settled branch has a real budget to fall back on",
        );
    }

    /// The retry is capped at once per rank. Without the cap, a rank no peer
    /// covers would buy a fresh fleet-wide fetch on every pass, reintroducing the
    /// amplification the coordinator exists to remove.
    #[test]
    fn requeue_is_capped_at_one_retry_per_rank() {
        let mut requeued: HashSet<KvWorkerId> = HashSet::new();
        let rank = worker_id("http://w1:30000", 0);

        // First time an uncovered rank is seen: eligible for a second sweep.
        assert!(
            requeued.insert(rank.clone()),
            "first sighting must be retried",
        );
        // Second time: must be delivered as-is (resolving to Uncovered) rather
        // than triggering another fetch.
        assert!(
            !requeued.insert(rank.clone()),
            "second sighting must NOT buy another fetch",
        );
    }

    /// Nothing queued means nothing added, and no spinning.
    #[tokio::test]
    async fn drain_ready_on_empty_queue_adds_nothing() {
        let (_tx, mut rx) = mpsc::channel::<ObligationBatch>(4);
        let mut pending = PendingSweep::from(obligation(0));
        assert_eq!(drain_ready(&mut rx, &mut pending), 0);
        assert_eq!(pending.obligations.len(), 1);
    }

    /// Cancellation while idle ends the coordinator rather than parking forever.
    #[tokio::test]
    async fn take_pending_stops_on_cancel() {
        let (_tx, mut rx) = mpsc::channel::<ObligationBatch>(4);
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(
            take_pending(&mut rx, &cancel).await.is_none(),
            "cancelled coordinator stops",
        );
    }

    /// All senders dropped with nothing pending ends the coordinator too.
    #[tokio::test]
    async fn take_pending_stops_when_senders_drop() {
        let (tx, mut rx) = mpsc::channel::<ObligationBatch>(4);
        let cancel = CancellationToken::new();
        drop(tx);
        assert!(
            take_pending(&mut rx, &cancel).await.is_none(),
            "closed channel stops the coordinator",
        );
    }

    /// Bundle of plumbing returned by `spawn_pump` so individual tests
    /// can destructure just the bits they need.
    struct PumpHarness {
        tree: Arc<HashTree>,
        engine_load: Arc<EngineLoadTable>,
        cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
        #[allow(dead_code)]
        live_set: Arc<Mutex<HashSet<KvWorkerId>>>,
        #[allow(dead_code)]
        cancel: CancellationToken,
        tx: mpsc::Sender<WorkerEvent>,
        pump: JoinHandle<()>,
        ctrl_tx: mpsc::Sender<PumpControl>,
        /// Obligations the pump handed back, e.g. a gap-driven retry.
        bootstrap_rx: mpsc::Receiver<ObligationBatch>,
        /// The vote registry the pump re-introspects into on `PublisherReset`.
        /// Read only by tests that drive that path.
        #[allow(dead_code)]
        oracle: Arc<BlockSizeOracle>,
    }

    /// Build a tree + cursors + live-set wired through `pump_loop` with
    /// the given workers pre-marked live. Bootstrap is pre-settled, so batches
    /// are applied immediately — the behaviour every pre-bootstrap test asserts.
    fn spawn_pump(live: &[KvWorkerId]) -> PumpHarness {
        spawn_pump_with_bootstrap(live, Arc::new(BootstrapTracker::disabled()))
    }

    /// As `spawn_pump`, but with a caller-supplied tracker so bootstrap
    /// hold-back, splicing, and abandonment can be driven directly.
    fn spawn_pump_with_bootstrap(
        live: &[KvWorkerId],
        bootstrap: Arc<BootstrapTracker>,
    ) -> PumpHarness {
        let tree = Arc::new(HashTree::new());
        let engine_load = EngineLoadTable::new();
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let live_set: Arc<Mutex<HashSet<KvWorkerId>>> =
            Arc::new(Mutex::new(live.iter().cloned().collect()));
        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::channel(4);
        let (ctrl_tx, ctrl_rx) = mpsc::channel(4);
        // Real queue so a gap-driven re-queue is observable rather than dropped.
        let (bootstrap_tx, bootstrap_rx) = mpsc::channel(16);
        let oracle = BlockSizeOracle::new();
        let pump = tokio::spawn(pump_loop(
            PumpDeps {
                tree: tree.clone(),
                engine_load: engine_load.clone(),
                cursors: cursors.clone(),
                live_workers: live_set.clone(),
                bootstrap: bootstrap.clone(),
                oracle: Arc::clone(&oracle),
                http: reqwest::Client::new(),
                // Empty peer set: a splice probe finds no witness and returns
                // `Unknown`, so these tests exercise the pump's own gates without
                // any network. Probe verdicts are driven directly instead.
                peers: Arc::new(PeerRegistry::new()),
                snapshot_http: reqwest::Client::new(),
                bootstrap_tx: bootstrap_tx.clone(),
                ctrl_tx: ctrl_tx.clone(),
            },
            cancel.clone(),
            rx,
            ctrl_rx,
        ));
        PumpHarness {
            tree,
            engine_load,
            cursors,
            live_set,
            cancel,
            tx,
            pump,
            ctrl_tx,
            bootstrap_rx,
            oracle,
        }
    }

    // -----------------------------------------------------------------------
    // Bootstrap hold-back / splice / bail-to-cold
    // -----------------------------------------------------------------------

    fn stored(parent: Option<i64>, hashes: Vec<i64>) -> KvCacheEvent {
        KvCacheEvent::BlockStored(BlockStored {
            parent_block_hash: parent,
            block_hashes: hashes,
            token_ids: vec![],
            block_size: 64,
            lora_id: None,
            medium: None,
        })
    }

    /// The obligation set for `ranks` as registered in `tracker`, mirroring what
    /// `add_worker` hands to `spawn_bootstrap`.
    fn obligations(tracker: &BootstrapTracker, ranks: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        ranks
            .iter()
            .map(|r| {
                (
                    r.clone(),
                    tracker.epoch_of(r).expect("rank must be registered"),
                )
            })
            .collect()
    }

    /// A tracker with one rank registered and a deadline far enough out that it
    /// never fires mid-test.
    fn pending_tracker(ids: &[KvWorkerId]) -> Arc<BootstrapTracker> {
        let t = Arc::new(BootstrapTracker::new(Duration::from_secs(3600)));
        t.register(ids);
        t
    }

    /// A snapshot carrying chain [100, 200] for `id`, watermarked at `cursor`.
    fn vetted_for(id: &KvWorkerId, cursor: i64) -> VettedSnapshot {
        vetted_for_workers(&[id], cursor)
    }

    /// As `vetted_for`, but every node carries every listed worker — for the
    /// tests that need one snapshot to span several ranks.
    fn vetted_for_workers(ids: &[&KvWorkerId], cursor: i64) -> VettedSnapshot {
        use crate::policies::kv_events::tree::SnapshotNode;
        let carriers: Vec<u32> = (0..ids.len() as u32).collect();
        VettedSnapshot::from_parts_for_test(
            ids.iter().map(|id| (*id).clone()).collect(),
            vec![
                SnapshotNode {
                    parent: None,
                    block_hash: 100,
                    workers: carriers.clone(),
                },
                SnapshotNode {
                    parent: Some(0),
                    block_hash: 200,
                    workers: carriers,
                },
            ],
            ids.iter().map(|id| ((*id).clone(), cursor)).collect(),
            0,
        )
    }

    /// While a rank is `Pending` its deltas must not reach the tree — applying
    /// them first would let a stale snapshot land on top of newer state.
    #[tokio::test]
    async fn pump_holds_batches_while_rank_pending() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump_with_bootstrap(
            std::slice::from_ref(&id),
            pending_tracker(std::slice::from_ref(&id)),
        );

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 6,
            batch: batch(vec![stored(None, vec![10, 20])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            h.tree.match_prefix(None, &[10, 20]).matched_blocks,
            0,
            "a held batch must not be applied",
        );
        assert!(h.cursors.lock().is_empty(), "no cursor for a held rank");
    }

    /// The splice: snapshot grafts, cursor seeds, then held deltas replay on
    /// top and the rank reports Recovered.
    #[tokio::test]
    async fn pump_splices_snapshot_then_drains_held_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Live delta continues exactly where the snapshot's watermark stops.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 6,
            batch: batch(vec![stored(Some(200), vec![300])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // Snapshot state present.
        assert_eq!(h.tree.match_prefix(None, &[100, 200]).matched_blocks, 2);
        // Held delta applied on top of it.
        let m = h.tree.match_prefix(None, &[100, 200, 300]);
        assert_eq!(m.matched_blocks, 3, "held batch must extend the snapshot");
        assert!(m.workers.contains(&id));
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(6));
        assert!(tracker.settled());
    }

    /// A held batch the snapshot already reflects is filtered by the seeded
    /// cursor — the reuse that makes cursor seeding sufficient to reconcile.
    #[tokio::test]
    async fn pump_seeded_cursor_filters_already_reflected_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // seq 5 EQUALS the snapshot's watermark: a redelivery of a batch the graft
        // already reflects. Equality is the boundary — `seq <= p` vs `seq < p` —
        // and re-applying it would re-insert blocks the producer may since have
        // removed.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 5,
            batch: batch(vec![stored(None, vec![777])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            h.tree.match_prefix(None, &[777]).matched_blocks,
            0,
            "a batch the snapshot already reflects must be filtered",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(5));
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
    }

    /// A hole between the snapshot watermark and the live stream means a delta
    /// was lost. Grafting anyway could leave a permanently stale entry, so the
    /// rank drops to cold: snapshot state cleared, live deltas still applied.
    ///
    /// This is the *deferred* path — the snapshot is grafted before any batch
    /// for the rank has arrived, so continuity can only be judged later. The
    /// pump's biased select drains the control channel first, which makes this
    /// the ordering that occurs naturally.
    #[tokio::test]
    async fn pump_detects_deferred_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Watermark 5, but the first live batch is seq 9 — 6..8 were lost.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // Gapped ranks are handed back for one retry, so the rank is Pending
        // rather than terminally Failed — the discarded graft is the same either
        // way.
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        // The live stream still applies: cold, not broken.
        assert!(h.tree.match_prefix(None, &[42]).workers.contains(&id));
        // Readiness now WAITS: the gapped rank went back to Pending for its retry,
        // so the tracker is unsettled until that resolves or the deadline expires.
        // Warming the tree is preferred over opening `/readyz` on a cold rank.
        assert!(!tracker.settled());
    }

    /// A gap is the costliest failure — a snapshot was fetched, grafted, then
    /// thrown away. So the rank is handed back for one more sweep instead of
    /// staying cold with budget unspent, and the retry is capped at one.
    #[tokio::test]
    async fn pump_requeues_a_gapped_rank_once() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Watermark 5, first live batch seq 9 — 6..8 lost, so this gaps.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        let requeued = h.bootstrap_rx.try_recv().expect("gapped rank re-queued");
        assert_eq!(requeued.obligations.len(), 1);
        assert_eq!(
            requeued.obligations[0].0, id,
            "the gapped rank itself is handed back",
        );
        assert_eq!(
            requeued.late_join,
            LateJoin::Refused,
            "a retry must not be spent on a snapshot fetched before the gap",
        );
        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "back to Pending so the next sweep may graft onto it",
        );

        // Capped: a second gap on the same rank must not buy another fetch.
        tracker.set(&id, BootstrapState::Failed);
        assert!(
            tracker.retry_after_gap(&id).is_none(),
            "one retry per rank, or a persistently gapping rank re-fetches forever",
        );
    }

    /// Same gap, detected on the *immediate* path: the batch is already held
    /// when the snapshot arrives, so the watermark can be checked at graft time.
    #[tokio::test]
    async fn pump_detects_held_queue_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        // Let the pump receive and hold the batch before the snapshot lands.
        // The pump's biased select would otherwise take the control message
        // first, which is the deferred path covered by the test above.
        tokio::time::sleep(Duration::from_millis(50)).await;

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        assert!(h.tree.match_prefix(None, &[42]).workers.contains(&id));
    }

    /// The gap boundary, both directions, on both the deferred and the held path.
    ///
    /// A one-batch hole is the likeliest ZMQ-HWM drop, and it is exactly the case
    /// an off-by-one in `seq > watermark + 1` would wave through. Previously both
    /// gap tests used a three-wide gap, so the boundary was pinned on one side
    /// only and `+ 1` -> `+ 2` survived mutation.
    #[tokio::test]
    async fn pump_gap_boundary_is_exact() {
        // watermark 5: seq 6 continues cleanly, seq 7 means batch 6 was lost.
        for (first_seq, expect_kept) in [(6i64, true), (7i64, false)] {
            let id = worker_id("http://w1", 0);
            let tracker = pending_tracker(std::slice::from_ref(&id));
            let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

            h.ctrl_tx
                .send(PumpControl::ApplySnapshot {
                    obligations: obligations(&tracker, std::slice::from_ref(&id)),
                    vetted: Box::new(vetted_for(&id, 5)),
                })
                .await
                .unwrap();
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq: first_seq,
                batch: batch(vec![stored(None, vec![9_000 + first_seq])]),
            })
            .await
            .unwrap();
            drop(h.tx);
            drop(h.ctrl_tx);
            h.pump.await.unwrap();

            let kept = h.tree.match_prefix(None, &[100, 200]).workers.contains(&id);
            assert_eq!(
                kept, expect_kept,
                "deferred path, watermark 5, first live seq {first_seq}",
            );
        }

        // Same boundary on the immediate (already-held) path.
        for (first_seq, expect_kept) in [(6i64, true), (7i64, false)] {
            let id = worker_id("http://w1", 0);
            let tracker = pending_tracker(std::slice::from_ref(&id));
            let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq: first_seq,
                batch: batch(vec![stored(None, vec![8_000 + first_seq])]),
            })
            .await
            .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
            h.ctrl_tx
                .send(PumpControl::ApplySnapshot {
                    obligations: obligations(&tracker, std::slice::from_ref(&id)),
                    vetted: Box::new(vetted_for(&id, 5)),
                })
                .await
                .unwrap();
            drop(h.tx);
            drop(h.ctrl_tx);
            h.pump.await.unwrap();

            let kept = h.tree.match_prefix(None, &[100, 200]).workers.contains(&id);
            assert_eq!(
                kept, expect_kept,
                "held path, watermark 5, first held seq {first_seq}",
            );
        }
    }

    /// A contiguous live batch after a graft must NOT be mistaken for a gap.
    #[tokio::test]
    async fn pump_contiguous_batch_after_graft_keeps_snapshot() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        // seq 6 continues directly from watermark 5.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 6,
            batch: batch(vec![stored(Some(200), vec![300])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        let m = h.tree.match_prefix(None, &[100, 200, 300]);
        assert_eq!(
            m.matched_blocks, 3,
            "snapshot must survive a contiguous delta"
        );
        assert!(m.workers.contains(&id));
    }

    /// No peer could supply a snapshot: held deltas must still be released, or
    /// the rank would buffer forever and never settle.
    #[tokio::test]
    async fn pump_abandon_bootstrap_releases_held_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: batch(vec![stored(None, vec![55, 66])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        let m = h.tree.match_prefix(None, &[55, 66]);
        assert_eq!(
            m.matched_blocks, 2,
            "held batches must be released, not lost"
        );
        assert!(m.workers.contains(&id));
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(3));
        assert!(tracker.settled());
    }

    /// Regression: a rank the peer's snapshot never mentions must still leave
    /// `Pending`.
    ///
    /// `ApplySnapshot` used to derive its work set from the snapshot's worker
    /// table, so a registered rank the peer knew nothing about (peer had not
    /// discovered that worker, or vetting dropped it as unknown) stayed `Pending`
    /// forever — buffering every batch, reporting Pending in the gauge, and
    /// routing cache-blind indefinitely.
    #[tokio::test]
    async fn pump_snapshot_fails_ranks_the_peer_did_not_cover() {
        let covered = worker_id("http://w1", 0);
        let uncovered = worker_id("http://w2", 0);
        let tracker = pending_tracker(&[covered.clone(), uncovered.clone()]);
        let h = spawn_pump_with_bootstrap(&[covered.clone(), uncovered.clone()], tracker.clone());

        // The uncovered rank has a batch held back; it must be released.
        h.tx.send(WorkerEvent::Batch {
            worker: uncovered.clone(),
            seq: 3,
            batch: batch(vec![stored(None, vec![555])]),
        })
        .await
        .unwrap();
        // Snapshot covers only `covered`, but the bootstrap owns both ranks.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, &[covered.clone(), uncovered.clone()]),
                vetted: Box::new(vetted_for(&covered, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&covered), Some(BootstrapState::Recovered));
        assert_eq!(
            tracker.state_of(&uncovered),
            Some(BootstrapState::Failed),
            "a rank absent from the snapshot must not stay Pending",
        );
        // Its held batch is released rather than stranded.
        assert!(h
            .tree
            .match_prefix(None, &[555])
            .workers
            .contains(&uncovered));
        assert!(tracker.settled(), "both ranks terminal ⇒ readiness opens");
    }

    /// Regression: a bootstrap task from a PREVIOUS incarnation must not graft.
    ///
    /// `spawn_bootstrap` runs detached for up to the whole deadline. If the
    /// worker is removed and re-added inside that window the rank is `Pending`
    /// again, so without an incarnation check the in-flight snapshot would be
    /// grafted onto the new incarnation and seed a watermark from the OLD
    /// publisher's numbering — after which every batch from the fresh publisher
    /// (restarting at seq 1) is filtered as out-of-order and the rank sits on
    /// stale state forever while reporting Recovered.
    #[tokio::test]
    async fn pump_snapshot_from_a_stale_incarnation_is_discarded() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let stale = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // The worker goes away and comes back: a new incarnation, Pending again.
        tracker.forget(std::slice::from_ref(&id));
        let fresh = tracker.register(std::slice::from_ref(&id));
        assert_ne!(stale[0].1, fresh[0].1, "test needs a new incarnation");

        // The old task's result finally arrives.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: stale,
                vetted: Box::new(vetted_for(&id, 5000)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            !h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "a previous incarnation's snapshot must not be grafted",
        );
        assert_eq!(
            h.cursors.lock().get(&id).copied(),
            None,
            "no stale watermark may be seeded, or the fresh publisher is filtered out",
        );
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
    }

    /// A graft must respect `live_workers` like every other tree write in the
    /// pump: a worker removed while the fetch was in flight must not get carriers.
    #[tokio::test]
    async fn pump_snapshot_skips_ranks_no_longer_live() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let obs = obligations(&tracker, std::slice::from_ref(&id));
        // Spawned with an EMPTY live set: the worker is gone.
        let h = spawn_pump_with_bootstrap(&[], tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs,
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            !h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "a deregistered worker must not gain carriers from a graft",
        );
        assert!(h.cursors.lock().is_empty());
    }

    /// A snapshot that arrives after its rank already went Failed must not be
    /// grafted under the live stream it lost the race to.
    #[tokio::test]
    async fn pump_snapshot_skips_ranks_no_longer_pending() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
            })
            .await
            .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(
            !h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "a late snapshot must not graft onto an already-live rank",
        );
    }

    /// Overflowing the hold-back queue leaves a hole no snapshot can splice
    /// across, so bootstrap is abandoned and the rank resumes live.
    #[tokio::test]
    async fn pump_overflowing_held_queue_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        for seq in 1..=(PENDING_BATCH_LIMIT as i64 + 2) {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        // The batch that tripped the limit, and everything after it, applies
        // directly; the discarded prefix does not.
        let last = PENDING_BATCH_LIMIT as i64 + 2;
        assert!(
            h.tree.match_prefix(None, &[last]).workers.contains(&id),
            "post-overflow batches must apply live",
        );
        assert_eq!(
            h.tree.match_prefix(None, &[1]).matched_blocks,
            0,
            "the discarded prefix must NOT be applied — replaying across a known \
             hole is what `discard_held` exists to prevent",
        );
        assert!(tracker.settled());
    }

    // -----------------------------------------------------------------------
    // Bootstrap state-machine contract, as an executable property
    //
    // WHY this exists rather than more example tests: every bug found in this
    // state machine so far has been a violation of one of two properties, not a
    // wrong value in a specific scenario. Round after round of scenario review
    // kept finding new interleavings; a randomised driver over the real pump
    // covers the interleavings directly and fails on the property instead.
    //
    // The contract:
    //   T (terminality) — once the bootstrap for a rank is resolved, that rank
    //     is never left `Pending`. A `Pending` rank buffers its events forever
    //     and holds the readiness gate, so this is the load-bearing property.
    //   L (latch) — `settled()` never transitions true -> false. Violating it
    //     turns a routine scale-up into a 503 on a serving replica.
    //
    // Cursor monotonicity is deliberately NOT asserted here: `fail_rank`
    // legitimately REMOVES a cursor, so a later lower value is correct after a
    // bail-to-cold. It is covered by the targeted tests above instead.
    // -----------------------------------------------------------------------

    /// Randomised interleavings of every control-plane and data-plane event,
    /// checked against the contract above.
    #[tokio::test]
    async fn property_bootstrap_contract_holds_under_random_interleavings() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        for seed in 0..96u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let n_workers = rng.gen_range(1..=3);
            let ranks: Vec<KvWorkerId> = (0..n_workers)
                .flat_map(|w| {
                    let dp = rng.gen_range(1..=2);
                    (0..dp)
                        .map(move |r| worker_id(&format!("http://w{w}"), r))
                        .collect::<Vec<_>>()
                })
                .collect();

            let tracker = pending_tracker(&ranks);
            let h = spawn_pump_with_bootstrap(&ranks, tracker.clone());

            let mut latched = false;
            let mut next_seq: HashMap<KvWorkerId, i64> = HashMap::new();
            // Ranks named in some control message's obligation set. Only these
            // are required to be terminal at the end — a rank nobody ever
            // resolved is legitimately still Pending until the deadline.
            let mut discharged: HashSet<KvWorkerId> = HashSet::new();
            // Set when a scale-up rank is registered, which legitimately leaves
            // an undischarged Pending entry behind.
            let mut registered_extra = false;

            for _ in 0..rng.gen_range(4..40) {
                let rank = ranks[rng.gen_range(0..ranks.len())].clone();
                match rng.gen_range(0..6) {
                    5 => {
                        // Scale-up: a brand-new rank appears mid-flight. This is
                        // what makes property L non-vacuous — without a fresh
                        // Pending entry after settlement, a missing latch could
                        // never be observed.
                        registered_extra = true;
                        tracker.register(&[worker_id(&format!("http://scaleup{seed}"), 0)]);
                    }
                    0 => {
                        // A live batch, sometimes with a deliberate seq gap.
                        let seq = next_seq.entry(rank.clone()).or_insert(1);
                        *seq += rng.gen_range(1..=3);
                        let s = *seq;
                        let _ =
                            h.tx.send(WorkerEvent::Batch {
                                worker: rank,
                                seq: s,
                                batch: batch(vec![stored(None, vec![s * 1000 + seed as i64])]),
                            })
                            .await;
                    }
                    1 => {
                        // The obligation set and the snapshot's coverage are
                        // generated INDEPENDENTLY. That decoupling is the whole
                        // point: a peer routinely knows nothing about some rank
                        // this replica registered, and the resulting
                        // "obligation without coverage" case is where the
                        // terminality property actually bites.
                        let obligation_ranks: Vec<KvWorkerId> = ranks
                            .iter()
                            .filter(|_| rng.gen_bool(0.7))
                            .cloned()
                            .collect();
                        let obligation_ranks = if obligation_ranks.is_empty() {
                            vec![rank.clone()]
                        } else {
                            obligation_ranks
                        };
                        // Covers an unrelated rank, or nothing at all.
                        let covered = if rng.gen_bool(0.3) {
                            None
                        } else {
                            Some(ranks[rng.gen_range(0..ranks.len())].clone())
                        };
                        let vetted = match covered {
                            Some(c) => vetted_for(&c, rng.gen_range(0..8)),
                            None => VettedSnapshot::from_parts_for_test(vec![], vec![], vec![], 0),
                        };
                        discharged.extend(obligation_ranks.iter().cloned());
                        let _ = h
                            .ctrl_tx
                            .send(PumpControl::ApplySnapshot {
                                obligations: obligations(&tracker, &obligation_ranks),
                                vetted: Box::new(vetted),
                            })
                            .await;
                    }
                    2 => {
                        let _ = h
                            .ctrl_tx
                            .send(PumpControl::AbandonBootstrap {
                                obligations: obligations(&tracker, &[rank]),
                            })
                            .await;
                    }
                    3 => {
                        let _ = h
                            .ctrl_tx
                            .send(PumpControl::ForgetRanks { ranks: vec![rank] })
                            .await;
                    }
                    _ => {
                        let _ =
                            h.tx.send(WorkerEvent::PublisherReset { worker: rank })
                                .await;
                    }
                }

                // L: sampled continuously, because the violation is a transition.
                let settled_now = tracker.settled();
                let regressed = latched && !settled_now;
                assert!(!regressed, "seed {seed}: settled() went true -> false");
                latched |= settled_now;
            }

            drop(h.tx);
            drop(h.ctrl_tx);
            h.pump.await.unwrap();

            // T: every rank some control message took responsibility for must be
            // terminal. Deliberately NO blanket AbandonBootstrap first — that
            // would satisfy this no matter what the ApplySnapshot handler did,
            // which is exactly how an earlier version of this test passed
            // against a handler that left uncovered ranks Pending.
            //
            // `None` is accepted: ForgetRanks models the worker being removed.
            for r in &discharged {
                match tracker.state_of(r) {
                    None => {}
                    Some(state) => assert!(
                        state.is_terminal(),
                        "seed {seed}: rank {r:?} was named in an obligation set but left \
                         Pending — it would buffer forever and hold the readiness gate",
                    ),
                }
            }
            if discharged.len() == ranks.len() && !registered_extra {
                assert!(
                    tracker.settled(),
                    "seed {seed}: every rank resolved but the readiness gate never opened",
                );
            }
        }
    }

    /// Count of one rank-outcome label, for the exactly-once assertions below.
    fn rank_count(tracker: &BootstrapTracker, label: &str) -> u64 {
        tracker
            .rank_outcome_counts()
            .into_iter()
            .find(|(l, _)| *l == label)
            .map(|(_, c)| c)
            .unwrap_or(0)
    }

    /// Graft a rank with NOTHING held, so its splice proof is deferred, and leave
    /// it in that state. The shape every probe test starts from.
    async fn graft_with_deferred_proof(
        id: &KvWorkerId,
        watermark: i64,
    ) -> (Arc<BootstrapTracker>, PumpHarness) {
        let tracker = pending_tracker(std::slice::from_ref(id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(id), tracker.clone());
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(id)),
                vetted: Box::new(vetted_for(id, watermark)),
            })
            .await
            .unwrap();
        (tracker, h)
    }

    /// A grafted-but-unproven rank must NOT be counted warm yet: the whole point
    /// of the split counter is that `warm` means "proven", so counting at graft
    /// time and demoting later would double-count under two labels.
    #[tokio::test]
    async fn pump_deferred_proof_is_not_counted_warm_until_proven() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        // Drain: the graft is handled before the channels close.
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert_eq!(
            rank_count(&tracker, "warm"),
            0,
            "an unproven splice must not be tallied warm",
        );
    }

    /// The contiguous batch that proves the splice is what tallies the rank warm
    /// — exactly once.
    #[tokio::test]
    async fn pump_deferred_proof_counts_warm_once_when_the_stream_joins_up() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        for seq in [6, 7, 8] {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert_eq!(
            rank_count(&tracker, "warm"),
            1,
            "three batches after the proof must still tally one warm rank",
        );
        assert!(
            h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "a proven splice keeps its grafted state",
        );
    }

    /// A peer that has applied a sequence ABOVE our watermark proves a batch we
    /// never received was published, so the grafted state goes.
    #[tokio::test]
    async fn pump_splice_probe_advanced_discards_grafted_state() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(PumpControl::SpliceProbe {
                rank: id.clone(),
                epoch,
                verdict: SpliceVerdict::Advanced,
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "positive evidence of a missed batch must discard the graft",
        );
        assert!(
            h.cursors.lock().get(&id).is_none(),
            "cursor must be dropped"
        );
        assert_eq!(rank_count(&tracker, "gap"), 1);
        assert_eq!(rank_count(&tracker, "warm"), 0);
    }

    /// Silence with no witness of advancement is NOT evidence of a hole. Keeping
    /// the tree here is the whole reason the timeout probes instead of discarding:
    /// a quiet fleet would otherwise lose every warm tree on a timer.
    #[tokio::test]
    async fn pump_splice_probe_no_advance_keeps_grafted_state_and_counts_warm() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(PumpControl::SpliceProbe {
                rank: id.clone(),
                epoch,
                verdict: SpliceVerdict::NoAdvance,
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(
            h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "proof by absence must keep the grafted state",
        );
        assert_eq!(rank_count(&tracker, "warm"), 1);
    }

    /// An unanswerable probe resolves nothing: the rank stays warm AND still
    /// awaits proof, so a later witness can still demote it. Resolving `Unknown`
    /// either way would make an unreachable fleet decide the question.
    #[tokio::test]
    async fn pump_splice_probe_unknown_leaves_the_question_open() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(PumpControl::SpliceProbe {
                rank: id.clone(),
                epoch,
                verdict: SpliceVerdict::Unknown,
            })
            .await
            .unwrap();
        // Still pending proof, so this second verdict must still be actionable.
        h.ctrl_tx
            .send(PumpControl::SpliceProbe {
                rank: id.clone(),
                epoch,
                verdict: SpliceVerdict::Advanced,
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "an Unknown verdict must not consume the pending proof; the later \
             Advanced verdict then gaps, which re-queues the rank",
        );
        assert_eq!(rank_count(&tracker, "warm"), 0);
    }

    /// A fleet that can never answer must not leave the rank probing forever: the
    /// graft is kept, but tallied under a label that says it was never witnessed
    /// rather than pretending it was proven.
    #[tokio::test]
    async fn pump_repeated_unknown_probes_resolve_as_unwitnessed() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        for _ in 0..MAX_UNKNOWN_PROBES {
            h.ctrl_tx
                .send(PumpControl::SpliceProbe {
                    rank: id.clone(),
                    epoch,
                    verdict: SpliceVerdict::Unknown,
                })
                .await
                .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "an unwitnessed graft is kept, not discarded",
        );
        assert!(h.tree.match_prefix(None, &[100, 200]).workers.contains(&id));
        assert_eq!(rank_count(&tracker, "warm_unwitnessed"), 1);
        assert_eq!(
            rank_count(&tracker, "warm"),
            0,
            "unwitnessed must not be conflated with proven",
        );
    }

    /// A probe answer for a worker that was removed and re-added while it was in
    /// flight must not touch the new incarnation's state.
    #[tokio::test]
    async fn pump_splice_probe_from_a_stale_incarnation_is_ignored() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let stale_epoch = tracker.epoch_of(&id).expect("registered");
        tracker.forget(std::slice::from_ref(&id));
        tracker.register(std::slice::from_ref(&id));

        h.ctrl_tx
            .send(PumpControl::SpliceProbe {
                rank: id.clone(),
                epoch: stale_epoch,
                verdict: SpliceVerdict::Advanced,
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "the re-registered incarnation must be left alone",
        );
        assert_eq!(rank_count(&tracker, "gap"), 0);
    }

    /// Peer-attempt and per-rank tallies must live in separate counters: one
    /// accepted fetch can settle several ranks, so a shared counter would be
    /// divisible by nothing.
    #[tokio::test]
    async fn rank_and_peer_outcome_tallies_are_separate() {
        let a = worker_id("http://w1", 0);
        let b = worker_id("http://w2", 0);
        let tracker = pending_tracker(&[a.clone(), b.clone()]);
        let h = spawn_pump_with_bootstrap(&[a.clone(), b.clone()], tracker.clone());
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, &[a.clone(), b.clone()]),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(rank_count(&tracker, "abandoned"), 2, "one count per rank");
        assert!(
            tracker.peer_outcome_counts().is_empty(),
            "no peer was contacted, so the peer counter must stay empty",
        );
    }

    /// S1: a restarted publisher renumbers from 1, so its cursor MUST be cleared.
    ///
    /// Without this, every post-restart batch has `seq < last_applied` and is
    /// dropped as out-of-order: the rank's tree freezes at pre-restart state
    /// forever while every metric reports healthy. The pump's `PublisherReset`
    /// handling previously had no test at all.
    #[tokio::test]
    async fn pump_publisher_reset_clears_cursor_so_restarted_stream_applies() {
        let id = worker_id("http://w1", 0);
        // Settled tracker: this is the steady-state path, not bootstrap.
        let h = spawn_pump(std::slice::from_ref(&id));

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![11])]),
        })
        .await
        .unwrap();
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        // A fresh publisher's first batch.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![stored(None, vec![22])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            h.tree.match_prefix(None, &[22]).workers.contains(&id),
            "a restarted publisher's stream must apply, not be filtered as stale",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(1));
    }

    /// A reset means the engine restarted, and its speculative-decoding
    /// config may have changed with it — when discovery emits nothing for it
    /// (silent in-place restart), the pump re-introspects the vote itself.
    #[tokio::test]
    async fn pump_publisher_reset_rechecks_the_workers_mode_vote() {
        let (url, _hits, _release) = serve_server_info(serde_json::json!({
            "status": 200,
            "speculative_algorithm": "EAGLE3",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "127.0.0.1",
                "endpoint_port_base": 19557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            },
        }))
        .await;
        let id = worker_id(&url, 0);

        let h = spawn_pump(std::slice::from_ref(&id));
        // The pre-restart fleet view: this worker votes unigram.
        h.oracle.report_worker(&url, false);
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if h.oracle.vote_of(&url) == Some(true) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the reset must trigger a re-introspection that flips the vote");
    }

    /// A recheck that gets no usable answer must leave the vote alone — and
    /// the assertion is fenced by the fetch's full retry budget arriving at
    /// the fake, so it cannot pass merely because the task is still running.
    #[tokio::test]
    async fn pump_recheck_keeps_the_vote_when_unanswered() {
        let (url, mut hits, _release) = serve_server_info(serde_json::json!({"status": 503})).await;
        let id = worker_id(&url, 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        h.oracle.report_worker(&url, true);
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        // A 5xx is retriable, so the fetch burns its whole 3-attempt budget
        // (`FETCH_MAX_ATTEMPTS` in discovery) before the detached task can
        // reach the vote path at all.
        for _ in 0..3 {
            tokio::time::timeout(Duration::from_secs(5), hits.recv())
                .await
                .expect("a retriable answer must exhaust the retry budget")
                .expect("the fake outlives the test");
        }
        assert_eq!(
            h.oracle.vote_of(&url),
            Some(true),
            "an unanswered recheck must not erase the vote",
        );
    }

    /// The fence: a recheck whose fetch outlives the worker's removal must
    /// not re-insert a vote the teardown already erased — no later
    /// `forget_worker` will ever clean a resurrected one.
    #[tokio::test]
    async fn pump_recheck_does_not_resurrect_a_removed_workers_vote() {
        let spec = serde_json::json!({
            "status": "hold",
            "speculative_algorithm": "EAGLE3",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "127.0.0.1",
                "endpoint_port_base": 19557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            },
        });
        let (url, mut hits, release) = serve_server_info(spec).await;
        let id = worker_id(&url, 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        h.oracle.report_worker(&url, false);
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        // The fetch is in flight against the held fake; the worker leaves
        // discovery before the answer arrives.
        tokio::time::timeout(Duration::from_secs(5), hits.recv())
            .await
            .expect("the fetch must have started")
            .expect("the fake outlives the test");
        h.live_set.lock().remove(&id);
        release.notify_waiters();
        // The recheck now completes: without the fence its bigram answer
        // would flip the vote severed from any live worker.
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            h.oracle.vote_of(&url),
            Some(false),
            "a recheck landing after removal must not resurrect the vote",
        );
    }

    /// S10: a reset while still Pending must bail the rank to cold.
    ///
    /// Post-reset seq 1 can never exceed the peer's watermark, so the gap check
    /// passes trivially, the stale tree is kept, and every real delta is filtered.
    #[tokio::test]
    async fn pump_publisher_reset_while_pending_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 900,
            batch: batch(vec![stored(None, vec![33])]),
        })
        .await
        .unwrap();
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Failed),
            "a reset mid-bootstrap must abandon, not wait for a snapshot it can no \
             longer splice",
        );
        assert!(tracker.settled());
    }

    /// S2: a terminal rank is immutable. An Abandon arriving after a successful
    /// graft must not wipe the rank's tree and demote it to Failed.
    #[tokio::test]
    async fn pump_abandon_after_successful_graft_does_not_undo_it() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let obs = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs.clone(),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        // A second bootstrap task for the same obligation set gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: obs })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "a late Abandon must not demote an already-Recovered rank",
        );
        assert!(
            h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "a late Abandon must not wipe a grafted tree",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(5));
    }

    /// S3: the stale-incarnation guard on the ABANDON arm, mirroring the one
    /// already covered on the ApplySnapshot arm.
    #[tokio::test]
    async fn pump_abandon_from_a_stale_incarnation_is_ignored() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let stale = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Remove + re-add: a new incarnation, Pending again.
        tracker.forget(std::slice::from_ref(&id));
        let fresh = tracker.register(std::slice::from_ref(&id));
        assert_ne!(stale[0].1, fresh[0].1);

        // The previous incarnation's task gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: stale })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "a stale Abandon must not force the new incarnation cold",
        );
    }

    /// S4: `retain_workers` must strip carriers for ranks that are not pending,
    /// or a rank already applying live deltas gets peer state grafted UNDERNEATH
    /// it — the stale splice the design refuses.
    #[tokio::test]
    async fn pump_snapshot_does_not_graft_carriers_for_non_pending_ranks() {
        let pending_rank = worker_id("http://w1", 0);
        let live_rank = worker_id("http://w2", 0);
        let tracker = pending_tracker(&[pending_rank.clone(), live_rank.clone()]);
        let obs = obligations(&tracker, &[pending_rank.clone(), live_rank.clone()]);
        let h =
            spawn_pump_with_bootstrap(&[pending_rank.clone(), live_rank.clone()], tracker.clone());

        // `live_rank` leaves Pending first: it is now applying live deltas.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&live_rank)),
            })
            .await
            .unwrap();
        // A snapshot covering BOTH ranks then arrives: every node carries both
        // worker-table slots.
        let vetted = vetted_for_workers(&[&pending_rank, &live_rank], 5);
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs,
                vetted: Box::new(vetted),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        let m = h.tree.match_prefix(None, &[100, 200]);
        assert!(
            m.workers.contains(&pending_rank),
            "the still-pending rank must be grafted",
        );
        assert!(
            !m.workers.contains(&live_rank),
            "a rank already applying live deltas must NOT receive peer carriers",
        );
    }

    /// S8/S9: `ForgetRanks` must drop the pump-local queue and splice proof, or a
    /// re-added worker inherits the dead incarnation's high sequence numbers and
    /// its fresh stream reads as a permanent gap.
    #[tokio::test]
    async fn pump_forget_ranks_drops_held_state_and_tree() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Held while Pending, plus grafted tree state to be torn down.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        h.ctrl_tx
            .send(PumpControl::ForgetRanks {
                ranks: vec![id.clone()],
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            !h.tree.match_prefix(None, &[100, 200]).workers.contains(&id),
            "ForgetRanks must clear the worker's carriers on the pump",
        );
        assert!(
            h.cursors.lock().get(&id).is_none(),
            "a stale cursor would filter the re-added worker's fresh stream",
        );
    }

    /// S8, the part the teardown assertions above cannot see: `held` itself.
    ///
    /// A leaked queue is only observable through its EFFECT on the next
    /// incarnation — the dead incarnation's high sequence numbers sit at the front
    /// of the queue, so the fresh publisher's low watermark reads as a permanent
    /// gap and the re-added worker can never bootstrap.
    #[tokio::test]
    async fn pump_forget_ranks_drops_held_queue_so_readd_can_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Old incarnation holds a high-sequence batch.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 900,
            batch: batch(vec![stored(None, vec![77])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Worker removed, then re-added: fresh incarnation, publisher renumbered.
        h.ctrl_tx
            .send(PumpControl::ForgetRanks {
                ranks: vec![id.clone()],
            })
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let fresh = tracker.register(std::slice::from_ref(&id));

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![stored(Some(200), vec![300])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: fresh,
                vetted: Box::new(vetted_for(&id, 0)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // With a leaked queue, front() is seq 900 against watermark 0 -> spurious
        // gap -> Failed and the graft discarded.
        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "a leaked held queue makes the re-added worker read as gapped forever",
        );
        assert!(h.tree.match_prefix(None, &[100, 200]).workers.contains(&id));
    }

    /// Direct test of the pump loop's tree application — no sockets.
    #[tokio::test]
    async fn pump_applies_block_stored_to_tree() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, tx, pump) = (h.tree, h.tx, h.pump);

        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![10, 20, 30],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[10, 20, 30]);
        assert_eq!(m.matched_blocks, 3);
        assert!(m.workers.contains(&id), "tree must hold the worker");
    }

    /// A `WorkerEvent::Load` lands in the engine-load table (gauge, no
    /// cursor) keyed by the worker URL, and does not touch the tree.
    #[tokio::test]
    async fn pump_applies_load_to_engine_load_table() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, engine_load, tx, pump) = (h.tree, h.engine_load, h.tx, h.pump);

        tx.send(WorkerEvent::Load {
            worker: id.clone(),
            load: LoadStat {
                num_running_reqs: 8,
                num_waiting_reqs: 4,
                num_tokens: 0,
                max_total_num_tokens: 0,
            },
        })
        .await
        .unwrap();
        drop(tx);
        pump.await.unwrap();

        let fresh = engine_load.snapshot_fresh(Instant::now());
        assert_eq!(fresh.get("http://w1").copied(), Some(12)); // 8 + 4
                                                               // Load events must not pollute the cache tree.
        assert_eq!(tree.node_count(), 0);
    }

    /// Out-of-order seq is filtered: a batch with seq <= last_applied is
    /// dropped silently and does not mutate the tree.
    #[tokio::test]
    async fn pump_filters_out_of_order_seq() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, cursors, tx, pump) = (h.tree, h.cursors, h.tx, h.pump);

        // Apply seq=5 with block 10.
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 5,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![10],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        // Then a duplicate-style seq=3 that tries to remove block 10. Must
        // be dropped.
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: batch(vec![KvCacheEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![10],
                medium: None,
            })]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[10]);
        assert_eq!(
            m.matched_blocks, 1,
            "out-of-order remove must not undo the prior insert",
        );
        assert_eq!(cursors.lock().get(&id).copied(), Some(5));
    }

    /// AllBlocksCleared wipes the worker's tree state entirely.
    #[tokio::test]
    async fn pump_handles_all_blocks_cleared() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, tx, pump) = (h.tree, h.tx, h.pump);

        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![1, 2],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 2,
            batch: batch(vec![KvCacheEvent::AllBlocksCleared]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(
            m.matched_blocks, 0,
            "AllBlocksCleared must purge the worker"
        );
    }

    /// The pump drops events whose worker is not in `live_workers`. This
    /// is the safety net against the remove-then-pump race: an event
    /// queued before `remove_worker` clears the live set must not mutate
    /// the tree.
    #[tokio::test]
    async fn pump_drops_events_from_detached_workers() {
        let live_id = worker_id("http://live", 0);
        let dead_id = worker_id("http://dead", 0);
        let h = spawn_pump(std::slice::from_ref(&live_id));
        let (tree, tx, pump) = (h.tree, h.tx, h.pump);

        // Event from a worker that was never added (or was already
        // removed). Must be dropped.
        tx.send(WorkerEvent::Batch {
            worker: dead_id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![42],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        // Sanity: a live event still applies.
        tx.send(WorkerEvent::Batch {
            worker: live_id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![99],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        assert_eq!(tree.match_prefix(None, &[42]).matched_blocks, 0);
        assert_eq!(tree.match_prefix(None, &[99]).matched_blocks, 1);
    }

    /// `add_worker` must reject a worker whose `EventConfig.block_size`
    /// disagrees with the previously-established oracle value. The
    /// router cannot hash prompts simultaneously at two block sizes;
    /// silently accepting the mismatched worker would destroy
    /// cache-aware routing quality for every request.
    #[tokio::test]
    async fn add_worker_rejects_block_size_mismatch() {
        let index = KvEventIndex::new();
        // First worker establishes block_size=64 via the oracle.
        index.block_size_oracle().try_set(64).unwrap();

        let bad_cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30100,
            topic: String::new(),
            load_port_base: None,
            block_size: 128,
            dp_size: 1,
            is_bigram: false,
        };
        index
            .add_worker("http://127.0.0.1:30100", Some(bad_cfg))
            .await;
        assert_eq!(
            index.known_worker_count(),
            0,
            "mismatched worker must not be registered"
        );
        assert_eq!(
            index.block_size_oracle().hash_config(),
            None,
            "a rejected worker must not vote in the fleet hashing mode",
        );
        index.shutdown().await;
    }

    /// A rolling DCP rollout is the realistic way two *healthy* workers behind
    /// one model come to report different block sizes — before the widening
    /// they all reported `page_size` and agreed. Whichever registers first
    /// latches the oracle; the other half of the fleet is then excluded from
    /// the index entirely, with no subscriber and no hashing-mode vote. Both
    /// sides are derived through `kv_event_block_size` so this breaks if the
    /// derivation changes.
    #[tokio::test]
    async fn add_worker_rejects_dcp_worker_joining_a_non_dcp_fleet() {
        let index = KvEventIndex::new();
        // dp_size=0 short-circuits before the subscriber spawn while still
        // running the block-size validation, as in the test below.
        let cfg = |port_base: u16, dcp_size: Option<u32>| EventConfig {
            host: "127.0.0.1".into(),
            port_base,
            topic: String::new(),
            load_port_base: None,
            block_size: crate::policies::kv_events::kv_event_block_size(
                64,
                dcp_size,
                "http://127.0.0.1",
            )
            .unwrap(),
            dp_size: 0,
            is_bigram: false,
        };

        index
            .add_worker("http://127.0.0.1:30300", Some(cfg(30300, Some(1))))
            .await;
        assert_eq!(index.block_size_oracle().get(), Some(64));

        index
            .add_worker("http://127.0.0.1:30301", Some(cfg(30301, Some(8))))
            .await;
        assert_eq!(
            index.block_size_oracle().get(),
            Some(64),
            "a DCP worker must not overwrite the latched fleet block size",
        );
        assert_eq!(
            index.block_size_oracle().vote_of("http://127.0.0.1:30301"),
            None,
            "a rejected worker casts no hashing-mode vote",
        );
        index.shutdown().await;
    }

    #[tokio::test]
    async fn add_worker_seeds_oracle_with_first_block_size() {
        // Without any prior priming, the first worker through `add_worker`
        // should publish its `EventConfig.block_size` into the oracle so
        // subsequent matching workers reconcile and mismatched ones fail.
        let index = KvEventIndex::new();
        assert_eq!(index.block_size_oracle().get(), None);

        // A dp_size=0 cfg short-circuits before the subscriber spawn but
        // still runs through the block-size validation.
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30200,
            topic: String::new(),
            load_port_base: None,
            block_size: 64,
            dp_size: 0,
            is_bigram: false,
        };
        index.add_worker("http://127.0.0.1:30200", Some(cfg)).await;
        assert_eq!(index.block_size_oracle().get(), Some(64));
        index.shutdown().await;
    }

    #[tokio::test]
    async fn add_worker_seeds_bigram_flag_from_event_config() {
        // The discovery->routing seam: add_worker must publish
        // EventConfig.is_bigram into the oracle (alongside block_size) so
        // select() picks the bigram hasher for EAGLE workers.
        let index = KvEventIndex::new();
        assert!(!index.block_size_oracle().is_bigram());
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30300,
            topic: String::new(),
            load_port_base: None,
            block_size: 64,
            dp_size: 1,
            is_bigram: true,
        };
        index.add_worker("http://127.0.0.1:30300", Some(cfg)).await;
        assert!(
            index.block_size_oracle().is_bigram(),
            "add_worker must seed the bigram flag from EventConfig"
        );
        index.shutdown().await;
    }

    /// A worker that yields zero usable ranks exits before the registration
    /// commit point. Its block size may stay (page size is fleet-wide either
    /// way), but it must NOT vote in the hashing mode: a rank-less worker
    /// publishes nothing, so its vote would skew selection toward a family
    /// that never carries a node.
    #[tokio::test]
    async fn add_worker_with_no_usable_ranks_leaves_no_mode_vote() {
        let index = KvEventIndex::new();
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30400,
            topic: String::new(),
            load_port_base: None,
            block_size: 64,
            dp_size: 0,
            is_bigram: true,
        };
        index.add_worker("http://127.0.0.1:30400", Some(cfg)).await;
        assert_eq!(
            index.block_size_oracle().hash_config(),
            None,
            "a zero-rank worker must not vote in the fleet hashing mode",
        );
        index.shutdown().await;
    }

    /// The rolling-update scenario through the real wiring: as the old
    /// (EAGLE) generation drains out of discovery, the derived fleet mode
    /// must converge with it — a vote left behind reruns the incident this
    /// registry exists to prevent.
    #[tokio::test]
    async fn remove_worker_converges_the_fleet_hashing_mode() {
        let index = KvEventIndex::new();
        let cfg = |port_base: u16, is_bigram: bool| EventConfig {
            host: "127.0.0.1".into(),
            port_base,
            topic: String::new(),
            load_port_base: None,
            block_size: 64,
            dp_size: 1,
            is_bigram,
        };
        let (eagle1, eagle2, dspark) = (
            "http://127.0.0.1:30501",
            "http://127.0.0.1:30502",
            "http://127.0.0.1:30503",
        );
        index.add_worker(eagle1, Some(cfg(30501, true))).await;
        index.add_worker(eagle2, Some(cfg(30502, true))).await;
        index.add_worker(dspark, Some(cfg(30503, false))).await;
        assert_eq!(
            index.block_size_oracle().hash_config(),
            Some((64, true)),
            "bigram majority while the old generation dominates",
        );
        assert!(index.block_size_oracle().is_bimodal());

        index.remove_worker(eagle1).await;
        index.remove_worker(eagle2).await;
        assert!(
            !index.block_size_oracle().is_bimodal(),
            "the drained generation must stop making the fleet bimodal",
        );
        assert_eq!(
            index.block_size_oracle().hash_config(),
            Some((64, false)),
            "the fleet converges to unigram with the surviving worker",
        );

        index.remove_worker(dspark).await;
        assert_eq!(
            index.block_size_oracle().hash_config(),
            None,
            "an empty fleet is unknown, not unigram",
        );
        index.shutdown().await;
    }

    /// `remove_worker` clears the worker's engine load and its expected mark,
    /// so a re-added worker does not inherit stale load. The worker advertises
    /// a load port (no publisher there; the subscriber just retries in the
    /// background and is cancelled on remove).
    #[tokio::test]
    async fn remove_worker_clears_engine_load() {
        let index = KvEventIndex::new();
        let url = "http://127.0.0.1:59123";
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 59123,
            topic: String::new(),
            load_port_base: Some(59223),
            block_size: 64,
            dp_size: 1,
            is_bigram: false,
        };
        index.add_worker(url, Some(cfg)).await;
        assert_eq!(index.engine_load().expected_count(), 1);

        let now = Instant::now();
        index.engine_load().set(
            url,
            0,
            LoadStat {
                num_running_reqs: 3,
                num_waiting_reqs: 1,
                num_tokens: 0,
                max_total_num_tokens: 0,
            },
            now,
        );
        assert!(index.engine_load().snapshot_fresh(now).contains_key(url));

        index.remove_worker(url).await;
        assert!(
            !index
                .engine_load()
                .snapshot_fresh(Instant::now())
                .contains_key(url),
            "remove_worker must clear engine load"
        );
        assert_eq!(index.engine_load().expected_count(), 0);
        index.shutdown().await;
    }
}
