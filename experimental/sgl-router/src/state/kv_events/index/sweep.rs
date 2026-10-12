//! One bounded peer sweep, from fetch through vet to a single `PumpControl` delivery.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::{KvEventIndex, ObligationBatch, PumpControl};
use crate::state::kv_events::block_size_oracle::BlockSizeOracle;
use crate::state::kv_events::bootstrap::{
    fetch_snapshot, BootstrapState, BootstrapTracker, FetchAnswer, PeerRegistry,
};
use crate::state::kv_events::bootstrap::{
    PeerSnapshot, SnapshotOutcome, SweepOutcome, VettedSnapshot,
};
use crate::state::kv_events::tree::KvWorkerId;

/// Delay between peer-sweep attempts while no usable peer has been found.
///
/// Short relative to the bootstrap deadline that bounds the whole sweep, so a
/// peer becoming available is picked up promptly.
const PEER_RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// Passes to skip a peer after its first retriable failure in a sweep.
///
/// Unreachable, vet-failed and non-covering peers are retried after a sit-out:
/// the peer may come back or discover our workers, and sitting out spares a
/// serving replica a multi-megabyte export every [`PEER_RETRY_INTERVAL`]. Each
/// further miss by the same peer within the sweep doubles its sit-out, up to
/// [`MAX_PEER_COOLDOWN_PASSES`], so a peer that keeps failing is asked rarely
/// while a newly discovered one is still tried on the next pass.
const PEER_COOLDOWN_PASSES: u32 = 4;

/// Longest sit-out a repeatedly failing peer backs off to, in retry-interval
/// time (the passes themselves add their fetch time on top).
const MAX_PEER_COOLDOWN: Duration = Duration::from_secs(30);

/// [`MAX_PEER_COOLDOWN`] in passes of [`PEER_RETRY_INTERVAL`].
const MAX_PEER_COOLDOWN_PASSES: u32 =
    (MAX_PEER_COOLDOWN.as_millis() / PEER_RETRY_INTERVAL.as_millis()) as u32;

/// Sit-out for a peer's `strike`-th retriable miss in one sweep (1-based):
/// [`PEER_COOLDOWN_PASSES`] doubled per repeat, capped at
/// [`MAX_PEER_COOLDOWN_PASSES`].
fn peer_cooldown_passes(strike: u32) -> u32 {
    let factor = 1u32
        .checked_shl(strike.saturating_sub(1))
        .unwrap_or(u32::MAX);
    PEER_COOLDOWN_PASSES
        .saturating_mul(factor)
        .min(MAX_PEER_COOLDOWN_PASSES)
}

/// Fetches the per-request timeout leaves room for within one deadline. The
/// sweep awaits candidates in turn, so one fetch must not consume the whole
/// deadline.
pub(super) const SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE: u32 = 4;

/// Preferred floor for the per-fetch timeout; a fleet-sized snapshot needs
/// tens of seconds for the producer's export build and the transfer. Yields to
/// `deadline / 2` and to the cap; see [`snapshot_fetch_timeout`].
pub(super) const SNAPSHOT_FETCH_TIMEOUT_FLOOR: Duration = Duration::from_secs(30);

/// Per-request timeout for a peer snapshot fetch, a strict fraction of the
/// configured bootstrap budget so several peers can be tried within it.
///
/// `cap` (from `--kv-bootstrap-fetch-timeout-cap-ms`, default
/// [`super::super::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP`]) bounds the
/// derivation, so a generous deadline (see `MAX_KV_BOOTSTRAP_TIMEOUT_MS`)
/// still bounds one progressing transfer; raise it when the snapshot body
/// outgrows the default.
///
/// The per-fetch bound is derived from the configured budget, not from the
/// time a given sweep has left, so a sweep running on a shrunken remainder is
/// bounded by its outer deadline rather than by this.
pub(super) fn snapshot_fetch_timeout(deadline: Duration, cap: Duration) -> Duration {
    // A pre-settled (bootstrap-disabled) tracker reports a zero deadline. The
    // client is still built, and a zero reqwest timeout means "expire
    // immediately" rather than "no timeout", so fall back to the floor.
    if deadline.is_zero() {
        return SNAPSHOT_FETCH_TIMEOUT_FLOOR;
    }
    // Yielding to the cap keeps `clamp` from panicking on a below-floor cap.
    let floor = SNAPSHOT_FETCH_TIMEOUT_FLOOR.min(deadline / 2).min(cap);
    (deadline / SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE).clamp(floor, cap)
}

/// The handles a bounded peer sweep needs, cloned out of [`KvEventIndex`] so the
/// sweep can run detached — or be awaited by the coordinator — without
/// borrowing `self`.
pub(super) struct BootstrapDeps {
    http: reqwest::Client,
    pub(super) peers: Arc<PeerRegistry>,
    pub(super) bootstrap: Arc<BootstrapTracker>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    oracle: Arc<BlockSizeOracle>,
    ctrl_tx: mpsc::Sender<PumpControl>,
}

impl BootstrapDeps {
    /// Budget for one sweep. Before readiness settles, the remainder of the
    /// tracker's single `/readyz` window; once settled, a full `timeout()` for
    /// a late-discovered worker (`time_remaining` saturates at zero).
    pub(super) fn deadline(&self) -> Duration {
        if self.bootstrap.settled() {
            return self.bootstrap.timeout();
        }
        self.bootstrap
            .time_remaining()
            .unwrap_or(self.bootstrap.timeout())
    }
}

/// Outcome of one bounded peer sweep.
pub(super) enum SweepResult {
    Found(VettedSnapshot),
    /// Discovery confirmed there are no siblings, so waiting cannot help.
    NoPeers,
    /// Every candidate's latest word was cold-or-terminal: an empty tree, or a
    /// permanent incompatibility. Waiting out the deadline cannot help —
    /// anything these peers learn later arrives over this replica's own event
    /// subscriptions anyway.
    FleetCold {
        /// Size of the candidate set the verdict was proven over — not the
        /// live registry, which may have changed since.
        peers_tried: usize,
    },
    TimedOut {
        /// Registry size at the deadline.
        peers_tried: usize,
        last_reason: Option<String>,
    },
    /// Every obligation the sweep was run for left `Pending` on its own
    /// incarnation without this sweep's snapshot — resolved from its stream's
    /// origin (`resolve_from_origin`), settled cold per rank, or forgotten.
    /// Nothing is owed to them, so delivery sends no control message.
    RanksResolved,
}

impl SweepResult {
    /// The metric-facing verdict, mirroring `VetError::outcome`: the closed
    /// label set lives in bootstrap.rs so no call site can mint new values.
    fn outcome(&self) -> SweepOutcome {
        match self {
            Self::Found(_) => SweepOutcome::Found,
            Self::NoPeers => SweepOutcome::NoPeers,
            Self::FleetCold { .. } => SweepOutcome::FleetCold,
            Self::TimedOut { .. } => SweepOutcome::TimedOut,
            Self::RanksResolved => SweepOutcome::RanksResolved,
        }
    }

    /// Size of the candidate set the verdict was proven over. `Found` reports
    /// 1 — the peer that answered — so a success is never mistaken for a
    /// verdict reached over nobody.
    fn peers_tried(&self) -> usize {
        match self {
            Self::Found(_) => 1,
            Self::NoPeers | Self::RanksResolved => 0,
            Self::FleetCold { peers_tried } | Self::TimedOut { peers_tried, .. } => *peers_tried,
        }
    }
}

/// Sweep peers until one yields a usable snapshot, every sibling proves it
/// has nothing to give, discovery proves there are none, or the budget runs
/// out.
///
/// Retries every `PEER_RETRY_INTERVAL` because the peer watch can deliver its
/// first list after worker discovery. Settles early when every candidate
/// answers cold or incompatible, and settles a single rank early when every
/// candidate proves it holds nothing for that rank. `freshness_floor` is
/// re-derived into a max age per attempt; a peer that has already answered is
/// asked for something newer still (see `SweepState::export_floor`).
///
/// Each pass sweeps only the obligations still `Pending` on the incarnation they
/// were registered under. A rank can leave `Pending` while the sweep runs — the
/// pump resolves it from its stream's origin, or the worker is removed — and a
/// rank nobody is waiting on must neither keep the sweep alive nor decide which
/// peer's snapshot is "covering".
pub(super) async fn sweep_until_deadline(
    deps: &BootstrapDeps,
    obligations: &[(KvWorkerId, u64)],
    deadline: Duration,
    freshness_floor: Instant,
) -> SweepResult {
    sweep_recording_settled(
        deps,
        obligations,
        deadline,
        freshness_floor,
        &mut Vec::new(),
    )
    .await
}

/// [`sweep_until_deadline`], also appending to `settled` every obligation it
/// released cold mid-sweep (see `settle_ranks_cold`).
///
/// For the coordinator, which decides after the sweep which obligations are
/// still owed a look: the pump may not have drained a mid-sweep release yet, so
/// a settled rank can still read `Pending` then, and only this list tells it
/// apart from one the sweep never resolved.
pub(super) async fn sweep_recording_settled(
    deps: &BootstrapDeps,
    obligations: &[(KvWorkerId, u64)],
    deadline: Duration,
    freshness_floor: Instant,
    settled: &mut Vec<(KvWorkerId, u64)>,
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
        let mut state = SweepState::new();
        let mut active: Vec<(KvWorkerId, u64)> = obligations.to_vec();
        loop {
            active.retain(|(rank, epoch)| still_pending(&deps.bootstrap, rank, *epoch));
            if active.is_empty() {
                return Some(SweepResult::RanksResolved);
            }
            let ranks: Vec<KvWorkerId> = active.iter().map(|(r, _)| r.clone()).collect();
            match sweep_peers(&ctx, &ranks, &mut state, &last_reason).await {
                SweepPass::Found(vetted) => return Some(SweepResult::Found(vetted)),
                SweepPass::FleetCold { peers_tried } => {
                    return Some(SweepResult::FleetCold { peers_tried })
                }
                SweepPass::NothingToRecover { ranks, peers_tried } => {
                    settled.extend(settle_ranks_cold(deps, &mut active, &ranks, peers_tried).await);
                    if active.is_empty() {
                        return Some(SweepResult::RanksResolved);
                    }
                }
                SweepPass::KeepLooking => {}
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
        Ok(Some(result)) => result,
        Ok(None) => SweepResult::NoPeers,
        Err(_) => SweepResult::TimedOut {
            peers_tried: deps.peers.len(),
            last_reason: last_reason.lock().clone(),
        },
    }
}

/// Release `ranks` cold now, mid-sweep, because every sibling proved it holds
/// nothing for them, and stop sweeping for them.
///
/// Sent from here rather than at delivery because the sweep keeps running for
/// its other ranks, and a rank no sibling can supply must not hold its batches —
/// or `/readyz` — until THEY resolve. Each obligation carries the incarnation it
/// was registered under, so the pump drops the release for a rank re-registered
/// since. The final delivery for this sweep still names these obligations; by
/// then they are not `Pending`, which every pump handler already treats as a
/// no-op.
///
/// Returns the obligations it released.
async fn settle_ranks_cold(
    deps: &BootstrapDeps,
    active: &mut Vec<(KvWorkerId, u64)>,
    ranks: &[KvWorkerId],
    peers_tried: usize,
) -> Vec<(KvWorkerId, u64)> {
    let (settled, rest): (Vec<_>, Vec<_>) = std::mem::take(active)
        .into_iter()
        .partition(|(r, _)| ranks.contains(r));
    *active = rest;
    info!(
        ranks = settled.len(),
        peers_tried,
        still_sweeping = active.len(),
        "kv-bootstrap: every sibling replica holds nothing for these ranks; \
         settling them cold without waiting out the deadline",
    );
    if deps
        .ctrl_tx
        .send(PumpControl::AbandonBootstrap {
            obligations: settled.clone(),
        })
        .await
        .is_err()
    {
        warn!("kv-bootstrap: pump is gone; per-rank cold settle discarded");
    }
    settled
}

/// Whether the obligation `(rank, epoch)` still has a rank waiting on it: the
/// rank is registered under that incarnation and has not left `Pending`.
pub(super) fn still_pending(bootstrap: &BootstrapTracker, rank: &KvWorkerId, epoch: u64) -> bool {
    bootstrap.epoch_of(rank) == Some(epoch)
        && bootstrap.state_of(rank) == Some(BootstrapState::Pending)
}

/// Turn a sweep result into the single [`PumpControl`] message its obligations
/// are owed. Every exit path that leaves ranks `Pending` sends exactly one,
/// which is what releases them; [`SweepResult::RanksResolved`] sends none,
/// because its ranks have already left `Pending`.
pub(super) async fn deliver_bootstrap(
    deps: &BootstrapDeps,
    obligations: Vec<(KvWorkerId, u64)>,
    result: SweepResult,
    deadline: Duration,
) {
    let n = obligations.len();
    deps.bootstrap
        .record_sweep_result(result.outcome(), result.peers_tried());
    match &result {
        SweepResult::Found(_) => {}
        SweepResult::RanksResolved => {
            debug!(
                ranks = n,
                "kv-bootstrap: every rank left Pending before a peer snapshot was needed",
            );
            return;
        }
        SweepResult::NoPeers => info!(
            ranks = n,
            "kv-bootstrap: no sibling replicas to bootstrap from; ranks will run cold",
        ),
        SweepResult::FleetCold { peers_tried } => info!(
            ranks = n,
            peers_tried,
            "kv-bootstrap: every sibling replica answered with an empty or \
             incompatible tree; settling cold without waiting out the deadline",
        ),
        SweepResult::TimedOut {
            peers_tried,
            last_reason,
        } => warn!(
            ranks = n,
            timeout_ms = deadline.as_millis(),
            peers_tried,
            last_reason = last_reason.as_deref().unwrap_or("none recorded"),
            "kv-bootstrap: no peer supplied a usable snapshot within the deadline; \
             ranks will run cold",
        ),
    }
    let msg = match result {
        SweepResult::Found(vetted) => PumpControl::ApplySnapshot {
            obligations,
            vetted: Box::new(vetted),
        },
        _ => PumpControl::AbandonBootstrap { obligations },
    };
    if deps.ctrl_tx.send(msg).await.is_err() {
        warn!("kv-bootstrap: pump is gone; bootstrap result discarded");
    }
}

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

/// Verdict of one pass over the candidate peers.
enum SweepPass {
    /// A peer's snapshot vetted and covers ranks we are bootstrapping.
    Found(VettedSnapshot),
    /// Nothing usable this pass, but at least one peer might have state later
    /// (unanswered, still bootstrapping, or warm-but-not-covering).
    KeepLooking,
    /// Every candidate's latest word was cold-or-terminal. Carries the size of
    /// the candidate set the verdict was proven over; see
    /// [`SweepResult::FleetCold`].
    FleetCold { peers_tried: usize },
    /// Not cold as a fleet, but every candidate's latest word says it holds
    /// nothing for `ranks` (see [`SweepState::nothing_to_recover`]). Only these
    /// ranks settle; the sweep goes on for the rest.
    NothingToRecover {
        ranks: Vec<KvWorkerId>,
        peers_tried: usize,
    },
}

/// Per-sweep peer state carried across passes.
struct SweepState {
    /// Peers whose snapshot is permanently incompatible (format, block size).
    /// A stable property of the peer for the life of the process, so they are
    /// never re-fetched.
    permanently_rejected: HashSet<String>,
    /// Retriable-failure sit-out, in remaining passes.
    cooldown: HashMap<String, u32>,
    /// Retriable misses per peer this sweep; sizes the next sit-out via
    /// [`peer_cooldown_passes`]. Never cleared: a peer that yields a usable
    /// snapshot ends the sweep.
    strikes: HashMap<String, u32>,
    /// Peers whose latest answer was an empty tree
    /// ([`PeerSnapshot::holds_no_state`], NOT the stricter
    /// [`PeerSnapshot::is_cold`] vetting uses — a peer mid-bootstrap that
    /// already holds nodes is not done, but it plainly HAS state). Carried
    /// across passes so a peer in cooldown keeps its last classification;
    /// dropped on any warmer or unknown answer, since the fleet proving cold
    /// is only meaningful when EVERY peer's latest word is "I have nothing".
    cold_witnessed: HashSet<String>,
    /// Per peer, the ranks its latest answer named in
    /// [`PeerSnapshot::empty_ranks`] — the per-rank sibling of
    /// `cold_witnessed`, with the same lifetime: kept across cooldowns,
    /// replaced by each answer, dropped when the peer stops answering. Wire
    /// identities, not [`KvWorkerId`]s, because this is evidence about a rank
    /// rather than a routing identity.
    holds_nothing: HashMap<String, HashSet<(String, u32)>>,
    /// Per peer, when its latest body arrived; see [`Self::export_floor`].
    received_at: HashMap<String, Instant>,
}

impl SweepState {
    fn new() -> Self {
        Self {
            permanently_rejected: HashSet::new(),
            cooldown: HashMap::new(),
            strikes: HashMap::new(),
            cold_witnessed: HashSet::new(),
            holds_nothing: HashMap::new(),
            received_at: HashMap::new(),
        }
    }

    /// The instant a fetch from `peer` must demand an export newer than: the
    /// sweep's `floor`, or the arrival of this peer's last body if later.
    ///
    /// The second term exists because a producer reuses its cached export
    /// for any requester whose `max_age` it meets (see
    /// `KvEventIndex::peer_snapshot_body`), and a floor fixed at sweep start is
    /// met by that export for the whole sweep: a re-fetch after a useless
    /// answer — no coverage, empty, nothing we know — would get the same
    /// document back even after the peer has since learned what we need. An export is always sampled before it is sent, so "newer than
    /// its arrival here" excludes exactly the answer already seen, whatever the
    /// transit latency, and forces the peer to take a fresh one.
    ///
    /// Only re-fetches ratchet, and only as often as cooldown lets this peer
    /// be asked. Requesters still share any build that started after the
    /// instant each demands, however long they queue behind it, because the
    /// producer pins that instant on arrival — so a herd re-fetching one peer
    /// still pays about one walk per build-duration of arrival spread.
    fn export_floor(&self, peer: &str, floor: Instant) -> Instant {
        self.received_at
            .get(peer)
            .map_or(floor, |&at| at.max(floor))
    }

    /// Record that `peer` answered with a body at `at`; see
    /// [`Self::export_floor`].
    fn note_received(&mut self, peer: &str, at: Instant) {
        self.received_at.insert(peer.to_string(), at);
    }

    /// A peer that did not answer has unknown warmth: it cannot count toward
    /// the all-cold verdict, and it sits out a while — the failures that
    /// land here are mostly stable for the life of the process (a transport it
    /// cannot complete, a body that will not inflate or parse), so re-fetching
    /// a multi-megabyte body from it on every pass is pure load on a replica
    /// that is itself serving traffic.
    fn note_unreachable(&mut self, peer: &str) {
        self.cold_witnessed.remove(peer);
        self.holds_nothing.remove(peer);
        self.start_cooldown(peer);
    }

    /// Record the temperature of an answered snapshot: an empty tree is a
    /// cold witness; anything with content — usable, still bootstrapping, or
    /// not covering us — means the fleet holds state worth waiting for.
    fn note_answer(&mut self, peer: &str, snap: &PeerSnapshot) {
        if snap.holds_no_state() {
            self.cold_witnessed.insert(peer.to_string());
        } else {
            self.cold_witnessed.remove(peer);
        }
        self.holds_nothing.insert(
            peer.to_string(),
            snap.empty_ranks
                .iter()
                .map(|w| (w.url.clone(), w.dp_rank))
                .collect(),
        );
    }

    /// Permanent rejection is terminal on its own — a peer whose state we can
    /// never consume has nothing to give us, whatever it holds.
    fn note_permanent_reject(&mut self, peer: &str) {
        self.permanently_rejected.insert(peer.to_string());
    }

    /// Sit out without touching the cold witness; contrast
    /// `note_unreachable`.
    fn note_sit_out(&mut self, peer: &str) {
        self.start_cooldown(peer);
    }

    /// Book one more retriable miss for `peer` and sit it out for the
    /// backed-off pass count.
    fn start_cooldown(&mut self, peer: &str) {
        let strike = self.strikes.entry(peer.to_string()).or_insert(0);
        *strike = strike.saturating_add(1);
        let passes = peer_cooldown_passes(*strike);
        self.cooldown.insert(peer.to_string(), passes);
    }

    /// True while `peer` sits out a retriable failure, decaying one pass.
    fn cooling(&mut self, peer: &str) -> bool {
        match self.cooldown.get_mut(peer) {
            Some(remaining) if *remaining > 0 => {
                *remaining -= 1;
                true
            }
            _ => false,
        }
    }

    /// The all-cold verdict: every candidate's latest classification says it
    /// has nothing to give. Peers skipped by cooldown count by their last
    /// classification; a candidate that has never answered (new to the set,
    /// or only ever unreachable) keeps the sweep waiting.
    ///
    /// An empty candidate set is NOT cold: `all()` on an empty iterator is
    /// vacuously true, and the peer watch regularly delivers its first list
    /// after worker discovery completes. An empty set means "no information"
    /// — handled by the `known_to_have_no_peers` path — not "everyone proved
    /// empty".
    fn fleet_is_cold(&self, candidates: &[String]) -> bool {
        !candidates.is_empty()
            && candidates
                .iter()
                .all(|p| self.permanently_rejected.contains(p) || self.cold_witnessed.contains(p))
    }

    /// The per-rank verdict: every candidate's latest word says it has nothing
    /// for `rank` — it named the rank in `empty_ranks`, or it is hopeless as a
    /// whole (the two `fleet_is_cold` classes). Same veto rules: a candidate
    /// that has never answered, or whose last answer does not name the rank
    /// (it may not have discovered it yet), keeps the rank waiting, and an
    /// empty candidate set proves nothing.
    ///
    /// Waiting cannot help such a rank: the answers were exported after this
    /// replica subscribed, so whatever a sibling learns about the rank from here
    /// on arrives over this replica's own subscription too. The one thing lost
    /// is what a sibling still bootstrapping the rank holds back from before
    /// our subscription — the price of not letting a fleet whose every member
    /// is waiting on every other member burn its whole deadline.
    fn nothing_to_recover(&self, rank: &KvWorkerId, candidates: &[String]) -> bool {
        let key = (rank.url.clone(), rank.dp_rank);
        !candidates.is_empty()
            && candidates.iter().all(|p| {
                self.permanently_rejected.contains(p)
                    || self.cold_witnessed.contains(p)
                    || self.holds_nothing.get(p).is_some_and(|s| s.contains(&key))
            })
    }
}

/// One pass over the candidates, returning the first snapshot that vets and
/// covers a rank.
async fn sweep_peers(
    ctx: &SweepCtx<'_>,
    ranks: &[KvWorkerId],
    state: &mut SweepState,
    last_reason: &Mutex<Option<String>>,
) -> SweepPass {
    let SweepCtx {
        http,
        peers,
        bootstrap,
        live_workers,
        oracle,
        freshness_floor,
    } = ctx;
    // Both halves or neither: acting on a block size whose companion hashing
    // mode is not yet published would vet a peer's tree against an identity
    // this replica has not established. See `BlockSizeOracle::hash_config`.
    // Only the size is vetted on — hashing mode deliberately is not, see
    // `VettedSnapshot::from_wire`.
    let Some((local_block_size, local_bigram)) = oracle.hash_config() else {
        return SweepPass::KeepLooking;
    };
    // A fetch that yielded nothing: name it for the terminal log and tally it.
    // The `last_reason` guard drops at the end of its statement.
    let miss = |peer: &str, outcome: SnapshotOutcome, detail: &str| {
        *last_reason.lock() = Some(format!("{peer}: {detail}"));
        bootstrap.record_peer_outcome(outcome, peer, Some(detail));
    };
    let candidates = peers.candidates();
    for peer in &candidates {
        if state.permanently_rejected.contains(peer) {
            continue;
        }
        if state.cooling(peer) {
            continue;
        }
        // Ask for an export that beats the floor — and, once this peer has
        // answered, that beats its last answer too (see
        // `SweepState::export_floor`). Derived per attempt, not once: the
        // condition is "newer than that instant", and only the age it
        // corresponds to moves as the sweep retries.
        let max_age = state.export_floor(peer, *freshness_floor).elapsed();
        let fetched = match fetch_snapshot(http, peer, Some(max_age)).await {
            Ok(FetchAnswer::Body(s)) => {
                state.note_received(peer, Instant::now());
                Ok(s)
            }
            // Reachable but no usable body — the status names which kind of
            // wrong: 404 is an older router image that does not serve the
            // route, 5xx is a sick sibling. Both retriable.
            Ok(FetchAnswer::NoBody(status)) => Err(format!("answered HTTP {status}")),
            // `{e:#}` for the anyhow chain: the bare Display prints only the
            // outermost message, dropping the reqwest/io cause that names what
            // actually went wrong.
            Err(e) => Err(format!("{e:#}")),
        };
        let snap = match fetched {
            Ok(snap) => snap,
            Err(detail) => {
                state.note_unreachable(peer);
                miss(peer, SnapshotOutcome::Unreachable, &detail);
                continue;
            }
        };
        // Classify before vetting: an empty tree is a cold witness even when the
        // snapshot is otherwise well formed.
        state.note_answer(peer, &snap);
        let peer_bigram = snap.is_bigram;
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
                    // Usually a warm peer that has not yet discovered our worker.
                    // Fixed text so the per-peer log limiter dedups repeats.
                    state.note_sit_out(peer);
                    miss(
                        peer,
                        SnapshotOutcome::ColdPeer,
                        "snapshot tracks none of the ranks being bootstrapped",
                    );
                    continue;
                }
                // How much of the copy can steer a selection: carried nodes
                // answer a query, structure nodes are match paths only. See
                // `VettedSnapshot::carrier_counts`.
                let (carried, structure) = vetted.carrier_counts();
                // The producer's hashing mode is logged rather than vetted
                // (see `VettedSnapshot::from_wire`), but a disagreement means
                // the fleet is mid-rollout across a spec-config change and the
                // grafted blocks may not match — worth naming when it happens.
                if peer_bigram != local_bigram {
                    info!(
                        peer = %peer,
                        peer_bigram,
                        local_bigram,
                        "kv-bootstrap: peer reports a different fleet hashing mode; \
                         grafting anyway, since the surviving carriers are workers \
                         this replica discovered itself",
                    );
                }
                info!(
                    peer = %peer,
                    nodes = vetted.node_count(),
                    carried_nodes = carried,
                    structure_nodes = structure,
                    workers = vetted.worker_count(),
                    dropped_workers = vetted.dropped_workers(),
                    "kv-bootstrap: snapshot accepted; handing to pump",
                );
                bootstrap.record_peer_outcome(SnapshotOutcome::Accepted, peer, None);
                return SweepPass::Found(vetted);
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
                    state.note_permanent_reject(peer);
                }
                state.note_sit_out(peer);
                miss(peer, e.outcome(), &e.to_string());
            }
        }
    }
    let verdict = if state.fleet_is_cold(&candidates) {
        SweepPass::FleetCold {
            peers_tried: candidates.len(),
        }
    } else {
        let barren: Vec<KvWorkerId> = ranks
            .iter()
            .filter(|r| state.nothing_to_recover(r, &candidates))
            .cloned()
            .collect();
        if barren.is_empty() {
            return SweepPass::KeepLooking;
        }
        SweepPass::NothingToRecover {
            ranks: barren,
            peers_tried: candidates.len(),
        }
    };
    // Discard the cold verdict — fleet-wide or per rank — if candidate
    // membership changed mid-pass (compared as sets; a same-length swap
    // counts): the newcomer was never consulted.
    if peers.candidates().into_iter().collect::<HashSet<_>>()
        != candidates.iter().cloned().collect::<HashSet<_>>()
    {
        // Not silent: this is how a flapping EndpointSlice turns a cold
        // fleet's quick settle into a full-deadline wait.
        info!("kv-bootstrap: candidate set changed mid-pass; discarding the cold verdict");
        return SweepPass::KeepLooking;
    }
    verdict
}

impl KvEventIndex {
    /// Fetch a snapshot for one batch of obligations from the first peer, in
    /// shuffled order, whose snapshot vets and covers at least one of its ranks,
    /// and hand the result to the pump.
    ///
    /// Runs detached: `/readyz` is gated by the tracker, not by awaiting this,
    /// so a slow peer delays readiness only up to the bootstrap deadline. Every
    /// exit path that leaves ranks `Pending` sends exactly one [`PumpControl`]
    /// message, which is what guarantees the held-back batches are eventually
    /// released (see `deliver_bootstrap`).
    pub(super) fn spawn_bootstrap(&self, batch: ObligationBatch) {
        let deps = self.bootstrap_deps();
        tokio::spawn(async move {
            let ObligationBatch {
                obligations,
                holding_since,
                late_join: _,
            } = batch;
            let deadline = deps.deadline();
            let result = sweep_until_deadline(&deps, &obligations, deadline, holding_since).await;
            deliver_bootstrap(&deps, obligations, result, deadline).await;
        });
    }

    /// Clone the handles a sweep needs, so it can run detached — or be awaited
    /// by the coordinator — without borrowing `self`.
    pub(super) fn bootstrap_deps(&self) -> BootstrapDeps {
        BootstrapDeps {
            http: self.snapshot_http.clone(),
            peers: Arc::clone(&self.peers),
            bootstrap: Arc::clone(&self.bootstrap),
            live_workers: Arc::clone(&self.live_workers),
            oracle: Arc::clone(&self.block_size_oracle),
            ctrl_tx: self.ctrl_tx.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::state::kv_events::bootstrap::{WireWorker, SNAPSHOT_FORMAT, SNAPSHOT_PATH};
    use crate::state::kv_events::tree::SnapshotNode;

    /// One snapshot fetch never consumes the whole bootstrap deadline, so the
    /// sweep can reach another candidate. The deadlines span short explicit
    /// budgets, where the half-deadline bound rather than the divisor binds,
    /// up to past the configured default.
    #[test]
    fn snapshot_fetch_timeout_is_a_strict_fraction_of_every_nonzero_deadline() {
        use crate::state::kv_events::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP as CAP;
        let default_secs = crate::config::DEFAULT_KV_BOOTSTRAP_TIMEOUT_MS / 1_000;
        assert_eq!(
            default_secs, 600,
            "test's premise: the default budget is 10 minutes"
        );

        for deadline_secs in [1, 2, 5, 10, 20, 30, 60, 120, default_secs, 3600] {
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
        // The default budget funds a snapshot-sized fetch: the divisor governs
        // and lands above the floor, so the constructor's below-floor warning
        // stays quiet. Pinned so a change to any of the constants is
        // deliberate.
        let at_default = snapshot_fetch_timeout(Duration::from_secs(default_secs), CAP);
        assert_eq!(at_default, Duration::from_secs(150));
        assert!(
            at_default >= SNAPSHOT_FETCH_TIMEOUT_FLOOR,
            "the default budget must not trip the constructor's warning",
        );
        // A short explicit budget still derives a bound below the floor, the
        // case the constructor warns about. The half-deadline bound binds
        // there, so raising the cap changes nothing.
        let short = Duration::from_secs(20);
        assert_eq!(snapshot_fetch_timeout(short, CAP), Duration::from_secs(10));
        assert!(snapshot_fetch_timeout(short, CAP) < SNAPSHOT_FETCH_TIMEOUT_FLOOR);
        assert_eq!(
            snapshot_fetch_timeout(short, CAP),
            snapshot_fetch_timeout(short, Duration::from_secs(3600)),
        );
        // A zero deadline (bootstrap disabled) yields the floor, not a zero
        // timeout that would expire immediately.
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
        // A long deadline derives above the default cap, which holds it at
        // `CAP`; a configured cap binds at exactly its value.
        use crate::state::kv_events::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP as CAP;
        assert_eq!(snapshot_fetch_timeout(Duration::from_secs(3600), CAP), CAP);
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
            Duration::from_millis(crate::config::DEFAULT_KV_BOOTSTRAP_FETCH_TIMEOUT_CAP_MS),
        );
    }

    /// The CLI's minimum fetch cap is the per-fetch floor, so a validated cap
    /// never binds below it.
    #[test]
    fn snapshot_fetch_timeout_floor_matches_the_config_minimum_cap() {
        assert_eq!(
            SNAPSHOT_FETCH_TIMEOUT_FLOOR,
            Duration::from_millis(crate::config::MIN_KV_BOOTSTRAP_FETCH_TIMEOUT_CAP_MS),
        );
    }

    // ---- sweep early exit (SweepResult::FleetCold) ----

    /// A replica that has published a block size but not yet its hashing mode
    /// must not vet anything. That window is real — `add_worker` writes the two
    /// in sequence — and a peer's tree vetted inside it is compared against an
    /// identity this replica has not established, so every grafted block is one
    /// it can never match.
    #[tokio::test]
    async fn the_sweep_refuses_to_vet_on_a_half_published_hash_config() {
        let deps = sweep_deps(vec!["http://peer:30000".into()], 64, &["http://w1:30000"]);
        // Undo the mode half that `sweep_deps` publishes, keeping the size.
        let half = BlockSizeOracle::new();
        half.try_set(64).unwrap();
        assert_eq!(half.get(), Some(64));
        assert_eq!(half.hash_config(), None);
        let deps = BootstrapDeps {
            oracle: half,
            ..deps
        };

        let ctx = SweepCtx {
            http: &deps.http,
            peers: &deps.peers,
            bootstrap: &deps.bootstrap,
            live_workers: &deps.live_workers,
            oracle: &deps.oracle,
            freshness_floor: Instant::now(),
        };
        let mut state = SweepState::new();
        let reason = Mutex::new(None);
        let pass = sweep_peers(
            &ctx,
            &[worker_id("http://w1:30000", 0)],
            &mut state,
            &reason,
        )
        .await;
        assert!(
            matches!(pass, SweepPass::KeepLooking),
            "no peer may be consulted before the hashing identity is established",
        );
        assert_eq!(
            deps.bootstrap.peer_outcome_counts(),
            vec![],
            "and no fetch was attempted, so no peer outcome is tallied",
        );
    }

    fn sweep_deps(peers: Vec<String>, block_size: u32, live: &[&str]) -> BootstrapDeps {
        sweep_deps_with_ctrl(peers, block_size, live).0
    }

    /// [`sweep_deps`], keeping the pump end of the control channel.
    fn sweep_deps_with_ctrl(
        peers: Vec<String>,
        block_size: u32,
        live: &[&str],
    ) -> (BootstrapDeps, mpsc::Receiver<PumpControl>) {
        let registry = Arc::new(PeerRegistry::new());
        registry.replace(peers);
        let oracle = BlockSizeOracle::new();
        oracle.try_set(block_size).expect("first set establishes");
        // `hash_config` is both-or-neither, and the sweep refuses to vet
        // without it.
        oracle.set_bigram(false);
        let (ctrl_tx, ctrl_rx) = mpsc::channel(8);
        let deps = BootstrapDeps {
            http: reqwest::Client::new(),
            peers: registry,
            bootstrap: Arc::new(BootstrapTracker::new(Duration::from_secs(3600))),
            live_workers: Arc::new(Mutex::new(live.iter().map(|u| worker_id(u, 0)).collect())),
            oracle,
            ctrl_tx,
        };
        (deps, ctrl_rx)
    }

    /// A snapshot with real content, carried by `carrier` only.
    fn warm_snapshot(carrier: &str, block_size: u32) -> PeerSnapshot {
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size,
            is_bigram: false,
            producer_ready: true,
            workers: vec![WireWorker {
                url: carrier.into(),
                dp_rank: 0,
            }],
            cursors: vec![(0, 5)],
            nodes: vec![SnapshotNode {
                parent: None,
                block_hash: 111,
                workers: vec![0],
                tiers: vec![],
            }],
            empty_ranks: vec![],
        }
    }

    /// A fleet whose siblings all hold empty trees settles cold at the end of
    /// the first pass instead of holding readiness until the deadline.
    #[tokio::test]
    async fn sweep_settles_early_when_every_peer_is_cold() {
        let (p1, q1) = serve_snapshot_sequence(vec![witness_snapshot(&[])]).await;
        let (p2, q2) = serve_snapshot_sequence(vec![witness_snapshot(&[])]).await;
        let deps = sweep_deps(vec![p1, p2], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        // A 3600s deadline would hang a broken test; the outer timeout fails
        // fast instead, and reaching it IS the regression.
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(
                &deps,
                &deps.bootstrap.register(&ranks),
                Duration::from_secs(3600),
                Instant::now(),
            ),
        )
        .await
        .expect("a cold fleet settles immediately, not at the deadline");
        assert!(
            matches!(result, SweepResult::FleetCold { peers_tried: 2 }),
            "every peer proved empty, so the sweep must not wait out the deadline",
        );
        // One fetch per peer: the verdict lands at the end of the FIRST pass,
        // not after a cooldown round of re-fetches.
        assert_eq!(q1.lock().expect("queries lock").len(), 1);
        assert_eq!(q2.lock().expect("queries lock").len(), 1);
    }

    /// An unreachable peer says nothing about its warmth — it may be a warm
    /// sibling mid-restart — so its presence must veto the early cold exit.
    #[tokio::test]
    async fn sweep_waits_out_the_deadline_when_a_peer_is_unreachable() {
        let (cold, _q) = serve_snapshot_sequence(vec![witness_snapshot(&[])]).await;
        // Nothing listens on port 1: a fast connection-refused, not a hang.
        let deps = sweep_deps(
            vec![cold, "http://127.0.0.1:1".into()],
            64,
            &["http://w1:30000"],
        );
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = sweep_until_deadline(
            &deps,
            &deps.bootstrap.register(&ranks),
            Duration::from_millis(500),
            Instant::now(),
        )
        .await;
        assert!(
            matches!(result, SweepResult::TimedOut { peers_tried: 2, .. }),
            "an unanswered peer keeps the sweep waiting until the deadline",
        );
    }

    /// A warm peer that holds no blocks for the bootstrapping ranks is still a
    /// warm peer: the fleet HAS state, so the sweep keeps looking rather than
    /// declaring the fleet cold. And the answer is still an attempt: tallied,
    /// and named as the reason the deadline ran out.
    #[tokio::test]
    async fn sweep_waits_out_the_deadline_when_a_peer_is_warm_but_uncovering() {
        let (warm, q) = serve_snapshot_sequence(vec![warm_snapshot("http://w2:30000", 64)]).await;
        let deps = sweep_deps(vec![warm], 64, &["http://w1:30000", "http://w2:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = sweep_until_deadline(
            &deps,
            &deps.bootstrap.register(&ranks),
            Duration::from_millis(500),
            Instant::now(),
        )
        .await;
        let SweepResult::TimedOut {
            peers_tried: 1,
            last_reason,
        } = result
        else {
            panic!("a warm peer keeps the sweep waiting even when it covers nothing we need");
        };
        assert!(
            last_reason
                .as_deref()
                .is_some_and(|r| r.contains("tracks none of the ranks")),
            "the timeout must name why the answering peer was not used; got {last_reason:?}",
        );
        assert!(
            deps.bootstrap
                .peer_outcome_counts()
                .iter()
                .any(|(label, n)| *label == "cold_peer" && *n >= 1),
            "a non-covering answer is a fetch attempt and must be tallied; got {:?}",
            deps.bootstrap.peer_outcome_counts(),
        );
        assert!(
            q.lock().expect("queries lock").len() <= 2,
            "a non-covering answer cools the peer down; it is not re-fetched every pass",
        );
    }

    /// Permanent rejection (here a block-size mismatch) is as terminal as an
    /// empty tree: retrying cannot change the answer, so a fleet of nothing
    /// but incompatible peers also settles early.
    #[tokio::test]
    async fn sweep_settles_early_when_every_peer_is_permanently_rejected() {
        let (p, q) = serve_snapshot_sequence(vec![warm_snapshot("http://w1:30000", 999)]).await;
        let deps = sweep_deps(vec![p], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(
                &deps,
                &deps.bootstrap.register(&ranks),
                Duration::from_secs(3600),
                Instant::now(),
            ),
        )
        .await
        .expect("an incompatible fleet settles immediately, not at the deadline");
        assert!(
            matches!(result, SweepResult::FleetCold { peers_tried: 1 }),
            "permanent rejection must count toward the all-cold verdict",
        );
        assert_eq!(
            q.lock().expect("queries lock").len(),
            1,
            "a permanently rejected peer is never re-fetched",
        );
    }

    /// An empty candidate set is not a cold fleet: `all()` on an empty
    /// iterator is vacuously true, and the peer set legitimately dips to
    /// empty during EndpointSlice repacks — exactly the race the retry loop
    /// exists for (worker discovery wins against the peer watch). This
    /// registry HAS seen peers, so `known_to_have_no_peers` stays false and
    /// only the verdict's own `is_empty` guard stands between this and a
    /// wrong `FleetCold`.
    #[tokio::test]
    async fn sweep_does_not_settle_cold_on_a_transiently_empty_peer_set() {
        let deps = sweep_deps(vec!["http://127.0.0.1:1".into()], 64, &["http://w1:30000"]);
        deps.peers.replace(vec![]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = sweep_until_deadline(
            &deps,
            &deps.bootstrap.register(&ranks),
            Duration::from_millis(500),
            Instant::now(),
        )
        .await;
        assert!(
            matches!(result, SweepResult::TimedOut { .. }),
            "an empty candidate set is no information, not a cold fleet",
        );
    }

    /// A registry that never had peers maps to `NoPeers`, not `FleetCold`:
    /// the two deliver identically but log differently, and "no siblings
    /// exist" is a different operational fact from "siblings proved empty".
    #[tokio::test]
    async fn sweep_reports_no_peers_when_discovery_confirms_none() {
        let deps = sweep_deps(vec![], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(
                &deps,
                &deps.bootstrap.register(&ranks),
                Duration::from_secs(3600),
                Instant::now(),
            ),
        )
        .await
        .expect("a confirmed-empty discovery settles immediately");
        assert!(matches!(result, SweepResult::NoPeers));
    }

    /// A peer that warms up mid-sweep must be FOUND, not FleetColded on a
    /// stale cold classification. The unreachable second peer is what keeps
    /// the sweep alive past the cold peer's cooldown: without an unclassified
    /// candidate in the mix, the fleet settles cold at the end of the pass
    /// that classifies it — before its cooldown ever lets a warmer answer
    /// through — by design.
    #[tokio::test]
    async fn sweep_finds_a_peer_that_warms_during_the_sweep() {
        let warm = warm_snapshot("http://w1:30000", 64);
        let (flipper, _q) = serve_snapshot_sequence(vec![witness_snapshot(&[]), warm]).await;
        let deps = sweep_deps(
            vec![flipper, "http://127.0.0.1:1".into()],
            64,
            &["http://w1:30000"],
        );
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(
                &deps,
                &deps.bootstrap.register(&ranks),
                Duration::from_secs(3600),
                Instant::now(),
            ),
        )
        .await
        .expect("the warm answer ends the sweep on its refetch pass");
        assert!(
            matches!(result, SweepResult::Found(_)),
            "a peer that warms mid-sweep must be found, not settled cold",
        );
    }

    /// The cold→warm remove itself: a peer re-fetched as warm-but-uncovering
    /// must drop its cold witness, or its stale classification would let the
    /// fleet settle cold while a warm peer exists. Drives `sweep_peers`
    /// directly so the refetch is not hostage to pass timing.
    #[tokio::test]
    async fn a_warm_answer_erases_a_peers_cold_witness() {
        let uncovering = warm_snapshot("http://w2:30000", 64);
        let (flipper, _q) = serve_snapshot_sequence(vec![witness_snapshot(&[]), uncovering]).await;
        // The unreachable second candidate keeps each pass from settling
        // cold on the flipper's classification alone.
        let deps = sweep_deps(
            vec![flipper.clone(), "http://127.0.0.1:1".into()],
            64,
            &["http://w1:30000", "http://w2:30000"],
        );
        let ctx = SweepCtx {
            http: &deps.http,
            peers: &deps.peers,
            bootstrap: &deps.bootstrap,
            live_workers: &deps.live_workers,
            oracle: &deps.oracle,
            freshness_floor: Instant::now(),
        };
        let last_reason = Mutex::new(None);
        let mut state = SweepState::new();
        let ranks = vec![worker_id("http://w1:30000", 0)];

        let pass1 = sweep_peers(&ctx, &ranks, &mut state, &last_reason).await;
        assert!(matches!(pass1, SweepPass::KeepLooking));
        assert!(state.cold_witnessed.contains(&flipper));

        // Skip the cooldown so pass 2 re-fetches immediately.
        state.cooldown.clear();
        let pass2 = sweep_peers(&ctx, &ranks, &mut state, &last_reason).await;
        assert!(matches!(pass2, SweepPass::KeepLooking));
        assert!(
            !state.cold_witnessed.contains(&flipper),
            "the warm answer must erase the cold witness",
        );
        // And with the witness gone, a fleet of {warm-uncovering, rejected}
        // is NOT cold — the warm peer might cover the ranks later.
        state.note_permanent_reject("http://c:30000");
        assert!(!state.fleet_is_cold(&[flipper.clone(), "http://c:30000".into()]));
    }

    /// The verdict primitives, pinned directly: cold counts, unknown vetoes,
    /// rejection counts, empty is not cold.
    #[test]
    fn fleet_is_cold_only_when_every_candidate_proved_hopeless() {
        let mut state = SweepState::new();
        let a = "http://a:30000".to_string();
        let cold = witness_snapshot(&[]);
        let warm = warm_snapshot("http://w1:30000", 64);
        let candidates = std::slice::from_ref(&a);

        // No information at all is not cold.
        assert!(!state.fleet_is_cold(&[]));
        assert!(!state.fleet_is_cold(candidates));

        state.note_answer(&a, &cold);
        assert!(state.fleet_is_cold(candidates));

        // An unreachable spell erases the witness: unknown warmth vetoes.
        state.note_unreachable(&a);
        assert!(!state.fleet_is_cold(candidates));

        // Re-cold, then a warm answer erases it again.
        state.note_answer(&a, &cold);
        state.note_answer(&a, &warm);
        assert!(!state.fleet_is_cold(candidates));

        // Settled-empty (ready flag set, zero nodes) is a cold witness too:
        // `holds_no_state` looks only at nodes.
        let settled_empty = PeerSnapshot {
            producer_ready: true,
            nodes: vec![],
            ..warm_snapshot("http://a:30000", 64)
        };
        state.note_answer(&a, &settled_empty);
        assert!(state.fleet_is_cold(candidates));

        // A peer still bootstrapping reports `producer_ready: false` while
        // holding nodes: vetting refuses it, but it is not a cold witness.
        let mid_bootstrap = PeerSnapshot {
            producer_ready: false,
            ..warm_snapshot("http://w1:30000", 64)
        };
        assert!(mid_bootstrap.is_cold(), "vetting still refuses to graft it");
        state.note_answer(&a, &mid_bootstrap);
        assert!(
            !state.fleet_is_cold(candidates),
            "a peer still bootstrapping with a non-empty tree is not a cold witness",
        );

        // Permanent rejection is terminal on its own.
        let mut state = SweepState::new();
        state.note_permanent_reject(&a);
        assert!(state.fleet_is_cold(candidates));
    }

    /// End to end for the same hazard: the only candidate is a sibling
    /// mid-bootstrap that already holds blocks. The sweep must keep looking
    /// (and find it once it settles), not settle cold on the first pass.
    #[tokio::test]
    async fn sweep_waits_for_a_sibling_that_is_still_bootstrapping() {
        let mid_bootstrap = PeerSnapshot {
            producer_ready: false,
            ..warm_snapshot("http://w1:30000", 64)
        };
        let (peer, _q) =
            serve_snapshot_sequence(vec![mid_bootstrap, warm_snapshot("http://w1:30000", 64)])
                .await;
        let deps = sweep_deps(vec![peer], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(
                &deps,
                &deps.bootstrap.register(&ranks),
                Duration::from_secs(3600),
                Instant::now(),
            ),
        )
        .await
        .expect("the peer settles on its second answer, well inside the deadline");
        assert!(
            matches!(result, SweepResult::Found(_)),
            "a peer that holds a tree but has not settled must keep the sweep alive",
        );
    }

    /// Cooldown sits a peer out for exactly the configured passes, then lets
    /// it be refetched — a peer whose entry reaches zero is never stuck.
    #[test]
    fn cooling_sits_out_exactly_the_configured_passes() {
        let mut state = SweepState::new();
        state.note_sit_out("http://a:30000");
        for _ in 0..PEER_COOLDOWN_PASSES {
            assert!(state.cooling("http://a:30000"));
        }
        assert!(
            !state.cooling("http://a:30000"),
            "a peer at zero is refetched, never stuck",
        );
        assert!(!state.cooling("http://never-seen:30000"));
    }

    /// Each further miss by the same peer doubles its sit-out, capped at
    /// `MAX_PEER_COOLDOWN` worth of passes; both miss kinds share the count,
    /// and another peer starts from the base.
    #[test]
    fn cooldown_backs_off_exponentially_per_peer_up_to_the_cap() {
        assert_eq!(MAX_PEER_COOLDOWN_PASSES, 120, "30s of 250ms passes");
        let sat_out = |state: &mut SweepState, peer: &str| {
            let mut passes = 0;
            while state.cooling(peer) {
                passes += 1;
            }
            passes
        };
        let a = "http://a:30000";
        let mut state = SweepState::new();
        let mut seen = Vec::new();
        for strike in 0..8 {
            if strike % 2 == 0 {
                state.note_sit_out(a);
            } else {
                state.note_unreachable(a);
            }
            seen.push(sat_out(&mut state, a));
        }
        assert_eq!(seen, [4, 8, 16, 32, 64, 120, 120, 120]);

        state.note_sit_out("http://b:30000");
        assert_eq!(
            sat_out(&mut state, "http://b:30000"),
            PEER_COOLDOWN_PASSES,
            "strikes are per peer",
        );
        assert_eq!(peer_cooldown_passes(u32::MAX), MAX_PEER_COOLDOWN_PASSES);
    }

    /// A same-length membership swap mid-pass — one pod replaced at constant
    /// replica count, the rolling-update norm — must veto the cold-fleet
    /// verdict even though the registry LENGTH is unchanged: the stale
    /// candidate list proved cold, but the live newcomer was never consulted.
    /// The swapping server stands in for the EndpointSlice informer landing a
    /// `replace` while the pass was in flight.
    #[tokio::test]
    async fn sweep_survives_a_same_length_membership_swap_mid_pass() {
        let cold = witness_snapshot(&[]);
        let registry = Arc::new(PeerRegistry::new());
        let reg_in_handler = Arc::clone(&registry);
        let app = axum::Router::new().route(
            SNAPSHOT_PATH,
            axum::routing::get(move || {
                let registry = Arc::clone(&reg_in_handler);
                let snap = cold.clone();
                async move {
                    // First fetch: the informer swaps this peer for one that
                    // is down — same length, different membership.
                    registry.replace(vec!["http://127.0.0.1:1".to_string()]);
                    axum::Json(snap)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        registry.replace(vec![format!("http://{addr}")]);

        let deps = BootstrapDeps {
            peers: registry,
            ..sweep_deps(vec![], 64, &["http://w1:30000"])
        };
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = sweep_until_deadline(
            &deps,
            &deps.bootstrap.register(&ranks),
            Duration::from_millis(500),
            Instant::now(),
        )
        .await;
        assert!(
            matches!(result, SweepResult::TimedOut { .. }),
            "a mid-pass membership swap must veto the verdict, length unchanged or not",
        );
    }

    /// `deliver_bootstrap` is the one recording point for the sweep metric:
    /// every terminal verdict tallies exactly once, so `fleet_cold` stays
    /// distinguishable from `timed_out` on dashboards.
    #[tokio::test]
    async fn deliver_bootstrap_records_the_sweep_verdict_once() {
        let (deps, mut ctrl_rx) = sweep_deps_with_ctrl(vec![], 64, &[]);
        let bootstrap = Arc::clone(&deps.bootstrap);
        deliver_bootstrap(
            &deps,
            vec![],
            SweepResult::FleetCold { peers_tried: 3 },
            Duration::from_secs(5),
        )
        .await;
        assert!(
            matches!(
                ctrl_rx.recv().await,
                Some(PumpControl::AbandonBootstrap { .. })
            ),
            "a cold fleet releases its ranks through the same abandon path",
        );
        assert!(
            bootstrap.sweep_result_counts().contains(&("fleet_cold", 1)),
            "the verdict must be tallied exactly once; got {:?}",
            bootstrap.sweep_result_counts(),
        );
    }

    // ---- per-rank settle, freshness ratchet, resolved ranks ----

    /// A warm sibling (a real tree, carried by `carrier`) that names each of
    /// `empty` in `empty_ranks`: subscribed to it, holding nothing for it.
    fn warm_snapshot_holding_nothing_for(carrier: &str, empty: &[&str]) -> PeerSnapshot {
        PeerSnapshot {
            empty_ranks: empty
                .iter()
                .map(|u| WireWorker {
                    url: (*u).to_string(),
                    dp_rank: 0,
                })
                .collect(),
            ..warm_snapshot(carrier, 64)
        }
    }

    /// The fresh-engine incident from the consumer side: every sibling is warm
    /// but each is itself waiting on the new rank, so none will ever cover it.
    /// Once they all say so the rank settles cold rather than waiting out the
    /// deadline.
    #[tokio::test]
    async fn sweep_settles_a_rank_every_peer_holds_nothing_for() {
        let body = warm_snapshot_holding_nothing_for("http://w2:30000", &["http://w1:30000"]);
        let (p1, _q1) = serve_snapshot_sequence(vec![body.clone()]).await;
        let (p2, _q2) = serve_snapshot_sequence(vec![body]).await;
        let (deps, mut ctrl_rx) =
            sweep_deps_with_ctrl(vec![p1, p2], 64, &["http://w1:30000", "http://w2:30000"]);
        let obligations = deps.bootstrap.register(&[worker_id("http://w1:30000", 0)]);
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(
                &deps,
                &obligations,
                Duration::from_secs(3600),
                Instant::now(),
            ),
        )
        .await
        .expect("a rank no sibling can supply settles now, not at the deadline");
        assert!(matches!(result, SweepResult::RanksResolved));
        match ctrl_rx.try_recv() {
            Ok(PumpControl::AbandonBootstrap { obligations: sent }) => {
                assert_eq!(sent, obligations, "released under its own incarnation")
            }
            other => panic!("expected the rank released cold, got {other:?}"),
        }
    }

    /// Per rank, not per sweep: a rank some sibling might still supply keeps
    /// the sweep alive, but must not hold back the one nobody can.
    #[tokio::test]
    async fn sweep_settles_only_the_ranks_every_peer_holds_nothing_for() {
        let body = warm_snapshot_holding_nothing_for("http://w2:30000", &["http://w1:30000"]);
        let (p, _q) = serve_snapshot_sequence(vec![body]).await;
        let (deps, mut ctrl_rx) = sweep_deps_with_ctrl(
            vec![p],
            64,
            &["http://w1:30000", "http://w2:30000", "http://w3:30000"],
        );
        let barren = worker_id("http://w1:30000", 0);
        let unknown = worker_id("http://w3:30000", 0);
        let obligations = deps.bootstrap.register(&[barren.clone(), unknown.clone()]);
        let result = sweep_until_deadline(
            &deps,
            &obligations,
            Duration::from_millis(800),
            Instant::now(),
        )
        .await;
        assert!(
            matches!(result, SweepResult::TimedOut { .. }),
            "the rank the peer does not speak for keeps the sweep waiting",
        );
        match ctrl_rx.try_recv() {
            Ok(PumpControl::AbandonBootstrap { obligations: sent }) => {
                let ranks: Vec<_> = sent.into_iter().map(|(r, _)| r).collect();
                assert_eq!(
                    ranks,
                    vec![barren],
                    "only the rank nobody can supply settles"
                );
            }
            other => panic!("expected a mid-sweep release, got {other:?}"),
        }
        assert!(
            ctrl_rx.try_recv().is_err(),
            "nothing else is released mid-sweep"
        );
    }

    /// A sibling that does not name the rank may simply not have discovered it
    /// yet — the case `covers_any` exists for — so one such sibling vetoes the
    /// per-rank settle, however many others hold nothing.
    #[tokio::test]
    async fn sweep_keeps_waiting_on_a_rank_while_any_peer_is_silent_on_it() {
        let (names_it, _q1) = serve_snapshot_sequence(vec![warm_snapshot_holding_nothing_for(
            "http://w2:30000",
            &["http://w1:30000"],
        )])
        .await;
        let (silent, _q2) =
            serve_snapshot_sequence(vec![warm_snapshot("http://w2:30000", 64)]).await;
        let (deps, mut ctrl_rx) = sweep_deps_with_ctrl(
            vec![names_it, silent],
            64,
            &["http://w1:30000", "http://w2:30000"],
        );
        let obligations = deps.bootstrap.register(&[worker_id("http://w1:30000", 0)]);
        let result = sweep_until_deadline(
            &deps,
            &obligations,
            Duration::from_millis(500),
            Instant::now(),
        )
        .await;
        assert!(matches!(result, SweepResult::TimedOut { .. }));
        assert!(ctrl_rx.try_recv().is_err(), "no rank released");
    }

    /// The per-rank verdict primitives: every candidate must speak for the rank
    /// — by naming it, or by being hopeless as a whole — and silence, an
    /// unreachable spell, or an empty candidate set vetoes.
    #[test]
    fn nothing_to_recover_needs_every_candidate_to_speak_for_the_rank() {
        let mut state = SweepState::new();
        let (a, b) = ("http://a:30000".to_string(), "http://b:30000".to_string());
        let candidates = [a.clone(), b.clone()];
        let rank = worker_id("http://w1:30000", 0);
        let other = worker_id("http://w9:30000", 0);
        let names_it = warm_snapshot_holding_nothing_for("http://w2:30000", &["http://w1:30000"]);

        assert!(
            !state.nothing_to_recover(&rank, &[]),
            "no candidates, no proof"
        );
        state.note_answer(&a, &names_it);
        assert!(
            !state.nothing_to_recover(&rank, &candidates),
            "b never answered"
        );
        state.note_answer(&b, &warm_snapshot("http://w2:30000", 64));
        assert!(
            !state.nothing_to_recover(&rank, &candidates),
            "b is silent on it"
        );
        state.note_answer(&b, &names_it);
        assert!(state.nothing_to_recover(&rank, &candidates));
        assert!(
            !state.nothing_to_recover(&other, &candidates),
            "evidence is per rank"
        );
        state.note_unreachable(&b);
        assert!(
            !state.nothing_to_recover(&rank, &candidates),
            "an unreachable spell erases the evidence",
        );
        state.note_answer(&b, &witness_snapshot(&[]));
        assert!(
            state.nothing_to_recover(&rank, &candidates),
            "an empty tree has nothing for any rank",
        );
        let mut state = SweepState::new();
        state.note_answer(&a, &names_it);
        state.note_permanent_reject(&b);
        assert!(
            state.nothing_to_recover(&rank, &candidates),
            "a peer we can never consume has nothing for us",
        );
    }

    /// Serve a REAL producer on the snapshot path, honouring `max_age_ms` the
    /// way the route does — the ratchet is only observable against the
    /// producer's own cache.
    async fn serve_producer(index: Arc<KvEventIndex>) -> String {
        use crate::state::kv_events::bootstrap::{MAX_AGE_PARAM, PRODUCER_CACHE_TTL};
        let app = axum::Router::new().route(
            SNAPSHOT_PATH,
            axum::routing::get(move |q: axum::extract::Query<HashMap<String, String>>| {
                let index = Arc::clone(&index);
                async move {
                    let max_age = q
                        .get(MAX_AGE_PARAM)
                        .and_then(|v| v.parse().ok())
                        .map_or(PRODUCER_CACHE_TTL, Duration::from_millis);
                    index.peer_snapshot_body(max_age).await.identity
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// The freshness ratchet: a peer that answered without covering our rank
    /// and has SINCE learned it must be able to say so. Against a floor fixed
    /// at sweep start alone, the producer would replay the export it built
    /// right after that floor for the whole sweep.
    #[tokio::test]
    async fn a_refetch_demands_an_export_newer_than_the_peers_last_answer() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(64).expect("first set establishes");
        oracle.set_bigram(false);
        let producer =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        let (ours, theirs) = (
            worker_id("http://w1:30000", 0),
            worker_id("http://w2:30000", 0),
        );
        producer.seed_stored_block_for_test(&theirs, 3, 222);
        let peer = serve_producer(Arc::clone(&producer)).await;

        let deps = sweep_deps(vec![peer], 64, &["http://w1:30000", "http://w2:30000"]);
        // Far enough behind the first export that reusing it under the floor
        // alone is not a matter of transit-latency luck.
        let floor = Instant::now();
        tokio::time::sleep(Duration::from_millis(200)).await;
        let ctx = SweepCtx {
            http: &deps.http,
            peers: &deps.peers,
            bootstrap: &deps.bootstrap,
            live_workers: &deps.live_workers,
            oracle: &deps.oracle,
            freshness_floor: floor,
        };
        let last_reason = Mutex::new(None);
        let mut state = SweepState::new();
        let ranks = vec![ours.clone()];

        let first = sweep_peers(&ctx, &ranks, &mut state, &last_reason).await;
        assert!(
            !matches!(first, SweepPass::Found(_)),
            "the peer does not know our rank yet",
        );
        // The peer learns our rank after answering; its cache still holds the
        // export it answered with, which does meet the sweep's floor.
        producer.seed_stored_block_for_test(&ours, 5, 111);
        state.cooldown.clear();
        let second = sweep_peers(&ctx, &ranks, &mut state, &last_reason).await;
        match second {
            SweepPass::Found(vetted) => assert_eq!(vetted.cursor_for(&ours), Some(5)),
            _ => panic!("the re-fetch must see a fresh export, not a replay of the last answer"),
        }
    }

    #[test]
    fn export_floor_is_the_later_of_the_sweep_floor_and_the_last_answer() {
        let mut state = SweepState::new();
        let floor = Instant::now();
        let peer = "http://a:30000";
        assert_eq!(state.export_floor(peer, floor), floor, "never answered");
        let answered = floor + Duration::from_secs(1);
        state.note_received(peer, answered);
        assert_eq!(state.export_floor(peer, floor), answered);
        assert_eq!(
            state.export_floor("http://b:30000", floor),
            floor,
            "per peer: another peer's answer ratchets nothing here",
        );
        let later_floor = answered + Duration::from_secs(1);
        assert_eq!(
            state.export_floor(peer, later_floor),
            later_floor,
            "the ratchet never loosens the sweep's own floor",
        );
    }

    /// Once the pump resolves a rank from its stream's origin, the sweep
    /// launched for it must stop. Here the only peer is warm but will never
    /// cover the rank, so only the rank leaving `Pending` can end the sweep
    /// before its deadline.
    #[tokio::test]
    async fn sweep_stops_once_every_rank_has_left_pending() {
        let (warm, _q) = serve_snapshot_sequence(vec![warm_snapshot("http://w2:30000", 64)]).await;
        let deps = sweep_deps(vec![warm], 64, &["http://w1:30000", "http://w2:30000"]);
        let rank = worker_id("http://w1:30000", 0);
        let obligations = deps.bootstrap.register(std::slice::from_ref(&rank));
        let tracker = Arc::clone(&deps.bootstrap);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            tracker.set(&rank, BootstrapState::Recovered);
        });
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(
                &deps,
                &obligations,
                Duration::from_secs(3600),
                Instant::now(),
            ),
        )
        .await
        .expect("a sweep with nobody left waiting must end, not run to the deadline");
        assert!(matches!(result, SweepResult::RanksResolved));
    }

    /// Obligations nobody is waiting on — resolved already, or superseded by a
    /// re-registration — cost no fetch at all.
    #[tokio::test]
    async fn sweep_fetches_nothing_for_obligations_nobody_waits_on() {
        let (warm, q) = serve_snapshot_sequence(vec![warm_snapshot("http://w1:30000", 64)]).await;
        let deps = sweep_deps(vec![warm], 64, &["http://w1:30000", "http://w2:30000"]);
        let resolved = worker_id("http://w1:30000", 0);
        let superseded = worker_id("http://w2:30000", 0);
        let obligations = deps
            .bootstrap
            .register(&[resolved.clone(), superseded.clone()]);
        deps.bootstrap.set(&resolved, BootstrapState::Recovered);
        deps.bootstrap.forget(std::slice::from_ref(&superseded));
        deps.bootstrap.register(std::slice::from_ref(&superseded));
        // The stale incarnation's obligation is still the one handed over.
        assert_ne!(deps.bootstrap.epoch_of(&superseded), Some(obligations[1].1));

        let result =
            sweep_until_deadline(&deps, &obligations, Duration::from_secs(5), Instant::now()).await;
        assert!(matches!(result, SweepResult::RanksResolved));
        assert!(
            q.lock().expect("queries lock").is_empty(),
            "no fetch for ranks nobody is waiting on",
        );
    }

    /// `RanksResolved` owes its ranks nothing: they already left `Pending`. It
    /// is still tallied, so the sweep counter sums to the sweeps run.
    #[tokio::test]
    async fn deliver_bootstrap_sends_nothing_for_resolved_ranks() {
        let (deps, mut ctrl_rx) = sweep_deps_with_ctrl(vec![], 64, &[]);
        let bootstrap = Arc::clone(&deps.bootstrap);
        deliver_bootstrap(
            &deps,
            vec![(worker_id("http://w1:30000", 0), 1)],
            SweepResult::RanksResolved,
            Duration::from_secs(5),
        )
        .await;
        drop(deps);
        assert!(
            ctrl_rx.recv().await.is_none(),
            "no control message for ranks that already left Pending",
        );
        assert!(bootstrap
            .sweep_result_counts()
            .contains(&("ranks_resolved", 1)));
    }
}
