// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle bundle for the KV-event index.
//!
//! Couples the three submodules that are independent in their own right but
//! always operate together in production:
//!
//! - [`HashTree`] — the cache-aware routing index keyed by SGLang block hash.
//! - [`EngineReportedLoadTable`] — engine-reported per-worker load.
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

use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::block_size_oracle::BlockSizeOracle;
use super::bootstrap::{
    BootstrapState, BootstrapTracker, PeerRegistry, RankOutcome, VettedSnapshot,
    SNAPSHOT_FETCH_CONNECT_TIMEOUT, SNAPSHOT_FETCH_READ_TIMEOUT,
};
use super::discovery::{fetch_event_config, EventConfig};
use super::subscriber::{KvEventSubscriberRegistry, SubKind, WorkerEvent};
use super::tally::{EventKind, EventTally};
use super::tree::{HashTree, KvWorkerId, Tiers};
use super::wire::{KvCacheEvent, KvEventBatch};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
use coordinator::bootstrap_coordinator;
use fallback::{demote_unproven_rank, fail_rank, resolve_from_origin, resolve_gap};
use graft::{apply_snapshot, leaves_gap, still_owed};
use probe::{
    spawn_splice_probe, PendingProof, ProbeTarget, SpliceVerdict, MAX_UNKNOWN_PROBES,
    SPLICE_PROOF_SWEEP_INTERVAL, SPLICE_PROOF_TIMEOUT,
};
use producer::CachedSnapshot;
use sweep::{
    snapshot_fetch_timeout, SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE, SNAPSHOT_FETCH_TIMEOUT_FLOOR,
};

mod coordinator;
mod fallback;
mod graft;
mod probe;
mod producer;
mod sweep;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

pub use producer::SnapshotBody;

/// Channel buffer between the subscriber registry and the pump task.
///
/// Bounded so a misbehaving publisher cannot exhaust memory.  Realistic
/// per-worker event rates are < 1 kHz; a 1024-deep buffer absorbs a
/// half-second burst at 2 kHz before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

/// Per-rank cap on batches held back while that rank is
/// [`BootstrapState::Pending`].
///
/// A rank that overflows this cannot be spliced (see `fail_rank`), so the
/// cap trades a small amount of memory for the chance to bootstrap at all.
/// Sized to match `EVENT_CHANNEL_BUFFER`: if the pump is that far behind, the
/// snapshot is not arriving in time anyway.
const PENDING_BATCH_LIMIT: usize = 1024;

/// Depth of the obligation queue feeding the coordinator, sized well past a
/// fleet's worker count. Overflow falls back as [`KvEventIndex::enqueue_bootstrap`]
/// describes.
const BOOTSTRAP_QUEUE_DEPTH: usize = 1024;

/// Sequence number of the FIRST batch a publisher ever emits.
///
/// SGLang's `ZmqEventPublisher` numbers batches from `itertools.count()`, and it
/// is constructed once per scheduler process, alongside an empty radix cache. A
/// `Pending` rank whose first held batch carries this number has therefore
/// received its publisher's stream from the beginning: no block exists on that
/// engine that the held batches do not describe, so no sibling can hand over
/// anything the rank lacks, and the rank is resolved on the spot rather than
/// swept for. See `resolve_from_origin`.
///
/// Only an exact match on the FIRST held batch counts. A first batch at 1
/// means batch 0 was missed — ZMQ's slow-joiner window drops whatever is
/// published before the SUB filter reaches the publisher — and that batch may
/// have stored blocks, so the rank keeps waiting for a snapshot. A batch 0
/// arriving later is a publisher restart instead; see the regression arms in
/// `pump_loop`.
const STREAM_ORIGIN_SEQ: i64 = 0;

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
    /// A worker was removed: drop everything the pump keeps for these ranks —
    /// held batches, pending splice proofs, tree carriers and cursors — so a
    /// re-added worker's fresh publisher inherits none of it.
    ///
    /// `done` fires once teardown has run, so nothing of these ranks survives
    /// `remove_worker`.
    ForgetRanks {
        ranks: Vec<KvWorkerId>,
        done: Option<oneshot::Sender<()>>,
    },
    /// Result of asking the fleet whether a rank's publisher moved past the
    /// watermark of a snapshot whose splice was never proven locally.
    ///
    /// The probe runs off-pump because it does network I/O; the verdict comes
    /// back here so the tree write stays on the single writer.
    ///
    /// `epoch` and `watermark` name the graft the verdict is about; see
    /// [`PendingProof::watermark`].
    SpliceProbe {
        rank: KvWorkerId,
        epoch: u64,
        watermark: i64,
        verdict: SpliceVerdict,
    },
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

/// Ranks whose socket port is representable for a publisher range. This is
/// shared by lifecycle bookkeeping and the subscriber registry contract so an
/// expected load rank always has a corresponding SUB socket.
fn subscribable_ranks(port_base: u16, dp_size: u32) -> Vec<u32> {
    let port_base = u32::from(port_base);
    (0..dp_size)
        .filter(|rank| port_base.saturating_add(*rank) <= u32::from(u16::MAX))
        .collect()
}

/// The read-only handles the `/metrics` scrape pulls the KV storage-tier
/// series from. Narrower than an [`KvEventIndex`] handle on purpose: a route
/// has no business calling `add_worker` / `remove_worker` / `shutdown`.
#[derive(Clone)]
pub struct KvIndexMetrics {
    pub(crate) tree: Arc<HashTree>,
    pub(crate) tally: Arc<EventTally>,
}

impl KvIndexMetrics {
    pub fn new(tree: Arc<HashTree>, tally: Arc<EventTally>) -> Self {
        Self { tree, tally }
    }
}

/// Obligations handed to the coordinator and the instant their ranks began
/// holding; a peer's export must be newer to splice.
struct ObligationBatch {
    obligations: Vec<(KvWorkerId, u64)>,
    /// Folded into [`PendingSweep::freshness_floor`], which the sweep asks with.
    ///
    /// [`PendingSweep::freshness_floor`]: coordinator::PendingSweep::freshness_floor
    holding_since: Instant,
    /// Whether this batch may ride a sweep already in flight.
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
    /// Used by discovery, whose workers land during the first fetch; a rank
    /// that gaps this way still has its one retry.
    Permitted,
    /// Wait for a sweep that asks on this batch's behalf.
    ///
    /// Used by a gap retry, for which riding along is equivalent to dropping
    /// it: `gap_retried` caps it at one, and a snapshot taken before the rank
    /// resumed holding re-gaps by construction. Deferring costs one loop
    /// iteration, since the coordinator re-enters `take_pending` as soon as it
    /// has delivered.
    Refused,
}

/// Bundle of `HashTree` + `KvEventSubscriberRegistry` + pump task.
///
/// Construct one instance per router process and hand it to the worker
/// manager as `Option<Arc<KvEventIndex>>` — `None` disables the cache-aware
/// routing path entirely.
pub struct KvEventIndex {
    tree: Arc<HashTree>,
    maintain_tree: bool,
    subscribers: Arc<KvEventSubscriberRegistry>,
    /// Second registry subscribing to the load topic (one per worker rank),
    /// feeding `LoadStat` snapshots into `engine_reported_load`. Shares the pump
    /// channel with `subscribers`; keyed independently so KV and load
    /// subscribers for the same worker don't collide.
    load_subscribers: Arc<KvEventSubscriberRegistry>,
    /// Engine-reported per-worker load, written by the pump from
    /// `WorkerEvent::Load` and captured at request ingress.
    engine_reported_load: Arc<EngineReportedLoadTable>,
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
    /// restart from 0.
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    /// Applied events by kind and storage medium, for the `/metrics` scrape.
    /// Written only by the pump.
    tally: Arc<EventTally>,
    /// Per-rank bootstrap progress; also what `/readyz` consults.
    bootstrap: Arc<BootstrapTracker>,
    /// Sibling replicas a snapshot may be pulled from. See [`PeerRegistry`].
    peers: Arc<PeerRegistry>,
    /// Control channel into the pump, so snapshot grafting happens on the
    /// single writer rather than in the bootstrap task.
    ctrl_tx: mpsc::Sender<PumpControl>,
    /// Client for snapshot fetches. Not `http`: its 2s total timeout suits
    /// `/server_info` but cannot fit a multi-megabyte body, and every large
    /// snapshot would be booked `unreachable`.
    snapshot_http: reqwest::Client,
    /// Obligations waiting for the coordinator to fold them into the sweep that
    /// is in flight, or to start one. See [`bootstrap_coordinator`].
    bootstrap_tx: mpsc::Sender<ObligationBatch>,
    /// Last built snapshot; see [`KvEventIndex::peer_snapshot_body`].
    snapshot_cache: Arc<AsyncMutex<Option<CachedSnapshot>>>,
    /// Worker-sourced `page_size` shared with prefix providers.
    /// `add_worker` calls `try_set(cfg.block_size)` so the first worker
    /// establishes the value; subsequent workers that disagree are
    /// rejected (logged + not subscribed). Prefix providers read it at routing
    /// time to size their `compute_block_hashes` calls.
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl KvEventIndex {
    /// Build an empty index and spawn the pump task.
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
    /// the same oracle the index seeds is available to prefix providers.
    /// Tests use this to pre-populate the
    /// oracle and exercise the mismatch-rejection path.
    pub fn new_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::new_with_mode(
            http,
            block_size_oracle,
            Arc::new(BootstrapTracker::disabled()),
            true,
        )
    }

    /// Constructor that enables peer bootstrap with the supplied tracker.
    pub fn new_with_bootstrap(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        bootstrap: Arc<BootstrapTracker>,
    ) -> Arc<Self> {
        Self::new_with_mode(http, block_size_oracle, bootstrap, true)
    }

    /// Discovers worker hash metadata only: seeds the shared [`BlockSizeOracle`]
    /// but neither subscribes to KV events nor maintains the local tree, because
    /// an external Indexer is the routing signal.
    pub fn new_metadata_only_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::new_with_mode(
            http,
            block_size_oracle,
            Arc::new(BootstrapTracker::disabled()),
            false,
        )
    }

    fn new_with_mode(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        bootstrap: Arc<BootstrapTracker>,
        maintain_tree: bool,
    ) -> Arc<Self> {
        // Connect and read timeouts cut a gone or stalled peer; the total
        // bounds a progressing transfer to a fraction of the deadline so the
        // sweep can reach another candidate (see `snapshot_fetch_timeout`).
        let per_fetch = snapshot_fetch_timeout(bootstrap.timeout(), bootstrap.fetch_cap());
        // A short `--kv-bootstrap-timeout-ms` can derive a per-fetch bound
        // below the floor that the cap cannot lift; warn once so the operator
        // raises the deadline.
        if bootstrap.enabled() && per_fetch < SNAPSHOT_FETCH_TIMEOUT_FLOOR {
            warn!(
                per_fetch_ms = per_fetch.as_millis(),
                bootstrap_timeout_ms = bootstrap.timeout().as_millis(),
                fetch_cap_ms = bootstrap.fetch_cap().as_millis(),
                floor_ms = SNAPSHOT_FETCH_TIMEOUT_FLOOR.as_millis(),
                suggested_bootstrap_timeout_ms = (SNAPSHOT_FETCH_TIMEOUT_FLOOR
                    * SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE)
                    .as_millis(),
                "kv-bootstrap: the per-fetch timeout derived from --kv-bootstrap-timeout-ms \
                 is below the floor a multi-megabyte snapshot needs, so peers will be \
                 booked unreachable and every rank will boot cold; raise \
                 --kv-bootstrap-timeout-ms (the fetch cap cannot lift this on its own)",
            );
        }
        let snapshot_http = match reqwest::Client::builder()
            .connect_timeout(SNAPSHOT_FETCH_CONNECT_TIMEOUT)
            .read_timeout(SNAPSHOT_FETCH_READ_TIMEOUT)
            .timeout(per_fetch)
            // A sibling router never redirects this route, so a redirect is
            // either a misconfigured peer or a hostile one steering the fetch
            // — and its multi-gigabyte buffering budget — at an arbitrary
            // in-cluster URL. Refuse to follow: the 3xx lands as
            // `FetchAnswer::NoBody` and the peer is just not a source.
            .redirect(reqwest::redirect::Policy::none())
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                // Not silent: the fallback's total timeout is sized for
                // `/server_info`, so every large snapshot would then time out
                // and be booked `unreachable` with nothing pointing here.
                warn!(
                    error = %e,
                    "kv-bootstrap: snapshot client failed to build; falling back to the \
                     introspection client, whose timeout cannot fit a large snapshot",
                );
                http.clone()
            }
        };
        let tree = Arc::new(HashTree::new());
        let (tx, rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let (ctrl_tx, ctrl_rx) = mpsc::channel::<PumpControl>(16);
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(tx.clone()));
        let load_subscribers = Arc::new(KvEventSubscriberRegistry::with_kind(tx, SubKind::Load));
        let engine_reported_load = EngineReportedLoadTable::new();
        let cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>> = Arc::new(Mutex::new(HashMap::new()));
        let live_workers: Arc<Mutex<HashSet<KvWorkerId>>> = Arc::new(Mutex::new(HashSet::new()));
        let pump_cancel = CancellationToken::new();
        let peers = Arc::new(PeerRegistry::new());
        let (bootstrap_tx, bootstrap_rx) = mpsc::channel(BOOTSTRAP_QUEUE_DEPTH);
        let tally = Arc::new(EventTally::new());
        let pump = tokio::spawn(pump_loop(
            PumpDeps {
                tally: Arc::clone(&tally),
                tree: tree.clone(),
                engine_reported_load: engine_reported_load.clone(),
                cursors: cursors.clone(),
                live_workers: live_workers.clone(),
                bootstrap: Arc::clone(&bootstrap),
                peers: Arc::clone(&peers),
                snapshot_http: snapshot_http.clone(),
                bootstrap_tx: bootstrap_tx.clone(),
                ctrl_tx: ctrl_tx.downgrade(),
            },
            pump_cancel.clone(),
            rx,
            ctrl_rx,
        ));
        let index = Arc::new(Self {
            tree,
            maintain_tree,
            subscribers,
            load_subscribers,
            engine_reported_load,
            pump: Mutex::new(Some(pump)),
            pump_cancel: pump_cancel.clone(),
            workers: Mutex::new(HashMap::new()),
            http,
            live_workers,
            cursors,
            tally,
            bootstrap,
            peers,
            ctrl_tx,
            snapshot_http,
            bootstrap_tx,
            snapshot_cache: Arc::new(AsyncMutex::new(None)),
            block_size_oracle,
        });
        // Same gate as registration, so a coordinator exists exactly when
        // obligations can be produced.
        if index.peer_bootstrap_enabled() {
            tokio::spawn(bootstrap_coordinator(
                bootstrap_rx,
                Arc::downgrade(&index),
                pump_cancel,
            ));
        }
        index
    }

    /// Shared handle to the bootstrap tracker. `/readyz` reads it to decide
    /// whether initial bootstrap has settled; the metrics surface reads it for
    /// the per-rank state gauge.
    pub fn bootstrap(&self) -> Arc<BootstrapTracker> {
        Arc::clone(&self.bootstrap)
    }

    /// Shared accessor for the per-process block-size oracle.
    pub fn block_size_oracle(&self) -> Arc<BlockSizeOracle> {
        Arc::clone(&self.block_size_oracle)
    }

    /// Clone the underlying tree handle for cache-aware selection and
    /// metrics. The pump is the sole writer; callers should treat the
    /// returned handle as read-only.
    pub fn tree(&self) -> Arc<HashTree> {
        self.tree.clone()
    }

    /// Handles for the `/metrics` storage-tier series, or `None` when this
    /// router does not maintain a local tree.
    ///
    /// In metadata-only mode (an external Indexer is the routing signal) no KV
    /// subscription is opened, so every tier series would be a structural
    /// zero — while their own HELP text tells the operator to read a zero
    /// `CPU_PINNED` row as "the tier stream is not reaching the router". That
    /// is a different fault with a different fix, so emit nothing rather than
    /// a confidently wrong zero.
    pub fn metrics_source(&self) -> Option<KvIndexMetrics> {
        self.maintain_tree.then(|| KvIndexMetrics {
            tree: Arc::clone(&self.tree),
            tally: Arc::clone(&self.tally),
        })
    }

    /// Shared handle to the peer registry.
    pub fn peers(&self) -> Arc<PeerRegistry> {
        Arc::clone(&self.peers)
    }

    /// Shared accessor for the engine-load table. Load values are written solely by the pump
    /// (from `LoadStat` events); `add_worker` / `remove_worker` here manage
    /// the expected set and per-worker eviction.
    pub fn engine_reported_load(&self) -> Arc<EngineReportedLoadTable> {
        Arc::clone(&self.engine_reported_load)
    }

    /// Register a worker. If `preresolved` is `Some`, the caller has
    /// already fetched `/server_info` (worker manager path) and we skip
    /// the internal HTTP round-trip; otherwise (standalone callers,
    /// e.g. integration tests) we fall back to `fetch_event_config`.
    ///
    /// Opens one ZMQ SUB per advertised DP rank for each usable stream. In
    /// metadata-only mode KV subscriptions remain disabled, but the separate
    /// #34608 load stream is still attached when its full descriptor exists.
    /// If the worker is not publishing KV events (older SGLang, opt-out
    /// config), this is a logged no-op — the worker still routes via the
    /// non-cache-aware policies.
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
        // Reconcile this worker's `page_size` with the oracle BEFORE
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
                "kv-events: worker page_size disagrees with established block_size; \
                 skipping worker — cache-aware routing requires every worker to publish \
                 at the same block size",
            );
            return;
        }
        // Establish the bigram flag alongside block_size. EAGLE-family workers
        // hash KV blocks over token bigrams, so the policy must use the bigram
        // hasher for its query hashes to match the worker's stored hashes.
        self.block_size_oracle.set_bigram(cfg.is_bigram);
        let kv_dp_ranks = if self.maintain_tree {
            subscribable_ranks(cfg.port_base, cfg.dp_size)
        } else {
            Vec::new()
        };
        let load_descriptor_complete = cfg.load_port_base.is_some() && cfg.load_topic.is_some();
        if cfg.load_port_base.is_some() != cfg.load_topic.is_some() {
            warn!(
                worker_url = %worker_url,
                load_port_base = ?cfg.load_port_base,
                load_topic = ?cfg.load_topic,
                "kv-events: incomplete load descriptor; refusing load subscription"
            );
        }
        let load_dp_ranks = cfg
            .load_port_base
            .filter(|_| load_descriptor_complete)
            .map(|port_base| subscribable_ranks(port_base, cfg.dp_size))
            .unwrap_or_default();
        let mut dp_ranks = kv_dp_ranks.clone();
        dp_ranks.extend(load_dp_ranks.iter().copied());
        dp_ranks.sort_unstable();
        dp_ranks.dedup();
        if dp_ranks.is_empty() {
            warn!(
                worker_url = %worker_url,
                port_base = cfg.port_base,
                dp_size = cfg.dp_size,
                "kv-events: no usable KV or load publisher ranks; skipping worker",
            );
            return;
        }
        if self.maintain_tree {
            info!(
                worker_url = %worker_url,
                dp_size = cfg.dp_size,
                port_base = cfg.port_base,
                load_port_base = ?cfg.load_port_base,
                block_size = cfg.block_size,
                is_bigram = cfg.is_bigram,
                "kv-events: subscribing",
            );
        } else {
            info!(
                worker_url = %worker_url,
                dp_size = cfg.dp_size,
                load_port_base = ?cfg.load_port_base,
                block_size = cfg.block_size,
                is_bigram = cfg.is_bigram,
                "kv-events: external Indexer configured; subscribing only to engine load",
            );
        }
        // Mark every rank live BEFORE the subscriber starts so any event
        // it queues is accepted by the pump.
        {
            let mut live = self.live_workers.lock();
            for &rank in &dp_ranks {
                live.insert(KvWorkerId {
                    url: worker_url.to_string(),
                    dp_rank: rank,
                });
            }
        }
        // Register for bootstrap BEFORE the subscriber starts too, so the very
        // first batch is held back rather than applied ahead of the snapshot.
        // Ordering matters in the other direction as well: the subscription
        // must be live before the snapshot is fetched, so no delta can fall
        // into the gap between the peer's export and our first received batch.
        //
        // KV ranks only. A load-only rank publishes no KV stream, so there is
        // nothing to hold and no watermark to splice against; registering it
        // would leave an obligation no snapshot can ever discharge.
        let bootstrap_ranks: Vec<KvWorkerId> = kv_dp_ranks
            .iter()
            .map(|&rank| KvWorkerId::new(worker_url.to_string(), rank))
            .collect();
        let bootstrap_obligations = self.register_for_bootstrap(&bootstrap_ranks);
        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerEntry {
                dp_ranks: dp_ranks.clone(),
            },
        );
        if self.maintain_tree && !kv_dp_ranks.is_empty() {
            self.subscribers.add_worker(worker_url, &cfg).await;
        }
        if !bootstrap_obligations.is_empty() {
            // Stamped HERE, after `subscribers.add_worker` — not at `register`.
            // A peer's export has to beat the subscription, and anything earlier
            // would let the sweep accept a snapshot taken during the subscribe
            // window, which is precisely the hole the watermark check would then
            // reject as `Gap`. `subscribers.add_worker` only spawns the SUB
            // tasks, though: the connect completes asynchronously, so an export
            // taken between this stamp and the connect can still gap, and the
            // splice check is what catches it.
            let batch = ObligationBatch {
                obligations: bootstrap_obligations,
                holding_since: Instant::now(),
                late_join: LateJoin::Permitted,
            };
            // The stamp travels with the batch, so whichever sweep picks it up
            // asks for an export newer than it.
            self.enqueue_bootstrap(batch);
        }
        // Mark only the ranks that have an actual SUB socket. `EngineReportedLoadTable`
        // then rejects missing or stale advertised ranks as a whole worker.
        if !load_dp_ranks.is_empty() {
            for rank in &load_dp_ranks {
                self.engine_reported_load
                    .mark_expected_rank(worker_url, *rank);
            }
            self.load_subscribers.add_worker(worker_url, &cfg).await;
        }
    }

    /// Whether this worker's ranks enter the bootstrap state machine. Keyed on
    /// configured, not settled, so a late worker still gets an obligation; and
    /// independent of visible peers, because registering arms the deadline.
    fn peer_bootstrap_enabled(&self) -> bool {
        self.bootstrap.enabled()
    }

    /// Register the ranks a sweep should run for, returning their obligations.
    ///
    /// Only ranks the tracker does not already hold yield one:
    /// `reconcile_unresolved_workers` re-calls `add_worker` for a worker it
    /// already knows, and a rank that is `Pending` already has a sweep while a
    /// terminal one has nothing left to fetch. A rank `remove_worker` forgot
    /// registers as a new incarnation.
    fn register_for_bootstrap(&self, ranks: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        if !self.peer_bootstrap_enabled() {
            return Vec::new();
        }
        self.bootstrap.register(ranks)
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
        // 3. Drop the worker's engine load and each rank's bootstrap state, so
        //    a rank that never finished cannot hold the readiness gate open.
        //    Any event already in the mpsc buffer at this point will be
        //    filtered by the live-set check inside the pump.
        self.engine_reported_load.forget_worker(worker_url);
        self.bootstrap.forget(&ids);
        // 4. Hand the per-rank teardown to the pump: `held`,
        //    `awaiting_splice_proof`, the tree carriers, and the cursor. Doing
        //    the tree/cursor half here would race a graft already past its
        //    gates (see the `ForgetRanks` arm).
        //
        //    `bootstrap.forget` above must precede this send: it is what makes
        //    an in-flight ApplySnapshot for these ranks a no-op.
        let (done_tx, done_rx) = oneshot::channel();
        let queued = self
            .ctrl_tx
            .send(PumpControl::ForgetRanks {
                ranks: ids.clone(),
                done: Some(done_tx),
            })
            .await
            .is_ok();
        // Wait, so a worker that flaps straight back cannot have its fresh
        // state wiped by this forget.
        if !queued || done_rx.await.is_err() {
            // Only reachable once the pump has exited, i.e. during shutdown.
            // Clean up inline so the state does not outlive the worker in that
            // case; there is no writer left to race.
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

/// Shared state the pump task reads and writes.
struct PumpDeps {
    tally: Arc<EventTally>,
    tree: Arc<HashTree>,
    engine_reported_load: Arc<EngineReportedLoadTable>,
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    bootstrap: Arc<BootstrapTracker>,
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
    ///
    /// Weak so the pump does not keep its own control channel open: the channel
    /// still closes when every external sender is gone, which is what the
    /// `ctrl_open` latch in `pump_loop` exists to observe. A probe upgrades it
    /// for the duration of its pass.
    ctrl_tx: mpsc::WeakSender<PumpControl>,
}

/// Drain `WorkerEvent`s: apply KV `Batch`es to the tree and `Load` snapshots
/// to the engine-load table. Out-of-order (seq ≤ last_applied) and stale
/// (worker not in `live_workers`) KV batches are skipped; `Load` is a gauge
/// with no seq. `PublisherReset` events clear the cursor so a publisher
/// restarting from seq=0 (after sending END_SEQ) is not filtered. Without
/// that event, a regression is still read as a restart where it is
/// unambiguous: any backwards step in a Pending rank's received sequence, and
/// a batch 0 behind any cursor — both replace the rank's state with the new
/// stream rather than being skipped.
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
        tally,
        tree,
        engine_reported_load,
        cursors,
        live_workers,
        bootstrap,
        peers,
        snapshot_http,
        bootstrap_tx,
        ctrl_tx,
    } = deps;
    let pump_state = PumpState {
        tree: &tree,
        cursors: &cursors,
        tally: &tally,
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
                    Some(PumpControl::ForgetRanks { ranks, done }) => {
                        for rank in ranks {
                            held.remove(&rank);
                            awaiting_splice_proof.remove(&rank);
                            // On the single writer so it is strictly ordered with
                            // any graft: a graft queued before this is undone here;
                            // one queued after fails its gates.
                            tree.clear_worker(&rank);
                            cursors.lock().remove(&rank);
                        }
                        if let Some(done) = done {
                            let _ = done.send(());
                        }
                    }
                    Some(PumpControl::AbandonBootstrap { obligations }) => {
                        for (rank, epoch) in obligations {
                            if !still_owed(&pump_state, &rank, epoch) {
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
                    Some(PumpControl::SpliceProbe {
                        rank,
                        epoch,
                        watermark,
                        verdict,
                    }) => {
                        // The rank may have proven itself, been forgotten,
                        // been re-registered, or been re-grafted by a gap retry
                        // while the probe was in flight. The epoch rules out
                        // another incarnation; the watermark rules out another
                        // graft of this one, which a retry makes under the same
                        // epoch.
                        let proof = match awaiting_splice_proof.get_mut(&rank) {
                            Some(proof)
                                if proof.watermark == watermark
                                    && bootstrap.epoch_of(&rank) == Some(epoch) =>
                            {
                                proof
                            }
                            _ => {
                                debug!(
                                    worker = ?rank,
                                    epoch,
                                    watermark,
                                    "kv-bootstrap: dropping a probe verdict that no longer \
                                     addresses live state",
                                );
                                continue;
                            }
                        };
                        match verdict {
                            SpliceVerdict::Advanced => {
                                awaiting_splice_proof.remove(&rank);
                                demote_unproven_rank(&pump_state, &mut held, &rank);
                            }
                            SpliceVerdict::NoAdvance => {
                                debug!(
                                    worker = ?rank,
                                    watermark,
                                    "kv-bootstrap: no peer is past the watermark; treating \
                                     the silent stream as continuous",
                                );
                                awaiting_splice_proof.remove(&rank);
                                bootstrap.record_rank_outcome(RankOutcome::Warm);
                            }
                            // No witness. Keep the rank warm and ask again — but
                            // not forever: a fleet that never answers (a
                            // single-replica deployment, say) would otherwise
                            // leave the verdict unresolved and the probe looping.
                            SpliceVerdict::Unknown => {
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
                // Do NOT discard on silence alone. Reaching the deferred path
                // means nothing arrived between subscribing and grafting, and
                // the subscriber is live before the snapshot is fetched — so
                // silence is far more often "this rank published nothing" than
                // "we lost a delta". Discarding on a timer would throw away a
                // healthy warm tree on every quiet fleet, which is the exact
                // regression this feature exists to prevent. Ask the fleet
                // instead, and act only on positive evidence.
                let now = Instant::now();
                let mut targets = Vec::new();
                for (rank, proof) in awaiting_splice_proof
                    .iter_mut()
                    .filter(|(_, p)| p.due_for_probe(SPLICE_PROOF_TIMEOUT))
                {
                    // Re-armed before probing so retries are spaced by the
                    // timeout, never by the sweep tick.
                    proof.since = now;
                    // Forgotten: the `ForgetRanks` that follows drops the entry.
                    let Some(epoch) = bootstrap.epoch_of(rank) else {
                        continue;
                    };
                    targets.push(ProbeTarget {
                        rank: rank.clone(),
                        watermark: proof.watermark,
                        epoch,
                    });
                }
                // One pass for every due rank: a single cursor table per peer
                // answers all of them, so asking per rank would multiply the
                // fleet's work by the number of idle ranks.
                if !targets.is_empty() {
                    if let Some(ctrl_tx) = ctrl_tx.upgrade() {
                        spawn_splice_probe(
                            snapshot_http.clone(),
                            Arc::clone(&peers),
                            ctrl_tx,
                            targets,
                        );
                    }
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
                engine_reported_load.set(&worker.url, worker.dp_rank, load, Instant::now());
            }
            WorkerEvent::PublisherReset { worker } => {
                // A reset means the engine restarted with an empty cache and
                // renumbers from 0: any splice proof, grafted state or held
                // batch describes a cache that no longer exists, so drop all of
                // it and relearn from the new stream.
                if awaiting_splice_proof.remove(&worker).is_some() {
                    bootstrap.record_rank_outcome(RankOutcome::PublisherReset);
                }
                let state = bootstrap.state_of(&worker);
                tree.clear_worker(&worker);
                match state {
                    Some(BootstrapState::Recovered) => {
                        bootstrap.set(&worker, BootstrapState::Failed);
                    }
                    Some(BootstrapState::Pending) => {
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
                    _ => {}
                }
                if cursors.lock().remove(&worker).is_some() {
                    info!(
                        worker = ?worker,
                        "kv-events pump: publisher reset; cursor cleared",
                    );
                }
            }
            WorkerEvent::Batch {
                worker,
                seq,
                mut batch,
            } => {
                // A rank still awaiting its snapshot holds its batches: applying
                // them first would put live deltas *under* the snapshot, where a
                // stale `BlockStored` from the peer could resurrect a block this
                // rank has already evicted.
                if bootstrap.enabled()
                    && bootstrap.state_of(&worker) == Some(BootstrapState::Pending)
                {
                    let last_held = held.get(&worker).and_then(|q| q.back()).map(|(s, _)| *s);
                    // The last seq this rank RECEIVED: the held queue's tail, or
                    // with nothing held, its cursor. A Pending rank's cursor is
                    // a raw arrival seq, never a graft watermark: a graft moves
                    // the rank out of Pending, every path that discards one
                    // clears the cursor it seeded, and a fresh incarnation
                    // starts with none, because its `ForgetRanks` is queued on
                    // the control channel, which the pump drains first.
                    let last_received = last_held.or_else(|| cursors.lock().get(&worker).copied());
                    if last_held.is_none() && seq == STREAM_ORIGIN_SEQ {
                        // Nothing to wait on a peer for: this stream starts at
                        // its publisher's origin.
                        resolve_from_origin(&pump_state, &mut held, &worker);
                    } else if last_received.is_some_and(|last| seq < last) {
                        // Received order is raw arrival order — no watermark is
                        // seeded into it — and PUB/SUB neither reorders nor
                        // replays, so a regression in it is the publisher
                        // renumbering: the engine restarted in place without an
                        // `END_SEQ`. Anything queued is a dead stream. Holding
                        // on would graft a snapshot whose old-numbering watermark
                        // then filters the entire new stream, with no gap ever
                        // detected — dead state served warm, live updates lost.
                        warn!(
                            worker = ?worker,
                            last_received_seq = last_received,
                            seq,
                            "kv-bootstrap: sequence regressed while holding; the publisher \
                             restarted without END_SEQ, discarding the dead stream's batches",
                        );
                        if seq == STREAM_ORIGIN_SEQ {
                            resolve_from_origin(&pump_state, &mut held, &worker);
                        } else {
                            // The new stream's head is gone too, so nothing can
                            // be spliced: run cold from this batch on.
                            fail_rank(
                                &pump_state,
                                &mut held,
                                &worker,
                                true,
                                RankOutcome::PublisherReset,
                            );
                        }
                    } else if held.get(&worker).map_or(0, VecDeque::len) >= PENDING_BATCH_LIMIT {
                        // Dropping from the middle of the stream would leave a
                        // hole the snapshot cannot be spliced across, so give up
                        // on bootstrapping this rank and let it run live. The
                        // queue itself is intact (nothing was dropped to reach
                        // the cap), so it is replayed rather than discarded.
                        warn!(
                            worker = ?worker,
                            limit = PENDING_BATCH_LIMIT,
                            "kv-bootstrap: held-batch limit reached before a snapshot arrived; \
                             abandoning bootstrap for this rank",
                        );
                        fail_rank(
                            &pump_state,
                            &mut held,
                            &worker,
                            false,
                            RankOutcome::Overflow,
                        );
                    } else {
                        // The cap counts batches, not bytes, and `token_ids` is
                        // the unbounded part of a batch (one entry per prompt
                        // token) that `apply_batch` never reads. Shed it so a
                        // long-context fleet cannot hold gigabytes at boot.
                        for event in &mut batch.events {
                            if let KvCacheEvent::BlockStored(b) = event {
                                b.token_ids = Vec::new();
                            }
                        }
                        if let Some(queue) = held.get_mut(&worker) {
                            queue.push_back((seq, batch));
                        } else {
                            held.entry(worker.clone())
                                .or_default()
                                .push_back((seq, batch));
                        }
                        continue;
                    }
                    // Falls through: the rank is no longer Pending, so this
                    // batch is applied directly below.
                } else if seq == STREAM_ORIGIN_SEQ && cursors.lock().contains_key(&worker) {
                    // The resolved-rank counterpart of the regression above,
                    // narrowed to batch 0. Here the comparison is against the
                    // CURSOR, which a graft may have seeded ahead of anything
                    // received, so `seq <= cursor` in general means "already
                    // reflected" and stays filtered by `apply_batch`. Batch 0 is
                    // the exception: any cursor means batch 0 is already
                    // reflected, so another one is a restarted publisher —
                    // whose whole stream the old cursor would otherwise filter
                    // until it overtook it. (Were it a redelivery at cursor 0,
                    // clearing and re-applying batch 0 rebuilds the same state,
                    // since batch 0 is all such a cursor reflects. Were it a
                    // graft the pump took before this rank's first batch, the
                    // stream from its origin rebuilds what the graft held.) The gap
                    // check below cannot see this: it flags forward holes, and
                    // a regressed seq passes it as proof of continuity.
                    warn!(
                        worker = ?worker,
                        "kv-events pump: batch 0 behind a later cursor; replacing this rank's \
                         state with the stream from its origin (the publisher restarted without \
                         END_SEQ, or a graft landed before its first batch)",
                    );
                    if awaiting_splice_proof.remove(&worker).is_some() {
                        // The graft's verdict is final now: its state is gone,
                        // and what replaces it is the new stream from its origin.
                        bootstrap.record_rank_outcome(RankOutcome::FromOrigin);
                    }
                    tree.clear_worker(&worker);
                    cursors.lock().remove(&worker);
                }
                // First batch after a graft proves — or disproves — that the
                // snapshot joins up with this rank's live stream.
                let proof = if awaiting_splice_proof.is_empty() {
                    None
                } else {
                    awaiting_splice_proof.remove(&worker)
                };
                if let Some(proof) = proof {
                    if leaves_gap(seq, proof.watermark) {
                        warn!(
                            worker = ?worker,
                            peer_cursor = proof.watermark,
                            first_live_seq = seq,
                            "kv-bootstrap: sequence gap between snapshot and live stream; \
                             discarding snapshot state for this rank to avoid stale cache entries",
                        );
                        // Held, not applied: `resolve_gap` either replays it after
                        // clearing, or keeps it for the retry's graft.
                        held.entry(worker.clone())
                            .or_default()
                            .push_back((seq, batch));
                        resolve_gap(&pump_state, &mut held, &worker);
                        continue;
                    } else {
                        // The deferred check passed: this rank's grafted state is
                        // now proven continuous with its live stream, which is the
                        // point at which it counts as warm.
                        bootstrap.record_rank_outcome(RankOutcome::Warm);
                    }
                }
                apply_batch(&tree, &cursors, &tally, &worker, seq, &batch);
            }
        }
    }
}

/// The pump-owned handles the bootstrap helpers share.
struct PumpState<'a> {
    tree: &'a HashTree,
    cursors: &'a Mutex<HashMap<KvWorkerId, i64>>,
    tally: &'a EventTally,
    bootstrap: &'a BootstrapTracker,
    live_workers: &'a Mutex<HashSet<KvWorkerId>>,
    /// Obligation queue, so the graft path can hand a gapped rank back for
    /// another sweep. See [`resolve_gap`].
    bootstrap_tx: &'a mpsc::Sender<ObligationBatch>,
}

/// Apply one batch, honouring the cursor's out-of-order filter.
///
/// This is the single place tree deltas are written, whether the batch came
/// straight off the wire or out of a bootstrap hold-back queue — which is what
/// makes cursor seeding sufficient to reconcile a snapshot with the live stream.
fn apply_batch(
    tree: &HashTree,
    cursors: &Mutex<HashMap<KvWorkerId, i64>>,
    tally: &EventTally,
    worker: &KvWorkerId,
    seq: i64,
    batch: &KvEventBatch,
) {
    // Bound on its own line: an `if let` scrutinee would keep the guard alive
    // through the logging below.
    let prev = cursors.lock().get(worker).copied();
    if let Some(p) = prev {
        if seq <= p {
            debug!(
                worker = ?worker,
                seq,
                last_applied = p,
                "kv-events pump: out-of-order batch; skipping",
            );
            return;
        }
        // The publisher's seq is dense, so a jump is exactly the batches ZMQ
        // dropped at its high-water mark. A removal clears only its own tier,
        // so losing a block's last removal leaves the worker owning it until
        // the next AllBlocksCleared or teardown; only the sequence shows that
        // happened. The operator-visible signature is tree coverage above 1.
        let lost = (seq - p - 1) as u64;
        if lost > 0 {
            tally.record_lost_batches(lost);
            warn!(
                worker = ?worker,
                seq,
                last_applied = p,
                lost,
                "kv-events pump: sequence gap; batches were dropped in transit and the tree may hold stale tiers for this worker",
            );
        }
    }
    for event in &batch.events {
        match event {
            // The `medium` tag decides which tier a store lands on and which
            // tier a removal clears, so a device eviction leaves a worker that
            // still holds the block on host as an owner — see the tree's
            // "Storage tiers" docs.
            KvCacheEvent::BlockStored(b) => {
                tally.record(
                    EventKind::BlockStored,
                    b.medium.as_deref(),
                    b.block_hashes.len(),
                );
                tree.insert_tiered(
                    worker,
                    b.parent_block_hash,
                    &b.block_hashes,
                    Tiers::for_store(b.medium.as_deref()),
                );
            }
            KvCacheEvent::BlockRemoved(b) => {
                tally.record(
                    EventKind::BlockRemoved,
                    b.medium.as_deref(),
                    b.block_hashes.len(),
                );
                tree.remove_tiered(
                    worker,
                    &b.block_hashes,
                    Tiers::for_remove(b.medium.as_deref()),
                );
            }
            KvCacheEvent::AllBlocksCleared => {
                tally.record(EventKind::AllBlocksCleared, None, 0);
                tree.clear_worker(worker);
            }
        }
    }
    let mut guard = cursors.lock();
    if let Some(cursor) = guard.get_mut(worker) {
        *cursor = seq;
    } else {
        guard.insert(worker.clone(), seq);
    }
}
