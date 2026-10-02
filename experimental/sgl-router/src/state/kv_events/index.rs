// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle bundle for the KV-event index: the [`HashTree`], the engine-reported load table,
//! the KV and load [`KvEventSubscriberRegistry`]s, and the pump that applies their events.
//!
//! `remove_worker` drops ranks from `live_workers` before joining their subscribers,
//! and the pump filters every event through that set,
//! so an event still buffered for a torn-down worker cannot re-insert tree state.

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

/// Bounded so a misbehaving publisher cannot exhaust memory;
/// 1024 absorbs a half-second 2 kHz burst before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

/// Per-rank cap on batches held while [`BootstrapState::Pending`]; overflow abandons bootstrap.
/// Matches `EVENT_CHANNEL_BUFFER`: a pump that far behind will not get the snapshot in time.
const PENDING_BATCH_LIMIT: usize = 1024;

/// Sized well past a fleet's worker count;
/// overflow falls back as [`KvEventIndex::enqueue_bootstrap`] describes.
const BOOTSTRAP_QUEUE_DEPTH: usize = 1024;

/// Seq of a publisher's first batch: SGLang's `ZmqEventPublisher` counts from 0 and is built
/// once per scheduler alongside an empty radix cache, so a `Pending` rank whose FIRST held
/// batch is 0 has seen every block and needs no snapshot (see `resolve_from_origin`).
/// A first batch of 1 means ZMQ's slow-joiner window dropped batch 0, so the rank still waits;
/// a batch 0 arriving later is a publisher restart (see `pump_loop`).
const STREAM_ORIGIN_SEQ: i64 = 0;

/// Control-plane messages for the pump, the tree's single writer (see [`super::tree`]);
/// the bootstrap task only fetches and vets snapshots, never writes the tree.
#[derive(Debug)]
enum PumpControl {
    /// Graft a vetted snapshot, seed cursors, then release each rank's held batches.
    /// Every rank in `obligations` leaves `Pending` whether or not the snapshot covered it;
    /// each carries its registered incarnation, so a stale task cannot graft onto a re-add.
    ApplySnapshot {
        obligations: Vec<(KvWorkerId, u64)>,
        vetted: Box<VettedSnapshot>,
    },
    /// Release the ranks' held batches and mark them [`BootstrapState::Failed`];
    /// sent when no peer can supply a snapshot or the bootstrap deadline fires.
    AbandonBootstrap { obligations: Vec<(KvWorkerId, u64)> },
    /// Drop everything the pump keeps for removed ranks (held batches, splice proofs,
    /// tree carriers, cursors); `done` fires once that teardown has run.
    ForgetRanks {
        ranks: Vec<KvWorkerId>,
        done: Option<oneshot::Sender<()>>,
    },
    /// Off-pump probe verdict on whether a rank's publisher moved past the watermark
    /// of a graft whose splice was never proven locally;
    /// `epoch` and `watermark` name that graft (see [`PendingProof::watermark`]).
    SpliceProbe {
        rank: KvWorkerId,
        epoch: u64,
        watermark: i64,
        verdict: SpliceVerdict,
    },
}

/// The DP ranks actually subscribed, which `remove_worker` clears;
/// not the advertised `dp_size`, whose ports may overflow `u16`.
#[derive(Debug, Clone)]
struct WorkerEntry {
    dp_ranks: Vec<u32>,
}

/// Ranks whose port fits in `u16`; shared with the subscriber registry
/// so every expected load rank has a SUB socket.
fn subscribable_ranks(port_base: u16, dp_size: u32) -> Vec<u32> {
    let port_base = u32::from(port_base);
    (0..dp_size)
        .filter(|rank| port_base.saturating_add(*rank) <= u32::from(u16::MAX))
        .collect()
}

/// Read-only handles for the `/metrics` KV storage-tier series,
/// narrower than [`KvEventIndex`] so a route cannot call lifecycle methods.
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
    /// Folded into `coordinator::PendingSweep::freshness_floor`, which the sweep asks with.
    holding_since: Instant,
    /// Whether this batch may ride a sweep already in flight.
    late_join: LateJoin,
}

/// Whether a batch may ride a sweep already in flight, whose export may predate the
/// batch's `holding_since` and so resolve [`RankOutcome::Gap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LateJoin {
    /// Ride it regardless; used by discovery,
    /// whose workers land during the first fetch and still have their one gap retry.
    Permitted,
    /// Wait for a sweep that asks on this batch's behalf; used by a gap retry,
    /// which `gap_retried` caps at one and which an older snapshot re-gaps by construction.
    Refused,
}

/// One per router process, held by the worker manager as `Option<Arc<KvEventIndex>>`;
/// `None` disables cache-aware routing.
pub struct KvEventIndex {
    tree: Arc<HashTree>,
    maintain_tree: bool,
    subscribers: Arc<KvEventSubscriberRegistry>,
    /// Load-topic subscribers feeding `engine_reported_load`; shares the pump channel
    /// but is keyed separately so KV and load subscribers for a worker don't collide.
    load_subscribers: Arc<KvEventSubscriberRegistry>,
    /// Engine-reported per-worker load, written by the pump from
    /// `WorkerEvent::Load` and captured at request ingress.
    engine_reported_load: Arc<EngineReportedLoadTable>,
    pump: Mutex<Option<JoinHandle<()>>>,
    pump_cancel: CancellationToken,
    workers: Mutex<HashMap<String, WorkerEntry>>,
    http: reqwest::Client,
    /// Attached `(worker_url, dp_rank)` pairs; the pump drops events for any other,
    /// so a batch queued before `remove_worker` cannot re-pollute the tree.
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    /// Last-applied seq per rank, filtering the subscriber's un-deduplicated stream;
    /// cleared on `remove_worker` since a re-added worker's publisher restarts from 0.
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
    /// Not `http`, whose 2s total timeout cannot fit a multi-megabyte snapshot
    /// and would book every large one `unreachable`.
    snapshot_http: reqwest::Client,
    /// Obligations waiting for the coordinator to fold them into the sweep that
    /// is in flight, or to start one. See [`bootstrap_coordinator`].
    bootstrap_tx: mpsc::Sender<ObligationBatch>,
    /// Last built snapshot; see [`KvEventIndex::peer_snapshot_body`].
    snapshot_cache: Arc<AsyncMutex<Option<CachedSnapshot>>>,
    /// Worker-sourced `page_size` shared with prefix providers;
    /// the first worker sets it and `add_worker` rejects any worker that disagrees.
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

    /// Takes a pre-shared [`BlockSizeOracle`] so prefix providers see the one this index seeds.
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

    /// Metadata-only: seeds the [`BlockSizeOracle`] without KV subscriptions or a local tree,
    /// because an external Indexer is the routing signal.
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
        // Connect/read timeouts cut a dead peer; the total caps a slow transfer
        // so the sweep can still reach another candidate (see `snapshot_fetch_timeout`).
        let per_fetch = snapshot_fetch_timeout(bootstrap.timeout(), bootstrap.fetch_cap());
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
        let snapshot_http = reqwest::Client::builder()
            .connect_timeout(SNAPSHOT_FETCH_CONNECT_TIMEOUT)
            .read_timeout(SNAPSHOT_FETCH_READ_TIMEOUT)
            .timeout(per_fetch)
            // A sibling router never redirects this route; following a 3xx would let a bad peer
            // aim a multi-gigabyte fetch anywhere, so it lands as `FetchAnswer::NoBody`.
            .redirect(reqwest::redirect::Policy::none())
            // No fallback to the introspection client, which follows redirects;
            // `build()` fails only if TLS cannot initialize, which `new()` already treats as fatal.
            .build()
            .expect("snapshot http client builds: no fallible builder options are set");
        let tree = Arc::new(HashTree::new());
        let (tx, rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let (ctrl_tx, ctrl_rx) = mpsc::channel::<PumpControl>(16);
        let tally = Arc::new(EventTally::new());
        let subscribers =
            Arc::new(KvEventSubscriberRegistry::new(tx.clone()).with_tally(Arc::clone(&tally)));
        let load_subscribers = Arc::new(KvEventSubscriberRegistry::with_kind(tx, SubKind::Load));
        let engine_reported_load = EngineReportedLoadTable::new();
        let cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>> = Arc::new(Mutex::new(HashMap::new()));
        let live_workers: Arc<Mutex<HashSet<KvWorkerId>>> = Arc::new(Mutex::new(HashSet::new()));
        let pump_cancel = CancellationToken::new();
        let peers = Arc::new(PeerRegistry::new());
        let (bootstrap_tx, bootstrap_rx) = mpsc::channel(BOOTSTRAP_QUEUE_DEPTH);
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

    /// Shared handle to the bootstrap tracker, which also gates `/readyz`.
    pub fn bootstrap(&self) -> Arc<BootstrapTracker> {
        Arc::clone(&self.bootstrap)
    }

    /// Shared accessor for the per-process block-size oracle.
    pub fn block_size_oracle(&self) -> Arc<BlockSizeOracle> {
        Arc::clone(&self.block_size_oracle)
    }

    /// Read-only by contract: the pump is the tree's sole writer.
    pub fn tree(&self) -> Arc<HashTree> {
        self.tree.clone()
    }

    /// Handles for the `/metrics` storage-tier series, or `None` without a local tree,
    /// where every series would be a zero that its HELP text reads as a broken tier stream.
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

    /// Engine-load table; the pump writes values,
    /// while `add_worker` / `remove_worker` manage expected ranks and eviction.
    pub fn engine_reported_load(&self) -> Arc<EngineReportedLoadTable> {
        Arc::clone(&self.engine_reported_load)
    }

    /// Register a worker, opening one ZMQ SUB per DP rank on each advertised stream;
    /// `preresolved` skips the `/server_info` fetch. Metadata-only mode still attaches
    /// the load stream, and a worker that publishes nothing is a logged no-op.
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
        // Before any subscriber state exists: a worker with another `page_size` would hash
        // the same prompt to different blocks, silently breaking cache-aware routing.
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
        // EAGLE-family workers hash KV blocks over token bigrams,
        // so query hashes must use the bigram hasher to match.
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
        // Register for bootstrap before subscribing, so the first batch is held, not applied
        // ahead of the snapshot; the subscription must in turn be live before the fetch,
        // so no delta falls between the peer's export and our first batch.
        // KV ranks only: a load-only rank's obligation could never be discharged.
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
            // Stamped after subscribing, so the sweep rejects exports from the subscribe window.
            // The SUB connect completes asynchronously, so an export just after this stamp
            // can still gap; the splice check catches that.
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

    /// Keyed on configured rather than settled, so a late worker still gets an obligation,
    /// and independent of visible peers, because registering arms the deadline.
    fn peer_bootstrap_enabled(&self) -> bool {
        self.bootstrap.enabled()
    }

    /// Obligations only for ranks the tracker does not hold, since `add_worker` can re-run
    /// for a known worker: a `Pending` rank already has a sweep, a terminal one has nothing
    /// to fetch. A rank `remove_worker` forgot registers as a new incarnation.
    fn register_for_bootstrap(&self, ranks: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        if !self.peer_bootstrap_enabled() {
            return Vec::new();
        }
        self.bootstrap.register(ranks)
    }

    /// Tear down a worker's subscribers and tree state; a no-op for an unknown worker.
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
        // Before the subscriber join, so events still buffered are filtered by the pump.
        {
            let mut live = self.live_workers.lock();
            for id in &ids {
                live.remove(id);
            }
        }
        self.subscribers.remove_worker(worker_url).await;
        self.load_subscribers.remove_worker(worker_url).await;
        // So a rank that never finished bootstrap cannot hold the readiness gate open.
        self.engine_reported_load.forget_worker(worker_url);
        self.bootstrap.forget(&ids);
        // The pump clears the tree and cursor: doing it here would race a graft past its gates.
        // `bootstrap.forget` must precede this send; it makes an in-flight ApplySnapshot a no-op.
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
            // Only reachable once the pump has exited during shutdown,
            // so there is no writer left to race this inline cleanup.
            debug!("kv-events: pump is gone; tearing down worker state inline");
            let mut cursors = self.cursors.lock();
            for id in &ids {
                self.tree.clear_worker(id);
                cursors.remove(id);
            }
        }
    }

    /// Subscribed worker URLs, excluding any whose discovery failed or found no publisher.
    pub fn known_worker_count(&self) -> usize {
        self.workers.lock().len()
    }

    /// Stop the subscribers first so nothing more is queued,
    /// then cancel the pump, discarding buffered events.
    pub async fn shutdown(&self) {
        self.subscribers.shutdown().await;
        self.load_subscribers.shutdown().await;
        self.pump_cancel.cancel();
        let handle = self.pump.lock().take();
        if let Some(h) = handle {
            // Guards a pathological runtime teardown; the pump normally exits within one poll.
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
    /// For the splice probe only; the pump never fetches bootstrap snapshots itself.
    peers: Arc<PeerRegistry>,
    snapshot_http: reqwest::Client,
    /// Obligation queue, so a gap-discarded rank can be handed back for another
    /// sweep instead of staying cold with budget unspent.
    bootstrap_tx: mpsc::Sender<ObligationBatch>,
    /// Loopback for probe verdicts; weak so the pump does not keep its own channel open,
    /// which must still close (for `ctrl_open`) once every external sender is gone.
    ctrl_tx: mpsc::WeakSender<PumpControl>,
}

/// Sole writer of tree state, snapshot grafts included (see [`super::tree`]).
/// A restart without `PublisherReset` is detected where unambiguous:
/// any regression while `Pending`, or a batch 0 behind any cursor.
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
    // Grafted ranks whose continuity with the live stream is unproven. A graft can precede
    // any batch, so the check runs on the first held or live batch,
    // or in the sweep below if none arrives.
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
                        // The epoch rules out another incarnation; the watermark,
                        // a gap retry's re-graft under the same epoch.
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
                            // No witness: stay warm and re-ask, up to a cap,
                            // since a single-replica fleet can never answer.
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
            // Timer-driven, not event-driven: a rank that goes quiet right after a graft
            // would otherwise never be checked.
            _ = proof_sweep.tick(), if !awaiting_splice_proof.is_empty() => {
                // Never discard on silence alone: the subscriber was live before the fetch,
                // so a quiet rank most likely published nothing. Probe the fleet instead,
                // and act only on positive evidence.
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
                // One probe for all due ranks: a peer's single cursor table answers every one.
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

        // Load-bearing: `remove_worker` clears the live set before joining subscribers,
        // so a still-buffered event from a detached worker must not reach the tree.
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
                // Gauge: last value wins, no sequence or dedup.
                engine_reported_load.set(&worker.url, worker.dp_rank, load, Instant::now());
            }
            WorkerEvent::PublisherReset { worker } => {
                // The engine restarted with an empty cache and renumbers from 0,
                // so every proof, graft and held batch for this rank is dead.
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
                // A `Pending` rank holds its batches: applied under the snapshot, a stale peer
                // `BlockStored` could resurrect a block this rank already evicted.
                if bootstrap.enabled()
                    && bootstrap.state_of(&worker) == Some(BootstrapState::Pending)
                {
                    let last_held = held.get(&worker).and_then(|q| q.back()).map(|(s, _)| *s);
                    // Last seq RECEIVED. A Pending rank's cursor is never a graft watermark:
                    // a graft leaves Pending, discarding one clears its cursor,
                    // and a fresh incarnation's `ForgetRanks` is drained first.
                    let last_received = last_held.or_else(|| cursors.lock().get(&worker).copied());
                    if last_held.is_none() && seq == STREAM_ORIGIN_SEQ {
                        // Nothing to wait on a peer for: this stream starts at
                        // its publisher's origin.
                        resolve_from_origin(&pump_state, &mut held, &worker);
                    } else if last_received.is_some_and(|last| seq < last) {
                        // PUB/SUB neither reorders nor replays, so a regression means the
                        // publisher restarted without `END_SEQ`; a graft with the old watermark
                        // would filter the whole new stream and serve dead state as warm.
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
                        // Dropping mid-stream would leave an unspliceable hole, so run live;
                        // the intact queue is replayed rather than discarded.
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
                        // The cap counts batches, not bytes; `token_ids` grows per prompt token
                        // and `apply_batch` never reads it, so shed it.
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
                    // A graft may seed the cursor ahead of anything received, so `seq <= cursor`
                    // usually means already reflected; but any cursor reflects batch 0,
                    // so another batch 0 is a restarted publisher (replaying from origin is
                    // also safe for a redelivery). The gap check flags only forward holes.
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
                // The first batch after a graft proves or disproves its splice.
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
                        // Proven continuous with its live stream, so the rank now counts as warm.
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

/// The only writer of tree deltas, for live and held batches alike,
/// which is why seeding the cursor suffices to reconcile a snapshot with the live stream.
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
        // Seq is dense, so a jump is exactly the batches ZMQ dropped at its high-water mark.
        // A lost removal leaves the worker owning that block until AllBlocksCleared or teardown,
        // visible to operators only as tree coverage above 1.
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
            // `medium` picks the tier a store fills and a removal clears, so a device eviction
            // keeps a worker holding the block on host as owner (see "Storage tiers" in the tree).
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
