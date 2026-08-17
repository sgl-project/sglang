// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle bundle for the KV-event index.
//!
//! Couples the three submodules that are independent in their own right but
//! always operate together in production:
//!
//! - [`HashTree`] — the cache-aware routing index keyed by SGLang block hash.
//! - [`KvEventSubscriberRegistry`] — one ZMQ SUB connection per `(worker_url,
//!   dp_rank)`.
//! - A pump task that drains [`WorkerEvent`]s from the subscriber and applies
//!   them to the tree.
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

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::block_size_oracle::BlockSizeOracle;
use super::discovery::{fetch_event_config, EventConfig};
use super::replay::{fetch_replay_batches, ReplayError};
use super::subscriber::{KvEventSubscriberRegistry, WorkerEvent};
use super::tree::{HashTree, KvWorkerId};
use super::wire::{KvCacheEvent, KvEventBatch};
use crate::server::metrics::{KvEventGapOutcome, MetricsRegistry};

/// Channel buffer between the subscriber registry and the pump task.
///
/// Bounded so a misbehaving publisher cannot exhaust memory.  Realistic
/// per-worker event rates are < 1 kHz; a 1024-deep buffer absorbs a
/// half-second burst at 2 kHz before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

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

/// Per-rank replay socket target, derived from `EventConfig.replay`.
#[derive(Debug, Clone)]
struct ReplayTarget {
    endpoint: String,
    buffer_steps: usize,
}

/// Bundle of `HashTree` + `KvEventSubscriberRegistry` + pump task.
///
/// Construct one instance per router process and hand it to the worker
/// manager as `Option<Arc<KvEventIndex>>` — `None` disables the cache-aware
/// routing path entirely.
pub struct KvEventIndex {
    tree: Arc<HashTree>,
    subscribers: Arc<KvEventSubscriberRegistry>,
    pump: Mutex<Option<JoinHandle<()>>>,
    pump_cancel: CancellationToken,
    workers: Mutex<HashMap<String, WorkerEntry>>,
    http: reqwest::Client,
    /// Set of currently-attached `(worker_url, dp_rank)` pairs. The pump
    /// drops any event whose `worker` is not in this set, so a batch
    /// queued by a subscriber that was torn down by `remove_worker` does
    /// not re-pollute the tree after `clear_worker` ran.
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    /// Optional per-rank replay endpoint. Missing means the worker can still be
    /// subscribed live, but a sequence gap can only be logged/metriced before
    /// falling back to legacy current-batch application.
    replay_targets: Arc<Mutex<HashMap<KvWorkerId, ReplayTarget>>>,
    /// Per-`(worker_url, dp_rank)` last-applied sequence number. The
    /// subscriber forwards every batch with no de-dup; this map filters
    /// any batch whose `seq` is not strictly greater than the previously
    /// applied one. Cleared on `remove_worker` because a re-added worker
    /// may legitimately have a fresh publisher whose sequence numbers
    /// restart from 1.
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    /// Worker-sourced `page_size` shared with the cache-aware-zmq policy.
    /// `add_worker` calls `try_set(cfg.block_size)` so the first worker
    /// establishes the value; subsequent workers that disagree are
    /// rejected (logged + not subscribed). The policy reads it at routing
    /// time to size its `compute_block_hashes` call.
    block_size_oracle: Arc<BlockSizeOracle>,
    metrics: Arc<OnceLock<Arc<MetricsRegistry>>>,
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
    /// the same oracle the index seeds is the one the cache-aware-zmq
    /// policy reads at routing time. Tests use this to pre-populate the
    /// oracle and exercise the mismatch-rejection path.
    pub fn new_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        let tree = Arc::new(HashTree::new());
        let (tx, rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(tx));
        let cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>> = Arc::new(Mutex::new(HashMap::new()));
        let live_workers: Arc<Mutex<HashSet<KvWorkerId>>> = Arc::new(Mutex::new(HashSet::new()));
        let replay_targets: Arc<Mutex<HashMap<KvWorkerId, ReplayTarget>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let metrics = Arc::new(OnceLock::new());
        let pump_cancel = CancellationToken::new();
        let pump = tokio::spawn(pump_loop(
            tree.clone(),
            cursors.clone(),
            live_workers.clone(),
            replay_targets.clone(),
            metrics.clone(),
            pump_cancel.clone(),
            rx,
        ));
        Arc::new(Self {
            tree,
            subscribers,
            pump: Mutex::new(Some(pump)),
            pump_cancel,
            workers: Mutex::new(HashMap::new()),
            http,
            live_workers,
            replay_targets,
            cursors,
            block_size_oracle,
            metrics,
        })
    }

    /// Attach the process metrics registry to the background pump.
    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
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
        // Mark every rank live BEFORE the subscriber starts so any event
        // it queues is accepted by the pump.
        {
            let mut live = self.live_workers.lock();
            let mut replay_targets = self.replay_targets.lock();
            for &rank in &dp_ranks {
                let id = KvWorkerId {
                    url: worker_url.to_string(),
                    dp_rank: rank,
                };
                live.insert(id.clone());
                match replay_target_for_rank(&cfg, rank) {
                    Some(target) => {
                        replay_targets.insert(id, target);
                    }
                    None => {
                        if cfg.replay.is_some() {
                            warn!(
                                worker_url = %worker_url,
                                dp_rank = rank,
                                "kv-events: replay endpoint port overflows u16 for rank; gap recovery disabled for this rank",
                            );
                        }
                        replay_targets.remove(&id);
                    }
                }
            }
        }
        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerEntry {
                dp_ranks: dp_ranks.clone(),
            },
        );
        self.subscribers.add_worker(worker_url, &cfg).await;
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
        // 2. Cancel and join the per-rank subscriber tasks. No further
        //    events for these ranks will be queued after this returns.
        self.subscribers.remove_worker(worker_url).await;
        // 3. Drop each rank's tree state and cursor. Any event already in
        //    the mpsc buffer at this point will be filtered by the
        //    live-set check inside the pump.
        let mut cursors = self.cursors.lock();
        let mut replay_targets = self.replay_targets.lock();
        for id in &ids {
            self.tree.clear_worker(id);
            cursors.remove(id);
            replay_targets.remove(id);
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

fn replay_target_for_rank(cfg: &EventConfig, rank: u32) -> Option<ReplayTarget> {
    let replay = cfg.replay.as_ref()?;
    let port = u32::from(replay.port_base) + rank;
    if port > u32::from(u16::MAX) {
        return None;
    }
    Some(ReplayTarget {
        endpoint: format!("tcp://{}:{port}", replay.host),
        buffer_steps: replay.buffer_steps,
    })
}

/// Drain `WorkerEvent`s and apply each batch to the tree. Out-of-order
/// (seq ≤ last_applied) and stale (worker not in `live_workers`) batches are
/// skipped. A sequence gap triggers replay when the worker advertised a replay
/// endpoint; if replay is unavailable or cannot prove a continuous prefix, the
/// pump records/logs the gap and falls back to the original behavior of applying
/// the current live batch.
async fn pump_loop(
    tree: Arc<HashTree>,
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    replay_targets: Arc<Mutex<HashMap<KvWorkerId, ReplayTarget>>>,
    metrics: Arc<OnceLock<Arc<MetricsRegistry>>>,
    cancel: CancellationToken,
    mut rx: mpsc::Receiver<WorkerEvent>,
) {
    loop {
        let ev = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                info!("kv-events pump: shutdown requested; exiting");
                return;
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
            WorkerEvent::PublisherReset { worker } => {
                let had_cursor = cursors.lock().remove(&worker).is_some();
                if had_cursor {
                    info!(
                        worker = ?worker,
                        "kv-events pump: publisher reset; cursor cleared",
                    );
                }
            }
            WorkerEvent::Batch { worker, seq, batch } => {
                let prev = cursors.lock().get(&worker).copied();
                if let Some(p) = prev {
                    if seq <= p {
                        debug!(
                            worker = ?worker,
                            seq,
                            last_applied = p,
                            "kv-events pump: out-of-order batch; skipping",
                        );
                        continue;
                    }
                }

                let prev = cursors.lock().get(&worker).copied();
                if let Some(p) = prev {
                    if seq > p.saturating_add(1) {
                        let expected = p.saturating_add(1);
                        let target = replay_targets.lock().get(&worker).cloned();
                        let Some(target) = target else {
                            record_gap_metric(&metrics, KvEventGapOutcome::NoReplayEndpoint);
                            warn!(
                                worker_url = %worker.url,
                                dp_rank = worker.dp_rank,
                                gap_start_seq = expected,
                                live_seq = seq,
                                "kv-events pump: sequence gap observed but worker has no replay endpoint; applying current live batch with legacy behavior",
                            );
                            apply_batch(&tree, &worker, &batch);
                            cursors.lock().insert(worker, seq);
                            continue;
                        };
                        match fetch_replay_batches(&target.endpoint, expected, target.buffer_steps)
                            .await
                        {
                            Ok(replayed) => {
                                match apply_replayed_batches(
                                    &tree, &cursors, &worker, expected, seq, replayed,
                                ) {
                                    Ok(last_replayed) => {
                                        record_gap_metric(&metrics, KvEventGapOutcome::Replayed);
                                        info!(
                                            worker_url = %worker.url,
                                            dp_rank = worker.dp_rank,
                                            gap_start_seq = expected,
                                            live_seq = seq,
                                            last_replayed_seq = last_replayed,
                                            "kv-events pump: gap replayed successfully",
                                        );
                                    }
                                    Err(err) => {
                                        record_gap_metric(&metrics, KvEventGapOutcome::BufferMiss);
                                        warn!(
                                            worker_url = %worker.url,
                                            dp_rank = worker.dp_rank,
                                            gap_start_seq = expected,
                                            live_seq = seq,
                                            error = %err,
                                            "kv-events pump: replay did not cover gap; applying current live batch with legacy behavior",
                                        );
                                        apply_batch(&tree, &worker, &batch);
                                        cursors.lock().insert(worker, seq);
                                        continue;
                                    }
                                }
                            }
                            Err(err) => {
                                let outcome = match err {
                                    ReplayError::BatchLimitExceeded { .. } => {
                                        KvEventGapOutcome::BufferMiss
                                    }
                                    _ => KvEventGapOutcome::ReplayFailed,
                                };
                                record_gap_metric(&metrics, outcome);
                                warn!(
                                    worker_url = %worker.url,
                                    dp_rank = worker.dp_rank,
                                    gap_start_seq = expected,
                                    live_seq = seq,
                                    error = %err,
                                    "kv-events pump: replay request failed; applying current live batch with legacy behavior",
                                );
                                apply_batch(&tree, &worker, &batch);
                                cursors.lock().insert(worker, seq);
                                continue;
                            }
                        }
                    }
                }

                let prev = cursors.lock().get(&worker).copied();
                if let Some(p) = prev {
                    if seq <= p {
                        debug!(
                            worker = ?worker,
                            seq,
                            last_applied = p,
                            "kv-events pump: live batch already covered by replay; skipping",
                        );
                        continue;
                    }
                    if seq > p.saturating_add(1) {
                        record_gap_metric(&metrics, KvEventGapOutcome::BufferMiss);
                        warn!(
                            worker_url = %worker.url,
                            dp_rank = worker.dp_rank,
                            gap_start_seq = p.saturating_add(1),
                            live_seq = seq,
                            "kv-events pump: sequence gap remains after replay attempt; applying current live batch with legacy behavior",
                        );
                    }
                }

                apply_batch(&tree, &worker, &batch);
                cursors.lock().insert(worker, seq);
            }
        }
    }
}

fn apply_batch(tree: &HashTree, worker: &KvWorkerId, batch: &KvEventBatch) {
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
}

#[derive(Debug, thiserror::Error)]
enum ReplayContinuityError {
    #[error("replay returned no batches")]
    Empty,
    #[error("replay sequence discontinuity: expected {expected}, got {got}")]
    NonContiguous { expected: i64, got: i64 },
    #[error("replay ended at {last:?}, needed at least {needed}")]
    Short { last: Option<i64>, needed: i64 },
}

fn apply_replayed_batches(
    tree: &HashTree,
    cursors: &Mutex<HashMap<KvWorkerId, i64>>,
    worker: &KvWorkerId,
    start_seq: i64,
    live_seq: i64,
    replayed: Vec<(i64, KvEventBatch)>,
) -> Result<i64, ReplayContinuityError> {
    if replayed.is_empty() {
        return Err(ReplayContinuityError::Empty);
    }
    let mut expected = start_seq;
    let mut last_applied = None;
    for (seq, batch) in replayed {
        if seq != expected {
            return Err(ReplayContinuityError::NonContiguous { expected, got: seq });
        }
        apply_batch(tree, worker, &batch);
        cursors.lock().insert(worker.clone(), seq);
        last_applied = Some(seq);
        if seq >= live_seq {
            return Ok(seq);
        }
        expected = seq.saturating_add(1);
    }
    let needed = live_seq.saturating_sub(1);
    match last_applied {
        Some(last) if last >= needed => Ok(last),
        other => Err(ReplayContinuityError::Short {
            last: other,
            needed,
        }),
    }
}

fn record_gap_metric(metrics: &OnceLock<Arc<MetricsRegistry>>, outcome: KvEventGapOutcome) {
    if let Some(registry) = metrics.get() {
        registry.record_kv_event_gap(outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::kv_events::wire::{BlockRemoved, BlockStored, KvEventBatch};
    use bytes::Bytes;
    use rmp::encode as mp;
    use zeromq::{Endpoint, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

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

    fn block_stored_batch(block_hashes: &[i64]) -> KvEventBatch {
        batch(vec![KvCacheEvent::BlockStored(BlockStored {
            parent_block_hash: None,
            block_hashes: block_hashes.to_vec(),
            token_ids: vec![],
            block_size: 64,
            lora_id: None,
            medium: None,
        })])
    }

    fn encode_block_stored_batch(block_hashes: &[i64]) -> Vec<u8> {
        let mut buf = Vec::new();
        mp::write_array_len(&mut buf, 3).unwrap();
        mp::write_f64(&mut buf, 0.0).unwrap();
        mp::write_array_len(&mut buf, 1).unwrap();
        mp::write_array_len(&mut buf, 7).unwrap();
        mp::write_str(&mut buf, "BlockStored").unwrap();
        mp::write_array_len(&mut buf, block_hashes.len() as u32).unwrap();
        for hash in block_hashes {
            mp::write_sint(&mut buf, *hash).unwrap();
        }
        mp::write_nil(&mut buf).unwrap();
        mp::write_array_len(&mut buf, 0).unwrap();
        mp::write_uint(&mut buf, 64).unwrap();
        mp::write_nil(&mut buf).unwrap();
        mp::write_nil(&mut buf).unwrap();
        mp::write_nil(&mut buf).unwrap();
        buf
    }

    async fn spawn_replay_router(
        expected_start_seq: i64,
        replies: Vec<(i64, Vec<u8>)>,
    ) -> (String, JoinHandle<()>) {
        let mut router = RouterSocket::new();
        let endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind replay ROUTER");
        let port = match endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };
        let handle = tokio::spawn(async move {
            let request = router.recv().await.expect("receive replay request");
            assert_eq!(request.len(), 3);
            let identity = request.get(0).expect("identity").clone();
            assert!(request.get(1).expect("delimiter").is_empty());
            assert_eq!(
                request.get(2).expect("start seq").as_ref(),
                &expected_start_seq.to_be_bytes()
            );
            for (seq, payload) in replies {
                router
                    .send(replay_reply(identity.clone(), seq, payload))
                    .await
                    .expect("send replay reply");
            }
            router
                .send(replay_reply(identity, -1, Vec::new()))
                .await
                .expect("send replay END");
        });
        (format!("tcp://127.0.0.1:{port}"), handle)
    }

    fn replay_reply(identity: Bytes, seq: i64, payload: Vec<u8>) -> ZmqMessage {
        let mut msg = ZmqMessage::from(identity);
        msg.push_back(Bytes::new());
        msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
        msg.push_back(Bytes::from(payload));
        msg
    }

    /// Bundle of plumbing returned by `spawn_pump` so individual tests
    /// can destructure just the bits they need.
    struct PumpHarness {
        tree: Arc<HashTree>,
        cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
        #[allow(dead_code)]
        live_set: Arc<Mutex<HashSet<KvWorkerId>>>,
        replay_targets: Arc<Mutex<HashMap<KvWorkerId, ReplayTarget>>>,
        metrics: Arc<MetricsRegistry>,
        #[allow(dead_code)]
        cancel: CancellationToken,
        tx: mpsc::Sender<WorkerEvent>,
        pump: JoinHandle<()>,
    }

    /// Build a tree + cursors + live-set wired through `pump_loop` with
    /// the given workers pre-marked live.
    fn spawn_pump(live: &[KvWorkerId]) -> PumpHarness {
        let tree = Arc::new(HashTree::new());
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let live_set: Arc<Mutex<HashSet<KvWorkerId>>> =
            Arc::new(Mutex::new(live.iter().cloned().collect()));
        let replay_targets: Arc<Mutex<HashMap<KvWorkerId, ReplayTarget>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let metrics_lock = Arc::new(OnceLock::new());
        let metrics = MetricsRegistry::new();
        metrics_lock.set(Arc::clone(&metrics)).unwrap();
        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::channel(4);
        let pump = tokio::spawn(pump_loop(
            tree.clone(),
            cursors.clone(),
            live_set.clone(),
            replay_targets.clone(),
            metrics_lock,
            cancel.clone(),
            rx,
        ));
        PumpHarness {
            tree,
            cursors,
            live_set,
            replay_targets,
            metrics,
            cancel,
            tx,
            pump,
        }
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

    /// Conservative gap fallback: without a replay endpoint, the new code must
    /// keep the original behavior of applying the current live batch.
    #[tokio::test]
    async fn pump_gap_without_replay_keeps_legacy_current_batch_application() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, cursors, metrics, tx, pump) = (h.tree, h.cursors, h.metrics, h.tx, h.pump);

        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: block_stored_batch(&[10]),
        })
        .await
        .unwrap();
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: block_stored_batch(&[30]),
        })
        .await
        .unwrap();
        drop(tx);
        pump.await.unwrap();

        assert_eq!(tree.match_prefix(None, &[10]).matched_blocks, 1);
        assert_eq!(
            tree.match_prefix(None, &[30]).matched_blocks,
            1,
            "gap fallback should still apply the current live batch like upstream/main",
        );
        assert_eq!(cursors.lock().get(&id).copied(), Some(3));
        let out = metrics.render();
        assert!(out.contains(r#"sgl_router_kv_event_gaps_total{outcome="no_replay_endpoint"} 1"#));
    }

    /// With replay metadata, a gap is filled from the ROUTER socket before the
    /// triggering live batch is considered.
    #[tokio::test]
    async fn pump_replays_gap_before_current_live_batch() {
        let id = worker_id("http://w1", 0);
        let (endpoint, replay_server) =
            spawn_replay_router(2, vec![(2, encode_block_stored_batch(&[20]))]).await;
        let h = spawn_pump(std::slice::from_ref(&id));
        h.replay_targets.lock().insert(
            id.clone(),
            ReplayTarget {
                endpoint,
                buffer_steps: 16,
            },
        );
        let (tree, cursors, metrics, tx, pump) = (h.tree, h.cursors, h.metrics, h.tx, h.pump);

        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: block_stored_batch(&[10]),
        })
        .await
        .unwrap();
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: block_stored_batch(&[30]),
        })
        .await
        .unwrap();
        drop(tx);
        pump.await.unwrap();
        replay_server.await.unwrap();

        assert_eq!(tree.match_prefix(None, &[10]).matched_blocks, 1);
        assert_eq!(
            tree.match_prefix(None, &[20]).matched_blocks,
            1,
            "replayed seq=2 batch should be applied",
        );
        assert_eq!(
            tree.match_prefix(None, &[30]).matched_blocks,
            1,
            "triggering live seq=3 batch should still be applied",
        );
        assert_eq!(cursors.lock().get(&id).copied(), Some(3));
        let out = metrics.render();
        assert!(out.contains(r#"sgl_router_kv_event_gaps_total{outcome="replayed"} 1"#));
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
            block_size: 128,
            dp_size: 1,
            replay: None,
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
            block_size: 64,
            dp_size: 0,
            replay: None,
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
        // dp_size=0 short-circuits the subscriber spawn but still runs the seed.
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30300,
            topic: String::new(),
            block_size: 64,
            dp_size: 0,
            replay: None,
            is_bigram: true,
        };
        index.add_worker("http://127.0.0.1:30300", Some(cfg)).await;
        assert!(
            index.block_size_oracle().is_bigram(),
            "add_worker must seed the bigram flag from EventConfig"
        );
        index.shutdown().await;
    }
}
