// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle facade for the KV-event placement index.
//!
//! [`KvEventIndex`] discovers a worker's event endpoints, owns one subscriber
//! per independently routable DP replica, and wires those streams into the
//! internal synchronization state machine. Snapshot protocol handling lives in
//! [`super::snapshot`]; bootstrap and catch-up rules live in [`super::sync`].
//!
//! `add_worker` / `remove_worker` are driven from the worker manager on every
//! `DiscoveryEvent::Added` / `DiscoveryEvent::Removed`.
//!
//! # Race avoidance
//!
//! Worker lifecycle operations are serialized because they contain async
//! subscriber joins. The event pump additionally checks a `replica_modes` map;
//! entries are removed before subscriber tasks are joined, so an already
//! queued event cannot reinsert placement state after detach.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::block_size_oracle::BlockSizeOracle;
use super::discovery::{fetch_event_config, EventConfig};
use super::subscriber::{KvEventSubscriberRegistry, WorkerEvent};
use super::sync::{
    pump_loop, snapshot_target_for_rank, spawn_snapshot_sync, ReplicaIndexMode, SnapshotSyncHandle,
};
use super::tree::{HashTree, KvWorkerId};

/// Channel buffer between the subscriber registry and the pump task.
///
/// Bounded so a misbehaving publisher cannot exhaust memory. Realistic
/// per-worker event rates are < 1 kHz; a 1024-deep buffer absorbs a
/// half-second burst at 2 kHz before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

/// Per-worker bookkeeping kept inside [`KvEventIndex`] so `remove_worker`
/// knows which DP ranks were actually subscribed (not the advertised
/// `dp_size`, which may overflow `u16` and skip ranks).
#[derive(Debug)]
struct WorkerEntry {
    /// DP ranks that were successfully spawned for this worker.
    dp_ranks: Vec<u32>,
    /// Snapshot fetch loops for ranks that advertised snapshot-v1.
    snapshot_sync: Vec<SnapshotSyncHandle>,
}

/// Bundle of placement tree, subscriber registry, and synchronization pump.
///
/// Construct one instance per router process and hand it to the worker manager
/// as `Option<Arc<KvEventIndex>>`; `None` disables cache-aware routing.
pub struct KvEventIndex {
    tree: Arc<HashTree>,
    maintain_tree: bool,
    subscribers: Arc<KvEventSubscriberRegistry>,
    event_tx: mpsc::Sender<WorkerEvent>,
    pump: Mutex<Option<JoinHandle<()>>>,
    pump_cancel: CancellationToken,
    workers: Mutex<HashMap<String, WorkerEntry>>,
    /// Serializes add/remove/shutdown across their async task boundaries.
    lifecycle: AsyncMutex<()>,
    http: reqwest::Client,
    /// Indexing mode for every attached `(worker_url, dp_rank)` pair. The pump
    /// drops an event when its replica is absent from this map.
    replica_modes: Arc<Mutex<HashMap<KvWorkerId, ReplicaIndexMode>>>,
    /// Per-replica coalescing trigger used on gaps, resets, and epoch changes.
    resync_triggers: Arc<Mutex<HashMap<KvWorkerId, mpsc::Sender<()>>>>,
    /// Last sequence applied for each `(worker_url, dp_rank)`. Cleared on
    /// detach because a rebuilt publisher may restart its counter from zero.
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    /// Worker-sourced page size shared with the cache-aware-zmq policy.
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl KvEventIndex {
    /// Build an empty index and spawn the event pump.
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

    /// Constructor that accepts the oracle shared with the routing policy.
    pub fn new_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::new_with_mode(http, block_size_oracle, true)
    }

    /// Discovers worker hash metadata only: seeds the shared [`BlockSizeOracle`]
    /// but neither subscribes to KV events nor maintains the local tree, because
    /// an external Indexer is the routing signal.
    pub fn new_metadata_only_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::new_with_mode(http, block_size_oracle, false)
    }

    fn new_with_mode(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        maintain_tree: bool,
    ) -> Arc<Self> {
        let tree = Arc::new(HashTree::new());
        let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(event_tx.clone()));
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let replica_modes = Arc::new(Mutex::new(HashMap::new()));
        let resync_triggers = Arc::new(Mutex::new(HashMap::new()));
        let pump_cancel = CancellationToken::new();
        let pump = tokio::spawn(pump_loop(
            tree.clone(),
            cursors.clone(),
            replica_modes.clone(),
            resync_triggers.clone(),
            pump_cancel.clone(),
            event_rx,
        ));
        Arc::new(Self {
            tree,
            maintain_tree,
            subscribers,
            event_tx,
            pump: Mutex::new(Some(pump)),
            pump_cancel,
            workers: Mutex::new(HashMap::new()),
            lifecycle: AsyncMutex::new(()),
            http,
            replica_modes,
            resync_triggers,
            cursors,
            block_size_oracle,
        })
    }

    /// Shared accessor for the per-process block-size oracle.
    pub fn block_size_oracle(&self) -> Arc<BlockSizeOracle> {
        Arc::clone(&self.block_size_oracle)
    }

    /// Clone the tree handle used by cache-aware selection and metrics.
    pub fn tree(&self) -> Arc<HashTree> {
        self.tree.clone()
    }

    /// Register a worker and start one live subscriber per valid DP rank.
    ///
    /// If `preresolved` is `Some`, the worker manager has already fetched
    /// `/server_info`; standalone callers can pass `None` and let this method
    /// perform the bounded introspection request. Workers without a usable KV
    /// event descriptor remain eligible for non-cache-aware routing.
    pub async fn add_worker(&self, worker_url: &str, preresolved: Option<EventConfig>) {
        let _lifecycle_guard = self.lifecycle.lock().await;
        if self.workers.lock().contains_key(worker_url) {
            debug!(
                worker_url = %worker_url,
                "kv-events: worker already registered; ignoring duplicate add",
            );
            return;
        }

        let cfg = match preresolved {
            Some(cfg) => cfg,
            None => match fetch_event_config(worker_url, &self.http).await {
                Ok(Some(cfg)) => cfg,
                Ok(None) => {
                    info!(
                        worker_url = %worker_url,
                        "kv-events: worker is not publishing; cache-aware routing disabled for this worker",
                    );
                    return;
                }
                Err(error) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %error,
                        "kv-events: /server_info introspection failed; skipping subscriber",
                    );
                    return;
                }
            },
        };

        if let Some(snapshot) = &cfg.snapshot {
            if snapshot.protocol_version != 1 {
                warn!(
                    worker_url = %worker_url,
                    protocol_version = snapshot.protocol_version,
                    "kv-events: unsupported placement snapshot protocol; using legacy live-only indexing",
                );
            }
        }

        // Hashing at different block sizes would silently destroy match
        // quality, so validate before creating any subscriber state.
        if let Err(error) = self.block_size_oracle.try_set(cfg.block_size) {
            warn!(
                worker_url = %worker_url,
                established_block_size = error.established,
                worker_block_size = error.candidate,
                "kv-events: worker page_size disagrees with established block_size; \
                 skipping worker — cache-aware routing requires every worker to publish \
                 at the same block size",
            );
            return;
        }
        self.block_size_oracle.set_bigram(cfg.is_bigram);
        if !self.maintain_tree {
            info!(
                worker_url = %worker_url,
                block_size = cfg.block_size,
                is_bigram = cfg.is_bigram,
                "kv-events: external Indexer configured; discovered hash metadata without subscribing"
            );
            return;
        }

        // Compute the DP ranks that will actually be subscribed (skip
        // ranks whose port overflows u16; the subscriber will warn on
        // each skipped rank).
        let port_base = u32::from(cfg.port_base);
        let dp_ranks: Vec<u32> = (0..cfg.dp_size)
            .filter(|rank| (port_base + rank) <= u32::from(u16::MAX))
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

        info!(
            worker_url = %worker_url,
            dp_size = cfg.dp_size,
            port_base = cfg.port_base,
            block_size = cfg.block_size,
            is_bigram = cfg.is_bigram,
            "kv-events: subscribing",
        );

        // Attach before spawning SUB tasks so the pump accepts a batch queued
        // immediately after connect.
        {
            let mut replica_modes = self.replica_modes.lock();
            for &dp_rank in &dp_ranks {
                let worker = KvWorkerId::new(worker_url.to_owned(), dp_rank);
                let mode = if snapshot_target_for_rank(&cfg, dp_rank).is_some() {
                    ReplicaIndexMode::SnapshotRecoverable
                } else {
                    ReplicaIndexMode::LegacyBestEffort
                };
                replica_modes.insert(worker, mode);
            }
        }

        self.subscribers.add_worker(worker_url, &cfg).await;

        let mut snapshot_sync = Vec::new();
        for &dp_rank in &dp_ranks {
            let Some(target) = snapshot_target_for_rank(&cfg, dp_rank) else {
                continue;
            };
            let worker = KvWorkerId::new(worker_url.to_owned(), dp_rank);
            let handle = spawn_snapshot_sync(worker.clone(), target, self.event_tx.clone());
            self.resync_triggers
                .lock()
                .insert(worker, handle.trigger.clone());
            let _ = handle.trigger.try_send(());
            snapshot_sync.push(handle);
        }

        self.workers.lock().insert(
            worker_url.to_owned(),
            WorkerEntry {
                dp_ranks,
                snapshot_sync,
            },
        );
    }

    /// Detach a worker, stop its tasks, and remove all of its placement state.
    pub async fn remove_worker(&self, worker_url: &str) {
        let _lifecycle_guard = self.lifecycle.lock().await;
        let Some(entry) = self.workers.lock().remove(worker_url) else {
            return;
        };
        let WorkerEntry {
            dp_ranks,
            snapshot_sync,
        } = entry;
        let workers: Vec<KvWorkerId> = dp_ranks
            .into_iter()
            .map(|dp_rank| KvWorkerId::new(worker_url.to_owned(), dp_rank))
            .collect();

        // Mark dead before joining SUB tasks so already-buffered events are
        // filtered by the pump.
        {
            let mut replica_modes = self.replica_modes.lock();
            let mut resync_triggers = self.resync_triggers.lock();
            for worker in &workers {
                replica_modes.remove(worker);
                resync_triggers.remove(worker);
            }
        }
        tokio::join!(
            cancel_and_join_snapshot_tasks(worker_url, snapshot_sync),
            self.subscribers.remove_worker(worker_url),
        );

        // Drain the pump through an explicit marker before clearing the tree.
        // This also releases any pending snapshot acknowledgement.
        for worker in &workers {
            let (ack, received) = oneshot::channel();
            if self
                .event_tx
                .send(WorkerEvent::Detached {
                    worker: worker.clone(),
                    ack,
                })
                .await
                .is_err()
            {
                break;
            }
            if received.await.is_err() {
                warn!(
                    worker = ?worker,
                    "kv-events: event pump stopped before detached worker was drained",
                );
            }
        }

        for worker in &workers {
            self.tree.clear_worker(worker);
            self.cursors.lock().remove(worker);
        }
    }

    /// Number of worker URLs currently attached to the index.
    pub fn known_worker_count(&self) -> usize {
        self.workers.lock().len()
    }

    /// Stop snapshot tasks, live subscribers, and finally the event pump.
    pub async fn shutdown(&self) {
        let _lifecycle_guard = self.lifecycle.lock().await;
        let sync_handles: Vec<SnapshotSyncHandle> = {
            let mut workers = self.workers.lock();
            workers
                .values_mut()
                .flat_map(|entry| std::mem::take(&mut entry.snapshot_sync))
                .collect()
        };
        tokio::join!(
            cancel_and_join_snapshot_tasks("<shutdown>", sync_handles),
            self.subscribers.shutdown(),
        );

        self.pump_cancel.cancel();
        let handle = self.pump.lock().take();
        if let Some(handle) = handle {
            match tokio::time::timeout(Duration::from_secs(2), handle).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    warn!(error = %error, "kv-events pump task did not join cleanly")
                }
                Err(_) => warn!("kv-events pump task did not stop within 2s"),
            }
        }
    }
}

async fn cancel_and_join_snapshot_tasks(worker_url: &str, handles: Vec<SnapshotSyncHandle>) {
    for handle in &handles {
        handle.cancel.cancel();
    }
    for handle in handles {
        if let Err(error) = handle.join.await {
            warn!(
                worker_url = %worker_url,
                error = %error,
                "kv-events: snapshot sync task did not join cleanly",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event_config(block_size: u32, dp_size: u32, is_bigram: bool) -> EventConfig {
        EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30_100,
            topic: String::new(),
            block_size,
            dp_size,
            snapshot: None,
            is_bigram,
        }
    }

    #[tokio::test]
    async fn add_worker_rejects_block_size_mismatch() {
        let index = KvEventIndex::new();
        index.block_size_oracle().try_set(64).unwrap();

        index
            .add_worker("http://127.0.0.1:30100", Some(event_config(128, 1, false)))
            .await;

        assert_eq!(index.known_worker_count(), 0);
        index.shutdown().await;
    }

    #[tokio::test]
    async fn first_worker_seeds_hashing_configuration() {
        let index = KvEventIndex::new();
        assert_eq!(index.block_size_oracle().get(), None);

        // dp_size=0 stops before socket creation but still validates hashing.
        index
            .add_worker("http://127.0.0.1:30200", Some(event_config(64, 0, true)))
            .await;

        assert_eq!(index.block_size_oracle().get(), Some(64));
        assert!(index.block_size_oracle().is_bigram());
        index.shutdown().await;
    }

    #[tokio::test]
    async fn metadata_only_mode_seeds_oracle_without_registering_subscribers() {
        let oracle = BlockSizeOracle::new();
        let index = KvEventIndex::new_metadata_only_with_http_and_oracle(
            reqwest::Client::new(),
            Arc::clone(&oracle),
        );
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30400,
            topic: "kv-events".into(),
            block_size: 64,
            dp_size: 2,
            snapshot: None,
            is_bigram: true,
        };

        index.add_worker("http://127.0.0.1:30400", Some(cfg)).await;

        assert_eq!(oracle.get(), Some(64));
        assert!(oracle.is_bigram());
        assert_eq!(index.known_worker_count(), 0);
        index.shutdown().await;
    }
}
