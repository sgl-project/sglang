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
//! `KvEventIndex::add_worker` / `remove_worker` are called from the worker
//! manager (`workers::manager`) on every `DiscoveryEvent::Added` /
//! `DiscoveryEvent::Removed`.  The selector that consumes the tree (cache-
//! aware-zmq) lands in M4.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use super::discovery::{fetch_event_config, EventConfig};
use super::subscriber::{KvEventSubscriberRegistry, WorkerEvent};
use super::tree::{HashTree, KvWorkerId};
use super::wire::KvCacheEvent;

/// Channel buffer between the subscriber registry and the pump task.
///
/// Bounded so a misbehaving publisher cannot exhaust memory.  Realistic
/// per-worker event rates are < 1 kHz; a 1024-deep buffer absorbs a
/// half-second burst at 2 kHz before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

/// Per-worker bookkeeping kept inside [`KvEventIndex`] so `remove_worker`
/// knows the DP fan-out without re-fetching `/server_info`.
#[derive(Debug, Clone)]
struct WorkerEntry {
    dp_size: u32,
}

/// Bundle of `HashTree` + `KvEventSubscriberRegistry` + pump task.
///
/// Construct one instance per router process and hand it to the worker
/// manager as `Option<Arc<KvEventIndex>>` — `None` disables the cache-
/// aware-zmq path entirely (i.e. all selection policies are pure
/// round-robin / random / power-of-two).
pub struct KvEventIndex {
    tree: Arc<HashTree>,
    subscribers: Arc<KvEventSubscriberRegistry>,
    pump: Mutex<Option<JoinHandle<()>>>,
    workers: Mutex<HashMap<String, WorkerEntry>>,
    http: reqwest::Client,
    /// Per-`(worker_url, dp_rank)` last-applied sequence number.  The
    /// subscriber forwards every batch with no de-dup; this map filters
    /// any batch whose `seq` is not strictly greater than the previously
    /// applied one.  Used for cursor recovery semantics on subscriber
    /// restart (manager-level resume; ZMQ itself does not replay).
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
}

impl KvEventIndex {
    /// Build an empty index and spawn the pump task.  The pump runs until
    /// the index is dropped or every sender to the internal mpsc closes.
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
        let tree = Arc::new(HashTree::new());
        let (tx, rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(tx));
        let cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>> = Arc::new(Mutex::new(HashMap::new()));
        let pump = tokio::spawn(pump_loop(tree.clone(), cursors.clone(), rx));
        Arc::new(Self {
            tree,
            subscribers,
            pump: Mutex::new(Some(pump)),
            workers: Mutex::new(HashMap::new()),
            http,
            cursors,
        })
    }

    /// Return a clone of the underlying tree handle for cache-aware
    /// selection (M4) and metrics.
    pub fn tree(&self) -> Arc<HashTree> {
        self.tree.clone()
    }

    /// Register a worker.  Fetches `/server_info` to learn the publisher
    /// endpoint and opens one SUB per advertised DP rank.  If the worker
    /// is not publishing KV events (older SGLang, opt-out config), this
    /// is a logged no-op — the worker still routes via the non-cache-
    /// aware policies.
    pub async fn add_worker(&self, worker_url: &str) {
        let cfg: EventConfig = match fetch_event_config(worker_url, &self.http).await {
            Ok(Some(c)) => c,
            Ok(None) => {
                debug!(
                    worker_url = %worker_url,
                    "kv-events: worker is not publishing; skipping subscriber",
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
        };
        info!(
            worker_url = %worker_url,
            dp_size = cfg.dp_size,
            port_base = cfg.port_base,
            "kv-events: subscribing",
        );
        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerEntry {
                dp_size: cfg.dp_size,
            },
        );
        self.subscribers.add_worker(worker_url, &cfg).await;
    }

    /// Tear down a worker's subscribers and clear it from the tree.
    /// Idempotent: a remove for a worker that was never added is a no-op.
    pub async fn remove_worker(&self, worker_url: &str) {
        let Some(entry) = self.workers.lock().remove(worker_url) else {
            return;
        };
        self.subscribers.remove_worker(worker_url).await;
        // Drop each rank's tree state.  We don't retain the cursor — a
        // re-added worker may legitimately have a fresh publisher whose
        // sequence numbers restart from 1.
        let mut cursors = self.cursors.lock();
        for dp_rank in 0..entry.dp_size {
            let id = KvWorkerId {
                url: worker_url.to_string(),
                dp_rank,
            };
            self.tree.clear_worker(&id);
            cursors.remove(&id);
        }
    }

    /// Shut down the pump task.  Must be awaited so in-flight batches are
    /// fully applied before the runtime tears down; cancels any
    /// outstanding SUB connections via the subscriber registry.
    pub async fn shutdown(&self) {
        self.subscribers.shutdown().await;
        let handle = self.pump.lock().take();
        if let Some(h) = handle {
            // The pump's mpsc rx ends when every cloned Sender drops; the
            // registry's shutdown above releases the per-worker senders.
            // A short timeout guards against pathological cases.
            match tokio::time::timeout(Duration::from_secs(2), h).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!(error = %e, "kv-events pump task did not join cleanly"),
                Err(_) => warn!("kv-events pump task did not stop within 2s"),
            }
        }
    }
}

/// Drain `WorkerEvent`s and apply each batch to the tree.  Out-of-order /
/// duplicate batches (seq ≤ last_applied) are skipped; the cursor map is
/// updated on every applied batch.
async fn pump_loop(
    tree: Arc<HashTree>,
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    mut rx: mpsc::Receiver<WorkerEvent>,
) {
    while let Some(ev) = rx.recv().await {
        let prev = cursors.lock().get(&ev.worker).copied();
        if let Some(p) = prev {
            if ev.seq <= p {
                debug!(
                    worker = ?ev.worker,
                    seq = ev.seq,
                    last_applied = p,
                    "kv-events pump: out-of-order batch; skipping",
                );
                continue;
            }
        }
        for event in &ev.batch.events {
            match event {
                KvCacheEvent::BlockStored(b) => {
                    tree.insert(&ev.worker, b.parent_block_hash, &b.block_hashes);
                }
                KvCacheEvent::BlockRemoved(b) => {
                    tree.remove(&ev.worker, &b.block_hashes);
                }
                KvCacheEvent::AllBlocksCleared => {
                    tree.clear_worker(&ev.worker);
                }
            }
        }
        cursors.lock().insert(ev.worker.clone(), ev.seq);
    }
    debug!("kv-events pump: receiver closed; exiting");
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Direct test of the pump loop's tree application — no sockets.
    #[tokio::test]
    async fn pump_applies_block_stored_to_tree() {
        let tree = Arc::new(HashTree::new());
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = mpsc::channel(4);
        let pump = tokio::spawn(pump_loop(tree.clone(), cursors.clone(), rx));

        let id = worker_id("http://w1", 0);
        tx.send(WorkerEvent {
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
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[10, 20, 30]);
        assert_eq!(m.matched_blocks, 3);
        assert!(m.workers.contains(&id), "tree must hold the worker");
    }

    /// Out-of-order seq is filtered: a batch with seq <= last_applied is
    /// dropped silently and does not mutate the tree.
    #[tokio::test]
    async fn pump_filters_out_of_order_seq() {
        let tree = Arc::new(HashTree::new());
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = mpsc::channel(4);
        let pump = tokio::spawn(pump_loop(tree.clone(), cursors.clone(), rx));

        let id = worker_id("http://w1", 0);
        // Apply seq=5 with block 10.
        tx.send(WorkerEvent {
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
        // Then a duplicate-style seq=3 that tries to remove block 10.  Must
        // be dropped.
        tx.send(WorkerEvent {
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
        let tree = Arc::new(HashTree::new());
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = mpsc::channel(4);
        let pump = tokio::spawn(pump_loop(tree.clone(), cursors.clone(), rx));

        let id = worker_id("http://w1", 0);
        tx.send(WorkerEvent {
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
        tx.send(WorkerEvent {
            worker: id.clone(),
            seq: 2,
            batch: batch(vec![KvCacheEvent::AllBlocksCleared]),
        })
        .await
        .unwrap();
        drop(tx);
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 0, "AllBlocksCleared must purge the worker");
    }
}
