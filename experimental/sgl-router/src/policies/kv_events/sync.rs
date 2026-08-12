// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Snapshot bootstrap and live-event catch-up for the KV placement index.
//!
//! This module owns the per-replica synchronization state machine. The public
//! [`super::index::KvEventIndex`] remains the lifecycle facade: it discovers
//! workers, starts subscribers, and attaches or detaches replicas. Snapshot
//! fetching, barrier matching, gap detection, and atomic publication live here
//! so those recovery rules do not leak into lifecycle orchestration.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::discovery::EventConfig;
use super::snapshot::{fetch_snapshot, PlacementSnapshot};
use super::subscriber::WorkerEvent;
use super::tree::{HashTree, KvWorkerId};
use super::wire::{KvCacheEvent, KvEventBatch};

/// A snapshot-capable replica remains invisible to cache-aware routing until
/// its snapshot barrier has been observed. Bound the temporary live-event
/// buffer so a missing barrier cannot grow router memory without limit.
const MAX_SYNC_BUFFERED_BATCHES: usize = 4096;
const SNAPSHOT_ACK_TIMEOUT: Duration = Duration::from_secs(8);
const SNAPSHOT_RETRY_BASE: Duration = Duration::from_millis(100);
const SNAPSHOT_RETRY_CAP: Duration = Duration::from_secs(2);

/// Consistency contract used when applying one replica's KV events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReplicaIndexMode {
    /// Preserve the legacy best-effort behavior: reject duplicates and stale
    /// events, but accept forward sequence gaps because no recovery source is
    /// available.
    LegacyBestEffort,
    /// Require a matching epoch and contiguous sequence numbers. Bootstrap or
    /// recover from a gap by rebuilding the replica from a snapshot before it
    /// becomes visible to routing.
    SnapshotRecoverable,
}

#[derive(Debug)]
pub(super) struct SnapshotSyncHandle {
    pub(super) cancel: CancellationToken,
    pub(super) trigger: mpsc::Sender<()>,
    pub(super) join: JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub(super) struct SnapshotTarget {
    endpoint: String,
}

pub(super) fn snapshot_target_for_rank(cfg: &EventConfig, dp_rank: u32) -> Option<SnapshotTarget> {
    let snapshot = cfg.snapshot.as_ref()?;
    if snapshot.protocol_version != 1 {
        return None;
    }
    let port = u16::try_from(u32::from(snapshot.port_base) + dp_rank).ok()?;
    Some(SnapshotTarget {
        endpoint: format!("tcp://{}:{port}", snapshot.host),
    })
}

pub(super) fn spawn_snapshot_sync(
    worker: KvWorkerId,
    target: SnapshotTarget,
    event_tx: mpsc::Sender<WorkerEvent>,
) -> SnapshotSyncHandle {
    // Capacity one intentionally coalesces repeated gap/reset signals while a
    // fetch is already in progress.
    let (trigger, trigger_rx) = mpsc::channel(1);
    let cancel = CancellationToken::new();
    let join = tokio::spawn(snapshot_sync_loop(
        worker,
        target,
        event_tx,
        cancel.clone(),
        trigger_rx,
    ));
    SnapshotSyncHandle {
        cancel,
        trigger,
        join,
    }
}

async fn snapshot_sync_loop(
    worker: KvWorkerId,
    target: SnapshotTarget,
    event_tx: mpsc::Sender<WorkerEvent>,
    cancel: CancellationToken,
    mut trigger_rx: mpsc::Receiver<()>,
) {
    let mut next_generation = 1_u64;
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            trigger = trigger_rx.recv() => {
                if trigger.is_none() {
                    return;
                }
            }
        }

        let mut delay = SNAPSHOT_RETRY_BASE;
        loop {
            let generation = next_generation;
            next_generation = next_generation.saturating_add(1);
            let (begin_tx, begin_rx) = oneshot::channel();
            let begin = WorkerEvent::BeginSync {
                worker: worker.clone(),
                generation,
                ack: begin_tx,
            };
            let sent = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                sent = event_tx.send(begin) => sent,
            };
            if sent.is_err() {
                return;
            }
            let armed = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                armed = tokio::time::timeout(SNAPSHOT_ACK_TIMEOUT, begin_rx) => armed,
            };
            let armed_ok = match armed {
                Ok(Ok(true)) => true,
                Ok(Ok(false)) => {
                    debug!(
                        worker = ?worker,
                        generation,
                        "kv-events: discarded stale resync trigger for READY replica",
                    );
                    break;
                }
                Ok(Err(_)) => {
                    warn!(
                        worker = ?worker,
                        generation,
                        "kv-events: BeginSync acknowledgement sender dropped; retrying",
                    );
                    false
                }
                Err(_) => {
                    warn!(
                        worker = ?worker,
                        generation,
                        "kv-events: timed out arming snapshot synchronization; retrying",
                    );
                    false
                }
            };
            if !armed_ok {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(delay) => {}
                }
                delay = (delay * 2).min(SNAPSHOT_RETRY_CAP);
                continue;
            }

            let result = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                result = fetch_snapshot(&target.endpoint, worker.dp_rank) => result,
            };
            match result {
                Ok(snapshot) => {
                    let epoch = snapshot.header.epoch.clone();
                    let barrier_seq = snapshot.header.barrier_seq;
                    let (ack_tx, ack_rx) = oneshot::channel();
                    let event = WorkerEvent::Snapshot {
                        worker: worker.clone(),
                        generation,
                        snapshot,
                        ack: ack_tx,
                    };
                    let sent = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        sent = event_tx.send(event) => sent,
                    };
                    if sent.is_err() {
                        return;
                    }
                    let ack = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        ack = tokio::time::timeout(SNAPSHOT_ACK_TIMEOUT, ack_rx) => ack,
                    };
                    match ack {
                        Ok(Ok(Ok(()))) => {
                            // Do not drain a queued trigger here. The pump may
                            // have acknowledged this snapshot and then observed
                            // a fresh gap before this task wakes up; discarding
                            // that signal would leave the replica SYNCING with
                            // no future recovery attempt. If the trigger is
                            // stale, the next BeginSync handshake rejects it
                            // before another provider request is sent.
                            info!(
                                worker = ?worker,
                                epoch,
                                barrier_seq,
                                "kv-events: placement snapshot synchronized; replica is READY",
                            );
                            break;
                        }
                        Ok(Ok(Err(reason))) => warn!(
                            worker = ?worker,
                            epoch,
                            barrier_seq,
                            reason,
                            "kv-events: placement snapshot rejected; retrying",
                        ),
                        Ok(Err(_)) => warn!(
                            worker = ?worker,
                            "kv-events: snapshot acknowledgement sender dropped; retrying",
                        ),
                        Err(_) => warn!(
                            worker = ?worker,
                            epoch,
                            barrier_seq,
                            "kv-events: timed out waiting for snapshot barrier; retrying",
                        ),
                    }
                }
                Err(error) => warn!(
                    worker = ?worker,
                    endpoint = %target.endpoint,
                    error = %error,
                    "kv-events: placement snapshot fetch failed; retrying",
                ),
            }

            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
            delay = (delay * 2).min(SNAPSHOT_RETRY_CAP);
        }
    }
}

#[derive(Debug)]
struct BufferedBatch {
    seq: i64,
    epoch: String,
    barrier_id: Option<String>,
    batch: KvEventBatch,
}

#[derive(Debug)]
struct PendingSnapshot {
    snapshot: PlacementSnapshot,
    ack: oneshot::Sender<Result<(), String>>,
}

#[derive(Debug)]
enum ReplicaSyncState {
    Syncing {
        generation: u64,
        buffered: Vec<BufferedBatch>,
        pending: Option<PendingSnapshot>,
    },
    Ready {
        epoch: String,
    },
}

impl ReplicaSyncState {
    fn syncing() -> Self {
        Self::Syncing {
            generation: 0,
            buffered: Vec::new(),
            pending: None,
        }
    }
}

fn trigger_resync(
    worker: &KvWorkerId,
    resync_triggers: &Mutex<HashMap<KvWorkerId, mpsc::Sender<()>>>,
) {
    if let Some(trigger) = resync_triggers.lock().get(worker).cloned() {
        let _ = trigger.try_send(());
    }
}

type PlacementRecords = HashMap<i64, Option<i64>>;
type OrderedPlacementRecords = Vec<(Option<i64>, Vec<i64>)>;

fn insert_record_chain(
    records: &mut PlacementRecords,
    mut parent: Option<i64>,
    block_hashes: &[i64],
) {
    for &hash in block_hashes {
        records.insert(hash, parent);
        parent = Some(hash);
    }
}

fn apply_batch_to_records(records: &mut PlacementRecords, batch: &KvEventBatch) {
    for event in &batch.events {
        match event {
            KvCacheEvent::BlockStored(block) => {
                insert_record_chain(records, block.parent_block_hash, &block.block_hashes);
            }
            KvCacheEvent::BlockRemoved(block) => {
                for hash in &block.block_hashes {
                    records.remove(hash);
                }
            }
            KvCacheEvent::AllBlocksCleared => records.clear(),
        }
    }
}

fn records_in_parent_order(records: &PlacementRecords) -> Result<OrderedPlacementRecords, String> {
    let mut indegree: HashMap<i64, usize> = records.keys().copied().map(|hash| (hash, 0)).collect();
    let mut children: HashMap<i64, Vec<i64>> = HashMap::new();

    for (&hash, &parent) in records {
        if let Some(parent) = parent.filter(|parent| records.contains_key(parent)) {
            *indegree
                .get_mut(&hash)
                .expect("every placement record has an indegree") += 1;
            children.entry(parent).or_default().push(hash);
        }
    }

    let mut ready: VecDeque<i64> = indegree
        .iter()
        .filter_map(|(&hash, &degree)| (degree == 0).then_some(hash))
        .collect();
    let mut ordered = Vec::with_capacity(records.len());
    while let Some(hash) = ready.pop_front() {
        ordered.push((records[&hash], vec![hash]));
        if let Some(block_children) = children.get(&hash) {
            for child in block_children {
                let degree = indegree
                    .get_mut(child)
                    .expect("every child placement record has an indegree");
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(*child);
                }
            }
        }
    }

    if ordered.len() != records.len() {
        return Err("snapshot placement records contain a parent cycle".into());
    }
    Ok(ordered)
}

fn apply_batch_to_tree(tree: &HashTree, worker: &KvWorkerId, batch: &KvEventBatch) {
    for event in &batch.events {
        match event {
            KvCacheEvent::BlockStored(block) => {
                tree.insert(worker, block.parent_block_hash, &block.block_hashes);
            }
            KvCacheEvent::BlockRemoved(block) => tree.remove(worker, &block.block_hashes),
            KvCacheEvent::AllBlocksCleared => tree.clear_worker(worker),
        }
    }
}

enum SnapshotFinish {
    Waiting {
        buffered: Vec<BufferedBatch>,
        pending: PendingSnapshot,
    },
    Ready {
        state: ReplicaSyncState,
        ack: oneshot::Sender<Result<(), String>>,
    },
    Invalid {
        reason: String,
        ack: oneshot::Sender<Result<(), String>>,
    },
}

fn finish_snapshot_sync(
    tree: &HashTree,
    cursors: &Mutex<HashMap<KvWorkerId, i64>>,
    worker: &KvWorkerId,
    mut buffered: Vec<BufferedBatch>,
    pending: PendingSnapshot,
) -> SnapshotFinish {
    let PendingSnapshot { snapshot, ack } = pending;
    let PlacementSnapshot { header, blocks } = snapshot;
    buffered.retain(|batch| batch.epoch == header.epoch && batch.seq >= header.barrier_seq);
    buffered.sort_by_key(|batch| batch.seq);

    let mut matching_barriers = buffered.iter().filter(|batch| {
        batch.seq == header.barrier_seq
            && batch.barrier_id.as_deref() == Some(header.barrier_id.as_str())
    });
    let Some(barrier) = matching_barriers.next() else {
        return SnapshotFinish::Waiting {
            buffered,
            pending: PendingSnapshot {
                snapshot: PlacementSnapshot { header, blocks },
                ack,
            },
        };
    };
    if !barrier.batch.events.is_empty() {
        return SnapshotFinish::Invalid {
            reason: "snapshot barrier batch must be empty".into(),
            ack,
        };
    }
    if matching_barriers.next().is_some() {
        return SnapshotFinish::Invalid {
            reason: "snapshot stream contains duplicate matching barriers".into(),
            ack,
        };
    }

    let mut expected = header.barrier_seq;
    for batch in &buffered {
        if batch.seq != expected {
            return SnapshotFinish::Invalid {
                reason: format!(
                    "event sequence discontinuity while applying snapshot: expected seq {expected}, got {}",
                    batch.seq
                ),
                ack,
            };
        }
        expected = expected.saturating_add(1);
    }

    let mut records = PlacementRecords::with_capacity(blocks.len());
    for block in blocks {
        insert_record_chain(&mut records, block.parent_block_hash, &block.block_hashes);
    }
    // The first batch is the validated empty barrier. Every remaining batch
    // is a strictly contiguous post-snapshot update.
    for batch in buffered.iter().skip(1) {
        apply_batch_to_records(&mut records, &batch.batch);
    }
    let records = match records_in_parent_order(&records) {
        Ok(records) => records,
        Err(reason) => return SnapshotFinish::Invalid { reason, ack },
    };

    let last_applied = expected.saturating_sub(1);
    tree.replace_worker(worker, &records);
    cursors.lock().insert(worker.clone(), last_applied);
    SnapshotFinish::Ready {
        state: ReplicaSyncState::Ready {
            epoch: header.epoch.clone(),
        },
        ack,
    }
}

fn handle_legacy_event(
    tree: &HashTree,
    cursors: &Mutex<HashMap<KvWorkerId, i64>>,
    ev: WorkerEvent,
) {
    match ev {
        WorkerEvent::PublisherReset { worker } => {
            if cursors.lock().remove(&worker).is_some() {
                info!(
                    worker = ?worker,
                    "kv-events pump: publisher reset; cursor cleared",
                );
            }
        }
        WorkerEvent::DecodeFailed { .. } => {
            // Legacy indexing has no recovery source. Preserve its previous
            // best-effort behavior and wait for the next decodable batch.
        }
        WorkerEvent::Batch {
            worker, seq, batch, ..
        } => {
            let prev = cursors.lock().get(&worker).copied();
            if prev.is_some_and(|last| seq <= last) {
                debug!(
                    worker = ?worker,
                    seq,
                    last_applied = prev,
                    "kv-events pump: out-of-order batch; skipping",
                );
                return;
            }
            apply_batch_to_tree(tree, &worker, &batch);
            cursors.lock().insert(worker, seq);
        }
        WorkerEvent::Snapshot { ack, .. } => {
            let _ = ack.send(Err("snapshot protocol is disabled for this replica".into()));
        }
        WorkerEvent::BeginSync { ack, .. } => {
            let _ = ack.send(false);
        }
        WorkerEvent::Detached { .. } => unreachable!("handled before mode dispatch"),
    }
}

fn handle_recoverable_event(
    tree: &HashTree,
    cursors: &Mutex<HashMap<KvWorkerId, i64>>,
    sync_states: &mut HashMap<KvWorkerId, ReplicaSyncState>,
    resync_triggers: &Mutex<HashMap<KvWorkerId, mpsc::Sender<()>>>,
    ev: WorkerEvent,
) {
    match ev {
        WorkerEvent::BeginSync {
            worker,
            generation,
            ack,
        } => {
            let state = sync_states
                .remove(&worker)
                .unwrap_or_else(ReplicaSyncState::syncing);
            match state {
                ReplicaSyncState::Ready { epoch } => {
                    sync_states.insert(worker, ReplicaSyncState::Ready { epoch });
                    let _ = ack.send(false);
                }
                ReplicaSyncState::Syncing {
                    generation: current_generation,
                    pending,
                    ..
                } if generation > current_generation => {
                    if let Some(stale) = pending {
                        let _ = stale
                            .ack
                            .send(Err("superseded by a newer sync attempt".into()));
                    }
                    sync_states.insert(
                        worker,
                        ReplicaSyncState::Syncing {
                            generation,
                            buffered: Vec::new(),
                            pending: None,
                        },
                    );
                    let _ = ack.send(true);
                }
                state @ ReplicaSyncState::Syncing { .. } => {
                    sync_states.insert(worker, state);
                    let _ = ack.send(false);
                }
            }
        }
        WorkerEvent::PublisherReset { worker } => {
            tree.clear_worker(&worker);
            cursors.lock().remove(&worker);
            if let Some(ReplicaSyncState::Syncing {
                pending: Some(pending),
                ..
            }) = sync_states.remove(&worker)
            {
                let _ = pending
                    .ack
                    .send(Err("publisher reset before snapshot became ready".into()));
            }
            sync_states.insert(worker.clone(), ReplicaSyncState::syncing());
            trigger_resync(&worker, resync_triggers);
            info!(worker = ?worker, "kv-events pump: publisher reset; replica is NOT_READY");
        }
        WorkerEvent::DecodeFailed { worker, seq } => {
            tree.clear_worker(&worker);
            cursors.lock().remove(&worker);
            if let Some(ReplicaSyncState::Syncing {
                pending: Some(pending),
                ..
            }) = sync_states.remove(&worker)
            {
                let _ = pending
                    .ack
                    .send(Err(format!("failed to decode live event at seq {seq}")));
            }
            sync_states.insert(worker.clone(), ReplicaSyncState::syncing());
            trigger_resync(&worker, resync_triggers);
            warn!(
                worker = ?worker,
                seq,
                "kv-events pump: live event decode failed; replica is NOT_READY",
            );
        }
        WorkerEvent::Batch {
            worker,
            seq,
            epoch,
            barrier_id,
            batch,
        } => {
            let Some(epoch) = epoch else {
                tree.clear_worker(&worker);
                cursors.lock().remove(&worker);
                sync_states.insert(worker.clone(), ReplicaSyncState::syncing());
                trigger_resync(&worker, resync_triggers);
                warn!(
                    worker = ?worker,
                    "kv-events pump: snapshot-capable replica emitted no epoch; resynchronizing",
                );
                return;
            };
            let state = sync_states
                .remove(&worker)
                .unwrap_or_else(ReplicaSyncState::syncing);
            match state {
                ReplicaSyncState::Ready { epoch: ready_epoch } => {
                    let prev = cursors.lock().get(&worker).copied();
                    if epoch == ready_epoch && prev.is_some_and(|last| seq <= last) {
                        debug!(
                            worker = ?worker,
                            seq,
                            last_applied = prev,
                            "kv-events pump: duplicate snapshot-era batch; skipping",
                        );
                        sync_states.insert(worker, ReplicaSyncState::Ready { epoch: ready_epoch });
                        return;
                    }
                    if epoch != ready_epoch || prev.is_none_or(|last| seq != last.saturating_add(1))
                    {
                        let reason = if epoch != ready_epoch {
                            format!("publisher epoch changed from {ready_epoch} to {epoch}")
                        } else {
                            format!(
                                "event sequence gap: expected {}, got {seq}",
                                prev.unwrap_or_default().saturating_add(1)
                            )
                        };
                        tree.clear_worker(&worker);
                        cursors.lock().remove(&worker);
                        sync_states.insert(
                            worker.clone(),
                            ReplicaSyncState::Syncing {
                                generation: 0,
                                buffered: vec![BufferedBatch {
                                    seq,
                                    epoch,
                                    barrier_id,
                                    batch,
                                }],
                                pending: None,
                            },
                        );
                        trigger_resync(&worker, resync_triggers);
                        warn!(
                            worker = ?worker,
                            reason,
                            "kv-events pump: placement view invalidated; replica is NOT_READY",
                        );
                        return;
                    }
                    apply_batch_to_tree(tree, &worker, &batch);
                    cursors.lock().insert(worker.clone(), seq);
                    sync_states.insert(worker, ReplicaSyncState::Ready { epoch: ready_epoch });
                }
                ReplicaSyncState::Syncing {
                    generation,
                    mut buffered,
                    mut pending,
                } => {
                    if pending
                        .as_ref()
                        .is_some_and(|snapshot| snapshot.snapshot.header.epoch != epoch)
                    {
                        if let Some(stale) = pending.take() {
                            let _ = stale.ack.send(Err(format!(
                                "publisher epoch changed to {epoch} during synchronization"
                            )));
                        }
                        buffered.clear();
                        trigger_resync(&worker, resync_triggers);
                    } else if buffered.last().is_some_and(|last| last.epoch != epoch) {
                        buffered.clear();
                        trigger_resync(&worker, resync_triggers);
                    }
                    buffered.push(BufferedBatch {
                        seq,
                        epoch,
                        barrier_id,
                        batch,
                    });
                    if buffered.len() > MAX_SYNC_BUFFERED_BATCHES {
                        if let Some(stale) = pending.take() {
                            let _ = stale.ack.send(Err(
                                "live-event buffer overflow while waiting for barrier".into(),
                            ));
                        }
                        buffered.clear();
                        trigger_resync(&worker, resync_triggers);
                        warn!(
                            worker = ?worker,
                            cap = MAX_SYNC_BUFFERED_BATCHES,
                            "kv-events pump: snapshot catch-up buffer overflow; retrying",
                        );
                    }

                    let Some(pending) = pending else {
                        sync_states.insert(
                            worker,
                            ReplicaSyncState::Syncing {
                                generation,
                                buffered,
                                pending: None,
                            },
                        );
                        return;
                    };
                    match finish_snapshot_sync(tree, cursors, &worker, buffered, pending) {
                        SnapshotFinish::Waiting { buffered, pending } => {
                            sync_states.insert(
                                worker,
                                ReplicaSyncState::Syncing {
                                    generation,
                                    buffered,
                                    pending: Some(pending),
                                },
                            );
                        }
                        SnapshotFinish::Ready { state, ack } => {
                            sync_states.insert(worker, state);
                            let _ = ack.send(Ok(()));
                        }
                        SnapshotFinish::Invalid { reason, ack } => {
                            tree.clear_worker(&worker);
                            cursors.lock().remove(&worker);
                            sync_states.insert(worker.clone(), ReplicaSyncState::syncing());
                            let _ = ack.send(Err(reason.clone()));
                            trigger_resync(&worker, resync_triggers);
                            warn!(worker = ?worker, reason, "kv-events pump: snapshot catch-up failed");
                        }
                    }
                }
            }
        }
        WorkerEvent::Snapshot {
            worker,
            generation,
            snapshot,
            ack,
        } => {
            let state = sync_states
                .remove(&worker)
                .unwrap_or_else(ReplicaSyncState::syncing);
            let buffered = match state {
                ReplicaSyncState::Ready { epoch } => {
                    sync_states.insert(worker, ReplicaSyncState::Ready { epoch });
                    let _ = ack.send(Err("snapshot arrived after replica became READY".into()));
                    return;
                }
                ReplicaSyncState::Syncing {
                    generation: current_generation,
                    buffered,
                    pending,
                } if generation == current_generation => {
                    if let Some(stale) = pending {
                        let _ = stale.ack.send(Err("superseded by a newer snapshot".into()));
                    }
                    buffered
                }
                state @ ReplicaSyncState::Syncing { .. } => {
                    sync_states.insert(worker, state);
                    let _ = ack.send(Err(format!(
                        "stale snapshot generation {generation} was not armed"
                    )));
                    return;
                }
            };
            let mut buffered = buffered;
            buffered.retain(|batch| batch.epoch == snapshot.header.epoch);
            let pending = PendingSnapshot { snapshot, ack };
            match finish_snapshot_sync(tree, cursors, &worker, buffered, pending) {
                SnapshotFinish::Waiting { buffered, pending } => {
                    sync_states.insert(
                        worker,
                        ReplicaSyncState::Syncing {
                            generation,
                            buffered,
                            pending: Some(pending),
                        },
                    );
                }
                SnapshotFinish::Ready { state, ack } => {
                    sync_states.insert(worker, state);
                    let _ = ack.send(Ok(()));
                }
                SnapshotFinish::Invalid { reason, ack } => {
                    tree.clear_worker(&worker);
                    cursors.lock().remove(&worker);
                    sync_states.insert(worker.clone(), ReplicaSyncState::syncing());
                    let _ = ack.send(Err(reason.clone()));
                    trigger_resync(&worker, resync_triggers);
                    debug!(
                        worker = ?worker,
                        reason,
                        "kv-events pump: rejected snapshot before publication",
                    );
                }
            }
        }
        WorkerEvent::Detached { .. } => unreachable!("handled before mode dispatch"),
    }
}

/// Drain [`WorkerEvent`]s and maintain the placement tree.
///
/// Legacy replicas retain the previous live-only behavior. Snapshot-capable
/// replicas move through `SYNCING -> READY`; only READY state is present in
/// the shared tree. A gap, epoch change, or publisher reset clears the replica
/// immediately and starts a fresh snapshot attempt.
pub(super) async fn pump_loop(
    tree: Arc<HashTree>,
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    replica_modes: Arc<Mutex<HashMap<KvWorkerId, ReplicaIndexMode>>>,
    resync_triggers: Arc<Mutex<HashMap<KvWorkerId, mpsc::Sender<()>>>>,
    cancel: CancellationToken,
    mut rx: mpsc::Receiver<WorkerEvent>,
) {
    let mut sync_states: HashMap<KvWorkerId, ReplicaSyncState> = HashMap::new();
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
        // load-bearing: `remove_worker` clears the replica's mode entry BEFORE
        // joining the subscriber task, so any event still buffered when the
        // pump reaches it would otherwise re-pollute the tree.
        if let WorkerEvent::Detached { worker, ack } = ev {
            if let Some(ReplicaSyncState::Syncing {
                pending: Some(pending),
                ..
            }) = sync_states.remove(&worker)
            {
                let _ = pending.ack.send(Err("worker detached".into()));
            }
            let _ = ack.send(());
            continue;
        }

        let worker = ev.worker();
        let Some(mode) = replica_modes.lock().get(worker).copied() else {
            debug!(
                worker = ?worker,
                "kv-events pump: dropping event from detached worker",
            );
            continue;
        };

        match mode {
            ReplicaIndexMode::LegacyBestEffort => {
                handle_legacy_event(&tree, &cursors, ev);
            }
            ReplicaIndexMode::SnapshotRecoverable => {
                handle_recoverable_event(&tree, &cursors, &mut sync_states, &resync_triggers, ev);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::kv_events::snapshot::{SnapshotBlock, SnapshotHeader};
    use crate::policies::kv_events::wire::{BlockRemoved, BlockStored};
    use std::collections::HashSet;
    use tokio::time::timeout;
    use zeromq::{Endpoint, RouterSocket, Socket, SocketRecv};

    struct PumpHarness {
        tree: Arc<HashTree>,
        cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
        tx: mpsc::Sender<WorkerEvent>,
        pump: JoinHandle<()>,
    }

    fn worker(url: &str) -> KvWorkerId {
        KvWorkerId::new(url.into(), 0)
    }

    fn batch(events: Vec<KvCacheEvent>) -> KvEventBatch {
        KvEventBatch {
            ts: 0.0,
            events,
            attn_dp_rank: None,
        }
    }

    fn stored(parent_block_hash: Option<i64>, block_hashes: Vec<i64>) -> KvCacheEvent {
        KvCacheEvent::BlockStored(BlockStored {
            parent_block_hash,
            block_hashes,
            token_ids: vec![],
            block_size: 64,
            lora_id: None,
            medium: None,
        })
    }

    fn spawn_pump(live: &[KvWorkerId], recoverable_workers: &[KvWorkerId]) -> PumpHarness {
        let tree = Arc::new(HashTree::new());
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let recoverable_workers: HashSet<_> = recoverable_workers.iter().collect();
        let replica_modes = Arc::new(Mutex::new(
            live.iter()
                .cloned()
                .map(|worker| {
                    let mode = if recoverable_workers.contains(&worker) {
                        ReplicaIndexMode::SnapshotRecoverable
                    } else {
                        ReplicaIndexMode::LegacyBestEffort
                    };
                    (worker, mode)
                })
                .collect(),
        ));
        let resync_triggers = Arc::new(Mutex::new(HashMap::new()));
        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::channel(16);
        let pump = tokio::spawn(pump_loop(
            tree.clone(),
            cursors.clone(),
            replica_modes,
            resync_triggers,
            cancel,
            rx,
        ));
        PumpHarness {
            tree,
            cursors,
            tx,
            pump,
        }
    }

    fn live_event(
        worker: KvWorkerId,
        seq: i64,
        epoch: Option<&str>,
        barrier_id: Option<&str>,
        events: Vec<KvCacheEvent>,
    ) -> WorkerEvent {
        WorkerEvent::Batch {
            worker,
            seq,
            epoch: epoch.map(str::to_owned),
            barrier_id: barrier_id.map(str::to_owned),
            batch: batch(events),
        }
    }

    fn snapshot(replica_rank: u32, barrier_seq: i64, epoch: &str) -> PlacementSnapshot {
        PlacementSnapshot {
            header: SnapshotHeader {
                version: 1,
                epoch: epoch.into(),
                replica_rank,
                resume_seq: barrier_seq + 1,
                barrier_seq,
                barrier_id: "barrier-a".into(),
                record_count: 1,
            },
            blocks: vec![SnapshotBlock {
                parent_block_hash: None,
                block_hashes: vec![10],
            }],
        }
    }

    async fn begin_sync(
        tx: &mpsc::Sender<WorkerEvent>,
        worker: &KvWorkerId,
        generation: u64,
    ) -> bool {
        let (ack, received) = oneshot::channel();
        tx.send(WorkerEvent::BeginSync {
            worker: worker.clone(),
            generation,
            ack,
        })
        .await
        .unwrap();
        received.await.unwrap()
    }

    async fn publish_snapshot(
        tx: &mpsc::Sender<WorkerEvent>,
        worker: &KvWorkerId,
        epoch: &str,
        barrier_seq: i64,
    ) {
        assert!(begin_sync(tx, worker, 1).await);
        tx.send(live_event(
            worker.clone(),
            barrier_seq,
            Some(epoch),
            Some("barrier-a"),
            vec![],
        ))
        .await
        .unwrap();
        let (ack, received) = oneshot::channel();
        tx.send(WorkerEvent::Snapshot {
            worker: worker.clone(),
            generation: 1,
            snapshot: snapshot(worker.dp_rank, barrier_seq, epoch),
            ack,
        })
        .await
        .unwrap();
        assert_eq!(received.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn cancelling_snapshot_fetch_stops_without_emitting_a_snapshot() {
        let mut provider = RouterSocket::new();
        let endpoint = provider
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind snapshot provider");
        let port = match endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };
        let (event_tx, mut event_rx) = mpsc::channel(1);
        let handle = spawn_snapshot_sync(
            worker("http://snapshot"),
            SnapshotTarget {
                endpoint: format!("tcp://127.0.0.1:{port}"),
            },
            event_tx,
        );

        handle.trigger.send(()).await.unwrap();
        let begin = timeout(Duration::from_secs(1), event_rx.recv())
            .await
            .expect("BeginSync was not emitted")
            .expect("snapshot event channel closed");
        match begin {
            WorkerEvent::BeginSync {
                generation, ack, ..
            } => {
                assert_eq!(generation, 1);
                ack.send(true).unwrap();
            }
            other => panic!("expected BeginSync, got {other:?}"),
        }
        timeout(Duration::from_secs(1), provider.recv())
            .await
            .expect("snapshot request was not received")
            .expect("snapshot provider recv failed");

        handle.cancel.cancel();
        timeout(Duration::from_millis(500), handle.join)
            .await
            .expect("snapshot task did not stop after cancellation")
            .expect("snapshot task panicked");
        assert!(event_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn legacy_replica_applies_events_and_filters_duplicates() {
        let id = worker("http://legacy");
        let harness = spawn_pump(std::slice::from_ref(&id), &[]);

        harness
            .tx
            .send(live_event(
                id.clone(),
                5,
                None,
                None,
                vec![stored(None, vec![10])],
            ))
            .await
            .unwrap();
        harness
            .tx
            .send(live_event(
                id.clone(),
                3,
                None,
                None,
                vec![KvCacheEvent::BlockRemoved(BlockRemoved {
                    block_hashes: vec![10],
                    medium: None,
                })],
            ))
            .await
            .unwrap();
        drop(harness.tx);
        harness.pump.await.unwrap();

        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 1);
        assert_eq!(harness.cursors.lock().get(&id).copied(), Some(5));
    }

    #[tokio::test]
    async fn snapshot_replica_is_hidden_until_barrier_then_published() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));

        // This predates the snapshot cut and must not leak into routing.
        harness
            .tx
            .send(live_event(
                id.clone(),
                4,
                Some("epoch-a"),
                None,
                vec![stored(None, vec![4])],
            ))
            .await
            .unwrap();
        publish_snapshot(&harness.tx, &id, "epoch-a", 5).await;
        harness
            .tx
            .send(live_event(
                id.clone(),
                6,
                Some("epoch-a"),
                None,
                vec![stored(Some(10), vec![60])],
            ))
            .await
            .unwrap();
        drop(harness.tx);
        harness.pump.await.unwrap();

        assert_eq!(harness.tree.match_prefix(None, &[4]).matched_blocks, 0);
        assert_eq!(harness.tree.match_prefix(None, &[10, 60]).matched_blocks, 2);
        assert_eq!(harness.cursors.lock().get(&id).copied(), Some(6));
    }

    #[tokio::test]
    async fn stale_begin_sync_does_not_invalidate_a_ready_replica() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        publish_snapshot(&harness.tx, &id, "epoch-a", 5).await;

        assert!(!begin_sync(&harness.tx, &id, 2).await);
        drop(harness.tx);
        harness.pump.await.unwrap();

        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 1);
        assert_eq!(harness.cursors.lock().get(&id).copied(), Some(5));
    }

    #[tokio::test]
    async fn stale_snapshot_generation_cannot_replace_the_armed_attempt() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        assert!(begin_sync(&harness.tx, &id, 2).await);

        let (stale_ack, stale_received) = oneshot::channel();
        harness
            .tx
            .send(WorkerEvent::Snapshot {
                worker: id.clone(),
                generation: 1,
                snapshot: snapshot(id.dp_rank, 5, "epoch-a"),
                ack: stale_ack,
            })
            .await
            .unwrap();
        assert!(matches!(
            stale_received.await.unwrap(),
            Err(reason) if reason.contains("stale snapshot generation 1")
        ));

        harness
            .tx
            .send(live_event(
                id.clone(),
                5,
                Some("epoch-a"),
                Some("barrier-a"),
                vec![],
            ))
            .await
            .unwrap();
        let (ack, received) = oneshot::channel();
        harness
            .tx
            .send(WorkerEvent::Snapshot {
                worker: id.clone(),
                generation: 2,
                snapshot: snapshot(id.dp_rank, 5, "epoch-a"),
                ack,
            })
            .await
            .unwrap();
        assert_eq!(received.await.unwrap(), Ok(()));

        drop(harness.tx);
        harness.pump.await.unwrap();
        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 1);
    }

    #[tokio::test]
    async fn unordered_snapshot_records_are_published_in_parent_order() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        assert!(begin_sync(&harness.tx, &id, 1).await);
        harness
            .tx
            .send(live_event(
                id.clone(),
                5,
                Some("epoch-a"),
                Some("barrier-a"),
                vec![],
            ))
            .await
            .unwrap();

        let mut unordered = snapshot(id.dp_rank, 5, "epoch-a");
        unordered.header.record_count = 2;
        unordered.blocks = vec![
            SnapshotBlock {
                parent_block_hash: Some(10),
                block_hashes: vec![20],
            },
            SnapshotBlock {
                parent_block_hash: None,
                block_hashes: vec![10],
            },
        ];
        let (ack, received) = oneshot::channel();
        harness
            .tx
            .send(WorkerEvent::Snapshot {
                worker: id.clone(),
                generation: 1,
                snapshot: unordered,
                ack,
            })
            .await
            .unwrap();
        assert_eq!(received.await.unwrap(), Ok(()));

        drop(harness.tx);
        harness.pump.await.unwrap();
        assert_eq!(harness.tree.match_prefix(None, &[10, 20]).matched_blocks, 2);
    }

    #[tokio::test]
    async fn duplicate_seq_during_snapshot_catch_up_is_rejected() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        assert!(begin_sync(&harness.tx, &id, 1).await);

        harness
            .tx
            .send(live_event(
                id.clone(),
                5,
                Some("epoch-a"),
                Some("barrier-a"),
                vec![],
            ))
            .await
            .unwrap();
        harness
            .tx
            .send(live_event(
                id.clone(),
                5,
                Some("epoch-a"),
                None,
                vec![stored(None, vec![50])],
            ))
            .await
            .unwrap();

        let (ack, received) = oneshot::channel();
        harness
            .tx
            .send(WorkerEvent::Snapshot {
                worker: id.clone(),
                generation: 1,
                snapshot: snapshot(id.dp_rank, 5, "epoch-a"),
                ack,
            })
            .await
            .unwrap();

        let result = received.await.unwrap();
        assert!(matches!(
            result,
            Err(reason) if reason.contains("expected seq 6, got 5")
        ));

        drop(harness.tx);
        harness.pump.await.unwrap();
        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 0);
        assert_eq!(harness.tree.match_prefix(None, &[50]).matched_blocks, 0);
        assert_eq!(harness.cursors.lock().get(&id), None);
    }

    #[tokio::test]
    async fn non_empty_snapshot_barrier_is_rejected() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        assert!(begin_sync(&harness.tx, &id, 1).await);

        harness
            .tx
            .send(live_event(
                id.clone(),
                5,
                Some("epoch-a"),
                Some("barrier-a"),
                vec![stored(None, vec![50])],
            ))
            .await
            .unwrap();

        let (ack, received) = oneshot::channel();
        harness
            .tx
            .send(WorkerEvent::Snapshot {
                worker: id.clone(),
                generation: 1,
                snapshot: snapshot(id.dp_rank, 5, "epoch-a"),
                ack,
            })
            .await
            .unwrap();

        assert!(matches!(
            received.await.unwrap(),
            Err(reason) if reason.contains("barrier batch must be empty")
        ));

        drop(harness.tx);
        harness.pump.await.unwrap();
        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 0);
        assert_eq!(harness.tree.match_prefix(None, &[50]).matched_blocks, 0);
        assert_eq!(harness.cursors.lock().get(&id), None);
    }

    #[tokio::test]
    async fn steady_state_gap_invalidates_snapshot_replica() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        publish_snapshot(&harness.tx, &id, "epoch-a", 5).await;

        // seq=6 is missing, so the entire replica view becomes untrusted.
        harness
            .tx
            .send(live_event(id.clone(), 7, Some("epoch-a"), None, vec![]))
            .await
            .unwrap();
        drop(harness.tx);
        harness.pump.await.unwrap();

        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 0);
        assert_eq!(harness.cursors.lock().get(&id), None);
    }

    #[tokio::test]
    async fn decode_failure_immediately_invalidates_snapshot_replica() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        publish_snapshot(&harness.tx, &id, "epoch-a", 5).await;

        harness
            .tx
            .send(WorkerEvent::DecodeFailed {
                worker: id.clone(),
                seq: 6,
            })
            .await
            .unwrap();
        drop(harness.tx);
        harness.pump.await.unwrap();

        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 0);
        assert_eq!(harness.cursors.lock().get(&id), None);
    }

    #[tokio::test]
    async fn epoch_change_invalidates_snapshot_replica() {
        let id = worker("http://snapshot");
        let harness = spawn_pump(std::slice::from_ref(&id), std::slice::from_ref(&id));
        publish_snapshot(&harness.tx, &id, "epoch-a", 5).await;

        harness
            .tx
            .send(live_event(id.clone(), 6, Some("epoch-b"), None, vec![]))
            .await
            .unwrap();
        drop(harness.tx);
        harness.pump.await.unwrap();

        assert_eq!(harness.tree.match_prefix(None, &[10]).matched_blocks, 0);
    }

    #[tokio::test]
    async fn epoch_change_invalidates_only_the_restarted_dp_rank() {
        let rank0 = KvWorkerId::new("http://snapshot".into(), 0);
        let rank1 = KvWorkerId::new("http://snapshot".into(), 1);
        let workers = [rank0.clone(), rank1.clone()];
        let harness = spawn_pump(&workers, &workers);
        publish_snapshot(&harness.tx, &rank0, "rank-0-epoch-a", 5).await;
        publish_snapshot(&harness.tx, &rank1, "rank-1-epoch-a", 5).await;

        let before_restart = harness.tree.match_prefix(None, &[10]);
        assert_eq!(before_restart.workers, workers.into_iter().collect());

        harness
            .tx
            .send(live_event(
                rank0.clone(),
                6,
                Some("rank-0-epoch-b"),
                None,
                vec![],
            ))
            .await
            .unwrap();
        drop(harness.tx);
        harness.pump.await.unwrap();

        let after_restart = harness.tree.match_prefix(None, &[10]);
        assert_eq!(after_restart.workers, HashSet::from([rank1.clone()]));
        assert_eq!(harness.cursors.lock().get(&rank0), None);
        assert_eq!(harness.cursors.lock().get(&rank1).copied(), Some(5));
    }
}
