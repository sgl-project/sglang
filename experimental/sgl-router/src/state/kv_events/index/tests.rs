//! Tests of the `KvEventIndex` facade and its pump.

use super::test_support::*;
use super::*;
use crate::state::kv_events::wire::{BlockRemoved, BlockStored};
use crate::state::load_monitor::engine_reported_load::LoadStat;

/// A restarted publisher renumbers from 0, so its cursor MUST be cleared.
///
/// Without this, every post-restart batch has `seq < last_applied` and is
/// dropped as out-of-order: the rank's tree freezes at pre-restart state
/// forever while every metric reports healthy.
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
        h.tree.match_prefix(None, &[22]).workers().contains(&id),
        "a restarted publisher's stream must apply, not be filtered as stale",
    );
    assert_eq!(h.cursors.lock().get(&id).copied(), Some(1));
    // The reset also clears pre-restart state, on this bootstrap-disabled
    // path too: the restarted engine holds none of those blocks.
    assert!(
        !h.tree.match_prefix(None, &[11]).workers().contains(&id),
        "pre-reset blocks must not survive as false cache hits",
    );
}

/// The same restart without the `END_SEQ` that raises `PublisherReset` — the
/// engine killed and restarted in place. Its batch 0 lands behind the old
/// stream's cursor; skipping it as out-of-order would filter the whole new
/// stream until it overtook the old numbering, serving the dead cache's blocks
/// meanwhile.
#[tokio::test]
async fn pump_batch_zero_behind_a_cursor_restarts_the_rank_without_bootstrap() {
    let id = worker_id("http://w1", 0);
    let h = spawn_pump(std::slice::from_ref(&id));

    for (seq, hash) in [(9, 11), (0, 22), (1, 33)] {
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq,
            batch: batch(vec![stored(None, vec![hash])]),
        })
        .await
        .unwrap();
    }
    drop(h.tx);
    drop(h.ctrl_tx);
    h.pump.await.unwrap();

    assert!(
        !h.tree.match_prefix(None, &[11]).workers().contains(&id),
        "the dead stream's blocks must not survive as false cache hits",
    );
    for live in [22, 33] {
        assert!(
            h.tree.match_prefix(None, &[live]).workers().contains(&id),
            "new-stream block {live} must be applied, not filtered",
        );
    }
    assert_eq!(h.cursors.lock().get(&id).copied(), Some(1));
}

/// A backwards step that does not land on batch 0 is still filtered: past the
/// origin, a cursor may be a graft watermark ahead of anything received, so a
/// lower seq means "already reflected", not "restarted".
#[tokio::test]
async fn pump_regression_past_the_origin_is_still_filtered_without_bootstrap() {
    let id = worker_id("http://w1", 0);
    let h = spawn_pump(std::slice::from_ref(&id));

    for (seq, hash) in [(9, 11), (3, 22)] {
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq,
            batch: batch(vec![stored(None, vec![hash])]),
        })
        .await
        .unwrap();
    }
    drop(h.tx);
    drop(h.ctrl_tx);
    h.pump.await.unwrap();

    assert!(h.tree.match_prefix(None, &[11]).workers().contains(&id));
    assert!(!h.tree.match_prefix(None, &[22]).workers().contains(&id));
    assert_eq!(h.cursors.lock().get(&id).copied(), Some(9));
}

/// `ForgetRanks` must drop the pump-local queue and splice proof, or a
/// re-added worker inherits the dead incarnation's high sequence numbers and
/// its fresh stream reads as a permanent gap.
#[tokio::test]
async fn pump_forget_ranks_drops_held_state_and_tree() {
    let id = worker_id("http://w1", 0);
    let tracker = pending_tracker(std::slice::from_ref(&id));
    let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

    // Grafted tree state and a deferred splice proof to be torn down.
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
            done: None,
        })
        .await
        .unwrap();
    drop(h.tx);
    drop(h.ctrl_tx);
    h.pump.await.unwrap();

    assert!(
        !h.tree
            .match_prefix(None, &[100, 200])
            .workers()
            .contains(&id),
        "ForgetRanks must clear the worker's carriers on the pump",
    );
    assert!(
        h.cursors.lock().get(&id).is_none(),
        "a stale cursor would filter the re-added worker's fresh stream",
    );
}

/// `ForgetRanks` drops `held` itself.
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
            done: None,
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
    assert!(h
        .tree
        .match_prefix(None, &[100, 200])
        .workers()
        .contains(&id));
}

/// A publisher reset means the engine came back with an empty cache, so
/// the grafted blocks are now false cache hits that nothing will ever
/// evict — the `BlockRemoved` for them was the old publisher's to send.
/// And the dropped splice proof has to be tallied, or the rank's verdict
/// never lands and `bootstrap_rank_total` stops summing to the ranks that
/// finished.
#[tokio::test]
async fn pump_publisher_reset_after_a_graft_clears_the_tree_and_tallies() {
    let id = worker_id("http://w1", 0);
    let tracker = pending_tracker(std::slice::from_ref(&id));
    let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), Arc::clone(&tracker));

    // Graft with no held batch, so the splice proof is deferred.
    h.ctrl_tx
        .send(PumpControl::ApplySnapshot {
            obligations: obligations(&tracker, std::slice::from_ref(&id)),
            vetted: Box::new(vetted_for(&id, 7)),
        })
        .await
        .unwrap();
    h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
        .await
        .unwrap();
    drop(h.tx);
    drop(h.ctrl_tx);
    h.pump.await.unwrap();

    assert!(
        !h.tree
            .match_prefix(None, &[100, 200])
            .workers()
            .contains(&id),
        "a restarted engine holds none of the grafted blocks",
    );
    assert_eq!(
        rank_count(&tracker, RankOutcome::PublisherReset.as_label()),
        1,
        "the discarded proof must land a verdict",
    );
    assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
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
    assert!(m.holds(&id), "tree must hold the worker");
}

/// The pump must carry each event's `medium` into the tree. The engine's
/// write-back sequence for a backed-up block is a host-tagged store
/// followed by a device-tagged removal; applied tier-blind, the removal
/// erased the worker and every repeat of that prefix routed cold for the
/// whole host retention horizon.
#[tokio::test]
async fn pump_keeps_host_backed_block_owned_across_device_eviction() {
    let id = worker_id("http://w1", 0);
    let h = spawn_pump(std::slice::from_ref(&id));
    let (tree, tx, pump) = (h.tree, h.tx, h.pump);

    let stored = |medium: Option<&str>| {
        KvCacheEvent::BlockStored(BlockStored {
            parent_block_hash: None,
            block_hashes: vec![10, 20],
            token_ids: vec![],
            block_size: 64,
            lora_id: None,
            medium: medium.map(str::to_owned),
        })
    };
    let removed = |medium: Option<&str>| {
        KvCacheEvent::BlockRemoved(BlockRemoved {
            block_hashes: vec![20],
            medium: medium.map(str::to_owned),
        })
    };
    tx.send(WorkerEvent::Batch {
        worker: id.clone(),
        seq: 1,
        batch: batch(vec![
            stored(Some("GPU")),
            stored(Some("CPU_PINNED")),
            removed(Some("GPU")),
        ]),
    })
    .await
    .unwrap();
    drop(tx);
    pump.await.unwrap();

    let m = tree.match_prefix(None, &[10, 20]);
    assert_eq!(m.matched_blocks, 2, "host copy keeps the block routable");
    assert!(m.holds(&id));
    assert!(!m.device_workers().contains(&id), "device copy is gone");
}

/// The metadata-only gate. Its whole justification is that a structural
/// zero would be read as "the tier stream is not reaching the router" — a
/// different fault with a different fix — so the gate itself needs pinning:
/// inverting it leaves every test green while `/metrics` starts lying.
#[tokio::test]
async fn metrics_source_is_none_only_without_a_local_tree() {
    let http = reqwest::Client::builder().build().unwrap();
    let with_tree = KvEventIndex::new_with_http_and_oracle(http.clone(), BlockSizeOracle::new());
    assert!(
        with_tree.metrics_source().is_some(),
        "a router maintaining its own tree must publish the tier series",
    );
    let metadata_only =
        KvEventIndex::new_metadata_only_with_http_and_oracle(http, BlockSizeOracle::new());
    assert!(
        metadata_only.metrics_source().is_none(),
        "an external-Indexer router must emit nothing rather than a structural zero",
    );
}

/// Every applied event is tallied by kind and medium, blocks included, so
/// the scrape can show the tier stream the tree is consuming. An
/// out-of-order batch is filtered before the tally and must not count.
#[tokio::test]
async fn pump_tallies_applied_events_by_medium() {
    let id = worker_id("http://w1", 0);
    let h = spawn_pump(std::slice::from_ref(&id));
    let (tally, tx, pump) = (h.tally, h.tx, h.pump);

    let stored = |medium: Option<&str>, hashes: Vec<i64>| {
        KvCacheEvent::BlockStored(BlockStored {
            parent_block_hash: None,
            block_hashes: hashes,
            token_ids: vec![],
            block_size: 64,
            lora_id: None,
            medium: medium.map(str::to_owned),
        })
    };
    tx.send(WorkerEvent::Batch {
        worker: id.clone(),
        seq: 2,
        batch: batch(vec![
            stored(Some("GPU"), vec![10, 20, 30]),
            stored(Some("CPU_PINNED"), vec![10, 20, 30]),
            KvCacheEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![30],
                medium: Some("GPU".into()),
            }),
        ]),
    })
    .await
    .unwrap();
    // Out of order: filtered, must not be tallied.
    tx.send(WorkerEvent::Batch {
        worker: id.clone(),
        seq: 1,
        batch: batch(vec![stored(None, vec![99])]),
    })
    .await
    .unwrap();
    // A gap: seq 3 and 4 were dropped in transit. Counted, because a
    // tagged removal now clears only its own tier, so a lost batch can
    // strand a tier the tree will never clear on its own.
    tx.send(WorkerEvent::Batch {
        worker: id.clone(),
        seq: 5,
        batch: batch(vec![KvCacheEvent::AllBlocksCleared]),
    })
    .await
    .unwrap();
    drop(tx);
    pump.await.unwrap();

    assert_eq!(tally.batches_lost(), 2, "seq 3 and 4 never arrived");
    let rows = tally.snapshot();
    let cell = |event: &str, medium: &str| {
        rows.iter()
            .find(|r| r.event == event && r.medium == medium)
            .cloned()
            .expect("cell rendered")
    };
    assert_eq!(cell("block_stored", "GPU").blocks, 3);
    assert_eq!(cell("block_stored", "CPU_PINNED").blocks, 3);
    assert_eq!(cell("block_removed", "GPU").events, 1);
    assert_eq!(
        cell("block_stored", "untagged").events,
        0,
        "the out-of-order batch was filtered before the tally",
    );
    assert_eq!(
        cell("all_blocks_cleared", "untagged").events,
        1,
        "a clear carries no medium and lands on the untagged row",
    );
}

/// A `WorkerEvent::Load` lands in the engine-load table (gauge, no
/// cursor) keyed by the worker URL, and does not touch the tree.
#[tokio::test]
async fn pump_applies_load_to_engine_load_table() {
    let id = worker_id("http://w1", 0);
    let h = spawn_pump(std::slice::from_ref(&id));
    let (tree, engine_reported_load, tx, pump) = (h.tree, h.engine_reported_load, h.tx, h.pump);

    tx.send(WorkerEvent::Load {
        worker: id.clone(),
        load: LoadStat {
            num_running_reqs: 8,
            num_waiting_reqs: 4,
            num_tokens: 0,
            max_total_num_tokens: 0,
            native_cache: None,
        },
    })
    .await
    .unwrap();
    drop(tx);
    pump.await.unwrap();

    let snapshot = engine_reported_load.capture_snapshot(Instant::now());
    let load = snapshot.fresh_load_for_url("http://w1").unwrap();
    assert_eq!(load.num_running_reqs + load.num_waiting_reqs, 12);
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
        load_topic: None,
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
        load_topic: None,
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
    // dp_size=0 short-circuits the subscriber spawn but still runs the seed.
    let cfg = EventConfig {
        host: "127.0.0.1".into(),
        port_base: 30300,
        topic: String::new(),
        load_port_base: None,
        load_topic: None,
        block_size: 64,
        dp_size: 0,
        is_bigram: true,
    };
    index.add_worker("http://127.0.0.1:30300", Some(cfg)).await;
    assert!(
        index.block_size_oracle().is_bigram(),
        "add_worker must seed the bigram flag from EventConfig"
    );
    index.shutdown().await;
}

#[tokio::test]
async fn metadata_only_mode_keeps_the_load_subscriber() {
    let oracle = BlockSizeOracle::new();
    let index = KvEventIndex::new_metadata_only_with_http_and_oracle(
        reqwest::Client::new(),
        Arc::clone(&oracle),
    );
    let cfg = EventConfig {
        host: "127.0.0.1".into(),
        port_base: 30400,
        topic: "kv-events".into(),
        load_port_base: Some(30410),
        load_topic: Some("load".into()),
        block_size: 64,
        dp_size: 2,
        is_bigram: true,
    };

    index.add_worker("http://127.0.0.1:30400", Some(cfg)).await;

    assert_eq!(oracle.get(), Some(64));
    assert!(oracle.is_bigram());
    assert_eq!(index.known_worker_count(), 1);
    assert_eq!(index.engine_reported_load().expected_count(), 1);
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
        load_topic: Some("load".into()),
        block_size: 64,
        dp_size: 1,
        is_bigram: false,
    };
    index.add_worker(url, Some(cfg)).await;
    assert_eq!(index.engine_reported_load().expected_count(), 1);

    let now = Instant::now();
    index.engine_reported_load().set(
        url,
        0,
        LoadStat {
            num_running_reqs: 3,
            num_waiting_reqs: 1,
            num_tokens: 0,
            max_total_num_tokens: 0,
            native_cache: None,
        },
        now,
    );
    assert!(index
        .engine_reported_load()
        .capture_snapshot(now)
        .fresh_load_for_url(url)
        .is_some());

    index.remove_worker(url).await;
    assert!(
        index
            .engine_reported_load()
            .capture_snapshot(Instant::now())
            .fresh_load_for_url(url)
            .is_none(),
        "remove_worker must clear engine load"
    );
    assert_eq!(index.engine_reported_load().expected_count(), 0);
    index.shutdown().await;
}

/// `remove_worker` returns only after every rank's tree state is cleared
/// (the pump runs the teardown and signals `done`).
#[tokio::test]
async fn remove_worker_clears_every_rank_before_returning() {
    let index = KvEventIndex::new();
    let url = "http://127.0.0.1:59124";
    let cfg = EventConfig {
        host: "127.0.0.1".into(),
        port_base: 59124,
        topic: String::new(),
        load_port_base: None,
        load_topic: None,
        block_size: 64,
        dp_size: 2,
        is_bigram: false,
    };
    index.add_worker(url, Some(cfg)).await;

    // Stand in for events the pump already applied for both ranks.
    let tree = index.tree();
    let (r0, r1) = (worker_id(url, 0), worker_id(url, 1));
    tree.insert(&r0, None, &[10, 20]);
    tree.insert(&r1, None, &[30, 40]);
    assert!(tree.match_prefix(None, &[10, 20]).holds(&r0));
    assert!(tree.match_prefix(None, &[30, 40]).holds(&r1));

    // The pump is live, so the teardown below runs on it rather than on
    // the inline fallback.
    assert!(
        index.pump.lock().as_ref().is_some_and(|h| !h.is_finished()),
        "pump task must still be live, or the inline fallback does the teardown",
    );

    index.remove_worker(url).await;

    assert_eq!(
        tree.match_prefix(None, &[10, 20]).matched_blocks,
        0,
        "remove_worker must clear the removed worker's tree state",
    );
    assert_eq!(
        tree.match_prefix(None, &[30, 40]).matched_blocks,
        0,
        "every dp rank of the removed worker must be cleared",
    );
    assert_eq!(tree.node_count(), 0, "cleared chains must prune");
    index.shutdown().await;
}
