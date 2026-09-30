//! Pump and peer test fixtures shared by `index::tests` and the child test modules.

use super::*;
use crate::state::kv_events::tree::SnapshotNode;
use crate::state::kv_events::wire::{BlockStored, KvEventBatch};

pub(super) fn worker_id(url: &str, rank: u32) -> KvWorkerId {
    KvWorkerId {
        url: url.into(),
        dp_rank: rank,
    }
}

pub(super) fn batch(events: Vec<KvCacheEvent>) -> KvEventBatch {
    KvEventBatch {
        ts: 0.0,
        events,
        attn_dp_rank: None,
    }
}

/// Bundle of plumbing returned by `spawn_pump` so individual tests
/// can destructure just the bits they need.
pub(super) struct PumpHarness {
    pub(super) tree: Arc<HashTree>,
    pub(super) engine_reported_load: Arc<EngineReportedLoadTable>,
    pub(super) cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    pub(super) tally: Arc<EventTally>,
    #[allow(dead_code)]
    pub(super) live_set: Arc<Mutex<HashSet<KvWorkerId>>>,
    #[allow(dead_code)]
    pub(super) cancel: CancellationToken,
    pub(super) tx: mpsc::Sender<WorkerEvent>,
    pub(super) pump: JoinHandle<()>,
    pub(super) ctrl_tx: mpsc::Sender<PumpControl>,
}

/// Build a tree + cursors + live-set wired through `pump_loop` with
/// the given workers pre-marked live. Bootstrap is pre-settled, so batches
/// are applied immediately — the behaviour every pre-bootstrap test asserts.
pub(super) fn spawn_pump(live: &[KvWorkerId]) -> PumpHarness {
    spawn_pump_with_bootstrap(live, Arc::new(BootstrapTracker::disabled()))
}

/// As `spawn_pump`, but with a caller-supplied tracker so bootstrap
/// hold-back, splicing, and abandonment can be driven directly.
pub(super) fn spawn_pump_with_bootstrap(
    live: &[KvWorkerId],
    bootstrap: Arc<BootstrapTracker>,
) -> PumpHarness {
    let tree = Arc::new(HashTree::new());
    let engine_reported_load = EngineReportedLoadTable::new();
    let cursors = Arc::new(Mutex::new(HashMap::new()));
    let live_set: Arc<Mutex<HashSet<KvWorkerId>>> =
        Arc::new(Mutex::new(live.iter().cloned().collect()));
    let cancel = CancellationToken::new();
    let (tx, rx) = mpsc::channel(4);
    let (ctrl_tx, ctrl_rx) = mpsc::channel(4);
    let tally = Arc::new(EventTally::new());
    let pump = tokio::spawn(pump_loop(
        PumpDeps {
            tally: Arc::clone(&tally),
            tree: tree.clone(),
            engine_reported_load: engine_reported_load.clone(),
            cursors: cursors.clone(),
            live_workers: live_set.clone(),
            bootstrap: bootstrap.clone(),
        },
        cancel.clone(),
        rx,
        ctrl_rx,
    ));
    PumpHarness {
        tree,
        engine_reported_load,
        cursors,
        tally,
        live_set,
        cancel,
        tx,
        pump,
        ctrl_tx,
    }
}

pub(super) fn stored(parent: Option<i64>, hashes: Vec<i64>) -> KvCacheEvent {
    KvCacheEvent::BlockStored(BlockStored {
        parent_block_hash: parent,
        block_hashes: hashes,
        token_ids: vec![],
        block_size: 64,
        lora_id: None,
        medium: None,
    })
}

/// The obligation set for `ranks` as registered in `tracker`, as
/// `add_worker` gets it from `register`.
pub(super) fn obligations(
    tracker: &BootstrapTracker,
    ranks: &[KvWorkerId],
) -> Vec<(KvWorkerId, u64)> {
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
pub(super) fn pending_tracker(ids: &[KvWorkerId]) -> Arc<BootstrapTracker> {
    let t = Arc::new(BootstrapTracker::new(Duration::from_secs(3600)));
    t.register(ids);
    t
}

/// A snapshot carrying chain [100, 200] for `id`, watermarked at `cursor`.
pub(super) fn vetted_for(id: &KvWorkerId, cursor: i64) -> VettedSnapshot {
    vetted_for_workers(&[id], cursor)
}

/// As `vetted_for`, but every node carries every listed worker — for the
/// tests that need one snapshot to span several ranks.
pub(super) fn vetted_for_workers(ids: &[&KvWorkerId], cursor: i64) -> VettedSnapshot {
    let carriers: Vec<u32> = (0..ids.len() as u32).collect();
    VettedSnapshot::from_parts_for_test(
        ids.iter().map(|id| (*id).clone()).collect(),
        vec![
            SnapshotNode {
                parent: None,
                block_hash: 100,
                workers: carriers.clone(),
                tiers: vec![],
            },
            SnapshotNode {
                parent: Some(0),
                block_hash: 200,
                workers: carriers,
                tiers: vec![],
            },
        ],
        ids.iter().map(|id| ((*id).clone(), cursor)).collect(),
        0,
    )
}

/// Count of one rank-outcome label, for the pump tests' exactly-once assertions.
pub(super) fn rank_count(tracker: &BootstrapTracker, label: &str) -> u64 {
    tracker
        .rank_outcome_counts()
        .into_iter()
        .find(|(l, _)| *l == label)
        .map(|(_, c)| c)
        .unwrap_or(0)
}

/// Graft a rank with NOTHING held, so its splice proof is deferred, and leave
/// it in that state. The shape the deferred-proof tests start from.
pub(super) async fn graft_with_deferred_proof(
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
