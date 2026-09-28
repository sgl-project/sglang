//! Run sweeps single-flight and fold the ranks discovered meanwhile into them.

use std::collections::HashSet;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::sweep::{deliver_bootstrap, still_pending, sweep_recording_settled, SweepResult};
use super::{KvEventIndex, LateJoin, ObligationBatch};
use crate::state::kv_events::bootstrap::{BootstrapTracker, SweepOutcome};
use crate::state::kv_events::tree::KvWorkerId;

/// Obligations taken off the queue for one sweep, with the freshness every one
/// of them requires.
pub(super) struct PendingSweep {
    obligations: Vec<(KvWorkerId, u64)>,
    /// The newest `holding_since` among batches absorbed before the sweep: its
    /// one snapshot must postdate every rank in it. An `Instant` rather than a
    /// max age, so each retry's `elapsed()` asks for the same condition
    /// (`exported_at > floor`).
    freshness_floor: Instant,
    /// The newest `holding_since` of every batch folded in, including those
    /// admitted after the floor froze. What a batch built from this sweep's
    /// ranks must carry, since it may hold ranks that began holding after the
    /// floor.
    newest_holding: Instant,
    /// Batches that arrived too late for this sweep to have asked on their
    /// behalf and refused to ride along. Handed back for the next one.
    deferred: Vec<ObligationBatch>,
}

impl PendingSweep {
    /// Fold a batch in before the sweep starts, tightening the floor it will ask
    /// with.
    fn absorb(&mut self, batch: ObligationBatch) {
        self.freshness_floor = self.freshness_floor.max(batch.holding_since);
        self.fold(batch);
    }

    /// Add `batch`'s obligations without touching the floor.
    fn fold(&mut self, batch: ObligationBatch) {
        self.newest_holding = self.newest_holding.max(batch.holding_since);
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
            self.fold(batch);
        }
    }

    /// Keep only obligations a graft could still discharge, once each.
    ///
    /// A queued obligation can go stale before its sweep runs — the rank
    /// resolved, or its worker was removed and re-added — and would buy a
    /// fleet-wide fetch the pump then discards at its `Pending` gate. And the
    /// same rank can arrive twice (a coverage retry merged into a sweep that
    /// already holds it); a duplicate would be delivered alongside that rank's
    /// own retry and resolve it `Uncovered` before the retry lands.
    fn retain_graftable(&mut self, tracker: &BootstrapTracker) {
        // Keyed by epoch alone: the tracker mints epochs from one counter, and
        // the `epoch_of` check ties each surviving epoch to exactly one rank.
        let mut seen: HashSet<u64> = HashSet::with_capacity(self.obligations.len());
        self.obligations
            .retain(|(rank, epoch)| still_pending(tracker, rank, *epoch) && seen.insert(*epoch));
    }
}

impl From<ObligationBatch> for PendingSweep {
    fn from(batch: ObligationBatch) -> Self {
        Self {
            obligations: batch.obligations,
            freshness_floor: batch.holding_since,
            newest_holding: batch.holding_since,
            deferred: Vec::new(),
        }
    }
}

/// Run bootstrap sweeps one at a time, and let every rank queued while a sweep
/// is in flight share its snapshot.
///
/// A peer snapshot is fleet-wide — one body carries the whole worker table,
/// every cursor and the whole tree — so one fetch serves every pending rank,
/// where a sweep per discovered engine would re-download it once per engine.
/// The fetch itself is the batching window: it outlasts the watch event that
/// delivers a fleet, so ranks discovered meanwhile join its delivery at no
/// latency cost.
///
/// Sharing is safe because a graft is judged by sequence, not by clock: it
/// splices only when the rank's live stream resumes at `peer_cursor + 1`, and
/// each rank is judged on its own, so one whose publisher advanced resolves
/// [`RankOutcome::Gap`]. A rank the accepted peer does not cover gets one sweep
/// of its own before it resolves [`RankOutcome::Uncovered`].
///
/// Every fetch attempt states [`PendingSweep::freshness_floor`], which speaks
/// for the ranks the sweep started with. A rank merged in later may be
/// delivered against an older export; [`LateJoin`] says which batches accept
/// that.
///
/// [`RankOutcome::Gap`]: crate::state::kv_events::bootstrap::RankOutcome::Gap
/// [`RankOutcome::Uncovered`]: crate::state::kv_events::bootstrap::RankOutcome::Uncovered
pub(super) async fn bootstrap_coordinator(
    mut rx: mpsc::Receiver<ObligationBatch>,
    index: std::sync::Weak<KvEventIndex>,
    cancel: CancellationToken,
) {
    // Obligations already given a second sweep for want of coverage. A rank no
    // peer covers must not buy a fresh fleet-wide fetch on every pass, so each
    // gets at most one retry. Keyed by incarnation, so a worker removed and re-added is
    // not denied its own, and pruned once that incarnation is gone.
    let mut requeued: HashSet<(KvWorkerId, u64)> = HashSet::new();
    loop {
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
        requeued.retain(|(rank, epoch)| deps.bootstrap.epoch_of(rank) == Some(*epoch));
        pending.retain_graftable(&deps.bootstrap);
        if pending.obligations.is_empty() {
            // Every rank left `Pending` before its sweep could start — most
            // often resolved from its stream's origin by a fresh engine's
            // first batch. The seed gate reads "no sweep verdict yet" as a boot
            // sweep still in flight, so with no rank left waiting on any sweep
            // this records the verdict a sweep started a moment later would
            // reach before its first fetch; otherwise `/readyz` would hold for
            // a sweep that never runs. While any rank is still `Pending`, the
            // sweep that owns it reports instead, and an early verdict here
            // would open the gate ahead of it.
            if !deps.bootstrap.any_pending() {
                deps.bootstrap
                    .record_sweep_result(SweepOutcome::RanksResolved, 0);
            }
            continue;
        }
        let deadline = deps.deadline();
        let mut settled = Vec::new();

        // ONE sweep, awaited here rather than spawned — that is what makes this
        // single-flight. Obligations discovered while it runs pile up in the
        // channel and are merged below, so they ride this same snapshot.
        let started = Instant::now();
        info!(
            ranks = pending.obligations.len(),
            peers = deps.peers.len(),
            deadline_ms = deadline.as_millis(),
            "kv-bootstrap: sweeping sibling replicas for a tree snapshot",
        );
        // Raced against shutdown, which cancels the pump through the same token:
        // a sweep finishing after it has nobody to deliver to, and would keep
        // pulling multi-megabyte bodies from siblings for up to the deadline.
        let result = tokio::select! {
            _ = cancel.cancelled() => return,
            result = sweep_recording_settled(
                &deps,
                &pending.obligations,
                deadline,
                pending.freshness_floor,
                &mut settled,
            ) => result,
        };
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
        // Late arrivals can repeat a rank already here, and anything can have
        // resolved or been removed while the sweep ran. A rank the sweep settled
        // cold is owed nothing either, but can still read `Pending` until the
        // pump drains that release, so it is dropped by obligation, not by
        // state — otherwise it would buy a coverage retry, or a sweep of its own.
        pending.obligations.retain(|ob| !settled.contains(ob));
        pending.retain_graftable(&deps.bootstrap);

        // The sweep's coverage check (`covers_any`) only spoke for the ranks it
        // was given, so a rank merged afterwards may be absent from the accepted
        // peer's tree — the peer can itself be partially discovered and know one
        // of our workers but not another. Delivering such a rank into this
        // snapshot resolves it `Uncovered`, which is TERMINAL, where a sweep that
        // asked on its behalf would have kept looking for a peer that did cover
        // it. Give it that look instead.
        if let SweepResult::Found(ref vetted) = result {
            let covered = vetted.covered_ranks();
            let (deliver, retry): (Vec<_>, Vec<_>) = std::mem::take(&mut pending.obligations)
                .into_iter()
                .partition(|ob| covered.contains(&ob.0) || !requeued.insert(ob.clone()));
            pending.obligations = deliver;
            if !retry.is_empty() {
                // The newest instant anything here began holding, not the frozen
                // floor: a rank merged in after the sweep started began holding
                // after the floor, and asking only for the floor would let its
                // retry accept an export that predates it. Still not NOW, which
                // would force the peer into rebuilds these ranks do not need.
                pending.deferred.push(ObligationBatch {
                    obligations: retry,
                    holding_since: pending.newest_holding,
                    late_join: LateJoin::Permitted,
                });
            }
        }

        // A sweep that ends `RanksResolved` owes its own ranks nothing and sends
        // nothing: every one left `Pending` on its incarnation, or was settled and
        // dropped above. What remains was merged in after the sweep started and
        // was never swept. Folded into that delivery it would stay `Pending` with
        // nothing left to release it, so it gets a sweep of its own, asking for
        // the newest instant it began holding, as a coverage retry does.
        if matches!(result, SweepResult::RanksResolved) && !pending.obligations.is_empty() {
            pending.deferred.push(ObligationBatch {
                obligations: std::mem::take(&mut pending.obligations),
                holding_since: pending.newest_holding,
                late_join: LateJoin::Permitted,
            });
        }

        // Re-queue via the index rather than a sender this task owns: a
        // long-lived clone here would hold the channel open forever and kill
        // the all-senders-dropped exit.
        let live = index.upgrade();
        for batch in std::mem::take(&mut pending.deferred) {
            match &live {
                // Never folded into this delivery: that would spend the
                // freshness these ranks asked for.
                Some(idx) => idx.enqueue_bootstrap(batch),
                // Teardown: nothing is left to run another sweep, so deliver them
                // here rather than stranding them.
                None => pending.obligations.extend(batch.obligations),
            }
        }
        // Not held across the delivery, which would keep the index alive.
        drop(live);

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

impl KvEventIndex {
    /// Hand `batch` to the coordinator, or sweep it on its own when the queue is
    /// full or the coordinator is gone.
    ///
    /// Never drops it: an obligation nothing owns leaves its ranks `Pending`,
    /// which holds `/readyz` and holds their batches until the pump's per-rank
    /// cap overflows. Sweeping un-batched costs a fetch the coordinator would
    /// have shared, and keeps the freshness the batch asked for.
    pub(super) fn enqueue_bootstrap(&self, batch: ObligationBatch) {
        if let Err(e) = self.bootstrap_tx.try_send(batch) {
            warn!("kv-bootstrap: batch queue unavailable ({e}); sweeping un-batched");
            self.spawn_bootstrap(e.into_inner());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::test_support::*;
    use super::*;
    use crate::state::kv_events::bootstrap::BootstrapState;

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

    /// It returns as soon as one obligation is queued, without debouncing.
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

    /// A gap retry landing mid-sweep must not be spent on that sweep's snapshot.
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

    /// A rank merged in mid-sweep began holding after the frozen floor, so
    /// anything re-queued from this sweep must ask for at least its instant.
    /// The floor itself stays put: the request has already gone out under it.
    #[tokio::test]
    async fn late_admissions_raise_newest_holding_but_not_the_floor() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        tx.send(discovery_batch(0, base)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");

        let late = base + Duration::from_secs(2);
        tx.send(discovery_batch(1, late)).await.unwrap();
        drain_ready(&mut rx, &mut pending);
        assert_eq!(
            pending.freshness_floor, base,
            "the floor is frozen once the sweep is in flight",
        );
        assert_eq!(pending.newest_holding, late);
    }

    /// Only obligations a graft could still discharge reach a sweep: a rank that
    /// already resolved, or whose incarnation is gone, would cost a fleet-wide
    /// fetch for nothing, and a duplicate would defeat its own coverage retry.
    #[test]
    fn retain_graftable_keeps_each_pending_incarnation_once() {
        let pending_rank = worker_id("http://pending:30000", 0);
        let resolved = worker_id("http://resolved:30000", 0);
        let reregistered = worker_id("http://reregistered:30000", 0);
        let ranks = [pending_rank.clone(), resolved.clone(), reregistered.clone()];
        let tracker = pending_tracker(&ranks);
        let obs = obligations(&tracker, &ranks);
        tracker.set(&resolved, BootstrapState::Recovered);
        let stale_epoch = tracker.epoch_of(&reregistered).unwrap();
        tracker.forget(std::slice::from_ref(&reregistered));
        tracker.register(std::slice::from_ref(&reregistered));

        let mut obligations = obs.clone();
        obligations.push(obs[0].clone());
        let mut sweep = PendingSweep::from(ObligationBatch {
            obligations,
            holding_since: Instant::now(),
            late_join: LateJoin::Permitted,
        });
        sweep.retain_graftable(&tracker);

        assert_eq!(
            sweep.obligations,
            vec![obs[0].clone()],
            "resolved, superseded (epoch {stale_epoch}) and duplicate obligations are dropped",
        );
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

    /// A rank that joins a sweep whose own ranks all resolve without it — here
    /// from their stream's origin — must still get a sweep. `RanksResolved`
    /// sends no control message, so folded into that delivery the late joiner
    /// would sit `Pending`, holding its batches and `/readyz`, forever.
    #[tokio::test]
    async fn a_rank_that_joins_a_sweep_ending_ranks_resolved_gets_its_own() {
        use std::sync::Arc;

        use crate::state::kv_events::block_size_oracle::BlockSizeOracle;
        use crate::state::kv_events::bootstrap::{PeerSnapshot, WireWorker, SNAPSHOT_FORMAT};
        use crate::state::kv_events::tree::SnapshotNode;

        let (first, late, carrier) = (
            worker_id("http://w1:30000", 0),
            worker_id("http://w2:30000", 0),
            worker_id("http://w3:30000", 0),
        );
        // Warm, silent on `first` (so its sweep keeps looking), and naming
        // `late` as a rank it holds nothing for (so a sweep for it settles).
        let body = PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: false,
            producer_ready: true,
            workers: vec![WireWorker::from(&carrier)],
            cursors: vec![(0, 5)],
            nodes: vec![SnapshotNode {
                parent: None,
                block_hash: 111,
                workers: vec![0],
                tiers: vec![],
            }],
            empty_ranks: vec![WireWorker::from(&late)],
        };
        let (peer, _q) = serve_snapshot_sequence(vec![body]).await;

        let oracle = BlockSizeOracle::new();
        oracle.try_set(64).expect("first set establishes");
        oracle.set_bigram(false);
        let tracker = Arc::new(BootstrapTracker::new(Duration::from_secs(3600)));
        let index =
            KvEventIndex::new_with_bootstrap(reqwest::Client::new(), oracle, Arc::clone(&tracker));
        index.peers().replace(vec![peer]);
        for w in [&first, &late, &carrier] {
            index.live_workers.lock().insert(w.clone());
        }
        let ob = tracker.register(&[first.clone(), late.clone()]);
        let batch = |obligations| ObligationBatch {
            obligations,
            holding_since: Instant::now(),
            late_join: LateJoin::Permitted,
        };

        index.enqueue_bootstrap(batch(vec![ob[0].clone()]));
        // Let the coordinator start the sweep, then queue the late joiner.
        tokio::time::sleep(Duration::from_millis(100)).await;
        index.enqueue_bootstrap(batch(vec![ob[1].clone()]));
        tracker.set(&first, BootstrapState::Recovered);

        tokio::time::timeout(Duration::from_secs(10), async {
            while tracker.state_of(&late) == Some(BootstrapState::Pending) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the late joiner was swept and settled, not stranded Pending");
        assert_eq!(tracker.state_of(&late), Some(BootstrapState::Failed));
        index.shutdown().await;
    }

    /// A rank the sweep settles cold mid-sweep can still read `Pending` when the
    /// coordinator looks, because the pump has not drained that release yet. It
    /// was swept, so it is owed nothing: it must not be re-queued as though it
    /// had joined late, which would run (or at least tally) a second sweep.
    #[tokio::test]
    async fn a_rank_settled_cold_mid_sweep_is_not_swept_again() {
        use std::sync::Arc;

        use crate::state::kv_events::block_size_oracle::BlockSizeOracle;
        use crate::state::kv_events::bootstrap::{PeerSnapshot, WireWorker, SNAPSHOT_FORMAT};
        use crate::state::kv_events::tree::SnapshotNode;

        let (rank, carrier) = (
            worker_id("http://w1:30000", 0),
            worker_id("http://w2:30000", 0),
        );
        let body = PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: false,
            producer_ready: true,
            workers: vec![WireWorker::from(&carrier)],
            cursors: vec![(0, 5)],
            nodes: vec![SnapshotNode {
                parent: None,
                block_hash: 111,
                workers: vec![0],
                tiers: vec![],
            }],
            empty_ranks: vec![WireWorker::from(&rank)],
        };
        let (peer, queries) = serve_snapshot_sequence(vec![body]).await;

        let oracle = BlockSizeOracle::new();
        oracle.try_set(64).expect("first set establishes");
        oracle.set_bigram(false);
        let tracker = Arc::new(BootstrapTracker::new(Duration::from_secs(3600)));
        let index =
            KvEventIndex::new_with_bootstrap(reqwest::Client::new(), oracle, Arc::clone(&tracker));
        index.peers().replace(vec![peer]);
        for w in [&rank, &carrier] {
            index.live_workers.lock().insert(w.clone());
        }
        let obligations = tracker.register(std::slice::from_ref(&rank));
        index.enqueue_bootstrap(ObligationBatch {
            obligations,
            holding_since: Instant::now(),
            late_join: LateJoin::Permitted,
        });

        tokio::time::timeout(Duration::from_secs(10), async {
            while tracker.state_of(&rank) == Some(BootstrapState::Pending) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the rank settles cold");
        // Room for a wrongly re-queued batch to reach the coordinator.
        tokio::time::sleep(Duration::from_millis(300)).await;

        assert_eq!(tracker.state_of(&rank), Some(BootstrapState::Failed));
        assert!(
            tracker
                .sweep_result_counts()
                .contains(&("ranks_resolved", 1)),
            "one sweep, tallied once: {:?}",
            tracker.sweep_result_counts(),
        );
        assert_eq!(queries.lock().expect("queries lock").len(), 1, "one fetch");
        index.shutdown().await;
    }

    /// A router booting alongside fresh engines resolves every rank from its
    /// stream's origin, often before the coordinator takes the batch, so no
    /// sweep ever runs. Under `--kv-bootstrap-seed-required` the gate reads "no
    /// sweep verdict yet" as a boot sweep still in flight, so the skipped sweep
    /// must still leave its verdict, or `/readyz` holds until the gate's hard
    /// bound for a sweep that never runs.
    #[tokio::test]
    async fn a_batch_every_rank_left_before_its_sweep_still_records_a_verdict() {
        use std::sync::Arc;

        use crate::state::kv_events::block_size_oracle::BlockSizeOracle;

        let tracker = Arc::new(BootstrapTracker::new_with_opts(
            Duration::from_secs(3600),
            Duration::from_secs(60),
            true,
        ));
        let index = KvEventIndex::new_with_bootstrap(
            reqwest::Client::new(),
            BlockSizeOracle::new(),
            Arc::clone(&tracker),
        );
        let rank = worker_id("http://w1:30000", 0);
        let obligations = tracker.register(std::slice::from_ref(&rank));
        // What `resolve_from_origin` leaves, before the batch is taken.
        tracker.set(&rank, BootstrapState::Recovered);
        assert!(tracker.settled());
        assert!(
            !tracker.admit_ready(),
            "premise: the gate waits on a verdict"
        );

        index.enqueue_bootstrap(ObligationBatch {
            obligations,
            holding_since: Instant::now(),
            late_join: LateJoin::Permitted,
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while !tracker.admit_ready() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("a sweep with nothing left to do must not hold readiness");
        assert!(tracker
            .sweep_result_counts()
            .contains(&("ranks_resolved", 1)));
        index.shutdown().await;
    }

    /// The other side of that verdict: while some rank is still `Pending`, the
    /// sweep that owns it is the one that reports. A verdict recorded for an
    /// emptied batch would clear the gate's "no verdict yet" arm ahead of it,
    /// and a probe landing before that sweep's `TimedOut` could latch the
    /// replica ready without the seed `--kv-bootstrap-seed-required` asks for.
    #[tokio::test]
    async fn an_emptied_batch_leaves_the_verdict_to_a_rank_still_pending() {
        use std::sync::Arc;

        use crate::state::kv_events::block_size_oracle::BlockSizeOracle;

        let tracker = Arc::new(BootstrapTracker::new_with_opts(
            Duration::from_secs(3600),
            Duration::from_secs(60),
            true,
        ));
        let index = KvEventIndex::new_with_bootstrap(
            reqwest::Client::new(),
            BlockSizeOracle::new(),
            Arc::clone(&tracker),
        );
        let (resolved, waiting) = (
            worker_id("http://w1:30000", 0),
            worker_id("http://w2:30000", 0),
        );
        let obligations = tracker.register(&[resolved.clone(), waiting]);
        tracker.set(&resolved, BootstrapState::Recovered);

        index.enqueue_bootstrap(ObligationBatch {
            obligations: vec![obligations[0].clone()],
            holding_since: Instant::now(),
            late_join: LateJoin::Permitted,
        });
        // Long enough for the coordinator to take and drop the batch.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            tracker.sweep_result_counts().iter().all(|(_, n)| *n == 0),
            "no verdict while a rank still waits on its own sweep: {:?}",
            tracker.sweep_result_counts(),
        );
        index.shutdown().await;
    }
}
