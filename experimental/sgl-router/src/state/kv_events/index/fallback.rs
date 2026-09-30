//! Every way a rank leaves bootstrap without keeping its graft: fail, discard,
//! or resolve from its stream's origin with no graft needed.

use std::collections::{HashMap, VecDeque};

use tracing::info;

use super::{apply_batch, PumpState};
use crate::state::kv_events::bootstrap::{BootstrapState, RankOutcome};
use crate::state::kv_events::tree::KvWorkerId;
use crate::state::kv_events::wire::KvEventBatch;

/// Give up on bootstrapping `rank`: release whatever it held and mark it
/// [`BootstrapState::Failed`].
///
/// `discard_held` is for a queue whose contents describe a cache that no
/// longer exists (a publisher reset): drop it and resume from the live stream.
/// Otherwise the intact queue is replayed as live deltas.
pub(super) fn fail_rank(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
    discard_held: bool,
    outcome: RankOutcome,
) {
    // Only transition ranks that are actually mid-bootstrap; an
    // AbandonBootstrap racing a successful ApplySnapshot must not undo it.
    //
    // This gate is also what keeps the rank tally once-per-rank: a second
    // attempt to fail an already-resolved rank returns before recording.
    if st.bootstrap.state_of(rank) != Some(BootstrapState::Pending) {
        held.remove(rank);
        return;
    }
    discard_graft(st, rank, outcome);
    let queue = held.remove(rank).unwrap_or_default();
    if discard_held {
        return;
    }
    for (seq, batch) in queue {
        apply_batch(st.tree, st.cursors, st.tally, rank, seq, &batch);
    }
}

/// Mark `rank` [`BootstrapState::Failed`], tally `outcome`, and drop
/// everything a snapshot contributed for it: tree carriers and cursor.
pub(super) fn discard_graft(st: &PumpState<'_>, rank: &KvWorkerId, outcome: RankOutcome) {
    st.bootstrap.set(rank, BootstrapState::Failed);
    st.bootstrap.record_rank_outcome(outcome);
    st.tree.clear_worker(rank);
    st.cursors.lock().remove(rank);
}

/// Resolve a `Pending` rank whose first held batch is its publisher's first
/// batch ever (`STREAM_ORIGIN_SEQ`), without a snapshot.
///
/// # Why this is safe
///
/// A peer snapshot exists to supply blocks the engine stored BEFORE this
/// replica subscribed. A stream received from its origin has no such blocks:
/// the publisher is born with the scheduler, next to an empty radix cache, so
/// every block the engine holds was announced by a batch this rank is about to
/// apply. The resulting tree is exactly what a replica subscribed from engine
/// start holds, and is exposed to the same ZMQ drop risk as any live stream,
/// no more. Waiting instead is pure cost: every sibling that also watched the
/// engine start is itself holding this rank's batches, so no snapshot anywhere
/// covers it, and the rank burns the whole bootstrap deadline to learn nothing.
///
/// Clears the rank's held queue, tree state and cursor first. A rank on its
/// first incarnation has none of them; one whose batch 0 arrives behind held
/// batches or a cursor is watching an engine that restarted without a graceful
/// `END_SEQ`, so the old state describes a cache that no longer exists and its
/// cursor would filter the new stream.
///
/// Leaves the caller to apply the triggering batch. A snapshot that lands
/// later is discarded by `apply_snapshot`'s `Pending` gate.
pub(super) fn resolve_from_origin(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
) {
    held.remove(rank);
    st.tree.clear_worker(rank);
    st.cursors.lock().remove(rank);
    st.bootstrap.set(rank, BootstrapState::Recovered);
    st.bootstrap.record_rank_outcome(RankOutcome::FromOrigin);
    info!(
        worker = ?rank,
        "kv-bootstrap: rank's event stream starts at its publisher's origin; \
         its history is complete without a peer snapshot",
    );
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::*;

    /// A hole between the snapshot watermark and the live stream means a delta
    /// was lost. Grafting anyway could leave a permanently stale entry, so the
    /// rank drops to cold: snapshot state cleared, live deltas still applied.
    ///
    /// This is the *deferred* path — the snapshot is grafted before any batch
    /// for the rank has arrived, so continuity can only be judged later. The
    /// pump's biased select drains the control channel first, which makes this
    /// the ordering that occurs naturally.
    #[tokio::test]
    async fn pump_detects_deferred_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Watermark 5, but the first live batch is seq 9 — 6..8 were lost.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // A gapped rank is terminal.
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        // The live stream still applies: cold, not broken.
        assert!(h.tree.match_prefix(None, &[42]).workers().contains(&id));
        // Every rank is terminal, so readiness no longer waits on this one.
        assert!(tracker.settled());
    }

    /// Same gap, detected on the *immediate* path: the batch is already held
    /// when the snapshot arrives, so the watermark can be checked at graft time.
    #[tokio::test]
    async fn pump_detects_held_queue_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        // Let the pump receive and hold the batch before the snapshot lands.
        // The pump's biased select would otherwise take the control message
        // first, which is the deferred path covered by the test above.
        tokio::time::sleep(Duration::from_millis(50)).await;

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        assert!(h.tree.match_prefix(None, &[42]).workers().contains(&id));
    }

    /// No peer could supply a snapshot: held deltas must still be released, or
    /// the rank would buffer forever and never settle.
    #[tokio::test]
    async fn pump_abandon_bootstrap_releases_held_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: batch(vec![stored(None, vec![55, 66])]),
        })
        .await
        .unwrap();
        // Held first, so it is the abandon's replay that releases it.
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        let m = h.tree.match_prefix(None, &[55, 66]);
        assert_eq!(
            m.matched_blocks, 2,
            "held batches must be released, not lost"
        );
        assert!(m.workers().contains(&id));
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(3));
        assert!(tracker.settled());
    }

    /// Overflowing the hold-back queue abandons bootstrap, and the rank resumes
    /// live WITHOUT losing what it held: the queue reached the cap intact, so
    /// discarding it would drop every block stored while waiting — blocks the
    /// engine never re-announces.
    #[tokio::test]
    async fn pump_overflowing_held_queue_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        for seq in 1..=(PENDING_BATCH_LIMIT as i64 + 2) {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        // The held prefix is replayed, then the batch that tripped the limit
        // and everything after it apply directly.
        let last = PENDING_BATCH_LIMIT as i64 + 2;
        for seq in [1, PENDING_BATCH_LIMIT as i64, last] {
            assert!(
                h.tree.match_prefix(None, &[seq]).workers().contains(&id),
                "batch {seq} must be applied across the overflow",
            );
        }
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(last));
        assert_eq!(
            h.tally.batches_lost(),
            0,
            "the replay is contiguous, so no sequence gap may be recorded",
        );
        assert!(tracker.settled());
    }

    /// A reset while still Pending must bail the rank to cold.
    ///
    /// Post-reset seq 1 can never exceed the peer's watermark, so the gap check
    /// passes trivially, the stale tree is kept, and every real delta is filtered.
    #[tokio::test]
    async fn pump_publisher_reset_while_pending_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 900,
            batch: batch(vec![stored(None, vec![33])]),
        })
        .await
        .unwrap();
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Failed),
            "a reset mid-bootstrap must abandon, not wait for a snapshot it can no \
             longer splice",
        );
        assert!(tracker.settled());
    }

    /// A terminal rank is immutable. An Abandon arriving after a successful
    /// graft must not wipe the rank's tree and demote it to Failed.
    #[tokio::test]
    async fn pump_abandon_after_successful_graft_does_not_undo_it() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let obs = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs.clone(),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        // A second bootstrap task for the same obligation set gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: obs })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "a late Abandon must not demote an already-Recovered rank",
        );
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a late Abandon must not wipe a grafted tree",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(5));
    }

    /// The stale-incarnation guard on the ABANDON arm, mirroring the one
    /// already covered on the ApplySnapshot arm.
    #[tokio::test]
    async fn pump_abandon_from_a_stale_incarnation_is_ignored() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let stale = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Remove + re-add: a new incarnation, Pending again.
        tracker.forget(std::slice::from_ref(&id));
        let fresh = tracker.register(std::slice::from_ref(&id));
        assert_ne!(stale[0].1, fresh[0].1);

        // The previous incarnation's task gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: stale })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "a stale Abandon must not force the new incarnation cold",
        );
    }

    /// The fresh-engine incident: a rank whose first batch is its publisher's
    /// first batch ever has nothing a sibling could supply, so it must resolve
    /// on that batch instead of holding for a snapshot no peer will ever have.
    /// A snapshot that lands afterwards must not be grafted over the complete
    /// stream.
    #[tokio::test]
    async fn pump_resolves_a_rank_whose_stream_starts_at_the_origin() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        for (seq, parent, hashes) in [(0, None, vec![10, 20]), (1, Some(20), vec![30])] {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(parent, hashes)]),
            })
            .await
            .unwrap();
        }
        // Wait for the pump to reach the batches before the control message:
        // the pump polls control first, so sending both at once would race.
        await_cursor(&h, &id, 1).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(tracker.settled(), "the rank no longer holds readiness");
        let m = h.tree.match_prefix(None, &[10, 20, 30]);
        assert_eq!(m.matched_blocks, 3, "both batches applied in order");
        assert!(m.workers().contains(&id));
        assert_eq!(
            h.tree.match_prefix(None, &[100, 200]).matched_blocks,
            0,
            "a late snapshot must not be grafted under a complete stream",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(1));
        assert_eq!(rank_count(&tracker, "from_origin"), 1);
    }

    /// The conservative side of the origin rule: a first batch past the origin
    /// means batch 0 was missed, and that batch may have stored blocks, so the
    /// history is not provably complete and the rank keeps holding for a
    /// snapshot. Batches that keep rising behind it are ordinary held batches.
    #[tokio::test]
    async fn pump_keeps_holding_unless_the_hold_begins_at_the_origin() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        for seq in [1, 2, 3] {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq + 10])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(h.cursors.lock().is_empty(), "nothing applied");
        assert_eq!(rank_count(&tracker, "from_origin"), 0);
    }

    /// Wait until the pump has applied `id` through `until`, so a control
    /// message sent next cannot overtake the batches (the pump polls control
    /// first).
    async fn await_cursor(h: &PumpHarness, id: &KvWorkerId, until: i64) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while h.cursors.lock().get(id).copied() != Some(until) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the batches are applied, not held");
    }

    /// Send `seqs` for `id` (each storing block `seq + 1000`), then wait until
    /// the pump has applied through `until`.
    async fn send_and_await_cursor(h: &PumpHarness, id: &KvWorkerId, seqs: &[i64], until: i64) {
        for &seq in seqs {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq + 1000])]),
            })
            .await
            .unwrap();
        }
        await_cursor(h, id, until).await;
    }

    /// An engine restarted in place (same URL, no `END_SEQ`) while its rank is
    /// Pending: the new stream's batch 0 lands behind the dead stream's held
    /// batches. Holding on would let a later graft seed an old-numbering
    /// watermark that filters the whole new stream — dead state reported warm,
    /// live updates dropped. The regression must discard the dead prefix and
    /// resolve from the new stream's origin, so a snapshot arriving afterwards
    /// is moot.
    #[tokio::test]
    async fn pump_restart_while_holding_resolves_from_the_new_origin() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        send_and_await_cursor(&h, &id, &[40, 41, 0, 1], 1).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 50)),
            })
            .await
            .unwrap();
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 2,
            batch: batch(vec![stored(None, vec![1002])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        for dead in [1040, 1041, 100, 200] {
            assert_eq!(
                h.tree.match_prefix(None, &[dead]).matched_blocks,
                0,
                "block {dead} is from the dead stream or the moot snapshot",
            );
        }
        for live in [1000, 1001, 1002] {
            assert!(
                h.tree.match_prefix(None, &[live]).workers().contains(&id),
                "new-stream block {live} must be applied, not filtered",
            );
        }
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(2));
        assert_eq!(rank_count(&tracker, "from_origin"), 1);
    }

    /// A regression that does not land on batch 0 is still a restart, but the
    /// new stream's head was missed too, so nothing can be spliced: discard the
    /// dead prefix and run cold from here.
    #[tokio::test]
    async fn pump_restart_past_the_origin_while_holding_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        send_and_await_cursor(&h, &id, &[40, 41, 3], 3).await;
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert_eq!(h.tree.match_prefix(None, &[1040]).matched_blocks, 0);
        assert_eq!(h.tree.match_prefix(None, &[1041]).matched_blocks, 0);
        assert!(h.tree.match_prefix(None, &[1003]).workers().contains(&id));
        assert_eq!(rank_count(&tracker, "publisher_reset"), 1);
    }

    /// With batches held, the held tail is the latest seq received, so it is
    /// what a regression is measured against. Measuring against the cursor
    /// alone would miss a step back that stays above it.
    #[tokio::test]
    async fn pump_regression_is_measured_against_the_held_tail_first() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());
        // A raw cursor below every held batch.
        h.cursors.lock().insert(id.clone(), 50);
        for seq in [55, 56, 52] {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq + 1000])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert_eq!(rank_count(&tracker, "publisher_reset"), 1);
        assert!(h.tree.match_prefix(None, &[1052]).workers().contains(&id));
        assert_eq!(h.tree.match_prefix(None, &[1055]).matched_blocks, 0);
    }

    /// With nothing held, the cursor is the latest seq received, so a
    /// regression below it is a restart too.
    #[tokio::test]
    async fn pump_regression_with_nothing_held_is_measured_against_the_cursor() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());
        h.tree
            .insert_tiered(&id, None, &[900], Tiers::for_store(None));
        h.cursors.lock().insert(id.clone(), 50);
        send_and_await_cursor(&h, &id, &[3], 3).await;
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert_eq!(rank_count(&tracker, "publisher_reset"), 1);
        assert_eq!(h.tree.match_prefix(None, &[900]).matched_blocks, 0);
        assert!(h.tree.match_prefix(None, &[1003]).workers().contains(&id));
    }

    /// The resolved-rank path: a grafted, proven rank whose engine restarts in
    /// place sees batch 0 behind a cursor seeded in the old numbering. The gap
    /// check cannot flag a backwards step, and the cursor would filter the new
    /// stream until it overtook the old one — so batch 0 replaces the state.
    #[tokio::test]
    async fn pump_batch_zero_behind_a_resolved_cursor_replaces_the_rank_state() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 6,
            batch: batch(vec![stored(Some(200), vec![300])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        await_cursor(&h, &id, 6).await;

        send_and_await_cursor(&h, &id, &[0, 1], 1).await;
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            h.tree.match_prefix(None, &[100]).matched_blocks,
            0,
            "the restarted engine's cache no longer holds the grafted state",
        );
        for live in [1000, 1001] {
            assert!(
                h.tree.match_prefix(None, &[live]).workers().contains(&id),
                "new-stream block {live} must be applied, not filtered",
            );
        }
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(1));
        assert_eq!(
            rank_count(&tracker, "warm"),
            1,
            "the graft's own verdict stands"
        );
        assert_eq!(
            rank_count(&tracker, "from_origin"),
            0,
            "counted once, as warm"
        );
    }

    /// The boundary: a rank resolved from its origin that has applied ONLY
    /// batch 0 (cursor 0) restarts in place. Treating the second batch 0 as a
    /// duplicate would keep the dead batch's state and drop the new one.
    #[tokio::test]
    async fn pump_second_batch_zero_at_cursor_zero_replaces_the_rank_state() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());
        send_and_await_cursor(&h, &id, &[0], 0).await;
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 0,
            batch: batch(vec![stored(None, vec![7777])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(h.tree.match_prefix(None, &[1000]).matched_blocks, 0);
        assert!(h.tree.match_prefix(None, &[7777]).workers().contains(&id));
    }

    /// Same restart while the graft's splice is still unproven: the verdict is
    /// final at that batch — the grafted state is gone and the new stream from
    /// its origin replaces it — so it is tallied exactly once, not warm.
    #[tokio::test]
    async fn pump_batch_zero_behind_an_unproven_graft_settles_its_verdict() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        send_and_await_cursor(&h, &id, &[0], 0).await;
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(h.tree.match_prefix(None, &[100]).matched_blocks, 0);
        assert!(h.tree.match_prefix(None, &[1000]).workers().contains(&id));
        assert_eq!(rank_count(&tracker, "warm"), 0);
        assert_eq!(rank_count(&tracker, "from_origin"), 1);
    }

    /// A Pending rank carrying tree state and a cursor from an earlier stream
    /// whose next batch is 0: that engine restarted, so the old state describes
    /// a cache that is gone — and its cursor would filter every batch of the
    /// new stream.
    #[tokio::test]
    async fn pump_origin_resolution_discards_an_earlier_streams_state() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());
        h.tree
            .insert_tiered(&id, None, &[900], Tiers::for_store(None));
        h.cursors.lock().insert(id.clone(), 50);

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 0,
            batch: batch(vec![stored(None, vec![10])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert_eq!(
            h.tree.match_prefix(None, &[900]).matched_blocks,
            0,
            "state from the dead stream must go",
        );
        assert_eq!(h.tree.match_prefix(None, &[10]).matched_blocks, 1);
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(0));
    }
}
