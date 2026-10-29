//! The proof state for an unproven splice, and the fleet probe that settles it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tracing::{debug, warn};

use super::PumpControl;
use crate::state::kv_events::bootstrap::{fetch_cursors, PeerRegistry, SNAPSHOT_FORMAT};
use crate::state::kv_events::tree::KvWorkerId;

/// How long a grafted rank may wait for its own live stream to prove the splice
/// before the fleet is asked instead.
///
/// Nothing guarantees a first live batch ever comes: an idle rank would
/// otherwise serve grafted state forever with its continuity unproven, and a
/// `BlockRemoved` lost in the subscribe window would survive as a permanent
/// false cache hit. On expiry the rank is probed against the fleet's cursors,
/// not discarded; see the proof sweep in `pump_loop`.
pub(super) const SPLICE_PROOF_TIMEOUT: Duration = Duration::from_secs(30);

/// Consecutive unanswerable probes after which an unproven graft is kept and
/// tallied [`RankOutcome::WarmUnwitnessed`].
///
/// Without a stop the pump would re-probe an idle rank forever whenever the
/// fleet is unreachable — a single-replica deployment being the obvious case —
/// and the rank's verdict would never resolve in the metrics. Keeping rather
/// than discarding follows the same reasoning as the probe itself.
///
/// [`RankOutcome::WarmUnwitnessed`]: crate::state::kv_events::bootstrap::RankOutcome::WarmUnwitnessed
pub(super) const MAX_UNKNOWN_PROBES: u32 = 3;

/// How often the pump checks for splice proofs that never arrived. Coarse: the
/// deadline it enforces is [`SPLICE_PROOF_TIMEOUT`], not this.
pub(super) const SPLICE_PROOF_SWEEP_INTERVAL: Duration = Duration::from_secs(5);

/// Upper bound on one probe pass across the fleet.
///
/// One sweep tick short of [`SPLICE_PROOF_TIMEOUT`], so a pass has always
/// reported before the sweep could launch the next one for the same ranks:
/// passes never overlap, and a slow or hung peer cannot stack up concurrent
/// passes that each re-ask every peer. A peer not reached inside the budget is
/// simply not a witness this pass, exactly as if it had been unreachable.
const SPLICE_PROBE_BUDGET: Duration =
    SPLICE_PROOF_TIMEOUT.saturating_sub(SPLICE_PROOF_SWEEP_INTERVAL);
const _: () = assert!(
    !SPLICE_PROBE_BUDGET.is_zero()
        && SPLICE_PROBE_BUDGET.as_nanos() + SPLICE_PROOF_SWEEP_INTERVAL.as_nanos()
            <= SPLICE_PROOF_TIMEOUT.as_nanos()
);

/// A grafted rank whose continuity with the live stream is not yet proven.
#[derive(Debug, Clone, Copy)]
pub(super) struct PendingProof {
    /// Watermark the first arriving batch must not exceed by more than one.
    ///
    /// Also what a probe verdict must name to act on this entry: a gap retry
    /// re-grafts under the SAME incarnation, so the epoch alone cannot tell a
    /// verdict about the previous graft from one about this one.
    pub(super) watermark: i64,
    /// When the graft happened, or when the last probe was launched.
    pub(super) since: Instant,
    /// Consecutive probes that found no witness; see [`MAX_UNKNOWN_PROBES`].
    pub(super) unknown_probes: u32,
}

impl PendingProof {
    pub(super) fn new(watermark: i64, now: Instant) -> Self {
        Self {
            watermark,
            since: now,
            unknown_probes: 0,
        }
    }

    /// Whether the sweep should probe this rank now: once the graft, or the
    /// last probe, is older than `timeout`. A verdict that never lands (a
    /// panicked probe task, say) is therefore simply asked again.
    pub(super) fn due_for_probe(&self, timeout: Duration) -> bool {
        self.since.elapsed() >= timeout
    }
}

/// What the fleet says about a publisher's progress past an unproven watermark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpliceVerdict {
    /// Some peer has applied a sequence ABOVE our watermark. Sequence numbers
    /// come from the publisher, so that batch was emitted — and we never saw it,
    /// which is exactly the hole the splice check exists to catch.
    Advanced,
    /// A peer reported exactly the watermark and none is past it, so there is
    /// nothing we could have missed: the grafted state is continuous with a
    /// stream that has simply been silent.
    NoAdvance,
    /// Nobody answered, so the question stays open and the wait is re-armed.
    Unknown,
}

/// One rank a probe pass asks about, and the graft its verdict must still match
/// to be acted on (see [`PendingProof::watermark`]).
#[derive(Debug, Clone)]
pub(super) struct ProbeTarget {
    pub(super) rank: KvWorkerId,
    pub(super) watermark: i64,
    pub(super) epoch: u64,
}

/// Ask the fleet whether each target's publisher has moved past its watermark,
/// and post one verdict per target back to the pump.
///
/// Asks each peer for its cursor table alone, not a snapshot, and asks it ONCE
/// per pass for every target: the question is one integer per rank, and one
/// table answers it for all of them. The witness rules are on [`probe_fleet`].
pub(super) fn spawn_splice_probe(
    http: reqwest::Client,
    peers: Arc<PeerRegistry>,
    ctrl_tx: mpsc::Sender<PumpControl>,
    targets: Vec<ProbeTarget>,
) {
    tokio::spawn(async move {
        let verdicts = probe_fleet(&http, &peers, &targets, SPLICE_PROBE_BUDGET).await;
        for (target, verdict) in targets.into_iter().zip(verdicts) {
            let msg = PumpControl::SpliceProbe {
                rank: target.rank,
                epoch: target.epoch,
                watermark: target.watermark,
                verdict,
            };
            if let Err(e) = ctrl_tx.send(msg).await {
                // Only the pump's shutdown closes the receiver, and a gone pump
                // has no proof state left to care about — but say so for
                // forensics.
                debug!(
                    error = %e,
                    "kv-bootstrap: control channel closed; dropping probe verdicts",
                );
                return;
            }
        }
    });
}

/// One pass over the fleet: fetch each peer's cursor table at most once, within
/// `budget`, and judge every target against every table that arrives.
///
/// Any peer's cursor is admissible evidence: sequence numbers are the
/// publisher's, so a peer reporting one above our watermark proves a batch we
/// never received was emitted. A peer too cold to bootstrap from is still a
/// valid witness, which is why this reads the wire cursor directly instead of
/// vetting.
///
/// A peer is a witness for a target only when its table names the rank AT OR
/// ABOVE the watermark. A decodable body alone is not one — a peer that never
/// saw the rank cannot say its publisher has not moved — and neither is a
/// cursor below the watermark: that peer has not observed the stream even up
/// to the point in question (a stalled subscription, or a reset we missed), so
/// it cannot testify about what came after. Either would otherwise manufacture
/// `NoAdvance` out of ignorance and tally the rank warm as if proven.
async fn probe_fleet(
    http: &reqwest::Client,
    peers: &PeerRegistry,
    targets: &[ProbeTarget],
    budget: Duration,
) -> Vec<SpliceVerdict> {
    let mut verdicts = vec![SpliceVerdict::Unknown; targets.len()];
    let deadline = Instant::now() + budget;
    for peer in peers.candidates() {
        if verdicts.iter().all(|v| *v == SpliceVerdict::Advanced) {
            break;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        let snap = match tokio::time::timeout(remaining, fetch_cursors(http, &peer)).await {
            Ok(Ok(Some(snap))) => snap,
            // Non-200 is already logged inside fetch_body; a read failure gets
            // one line here, so a peer serving corrupt bodies is not
            // indistinguishable from a peer that never saw the rank. Either way
            // this peer is not a witness for this pass.
            Ok(Ok(None)) => continue,
            Ok(Err(e)) => {
                debug!(
                    peer = %peer,
                    error = %format_args!("{e:#}"),
                    "kv-bootstrap: splice probe could not read this peer",
                );
                continue;
            }
            Err(_) => {
                debug!(
                    peer = %peer,
                    budget_ms = budget.as_millis(),
                    "kv-bootstrap: splice probe ran out of budget; peers not yet \
                     asked are not witnesses this pass",
                );
                break;
            }
        };
        // Same rule as vetting: a format this build does not recognise may give
        // the cursor table a meaning it does not know.
        if snap.format != SNAPSHOT_FORMAT {
            debug!(
                peer = %peer,
                got = snap.format,
                want = SNAPSHOT_FORMAT,
                "kv-bootstrap: splice probe ignoring a peer with an unknown snapshot format",
            );
            continue;
        }
        for (target, verdict) in targets.iter().zip(verdicts.iter_mut()) {
            if *verdict == SpliceVerdict::Advanced {
                continue;
            }
            match snap.wire_cursor_for(&target.rank.url, target.rank.dp_rank) {
                Some(seq) if seq > target.watermark => {
                    warn!(
                        worker = ?target.rank,
                        peer = %peer,
                        watermark = target.watermark,
                        peer_cursor = seq,
                        "kv-bootstrap: a peer is past our unproven watermark, so a \
                         batch we never received was published; discarding grafted \
                         state for this rank",
                    );
                    *verdict = SpliceVerdict::Advanced;
                }
                Some(seq) if seq == target.watermark => {
                    *verdict = SpliceVerdict::NoAdvance;
                }
                _ => {}
            }
        }
    }
    verdicts
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::*;
    use super::*;
    use crate::state::kv_events::bootstrap::{PeerSnapshot, CURSORS_ONLY_PARAM};

    /// A peer that accepts and then never answers must not hold a pass past its
    /// budget: the pass ends with no witness rather than outliving the sweep.
    #[tokio::test]
    async fn probe_pass_gives_up_on_a_hung_peer_within_its_budget() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((sock, _)) = listener.accept().await {
                held.push(sock);
            }
        });
        let peers = PeerRegistry::new();
        peers.replace(vec![format!("http://{addr}")]);
        let targets = [ProbeTarget {
            rank: worker_id("http://w1:30000", 0),
            watermark: 10,
            epoch: 0,
        }];
        let verdicts = tokio::time::timeout(
            Duration::from_secs(5),
            probe_fleet(
                &reqwest::Client::new(),
                &peers,
                &targets,
                Duration::from_millis(200),
            ),
        )
        .await
        .expect("the budget, not the peer, must end the pass");
        assert_eq!(verdicts, vec![SpliceVerdict::Unknown]);
    }

    /// Run one probe pass against a one-peer fleet serving `snap`, returning
    /// the verdicts the probe posts to the pump-control channel — the same path
    /// a verdict takes in production — in target order. Asserts along the way
    /// that the peer was asked exactly once, for the cursor table alone.
    async fn probe_pass(snap: PeerSnapshot, probed: &[(KvWorkerId, i64)]) -> Vec<SpliceVerdict> {
        let (base, queries) = serve_snapshot_sequence(vec![snap]).await;
        let peers = Arc::new(PeerRegistry::new());
        peers.replace(vec![base]);
        let (tx, mut rx) = mpsc::channel(probed.len());
        let targets = probed
            .iter()
            .map(|(rank, watermark)| ProbeTarget {
                rank: rank.clone(),
                watermark: *watermark,
                epoch: 0,
            })
            .collect();
        spawn_splice_probe(reqwest::Client::new(), peers, tx, targets);
        let mut verdicts = Vec::new();
        for (rank, watermark) in probed {
            let outcome = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("a live one-peer fleet always gets an answer")
                .expect("the control channel outlives the probe");
            match outcome {
                PumpControl::SpliceProbe {
                    rank: got,
                    watermark: got_watermark,
                    verdict,
                    ..
                } => {
                    assert_eq!((&got, got_watermark), (rank, *watermark));
                    verdicts.push(verdict);
                }
                other => panic!("expected a splice-probe verdict, got {other:?}"),
            }
        }
        let queries = queries.lock().expect("queries lock");
        assert_eq!(queries.len(), 1, "one fetch per peer answers every target");
        assert!(
            queries.iter().all(|q| q
                .as_deref()
                .unwrap_or_default()
                .contains(CURSORS_ONLY_PARAM)),
            "every probe request must ask for the cursor table alone",
        );
        verdicts
    }

    async fn probe_once(snap: PeerSnapshot, probed: &KvWorkerId, watermark: i64) -> SpliceVerdict {
        probe_pass(snap, &[(probed.clone(), watermark)]).await[0]
    }

    /// A peer whose cursor for the rank is BELOW the watermark has not seen the
    /// stream even that far, so it cannot say the publisher stopped there.
    #[tokio::test]
    async fn probe_ignores_a_witness_below_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 4)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Unknown,
        );
    }

    /// Every due rank is judged from the same table: one request, one verdict
    /// per rank.
    #[tokio::test]
    async fn probe_pass_judges_every_target_from_one_fetch() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 11), ("http://w2:30000", 0, 7)]);
        assert_eq!(
            probe_pass(
                snap,
                &[
                    (worker_id("http://w1:30000", 0), 10),
                    (worker_id("http://w2:30000", 0), 7),
                ],
            )
            .await,
            vec![SpliceVerdict::Advanced, SpliceVerdict::NoAdvance],
        );
    }

    /// A peer whose cursor table does not NAME the probed rank has never
    /// observed its publisher, so its body alone must not resolve the verdict
    /// to NoAdvance.
    #[tokio::test]
    async fn probe_ignores_a_body_that_does_not_name_the_rank() {
        let snap = witness_snapshot(&[("http://other:30000", 0, 999)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Unknown,
        );
    }

    #[tokio::test]
    async fn probe_reports_no_advance_when_the_best_witness_is_at_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 10)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::NoAdvance,
        );
    }

    #[tokio::test]
    async fn probe_reports_advance_when_a_witness_is_past_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 11)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Advanced,
        );
    }

    /// A probe verdict as the probe task posts it.
    fn splice_verdict(
        rank: &KvWorkerId,
        epoch: u64,
        watermark: i64,
        verdict: SpliceVerdict,
    ) -> PumpControl {
        PumpControl::SpliceProbe {
            rank: rank.clone(),
            epoch,
            watermark,
            verdict,
        }
    }

    /// A peer that has applied a sequence ABOVE our watermark proves a batch we
    /// never received was published, so the grafted state goes.
    #[tokio::test]
    async fn pump_splice_probe_advanced_discards_grafted_state() {
        let id = worker_id("http://w1", 0);
        let (tracker, mut h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Advanced))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "positive evidence of a missed batch must discard the graft",
        );
        assert!(
            h.cursors.lock().get(&id).is_none(),
            "cursor must be dropped"
        );
        let requeued = h.bootstrap_rx.try_recv().expect("gapped rank re-queued");
        assert_eq!(requeued.obligations[0].0, id);
        assert_eq!(requeued.late_join, LateJoin::Refused);
        assert_eq!(
            rank_count(&tracker, "gap"),
            0,
            "a retried gap is not a verdict; the retry's outcome is the rank's one count",
        );
        assert_eq!(rank_count(&tracker, "warm"), 0);
    }

    /// A probe answers for the graft it was launched about. Once a batch 0 has
    /// replaced that graft with a restarted engine's stream, a verdict still in
    /// flight — its witnesses counted in the dead numbering — must not demote
    /// the new stream's state.
    #[tokio::test]
    async fn pump_drops_a_probe_verdict_about_a_graft_a_restart_replaced() {
        let id = worker_id("http://w1", 0);
        let (tracker, mut h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 0,
            batch: batch(vec![stored(None, vec![1000])]),
        })
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while h.cursors.lock().get(&id).copied() != Some(0) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the restarted stream's batch 0 is applied");
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Advanced))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(h.tree.match_prefix(None, &[1000]).workers().contains(&id));
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(0));
        assert!(h.bootstrap_rx.try_recv().is_err(), "nothing re-queued");
        assert_eq!(rank_count(&tracker, "from_origin"), 1);
    }

    /// Silence with no witness of advancement is NOT evidence of a hole. Keeping
    /// the tree here is the whole reason the timeout probes instead of discarding:
    /// a quiet fleet would otherwise lose every warm tree on a timer.
    #[tokio::test]
    async fn pump_splice_probe_no_advance_keeps_grafted_state_and_counts_warm() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::NoAdvance))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "proof by absence must keep the grafted state",
        );
        assert_eq!(rank_count(&tracker, "warm"), 1);
    }

    /// An unanswerable probe resolves nothing: the rank stays warm AND still
    /// awaits proof, so a later witness can still demote it. Resolving `Unknown`
    /// either way would make an unreachable fleet decide the question.
    #[tokio::test]
    async fn pump_splice_probe_unknown_leaves_the_question_open() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Unknown))
            .await
            .unwrap();
        // Still pending proof, so this second verdict must still be actionable.
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Advanced))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "an Unknown verdict must not consume the pending proof; the later \
             Advanced verdict then gaps, which re-queues the rank",
        );
        assert_eq!(rank_count(&tracker, "warm"), 0);
    }

    /// A fleet that can never answer must not leave the rank probing forever: the
    /// graft is kept, but tallied under a label that says it was never witnessed
    /// rather than pretending it was proven.
    #[tokio::test]
    async fn pump_repeated_unknown_probes_resolve_as_unwitnessed() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        for _ in 0..MAX_UNKNOWN_PROBES {
            h.ctrl_tx
                .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Unknown))
                .await
                .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "an unwitnessed graft is kept, not discarded",
        );
        assert!(h
            .tree
            .match_prefix(None, &[100, 200])
            .workers()
            .contains(&id));
        assert_eq!(rank_count(&tracker, "warm_unwitnessed"), 1);
        assert_eq!(
            rank_count(&tracker, "warm"),
            0,
            "unwitnessed must not be conflated with proven",
        );
    }

    /// A probe answer for a worker that was removed and re-added while it was in
    /// flight must not touch the new incarnation's state.
    #[tokio::test]
    async fn pump_splice_probe_from_a_stale_incarnation_is_ignored() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let stale_epoch = tracker.epoch_of(&id).expect("registered");
        tracker.forget(std::slice::from_ref(&id));
        tracker.register(std::slice::from_ref(&id));

        h.ctrl_tx
            .send(splice_verdict(&id, stale_epoch, 5, SpliceVerdict::Advanced))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "the re-registered incarnation must be left alone",
        );
        assert_eq!(rank_count(&tracker, "gap"), 0);
    }

    /// A gap retry re-grafts under the SAME epoch, so a verdict still in flight
    /// about the first graft's watermark (a second probe pass, say) must not
    /// demote the second graft: its question was about a different watermark.
    #[tokio::test]
    async fn pump_splice_probe_about_a_previous_graft_is_ignored() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        let advanced_past_5 = || splice_verdict(&id, epoch, 5, SpliceVerdict::Advanced);
        // First verdict gaps the graft and re-queues the rank as Pending.
        h.ctrl_tx.send(advanced_past_5()).await.unwrap();
        // The retry grafts a fresher snapshot, deferred again.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 9)),
            })
            .await
            .unwrap();
        // A late duplicate about the FIRST graft's watermark.
        h.ctrl_tx.send(advanced_past_5()).await.unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "the retry's graft must survive a verdict about its predecessor",
        );
        assert_eq!(rank_count(&tracker, "gap"), 0, "the first gap was retried");
    }
}
