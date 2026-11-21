//! Cross-rank weight updates across DP listeners (sgl-project/sglang#51333,
//! item 2). Two listeners are wired as each other's only peer in-process: a
//! [`LoopbackPeer`] hands a peer relay's encoded request to the other
//! handle's [`CoreHandle::forward_control_req`], as the `/internal/peer`
//! route does, and a forwarded update's JSON to its
//! [`CoreHandle::coordinate_forwarded_weight_update`], as the
//! `/internal/weight_update` route does. Listener A is DP rank 0's, the
//! coordinator. Each listener's scheduler is a [`FakeScheduler`] on its
//! `intake_rx`. Every test ends with neither listener holding admission.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;

use super::handle::{CoreConfig, CoreHandle, CoreMetadata};
use super::test_utils::accept_intake;
use super::weight_coordinator::PEER_REPLY_DEADLINE;
use super::weights::DiskUpdateOutcome;
use crate::api_server::core::fanout::{ControlTarget, RankId, RankReply, RequestBuilder};
use crate::api_server::core::{CoreError, CoreErrorKind, ReceivedAs};
use crate::message::config::DeploymentInfo;
use crate::message::ids::Rid;
use crate::message::io_struct::DiskWeightUpdate;
use crate::message::request::{GenerateRequest, Request, RequestKind};
use crate::message::response::ResponseItem;
use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

/// Long enough for two in-process updates on an idle CPU, short enough to
/// fail fast on the deadlock.
const UPDATE_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a relay waits for the other origin to start fanning out, so both
/// origins hold their own update lock before either relay arrives (two HTTP
/// posts in flight at once). Arbitrary; it only bounds the wait when the
/// other origin never fans out.
const RENDEZVOUS_WAIT: Duration = Duration::from_millis(200);

/// What a [`LoopbackPeer`] does with a peer relay or a forwarded update.
#[derive(Clone, Copy)]
enum PeerLink {
    /// Hand it to the peer listener, as its internal routes do.
    Relay,
    /// Never answer: a peer whose response does not come back.
    Silent,
    /// Fail at once, as a peer that is down (HTTP 503).
    Unavailable,
    /// Fail the first request as [`PeerLink::Unavailable`], relay the rest.
    DownOnce,
}

/// Lets each relay wait until both origins have started fanning out.
#[derive(Clone)]
struct Rendezvous(Arc<tokio::sync::watch::Sender<usize>>);

impl Rendezvous {
    fn new() -> Self {
        Self(Arc::new(tokio::sync::watch::Sender::new(0)))
    }

    async fn arrive(&self) {
        self.0.send_modify(|arrived| *arrived += 1);
        let mut arrived = self.0.subscribe();
        let _ = tokio::time::timeout(RENDEZVOUS_WAIT, arrived.wait_for(|n| *n >= 2)).await;
    }
}

/// The other DP rank's listener, reached in-process instead of over HTTP.
struct LoopbackPeer {
    ranks: [RankId; 1],
    /// Set once both handles exist: each one's config names the other.
    peer: Arc<OnceLock<CoreHandle>>,
    link: PeerLink,
    /// Whether a [`PeerLink::DownOnce`] link has failed its request yet.
    went_down: AtomicBool,
    rendezvous: Option<Rendezvous>,
}

impl LoopbackPeer {
    fn new(rank: RankId, peer: Arc<OnceLock<CoreHandle>>, link: PeerLink) -> Self {
        Self {
            ranks: [rank],
            peer,
            link,
            went_down: AtomicBool::new(false),
            rendezvous: None,
        }
    }

    fn peer(&self) -> &CoreHandle {
        self.peer.get().expect("peer wired before use")
    }

    /// Whether this request goes through, or how it fails.
    async fn pass(&self) -> Result<(), CoreError> {
        match self.link {
            PeerLink::Relay => Ok(()),
            PeerLink::Silent => futures::future::pending().await,
            PeerLink::Unavailable => Err(CoreError::Unavailable),
            PeerLink::DownOnce if !self.went_down.swap(true, Ordering::SeqCst) => {
                Err(CoreError::Unavailable)
            }
            PeerLink::DownOnce => Ok(()),
        }
    }
}

impl ControlTarget for LoopbackPeer {
    fn ranks(&self) -> &[RankId] {
        &self.ranks
    }

    fn control<'a>(&'a self, build: &'a RequestBuilder<'a>) -> BoxFuture<'a, Vec<RankReply>> {
        Box::pin(async move {
            if let Some(rendezvous) = &self.rendezvous {
                rendezvous.arrive().await;
            }
            let outcome = match self.pass().await {
                Ok(()) => {
                    let header = build().encode().expect("control requests encode");
                    self.peer().forward_control_req(header).await
                }
                Err(error) => Err(error),
            };
            vec![RankReply {
                rank: self.ranks[0],
                outcome,
            }]
        })
    }

    fn coordinate_weight_update(&self, body: Bytes) -> BoxFuture<'_, Result<Bytes, CoreError>> {
        Box::pin(async move {
            self.pass().await?;
            self.peer().coordinate_forwarded_weight_update(body).await
        })
    }
}

/// A listener's scheduler: answers each control request with success and
/// logs every weight update (a disk update by its `model_path`, any other
/// by its message type) in the order it applied them. Generations are taken
/// and left running. With `held`, it takes control requests but answers none
/// until released.
struct FakeScheduler {
    applied: Arc<Mutex<Vec<String>>>,
    generations: Arc<Mutex<Vec<Request>>>,
    release: Arc<tokio::sync::Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeScheduler {
    fn spawn(rank: RankId, intake_rx: flume::Receiver<TmEvent>, held: bool) -> Self {
        let applied = Arc::new(Mutex::new(Vec::new()));
        let generations = Arc::new(Mutex::new(Vec::new()));
        let release = Arc::new(tokio::sync::Notify::new());
        let task = tokio::spawn({
            let applied = applied.clone();
            let generations = generations.clone();
            let release = release.clone();
            async move {
                while let Ok(event) = intake_rx.recv_async().await {
                    let request = accept_intake(event);
                    let RequestKind::Control(control) = &request.kind else {
                        generations.lock().unwrap().push(request);
                        continue;
                    };
                    applied
                        .lock()
                        .unwrap()
                        .push(update_label(&control.encode().unwrap()));
                    if held {
                        release.notified().await;
                    }
                    // The caller may be gone (a dropped origin); nothing to answer then.
                    let _ = request
                        .sink
                        .try_send(ResponseItem::Control(update_reply(rank)));
                }
            }
        });
        Self {
            applied,
            generations,
            release,
            task,
        }
    }

    fn applied(&self) -> Vec<String> {
        self.applied.lock().unwrap().clone()
    }
}

impl Drop for FakeScheduler {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A weight update's message type, or for `UpdateWeightFromDiskReqInput`
/// (positional: `[tag, rid, ipc, model_path, ..]`) its `model_path`.
fn update_label(header: &[u8]) -> String {
    let rmpv::Value::Array(items) = rmpv::decode::read_value(&mut &header[..]).unwrap() else {
        panic!("positional wire");
    };
    match items[0].as_str() {
        Some("UpdateWeightFromDiskReqInput") => items[3].as_str().unwrap().to_owned(),
        Some(tag) => tag.to_owned(),
        None => panic!("tagged wire"),
    }
}

/// Rank `rank`'s answer: success, named after the rank, with `10 + rank`
/// paused requests so the merged outcome shows where each entry came from.
fn update_reply(rank: RankId) -> Bytes {
    Bytes::from(
        rmp_serde::to_vec_named(&serde_json::json!({
            "success": true,
            "message": format!("rank {rank}"),
            "num_paused_requests": 10 + rank,
        }))
        .unwrap(),
    )
}

fn disk_update(model_path: &str) -> DiskWeightUpdate {
    DiskWeightUpdate {
        model_path: model_path.into(),
        ..Default::default()
    }
}

/// One DP rank's listener in a `dp_size = 2` deployment, with its intake
/// and abort channels.
fn listener(
    dp_rank: RankId,
    peer: LoopbackPeer,
) -> (
    CoreHandle,
    flume::Receiver<TmEvent>,
    flume::Receiver<AbortSource>,
) {
    let (intake_tx, intake_rx) = flume::unbounded();
    let (abort_tx, abort_rx) = flume::unbounded();
    let handle = CoreHandle::new(
        intake_tx,
        abort_tx,
        CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: true,
            is_disaggregation: false,
            multimodal: false,
            metrics: None,
            request_log: None,
            generate_validator: None,
            abort_hook: None,
            mm_limits: Vec::new(),
            metadata: CoreMetadata {
                deployment_info: DeploymentInfo {
                    dp_size: 2,
                    dp_rank: Some(dp_rank),
                    ..Default::default()
                },
                ..Default::default()
            },
            peer_endpoints: vec![Arc::new(peer)],
        },
    );
    (handle, intake_rx, abort_rx)
}

/// Listeners A (DP rank 0, the coordinator) and B (DP rank 1), each the
/// other's only peer, with A's link to B and B's link to A as given.
struct Pair {
    a: CoreHandle,
    b: CoreHandle,
    scheduler_a: FakeScheduler,
    scheduler_b: FakeScheduler,
    aborts_a: flume::Receiver<AbortSource>,
    aborts_b: flume::Receiver<AbortSource>,
}

impl Pair {
    fn new(a_to_b: PeerLink, b_to_a: PeerLink, held_b: bool, rendezvous: bool) -> Self {
        let rendezvous = rendezvous.then(Rendezvous::new);
        let slot_a = Arc::new(OnceLock::new());
        let slot_b = Arc::new(OnceLock::new());
        let (a, intake_a, aborts_a) = listener(
            0,
            LoopbackPeer {
                rendezvous: rendezvous.clone(),
                ..LoopbackPeer::new(1, slot_b.clone(), a_to_b)
            },
        );
        let (b, intake_b, aborts_b) = listener(
            1,
            LoopbackPeer {
                rendezvous,
                ..LoopbackPeer::new(0, slot_a.clone(), b_to_a)
            },
        );
        assert!(slot_a.set(a.clone()).is_ok() && slot_b.set(b.clone()).is_ok());
        Self {
            a,
            b,
            scheduler_a: FakeScheduler::spawn(0, intake_a, false),
            scheduler_b: FakeScheduler::spawn(1, intake_b, held_b),
            aborts_a,
            aborts_b,
        }
    }

    fn assert_released(&self) {
        assert!(!self.a.inner.frontend.is_paused(), "listener A still held");
        assert!(!self.b.inner.frontend.is_paused(), "listener B still held");
    }

    async fn wait_released(&self) {
        tokio::time::timeout(UPDATE_TIMEOUT, async {
            while self.a.inner.frontend.is_paused() || self.b.inner.frontend.is_paused() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("both listeners released");
    }
}

/// Two weight updates started at once on two DP listeners both finish, and
/// both schedulers apply them in the same order, so the ranks end on the
/// same weights. Neither may wait on a lock the other holds while holding
/// its own; and dropping the peer relay's lock instead would let the two
/// schedulers apply the updates in opposite orders.
#[tokio::test]
async fn concurrent_updates_on_two_dp_listeners_both_finish() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Relay, false, true);
    let both = async {
        tokio::join!(
            pair.a
                .update_weights_from_disk(disk_update("/models/from-a")),
            pair.b
                .update_weights_from_disk(disk_update("/models/from-b")),
        )
    };
    let (from_a, from_b) = tokio::time::timeout(UPDATE_TIMEOUT, both)
        .await
        .unwrap_or_else(|_| {
            panic!(
                "cross-DP weight-update deadlock: neither update finished within \
                 {UPDATE_TIMEOUT:?}; scheduler A applied {:?}, scheduler B applied {:?}",
                pair.scheduler_a.applied(),
                pair.scheduler_b.applied(),
            )
        });
    assert!(
        from_a.is_ok() && from_b.is_ok(),
        "the second update waits its turn: {from_a:?} / {from_b:?}"
    );
    assert_eq!(pair.scheduler_a.applied().len(), 2);
    assert_eq!(
        pair.scheduler_a.applied(),
        pair.scheduler_b.applied(),
        "every rank applies the updates in the same order"
    );
    pair.assert_released();
}

/// An update sent to a listener other than DP rank 0's reaches every rank,
/// and its client gets every rank's outcome merged in rank order, as from
/// the coordinator.
#[tokio::test]
async fn an_update_sent_to_another_listener_returns_every_ranks_outcome_in_rank_order() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Relay, false, false);
    let outcome = tokio::time::timeout(
        UPDATE_TIMEOUT,
        pair.b.update_weights_from_disk(disk_update("/models/new")),
    )
    .await
    .expect("the update finishes");
    assert_eq!(
        outcome.unwrap(),
        DiskUpdateOutcome {
            success: true,
            message: "rank 0 | rank 1".into(),
            num_paused_requests: vec![10, 11],
        }
    );
    assert_eq!(pair.scheduler_a.applied(), ["/models/new"]);
    assert_eq!(pair.scheduler_b.applied(), ["/models/new"]);
    pair.assert_released();
}

/// A peer that refuses the relay fails that update only, with the peer's
/// category; the other listener's concurrent update still finishes, and
/// neither listener stays held. The rank that applied the failed update
/// keeps it.
#[tokio::test]
async fn an_update_whose_peer_fails_releases_both_listeners() {
    let pair = Pair::new(PeerLink::DownOnce, PeerLink::Relay, false, false);
    let both = async {
        tokio::join!(
            pair.a
                .update_weights_from_disk(disk_update("/models/from-a")),
            pair.b
                .update_weights_from_disk(disk_update("/models/from-b")),
        )
    };
    let (from_a, from_b) = tokio::time::timeout(UPDATE_TIMEOUT, both)
        .await
        .expect("a failing peer ends the update instead of waiting");
    let (failed, finished) = match (from_a, from_b) {
        (Err(error), Ok(outcome)) | (Ok(outcome), Err(error)) => (error, outcome),
        other => panic!("exactly one update fails: {other:?}"),
    };
    assert_eq!(failed.kind(), CoreErrorKind::Unavailable, "{failed}");
    assert!(finished.success);
    assert_eq!(pair.scheduler_a.applied().len(), 2, "no rollback on rank 0");
    assert_eq!(pair.scheduler_b.applied().len(), 1);
    pair.assert_released();
}

/// A peer whose answer never comes back costs a bounded wait: the update
/// fails at the coordinator's deadline, with an error naming that rank,
/// before the origin's own wait on the coordinator ends, and no listener
/// stays held.
#[tokio::test(start_paused = true)]
async fn an_update_whose_peer_never_answers_ends_and_releases_both_listeners() {
    let pair = Pair::new(PeerLink::Silent, PeerLink::Relay, false, false);
    // Virtual time: the runtime skips ahead while every task is waiting.
    let outcome = tokio::time::timeout(
        PEER_REPLY_DEADLINE * 2,
        pair.b.update_weights_from_disk(disk_update("/models/new")),
    )
    .await
    .expect("the coordinator gives up on a silent peer");
    let error = outcome.unwrap_err();
    assert_eq!(error.kind(), CoreErrorKind::DeadlineExceeded, "{error}");
    assert!(
        error.to_string().contains("dp rank 1"),
        "the coordinator's error, naming the rank: {error}"
    );
    pair.assert_released();
}

/// The origin's caller goes away (client disconnect) mid-fan-out: the
/// update still runs to the end on every rank, so the ranks do not end on
/// mixed weights. The peer stays held until its scheduler answers, and no
/// abort reaches that scheduler.
#[tokio::test]
async fn dropping_the_origin_mid_fan_out_still_finishes_the_update() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Relay, true, false);
    let origin = tokio::spawn({
        let b = pair.b.clone();
        async move { b.update_weights_from_disk(disk_update("/models/new")).await }
    });
    tokio::time::timeout(UPDATE_TIMEOUT, async {
        while pair.scheduler_b.applied().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the update reaches B's scheduler");
    origin.abort();
    let _ = origin.await;
    tokio::task::yield_now().await;
    assert!(
        pair.b.inner.frontend.is_paused(),
        "B stays held while its scheduler runs the update"
    );
    assert!(
        pair.aborts_b.try_recv().is_err(),
        "the update is not aborted on B"
    );
    pair.scheduler_b.release.notify_one();
    pair.wait_released().await;
    assert_eq!(pair.scheduler_a.applied(), ["/models/new"]);
    pair.assert_released();
}

/// With DP rank 0's listener unreachable, an update sent elsewhere fails at
/// once as unavailable, and reaches and holds nothing.
#[tokio::test]
async fn an_update_fails_at_once_when_the_coordinator_is_unreachable() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Unavailable, false, false);
    let outcome = tokio::time::timeout(
        UPDATE_TIMEOUT,
        pair.b.update_weights_from_disk(disk_update("/models/new")),
    )
    .await
    .expect("an unreachable coordinator fails the update at once");
    assert_eq!(outcome.unwrap_err().kind(), CoreErrorKind::Unavailable);
    assert!(pair.scheduler_a.applied().is_empty());
    assert!(pair.scheduler_b.applied().is_empty());
    pair.assert_released();
}

/// Begin and end sent to another listener reach every rank through the
/// coordinator, in order, and answer the merged outcome.
#[tokio::test]
async fn begin_and_end_weight_update_go_through_the_coordinator() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Relay, false, false);
    let begin = pair.b.begin_weight_update("all".into()).await.unwrap();
    let end = pair.b.end_weight_update().await.unwrap();
    for outcome in [begin, end] {
        assert!(outcome.success);
        assert_eq!(outcome.message, "rank 0 | rank 1");
    }
    let expected = ["BeginWeightUpdateReqInput", "EndWeightUpdateReqInput"];
    assert_eq!(pair.scheduler_a.applied(), expected);
    assert_eq!(pair.scheduler_b.applied(), expected);
    pair.assert_released();
}

/// `abort_all_requests` aborts the generations in flight on every listener,
/// not only on the one the update was sent to.
#[tokio::test]
async fn abort_all_requests_aborts_generations_on_every_listener() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Relay, false, false);
    let generate = |handle: CoreHandle, rid: &'static str| async move {
        handle
            .generate(
                GenerateRequest {
                    rid: Rid::from_client(rid),
                    input_ids: Some(vec![1]),
                    ..Default::default()
                },
                ReceivedAs::without_headers(),
            )
            .await
    };
    let running_a = generate(pair.a.clone(), "on-a").await.unwrap();
    let running_b = generate(pair.b.clone(), "on-b").await.unwrap();
    tokio::time::timeout(UPDATE_TIMEOUT, async {
        while [&pair.scheduler_a, &pair.scheduler_b]
            .iter()
            .any(|scheduler| scheduler.generations.lock().unwrap().is_empty())
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("both schedulers run a generation");
    let update = DiskWeightUpdate {
        abort_all_requests: true,
        ..disk_update("/models/new")
    };
    let outcome = tokio::time::timeout(UPDATE_TIMEOUT, pair.b.update_weights_from_disk(update))
        .await
        .expect("the update finishes");
    assert!(outcome.unwrap().success);
    for (listener, aborts, rid) in [("A", &pair.aborts_a, "on-a"), ("B", &pair.aborts_b, "on-b")] {
        match aborts.try_recv() {
            Ok(AbortSource::Request(aborted)) => {
                assert_eq!(aborted.client_facing(), rid, "{listener}")
            }
            other => panic!("listener {listener}'s generation was not aborted: {other:?}"),
        }
    }
    drop((running_a, running_b));
    pair.assert_released();
}

/// A forwarded update that reaches a listener other than DP rank 0's is
/// refused with 421, not forwarded again, so a wrong endpoint table cannot
/// loop it.
#[tokio::test]
async fn a_forwarded_update_is_refused_by_a_listener_that_does_not_coordinate() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Relay, false, false);
    let body = serde_json::json!({"kind": "end_weight_update"}).to_string();
    let error = pair
        .b
        .coordinate_forwarded_weight_update(Bytes::from(body))
        .await
        .unwrap_err();
    assert_eq!(error.http_status().as_u16(), 421, "{error}");
    assert!(pair.scheduler_a.applied().is_empty());
    assert!(pair.scheduler_b.applied().is_empty());
    pair.assert_released();
}
