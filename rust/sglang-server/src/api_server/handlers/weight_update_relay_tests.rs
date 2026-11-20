//! Concurrent weight updates across DP listeners (sgl-project/sglang#51333,
//! item 2). Two listeners are wired as each other's only peer in-process: a
//! [`LoopbackPeer`] hands the encoded request to the other handle's
//! [`CoreHandle::forward_control_req`], as the `/internal/peer` route does.
//! Each listener's scheduler is a [`FakeScheduler`] on its `intake_rx`.
//! Every test ends with neither listener holding admission.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;

use super::handle::{CoreConfig, CoreHandle, CoreMetadata};
use super::test_utils::accept_intake;
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout::{ControlTarget, RankId, RankReply, RequestBuilder};
use crate::message::config::DeploymentInfo;
use crate::message::io_struct::DiskWeightUpdate;
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::tokenizer_manager::wiring::TmEvent;

/// Long enough for two in-process updates on an idle CPU, short enough to
/// fail fast on the deadlock.
const UPDATE_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a relay waits for the other origin to start fanning out, so both
/// origins hold their own update lock before either relay arrives (two HTTP
/// posts in flight at once). Arbitrary; it only bounds the wait when the
/// other origin never fans out.
const RENDEZVOUS_WAIT: Duration = Duration::from_millis(200);

/// What a [`LoopbackPeer`] does with a request.
#[derive(Clone, Copy)]
enum PeerLink {
    /// Relay it to the peer listener, as `/internal/peer` does.
    Relay,
    /// Never answer: a peer whose response does not come back.
    Silent,
    /// Fail at once, as a peer that is down (HTTP 503).
    Unavailable,
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
    rendezvous: Option<Rendezvous>,
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
            let outcome = match self.link {
                PeerLink::Relay => {
                    let header = build().encode().expect("control requests encode");
                    let peer = self.peer.get().expect("peer wired before use");
                    peer.forward_control_req(header).await
                }
                PeerLink::Silent => futures::future::pending().await,
                PeerLink::Unavailable => Err(CoreError::Unavailable),
            };
            vec![RankReply {
                rank: self.ranks[0],
                outcome,
            }]
        })
    }
}

/// A listener's scheduler: answers each control request with success and
/// logs the `model_path` of every weight update, in the order it applied
/// them. With `held`, it takes requests but answers none until released.
struct FakeScheduler {
    applied: Arc<Mutex<Vec<String>>>,
    release: Arc<tokio::sync::Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeScheduler {
    fn spawn(intake_rx: flume::Receiver<TmEvent>, held: bool) -> Self {
        let applied = Arc::new(Mutex::new(Vec::new()));
        let release = Arc::new(tokio::sync::Notify::new());
        let task = tokio::spawn({
            let applied = applied.clone();
            let release = release.clone();
            async move {
                while let Ok(event) = intake_rx.recv_async().await {
                    let request = accept_intake(event);
                    let RequestKind::Control(control) = &request.kind else {
                        panic!("only control requests are sent in these tests");
                    };
                    applied
                        .lock()
                        .unwrap()
                        .push(model_path(&control.encode().unwrap()));
                    if held {
                        release.notified().await;
                    }
                    // The caller may be gone (a dropped origin); nothing to answer then.
                    let _ = request
                        .sink
                        .try_send(ResponseItem::Control(disk_update_reply()));
                }
            }
        });
        Self {
            applied,
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

/// `UpdateWeightFromDiskReqInput` is positional: `[tag, rid, ipc, model_path, ..]`.
fn model_path(header: &[u8]) -> String {
    let rmpv::Value::Array(items) = rmpv::decode::read_value(&mut &header[..]).unwrap() else {
        panic!("positional wire");
    };
    assert_eq!(items[0].as_str(), Some("UpdateWeightFromDiskReqInput"));
    items[3].as_str().unwrap().to_owned()
}

fn disk_update_reply() -> Bytes {
    Bytes::from(
        rmp_serde::to_vec_named(&serde_json::json!({
            "success": true, "message": "ok", "num_paused_requests": 0
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

/// One DP rank's listener in a `dp_size = 2` deployment.
fn listener(dp_rank: RankId, peer: LoopbackPeer) -> (CoreHandle, flume::Receiver<TmEvent>) {
    let (intake_tx, intake_rx) = flume::unbounded();
    let (abort_tx, _abort_rx) = flume::unbounded();
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
    (handle, intake_rx)
}

/// Listeners A (DP rank 0) and B (DP rank 1), each the other's only peer,
/// with A's link to B and B's link to A as given.
struct Pair {
    a: CoreHandle,
    b: CoreHandle,
    scheduler_a: FakeScheduler,
    scheduler_b: FakeScheduler,
}

impl Pair {
    fn new(a_to_b: PeerLink, b_to_a: PeerLink, held_b: bool, rendezvous: bool) -> Self {
        let rendezvous = rendezvous.then(Rendezvous::new);
        let slot_a = Arc::new(OnceLock::new());
        let slot_b = Arc::new(OnceLock::new());
        let (a, intake_a) = listener(
            0,
            LoopbackPeer {
                ranks: [1],
                peer: slot_b.clone(),
                link: a_to_b,
                rendezvous: rendezvous.clone(),
            },
        );
        let (b, intake_b) = listener(
            1,
            LoopbackPeer {
                ranks: [0],
                peer: slot_a.clone(),
                link: b_to_a,
                rendezvous,
            },
        );
        assert!(slot_a.set(a.clone()).is_ok() && slot_b.set(b.clone()).is_ok());
        Self {
            a,
            b,
            scheduler_a: FakeScheduler::spawn(intake_a, false),
            scheduler_b: FakeScheduler::spawn(intake_b, held_b),
        }
    }

    fn assert_released(&self) {
        assert!(!self.a.inner.frontend.is_paused(), "listener A still held");
        assert!(!self.b.inner.frontend.is_paused(), "listener B still held");
    }
}

/// Two weight updates started at once on two DP listeners both finish, and
/// both schedulers apply them in the same order, so the ranks end on the
/// same weights. Each origin holds its own update lock across the fan-out
/// while the relay waits for the peer's, so today this times out.
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
    // A fix may refuse one of the two (busy) rather than queue it; both must
    // still answer.
    assert!(
        from_a.is_ok() || from_b.is_ok(),
        "at least one update goes through: {from_a:?} / {from_b:?}"
    );
    assert_eq!(
        pair.scheduler_a.applied(),
        pair.scheduler_b.applied(),
        "every rank applies the updates in the same order"
    );
    pair.assert_released();
}

/// A peer whose answer never comes back must not hold the origin listener
/// forever. Ignored: the peer relay has no response deadline today
/// (`core/peer.rs` waits "as long as it takes"); enable with the fix.
#[ignore = "sgl-project/sglang#51333 item 2: no deadline on a peer's reply yet"]
#[tokio::test(start_paused = true)]
async fn an_update_whose_peer_never_answers_ends_and_releases_both_listeners() {
    let pair = Pair::new(PeerLink::Silent, PeerLink::Relay, false, false);
    // Virtual time: the runtime skips ahead while every task is waiting, so
    // this covers any deadline the fix picks below ten minutes.
    let outcome = tokio::time::timeout(
        Duration::from_secs(600),
        pair.a
            .update_weights_from_disk(disk_update("/models/from-a")),
    )
    .await
    .expect("the origin gives up on a silent peer");
    assert!(
        outcome.is_err(),
        "a rank that never answered fails the update"
    );
    pair.assert_released();
}

/// A peer that refuses the relay fails that update only; the other
/// listener's concurrent update still finishes, and neither stays held.
#[tokio::test]
async fn an_update_whose_peer_fails_releases_both_listeners() {
    let pair = Pair::new(PeerLink::Unavailable, PeerLink::Relay, false, true);
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
    assert!(from_a.is_err(), "rank 1 did not take A's update");
    assert!(from_b.is_ok(), "B's update reaches both ranks: {from_b:?}");
    pair.assert_released();
}

/// The origin's caller goes away (client disconnect) while the peer's
/// scheduler is still applying the update: both listeners are released
/// once that scheduler answers, and nothing stays held. In-process only:
/// whether the HTTP server drops the `/internal/peer` handler on disconnect
/// is not covered here.
#[tokio::test]
async fn dropping_the_origin_mid_fan_out_releases_both_listeners() {
    let pair = Pair::new(PeerLink::Relay, PeerLink::Relay, true, false);
    let origin = tokio::spawn({
        let a = pair.a.clone();
        async move {
            a.update_weights_from_disk(disk_update("/models/from-a"))
                .await
        }
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
    pair.scheduler_b.release.notify_one();
    tokio::time::timeout(UPDATE_TIMEOUT, async {
        while pair.a.inner.frontend.is_paused() || pair.b.inner.frontend.is_paused() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("both listeners released after the origin is dropped");
    pair.assert_released();
}
