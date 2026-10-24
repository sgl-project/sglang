//! Test fixtures shared by the handler modules' unit tests.

use super::handle::{CoreConfig, CoreHandle, CoreMetadata};
use crate::message::request::{GenerateRequest, Request, RequestKind};
use crate::message::response::ResponseItem;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

pub(super) struct Harness {
    pub(super) handle: CoreHandle,
    pub(super) intake_rx: flume::Receiver<TmEvent>,
    pub(super) abort_rx: flume::Receiver<AbortSource>,
}

impl Harness {
    pub(super) fn unbounded(response_capacity: usize) -> Self {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        Self {
            handle: test_handle(intake_tx, abort_tx, response_capacity),
            intake_rx,
            abort_rx,
        }
    }
}

pub(super) fn test_handle(
    intake_tx: flume::Sender<TmEvent>,
    abort_tx: flume::Sender<AbortSource>,
    response_capacity: usize,
) -> CoreHandle {
    configured_handle(
        intake_tx,
        abort_tx,
        response_capacity,
        Default::default(),
        false,
        false,
        Vec::new(),
    )
}

pub(super) fn configured_handle(
    intake_tx: flume::Sender<TmEvent>,
    abort_tx: flume::Sender<AbortSource>,
    response_capacity: usize,
    response_activity: ActivityCounter,
    startup_ready: bool,
    is_disaggregation: bool,
    mm_limits: Vec<(String, usize)>,
) -> CoreHandle {
    CoreHandle::new(
        intake_tx,
        abort_tx,
        CoreConfig {
            response_capacity,
            response_activity,
            startup_ready,
            is_disaggregation,
            mm_limits,
            peer_endpoints: Vec::new(),
            metadata: CoreMetadata::default(),
        },
    )
}

pub(super) fn generate(rid: &str) -> RequestKind {
    RequestKind::Generate(Box::new(GenerateRequest {
        rid: rid.into(),
        ..Default::default()
    }))
}

pub(super) fn accept_intake(event: TmEvent) -> Request {
    let TmEvent::Intake { request, admission } = event else {
        panic!("expected fresh intake request");
    };
    assert!(admission.try_accept(), "test request should still be live");
    request
}

/// Answer the one control request the local scheduler receives with
/// `reply`, as the Python scheduler would through the egress ring.
pub(super) async fn answer_local_control(
    intake_rx: &flume::Receiver<TmEvent>,
    reply: bytes::Bytes,
) {
    let request = accept_intake(intake_rx.recv_async().await.unwrap());
    assert!(matches!(request.kind, RequestKind::Control(_)));
    request
        .sink
        .try_send(ResponseItem::Control(reply))
        .expect("fresh response channel has room");
}
