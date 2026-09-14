//! Transport-neutral entry point into the Rust frontend pipeline.
//!
//! Wire adapters normalize their protocol into the existing in-process request
//! types, submit them through [`FrontendHandle`], and render the returned
//! [`ResponseItem`] values for their own transport. This module owns shared
//! preprocessing, runtime capabilities, and request lifetime; it deliberately
//! knows nothing about Axum, HTTP response shapes, Tonic, or protobuf.

use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::mpsc;

use crate::message::ids::Rid;
use crate::message::io_struct::ControlRequest;
use crate::message::request::{GenerateRequest, Request, RequestKind};
use crate::message::response::{ResponseItem, ResponseSink};
use crate::message::sampling::SamplingParams;
use crate::message::types::TokenIds;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission, TmEvent};
use crate::utils::error::Error;
use crate::utils::fsm::RequestState;

mod prefetch;

/// Sentinel host that makes the KV connector no-op. Parity with
/// `sglang.srt.disaggregation.utils.FAKE_BOOTSTRAP_HOST`.
const FAKE_BOOTSTRAP_HOST: &str = "2.2.2.2";

/// A transport-neutral failure while executing a frontend operation.
#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum FrontendError {
    /// The TokenizerManager intake loop has shut down, so callers may retry on
    /// another healthy server.
    #[error("service unavailable")]
    Unavailable,

    #[error("{0}")]
    InvalidArgument(String),

    #[error(transparent)]
    Pipeline(#[from] Error),

    #[error("response channel closed before terminal output")]
    ResponseClosed,

    #[error("{0}")]
    UnexpectedResponse(&'static str),
}

/// Semantic result of a deep frontend health check.
///
/// Expected lifecycle states are values rather than transport errors so each
/// adapter can render them according to its own protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HealthStatus {
    Healthy,
    NotReady,
    Stalled,
}

pub(crate) struct FrontendConfig {
    pub(crate) response_capacity: usize,
    pub(crate) response_activity: ActivityCounter,
    pub(crate) startup_ready: bool,
    pub(crate) is_disaggregation: bool,
    pub(crate) mm_limits: BTreeMap<String, usize>,
}

/// Cloneable capability for submitting work to the existing frontend runtime.
///
/// Only the runtime constructs this handle. Protocol adapters receive clones;
/// they cannot reach the raw stage wiring directly.
#[derive(Clone)]
pub(crate) struct FrontendHandle {
    inner: Arc<FrontendInner>,
}

struct FrontendInner {
    intake_tx: flume::Sender<TmEvent>,
    abort_tx: flume::Sender<AbortSource>,
    response_capacity: usize,
    response_activity: ActivityCounter,
    startup_ready: AtomicBool,
    is_disaggregation: bool,
    mm_limits: BTreeMap<String, usize>,
}

impl FrontendHandle {
    pub(crate) fn new(
        intake_tx: flume::Sender<TmEvent>,
        abort_tx: flume::Sender<AbortSource>,
        config: FrontendConfig,
    ) -> Self {
        Self {
            inner: Arc::new(FrontendInner {
                intake_tx,
                abort_tx,
                response_capacity: config.response_capacity,
                response_activity: config.response_activity,
                startup_ready: AtomicBool::new(config.startup_ready),
                is_disaggregation: config.is_disaggregation,
                mm_limits: config.mm_limits,
            }),
        }
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.inner.startup_ready.load(Ordering::Acquire)
    }

    /// Record successful completion of the main process's startup warmup.
    /// The adapter decides which response qualifies as that warmup; the state
    /// itself is shared by every transport.
    pub(crate) fn mark_ready(&self) {
        self.inner.startup_ready.store(true, Ordering::Release);
    }

    pub(crate) fn response_activity(&self) -> u64 {
        self.inner.response_activity.load(Ordering::Relaxed)
    }

    pub(crate) async fn generate(
        &self,
        mut request: GenerateRequest,
    ) -> Result<FrontendCall, FrontendError> {
        self.prepare_generations(std::slice::from_mut(&mut request))
            .await?;
        self.submit(RequestKind::Generate(Box::new(request))).await
    }

    /// Preprocess and submit a group atomically from the adapter's perspective:
    /// no request is admitted until shared multimodal preparation succeeds for
    /// the entire group. If admission later fails part-way through, dropping
    /// the already-created calls aborts those accepted requests.
    pub(crate) async fn generate_batch(
        &self,
        mut requests: Vec<GenerateRequest>,
    ) -> Result<Vec<FrontendCall>, FrontendError> {
        self.prepare_generations(&mut requests).await?;

        let mut calls = Vec::with_capacity(requests.len());
        for request in requests {
            calls.push(
                self.submit(RequestKind::Generate(Box::new(request)))
                    .await?,
            );
        }
        Ok(calls)
    }

    async fn prepare_generations(
        &self,
        requests: &mut [GenerateRequest],
    ) -> Result<(), FrontendError> {
        prefetch::prefetch_all(requests, &self.inner.mm_limits)
            .await
            .map_err(FrontendError::InvalidArgument)
    }

    pub(crate) async fn control(&self, request: ControlRequest) -> Result<Bytes, FrontendError> {
        let mut call = self.submit(RequestKind::Control(Box::new(request))).await?;
        match call.recv().await {
            Some(ResponseItem::Control(bytes)) => Ok(bytes),
            Some(ResponseItem::Error(error)) => Err(FrontendError::Pipeline(error)),
            Some(_) => Err(FrontendError::UnexpectedResponse(
                "unexpected generation output for control request",
            )),
            None => Err(FrontendError::ResponseClosed),
        }
    }

    pub(crate) async fn detokenize(&self, token_ids: TokenIds) -> Result<Bytes, FrontendError> {
        let mut call = self.submit(RequestKind::Detokenize { token_ids }).await?;
        match call.recv().await {
            Some(ResponseItem::Data(bytes)) => Ok(bytes),
            Some(ResponseItem::Error(error)) => Err(FrontendError::Pipeline(error)),
            Some(_) => Err(FrontendError::UnexpectedResponse(
                "unexpected output for detokenize request",
            )),
            None => Err(FrontendError::ResponseClosed),
        }
    }

    /// Confirm that scheduler output is moving. The response heartbeat is the
    /// signal rather than this probe's own result, matching Python's
    /// `last_receive_tstamp` behavior on a busy server.
    pub(crate) async fn probe_health(
        &self,
        timeout: Duration,
    ) -> Result<HealthStatus, FrontendError> {
        if !self.is_ready() {
            return Ok(HealthStatus::NotReady);
        }

        let baseline = self.response_activity();
        let probe = GenerateRequest {
            rid: Rid::new_health_check(),
            input_ids: Some(vec![0]),
            sampling_params: SamplingParams {
                max_new_tokens: Some(1),
                temperature: 0.0,
                ..Default::default()
            },
            stream: false,
            bootstrap_host: self
                .inner
                .is_disaggregation
                .then(|| FAKE_BOOTSTRAP_HOST.into()),
            bootstrap_room: self.inner.is_disaggregation.then_some(0),
            ..Default::default()
        };
        // The probe is intentionally not drained. Its call stays alive while
        // the heartbeat is observed, then aborts on drop to clean up a probe
        // that a busy scheduler skipped without producing a terminal frame.
        let _probe = self.generate(probe).await?;

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.response_activity() != baseline {
                return Ok(HealthStatus::Healthy);
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(HealthStatus::Stalled);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Submit one already-parsed in-process request.
    ///
    /// The async send preserves TokenizerManager inbox backpressure. Once the
    /// request is accepted, the returned call owns cancellation until it
    /// observes a terminal response.
    async fn submit(&self, kind: RequestKind) -> Result<FrontendCall, FrontendError> {
        let rid = request_rid(&kind);
        let expected_response = ExpectedResponse::for_request(&kind);
        let (response_tx, response_rx) = mpsc::channel(self.inner.response_capacity);
        let admission = RequestAdmission::pending();
        // Arm ownership before awaiting the bounded handoff. If this future is
        // cancelled after the event is queued but before the await observes
        // success, dropping `call` still cancels the request.
        let mut call = FrontendCall {
            rid: rid.clone(),
            response_rx,
            abort_tx: self.inner.abort_tx.clone(),
            admission: admission.clone(),
            expected_response,
            in_flight: true,
        };
        let request = Request {
            rid: rid.clone(),
            state: RequestState::Received,
            sink: ResponseSink::Local(response_tx),
            kind,
        };

        if self
            .inner
            .intake_tx
            .send_async(TmEvent::Intake { request, admission })
            .await
            .is_err()
        {
            // The channel returned the event, so Intake cannot own this request.
            // Disarm before returning the transport-neutral rejection.
            call.in_flight = false;
            tracing::error!(%rid, "tm inbox closed; request rejected");
            return Err(FrontendError::Unavailable);
        }

        Ok(call)
    }
}

/// One accepted request and its response stream.
///
/// Dropping an unfinished call either cancels a request that Intake has not yet
/// claimed or notifies the existing abort lane after admission. Reading a
/// terminal item disarms that cleanup automatically, so transports cannot
/// accidentally abort completed work or forget disconnect cleanup.
pub(crate) struct FrontendCall {
    rid: Rid,
    response_rx: mpsc::Receiver<ResponseItem>,
    abort_tx: flume::Sender<AbortSource>,
    admission: RequestAdmission,
    expected_response: ExpectedResponse,
    in_flight: bool,
}

impl FrontendCall {
    pub(crate) fn rid(&self) -> &Rid {
        &self.rid
    }

    pub(crate) async fn recv(&mut self) -> Option<ResponseItem> {
        let item = self.response_rx.recv().await;
        self.observe(item.as_ref());
        item
    }

    pub(crate) fn try_recv(&mut self) -> Result<ResponseItem, mpsc::error::TryRecvError> {
        let item = self.response_rx.try_recv()?;
        self.observe(Some(&item));
        Ok(item)
    }

    fn observe(&mut self, item: Option<&ResponseItem>) {
        if item.is_some_and(|item| self.expected_response.is_terminal(item)) {
            self.in_flight = false;
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_generation_parts(
        rid: Rid,
        response_rx: mpsc::Receiver<ResponseItem>,
        abort_tx: flume::Sender<AbortSource>,
    ) -> Self {
        let admission = RequestAdmission::pending();
        assert!(admission.try_accept(), "test calls start already admitted");
        Self {
            rid,
            response_rx,
            abort_tx,
            admission,
            expected_response: ExpectedResponse::Generate,
            in_flight: true,
        }
    }
}

impl Drop for FrontendCall {
    fn drop(&mut self) {
        if self.in_flight {
            // Pending cancellation is consumed by Intake before it starts work.
            // Once Intake has accepted the request, use the normal abort lane.
            if self.admission.cancel_requires_abort() {
                let _ = self.abort_tx.send(AbortSource::Guard(self.rid.clone()));
            }
            self.in_flight = false;
        }
    }
}

fn request_rid(kind: &RequestKind) -> Rid {
    match kind {
        // Generate IDs are finalized while the transport request is normalized.
        RequestKind::Generate(request) => request.rid.clone(),
        RequestKind::Control(request) => request.rid().into(),
        // Internal service calls are not client-addressable.
        RequestKind::Detokenize { .. } => Rid::new(),
    }
}

#[derive(Clone, Copy)]
enum ExpectedResponse {
    Generate,
    Control,
    Detokenize,
}

impl ExpectedResponse {
    fn for_request(kind: &RequestKind) -> Self {
        match kind {
            RequestKind::Generate(_) => Self::Generate,
            RequestKind::Control(_) => Self::Control,
            RequestKind::Detokenize { .. } => Self::Detokenize,
        }
    }

    fn is_terminal(self, item: &ResponseItem) -> bool {
        matches!(item, ResponseItem::Error(_))
            || matches!(
                (self, item),
                (Self::Generate, ResponseItem::Done(_))
                    | (Self::Control, ResponseItem::Control(_))
                    | (Self::Detokenize, ResponseItem::Data(_))
            )
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bytes::Bytes;

    use super::*;
    use crate::message::io_struct::GetInternalStateReq;
    use crate::message::multimodal::MmItem;
    use crate::message::request::{GenerateRequest, MmData};
    use crate::message::response::{ChunkEvent, SinkError};
    use crate::utils::error::Error;

    struct Harness {
        handle: FrontendHandle,
        intake_rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
    }

    impl Harness {
        fn unbounded(response_capacity: usize) -> Self {
            let (intake_tx, intake_rx) = flume::unbounded();
            let (abort_tx, abort_rx) = flume::unbounded();
            Self {
                handle: test_handle(intake_tx, abort_tx, response_capacity),
                intake_rx,
                abort_rx,
            }
        }
    }

    fn test_handle(
        intake_tx: flume::Sender<TmEvent>,
        abort_tx: flume::Sender<AbortSource>,
        response_capacity: usize,
    ) -> FrontendHandle {
        configured_handle(
            intake_tx,
            abort_tx,
            response_capacity,
            Default::default(),
            false,
            false,
            BTreeMap::new(),
        )
    }

    fn configured_handle(
        intake_tx: flume::Sender<TmEvent>,
        abort_tx: flume::Sender<AbortSource>,
        response_capacity: usize,
        response_activity: ActivityCounter,
        startup_ready: bool,
        is_disaggregation: bool,
        mm_limits: BTreeMap<String, usize>,
    ) -> FrontendHandle {
        FrontendHandle::new(
            intake_tx,
            abort_tx,
            FrontendConfig {
                response_capacity,
                response_activity,
                startup_ready,
                is_disaggregation,
                mm_limits,
            },
        )
    }

    fn generate(rid: &str) -> RequestKind {
        RequestKind::Generate(Box::new(GenerateRequest {
            rid: rid.into(),
            ..Default::default()
        }))
    }

    fn accept_intake(event: TmEvent) -> Request {
        let TmEvent::Intake { request, admission } = event else {
            panic!("expected fresh intake request");
        };
        assert!(admission.try_accept(), "test request should still be live");
        request
    }

    #[tokio::test]
    async fn submit_enqueues_received_request_and_connects_response_sink() {
        let harness = Harness::unbounded(8);
        let mut call = harness.handle.submit(generate("request-1")).await.unwrap();

        let request = accept_intake(harness.intake_rx.recv().unwrap());
        assert_eq!(request.rid.as_str(), "request-1");
        assert!(matches!(request.state, RequestState::Received));
        assert!(matches!(request.kind, RequestKind::Generate(_)));

        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent::default()))
            .unwrap();
        assert!(matches!(call.recv().await, Some(ResponseItem::Done(_))));
        drop(call);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn response_channel_uses_configured_capacity() {
        let harness = Harness::unbounded(1);
        let mut call = harness.handle.submit(generate("bounded")).await.unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());

        request
            .sink
            .try_send(ResponseItem::Frame(ChunkEvent::default()))
            .unwrap();
        assert_eq!(
            request
                .sink
                .try_send(ResponseItem::Frame(ChunkEvent::default())),
            Err(SinkError::Full)
        );
        assert!(matches!(call.recv().await, Some(ResponseItem::Frame(_))));
    }

    #[tokio::test]
    async fn closed_intake_returns_unavailable_without_aborting() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        drop(intake_rx);
        let handle = test_handle(intake_tx, abort_tx, 8);

        let result = handle.submit(generate("rejected")).await;
        assert!(matches!(result, Err(FrontendError::Unavailable)));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn submit_waits_for_intake_capacity() {
        let (intake_tx, intake_rx) = flume::bounded(1);
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = test_handle(intake_tx, abort_tx, 8);

        let first = handle.submit(generate("first")).await.unwrap();
        let second = handle.submit(generate("second"));
        tokio::pin!(second);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut second)
                .await
                .is_err()
        );

        let _ = intake_rx.recv().unwrap();
        let second = tokio::time::timeout(Duration::from_secs(1), &mut second)
            .await
            .expect("submission should resume when intake has capacity")
            .unwrap();
        drop(first);
        drop(second);
    }

    #[tokio::test]
    async fn cancellation_after_intake_acceptance_aborts_unobserved_handoff() {
        let (intake_tx, intake_rx) = flume::bounded(0);
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = test_handle(intake_tx, abort_tx, 8);
        let mut submit = Box::pin(handle.submit(generate("accepted-before-cancel")));

        // Park the send on the rendezvous channel. Receiving the event wakes
        // `submit`, but deliberately do not poll it again to observe success.
        assert!(matches!(
            futures::poll!(submit.as_mut()),
            std::task::Poll::Pending
        ));
        let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap() else {
            panic!("expected fresh intake request");
        };
        assert!(admission.try_accept());

        drop(submit);
        drop(request);
        assert!(matches!(
            abort_rx
                .try_recv()
                .expect("accepted cancellation must emit an abort"),
            AbortSource::Guard(rid) if rid.as_str() == "accepted-before-cancel"
        ));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn cancellation_before_intake_acceptance_suppresses_the_request() {
        let (intake_tx, intake_rx) = flume::bounded(0);
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = test_handle(intake_tx, abort_tx, 8);
        let mut submit = Box::pin(handle.submit(generate("cancelled-before-accept")));

        assert!(matches!(
            futures::poll!(submit.as_mut()),
            std::task::Poll::Pending
        ));
        let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap() else {
            panic!("expected fresh intake request");
        };

        drop(submit);
        assert!(!admission.try_accept());
        drop(request);
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn dropping_live_call_aborts_exactly_once() {
        let harness = Harness::unbounded(8);
        let call = harness.handle.submit(generate("live")).await.unwrap();
        let _request = accept_intake(harness.intake_rx.recv().unwrap());

        drop(call);
        assert!(matches!(
            harness.abort_rx.recv().unwrap(),
            AbortSource::Guard(rid) if rid.as_str() == "live"
        ));
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn terminal_items_disarm_only_the_matching_operation() {
        async fn assert_disarmed(kind: RequestKind, item: ResponseItem) {
            let harness = Harness::unbounded(8);
            let mut call = harness.handle.submit(kind).await.unwrap();
            let request = accept_intake(harness.intake_rx.recv().unwrap());
            request.sink.try_send(item).unwrap();

            let _ = call.recv().await;
            drop(call);
            assert!(harness.abort_rx.try_recv().is_err());
        }

        assert_disarmed(generate("done"), ResponseItem::Done(ChunkEvent::default())).await;
        assert_disarmed(
            generate("failed"),
            ResponseItem::Error(Error::Internal("failed".into())),
        )
        .await;
        assert_disarmed(
            RequestKind::Control(Box::new(ControlRequest::GetInternalStateReq(
                GetInternalStateReq::new("control".into()),
            ))),
            ResponseItem::Control(Bytes::from_static(b"control")),
        )
        .await;
        assert_disarmed(
            RequestKind::Detokenize {
                token_ids: vec![1, 2],
            },
            ResponseItem::Data(Bytes::from_static(b"data")),
        )
        .await;
    }

    #[tokio::test]
    async fn wrong_response_kind_keeps_generation_armed() {
        for item in [
            ResponseItem::Control(Bytes::from_static(b"control")),
            ResponseItem::Data(Bytes::from_static(b"data")),
        ] {
            let harness = Harness::unbounded(8);
            let mut call = harness.handle.submit(generate("wrong-kind")).await.unwrap();
            let request = accept_intake(harness.intake_rx.recv().unwrap());
            request.sink.try_send(item).unwrap();

            let _ = call.recv().await;
            drop(call);
            assert!(matches!(
                harness.abort_rx.recv().unwrap(),
                AbortSource::Guard(rid) if rid.as_str() == "wrong-kind"
            ));
        }
    }

    #[tokio::test]
    async fn nonterminal_frame_keeps_call_armed() {
        let harness = Harness::unbounded(8);
        let mut call = harness.handle.submit(generate("streaming")).await.unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());
        request
            .sink
            .try_send(ResponseItem::Frame(ChunkEvent::default()))
            .unwrap();

        assert!(matches!(call.recv().await, Some(ResponseItem::Frame(_))));
        drop(call);
        assert!(matches!(
            harness.abort_rx.recv().unwrap(),
            AbortSource::Guard(rid) if rid.as_str() == "streaming"
        ));
    }

    #[tokio::test]
    async fn response_channel_close_before_terminal_keeps_call_armed() {
        let harness = Harness::unbounded(8);
        let mut call = harness.handle.submit(generate("truncated")).await.unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());
        drop(request);

        assert!(call.recv().await.is_none());
        drop(call);
        assert!(matches!(
            harness.abort_rx.recv().unwrap(),
            AbortSource::Guard(rid) if rid.as_str() == "truncated"
        ));
    }

    #[tokio::test]
    async fn generate_batch_failure_aborts_already_accepted_calls() {
        let (intake_tx, intake_rx) = flume::bounded(0);
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = test_handle(intake_tx, abort_tx, 8);

        let submitter = tokio::spawn(async move {
            handle
                .generate_batch(vec![
                    GenerateRequest {
                        rid: "accepted".into(),
                        ..Default::default()
                    },
                    GenerateRequest {
                        rid: "rejected".into(),
                        ..Default::default()
                    },
                ])
                .await
        });

        // Accept the first rendezvous, then close the channel before the batch
        // task can submit its second item. Rolling the batch back must abort the
        // request that Intake already claimed.
        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        assert_eq!(request.rid.as_str(), "accepted");
        drop(intake_rx);
        assert!(matches!(
            submitter.await.unwrap(),
            Err(FrontendError::Unavailable)
        ));

        assert!(matches!(
            abort_rx.recv().unwrap(),
            AbortSource::Guard(rid) if rid.as_str() == "accepted"
        ));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn generation_preprocessing_rejects_before_intake() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            false,
            false,
            BTreeMap::from([("image".to_owned(), 1)]),
        );
        let request = GenerateRequest {
            mm: Some(Box::new(MmData {
                image_data: vec![
                    MmItem::Source("/not-read-before-limit-1".into()),
                    MmItem::Source("/not-read-before-limit-2".into()),
                ],
                ..Default::default()
            })),
            ..Default::default()
        };

        assert!(matches!(
            handle.generate(request).await,
            Err(FrontendError::InvalidArgument(message))
                if message == "Image count 2 exceeds limit 1 per request."
        ));
        assert!(intake_rx.try_recv().is_err());
        assert!(abort_rx.try_recv().is_err());
    }

    #[test]
    fn readiness_initialization_and_transition_are_shared() {
        let initially_unready = Harness::unbounded(8).handle;
        let observer = initially_unready.clone();
        assert!(!initially_unready.is_ready());
        initially_unready.mark_ready();
        assert!(observer.is_ready());

        let (intake_tx, _) = flume::unbounded();
        let (abort_tx, _) = flume::unbounded();
        let initially_ready = FrontendHandle::new(
            intake_tx,
            abort_tx,
            FrontendConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: BTreeMap::new(),
            },
        );
        assert!(initially_ready.is_ready());
    }

    #[tokio::test]
    async fn health_probe_reports_not_ready_without_submitting() {
        let harness = Harness::unbounded(8);
        assert_eq!(
            harness
                .handle
                .probe_health(Duration::from_secs(1))
                .await
                .unwrap(),
            HealthStatus::NotReady
        );
        assert!(harness.intake_rx.try_recv().is_err());
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn healthy_probe_uses_shared_activity_and_cleans_up() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let activity: ActivityCounter = Default::default();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            activity.clone(),
            true,
            false,
            BTreeMap::new(),
        );
        let probe = tokio::spawn(async move { handle.probe_health(Duration::from_secs(1)).await });

        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        let rid = request.rid.clone();
        assert!(rid.as_str().starts_with("HEALTH_CHECK_"));
        activity.fetch_add(1, Ordering::Relaxed);

        assert_eq!(probe.await.unwrap().unwrap(), HealthStatus::Healthy);
        assert!(matches!(
            abort_rx.recv().unwrap(),
            AbortSource::Guard(aborted) if aborted == rid
        ));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn stalled_pd_probe_has_bootstrap_pair_and_cleans_up() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            true,
            true,
            BTreeMap::new(),
        );

        let probe_task =
            tokio::spawn(
                async move { handle.probe_health(Duration::from_millis(1)).await.unwrap() },
            );
        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        let rid = request.rid.clone();
        let RequestKind::Generate(probe) = request.kind else {
            panic!("health must submit a generation request");
        };
        assert_eq!(probe.bootstrap_host.as_deref(), Some(FAKE_BOOTSTRAP_HOST));
        assert_eq!(probe.bootstrap_room, Some(0));
        assert_eq!(probe_task.await.unwrap(), HealthStatus::Stalled);
        assert!(matches!(
            abort_rx.recv().unwrap(),
            AbortSource::Guard(aborted) if aborted == rid
        ));
    }

    #[tokio::test]
    async fn health_probe_surfaces_closed_intake_as_operational_error() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        drop(intake_rx);
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            true,
            false,
            BTreeMap::new(),
        );

        assert!(matches!(
            handle.probe_health(Duration::ZERO).await,
            Err(FrontendError::Unavailable)
        ));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn control_operation_returns_transport_neutral_payload() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let task = tokio::spawn(async move {
            handle
                .control(ControlRequest::GetInternalStateReq(
                    GetInternalStateReq::new("control-id".into()),
                ))
                .await
        });

        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert_eq!(request.rid.as_str(), "control-id");
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::from_static(b"payload")))
            .unwrap();
        assert_eq!(task.await.unwrap().unwrap(), Bytes::from_static(b"payload"));
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn detokenize_operation_mints_an_internal_rid() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let task = tokio::spawn(async move { handle.detokenize(vec![1, 2]).await });

        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(!request.rid.as_str().is_empty());
        assert!(matches!(request.kind, RequestKind::Detokenize { .. }));
        request
            .sink
            .try_send(ResponseItem::Data(Bytes::from_static(b"decoded")))
            .unwrap();
        assert_eq!(task.await.unwrap().unwrap(), Bytes::from_static(b"decoded"));
        assert!(harness.abort_rx.try_recv().is_err());
    }
}
