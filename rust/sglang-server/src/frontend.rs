//! Transport-neutral entry point into the Rust frontend pipeline.
//!
//! Wire adapters normalize their protocol into [`FrontendRequest`], submit it
//! through [`FrontendHandle`], and render semantic [`FrontendEvent`]s for their
//! own transport. This module owns shared preprocessing, runtime translation,
//! capabilities, and request lifetime; it deliberately knows nothing about
//! Axum, HTTP response shapes, Tonic, or protobuf.

use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::message::config::{DisaggregationMode, ServerArgs};
use crate::message::ids::Rid;
use crate::message::io_struct::{ControlRequest, GetInternalStateReq};
use crate::message::request::{Request, RequestKind};
use crate::message::response::{ResponseItem, ResponseSink};
use crate::message::sampling::SamplingParams;
use crate::message::types::TokenIds;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission, TmEvent};
use crate::utils::error::Error;
use crate::utils::fsm::RequestState;

mod contract;
mod prefetch;

pub(crate) use contract::{
    FrontendError, FrontendErrorKind, FrontendEvent, FrontendOutput, FrontendRequest, HealthStatus,
    InternalState, ModelInfo, ServerInfo,
};

/// Sentinel host that makes the KV connector no-op. Parity with
/// `sglang.srt.disaggregation.utils.FAKE_BOOTSTRAP_HOST`.
const FAKE_BOOTSTRAP_HOST: &str = "2.2.2.2";

pub(crate) struct FrontendConfig {
    pub(crate) response_capacity: usize,
    pub(crate) response_activity: ActivityCounter,
    pub(crate) startup_ready: bool,
    pub(crate) is_disaggregation: bool,
    pub(crate) mm_limits: BTreeMap<String, usize>,
    pub(crate) metadata: FrontendMetadata,
}

/// Immutable, transport-independent metadata retained by the frontend.
/// Keeping this snapshot narrow avoids making adapters or the shared handle
/// depend on the full launch-configuration object.
#[derive(Clone, Debug, Default)]
pub(crate) struct FrontendMetadata {
    model_path: String,
    served_model_name: String,
    tokenizer_path: String,
    preferred_sampling_params: Option<crate::message::config::PreferredSamplingParams>,
    weight_version: Option<String>,
    load_format: Option<String>,
    reasoning_parser: Option<String>,
    tool_call_parser: Option<String>,
    disaggregation_mode: DisaggregationMode,
    max_context_length: u64,
    max_total_num_tokens: u64,
    version: String,
}

impl From<&ServerArgs> for FrontendMetadata {
    fn from(args: &ServerArgs) -> Self {
        Self {
            model_path: args.model_path.clone(),
            served_model_name: args.served_model_name.clone(),
            tokenizer_path: args.tokenizer_path.clone(),
            preferred_sampling_params: args.preferred_sampling_params.clone(),
            weight_version: args.weight_version.clone(),
            load_format: args.load_format.clone(),
            reasoning_parser: args.reasoning_parser.clone(),
            tool_call_parser: args.tool_call_parser.clone(),
            disaggregation_mode: args.disaggregation_mode,
            max_context_length: args.model_config.context_len,
            max_total_num_tokens: args.max_total_num_tokens,
            version: args.version.clone(),
        }
    }
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
    metadata: FrontendMetadata,
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
                metadata: config.metadata,
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

    fn response_activity(&self) -> u64 {
        self.inner.response_activity.load(Ordering::Relaxed)
    }

    pub(crate) async fn generate(
        &self,
        mut request: FrontendRequest,
    ) -> Result<FrontendCall, FrontendError> {
        self.prepare_generations(std::slice::from_mut(&mut request))
            .await?;
        self.submit(RequestKind::Generate(Box::new(request)))
            .await
            .map(FrontendCall::new)
    }

    /// Preprocess and submit a group atomically from the adapter's perspective:
    /// no request is admitted until shared multimodal preparation succeeds for
    /// the entire group. If admission later fails part-way through, dropping
    /// the already-created calls aborts those accepted requests.
    pub(crate) async fn generate_batch(
        &self,
        mut requests: Vec<FrontendRequest>,
    ) -> Result<Vec<FrontendCall>, FrontendError> {
        self.prepare_generations(&mut requests).await?;

        let mut calls = Vec::with_capacity(requests.len());
        for request in requests {
            calls.push(
                self.submit(RequestKind::Generate(Box::new(request)))
                    .await
                    .map(FrontendCall::new)?,
            );
        }
        Ok(calls)
    }

    async fn prepare_generations(
        &self,
        requests: &mut [FrontendRequest],
    ) -> Result<(), FrontendError> {
        prefetch::prefetch_all(requests, &self.inner.mm_limits)
            .await
            .map_err(FrontendError::InvalidArgument)
    }

    /// Return static model metadata without exposing the full launch config.
    pub(crate) fn model_info(&self) -> ModelInfo {
        let metadata = &self.inner.metadata;
        ModelInfo {
            model_path: metadata.model_path.clone(),
            served_model_name: metadata.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            is_generation: true,
            preferred_sampling_params: metadata.preferred_sampling_params.clone(),
            weight_version: metadata.weight_version.clone(),
            load_format: metadata.load_format.clone(),
            reasoning_parser: metadata.reasoning_parser.clone(),
            tool_call_parser: metadata.tool_call_parser.clone(),
            disaggregation_mode: metadata.disaggregation_mode,
        }
    }

    /// Return public server metadata with the current scheduler metrics.
    /// Raw control bytes and the scheduler's full launch-argument dump never
    /// cross the frontend boundary.
    pub(crate) async fn server_info(&self) -> Result<ServerInfo, FrontendError> {
        let internal_state = self.internal_state().await?;
        let metadata = &self.inner.metadata;
        Ok(ServerInfo {
            model_path: metadata.model_path.clone(),
            served_model_name: metadata.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            max_context_length: metadata.max_context_length,
            max_total_num_tokens: metadata.max_total_num_tokens,
            version: metadata.version.clone(),
            internal_states: vec![internal_state],
        })
    }

    async fn internal_state(&self) -> Result<InternalState, FrontendError> {
        let bytes = self
            .control(ControlRequest::GetInternalStateReq(
                GetInternalStateReq::new(Rid::new().to_string()),
            ))
            .await?;
        rmp_serde::from_slice::<InternalStateEnvelope>(&bytes)
            .map(|response| response.internal_state)
            .map_err(|error| {
                FrontendError::InvalidResponse(format!("invalid internal-state response: {error}"))
            })
    }

    async fn control(&self, request: ControlRequest) -> Result<bytes::Bytes, FrontendError> {
        let mut call = self.submit(RequestKind::Control(Box::new(request))).await?;
        match call.recv().await {
            Some(ResponseItem::Control(bytes)) => Ok(bytes),
            Some(ResponseItem::Error(error)) => Err(translate_runtime_error(error)),
            Some(_) => Err(FrontendError::InvalidResponse(
                "unexpected generation output for control request".into(),
            )),
            None => Err(FrontendError::ResponseTruncated),
        }
    }

    /// Decode token IDs into text without exposing the runtime's byte payload.
    pub(crate) async fn detokenize(&self, token_ids: TokenIds) -> Result<String, FrontendError> {
        let mut call = self.submit(RequestKind::Detokenize { token_ids }).await?;
        let bytes = match call.recv().await {
            Some(ResponseItem::Data(bytes)) => bytes,
            Some(ResponseItem::Error(Error::Validation(message))) => {
                return Err(FrontendError::InvalidArgument(message));
            }
            Some(ResponseItem::Error(error)) => return Err(translate_runtime_error(error)),
            Some(_) => {
                return Err(FrontendError::InvalidResponse(
                    "unexpected output for detokenize request".into(),
                ));
            }
            None => {
                return Err(FrontendError::Internal("reply channel closed".into()));
            }
        };
        String::from_utf8(bytes.to_vec()).map_err(|_| {
            FrontendError::InvalidResponse("detokenized prompt is not valid UTF-8".into())
        })
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
        let probe = FrontendRequest {
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
    async fn submit(&self, kind: RequestKind) -> Result<RuntimeCall, FrontendError> {
        let rid = request_rid(&kind);
        let expected_response = ExpectedResponse::for_request(&kind);
        let (response_tx, response_rx) = mpsc::channel(self.inner.response_capacity);
        let admission = RequestAdmission::pending();
        // Arm ownership before awaiting the bounded handoff. If this future is
        // cancelled after the event is queued but before the await observes
        // success, dropping `call` still cancels the request.
        let mut call = RuntimeCall {
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

fn translate_runtime_error(error: Error) -> FrontendError {
    match error {
        Error::Validation(message) => {
            FrontendError::InvalidArgument(format!("validation failed: {message}"))
        }
        Error::QueueFull => FrontendError::Overloaded("to_scheduler channel full".into()),
        Error::Disconnected => FrontendError::Cancelled("client disconnected".into()),
        error => FrontendError::Internal(error.to_string()),
    }
}

/// One accepted generation request and its semantic response stream.
///
/// Runtime response variants are translated here so adapters cannot depend on
/// scheduler/control encodings. Dropping an unfinished call retains the
/// cancellation guarantees owned by [`RuntimeCall`].
pub(crate) struct FrontendCall {
    runtime: RuntimeCall,
    events_finished: bool,
}

impl FrontendCall {
    fn new(runtime: RuntimeCall) -> Self {
        Self {
            runtime,
            events_finished: false,
        }
    }

    /// ID that adapters should return to their client.
    pub(crate) fn public_id(&self) -> &str {
        self.runtime.rid.client_facing()
    }

    /// Receive the next semantic event. A runtime channel that closes before a
    /// terminal response is converted into one terminal failure event; adapters
    /// never need to infer whether an empty runtime channel means success.
    pub(crate) async fn recv(&mut self) -> Option<FrontendEvent> {
        if self.events_finished {
            return None;
        }

        let event = match self.runtime.recv().await {
            Some(item) => generation_event(item),
            None => FrontendEvent::Failed(FrontendError::Internal(
                "response truncated before completion".into(),
            )),
        };
        self.events_finished = event.is_terminal();
        Some(event)
    }

    /// Append every semantic event that is ready without waiting. Channel
    /// emptiness and closure remain private runtime details.
    pub(crate) fn drain_ready(&mut self, events: &mut Vec<FrontendEvent>) {
        while !self.events_finished {
            let event = match self.runtime.try_recv() {
                Ok(item) => generation_event(item),
                // Preserve every queued event before reporting a premature
                // close. The next awaited `recv` translates that close into
                // one terminal failure, maintaining stream order while still
                // hiding channel mechanics from adapters.
                Err(mpsc::error::TryRecvError::Disconnected | mpsc::error::TryRecvError::Empty) => {
                    break;
                }
            };
            self.events_finished = event.is_terminal();
            events.push(event);
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
        Self::new(RuntimeCall {
            rid,
            response_rx,
            abort_tx,
            admission,
            expected_response: ExpectedResponse::Generate,
            in_flight: true,
        })
    }
}

fn generation_event(item: ResponseItem) -> FrontendEvent {
    match item {
        ResponseItem::Frame(output) => FrontendEvent::Delta(output.into()),
        ResponseItem::Done(output) => {
            if let Some((legacy_http_status, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                FrontendEvent::Failed(FrontendError::from_runtime_rejection(
                    message.to_owned(),
                    legacy_http_status,
                ))
            } else {
                FrontendEvent::Finished(output.into())
            }
        }
        ResponseItem::Error(error) => FrontendEvent::Failed(translate_runtime_error(error)),
        ResponseItem::Control(_) | ResponseItem::Data(_) => FrontendEvent::Failed(
            FrontendError::InvalidResponse("unexpected non-generation response".into()),
        ),
    }
}

/// Private scheduler response envelope. This MessagePack shape is decoded at
/// the runtime boundary and deliberately is not part of the frontend contract.
#[derive(serde::Deserialize)]
struct InternalStateEnvelope {
    #[serde(default)]
    internal_state: InternalState,
}

/// Runtime-facing request guard kept private behind [`FrontendCall`].
///
/// This type is the only owner of the raw response channel and admission/abort
/// capabilities. It is also used directly by typed unary operations such as
/// detokenization and internal-state lookup.
struct RuntimeCall {
    rid: Rid,
    response_rx: mpsc::Receiver<ResponseItem>,
    abort_tx: flume::Sender<AbortSource>,
    admission: RequestAdmission,
    expected_response: ExpectedResponse,
    in_flight: bool,
}

impl RuntimeCall {
    async fn recv(&mut self) -> Option<ResponseItem> {
        let item = self.response_rx.recv().await;
        self.observe(item.as_ref());
        item
    }

    fn try_recv(&mut self) -> Result<ResponseItem, mpsc::error::TryRecvError> {
        let item = self.response_rx.try_recv()?;
        self.observe(Some(&item));
        Ok(item)
    }

    fn observe(&mut self, item: Option<&ResponseItem>) {
        if item.is_some_and(|item| self.expected_response.is_terminal(item)) {
            self.in_flight = false;
        }
    }
}

impl Drop for RuntimeCall {
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
    use crate::message::response::ChunkEvent;

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
                metadata: FrontendMetadata::default(),
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

    #[test]
    fn runtime_errors_become_transport_neutral_categories() {
        assert!(matches!(
            translate_runtime_error(Error::Validation("bad input".into())),
            FrontendError::InvalidArgument(message)
                if message == "validation failed: bad input"
        ));
        assert!(matches!(
            translate_runtime_error(Error::QueueFull),
            FrontendError::Overloaded(message) if message == "to_scheduler channel full"
        ));
        assert!(matches!(
            translate_runtime_error(Error::Disconnected),
            FrontendError::Cancelled(message) if message == "client disconnected"
        ));

        assert!(matches!(
            translate_runtime_error(Error::Internal("bug".into())),
            FrontendError::Internal(message) if message == "internal error: bug"
        ));
    }

    #[test]
    fn scheduler_status_is_classified_once_for_non_http_adapters() {
        for (status, expected) in [
            (400, FrontendErrorKind::InvalidArgument),
            (404, FrontendErrorKind::NotFound),
            (408, FrontendErrorKind::DeadlineExceeded),
            (412, FrontendErrorKind::FailedPrecondition),
            (413, FrontendErrorKind::ResourceExhausted),
            (429, FrontendErrorKind::ResourceExhausted),
            (432, FrontendErrorKind::InvalidArgument),
            (499, FrontendErrorKind::Cancelled),
            (500, FrontendErrorKind::Internal),
            (503, FrontendErrorKind::Unavailable),
            (504, FrontendErrorKind::DeadlineExceeded),
        ] {
            let error = FrontendError::from_runtime_rejection("rejected".into(), status);
            assert_eq!(error.kind(), expected, "legacy status {status}");
            assert!(matches!(
                error,
                FrontendError::RuntimeRejected {
                    legacy_http_status,
                    ..
                } if legacy_http_status == status
            ));
        }
    }

    #[tokio::test]
    async fn generation_exposes_public_id_and_semantic_events() {
        let harness = Harness::unbounded(8);
        let mut call = harness
            .handle
            .generate(GenerateRequest {
                rid: Rid::from_client("public-id"),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(call.public_id(), "public-id");
        assert_ne!(call.runtime.rid.as_str(), "public-id");

        let request = accept_intake(harness.intake_rx.recv().unwrap());
        request
            .sink
            .try_send(ResponseItem::Frame(ChunkEvent {
                text: "delta".into(),
                ..Default::default()
            }))
            .unwrap();
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent {
                text: "final".into(),
                ..Default::default()
            }))
            .unwrap();

        assert!(matches!(
            call.recv().await,
            Some(FrontendEvent::Delta(FrontendOutput { text, .. })) if text == "delta"
        ));
        assert!(matches!(
            call.recv().await,
            Some(FrontendEvent::Finished(FrontendOutput { text, .. })) if text == "final"
        ));
        assert!(call.recv().await.is_none());
        drop(call);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn plain_scheduler_abort_remains_a_terminal_output() {
        let harness = Harness::unbounded(8);
        let mut call = harness
            .handle
            .generate(GenerateRequest {
                rid: "plain-abort".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent {
                finish_reason: serde_json::from_value(serde_json::json!({
                    "type": "abort",
                    "message": "Aborted",
                    "status_code": null,
                    "err_type": null
                }))
                .unwrap(),
                ..Default::default()
            }))
            .unwrap();

        assert!(matches!(
            call.recv().await,
            Some(FrontendEvent::Finished(FrontendOutput {
                finish_reason: Some(_),
                ..
            }))
        ));
        assert!(call.recv().await.is_none());
        drop(call);
        assert!(harness.abort_rx.try_recv().is_err());
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
        assert_eq!(task.await.unwrap().unwrap(), "decoded");
        assert!(harness.abort_rx.try_recv().is_err());
    }
}
