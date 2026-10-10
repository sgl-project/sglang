//! The shared handle: submission with admission and cancellation ownership,
//! runtime-to-semantic event translation, and the typed control operations
//! (model info, server info, detokenize, health probe).

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
use crate::message::request::{GenerateRequest, Request, RequestKind};
use crate::message::response::{ResponseItem, ResponseSink};
use crate::message::sampling::SamplingParams;
use crate::message::types::TokenIds;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission, TmEvent};
use crate::utils::error::Error;
use crate::utils::fsm::RequestState;

use super::{CoreError, CoreEvent, HealthStatus, prefetch};
use crate::message::info::{InternalState, ModelInfo, ServerInfo};

/// Sentinel host that makes the KV connector no-op. Parity with
/// `sglang.srt.disaggregation.utils.FAKE_BOOTSTRAP_HOST`.
const FAKE_BOOTSTRAP_HOST: &str = "2.2.2.2";

pub(crate) struct CoreConfig {
    pub(crate) response_capacity: usize,
    pub(crate) response_activity: ActivityCounter,
    pub(crate) startup_ready: bool,
    pub(crate) is_disaggregation: bool,
    pub(crate) mm_limits: BTreeMap<String, usize>,
    pub(crate) metadata: CoreMetadata,
}

/// Immutable, transport-independent metadata retained by the core.
/// Keeping this snapshot narrow avoids making adapters or the shared handle
/// depend on the full launch-configuration object.
#[derive(Clone, Debug, Default)]
pub(crate) struct CoreMetadata {
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

impl From<&ServerArgs> for CoreMetadata {
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

/// Cloneable capability for submitting work to the runtime pipeline.
///
/// Only the runtime constructs this handle. Protocol adapters receive clones;
/// they cannot reach the raw stage wiring directly.
#[derive(Clone)]
pub(crate) struct CoreHandle {
    inner: Arc<CoreInner>,
}

struct CoreInner {
    intake_tx: flume::Sender<TmEvent>,
    abort_tx: flume::Sender<AbortSource>,
    response_capacity: usize,
    response_activity: ActivityCounter,
    startup_ready: AtomicBool,
    is_disaggregation: bool,
    mm_limits: BTreeMap<String, usize>,
    metadata: CoreMetadata,
}

impl CoreHandle {
    pub(crate) fn new(
        intake_tx: flume::Sender<TmEvent>,
        abort_tx: flume::Sender<AbortSource>,
        config: CoreConfig,
    ) -> Self {
        Self {
            inner: Arc::new(CoreInner {
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

    /// Record successful completion of the startup warmup. The state is
    /// shared by every transport.
    pub(crate) fn mark_ready(&self) {
        self.inner.startup_ready.store(true, Ordering::Release);
    }

    fn response_activity(&self) -> u64 {
        self.inner.response_activity.load(Ordering::Relaxed)
    }

    pub(crate) async fn generate(
        &self,
        mut request: GenerateRequest,
    ) -> Result<CoreCall, CoreError> {
        self.prepare_generations(std::slice::from_mut(&mut request))
            .await?;
        self.submit(RequestKind::Generate(Box::new(request)))
            .await
            .map(CoreCall::new)
    }

    /// Preprocess and submit a group atomically from the adapter's perspective:
    /// no request is admitted until shared multimodal preparation succeeds for
    /// the entire group. If admission later fails part-way through, dropping
    /// the already-created calls aborts those accepted requests.
    pub(crate) async fn generate_batch(
        &self,
        mut requests: Vec<GenerateRequest>,
    ) -> Result<Vec<CoreCall>, CoreError> {
        self.prepare_generations(&mut requests).await?;

        let mut calls = Vec::with_capacity(requests.len());
        for request in requests {
            calls.push(
                self.submit(RequestKind::Generate(Box::new(request)))
                    .await
                    .map(CoreCall::new)?,
            );
        }
        Ok(calls)
    }

    async fn prepare_generations(&self, requests: &mut [GenerateRequest]) -> Result<(), CoreError> {
        prefetch::prefetch_all(requests, &self.inner.mm_limits)
            .await
            .map_err(CoreError::InvalidArgument)
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
    /// cross the core boundary.
    pub(crate) async fn server_info(&self) -> Result<ServerInfo, CoreError> {
        let internal_state = self.internal_state().await?;
        let metadata = &self.inner.metadata;
        Ok(ServerInfo {
            model_path: metadata.model_path.clone(),
            served_model_name: metadata.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            max_context_length: metadata.max_context_length,
            max_total_num_tokens: metadata.max_total_num_tokens,
            version: metadata.version.clone(),
            frontend: "rust",
            internal_states: vec![internal_state],
        })
    }

    async fn internal_state(&self) -> Result<InternalState, CoreError> {
        let bytes = self
            .control(ControlRequest::GetInternalStateReq(
                GetInternalStateReq::new(Rid::new().to_string()),
            ))
            .await?;
        rmp_serde::from_slice::<InternalStateEnvelope>(&bytes)
            .map(|response| response.internal_state)
            .map_err(|error| {
                CoreError::InvalidResponse(format!("invalid internal-state response: {error}"))
            })
    }

    async fn control(&self, request: ControlRequest) -> Result<bytes::Bytes, CoreError> {
        let mut call = self.submit(RequestKind::Control(Box::new(request))).await?;
        match call.recv().await {
            Some(ResponseItem::Control(bytes)) => Ok(bytes),
            Some(ResponseItem::Error(error)) => Err(translate_runtime_error(error)),
            Some(_) => Err(CoreError::InvalidResponse(
                "unexpected generation output for control request".into(),
            )),
            None => Err(CoreError::ResponseTruncated),
        }
    }

    /// Decode token IDs into text without exposing the runtime's byte payload.
    pub(crate) async fn detokenize(&self, token_ids: TokenIds) -> Result<String, CoreError> {
        let mut call = self.submit(RequestKind::Detokenize { token_ids }).await?;
        let bytes = match call.recv().await {
            Some(ResponseItem::Data(bytes)) => bytes,
            Some(ResponseItem::Error(Error::Validation(message))) => {
                return Err(CoreError::InvalidArgument(message));
            }
            Some(ResponseItem::Error(error)) => return Err(translate_runtime_error(error)),
            Some(_) => {
                return Err(CoreError::InvalidResponse(
                    "unexpected output for detokenize request".into(),
                ));
            }
            None => {
                return Err(CoreError::Internal("reply channel closed".into()));
            }
        };
        String::from_utf8(bytes.to_vec())
            .map_err(|_| CoreError::InvalidResponse("detokenized prompt is not valid UTF-8".into()))
    }

    /// Confirm that scheduler output is moving. The response heartbeat is the
    /// signal rather than this probe's own result, matching Python's
    /// `last_receive_tstamp` behavior on a busy server.
    pub(crate) async fn probe_health(&self, timeout: Duration) -> Result<HealthStatus, CoreError> {
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
        let deadline = tokio::time::Instant::now() + timeout;
        let observe_activity = async {
            loop {
                if self.response_activity() != baseline {
                    return HealthStatus::Healthy;
                }
                let now = tokio::time::Instant::now();
                if now >= deadline {
                    return HealthStatus::Stalled;
                }
                tokio::time::sleep_until((now + Duration::from_millis(50)).min(deadline)).await;
            }
        };
        tokio::pin!(observe_activity);

        // Monitor the heartbeat and deadline even while intake is backpressured.
        // Dropping a pending submission cancels it before admission. Once admitted,
        // the undrained call stays alive until health is decided, then aborts on drop.
        let _probe = tokio::select! {
            biased;
            result = self.generate(probe) => result?,
            status = &mut observe_activity => return Ok(status),
        };
        Ok(observe_activity.await)
    }

    /// One generation mirroring the Python `_execute_server_warmup` text request;
    /// VLMs warm only the text path. The caller decides readiness.
    pub(crate) async fn warm_up(&self, skip_tokenizer_init: bool) -> Result<(), CoreError> {
        let (text, input_ids) = if skip_tokenizer_init {
            (None, Some(vec![10, 11, 12]))
        } else {
            (Some("The capital city of France is".to_string()), None)
        };
        let mut call = self
            .generate(GenerateRequest {
                rid: Rid::new(),
                text,
                input_ids,
                sampling_params: SamplingParams {
                    max_new_tokens: Some(8),
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
            })
            .await?;
        loop {
            match call.recv().await {
                Some(CoreEvent::Finished(_)) => return Ok(()),
                Some(CoreEvent::Failed(e)) => return Err(e),
                Some(CoreEvent::Delta(_)) => {}
                None => return Err(CoreError::ResponseTruncated),
            }
        }
    }

    /// Submit one already-parsed in-process request.
    ///
    /// The async send preserves TokenizerManager inbox backpressure. Once the
    /// request is accepted, the returned call owns cancellation until it
    /// observes a terminal response.
    async fn submit(&self, kind: RequestKind) -> Result<RuntimeCall, CoreError> {
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
            return Err(CoreError::Unavailable);
        }

        Ok(call)
    }
}

fn translate_runtime_error(error: Error) -> CoreError {
    match error {
        Error::Validation(message) => {
            CoreError::InvalidArgument(format!("validation failed: {message}"))
        }
        Error::QueueFull => CoreError::Overloaded("to_scheduler channel full".into()),
        Error::Disconnected => CoreError::Cancelled("client disconnected".into()),
        error => CoreError::Internal(error.to_string()),
    }
}

/// One accepted generation request and its semantic response stream.
///
/// Runtime response variants are translated here so adapters cannot depend on
/// scheduler/control encodings. Dropping an unfinished call retains the
/// cancellation guarantees owned by [`RuntimeCall`].
pub(crate) struct CoreCall {
    runtime: RuntimeCall,
    events_finished: bool,
}

impl CoreCall {
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
    pub(crate) async fn recv(&mut self) -> Option<CoreEvent> {
        if self.events_finished {
            return None;
        }

        let event = match self.runtime.recv().await {
            Some(item) => generation_event(item),
            None => CoreEvent::Failed(CoreError::Internal(
                "response truncated before completion".into(),
            )),
        };
        self.events_finished = event.is_terminal();
        Some(event)
    }

    /// Append every semantic event that is ready without waiting. Channel
    /// emptiness and closure remain private runtime details.
    pub(crate) fn drain_ready(&mut self, events: &mut Vec<CoreEvent>) {
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

/// Await the next event from `call`, then drain whatever queued behind it (so the
/// caller can coalesce a backlog, as Python's `state.out_list` does), handing the
/// call back for `FuturesUnordered` to re-poll. An empty result means the semantic
/// stream was already exhausted. Shared by the HTTP and gRPC batch multiplexers.
pub(crate) async fn recv_indexed(
    index: usize,
    mut call: CoreCall,
) -> (usize, CoreCall, Vec<CoreEvent>) {
    let mut items = Vec::new();
    match call.recv().await {
        Some(item) => items.push(item),
        None => return (index, call, items), // already exhausted
    }
    call.drain_ready(&mut items);
    (index, call, items)
}

fn generation_event(item: ResponseItem) -> CoreEvent {
    match item {
        ResponseItem::Frame(output) => CoreEvent::Delta(output.into()),
        ResponseItem::Done(output) => {
            if let Some((legacy_http_status, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                CoreEvent::Failed(CoreError::from_runtime_rejection(
                    message.to_owned(),
                    legacy_http_status,
                ))
            } else {
                CoreEvent::Finished(output.into())
            }
        }
        ResponseItem::Error(error) => CoreEvent::Failed(translate_runtime_error(error)),
        ResponseItem::Control(_) | ResponseItem::Data(_) => CoreEvent::Failed(
            CoreError::InvalidResponse("unexpected non-generation response".into()),
        ),
    }
}

/// Private scheduler response envelope. This MessagePack shape is decoded at
/// the runtime boundary and deliberately is not part of the core contract.
#[derive(serde::Deserialize)]
struct InternalStateEnvelope {
    #[serde(default)]
    internal_state: InternalState,
}

/// Runtime-facing request guard kept private behind [`CoreCall`].
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
    use crate::api_server::core::{CoreErrorKind, CoreOutput};
    use crate::message::io_struct::GetInternalStateReq;
    use crate::message::multimodal::MmItem;
    use crate::message::request::MmData;
    use crate::message::response::ChunkEvent;

    struct Harness {
        handle: CoreHandle,
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
    ) -> CoreHandle {
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
                metadata: CoreMetadata::default(),
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
            CoreError::InvalidArgument(message)
                if message == "validation failed: bad input"
        ));
        assert!(matches!(
            translate_runtime_error(Error::QueueFull),
            CoreError::Overloaded(message) if message == "to_scheduler channel full"
        ));
        assert!(matches!(
            translate_runtime_error(Error::Disconnected),
            CoreError::Cancelled(message) if message == "client disconnected"
        ));

        assert!(matches!(
            translate_runtime_error(Error::Internal("bug".into())),
            CoreError::Internal(message) if message == "internal error: bug"
        ));
    }

    #[test]
    fn scheduler_status_is_classified_once_for_non_http_adapters() {
        for (status, expected) in [
            (400, CoreErrorKind::InvalidArgument),
            (404, CoreErrorKind::NotFound),
            (408, CoreErrorKind::DeadlineExceeded),
            (412, CoreErrorKind::FailedPrecondition),
            (413, CoreErrorKind::ResourceExhausted),
            (429, CoreErrorKind::ResourceExhausted),
            (432, CoreErrorKind::InvalidArgument),
            (499, CoreErrorKind::Cancelled),
            (500, CoreErrorKind::Internal),
            (503, CoreErrorKind::Unavailable),
            (504, CoreErrorKind::DeadlineExceeded),
        ] {
            let error = CoreError::from_runtime_rejection("rejected".into(), status);
            assert_eq!(error.kind(), expected, "legacy status {status}");
            assert!(matches!(
                error,
                CoreError::RuntimeRejected {
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
            Some(CoreEvent::Delta(CoreOutput { text, .. })) if text == "delta"
        ));
        assert!(matches!(
            call.recv().await,
            Some(CoreEvent::Finished(CoreOutput { text, .. })) if text == "final"
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
            Some(CoreEvent::Finished(CoreOutput {
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
            Err(CoreError::Unavailable)
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
            Err(CoreError::InvalidArgument(message))
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
    async fn health_timeout_includes_pending_admission() {
        let (intake_tx, intake_rx) = flume::bounded(0);
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            true,
            false,
            BTreeMap::new(),
        );
        let mut probe = Box::pin(handle.probe_health(Duration::from_millis(1)));
        let result = match futures::poll!(&mut probe) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => {
                tokio::time::sleep(Duration::from_millis(10)).await;
                futures::FutureExt::now_or_never(probe)
                    .expect("health timeout must include waiting for intake capacity")
            }
        };
        assert_eq!(result.unwrap(), HealthStatus::Stalled);
        assert!(intake_rx.try_recv().is_err());
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn healthy_activity_does_not_wait_for_probe_admission() {
        let (intake_tx, intake_rx) = flume::bounded(0);
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
        let mut probe = Box::pin(handle.probe_health(Duration::from_secs(1)));
        assert!(matches!(
            futures::poll!(&mut probe),
            std::task::Poll::Pending
        ));

        activity.fetch_add(1, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(60)).await;
        let result = futures::FutureExt::now_or_never(probe).expect(
            "scheduler activity must keep a busy server healthy during intake backpressure",
        );
        assert_eq!(result.unwrap(), HealthStatus::Healthy);
        assert!(intake_rx.try_recv().is_err());
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

        let probe_task = tokio::spawn(async move {
            handle
                .probe_health(Duration::from_millis(50))
                .await
                .unwrap()
        });
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
    async fn warm_up_succeeds_on_finished_generation() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let warm_up = tokio::spawn(async move { handle.warm_up(false).await });

        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        let RequestKind::Generate(warmup) = &request.kind else {
            panic!("warmup must submit a generation request");
        };
        assert_eq!(
            warmup.text.as_deref(),
            Some("The capital city of France is")
        );
        assert_eq!(warmup.input_ids, None);
        assert_eq!(warmup.sampling_params.max_new_tokens, Some(8));
        assert_eq!(warmup.bootstrap_host, None);
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent::default()))
            .unwrap();

        assert!(warm_up.await.unwrap().is_ok());
        assert!(!harness.handle.is_ready());
    }

    #[tokio::test]
    async fn pd_warm_up_without_tokenizer_fails_on_truncated_stream() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            false,
            true,
            BTreeMap::new(),
        );
        let warm_up = tokio::spawn(async move { handle.warm_up(true).await });

        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        let RequestKind::Generate(warmup) = &request.kind else {
            panic!("warmup must submit a generation request");
        };
        assert_eq!(warmup.text, None);
        assert_eq!(warmup.input_ids, Some(vec![10, 11, 12]));
        assert_eq!(warmup.bootstrap_host.as_deref(), Some(FAKE_BOOTSTRAP_HOST));
        assert_eq!(warmup.bootstrap_room, Some(0));
        drop(request);

        assert!(warm_up.await.unwrap().is_err());
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
