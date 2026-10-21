//! The shared handle: submission with admission and cancellation ownership,
//! runtime-to-semantic event translation, and the typed control operations
//! (model info, model list, server info, flush cache, weights check, tokenize,
//! detokenize, health probe).

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use sglang_api_types::api::v1 as api;
use tokio::sync::{mpsc, oneshot};

use crate::message::config::{DisaggregationMode, ServerArgs};
use crate::message::ids::Rid;
use crate::message::io_struct::{
    CheckWeightsReqInput, ControlRequest, FlushCacheReqInput, GetInternalStateReq,
};
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
    pub(crate) mm_limits: Vec<(String, usize)>,
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
    host: String,
    port: u16,
    incremental_streaming_output: bool,
    deployment_info: crate::message::config::DeploymentInfo,
    launch_server_info: Arc<serde_json::Map<String, serde_json::Value>>,
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
            host: args.host.clone(),
            port: args.port,
            incremental_streaming_output: args.incremental_streaming_output,
            deployment_info: args.deployment_info.clone(),
            launch_server_info: args.launch_server_info.0.clone(),
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
    mm_limits: Vec<(String, usize)>,
    metadata: CoreMetadata,
    generations: Generations,
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
                generations: Generations::default(),
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
        let mut generation = self
            .inner
            .generations
            .register(request.rid.clone(), self.inner.abort_tx.clone());
        tokio::select! {
            biased;
            _ = generation.cancelled() => return Err(CoreError::Cancelled("request aborted".into())),
            result = self.prepare_generations(std::slice::from_mut(&mut request)) => result?,
        }
        self.submit_generation(request, generation)
            .await
            .map(CoreCall::new)
    }

    /// Submit core-owned work without exposing it through the public Abort
    /// registry. Drop still owns cancellation through `RuntimeCall`.
    async fn generate_internal(&self, mut request: GenerateRequest) -> Result<CoreCall, CoreError> {
        self.prepare_generations(std::slice::from_mut(&mut request))
            .await?;
        self.submit(RequestKind::Generate(Box::new(request)))
            .await
            .map(CoreCall::new)
    }

    /// Prepare the group before submitting any surviving request. Explicit
    /// cancellation produces only that item's terminal result, preserving batch
    /// order. Other preparation/admission failures still reject the group and
    /// dropping already-created calls aborts any accepted requests.
    pub(crate) async fn generate_batch(
        &self,
        mut requests: Vec<GenerateRequest>,
    ) -> Result<Vec<CoreCall>, CoreError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let mut generations: Vec<_> = requests
            .iter()
            .map(|request| {
                self.inner
                    .generations
                    .register(request.rid.clone(), self.inner.abort_tx.clone())
            })
            .collect();
        // Fetch concurrently, but stop only the cancelled item's preparation.
        // Per-item cancellation lets try_join_all use wake-driven polling for
        // large batches, without rescanning every receiver on each media wake.
        futures::future::try_join_all(requests.iter_mut().zip(&mut generations).map(
            |(request, generation)| async move {
                tokio::select! {
                    biased;
                    _ = generation.cancelled() => Ok(()),
                    result = self.prepare_generations(std::slice::from_mut(request)) => result,
                }
            },
        ))
        .await?;

        let mut calls = Vec::with_capacity(requests.len());
        for (request, generation) in requests.into_iter().zip(generations) {
            let rid = request.rid.clone();
            let call = match self.submit_generation(request, generation).await {
                Ok(call) => CoreCall::new(call),
                // Cancellation may happen during preparation or while waiting
                // for intake. Keep this slot without rolling back siblings.
                Err(CoreError::Cancelled(_)) => {
                    CoreCall::cancelled(rid, self.inner.abort_tx.clone())
                }
                Err(error) => return Err(error),
            };
            calls.push(call);
        }
        Ok(calls)
    }

    /// Cancel active generations by public ID, or all active generations.
    /// The admission gate and internal IDs keep cancellation safe across
    /// preprocessing, queueing, duplicate public IDs, and ID reuse.
    pub(crate) fn abort(&self, rid: Option<&str>) -> bool {
        self.inner.generations.abort(rid, &self.inner.abort_tx)
    }

    async fn submit_generation(
        &self,
        request: GenerateRequest,
        generation: Generation,
    ) -> Result<RuntimeCall, CoreError> {
        self.submit_with_generation(RequestKind::Generate(Box::new(request)), Some(generation))
            .await
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

    /// Return the served model cards shared by HTTP and gRPC discovery.
    /// This is static metadata; no runtime request is submitted.
    pub(crate) fn list_models(&self) -> Vec<api::ModelCard> {
        let metadata = &self.inner.metadata;
        vec![api::ModelCard {
            id: metadata.served_model_name.clone(),
            root: metadata.served_model_name.clone(),
            parent: None,
            max_model_len: metadata.max_context_length,
        }]
    }

    /// Return public server metadata with the current scheduler metrics.
    /// Raw control bytes and the scheduler's full launch-argument dump never
    /// cross the core boundary.
    pub(crate) async fn server_info(&self) -> Result<ServerInfo, CoreError> {
        let mut internal_state = self.internal_state().await?;
        let metadata = &self.inner.metadata;
        Ok(ServerInfo {
            launch_fields: metadata.launch_server_info.clone(),
            model_path: metadata.model_path.clone(),
            served_model_name: metadata.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            max_context_length: metadata.max_context_length,
            max_total_num_tokens: metadata.max_total_num_tokens,
            version: metadata.version.clone(),
            frontend: "rust",
            startup_time: internal_state.rust_server_startup_time.take(),
            internal_states: vec![internal_state],
            host: metadata.host.clone(),
            port: metadata.port,
            incremental_streaming_output: metadata.incremental_streaming_output,
            disaggregation_mode: metadata.disaggregation_mode,
            deployment_info: metadata.deployment_info.clone(),
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

    /// Ask the scheduler to flush its caches, waiting up to `timeout_s` for it
    /// to go idle. The scheduler runs its own flush handler in its own loop;
    /// this only relays the request and its outcome.
    pub(crate) async fn flush_cache(&self, timeout_s: f64) -> Result<FlushCacheOutcome, CoreError> {
        let bytes = self
            .control(ControlRequest::FlushCacheReqInput(FlushCacheReqInput::new(
                Rid::new().to_string(),
                Some(timeout_s),
            )))
            .await?;
        rmp_serde::from_slice::<FlushCacheOutcome>(&bytes).map_err(|error| {
            CoreError::InvalidResponse(format!("invalid flush-cache response: {error}"))
        })
    }

    /// Ask every scheduler rank to run the weight checker's `action` on the
    /// runners `selector` names, and relay each rank's checksum record.
    pub(crate) async fn check_weights(
        &self,
        action: String,
        allow_quant_error: bool,
        skip_tensor_list: Option<Vec<String>>,
        selector: String,
    ) -> Result<CheckWeightsOutcome, CoreError> {
        let bytes = self
            .control(ControlRequest::CheckWeightsReqInput(
                CheckWeightsReqInput::new(
                    Rid::new().to_string(),
                    action,
                    allow_quant_error,
                    skip_tensor_list,
                    selector,
                ),
            ))
            .await?;
        rmp_serde::from_slice::<CheckWeightsOutcome>(&bytes).map_err(|error| {
            CoreError::InvalidResponse(format!("invalid weights-check response: {error}"))
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
    pub(crate) async fn detokenize(
        &self,
        token_ids: TokenIds,
        skip_special_tokens: bool,
    ) -> Result<String, CoreError> {
        let mut call = self
            .submit(RequestKind::Detokenize {
                token_ids,
                skip_special_tokens,
            })
            .await?;
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

    /// Encode one text into token IDs on the tokenizer pool.
    pub(crate) async fn tokenize(
        &self,
        text: String,
        add_special_tokens: bool,
    ) -> Result<TokenIds, CoreError> {
        let mut call = self
            .submit(RequestKind::Tokenize {
                text,
                add_special_tokens,
            })
            .await?;
        match call.recv().await {
            Some(ResponseItem::TokenIds(ids)) => Ok(ids),
            Some(ResponseItem::Error(Error::Validation(message))) => {
                Err(CoreError::InvalidArgument(message))
            }
            Some(ResponseItem::Error(error)) => Err(translate_runtime_error(error)),
            Some(_) => Err(CoreError::InvalidResponse(
                "unexpected output for tokenize request".into(),
            )),
            None => Err(CoreError::Internal("reply channel closed".into())),
        }
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
        // The probe is intentionally not drained. Its call stays alive while
        // the heartbeat is observed, then aborts on drop to clean up a probe
        // that a busy scheduler skipped without producing a terminal frame.
        let _probe = self.generate_internal(probe).await?;

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

    /// One generation mirroring the Python `_execute_server_warmup` text request;
    /// VLMs warm only the text path. The caller decides readiness.
    pub(crate) async fn warm_up(&self, skip_tokenizer_init: bool) -> Result<(), CoreError> {
        let (text, input_ids) = if skip_tokenizer_init {
            (None, Some(vec![10, 11, 12]))
        } else {
            (Some("The capital city of France is".to_string()), None)
        };
        let mut call = self
            .generate_internal(GenerateRequest {
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
        self.submit_with_generation(kind, None).await
    }

    async fn submit_with_generation(
        &self,
        kind: RequestKind,
        generation: Option<Generation>,
    ) -> Result<RuntimeCall, CoreError> {
        let rid = request_rid(&kind);
        let expected_response = ExpectedResponse::for_request(&kind);
        let (response_tx, response_rx) = mpsc::channel(self.inner.response_capacity);
        let admission = generation
            .as_ref()
            .map(|g| g.admission.clone())
            .unwrap_or_else(RequestAdmission::pending);
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
            generation,
        };
        let request = Request {
            rid: rid.clone(),
            state: RequestState::Received,
            sink: ResponseSink::Local(response_tx),
            kind,
        };

        let sent = tokio::select! {
            biased;
            _ = async {
                match call.generation.as_mut() {
                    Some(generation) => generation.cancelled().await,
                    None => std::future::pending().await,
                }
            } => return Err(CoreError::Cancelled("request aborted".into())),
            result = self.inner.intake_tx.send_async(TmEvent::Intake { request, admission }) => result,
        };
        if sent.is_err() {
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

/// One generation's semantic response stream, including pre-admission cancellation.
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

    /// Retain a cancelled batch item's ID and position without runtime work.
    fn cancelled(rid: Rid, abort_tx: flume::Sender<AbortSource>) -> Self {
        let (response_tx, response_rx) = mpsc::channel(1);
        response_tx
            .try_send(ResponseItem::Error(Error::Disconnected))
            .expect("new response channel has room for its terminal error");
        Self::new(RuntimeCall {
            rid,
            response_rx,
            abort_tx,
            admission: RequestAdmission::pending(),
            expected_response: ExpectedResponse::Generate,
            in_flight: false,
            generation: None,
        })
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
            generation: None,
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
        ResponseItem::Control(_) | ResponseItem::Data(_) | ResponseItem::TokenIds(_) => {
            CoreEvent::Failed(CoreError::InvalidResponse(
                "unexpected non-generation response".into(),
            ))
        }
    }
}

/// The scheduler's `FlushCacheReqOutput`: whether the flush ran, and why not.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct FlushCacheOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

/// The scheduler's `CheckWeightsReqOutput`: whether every rank succeeded, why
/// not, and one checksum record per rank (`None` when no rank produced one).
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct CheckWeightsOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
    #[serde(default)]
    pub(crate) payload: Option<Vec<serde_json::Map<String, serde_json::Value>>>,
}

/// Private scheduler response envelope. This MessagePack shape is decoded at
/// the runtime boundary and deliberately is not part of the core contract.
#[derive(serde::Deserialize)]
struct InternalStateEnvelope {
    #[serde(default)]
    internal_state: InternalState,
}

type Entries = HashMap<String, HashMap<Rid, Entry>>;

/// Public-ID lookup for explicit cancellation. Internal IDs remain unique;
/// this index never replaces the runtime's request identity or admission gate.
#[derive(Clone, Default)]
struct Generations(Arc<Mutex<Entries>>);

struct Entry {
    admission: RequestAdmission,
    cancelled: oneshot::Sender<()>,
}

impl Generations {
    fn register(&self, rid: Rid, abort_tx: flume::Sender<AbortSource>) -> Generation {
        let admission = RequestAdmission::pending();
        let (cancelled, cancellation) = oneshot::channel();
        self.0
            .lock()
            .unwrap()
            .entry(rid.client_facing().into())
            .or_default()
            .insert(
                rid.clone(),
                Entry {
                    admission: admission.clone(),
                    cancelled,
                },
            );
        Generation {
            registry: self.clone(),
            rid,
            admission,
            cancellation,
            abort_tx,
            armed: true,
        }
    }

    fn abort(&self, rid: Option<&str>, abort_tx: &flume::Sender<AbortSource>) -> bool {
        // Detach a snapshot under the lock, then notify outside it. Completion
        // and a new request reusing the same public ID cannot remove each other.
        let selected = {
            let mut entries = self.0.lock().unwrap();
            match rid {
                Some(rid) => entries.remove(rid).into_iter().collect::<Vec<_>>(),
                None => std::mem::take(&mut *entries).into_values().collect(),
            }
        };
        let mut matched = false;
        for entries in selected {
            for (rid, entry) in entries {
                matched = true;
                // Stop backend work immediately, even if the client is not
                // polling its response stream. The same admission gate also
                // prevents a queued request from starting after cancellation.
                if entry.admission.cancel_requires_abort() {
                    let _ = abort_tx.send(AbortSource::Guard(rid));
                }
                let _ = entry.cancelled.send(());
            }
        }
        matched
    }

    fn remove(&self, rid: &Rid) {
        let mut entries = self.0.lock().unwrap();
        if let Some(group) = entries.get_mut(rid.client_facing()) {
            group.remove(rid);
            if group.is_empty() {
                entries.remove(rid.client_facing());
            }
        }
    }
}

/// Lives from preprocessing through the terminal response. Drop owns both
/// removal from the public-ID index and cancellation of unfinished work.
struct Generation {
    registry: Generations,
    rid: Rid,
    admission: RequestAdmission,
    cancellation: oneshot::Receiver<()>,
    abort_tx: flume::Sender<AbortSource>,
    armed: bool,
}

impl Generation {
    async fn cancelled(&mut self) {
        // Preparation and submission may both observe the same cancellation;
        // do not poll a oneshot receiver again after consuming its notification.
        if !self.is_cancelled() {
            let _ = (&mut self.cancellation).await;
        }
    }

    fn is_cancelled(&self) -> bool {
        self.admission.is_cancelled()
    }

    fn complete(&mut self) {
        self.admission.complete();
        self.registry.remove(&self.rid);
        self.armed = false;
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        if self.armed {
            self.registry.remove(&self.rid);
            if self.admission.cancel_requires_abort() {
                let _ = self.abort_tx.send(AbortSource::Guard(self.rid.clone()));
            }
        }
    }
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
    generation: Option<Generation>,
}

impl RuntimeCall {
    async fn recv(&mut self) -> Option<ResponseItem> {
        let item = tokio::select! {
            biased;
            _ = async {
                match self.generation.as_mut() {
                    Some(generation) => generation.cancelled().await,
                    None => std::future::pending().await,
                }
            } => Some(ResponseItem::Error(Error::Disconnected)),
            item = self.response_rx.recv() => item,
        };
        self.observe(item.as_ref());
        item
    }

    fn try_recv(&mut self) -> Result<ResponseItem, mpsc::error::TryRecvError> {
        let item = if self
            .generation
            .as_ref()
            .is_some_and(Generation::is_cancelled)
        {
            ResponseItem::Error(Error::Disconnected)
        } else {
            self.response_rx.try_recv()?
        };
        self.observe(Some(&item));
        Ok(item)
    }

    fn observe(&mut self, item: Option<&ResponseItem>) {
        if item.is_some_and(|item| self.expected_response.is_terminal(item)) {
            self.in_flight = false;
            if let Some(generation) = &mut self.generation {
                generation.complete();
            }
        }
    }
}

impl Drop for RuntimeCall {
    fn drop(&mut self) {
        if self.in_flight && self.generation.is_none() {
            // Pending cancellation is consumed by Intake before it starts work.
            // Once Intake has accepted the request, use the normal abort lane.
            if self.admission.cancel_requires_abort() {
                let rid = self.rid.clone();
                let source = match self.expected_response {
                    ExpectedResponse::Tokenize | ExpectedResponse::Detokenize => {
                        AbortSource::Local(rid)
                    }
                    ExpectedResponse::Generate | ExpectedResponse::Control => {
                        AbortSource::Guard(rid)
                    }
                };
                let _ = self.abort_tx.send(source);
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
        RequestKind::Detokenize { .. } | RequestKind::Tokenize { .. } => Rid::new(),
    }
}

#[derive(Clone, Copy)]
enum ExpectedResponse {
    Generate,
    Control,
    Detokenize,
    Tokenize,
}

impl ExpectedResponse {
    fn for_request(kind: &RequestKind) -> Self {
        match kind {
            RequestKind::Generate(_) => Self::Generate,
            RequestKind::Control(_) => Self::Control,
            RequestKind::Detokenize { .. } => Self::Detokenize,
            RequestKind::Tokenize { .. } => Self::Tokenize,
        }
    }

    fn is_terminal(self, item: &ResponseItem) -> bool {
        matches!(item, ResponseItem::Error(_))
            || matches!(
                (self, item),
                (Self::Generate, ResponseItem::Done(_))
                    | (Self::Control, ResponseItem::Control(_))
                    | (Self::Detokenize, ResponseItem::Data(_))
                    | (Self::Tokenize, ResponseItem::TokenIds(_))
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
    use crate::message::response::{ChunkEvent, frame_health_ack};
    use crate::runtime::Runnable;
    use crate::tokenizer_manager::from_scheduler::Dispatcher;
    use crate::tokenizer_manager::wiring::Senders;

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
            Vec::new(),
        )
    }

    fn configured_handle(
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

    #[test]
    fn list_models_returns_static_cards_without_runtime_work() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: false,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                metadata: CoreMetadata {
                    served_model_name: "served-model".into(),
                    max_context_length: 1_u64 << 32,
                    ..Default::default()
                },
            },
        );

        assert_eq!(
            handle.list_models(),
            vec![api::ModelCard {
                id: "served-model".into(),
                root: "served-model".into(),
                parent: None,
                max_model_len: 1_u64 << 32,
            }]
        );
        assert!(intake_rx.try_recv().is_err());
        assert!(abort_rx.try_recv().is_err());
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
            vec![("image".to_owned(), 1)],
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

    /// A busy scheduler skips the probe and acknowledges it after its next step.
    /// The dispatcher counts the acknowledgement as scheduler output, so the
    /// probe passes without any batch frame.
    #[tokio::test]
    async fn skipped_probe_passes_on_health_ack_and_cleans_up() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let activity: ActivityCounter = Default::default();
        let (from_scheduler_tx, from_scheduler_rx) =
            crate::tokenizer_manager::channel::from_scheduler(8);
        let (detok_tx, detok_rx) = flume::unbounded();
        let (_shutdown_tx, shutdown_rx) = flume::bounded(1);
        let dispatcher = Dispatcher::new(
            from_scheduler_rx,
            Senders {
                tok_manager_tx: flume::unbounded().0,
                abort_tx: flume::unbounded().0,
                tokenizer_tx: flume::unbounded().0,
                detokenizer_tx: vec![detok_tx],
            },
            activity.clone(),
            shutdown_rx,
        );
        let dispatcher = std::thread::spawn(move || dispatcher.run());
        let handle = configured_handle(intake_tx, abort_tx, 8, activity, true, false, Vec::new());
        let probe_handle = handle.clone();
        let probe =
            tokio::spawn(async move { probe_handle.probe_health(Duration::from_secs(1)).await });

        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        let rid = request.rid.clone();
        assert!(rid.as_str().starts_with("HEALTH_CHECK_"));
        assert!(
            !handle.abort(None),
            "internal probes must not enter the public abort-all registry"
        );
        assert!(from_scheduler_tx.push(frame_health_ack()));

        assert_eq!(probe.await.unwrap().unwrap(), HealthStatus::Healthy);
        assert!(matches!(
            abort_rx.recv().unwrap(),
            AbortSource::Guard(aborted) if aborted == rid
        ));
        assert!(abort_rx.try_recv().is_err());
        drop(from_scheduler_tx);
        dispatcher.join().unwrap();
        assert!(
            detok_rx.try_recv().is_err(),
            "an acknowledgement routes nothing"
        );
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
            Vec::new(),
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
        assert!(
            !harness.handle.abort(None),
            "startup warmup must not enter the public abort-all registry"
        );
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
            Vec::new(),
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
    async fn token_operations_mint_internal_rids() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let task = tokio::spawn(async move {
            handle
                .detokenize(vec![1, 2], false /* skip_special_tokens */)
                .await
        });

        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(!request.rid.as_str().is_empty());
        assert!(matches!(
            request.kind,
            RequestKind::Detokenize {
                skip_special_tokens: false,
                ..
            }
        ));
        request
            .sink
            .try_send(ResponseItem::Data(Bytes::from_static(b"decoded")))
            .unwrap();
        assert_eq!(task.await.unwrap().unwrap(), "decoded");

        let handle = harness.handle.clone();
        let task = tokio::spawn(async move {
            handle
                .tokenize("hi".into(), true /* add_special_tokens */)
                .await
        });
        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(!request.rid.as_str().is_empty());
        assert!(matches!(
            &request.kind,
            RequestKind::Tokenize {
                text,
                add_special_tokens: true,
            } if text == "hi"
        ));
        request
            .sink
            .try_send(ResponseItem::TokenIds(vec![1, 2]))
            .unwrap();
        assert_eq!(task.await.unwrap().unwrap(), vec![1, 2]);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    fn public_request(rid: &str) -> GenerateRequest {
        GenerateRequest {
            rid: Rid::from_client(rid),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn explicit_abort_during_batch_preprocessing_preserves_other_items() {
        for aborted in ["A", "B"] {
            let harness = Harness::unbounded(8);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = ["A", "B"]
                .into_iter()
                .map(|rid| {
                    let mut request = public_request(rid);
                    if rid == aborted {
                        request.mm = Some(Box::new(MmData {
                            image_data: vec![MmItem::Source(format!("http://{addr}/blocked.png"))],
                            ..Default::default()
                        }));
                    }
                    request
                })
                .collect();
            let mut batch = Box::pin(harness.handle.generate_batch(requests));
            assert!(futures::poll!(batch.as_mut()).is_pending());
            // Hold the media response open so cancellation happens during
            // preprocessing, not after the request reaches intake.
            let (media, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
                .await
                .unwrap()
                .unwrap();
            assert!(harness.intake_rx.try_recv().is_err());
            assert!(harness.handle.abort(Some(aborted)));
            let mut calls = tokio::time::timeout(Duration::from_secs(5), batch)
                .await
                .expect("cancelled media must not hold up its sibling")
                .unwrap();
            drop(media);

            assert_eq!(
                calls.iter().map(CoreCall::public_id).collect::<Vec<_>>(),
                ["A", "B"]
            );
            let request = accept_intake(harness.intake_rx.try_recv().unwrap());
            assert_ne!(request.rid.client_facing(), aborted);
            assert!(harness.intake_rx.try_recv().is_err());
            request
                .sink
                .try_send(ResponseItem::Done(ChunkEvent::default()))
                .unwrap();
            for call in &mut calls {
                let was_aborted = call.public_id() == aborted;
                let event = call.recv().await;
                if was_aborted {
                    assert!(matches!(
                        event,
                        Some(CoreEvent::Failed(CoreError::Cancelled(_)))
                    ));
                } else {
                    assert!(matches!(event, Some(CoreEvent::Finished(_))));
                }
                assert!(call.recv().await.is_none());
            }
            assert!(!harness.handle.abort(None));
            assert!(harness.abort_rx.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn explicit_abort_during_batch_admission_preserves_other_items() {
        let (intake_tx, intake_rx) = flume::bounded(0);
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = test_handle(intake_tx, abort_tx, 8);
        let mut batch = Box::pin(
            handle.generate_batch(["A", "B", "C"].into_iter().map(public_request).collect()),
        );
        assert!(futures::poll!(batch.as_mut()).is_pending());
        let first = accept_intake(intake_rx.recv_async().await.unwrap());
        assert_eq!(first.rid.client_facing(), "A");

        // A is accepted; B is blocked on intake; C has not been submitted yet.
        assert!(futures::poll!(batch.as_mut()).is_pending());
        assert!(handle.abort(Some("B")));
        assert!(futures::poll!(batch.as_mut()).is_pending());
        let last = accept_intake(intake_rx.recv_async().await.unwrap());
        assert_eq!(last.rid.client_facing(), "C");
        let mut calls = batch.await.unwrap();
        assert_eq!(
            calls.iter().map(CoreCall::public_id).collect::<Vec<_>>(),
            ["A", "B", "C"]
        );
        assert!(intake_rx.try_recv().is_err());
        assert!(abort_rx.try_recv().is_err(), "A must remain active");

        for request in [first, last] {
            request
                .sink
                .try_send(ResponseItem::Done(ChunkEvent::default()))
                .unwrap();
        }
        for (index, call) in calls.iter_mut().enumerate() {
            let event = call.recv().await;
            if index == 1 {
                assert!(matches!(
                    event,
                    Some(CoreEvent::Failed(CoreError::Cancelled(_)))
                ));
            } else {
                assert!(matches!(event, Some(CoreEvent::Finished(_))));
            }
            assert!(call.recv().await.is_none());
        }
        assert!(!handle.abort(None));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn explicit_abort_cancels_duplicates_but_not_later_id_reuse() {
        let harness = Harness::unbounded(8);
        let mut first = harness
            .handle
            .generate(public_request("same"))
            .await
            .unwrap();
        let mut second = harness
            .handle
            .generate(public_request("same"))
            .await
            .unwrap();
        let old_a = accept_intake(harness.intake_rx.recv().unwrap());
        let old_b = accept_intake(harness.intake_rx.recv().unwrap());
        assert!(harness.handle.abort(Some("same")));
        assert!(!harness.handle.abort(Some("same")));
        // Backend cancellation does not depend on polling either response.
        let aborted: std::collections::HashSet<_> = (0..2)
            .map(|_| harness.abort_rx.try_recv().unwrap().rid().clone())
            .collect();
        assert_eq!(aborted, [old_a.rid, old_b.rid].into_iter().collect());

        let new = harness
            .handle
            .generate(public_request("same"))
            .await
            .unwrap();
        let new_request = accept_intake(harness.intake_rx.recv().unwrap());
        for call in [&mut first, &mut second] {
            assert!(matches!(
                call.recv().await,
                Some(CoreEvent::Failed(CoreError::Cancelled(_)))
            ));
            assert!(call.recv().await.is_none());
        }
        drop((first, second));
        assert!(harness.abort_rx.try_recv().is_err());
        assert!(harness.handle.abort(Some("same")));
        assert_eq!(harness.abort_rx.try_recv().unwrap().rid(), &new_request.rid);
        drop(new);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn explicit_abort_before_admission_prevents_intake_acceptance() {
        let harness = Harness::unbounded(8);
        let mut call = harness
            .handle
            .generate(public_request("queued"))
            .await
            .unwrap();
        assert!(harness.handle.abort(Some("queued")));
        let TmEvent::Intake { admission, .. } = harness.intake_rx.recv().unwrap() else {
            panic!("expected intake");
        };
        assert!(!admission.try_accept());
        assert!(matches!(
            call.recv().await,
            Some(CoreEvent::Failed(CoreError::Cancelled(_)))
        ));
        drop(call);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn explicit_abort_wakes_a_submission_blocked_on_admission() {
        let (intake_tx, intake_rx) = flume::bounded(0);
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = test_handle(intake_tx, abort_tx, 8);
        let mut submit = Box::pin(handle.generate(public_request("blocked")));
        assert!(futures::poll!(submit.as_mut()).is_pending());
        assert!(handle.abort(Some("blocked")));
        assert!(matches!(submit.await, Err(CoreError::Cancelled(_))));
        if let Ok(TmEvent::Intake { admission, .. }) = intake_rx.try_recv() {
            assert!(!admission.try_accept());
        }
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn completed_and_dropped_calls_are_removed_from_abort_lookup() {
        let harness = Harness::unbounded(8);
        let mut call = harness
            .handle
            .generate(public_request("done"))
            .await
            .unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent::default()))
            .unwrap();
        assert!(matches!(call.recv().await, Some(CoreEvent::Finished(_))));
        assert!(!harness.handle.abort(Some("done")));
        drop(call);
        assert!(harness.abort_rx.try_recv().is_err());

        let call = harness
            .handle
            .generate(public_request("dropped"))
            .await
            .unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());
        drop(call);
        assert_eq!(harness.abort_rx.try_recv().unwrap().rid(), &request.rid);
        assert!(!harness.handle.abort(Some("dropped")));
        assert!(!harness.handle.abort(None));
    }

    #[tokio::test]
    async fn abort_all_excludes_detokenization_and_preserves_future_requests() {
        let harness = Harness::unbounded(8);
        let mut decode = Box::pin(harness.handle.detokenize(vec![1], true));
        assert!(futures::poll!(decode.as_mut()).is_pending());
        let decoder = accept_intake(harness.intake_rx.recv().unwrap());
        let a = harness.handle.generate(public_request("a")).await.unwrap();
        let b = harness.handle.generate(public_request("b")).await.unwrap();
        let req_a = accept_intake(harness.intake_rx.recv().unwrap());
        let req_b = accept_intake(harness.intake_rx.recv().unwrap());
        assert!(harness.handle.abort(None));
        let aborted: std::collections::HashSet<_> = (0..2)
            .map(|_| harness.abort_rx.try_recv().unwrap().rid().clone())
            .collect();
        assert_eq!(aborted, [req_a.rid, req_b.rid].into_iter().collect());
        decoder
            .sink
            .try_send(ResponseItem::Data(Bytes::from_static(b"text")))
            .unwrap();
        assert_eq!(decode.await.unwrap(), "text");
        drop((a, b));
        assert!(harness.abort_rx.try_recv().is_err());
        assert!(!harness.handle.abort(None));
        let future = harness
            .handle
            .generate(public_request("future"))
            .await
            .unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());
        drop(future);
        assert_eq!(harness.abort_rx.try_recv().unwrap().rid(), &request.rid);
    }

    #[tokio::test]
    async fn abort_is_observed_when_draining_buffered_frames() {
        let harness = Harness::unbounded(8);
        let mut call = harness
            .handle
            .generate(public_request("buffered"))
            .await
            .unwrap();
        let request = accept_intake(harness.intake_rx.recv().unwrap());
        request
            .sink
            .try_send(ResponseItem::Frame(ChunkEvent::default()))
            .unwrap();
        assert!(harness.handle.abort(Some("buffered")));
        let mut events = Vec::new();
        call.drain_ready(&mut events);
        assert!(matches!(
            events.as_slice(),
            [CoreEvent::Failed(CoreError::Cancelled(_))]
        ));
        assert!(call.recv().await.is_none());
    }
}
