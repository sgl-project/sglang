//! The shared handle: its construction, submission with admission and
//! cancellation ownership, and the runtime-call guard behind every operation.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use tokio::sync::{SetOnce, mpsc};

use crate::message::config::{DisaggregationMode, ServerArgs};
use crate::message::finish_reason::{AbortReason, FinishKind};
use crate::message::ids::Rid;
use crate::message::request::{Request, RequestKind};
use crate::message::response::{
    CachedTokensDetails, ChunkEvent, ChunkExtras, ResponseItem, ResponseSink,
};
use crate::multi_modality::worker::MmProcessor;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission, TmEvent};
use crate::utils::error::Error;
use crate::utils::fsm::RequestState;

use super::control::PEER_ABORT_MAX_IN_FLIGHT;
use super::generate::{Cancellation, Generation, Generations};
use crate::api_server::core::fanout::{ControlTarget, RankId};
use crate::api_server::core::{CoreError, DrainEnd, DrainState, FrontendMetrics};

/// How long the listeners keep open connections after a drain's requests
/// finished: enough to flush the last responses and close idle connections.
pub(crate) const GRACEFUL_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) struct CoreConfig {
    pub(crate) response_capacity: usize,
    pub(crate) response_activity: ActivityCounter,
    pub(crate) startup_ready: bool,
    pub(crate) is_disaggregation: bool,
    pub(crate) mm_limits: Vec<(String, usize)>,
    /// A multimodal model admits no generation until its MM pool's processor
    /// is installed ([`CoreHandle::set_mm_processor`]).
    pub(crate) multimodal: bool,
    pub(crate) metadata: CoreMetadata,
    /// Request metrics, when `--enable-metrics` is on.
    pub(crate) metrics: Option<Arc<FrontendMetrics>>,
    /// The targets answering for the DP ranks this process does not host,
    /// which every multi-rank control operation fans out to alongside the
    /// local scheduler (see [`fanout`]). Empty means this process answers
    /// for its own rank only.
    pub(crate) peer_endpoints: Vec<Arc<dyn ControlTarget>>,
}

/// Immutable, transport-independent metadata retained by the core.
/// Keeping this snapshot narrow avoids making adapters or the shared handle
/// depend on the full launch-configuration object.
#[derive(Clone, Debug, Default)]
pub(crate) struct CoreMetadata {
    pub(super) model_path: String,
    pub(super) served_model_name: String,
    pub(super) tokenizer_path: String,
    pub(super) preferred_sampling_params: Option<crate::message::config::PreferredSamplingParams>,
    pub(super) weight_version: Option<Arc<str>>,
    pub(super) load_format: Option<String>,
    pub(super) reasoning_parser: Option<String>,
    pub(super) tool_call_parser: Option<String>,
    pub(super) disaggregation_mode: DisaggregationMode,
    pub(super) max_context_length: u64,
    pub(super) max_total_num_tokens: u64,
    pub(super) version: String,
    pub(super) host: String,
    pub(super) port: u16,
    pub(super) incremental_streaming_output: bool,
    pub(super) deployment_info: crate::message::config::DeploymentInfo,
    pub(super) launch_server_info: Arc<serde_json::Map<String, serde_json::Value>>,
    pub(super) launch_model_info: Arc<serde_json::Map<String, serde_json::Value>>,
    /// Priority for a generation that sets none (see
    /// `ServerArgs::default_priority_value`).
    pub(super) default_priority: Option<i64>,
}

impl From<&ServerArgs> for CoreMetadata {
    fn from(args: &ServerArgs) -> Self {
        Self {
            model_path: args.model_path.clone(),
            served_model_name: args.served_model_name.clone(),
            tokenizer_path: args.public_tokenizer_path.clone(),
            preferred_sampling_params: args.preferred_sampling_params.clone(),
            weight_version: args.weight_version.as_deref().map(Arc::from),
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
            launch_model_info: args.launch_model_info.0.clone(),
            default_priority: args.default_priority_value,
        }
    }
}

/// Cloneable capability for submitting work to the runtime pipeline.
///
/// Only the runtime constructs this handle. Protocol adapters receive clones;
/// they cannot reach the raw stage wiring directly.
#[derive(Clone)]
pub(crate) struct CoreHandle {
    pub(super) inner: Arc<CoreInner>,
}

pub(super) struct CoreInner {
    pub(super) intake_tx: flume::Sender<TmEvent>,
    pub(super) abort_tx: flume::Sender<AbortSource>,
    pub(super) response_capacity: usize,
    pub(super) response_activity: ActivityCounter,
    pub(super) startup_ready: AtomicBool,
    pub(super) is_disaggregation: bool,
    pub(super) mm_limits: Vec<(String, usize)>,
    pub(super) multimodal: bool,
    /// The MM pool's processor, whose media hooks admission runs; set once
    /// when the pool starts (after the listeners are up), and awaited by a
    /// multimodal model's admission.
    pub(super) mm_processor: SetOnce<Arc<dyn MmProcessor>>,
    pub(super) metadata: CoreMetadata,
    pub(super) generations: Generations,
    pub(super) metrics: Option<Arc<FrontendMetrics>>,
    /// The DP rank the local scheduler answers for, and the targets for the rest.
    pub(super) dp_rank: RankId,
    pub(super) peer_endpoints: Vec<Arc<dyn ControlTarget>>,
    /// Shared by every transport (see [`CoreHandle::begin_drain`]).
    pub(super) drain: Arc<DrainState>,
    /// Overrides [`GRACEFUL_CLOSE_TIMEOUT`] (see
    /// [`CoreHandle::set_graceful_close_timeout`]).
    pub(super) graceful_close_timeout: OnceLock<Duration>,
    /// Relayed aborts in flight at once, across every abort (see
    /// [`CoreHandle::abort_peers`]).
    pub(super) peer_abort_permits: tokio::sync::Semaphore,
}

impl CoreHandle {
    pub(crate) fn new(
        intake_tx: flume::Sender<TmEvent>,
        abort_tx: flume::Sender<AbortSource>,
        config: CoreConfig,
    ) -> Self {
        let dp_rank = config.metadata.deployment_info.dp_rank.unwrap_or(0);
        Self {
            inner: Arc::new(CoreInner {
                intake_tx,
                abort_tx,
                response_capacity: config.response_capacity,
                response_activity: config.response_activity,
                startup_ready: AtomicBool::new(config.startup_ready),
                is_disaggregation: config.is_disaggregation,
                mm_limits: config.mm_limits,
                multimodal: config.multimodal,
                mm_processor: SetOnce::new(),
                metadata: config.metadata,
                generations: Generations::default(),
                metrics: config.metrics,
                dp_rank,
                peer_endpoints: config.peer_endpoints,
                drain: Arc::default(),
                graceful_close_timeout: OnceLock::new(),
                peer_abort_permits: tokio::sync::Semaphore::new(PEER_ABORT_MAX_IN_FLIGHT),
            }),
        }
    }

    /// Run `processor`'s media hooks during admission. The first processor
    /// installed wins; `false` if one already was.
    pub(crate) fn set_mm_processor(&self, processor: Arc<dyn MmProcessor>) -> bool {
        self.inner.mm_processor.set(processor).is_ok()
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.inner.startup_ready.load(Ordering::Acquire)
    }

    /// Whether health may answer 200: startup warmup finished (or was
    /// skipped) and, for a multimodal model, the MM pool's processor is
    /// installed, so a generation would be admitted rather than wait.
    pub(crate) fn is_serving(&self) -> bool {
        self.is_ready() && (!self.inner.multimodal || self.inner.mm_processor.get().is_some())
    }

    /// The request metrics, when enabled.
    pub(crate) fn metrics(&self) -> Option<&Arc<FrontendMetrics>> {
        self.inner.metrics.as_ref()
    }

    /// Record successful completion of the startup warmup. The state is
    /// shared by every transport.
    pub(crate) fn mark_ready(&self) {
        self.inner.startup_ready.store(true, Ordering::Release);
    }

    /// Begin the graceful drain (idempotent): health answers 503 from now
    /// on, requests are still admitted, and [`CoreHandle::drain_ended`]
    /// resolves once none is left in flight. `force` skips that wait.
    pub(crate) fn begin_drain(&self, force: bool) {
        self.inner.drain.begin(force);
    }

    /// Set the graceful-close timeout once, at boot; the first value wins.
    pub(crate) fn set_graceful_close_timeout(&self, timeout: Duration) {
        let _ = self.inner.graceful_close_timeout.set(timeout);
    }

    pub(crate) fn is_draining(&self) -> bool {
        self.inner.drain.is_draining()
    }

    /// Resolve once the drain has ended (see [`DrainState::ended`]); the
    /// listeners then close.
    pub(crate) async fn drain_ended(&self) -> DrainEnd {
        self.inner.drain.ended().await
    }

    /// Resolve once the drain must stop waiting for in-flight requests.
    pub(crate) async fn drain_forced(&self) {
        self.inner.drain.forced().await;
    }

    /// Resolve once the listeners must stop waiting for open connections:
    /// when the drain is forced, or the graceful-close timeout
    /// ([`GRACEFUL_CLOSE_TIMEOUT`] by default) after its requests drained. A
    /// request that raced the close, or a client that never acknowledges the
    /// HTTP/2 shutdown, must not hold the scheduler's shutdown.
    pub(crate) async fn drain_close_deadline(&self) {
        tokio::select! {
            () = self.drain_forced() => {}
            () = async {
                self.drain_ended().await;
                let timeout = self
                    .inner
                    .graceful_close_timeout
                    .get()
                    .copied()
                    .unwrap_or(GRACEFUL_CLOSE_TIMEOUT);
                tokio::time::sleep(timeout).await;
            } => {}
        }
    }

    /// Why the drain skipped its wait, in Python's words; `None` if it did not.
    pub(crate) fn drain_force_reason(&self) -> Option<&'static str> {
        self.inner.drain.force_reason()
    }

    pub(super) fn response_activity(&self) -> u64 {
        self.inner.response_activity.load(Ordering::Relaxed)
    }

    /// Submit one already-parsed in-process request.
    ///
    /// The async send preserves TokenizerManager inbox backpressure. Once the
    /// request is accepted, the returned call owns cancellation until it
    /// observes a terminal response.
    pub(super) async fn submit(&self, kind: RequestKind) -> Result<RuntimeCall, CoreError> {
        self.submit_with_generation(kind, None).await
    }

    pub(super) async fn submit_with_generation(
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
            last_output: LastOutput::default(),
            abort_deadline: None,
        };
        let request = Request {
            rid: rid.clone(),
            state: RequestState::Received,
            sink: ResponseSink::Local(response_tx),
            kind,
        };

        let sent = tokio::select! {
            biased;
            cancellation = async {
                match call.generation.as_mut() {
                    Some(generation) => generation.cancelled().await,
                    None => std::future::pending().await,
                }
            } => match cancellation {
                Cancellation::Error => {
                    return Err(CoreError::Cancelled("request aborted".into()));
                }
                // The request never starts; the call yields its abort output.
                Cancellation::Unstarted => {
                    call.in_flight = false;
                    return Ok(call);
                }
                // Intake took the event before the abort ran: the request is
                // sent, and the call ends on the scheduler's answer.
                Cancellation::Scheduler => Ok(()),
            },
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

pub(super) fn translate_runtime_error(error: Error) -> CoreError {
    match error {
        Error::Validation(message) => {
            CoreError::InvalidArgument(format!("validation failed: {message}"))
        }
        Error::Encode(error) => error.into(),
        Error::QueueFull => CoreError::Overloaded("to_scheduler channel full".into()),
        Error::Disconnected => CoreError::Cancelled("client disconnected".into()),
        error => CoreError::Internal(error.to_string()),
    }
}

/// Runtime-facing request guard kept private behind [`CoreCall`].
///
/// This type is the only owner of the raw response channel and admission/abort
/// capabilities. It is also used directly by typed unary operations such as
/// detokenization and internal-state lookup.
pub(super) struct RuntimeCall {
    pub(super) rid: Rid,
    pub(super) response_rx: mpsc::Receiver<ResponseItem>,
    pub(super) abort_tx: flume::Sender<AbortSource>,
    pub(super) admission: RequestAdmission,
    pub(super) expected_response: ExpectedResponse,
    pub(super) in_flight: bool,
    pub(super) generation: Option<Generation>,
    /// What the request's latest output reported, for a frontend abort output.
    pub(super) last_output: LastOutput,
    /// Set once the scheduler was asked to abort the request: when the call
    /// stops waiting for the scheduler's terminal output.
    pub(super) abort_deadline: Option<tokio::time::Instant>,
}

/// How long a call that `/abort_request` handed to the scheduler waits for
/// the scheduler's terminal output. The scheduler answers at its next step;
/// the bound only keeps a lost answer from holding the client forever.
const SCHEDULER_ABORT_DEADLINE: Duration = Duration::from_secs(30);

/// The request-level counts and cache split every scheduler output reports
/// in full, which a frontend abort output repeats.
#[derive(Clone, Default)]
pub(super) struct LastOutput {
    prompt_tokens: u32,
    cached_tokens: u32,
    reasoning_tokens: u64,
    cached_tokens_details: Option<CachedTokensDetails>,
}

/// What woke a call waiting for its next response.
enum Wake {
    Cancelled(Cancellation),
    Item(Option<ResponseItem>),
    AbortDeadline,
}

impl RuntimeCall {
    pub(super) async fn recv(&mut self) -> Option<ResponseItem> {
        let mut item = loop {
            let abort_deadline = self.abort_deadline;
            let wake = tokio::select! {
                biased;
                cancellation = async {
                    match self.generation.as_mut() {
                        Some(generation) => generation.cancelled().await,
                        None => std::future::pending().await,
                    }
                }, if abort_deadline.is_none() => Wake::Cancelled(cancellation),
                item = self.response_rx.recv() => Wake::Item(item),
                () = async {
                    match abort_deadline {
                        Some(deadline) => tokio::time::sleep_until(deadline).await,
                        None => std::future::pending().await,
                    }
                } => Wake::AbortDeadline,
            };
            match wake {
                Wake::Cancelled(Cancellation::Scheduler) => {
                    self.abort_deadline =
                        Some(tokio::time::Instant::now() + SCHEDULER_ABORT_DEADLINE);
                }
                Wake::Cancelled(cancellation) => break Some(self.cancelled_item(cancellation)),
                Wake::Item(Some(item)) => break Some(item),
                // A cancellation can tear the response sink down before its
                // notification is seen; it, not the closed sink, ends the call.
                Wake::Item(None) => {
                    let cancellation = self.generation.as_mut().and_then(Generation::try_cancelled);
                    break cancellation.map(|cancellation| self.cancelled_item(cancellation));
                }
                Wake::AbortDeadline => {
                    tracing::warn!(rid = %self.rid, "scheduler did not answer an abort; ending the request");
                    if self.admission.cancel_requires_abort() {
                        let _ = self.abort_tx.send(AbortSource::Guard(self.rid.clone()));
                    }
                    break Some(self.cancelled_item(Cancellation::Scheduler));
                }
            }
        };
        self.observe(item.as_ref());
        if let Some(item) = &mut item {
            self.name_client_rid(item);
        }
        item
    }

    pub(super) fn try_recv(&mut self) -> Result<ResponseItem, mpsc::error::TryRecvError> {
        let mut item = match self.generation.as_mut().and_then(Generation::try_cancelled) {
            Some(cancellation @ (Cancellation::Error | Cancellation::Unstarted)) => {
                self.cancelled_item(cancellation)
            }
            Some(Cancellation::Scheduler) | None => self.response_rx.try_recv()?,
        };
        self.observe(Some(&item));
        self.name_client_rid(&mut item);
        Ok(item)
    }

    /// The scheduler knows a request by the rid this server sent it, which
    /// carries [`Rid::from_client`]'s suffix. An abort message it writes
    /// reaches the client, so name the request there by the client's rid.
    ///
    /// The message may hold the rid raw (`{rid}`) or as Python's `repr`
    /// (`{rid=}`, `{rid!r}`), which escapes the client's part and quotes the
    /// whole. The suffix is ASCII hex after `#`, so neither form changes it
    /// or the quote `repr` picks: dropping the suffix leaves the client's rid
    /// in the form the scheduler wrote, as Python's frontend shows it.
    fn name_client_rid(&self, item: &mut ResponseItem) {
        let client_rid = self.rid.client_facing();
        if let ResponseItem::Done(output) = item
            && client_rid.len() != self.rid.len()
            && let Some(reason) = &mut output.finish_reason
        {
            reason.replace_in_abort_message(&self.rid.as_str()[client_rid.len()..], "");
        }
    }

    /// The item that ends this generation on `cancellation` without the
    /// scheduler's answer. The frontend abort output carries the plain abort
    /// finish reason Python's scheduler writes and what the request last
    /// reported, adds no tokens, and is marked as ended outside the output
    /// stream, as the scheduler marks an abort it answers from its queue.
    fn cancelled_item(&self, cancellation: Cancellation) -> ResponseItem {
        if cancellation == Cancellation::Error {
            return ResponseItem::Error(Error::Disconnected);
        }
        let last = &self.last_output;
        ResponseItem::Done(ChunkEvent {
            rid: self.rid.clone(),
            finish_reason: Some(
                FinishKind::Abort(Box::new(AbortReason {
                    message: Some("Aborted".into()),
                    status_code: None,
                    err_type: None,
                }))
                .into(),
            ),
            prompt_tokens: last.prompt_tokens,
            cached_tokens: last.cached_tokens,
            reasoning_tokens: last.reasoning_tokens,
            extras: Some(Box::new(ChunkExtras {
                abort_output: true,
                cached_tokens_details: last.cached_tokens_details.clone(),
                ..Default::default()
            })),
            ..Default::default()
        })
    }

    fn observe(&mut self, item: Option<&ResponseItem>) {
        if let Some(ResponseItem::Frame(output)) = item {
            let last = &mut self.last_output;
            last.prompt_tokens = output.prompt_tokens;
            last.cached_tokens = output.cached_tokens;
            last.reasoning_tokens = output.reasoning_tokens;
            if let Some(details) = output
                .extras
                .as_deref()
                .and_then(|extras| extras.cached_tokens_details.as_ref())
            {
                last.cached_tokens_details = Some(details.clone());
            }
        }
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
pub(super) enum ExpectedResponse {
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
    use super::*;
    use crate::api_server::handlers::test_utils::*;

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
}
