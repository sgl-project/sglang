//! Generation: submission with the public-ID abort index, the semantic call
//! stream adapters consume, and the per-call lifetime that owns cancellation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RequestKind};
use crate::message::response::ResponseItem;
use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission};
use crate::utils::error::Error;

use super::handle::{
    CoreHandle, ExpectedResponse, LastOutput, RuntimeCall, translate_runtime_error,
};
use crate::api_server::core::{CoreError, CoreEvent, InFlight, RequestMetrics, prefetch};

impl CoreHandle {
    pub(crate) async fn generate(&self, request: GenerateRequest) -> Result<CoreCall, CoreError> {
        // Counted from admission, media prefetch included: the drain waits
        // for it.
        let in_flight = self.inner.drain.track();
        let generation = self
            .inner
            .generations
            .register(request.rid.clone(), self.inner.abort_tx.clone());
        self.generate_registered(request, generation, in_flight)
            .await
    }

    /// Register a generation under the public ID `rid` before its request
    /// exists, so an explicit cancellation in the meantime is kept for
    /// [`Self::generate_pending`]. The hold counts as in flight: the drain
    /// waits for the work before `/generate` as for media prefetch.
    pub(crate) fn hold_generation(&self, rid: &str) -> PendingGeneration {
        PendingGeneration {
            generation: self
                .inner
                .generations
                .register(Rid::from_client(rid), self.inner.abort_tx.clone()),
            in_flight: self.inner.drain.track(),
        }
    }

    /// [`Self::generate`] for a request held by [`Self::hold_generation`]
    /// and claimed by it ([`PendingGeneration::claim`]): a cancellation that
    /// selected the hold ends the call without starting the request.
    pub(crate) async fn generate_pending(
        &self,
        request: GenerateRequest,
        pending: PendingGeneration,
    ) -> Result<CoreCall, CoreError> {
        debug_assert!(
            pending.generation.rid == request.rid,
            "the request claims its hold"
        );
        self.generate_registered(request, pending.generation, pending.in_flight)
            .await
    }

    async fn generate_registered(
        &self,
        mut request: GenerateRequest,
        mut generation: Generation,
        in_flight: InFlight,
    ) -> Result<CoreCall, CoreError> {
        let created_at = Instant::now();
        self.set_default_priority(&mut request);
        tokio::select! {
            biased;
            cancellation = generation.cancelled() => {
                if cancellation == Cancellation::Error {
                    return Err(CoreError::Cancelled("request aborted".into()));
                }
                // Submission observes the same cancellation and ends the call
                // on its abort output without starting the request.
            }
            result = self.prepare_generations(std::slice::from_mut(&mut request)) => result?,
        }
        let metrics = self.request_metrics(&request, created_at);
        self.submit_generation(request, generation)
            .await
            .map(|runtime| self.generation_call(runtime, metrics, Some(in_flight)))
    }

    /// Submit core-owned work without exposing it through the public Abort
    /// registry. Drop still owns cancellation through `RuntimeCall`.
    /// `log_metrics` is Python's flag of the same name: the health probe's
    /// request is not counted.
    pub(super) async fn generate_internal(
        &self,
        mut request: GenerateRequest,
        log_metrics: bool,
    ) -> Result<CoreCall, CoreError> {
        let created_at = Instant::now();
        self.set_default_priority(&mut request);
        // Counted from admission, media prefetch included: the drain waits
        // for it. The health probe is not a client request; the drain never
        // waits for it.
        let in_flight = log_metrics.then(|| self.inner.drain.track());
        self.prepare_generations(std::slice::from_mut(&mut request))
            .await?;
        let metrics = log_metrics
            .then(|| self.request_metrics(&request, created_at))
            .flatten();
        self.submit(RequestKind::Generate(Box::new(request)))
            .await
            .map(|runtime| self.generation_call(runtime, metrics, in_flight))
    }

    /// Prepare the group before submitting any surviving request. Explicit
    /// cancellation produces only that item's terminal result, preserving batch
    /// order. Other preparation/admission failures still reject the group and
    /// dropping already-created calls aborts any accepted requests.
    pub(crate) async fn generate_batch(
        &self,
        mut requests: Vec<GenerateRequest>,
    ) -> Result<Vec<CoreCall>, CoreError> {
        let created_at = Instant::now();
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        for request in &mut requests {
            self.set_default_priority(request);
        }
        // Counted from admission, media prefetch included (see `generate`).
        let in_flight: Vec<InFlight> = requests.iter().map(|_| self.inner.drain.track()).collect();
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
        for ((request, generation), in_flight) in
            requests.into_iter().zip(generations).zip(in_flight)
        {
            let rid = request.rid.clone();
            let metrics = self.request_metrics(&request, created_at);
            let call = match self.submit_generation(request, generation).await {
                Ok(call) => self.generation_call(call, metrics, Some(in_flight)),
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

    /// Cancel active generations by public ID, or all active generations,
    /// ending each call as `ending` says. The admission gate and internal IDs
    /// keep cancellation safe across preprocessing, queueing, duplicate public
    /// IDs, and ID reuse.
    pub(crate) fn abort(&self, rid: Option<&str>, ending: AbortEnding) -> bool {
        self.inner
            .generations
            .abort(rid, ending, &self.inner.abort_tx)
    }

    async fn submit_generation(
        &self,
        request: GenerateRequest,
        generation: Generation,
    ) -> Result<RuntimeCall, CoreError> {
        self.submit_with_generation(RequestKind::Generate(Box::new(request)), Some(generation))
            .await
    }

    fn generation_call(
        &self,
        runtime: RuntimeCall,
        metrics: Option<RequestMetrics>,
        in_flight: Option<InFlight>,
    ) -> CoreCall {
        CoreCall::new(
            runtime,
            self.inner.metadata.weight_version.clone(),
            metrics,
            in_flight,
        )
    }

    /// Python `_set_default_priority`: an unset priority takes the configured
    /// default instead of ranking as the scheduler's lowest. Applied before
    /// the metrics pick the request's `priority` label, as Python does.
    fn set_default_priority(&self, request: &mut GenerateRequest) {
        if request.priority.is_none() {
            request.priority = self.inner.metadata.default_priority;
        }
    }

    fn request_metrics(
        &self,
        request: &GenerateRequest,
        created_at: Instant,
    ) -> Option<RequestMetrics> {
        let metrics = self.inner.metrics.as_ref()?;
        Some(metrics.start_request(request, created_at))
    }

    async fn prepare_generations(&self, requests: &mut [GenerateRequest]) -> Result<(), CoreError> {
        // A multimodal model's requests wait here until the MM pool starts, so
        // none is admitted without the processor's media hooks. The wait has
        // no deadline: the pool starts during launch, and runtime shutdown
        // cancels a waiting handler with the api runtime.
        let processor = if self.inner.multimodal {
            Some(self.inner.mm_processor.wait().await)
        } else {
            self.inner.mm_processor.get()
        };
        prefetch::prefetch_all(requests, &self.inner.mm_limits, processor)
            .await
            .map_err(CoreError::from)
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
    /// The launch weight version, stamped on every output (see
    /// [`CoreOutput::weight_version`](crate::api_server::core::CoreOutput)).
    weight_version: Option<Arc<str>>,
    /// Observes every runtime item as the call receives it.
    metrics: Option<RequestMetrics>,
    /// Holds the drain open while this client request is alive.
    _in_flight: Option<InFlight>,
}

impl CoreCall {
    fn new(
        runtime: RuntimeCall,
        weight_version: Option<Arc<str>>,
        metrics: Option<RequestMetrics>,
        in_flight: Option<InFlight>,
    ) -> Self {
        Self {
            runtime,
            events_finished: false,
            weight_version,
            metrics,
            _in_flight: in_flight,
        }
    }

    fn observe(&mut self, item: Option<&ResponseItem>) {
        if let Some(metrics) = &mut self.metrics {
            metrics.observe(item);
        }
    }

    /// Retain a cancelled batch item's ID and position without runtime work.
    fn cancelled(rid: Rid, abort_tx: flume::Sender<AbortSource>) -> Self {
        let (response_tx, response_rx) = mpsc::channel(1);
        response_tx
            .try_send(ResponseItem::Error(Error::Disconnected))
            .expect("new response channel has room for its terminal error");
        Self::new(
            RuntimeCall {
                rid,
                response_rx,
                abort_tx,
                admission: RequestAdmission::pending(),
                expected_response: ExpectedResponse::Generate,
                in_flight: false,
                generation: None,
                last_output: LastOutput::default(),
                abort_deadline: None,
            },
            None,
            None,
            None,
        )
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

        let item = self.runtime.recv().await;
        self.observe(item.as_ref());
        let event = match item {
            Some(item) => self.generation_event(item),
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
                Ok(item) => {
                    self.observe(Some(&item));
                    self.generation_event(item)
                }
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
        Self::new(
            RuntimeCall {
                rid,
                response_rx,
                abort_tx,
                admission,
                expected_response: ExpectedResponse::Generate,
                in_flight: true,
                generation: None,
                last_output: LastOutput::default(),
                abort_deadline: None,
            },
            None,
            None,
            None,
        )
    }

    fn generation_event(&self, item: ResponseItem) -> CoreEvent {
        let mut event = generation_event(item);
        if let CoreEvent::Delta(output) | CoreEvent::Finished(output) = &mut event {
            output.weight_version = self.weight_version.clone();
        }
        event
    }
}

impl Drop for CoreCall {
    fn drop(&mut self) {
        if let Some(metrics) = &self.metrics {
            metrics.call_dropped();
        }
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
                    output.into(),
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

type Entries = HashMap<String, HashMap<Rid, Entry>>;

/// How an explicit cancellation ends the calls it selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbortEnding {
    /// A cancelled error, as the gRPC `Abort` RPC reports it.
    CancelledError,
    /// A terminal output whose finish reason is the abort, as Python's
    /// `/abort_request` ends a request.
    AbortFinishReason,
}

/// An explicit cancellation of a generation not yet started: how it ends the
/// generation, and whether it aborted all generations rather than naming
/// this one's ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aborted {
    pub ending: AbortEnding,
    pub abort_all: bool,
}

/// How a call learns to end from an explicit cancellation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cancellation {
    /// End now with a cancelled error; the runtime is told to drop the request.
    Error,
    /// End now on a frontend abort output: Intake never claimed the request.
    Unstarted,
    /// The scheduler was asked to abort the request and keeps its response
    /// sink: end on the scheduler's terminal output, partial generation
    /// included, or on a frontend abort output if it never comes.
    Scheduler,
}

/// Public-ID lookup for explicit cancellation. Internal IDs remain unique;
/// this index never replaces the runtime's request identity or admission gate.
#[derive(Clone, Default)]
pub(super) struct Generations(Arc<Mutex<Entries>>);

struct Entry {
    admission: RequestAdmission,
    /// The cancellation, and whether it was an abort of all generations.
    cancelled: oneshot::Sender<(Cancellation, bool)>,
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
            observed: None,
            abort_all: false,
            abort_tx,
            armed: true,
        }
    }

    fn abort(
        &self,
        rid: Option<&str>,
        ending: AbortEnding,
        abort_tx: &flume::Sender<AbortSource>,
    ) -> bool {
        // Detach a snapshot under the lock, then notify outside it. Completion
        // and a new request reusing the same public ID cannot remove each other.
        Self::notify(self.detach(rid), rid.is_none(), ending, abort_tx)
    }

    /// The entries an abort of `rid`, or of all, selects, taken out of the
    /// index.
    fn detach(&self, rid: Option<&str>) -> Vec<HashMap<Rid, Entry>> {
        let mut entries = self.0.lock().unwrap();
        match rid {
            Some(rid) => entries.remove(rid).into_iter().collect(),
            None => std::mem::take(&mut *entries).into_values().collect(),
        }
    }

    /// End the `selected` entries' generations as `ending` says;
    /// `abort_all` if the abort selected every generation.
    fn notify(
        selected: Vec<HashMap<Rid, Entry>>,
        abort_all: bool,
        ending: AbortEnding,
        abort_tx: &flume::Sender<AbortSource>,
    ) -> bool {
        let mut matched = false;
        for entries in selected {
            for (rid, entry) in entries {
                matched = true;
                // Stop backend work immediately, even if the client is not
                // polling its response stream. The same admission gate also
                // prevents a queued request from starting after cancellation.
                match ending {
                    AbortEnding::CancelledError => {
                        // Cancel admission before notifying, so a racing
                        // terminal cannot complete the request and skip the
                        // abort, and notify before the abort can tear down
                        // the response sink the call may be polling.
                        let requires_abort = entry.admission.cancel_requires_abort();
                        let _ = entry.cancelled.send((Cancellation::Error, abort_all));
                        if requires_abort {
                            let _ = abort_tx.send(AbortSource::Guard(rid));
                        }
                    }
                    AbortEnding::AbortFinishReason => {
                        if entry.admission.cancel_if_pending() {
                            let _ = entry.cancelled.send((Cancellation::Unstarted, abort_all));
                        } else if entry.admission.is_accepted() {
                            let _ = entry.cancelled.send((Cancellation::Scheduler, abort_all));
                            let _ = abort_tx.send(AbortSource::Request(rid));
                        }
                    }
                }
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
pub(super) struct Generation {
    registry: Generations,
    rid: Rid,
    pub(super) admission: RequestAdmission,
    cancellation: oneshot::Receiver<(Cancellation, bool)>,
    /// The cancellation once observed. Preparation, submission and the
    /// response stream may each observe it, and a oneshot receiver must not
    /// be polled again after consuming its notification.
    observed: Option<Cancellation>,
    /// Whether the observed cancellation was an abort of all generations.
    abort_all: bool,
    abort_tx: flume::Sender<AbortSource>,
    armed: bool,
}

impl Generation {
    pub(super) async fn cancelled(&mut self) -> Cancellation {
        if let Some(cancellation) = self.observed {
            return cancellation;
        }
        let received = (&mut self.cancellation).await;
        self.observe(received.ok())
    }

    fn observe(&mut self, received: Option<(Cancellation, bool)>) -> Cancellation {
        let (cancellation, abort_all) = received.unwrap_or((Cancellation::Error, false));
        self.abort_all = abort_all;
        *self.observed.insert(cancellation)
    }

    /// The cancellation, if it has already happened.
    pub(super) fn try_cancelled(&mut self) -> Option<Cancellation> {
        if self.observed.is_none() {
            match self.cancellation.try_recv() {
                Ok(received) => {
                    self.observe(Some(received));
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    self.observe(None);
                }
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
        }
        self.observed
    }

    pub(super) fn complete(&mut self) {
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

/// A generation registered for explicit cancellation before its request
/// exists ([`CoreHandle::hold_generation`]). Dropping it unregisters it.
pub(crate) struct PendingGeneration {
    generation: Generation,
    /// Holds the drain open until the request ends or the hold drops.
    in_flight: InFlight,
}

impl PendingGeneration {
    /// Make this the generation of `request`, which arrived for it. The
    /// request takes the held internal rid rather than the entry moving to
    /// the request's, so an abort that has already detached the entry still
    /// names the rid that is submitted. A request under another public ID
    /// cannot be reached by aborts naming the held one: it registers afresh,
    /// keeping a cancellation that already selected the hold.
    pub(crate) fn claim(self, request: &mut GenerateRequest) -> Self {
        let Self {
            generation: mut held,
            in_flight,
        } = self;
        if held.rid.client_facing() == request.rid.client_facing() {
            request.rid = held.rid.clone();
            return Self {
                generation: held,
                in_flight,
            };
        }
        tracing::warn!(
            held = held.rid.client_facing(),
            rid = request.rid.client_facing(),
            "a request held before /generate arrived under another rid; later aborts of the \
             held rid do not reach it"
        );
        let (cancellation, abort_all) = (held.try_cancelled(), held.abort_all);
        let mut fresh = held
            .registry
            .register(request.rid.clone(), held.abort_tx.clone());
        drop(held);
        if let Some(cancellation) = cancellation {
            fresh.admission.cancel_if_pending();
            fresh.observed = Some(cancellation);
            fresh.abort_all = abort_all;
        }
        Self {
            generation: fresh,
            in_flight,
        }
    }

    /// Resolves once an explicit cancellation selects the generation.
    pub(crate) async fn cancelled(&mut self) -> Aborted {
        let ending = match self.generation.cancelled().await {
            Cancellation::Error => AbortEnding::CancelledError,
            Cancellation::Unstarted | Cancellation::Scheduler => AbortEnding::AbortFinishReason,
        };
        Aborted {
            ending,
            abort_all: self.generation.abort_all,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::handle::{CoreConfig, CoreMetadata};
    use super::AbortEnding::{AbortFinishReason, CancelledError};
    use super::*;
    use crate::api_server::core::FrontendMetrics;
    use crate::api_server::core::{CoreErrorKind, CoreOutput, DrainEnd};
    use crate::api_server::handlers::test_utils::*;
    use crate::message::config::DisaggregationMode;
    use crate::message::multimodal::MmItem;
    use crate::message::request::MmData;
    use crate::message::response::{CachedTokensDetails, ChunkEvent, ChunkExtras};
    use crate::multi_modality::worker::MediaError;
    use crate::tokenizer_manager::wiring::TmEvent;
    use bytes::Bytes;
    use std::time::Duration;

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
            let error =
                CoreError::from_runtime_rejection("rejected".into(), status, Default::default());
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

    /// A generation that sets no priority is scheduled and counted under the
    /// configured default, as Python's `_set_default_priority` runs before
    /// `collect_metrics` picks the label.
    #[tokio::test]
    async fn unset_priority_takes_the_default_before_metrics() {
        let dir = std::env::temp_dir().join(format!("core-priority-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = crate::message::config::MetricsConfig {
            labels: [("priority".to_string(), String::new())].into(),
            priority_label: true,
            ..crate::message::config::MetricsConfig::for_dir(&dir)
        };
        let metrics = Arc::new(FrontendMetrics::new(&config, DisaggregationMode::Null).unwrap());
        let (intake_tx, intake_rx) = flume::unbounded();
        let handle = CoreHandle::new(
            intake_tx,
            flume::unbounded().0,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                multimodal: false,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata {
                    default_priority: Some(5),
                    ..Default::default()
                },
                metrics: Some(metrics.clone()),
            },
        );
        let mut call = handle.generate(GenerateRequest::default()).await.unwrap();
        let request = accept_intake(intake_rx.recv().unwrap());
        let RequestKind::Generate(generate) = &request.kind else {
            unreachable!()
        };
        assert_eq!(generate.priority, Some(5));
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent {
                completion_tokens: 1,
                finish_reason: Some(crate::message::finish_reason::FinishReason::Known(
                    crate::message::finish_reason::FinishKind::Stop { matched: None },
                )),
                ..Default::default()
            }))
            .unwrap();
        while call.recv().await.is_some() {}
        drop(call);

        metrics.flush().unwrap();
        let files = crate::utils::prometheus::read_dir_files(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let text = crate::utils::prometheus::render(&crate::utils::prometheus::merge(
            files
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice())),
        ));
        assert!(
            text.contains(r#"sglang:num_requests_total{is_streaming="false",priority="5"} 1.0"#),
            "{text}"
        );
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

    /// A scheduler abort that carried a status fails the call with that
    /// status and keeps the terminal output, and the scheduler's message
    /// names the request by the client's rid, not the suffixed one this
    /// server sent the scheduler — raw, or as Python's `repr` (`{rid=}`)
    /// with the client's escaping and quoting kept, as Python shows it.
    #[tokio::test]
    async fn status_abort_names_the_client_rid() {
        // (client rid, Python's `repr` of it)
        for (client_rid, client_repr) in [
            ("client-rid", "'client-rid'"),
            (r"back\slash", r"'back\\slash'"),
            ("new\nline", r"'new\nline'"),
            ("it's", r#""it's""#),
        ] {
            let harness = Harness::unbounded(8);
            let mut call = harness
                .handle
                .generate(GenerateRequest {
                    rid: Rid::from_client(client_rid),
                    ..Default::default()
                })
                .await
                .unwrap();
            let internal = call.runtime.rid.as_str().to_owned();
            let suffix = &internal[client_rid.len()..];
            // `repr` of the suffixed rid: the suffix lands inside the quotes.
            let (body, quote) = client_repr.split_at(client_repr.len() - 1);
            let internal_repr = format!("{body}{suffix}{quote}");
            let request = accept_intake(harness.intake_rx.recv().unwrap());
            request
                .sink
                .try_send(ResponseItem::Done(ChunkEvent {
                    prompt_tokens: 3,
                    finish_reason: serde_json::from_value(serde_json::json!({
                        "type": "abort",
                        "message": format!(
                            "Request {internal} failed: req.rid={internal_repr}"
                        ),
                        "status_code": 500,
                        "err_type": null
                    }))
                    .unwrap(),
                    ..Default::default()
                }))
                .unwrap();

            let Some(CoreEvent::Failed(error)) = call.recv().await else {
                panic!("a status abort fails the call");
            };
            let expected = format!("Request {client_rid} failed: req.rid={client_repr}");
            assert_eq!(error.http_status().as_u16(), 500);
            assert_eq!(error.to_string(), expected, "client rid {client_rid:?}");
            let output = error
                .into_rejected_output()
                .expect("a status abort keeps its output");
            assert_eq!(output.prompt_tokens, 3);
            assert_eq!(
                output.finish_reason.unwrap().abort_status(),
                Some((500, expected.as_str()))
            );
        }
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
            assert!(harness.handle.abort(Some(aborted), CancelledError));
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
            assert!(!harness.handle.abort(None, CancelledError));
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
        assert!(handle.abort(Some("B"), CancelledError));
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
        assert!(!handle.abort(None, CancelledError));
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
        assert!(harness.handle.abort(Some("same"), CancelledError));
        assert!(!harness.handle.abort(Some("same"), CancelledError));
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
        assert!(harness.handle.abort(Some("same"), CancelledError));
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
        assert!(harness.handle.abort(Some("queued"), CancelledError));
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

    /// A submission blocked on admission wakes on an explicit abort without
    /// starting the request: the gRPC ending fails it, the `/abort_request`
    /// ending returns a call that finishes on the abort.
    #[tokio::test]
    async fn explicit_abort_wakes_a_submission_blocked_on_admission() {
        for ending in [CancelledError, AbortFinishReason] {
            let (intake_tx, intake_rx) = flume::bounded(0);
            let (abort_tx, abort_rx) = flume::unbounded();
            let handle = test_handle(intake_tx, abort_tx, 8);
            let mut submit = Box::pin(handle.generate(public_request("blocked")));
            assert!(futures::poll!(submit.as_mut()).is_pending());
            assert!(handle.abort(Some("blocked"), ending));
            match (ending, submit.await) {
                (CancelledError, Err(CoreError::Cancelled(_))) => {}
                (AbortFinishReason, Ok(mut call)) => {
                    let Some(CoreEvent::Finished(output)) = call.recv().await else {
                        panic!("the call finishes on the abort");
                    };
                    let reason = output.finish_reason.unwrap();
                    assert_eq!(reason.kind_name(), Some("abort"));
                    assert_eq!(reason.abort_status(), None, "a plain abort");
                    assert!(call.recv().await.is_none());
                }
                (ending, result) => panic!("{ending:?}: unexpected {:?}", result.err()),
            }
            if let Ok(TmEvent::Intake { admission, .. }) = intake_rx.try_recv() {
                assert!(!admission.try_accept());
            }
            assert!(abort_rx.try_recv().is_err());
        }
    }

    /// An explicit cancellation is published before its backend abort, which
    /// lets Intake tear down the response sink. A call that then finds its
    /// sink closed ends on the cancellation, not as a truncated response.
    /// The abort lane here is a rendezvous, so the abort is held handing over
    /// its backend abort while the sink is dropped.
    #[tokio::test]
    async fn explicit_abort_ends_the_call_before_its_sink_closes() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::bounded(0);
        let handle = test_handle(intake_tx, abort_tx, 8);
        let mut call = handle.generate(public_request("raced")).await.unwrap();
        let TmEvent::Intake { request, admission } = intake_rx.recv().unwrap() else {
            panic!("expected intake");
        };
        assert!(admission.try_accept());
        let aborter = {
            let handle = handle.clone();
            std::thread::spawn(move || handle.abort(Some("raced"), CancelledError))
        };
        while admission.is_accepted() {
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(20));
        drop(request);
        assert!(matches!(
            call.recv().await,
            Some(CoreEvent::Failed(CoreError::Cancelled(message))) if message == "client disconnected"
        ));
        assert!(
            matches!(abort_rx.recv().unwrap(), AbortSource::Guard(rid) if rid.client_facing() == "raced")
        );
        assert!(aborter.join().unwrap());
    }

    /// A claimed hold's request is submitted under the held internal rid, so
    /// an abort that detached the held entry before the request was admitted
    /// still names the submitted request once it notifies. A request under
    /// another public ID keeps a cancellation the hold already saw.
    #[tokio::test]
    async fn a_claimed_hold_keeps_the_rid_a_detached_abort_names() {
        for ending in [AbortFinishReason, CancelledError] {
            let harness = Harness::unbounded(8);
            let pending = harness.handle.hold_generation("held");
            let detached = harness.handle.inner.generations.detach(Some("held"));
            let mut request = public_request("held");
            let pending = pending.claim(&mut request);
            let handle = harness.handle.clone();
            let call = tokio::spawn(async move { handle.generate_pending(request, pending).await });
            let submitted = accept_intake(harness.intake_rx.recv_async().await.unwrap());
            assert!(Generations::notify(
                detached,
                false,
                ending,
                &harness.handle.inner.abort_tx
            ));
            let aborted = harness
                .abort_rx
                .try_recv()
                .expect("the abort reaches the runtime");
            assert_eq!(aborted.rid(), &submitted.rid, "{ending:?}");
            call.abort();
        }

        let harness = Harness::unbounded(8);
        let pending = harness.handle.hold_generation("held");
        assert!(harness.handle.abort(Some("held"), AbortFinishReason));
        let mut request = public_request("other");
        let pending = pending.claim(&mut request);
        assert_eq!(request.rid.client_facing(), "other");
        let mut call = harness
            .handle
            .generate_pending(request, pending)
            .await
            .unwrap();
        assert!(harness.intake_rx.try_recv().is_err());
        assert!(matches!(call.recv().await, Some(CoreEvent::Finished(_))));
    }

    /// A request still fetching its media when the drain begins holds the
    /// drain open, and is admitted once its media arrive.
    #[tokio::test]
    async fn drain_waits_for_a_request_in_media_prefetch() {
        use crate::message::types::TokenIds;
        use crate::multi_modality::worker::MmProcessor;

        struct SlowFetch {
            started: std::sync::Mutex<std::sync::mpsc::Sender<()>>,
            release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
        }

        impl MmProcessor for SlowFetch {
            fn process(
                &self,
                _input_ids: TokenIds,
                _mm: MmData,
            ) -> Result<crate::multi_modality::worker::MmProcessOutput, MediaError> {
                unreachable!("never encoded here")
            }

            fn fetch_media(
                &self,
                _source: &str,
                _modality: crate::multi_modality::encoded::MmModality,
                _budget: &crate::MediaByteBudget,
            ) -> Result<Option<Bytes>, MediaError> {
                self.started.lock().unwrap().send(()).unwrap();
                self.release.lock().unwrap().recv().unwrap();
                Ok(Some(Bytes::from_static(b"media")))
            }
        }

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let harness = Harness::unbounded(8);
        assert!(harness.handle.set_mm_processor(Arc::new(SlowFetch {
            started: std::sync::Mutex::new(started_tx),
            release: std::sync::Mutex::new(release_rx),
        })));
        let request = GenerateRequest {
            rid: "fetching".into(),
            input_ids: Some(vec![1]),
            mm: Some(Box::new(MmData {
                image_data: vec![MmItem::Source("custom://slow".into())],
                ..Default::default()
            })),
            ..Default::default()
        };
        let handle = harness.handle.clone();
        let call = tokio::spawn(async move { handle.generate(request).await });
        tokio::task::spawn_blocking(move || started_rx.recv().unwrap())
            .await
            .unwrap();

        harness.handle.begin_drain(false);
        let ended =
            || tokio::time::timeout(Duration::from_millis(100), harness.handle.drain_ended());
        assert!(
            ended().await.is_err(),
            "the drain ignored a request in prefetch"
        );
        release_tx.send(()).unwrap();
        let call = call
            .await
            .unwrap()
            .expect("admitted after its media arrived");
        let _admitted = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(ended().await.is_err());
        drop(call);
        assert_eq!(ended().await.ok(), Some(DrainEnd::Drained));
    }

    /// A request held before `/generate` ([`CoreHandle::hold_generation`])
    /// holds the drain open from the hold, through its claim, until its call
    /// ends; a hold dropped unclaimed (a client gone during the layer's work)
    /// releases it.
    #[tokio::test]
    async fn drain_waits_for_a_held_request() {
        let harness = Harness::unbounded(8);
        let held = harness.handle.hold_generation("held");
        let dropped = harness.handle.hold_generation("dropped");
        harness.handle.begin_drain(false);
        let ended =
            || tokio::time::timeout(Duration::from_millis(100), harness.handle.drain_ended());
        drop(dropped);
        assert!(ended().await.is_err(), "the drain ignored a held request");
        let mut request = public_request("held");
        let held = held.claim(&mut request);
        let handle = harness.handle.clone();
        let call = tokio::spawn(async move { handle.generate_pending(request, held).await });
        let _admitted = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        let call = call.await.unwrap().expect("the held request is admitted");
        assert!(ended().await.is_err());
        drop(call);
        assert_eq!(ended().await.ok(), Some(DrainEnd::Drained));
    }

    /// An `/abort_request` that ends a call before Intake claims it records
    /// no finished request: the request never ran, as for an abort the
    /// scheduler answers from its queue.
    #[tokio::test]
    async fn unstarted_abort_request_records_no_finished_request() {
        let dir = std::env::temp_dir().join(format!("core-unstarted-abort-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = crate::message::config::MetricsConfig::for_dir(&dir);
        let metrics = Arc::new(FrontendMetrics::new(&config, DisaggregationMode::Null).unwrap());
        let (intake_tx, _intake_rx) = flume::bounded(0);
        let handle = CoreHandle::new(
            intake_tx,
            flume::unbounded().0,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                multimodal: false,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::default(),
                metrics: Some(metrics.clone()),
            },
        );
        let mut submit = Box::pin(handle.generate(public_request("blocked")));
        assert!(futures::poll!(submit.as_mut()).is_pending());
        assert!(handle.abort(Some("blocked"), AbortFinishReason));
        let mut call = submit.await.unwrap();
        assert!(matches!(call.recv().await, Some(CoreEvent::Finished(_))));
        drop(call);

        metrics.flush().unwrap();
        let files = crate::utils::prometheus::read_dir_files(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let text = crate::utils::prometheus::render(&crate::utils::prometheus::merge(
            files
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice())),
        ));
        assert!(
            text.contains(r#"sglang:num_requests_total{is_streaming="false"} 0.0"#),
            "{text}"
        );
    }

    /// A call handed to the scheduler by `/abort_request` ends on a frontend
    /// abort output when the scheduler's answer cannot come: its sink closed
    /// (Intake fell back to dropping it), or the deadline passed, which also
    /// stops the scheduler as a dropped call does. The output repeats what the
    /// request last reported, its cache split included.
    #[tokio::test(start_paused = true)]
    async fn scheduler_abort_without_an_answer_ends_on_a_frontend_abort_output() {
        for sink_closes in [true, false] {
            let harness = Harness::unbounded(8);
            let mut call = harness
                .handle
                .generate(public_request("held"))
                .await
                .unwrap();
            let request = accept_intake(harness.intake_rx.recv().unwrap());
            let details = CachedTokensDetails {
                device: 1,
                host: 2,
                storage: None,
                storage_backend: None,
            };
            request
                .sink
                .try_send(ResponseItem::Frame(ChunkEvent {
                    rid: request.rid.clone(),
                    token_ids: vec![7],
                    prompt_tokens: 5,
                    cached_tokens: 3,
                    completion_tokens: 1,
                    extras: Some(Box::new(ChunkExtras {
                        cached_tokens_details: Some(details.clone()),
                        ..Default::default()
                    })),
                    ..Default::default()
                }))
                .unwrap();
            assert!(matches!(call.recv().await, Some(CoreEvent::Delta(_))));
            assert!(harness.handle.abort(Some("held"), AbortFinishReason));
            assert!(matches!(
                harness.abort_rx.try_recv(),
                Ok(AbortSource::Request(rid)) if rid == request.rid
            ));
            let rid = request.rid.clone();
            if sink_closes {
                drop(request);
            }

            let Some(CoreEvent::Finished(output)) = call.recv().await else {
                panic!("sink_closes={sink_closes}: the call finishes on an abort output");
            };
            assert_eq!(output.finish_reason.unwrap().kind_name(), Some("abort"));
            assert_eq!((output.prompt_tokens, output.cached_tokens), (5, 3));
            let extras = output.extras.unwrap();
            assert!(extras.abort_output);
            assert_eq!(extras.cached_tokens_details, Some(details));
            if sink_closes {
                assert!(harness.abort_rx.try_recv().is_err());
            } else {
                assert!(matches!(
                    harness.abort_rx.try_recv(),
                    Ok(AbortSource::Guard(aborted)) if aborted == rid
                ));
            }
        }
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
        assert!(!harness.handle.abort(Some("done"), CancelledError));
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
        assert!(!harness.handle.abort(Some("dropped"), CancelledError));
        assert!(!harness.handle.abort(None, CancelledError));
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
        assert!(harness.handle.abort(None, CancelledError));
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
        assert!(!harness.handle.abort(None, CancelledError));
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
        assert!(harness.handle.abort(Some("buffered"), CancelledError));
        let mut events = Vec::new();
        call.drain_ready(&mut events);
        assert!(matches!(
            events.as_slice(),
            [CoreEvent::Failed(CoreError::Cancelled(_))]
        ));
        assert!(call.recv().await.is_none());
    }

    /// A processor's media failures reach adapters as Python's loader classes:
    /// a client fault is a 400, a server fault a 500. Neither is admitted.
    #[tokio::test]
    async fn processor_media_failures_keep_their_status_class() {
        use crate::message::types::TokenIds;
        use crate::multi_modality::worker::MmProcessor;

        struct FailingFetch;

        impl MmProcessor for FailingFetch {
            fn process(
                &self,
                _input_ids: TokenIds,
                _mm: MmData,
            ) -> Result<crate::multi_modality::worker::MmProcessOutput, MediaError> {
                unreachable!("rejected during admission")
            }

            fn fetch_media(
                &self,
                source: &str,
                _modality: crate::multi_modality::encoded::MmModality,
                _budget: &crate::MediaByteBudget,
            ) -> Result<Option<Bytes>, MediaError> {
                Err(match source {
                    "custom://client" => MediaError::Client("bad media".into()),
                    _ => MediaError::Server("backend down".into()),
                })
            }
        }

        let harness = Harness::unbounded(8);
        assert!(harness.handle.set_mm_processor(Arc::new(FailingFetch)));
        assert!(!harness.handle.set_mm_processor(Arc::new(FailingFetch)));
        for (source, status, message) in [
            ("custom://client", 400, "bad media"),
            ("custom://server", 500, "backend down"),
        ] {
            let request = GenerateRequest {
                mm: Some(Box::new(MmData {
                    image_data: vec![MmItem::Source(source.into())],
                    ..Default::default()
                })),
                ..Default::default()
            };
            let error = harness.handle.generate(request).await.err().unwrap();
            assert_eq!(error.http_status().as_u16(), status, "{source}");
            assert_eq!(error.to_string(), message);
        }
        assert!(harness.intake_rx.try_recv().is_err());
        assert!(harness.abort_rx.try_recv().is_err());
    }
}
