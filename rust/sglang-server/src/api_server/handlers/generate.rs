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

use super::handle::{CoreHandle, ExpectedResponse, RuntimeCall, translate_runtime_error};
use crate::api_server::core::{CoreError, CoreEvent, RequestMetrics, prefetch};

impl CoreHandle {
    pub(crate) async fn generate(
        &self,
        mut request: GenerateRequest,
    ) -> Result<CoreCall, CoreError> {
        let created_at = Instant::now();
        let mut generation = self
            .inner
            .generations
            .register(request.rid.clone(), self.inner.abort_tx.clone());
        tokio::select! {
            biased;
            _ = generation.cancelled() => return Err(CoreError::Cancelled("request aborted".into())),
            result = self.prepare_generations(std::slice::from_mut(&mut request)) => result?,
        }
        let metrics = self.request_metrics(&request, created_at);
        self.submit_generation(request, generation)
            .await
            .map(|runtime| self.generation_call(runtime, metrics))
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
        self.prepare_generations(std::slice::from_mut(&mut request))
            .await?;
        let metrics = log_metrics
            .then(|| self.request_metrics(&request, created_at))
            .flatten();
        self.submit(RequestKind::Generate(Box::new(request)))
            .await
            .map(|runtime| self.generation_call(runtime, metrics))
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
            let metrics = self.request_metrics(&request, created_at);
            let call = match self.submit_generation(request, generation).await {
                Ok(call) => self.generation_call(call, metrics),
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

    fn generation_call(&self, runtime: RuntimeCall, metrics: Option<RequestMetrics>) -> CoreCall {
        CoreCall::new(runtime, self.inner.metadata.weight_version.clone(), metrics)
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
        prefetch::prefetch_all(
            requests,
            &self.inner.mm_limits,
            self.inner.mm_processor.get(),
        )
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
}

impl CoreCall {
    fn new(
        runtime: RuntimeCall,
        weight_version: Option<Arc<str>>,
        metrics: Option<RequestMetrics>,
    ) -> Self {
        Self {
            runtime,
            events_finished: false,
            weight_version,
            metrics,
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
            },
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
            },
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

/// Public-ID lookup for explicit cancellation. Internal IDs remain unique;
/// this index never replaces the runtime's request identity or admission gate.
#[derive(Clone, Default)]
pub(super) struct Generations(Arc<Mutex<Entries>>);

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
pub(super) struct Generation {
    registry: Generations,
    rid: Rid,
    pub(super) admission: RequestAdmission,
    cancellation: oneshot::Receiver<()>,
    abort_tx: flume::Sender<AbortSource>,
    armed: bool,
}

impl Generation {
    pub(super) async fn cancelled(&mut self) {
        // Preparation and submission may both observe the same cancellation;
        // do not poll a oneshot receiver again after consuming its notification.
        if !self.is_cancelled() {
            let _ = (&mut self.cancellation).await;
        }
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.admission.is_cancelled()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::{CoreErrorKind, CoreOutput};
    use crate::api_server::handlers::test_utils::*;
    use crate::message::multimodal::MmItem;
    use crate::message::request::MmData;
    use crate::message::response::ChunkEvent;
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
