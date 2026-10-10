//! The `SglangService` implementation: per-RPC translation between the
//! `api.v1` messages and the shared core.

use std::pin::Pin;
use std::time::{Duration, Instant};

use futures::Stream;
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_server::SglangService;
use tonic::{Request, Response, Status};

use super::{info, native_api};
use crate::api_server::core::{CoreHandle, HealthStatus};
use crate::message::config::{PreferredSamplingParams, ServerArgs};
use crate::message::request::into_requests;
use crate::message::wire::fill_preferred_sampling;
use crate::utils::environ;

const DEFAULT_RESPONSE_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Narrow snapshot of launch policy the adapter needs per call. It
/// intentionally excludes listener, authentication, TLS, and lifecycle
/// configuration.
#[derive(Clone)]
struct AdapterConfig {
    preferred_sampling_params: Option<PreferredSamplingParams>,
    incremental_streaming_output: bool,
    /// Longest wait for the next event of any in-flight call before the RPC
    /// ends with DEADLINE_EXCEEDED.
    response_timeout: Duration,
    /// `/health_generate`'s heartbeat window (`SGLANG_HEALTH_CHECK_TIMEOUT`).
    health_timeout: Duration,
}

/// Tonic-facing implementation backed by the transport-neutral core.
pub(crate) struct GrpcService {
    core: CoreHandle,
    config: AdapterConfig,
}

impl GrpcService {
    pub(crate) fn new(core: CoreHandle, server_args: &ServerArgs) -> Self {
        Self {
            core,
            config: AdapterConfig {
                preferred_sampling_params: server_args.preferred_sampling_params.clone(),
                incremental_streaming_output: server_args.incremental_streaming_output,
                response_timeout: DEFAULT_RESPONSE_TIMEOUT,
                health_timeout: Duration::from_secs(
                    environ::env_i64("SGLANG_HEALTH_CHECK_TIMEOUT", 20).max(0) as u64,
                ),
            },
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(
        core: CoreHandle,
        preferred_sampling_params: Option<PreferredSamplingParams>,
        incremental_streaming_output: bool,
        response_timeout: Duration,
    ) -> Self {
        Self {
            core,
            config: AdapterConfig {
                preferred_sampling_params,
                incremental_streaming_output,
                response_timeout,
                health_timeout: Duration::from_millis(20),
            },
        }
    }
}

#[tonic::async_trait]
impl SglangService for GrpcService {
    type GenerateStream = ResponseStream<api::GenerateStreamItem>;

    /// `/generate` over gRPC: one schema, one fan-out, one admission path. A
    /// body that fails normalization ends the RPC with INVALID_ARGUMENT before
    /// anything reaches the scheduler, as the HTTP 400 does.
    async fn generate(
        &self,
        request: Request<api::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let mut request = request.into_inner();
        if let Some(preferred) = &self.config.preferred_sampling_params {
            request.sampling_params =
                fill_preferred_sampling(request.sampling_params.take(), &preferred.0)
                    .map_err(Status::internal)?;
        }
        // Protobuf has no null: a field the client left unset takes the
        // schema default, as an absent JSON key does.
        request.apply_absent_defaults();
        let stream = request.stream.unwrap_or(false);
        let (payloads, is_batch) =
            into_requests(request).map_err(|error| Status::invalid_argument(error.to_string()))?;
        // Python starts its request clock after normalization and before
        // tokenization / multimodal preprocessing; `e2e_latency` measures from
        // the same boundary here.
        let created_at = Instant::now();
        let calls = self
            .core
            .generate_batch(payloads)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(native_api::generate_stream(
            calls,
            native_api::StreamOptions {
                stream,
                incremental: self.config.incremental_streaming_output,
                with_index: is_batch,
                response_timeout: self.config.response_timeout,
                created_at,
            },
        )))
    }

    async fn health_check(
        &self,
        _request: Request<api::HealthCheckRequest>,
    ) -> Result<Response<api::HealthCheckResponse>, Status> {
        match self.core.probe_health(self.config.health_timeout).await {
            Ok(HealthStatus::Healthy) => {
                Ok(Response::new(api::HealthCheckResponse { healthy: true }))
            }
            Ok(HealthStatus::NotReady) => Err(Status::unavailable(
                "server is still completing its startup warmup",
            )),
            Ok(HealthStatus::Stalled) => Err(Status::unavailable(
                "no scheduler output within the health-check timeout",
            )),
            Err(error) => Err(native_api::status(error)),
        }
    }

    async fn get_model_info(
        &self,
        _request: Request<api::GetModelInfoRequest>,
    ) -> Result<Response<api::GetModelInfoResponse>, Status> {
        info::model_info(self.core.model_info())
            .map(Response::new)
            .map_err(Status::internal)
    }

    async fn get_server_info(
        &self,
        _request: Request<api::GetServerInfoRequest>,
    ) -> Result<Response<api::GetServerInfoResponse>, Status> {
        let server_info = self.core.server_info().await.map_err(native_api::status)?;
        info::server_info(server_info)
            .map(Response::new)
            .map_err(Status::internal)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::StreamExt;
    use sglang_api_types::api::v1 as api;
    use sglang_api_types::api::v1::generate_stream_item::Item;
    use sglang_api_types::api::v1::sglang_service_server::SglangService;
    use tonic::{Code, Request};

    use super::GrpcService;
    use crate::api_server::core::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::PreferredSamplingParams;
    use crate::message::finish_reason::FinishReason;
    use crate::message::ids::Rid;
    use crate::message::request::{GenerateRequest, Request as RuntimeRequest, RequestKind};
    use crate::message::response::{ChunkEvent, ResponseItem, ResponseSink};
    use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission, TmEvent};

    struct Harness {
        service: GrpcService,
        intake_rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
    }

    struct GenerationIntake {
        request: Box<GenerateRequest>,
        rid: Rid,
        sink: ResponseSink,
        admission: RequestAdmission,
    }

    impl Harness {
        fn new(response_capacity: usize, incremental: bool, response_timeout: Duration) -> Self {
            Self::build(response_capacity, incremental, response_timeout, true, None)
        }

        fn build(
            response_capacity: usize,
            incremental: bool,
            response_timeout: Duration,
            startup_ready: bool,
            preferred: Option<PreferredSamplingParams>,
        ) -> Self {
            let (intake_tx, intake_rx) = flume::unbounded();
            let (abort_tx, abort_rx) = flume::unbounded();
            let core = CoreHandle::new(
                intake_tx,
                abort_tx,
                CoreConfig {
                    response_capacity,
                    response_activity: Default::default(),
                    startup_ready,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: CoreMetadata::default(),
                },
            );
            Self {
                service: GrpcService::for_test(core, preferred, incremental, response_timeout),
                intake_rx,
                abort_rx,
            }
        }

        async fn next_generation(&self) -> GenerationIntake {
            let TmEvent::Intake { request, admission } = self
                .intake_rx
                .recv_async()
                .await
                .expect("request submitted")
            else {
                panic!("expected generation intake");
            };
            let RuntimeRequest {
                rid,
                sink,
                kind,
                state: _,
            } = request;
            let RequestKind::Generate(request) = kind else {
                panic!("expected generation request");
            };
            GenerationIntake {
                request,
                rid,
                sink,
                admission,
            }
        }
    }

    fn chunk(rid: &Rid, text: &str, token_id: i64, finished: bool) -> ChunkEvent {
        ChunkEvent {
            rid: rid.clone(),
            token_ids: vec![token_id],
            finish_reason: finished.then(stop_reason),
            prompt_tokens: 3,
            text: text.into(),
            completion_tokens: 1,
            stop_token_trimmed: false,
            extras: None,
        }
    }

    fn stop_reason() -> FinishReason {
        serde_json::from_value(serde_json::json!({"type": "stop"})).unwrap()
    }

    fn one_string(value: &str) -> api::StringOrList {
        api::StringOrList {
            value: Some(api::string_or_list::Value::One(value.into())),
        }
    }

    fn many_strings(values: &[&str]) -> api::StringOrList {
        api::StringOrList {
            value: Some(api::string_or_list::Value::Many(api::StringList {
                items: values.iter().map(|value| (*value).to_owned()).collect(),
            })),
        }
    }

    fn one_token_ids(ids: &[i64]) -> api::TokenIdsOrList {
        api::TokenIdsOrList {
            value: Some(api::token_ids_or_list::Value::One(api::TokenIds {
                ids: ids.to_vec(),
            })),
        }
    }

    fn text_request(text: &str, rid: &str) -> api::GenerateRequest {
        api::GenerateRequest {
            text: Some(one_string(text)),
            rid: Some(one_string(rid)),
            stream: Some(true),
            ..Default::default()
        }
    }

    fn ids_request(ids: &[i64], rid: Option<&str>) -> api::GenerateRequest {
        api::GenerateRequest {
            input_ids: Some(one_token_ids(ids)),
            rid: rid.map(one_string),
            stream: Some(true),
            ..Default::default()
        }
    }

    fn frame(item: api::GenerateStreamItem) -> api::GenerateResponse {
        match item.item {
            Some(Item::Frame(frame)) => frame,
            other => panic!("expected a frame, got {other:?}"),
        }
    }

    fn stream_error(item: api::GenerateStreamItem) -> api::GenerateStreamError {
        match item.item {
            Some(Item::Error(error)) => error,
            other => panic!("expected an error item, got {other:?}"),
        }
    }

    fn meta(frame: &api::GenerateResponse) -> &api::GenerateMetaInfo {
        frame
            .meta_info
            .as_ref()
            .expect("every frame carries meta_info")
    }

    fn output_ids(frame: &api::GenerateResponse) -> &[i64] {
        frame
            .output_ids
            .as_ref()
            .map_or(&[], |ids| ids.ids.as_slice())
    }

    #[tokio::test]
    async fn generate_streams_cumulative_frames_for_a_text_prompt() {
        let harness = Harness::new(2, false, Duration::from_secs(1));
        let mut stream = harness
            .service
            .generate(Request::new(text_request("prompt", "request-1")))
            .await
            .unwrap()
            .into_inner();

        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        assert_eq!(intake.request.text.as_deref(), Some("prompt"));
        assert!(intake.request.input_ids.is_none());
        assert!(intake.request.stream);
        assert_eq!(intake.rid.client_facing(), "request-1");
        // One chunk at a time: a queued backlog of cumulative frames collapses to
        // its last, so sending both up front would yield only the terminal frame.
        intake
            .sink
            .try_send(ResponseItem::Frame(chunk(&intake.rid, "Hel", 1, false)))
            .unwrap();
        let first = frame(stream.next().await.unwrap().unwrap());
        assert_eq!(first.text, "Hel");
        assert_eq!(output_ids(&first), [1]);
        assert_eq!(first.index, None);
        let first_meta = meta(&first);
        assert_eq!(first_meta.id, "request-1");
        assert_eq!(first_meta.prompt_tokens, 3);
        assert_eq!(first_meta.completion_tokens, 1);
        assert!(first_meta.finish_reason.is_none());
        assert!(first_meta.e2e_latency.is_none());

        intake
            .sink
            .try_send(ResponseItem::Done(chunk(&intake.rid, "lo", 2, true)))
            .unwrap();
        let finished = frame(stream.next().await.unwrap().unwrap());
        assert_eq!(finished.text, "Hello");
        assert_eq!(output_ids(&finished), [1, 2]);
        let finished_meta = meta(&finished);
        assert_eq!(finished_meta.completion_tokens, 2);
        assert!(matches!(
            finished_meta.finish_reason,
            Some(api::FinishReason {
                kind: Some(api::finish_reason::Kind::Stop(_))
            })
        ));
        assert!(finished_meta.e2e_latency.is_some());
        assert!(stream.next().await.is_none());
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn incremental_frames_carry_deltas_with_cumulative_count() {
        let harness = Harness::new(2, true, Duration::from_secs(1));
        let mut stream = harness
            .service
            .generate(Request::new(ids_request(&[4, 5], Some("tokens"))))
            .await
            .unwrap()
            .into_inner();

        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        assert_eq!(intake.request.input_ids.as_deref(), Some(&[4, 5][..]));
        assert!(intake.request.text.is_none());
        intake
            .sink
            .try_send(ResponseItem::Frame(chunk(&intake.rid, "A", 10, false)))
            .unwrap();
        intake
            .sink
            .try_send(ResponseItem::Done(chunk(&intake.rid, "B", 11, true)))
            .unwrap();

        let first = frame(stream.next().await.unwrap().unwrap());
        let finished = frame(stream.next().await.unwrap().unwrap());
        assert_eq!(output_ids(&first), [10]);
        assert_eq!(output_ids(&finished), [11]);
        assert_eq!(finished.text, "B");
        assert_eq!(meta(&finished).completion_tokens, 2);
        assert!(stream.next().await.is_none());
    }

    /// `stream: false` is a one-frame stream: deltas fold into the terminal
    /// result, and the server's incremental policy does not split a unary reply.
    #[tokio::test]
    async fn unary_request_yields_only_the_cumulative_terminal_frame() {
        let harness = Harness::new(2, true, Duration::from_secs(1));
        let mut request = text_request("prompt", "unary");
        request.stream = Some(false);
        let mut stream = harness
            .service
            .generate(Request::new(request))
            .await
            .unwrap()
            .into_inner();

        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        assert!(!intake.request.stream);
        intake
            .sink
            .try_send(ResponseItem::Frame(chunk(&intake.rid, "Hel", 1, false)))
            .unwrap();
        intake
            .sink
            .try_send(ResponseItem::Done(chunk(&intake.rid, "lo", 2, true)))
            .unwrap();

        let only = frame(stream.next().await.unwrap().unwrap());
        assert_eq!(only.text, "Hello");
        assert_eq!(output_ids(&only), [1, 2]);
        assert_eq!(meta(&only).completion_tokens, 2);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_drop_aborts_admitted_request() {
        let harness = Harness::new(1, false, Duration::from_secs(1));
        let mut stream = harness
            .service
            .generate(Request::new(ids_request(&[1], Some("cancel-me"))))
            .await
            .unwrap()
            .into_inner();
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        intake
            .sink
            .try_send(ResponseItem::Frame(chunk(&intake.rid, "partial", 1, false)))
            .unwrap();
        assert!(stream.next().await.unwrap().is_ok());
        drop(stream);
        let abort = harness.abort_rx.recv_async().await.unwrap();
        assert_eq!(abort.rid().client_facing(), "cancel-me");
        assert!(harness.abort_rx.try_recv().is_err());
    }

    /// A runtime failure after admission is this request's in-stream error item
    /// (the SSE error frame's twin), not an RPC status; the call is disarmed.
    #[tokio::test]
    async fn runtime_failure_rides_the_stream_as_an_error_item() {
        let harness = Harness::new(1, false, Duration::from_secs(1));
        let mut stream = harness
            .service
            .generate(Request::new(ids_request(&[1], None)))
            .await
            .unwrap()
            .into_inner();
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        intake
            .sink
            .try_send(ResponseItem::Error(crate::utils::error::Error::Validation(
                "bad sampling".into(),
            )))
            .unwrap();

        let error = stream_error(stream.next().await.unwrap().unwrap());
        assert_eq!(error.index, None);
        let body = error.error.unwrap();
        assert_eq!(body.code, 400);
        assert!(body.message.contains("bad sampling"));
        assert!(stream.next().await.is_none());
        drop(stream);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn per_chunk_timeout_after_progress_returns_deadline_exceeded_and_aborts() {
        let harness = Harness::new(1, false, Duration::from_millis(10));
        let mut stream = harness
            .service
            .generate(Request::new(ids_request(&[1], Some("too-slow"))))
            .await
            .unwrap()
            .into_inner();
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        intake
            .sink
            .try_send(ResponseItem::Frame(chunk(&intake.rid, "partial", 1, false)))
            .unwrap();
        assert!(stream.next().await.unwrap().is_ok());

        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(error.code(), Code::DeadlineExceeded);
        drop(stream);
        let abort = harness.abort_rx.recv_async().await.unwrap();
        assert_eq!(abort.rid().client_facing(), "too-slow");
    }

    #[tokio::test]
    async fn closed_intake_is_a_top_level_unavailable_status() {
        let (intake_tx, intake_rx) = flume::unbounded();
        drop(intake_rx);
        let (abort_tx, _abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 1,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: CoreMetadata::default(),
            },
        );
        let service = GrpcService::for_test(core, None, false, Duration::from_secs(1));

        let result = service
            .generate(Request::new(ids_request(&[1], None)))
            .await;
        let error = match result {
            Ok(_) => panic!("closed intake must reject the RPC"),
            Err(error) => error,
        };
        assert_eq!(error.code(), Code::Unavailable);
    }

    /// Normalization failures end the RPC before anything is submitted, with the
    /// status that mirrors the HTTP 400.
    #[tokio::test]
    async fn malformed_request_is_rejected_before_submission() {
        let harness = Harness::new(1, false, Duration::from_secs(1));
        let mut both = text_request("prompt", "both");
        both.input_ids = Some(one_token_ids(&[1]));
        let error = harness
            .service
            .generate(Request::new(both))
            .await
            .err()
            .expect("text and input_ids together must be rejected");
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(error.message().contains("text"));

        let error = harness
            .service
            .generate(Request::new(ids_request(&[], None)))
            .await
            .err()
            .expect("empty input_ids must be rejected");
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(harness.intake_rx.try_recv().is_err());
    }

    /// A list-form request fans out like the HTTP batch: every item is tagged
    /// with its position, and one item's failure neither ends the stream nor
    /// touches its siblings.
    #[tokio::test]
    async fn batch_items_are_indexed_and_fail_independently() {
        let harness = Harness::new(2, false, Duration::from_secs(1));
        let mut stream = harness
            .service
            .generate(Request::new(api::GenerateRequest {
                text: Some(many_strings(&["a", "b"])),
                stream: Some(true),
                ..Default::default()
            }))
            .await
            .unwrap()
            .into_inner();

        let first = harness.next_generation().await;
        let second = harness.next_generation().await;
        assert!(first.admission.try_accept());
        assert!(second.admission.try_accept());
        assert_eq!(first.request.text.as_deref(), Some("a"));
        assert_eq!(second.request.text.as_deref(), Some("b"));
        first
            .sink
            .try_send(ResponseItem::Error(crate::utils::error::Error::Validation(
                "first failed".into(),
            )))
            .unwrap();
        second
            .sink
            .try_send(ResponseItem::Done(chunk(&second.rid, "ok", 5, true)))
            .unwrap();

        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item.unwrap());
        }
        assert_eq!(items.len(), 2);
        let (errors, frames): (Vec<_>, Vec<_>) = items
            .into_iter()
            .partition(|item| matches!(item.item, Some(Item::Error(_))));
        let error = stream_error(errors.into_iter().next().unwrap());
        assert_eq!(error.index, Some(0));
        assert!(error.error.unwrap().message.contains("first failed"));
        let ok = frame(frames.into_iter().next().unwrap());
        assert_eq!(ok.index, Some(1));
        assert_eq!(ok.text, "ok");
        assert_eq!(meta(&ok).id, second.rid.client_facing());
        assert!(harness.abort_rx.try_recv().is_err());
    }

    /// Launch-time preferred sampling params fill the fields a protobuf request
    /// left unset, beneath the ones it set: the HTTP precedence at the gRPC entry.
    #[tokio::test]
    async fn preferred_sampling_params_fill_unset_fields() {
        let preferred = PreferredSamplingParams(serde_json::json!({
            "temperature": 0.25,
            "max_new_tokens": 32,
        }));
        let harness = Harness::build(1, false, Duration::from_secs(1), true, Some(preferred));
        let mut request = ids_request(&[1], None);
        request.sampling_params = Some(api::SamplingParamsOrList {
            value: Some(api::sampling_params_or_list::Value::One(
                api::SamplingParams {
                    temperature: Some(0.8),
                    ..Default::default()
                },
            )),
        });
        let _stream = harness
            .service
            .generate(Request::new(request))
            .await
            .unwrap()
            .into_inner();

        let intake = harness.next_generation().await;
        assert_eq!(intake.request.sampling_params.temperature, 0.8);
        assert_eq!(intake.request.sampling_params.max_new_tokens, Some(32));
    }

    /// Protobuf has no null: a sampling field the client left unset takes the
    /// schema default, as an absent JSON key does. `max_new_tokens` is the one
    /// field whose Rust `None` means "unbounded", so without the absent-defaults
    /// pass an unset value would lift the limit instead of applying 128.
    #[tokio::test]
    async fn unset_protobuf_sampling_fields_take_schema_defaults() {
        let harness = Harness::new(1, false, Duration::from_secs(1));
        let mut request = ids_request(&[1], None);
        request.sampling_params = Some(api::SamplingParamsOrList {
            value: Some(api::sampling_params_or_list::Value::One(
                api::SamplingParams {
                    temperature: Some(0.8),
                    ..Default::default()
                },
            )),
        });
        let _stream = harness
            .service
            .generate(Request::new(request))
            .await
            .unwrap()
            .into_inner();

        let intake = harness.next_generation().await;
        assert_eq!(intake.request.sampling_params.temperature, 0.8);
        assert_eq!(intake.request.sampling_params.max_new_tokens, Some(128));
        assert_eq!(intake.request.sampling_params.top_p, 1.0);
    }

    /// `HealthCheck` is `/health_generate`: not ready and stalled are both
    /// UNAVAILABLE, and the stalled probe is a real generation submission.
    #[tokio::test]
    async fn health_check_reports_not_ready_and_stalled_as_unavailable() {
        let not_ready = Harness::build(1, false, Duration::from_secs(1), false, None);
        let error = not_ready
            .service
            .health_check(Request::new(api::HealthCheckRequest {}))
            .await
            .expect_err("warmup must not report healthy");
        assert_eq!(error.code(), Code::Unavailable);
        assert!(not_ready.intake_rx.try_recv().is_err());

        let stalled = Harness::new(1, false, Duration::from_secs(1));
        let error = stalled
            .service
            .health_check(Request::new(api::HealthCheckRequest {}))
            .await
            .expect_err("no heartbeat within the window must not report healthy");
        assert_eq!(error.code(), Code::Unavailable);
        let probe = stalled.next_generation().await;
        assert_eq!(probe.request.input_ids.as_deref(), Some(&[0][..]));
    }
}
