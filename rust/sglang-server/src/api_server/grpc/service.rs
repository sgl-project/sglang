//! The `SglangService` implementation: per-RPC translation between the
//! `api.v1` messages and the shared core.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::Stream;
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_server::SglangService;
use tonic::{Request, Response, Status};

use super::{info, native_api};
use crate::api_server::core::HealthStatus;
use crate::api_server::handlers::{CoreHandle, HiCacheManager};
use crate::message::config::{PreferredSamplingParams, ServerArgs};
use crate::message::request::{GenerationPolicy, into_requests};
use crate::message::wire::{ClientMaxNewTokens, fill_preferred_sampling};
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
    has_tokenizer: bool,
    generation_policy: GenerationPolicy,
}

/// Tonic-facing implementation backed by the transport-neutral core.
pub(crate) struct GrpcService {
    core: CoreHandle,
    config: AdapterConfig,
    hicache_manager: Arc<HiCacheManager>,
}

impl GrpcService {
    pub(crate) fn new(
        core: CoreHandle,
        server_args: &ServerArgs,
        hicache_manager: Arc<HiCacheManager>,
    ) -> Self {
        Self {
            core,
            config: AdapterConfig {
                preferred_sampling_params: server_args.preferred_sampling_params.clone(),
                incremental_streaming_output: server_args.incremental_streaming_output,
                response_timeout: DEFAULT_RESPONSE_TIMEOUT,
                health_timeout: Duration::from_secs(
                    environ::env_i64("SGLANG_HEALTH_CHECK_TIMEOUT", 20).max(0) as u64,
                ),
                has_tokenizer: !server_args.skip_tokenizer_init,
                generation_policy: server_args.into(),
            },
            hicache_manager,
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
            hicache_manager: Arc::new(HiCacheManager::new(core.clone(), &ServerArgs::default())),
            core,
            config: AdapterConfig {
                preferred_sampling_params,
                incremental_streaming_output,
                response_timeout,
                health_timeout: Duration::from_millis(20),
                has_tokenizer: true,
                generation_policy: GenerationPolicy::default(),
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
        // Read before the fills below: only a `max_new_tokens` the client set
        // is checked against the context window.
        let client_max_new_tokens = ClientMaxNewTokens::of_proto(request.sampling_params.as_ref());
        if let Some(preferred) = &self.config.preferred_sampling_params {
            request.sampling_params =
                fill_preferred_sampling(request.sampling_params.take(), &preferred.0)
                    .map_err(Status::internal)?;
        }
        // Protobuf has no null: a field the client left unset takes the
        // schema default, as an absent JSON key does.
        request.apply_absent_defaults();
        let stream = request.stream.unwrap_or(false);
        let (mut payloads, is_batch) = into_requests(request, &self.config.generation_policy)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        client_max_new_tokens.mark(&mut payloads);
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

    async fn get_hi_cache_storage(
        &self,
        _request: Request<api::GetHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStatus>, Status> {
        Ok(Response::new(self.hicache_manager.status().await))
    }

    async fn attach_hi_cache_storage(
        &self,
        request: Request<api::AttachHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStorageResponse>, Status> {
        let output = self
            .hicache_manager
            .attach(request.into_inner())
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(output))
    }

    async fn detach_hi_cache_storage(
        &self,
        _request: Request<api::DetachHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStorageResponse>, Status> {
        let output = self
            .hicache_manager
            .detach()
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(output))
    }

    async fn clear_hi_cache_storage(
        &self,
        _request: Request<api::ClearHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStorageResponse>, Status> {
        let output = self
            .hicache_manager
            .clear()
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(output))
    }

    async fn list_models(
        &self,
        _request: Request<api::ListModelsRequest>,
    ) -> Result<Response<api::ListModelsResponse>, Status> {
        Ok(Response::new(api::ListModelsResponse {
            models: self.core.list_models(),
        }))
    }

    async fn detokenize(
        &self,
        request: Request<api::DetokenizeRequest>,
    ) -> Result<Response<api::DetokenizeResponse>, Status> {
        if !self.config.has_tokenizer {
            return Err(Status::unimplemented("detokenization requires a tokenizer"));
        }
        let text = tokio::time::timeout(
            self.config.response_timeout,
            self.core.detokenize(
                request.into_inner().tokens,
                true, /* skip_special_tokens */
            ),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("detokenization timed out"))?
        .map_err(native_api::status)?;
        Ok(Response::new(api::DetokenizeResponse { text }))
    }

    async fn abort(
        &self,
        request: Request<api::AbortRequest>,
    ) -> Result<Response<api::AbortResponse>, Status> {
        let request = request.into_inner();
        if request.abort_all != request.rid.is_empty() {
            return Err(Status::invalid_argument(
                "provide either a nonempty rid or abort_all=true, not both",
            ));
        }
        Ok(Response::new(api::AbortResponse {
            success: self
                .core
                .abort((!request.abort_all).then_some(request.rid.as_str())),
        }))
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
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::{ModelConfig, PreferredSamplingParams, ServerArgs};
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
                    multimodal: false,
                    peer_endpoints: Vec::new(),
                    metadata: CoreMetadata::from(&ServerArgs {
                        served_model_name: "test-model".into(),
                        model_config: ModelConfig {
                            context_len: 4096,
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                    metrics: None,
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
            cached_tokens: 2,
            reasoning_tokens: 0,
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
        assert_eq!(first_meta.cached_tokens, 2);
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

    /// Incremental frames carry each delta with the cumulative completion
    /// count and cached-token split, which a scheduler abort does not carry.
    #[tokio::test]
    async fn incremental_frames_carry_deltas_with_cumulative_count() {
        use crate::message::response::{CachedTokensDetails, ChunkExtras};

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
        let mut first = chunk(&intake.rid, "A", 10, false);
        first.extras = Some(Box::new(ChunkExtras {
            cached_tokens_details: Some(CachedTokensDetails {
                device: 1,
                host: 1,
                storage: None,
                storage_backend: None,
            }),
            ..Default::default()
        }));
        intake.sink.try_send(ResponseItem::Frame(first)).unwrap();
        let mut aborted = chunk(&intake.rid, "B", 11, true);
        aborted.extras = Some(Box::new(ChunkExtras {
            abort_output: true,
            ..Default::default()
        }));
        intake.sink.try_send(ResponseItem::Done(aborted)).unwrap();

        let first = frame(stream.next().await.unwrap().unwrap());
        let finished = frame(stream.next().await.unwrap().unwrap());
        assert_eq!(output_ids(&first), [10]);
        assert_eq!(output_ids(&finished), [11]);
        assert_eq!(finished.text, "B");
        assert_eq!(meta(&finished).completion_tokens, 2);
        let split = Some(api::CachedTokensDetails {
            device: 1,
            host: 1,
            storage: None,
            storage_backend: None,
        });
        assert_eq!(meta(&first).cached_tokens_details, split);
        assert_eq!(meta(&finished).cached_tokens_details, split);
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
                multimodal: false,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::default(),
                metrics: None,
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
    /// A filled `max_new_tokens` is marked defaulted; a set one is the client's.
    #[tokio::test]
    async fn preferred_sampling_params_fill_unset_fields() {
        let preferred = PreferredSamplingParams(serde_json::json!({
            "temperature": 0.25,
            "max_new_tokens": 32,
        }));
        let harness = Harness::build(1, false, Duration::from_secs(1), true, Some(preferred));
        for (max_new_tokens, expected) in [(None, (Some(32), true)), (Some(8), (Some(8), false))] {
            let mut request = ids_request(&[1], None);
            request.sampling_params = Some(api::SamplingParamsOrList {
                value: Some(api::sampling_params_or_list::Value::One(
                    api::SamplingParams {
                        temperature: Some(0.8),
                        max_new_tokens,
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
            assert_eq!(
                (
                    intake.request.sampling_params.max_new_tokens,
                    intake.request.max_new_tokens_defaulted,
                ),
                expected
            );
        }
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
        assert!(intake.request.max_new_tokens_defaulted);
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

    /// Check gRPC validation, scheduler results, and the configuration returned by GET.
    #[tokio::test]
    async fn hicache_controls_preserve_results_and_configuration() {
        let h = Harness::new(8, false, Duration::from_secs(5));
        let service = &h.service;
        let initial = service
            .get_hi_cache_storage(Request::new(api::GetHiCacheStorageRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(initial, api::HiCacheStatus::default());
        let error = service
            .attach_hi_cache_storage(Request::new(api::AttachHiCacheStorageRequest::default()))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(h.intake_rx.is_empty());

        for success in [false, true] {
            let response = hicache_reply(
                &h,
                "AttachHiCacheStorageReqInput",
                success,
                service.attach_hi_cache_storage(Request::new(api::AttachHiCacheStorageRequest {
                    hicache_storage_backend: Some("file".into()),
                    hicache_write_policy: Some("write_through".into()),
                    ..Default::default()
                })),
            )
            .await;
            assert_eq!(response.success, success);
            assert_eq!(response.message, "backend detail");
        }
        for success in [false, true] {
            let response = hicache_reply(
                &h,
                "ClearHiCacheReqInput",
                success,
                service.clear_hi_cache_storage(Request::new(api::ClearHiCacheStorageRequest {})),
            )
            .await;
            assert_eq!(response.success, success);
            assert!(response.message.is_empty());
        }
        for success in [false, true] {
            let response = hicache_reply(
                &h,
                "DetachHiCacheStorageReqInput",
                success,
                service.detach_hi_cache_storage(Request::new(api::DetachHiCacheStorageRequest {})),
            )
            .await;
            assert_eq!(response.success, success);
            assert_eq!(response.message, "backend detail");
        }
        let status = service
            .get_hi_cache_storage(Request::new(api::GetHiCacheStorageRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            status,
            api::HiCacheStatus {
                hicache_write_policy: Some("write_through".into()),
                ..Default::default()
            }
        );
    }

    // Drive the RPC and a simulated scheduler reply together, checking the request tag.
    async fn hicache_reply(
        h: &Harness,
        tag: &str,
        success: bool,
        call: impl std::future::Future<
            Output = Result<tonic::Response<api::HiCacheStorageResponse>, tonic::Status>,
        >,
    ) -> api::HiCacheStorageResponse {
        let scheduler = async {
            let TmEvent::Intake { request, admission } = h.intake_rx.recv_async().await.unwrap()
            else {
                panic!("expected control intake");
            };
            assert!(admission.try_accept());
            let RequestKind::Control(control) = &request.kind else {
                panic!("expected control request");
            };
            let wire: serde_json::Value =
                rmp_serde::from_slice(&control.encode().unwrap()).unwrap();
            assert_eq!(wire[0], tag);
            request
                .sink
                .try_send(ResponseItem::Control(bytes::Bytes::from(
                    rmp_serde::to_vec_named(
                        &serde_json::json!({"success": success, "message": "backend detail"}),
                    )
                    .unwrap(),
                )))
                .unwrap();
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            let (response, ()) = tokio::join!(call, scheduler);
            response.unwrap().into_inner()
        })
        .await
        .expect("HiCache RPC did not finish")
    }

    #[tokio::test]
    async fn list_models_uses_static_metadata_without_intake() {
        let harness = Harness::new(2, false, Duration::from_secs(1));
        let models = harness
            .service
            .list_models(Request::new(api::ListModelsRequest {}))
            .await
            .unwrap()
            .into_inner()
            .models;
        assert_eq!(
            models,
            vec![api::ModelCard {
                id: "test-model".into(),
                root: "test-model".into(),
                parent: None,
                max_model_len: 4096,
            }]
        );
        assert!(harness.intake_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn detokenize_returns_text_and_maps_invalid_ids() {
        let harness = Harness::new(2, false, Duration::from_secs(1));
        for invalid in [false, true] {
            let tokens = if invalid { vec![-1] } else { vec![1, 2] };
            let decode = harness
                .service
                .detokenize(Request::new(api::DetokenizeRequest {
                    tokens: tokens.clone(),
                }));
            let runtime = async {
                let TmEvent::Intake { request, admission } =
                    harness.intake_rx.recv_async().await.unwrap()
                else {
                    panic!("expected intake");
                };
                assert!(admission.try_accept());
                assert!(
                    matches!(&request.kind, RequestKind::Detokenize { token_ids, skip_special_tokens: true } if *token_ids == tokens)
                );
                let reply = if invalid {
                    ResponseItem::Error(crate::utils::error::Error::Validation(
                        "invalid token ID".into(),
                    ))
                } else {
                    ResponseItem::Data(bytes::Bytes::from_static("héllo".as_bytes()))
                };
                request.sink.try_send(reply).unwrap();
            };
            let (result, ()) = tokio::join!(decode, runtime);
            if invalid {
                assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
            } else {
                assert_eq!(result.unwrap().into_inner().text, "héllo");
            }
            assert!(harness.abort_rx.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn detokenize_rejects_missing_tokenizer_without_submission() {
        let mut harness = Harness::new(2, false, Duration::from_secs(1));
        harness.service.config.has_tokenizer = false;
        let error = harness
            .service
            .detokenize(Request::new(api::DetokenizeRequest { tokens: vec![1] }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::Unimplemented);
        assert!(harness.intake_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn detokenize_timeout_releases_the_internal_call() {
        let harness = Harness::new(2, false, Duration::from_millis(10));
        let decode = harness
            .service
            .detokenize(Request::new(api::DetokenizeRequest { tokens: vec![1] }));
        let runtime = async {
            let TmEvent::Intake { request, admission } =
                harness.intake_rx.recv_async().await.unwrap()
            else {
                panic!("expected intake");
            };
            assert!(admission.try_accept());
            let aborted = harness.abort_rx.recv_async().await.unwrap();
            assert_eq!(aborted.rid(), &request.rid);
        };
        let (result, ()) = tokio::join!(decode, runtime);
        assert_eq!(result.unwrap_err().code(), Code::DeadlineExceeded);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn abort_rpc_cancels_generation_with_an_in_stream_error() {
        let harness = Harness::new(2, false, Duration::from_secs(1));
        let mut stream = harness
            .service
            .generate(Request::new(ids_request(&[1], Some("abort-me"))))
            .await
            .unwrap()
            .into_inner();
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        let reply = harness
            .service
            .abort(Request::new(api::AbortRequest {
                rid: "abort-me".into(),
                abort_all: false,
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(reply.success);
        assert_eq!(harness.abort_rx.try_recv().unwrap().rid(), &intake.rid);
        let error = stream_error(stream.next().await.unwrap().unwrap());
        assert_eq!(error.error.unwrap().code, 499);
        assert!(stream.next().await.is_none());
        assert!(
            !harness
                .service
                .abort(Request::new(api::AbortRequest {
                    rid: "abort-me".into(),
                    abort_all: false,
                }))
                .await
                .unwrap()
                .into_inner()
                .success
        );
        drop(stream);
        assert!(harness.abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn abort_validates_selector_and_supports_all() {
        let harness = Harness::new(2, false, Duration::from_secs(1));
        for (rid, abort_all) in [("", false), ("ambiguous", true)] {
            let error = harness
                .service
                .abort(Request::new(api::AbortRequest {
                    rid: rid.into(),
                    abort_all,
                }))
                .await
                .unwrap_err();
            assert_eq!(error.code(), Code::InvalidArgument);
        }
        let stream = harness
            .service
            .generate(Request::new(ids_request(&[1], Some("all"))))
            .await
            .unwrap()
            .into_inner();
        let intake = harness.next_generation().await;
        assert!(intake.admission.try_accept());
        assert!(
            harness
                .service
                .abort(Request::new(api::AbortRequest {
                    rid: String::new(),
                    abort_all: true,
                }))
                .await
                .unwrap()
                .into_inner()
                .success
        );
        assert_eq!(harness.abort_rx.try_recv().unwrap().rid(), &intake.rid);
        drop(stream);
        assert!(harness.abort_rx.try_recv().is_err());
    }
}
