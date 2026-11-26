// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The engine's native gRPC interface (`sglang.api.v1.SglangService`),
//! served so a gRPC client can point at the router or at an engine. Each served
//! RPC is routed like its HTTP route and forwarded to the same RPC on an engine.

use crate::proxy::grpc::{GrpcResponse, Replies};
use crate::server::app::pod_id;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{outcome_from_status, RequestLogContext};
use crate::server::routes::chat::route_generate;
use crate::workers::GRPC_MAX_MESSAGE_BYTES;
use axum::http::HeaderMap;
use futures::{stream::BoxStream, StreamExt};
use sglang_api_types::api::v1 as proto;
use sglang_api_types::api::v1::sglang_service_server::{SglangService, SglangServiceServer};
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tonic::{Request, Response, Status};

type RpcResult<T> = Result<Response<T>, Status>;

fn unserved<T>(rpc: &str) -> RpcResult<T> {
    Err(Status::unimplemented(format!(
        "{rpc} is not served by sgl-router"
    )))
}

/// Serve until `shutdown` resolves, then let open calls finish.
pub async fn serve(
    listener: TcpListener,
    ctx: Arc<AppContext>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let service = SglangServiceServer::new(RouterGrpc { ctx })
        // Not the HTTP body cap (`max_chat_body_bytes`): the request goes on
        // to an engine over a GRPC_MAX_MESSAGE_BYTES channel, and grows on the
        // way, so an inbound message past half of it would fail only after
        // selection and rendering. Raising SGLANG_ROUTER_MAX_BODY_BYTES does
        // not raise this.
        .max_decoding_message_size(GRPC_MAX_MESSAGE_BYTES / 2)
        .max_encoding_message_size(GRPC_MAX_MESSAGE_BYTES);
    tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_incoming_shutdown(
            tokio_stream::wrappers::TcpListenerStream::new(listener),
            shutdown,
        )
        .await?;
    Ok(())
}

struct RouterGrpc {
    ctx: Arc<AppContext>,
}

type Routed = (Result<GrpcResponse, ApiError>, Option<RequestLogContext>);

/// The access-log line the HTTP edge middleware writes, for a gRPC call.
fn log_call(
    route: &str,
    headers: &HeaderMap,
    status: u16,
    log_context: Option<&RequestLogContext>,
    started: Instant,
) {
    tracing::info!(
        pod_id = %pod_id(),
        request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-"),
        path = route,
        status,
        outcome = log_context
            .map(|c| c.outcome)
            .unwrap_or_else(|| outcome_from_status(status))
            .as_str(),
        worker = log_context.map(|c| c.worker_url.as_str()).unwrap_or(""),
        engine_rid = log_context
            .and_then(|c| c.engine_rid.as_deref())
            .unwrap_or(""),
        model = log_context.map(|c| c.model_id.as_str()).unwrap_or(""),
        stream = log_context.is_some_and(|c| c.streaming),
        images = log_context.map_or(0, |c| c.media.images),
        videos = log_context.map_or(0, |c| c.media.videos),
        audios = log_context.map_or(0, |c| c.media.audios),
        latency_ms = started.elapsed().as_millis() as u64,
        "grpc_request",
    );
}

impl RouterGrpc {
    /// Run a routed call with the HTTP edge's metrics, in-flight count and access log.
    async fn finish(
        &self,
        headers: HeaderMap,
        call: impl Future<Output = Routed>,
    ) -> Result<Replies, Status> {
        let started = Instant::now();
        let route = "/sglang.api.v1.SglangService/Generate";
        let metrics = &self.ctx.metrics;
        metrics.record_ingress(route, "POST");
        let inflight = self.ctx.inflight_http.enter();
        let (result, log_context) = call.await;
        let status = match &result {
            Ok(response) => response.status.as_u16(),
            Err(error) => error.status_code().as_u16(),
        };
        metrics.record_response(route, "POST", status);
        log_call(route, &headers, status, log_context.as_ref(), started);
        let replies = result.map_err(ApiError::into_grpc_status)?.replies;
        // Counted until the client's stream ends, as an HTTP body is.
        Ok(replies
            .map(move |reply| {
                let _ = &inflight;
                reply
            })
            .boxed())
    }
}

// async_trait must see the expanded methods, including the unserved RPCs.
macro_rules! sglang_service {
    (
        $($rpc:ident($req:ident) -> $resp:ident;)*
        [$($stream_rpc:ident($stream_req:ident) -> $stream_ty:ident<$item:ident>;)*]
    ) => {
        #[tonic::async_trait]
        impl SglangService for RouterGrpc {
            type GenerateStream = Replies;

            async fn generate(
                &self,
                request: Request<proto::GenerateRequest>,
            ) -> RpcResult<Self::GenerateStream> {
                let (metadata, _, request) = request.into_parts();
                let headers = metadata.into_headers();
                let call = route_generate(&self.ctx, request, headers.clone());
                self.finish(headers, call).await.map(Response::new)
            }

            $(
                async fn $rpc(&self, _: Request<proto::$req>) -> RpcResult<proto::$resp> {
                    unserved(stringify!($rpc))
                }
            )*

            $(
                type $stream_ty = BoxStream<'static, Result<proto::$item, Status>>;

                async fn $stream_rpc(
                    &self,
                    _: Request<proto::$stream_req>,
                ) -> RpcResult<Self::$stream_ty> {
                    unserved(stringify!($stream_rpc))
                }
            )*
        }
    };
}

sglang_service! {
    health_check(HealthCheckRequest) -> HealthCheckResponse;
    get_model_info(GetModelInfoRequest) -> GetModelInfoResponse;
    get_server_info(GetServerInfoRequest) -> GetServerInfoResponse;
    get_hi_cache_storage(GetHiCacheStorageRequest) -> HiCacheStatus;
    attach_hi_cache_storage(AttachHiCacheStorageRequest) -> HiCacheStorageResponse;
    detach_hi_cache_storage(DetachHiCacheStorageRequest) -> HiCacheStorageResponse;
    clear_hi_cache_storage(ClearHiCacheStorageRequest) -> HiCacheStorageResponse;
    list_models(ListModelsRequest) -> ListModelsResponse;
    detokenize(DetokenizeRequest) -> DetokenizeResponse;
    abort(AbortRequest) -> AbortResponse;
    set_internal_state(SetInternalStateRequest) -> SetInternalStateResponse;
    start_profile(StartProfileRequest) -> ProfileResponse;
    stop_profile(StopProfileRequest) -> ProfileResponse;
    start_expert_distribution_record(ExpertDistributionRecordRequest) -> ExpertDistributionRecordResponse;
    stop_expert_distribution_record(ExpertDistributionRecordRequest) -> ExpertDistributionRecordResponse;
    dump_expert_distribution_record(ExpertDistributionRecordRequest) -> ExpertDistributionRecordResponse;
    slow_down(SlowDownRequest) -> SlowDownResponse;
    release_memory_occupation(ReleaseMemoryOccupationRequest) -> MemoryOccupationResponse;
    resume_memory_occupation(ResumeMemoryOccupationRequest) -> MemoryOccupationResponse;
    get_weights_by_name(GetWeightsByNameRequest) -> GetWeightsByNameResponse;
    init_weights_update_group(InitWeightsUpdateGroupRequest) -> WeightGroupResponse;
    destroy_weights_update_group(DestroyWeightsUpdateGroupRequest) -> WeightGroupResponse;
    init_weights_send_group_for_remote_instance(InitWeightsSendGroupForRemoteInstanceRequest) -> WeightGroupResponse;
    send_weights_to_remote_instance(SendWeightsToRemoteInstanceRequest) -> WeightGroupResponse;
    update_weights_from_disk(UpdateWeightsFromDiskRequest) -> UpdateWeightsFromDiskResponse;
    update_weights_from_distributed(UpdateWeightsFromDistributedRequest) -> WeightUpdateResponse;
    update_weights_from_tensor(UpdateWeightsFromTensorRequest) -> WeightUpdateResponse;
    update_weights_from_ipc(UpdateWeightsFromIpcRequest) -> WeightUpdateResponse;
    begin_weight_update(BeginWeightUpdateRequest) -> WeightUpdateResponse;
    end_weight_update(EndWeightUpdateRequest) -> WeightUpdateResponse;
    plugin_weight_update(PluginWeightUpdateRequest) -> PluginWeightUpdateResponse;
    load_lora_adapter(LoadLoraAdapterRequest) -> LoraUpdateResponse;
    load_lora_adapter_from_tensors(LoadLoraAdapterFromTensorsRequest) -> LoraUpdateResponse;
    unload_lora_adapter(UnloadLoraAdapterRequest) -> LoraUpdateResponse;
    get_remote_instance_transfer_engine_info(RemoteInstanceTransferEngineInfoRequest) -> RemoteInstanceTransferEngineInfoResponse;
    parse_function_call(ParseFunctionCallRequest) -> ParseFunctionCallResponse;
    separate_reasoning(SeparateReasoningRequest) -> SeparateReasoningResponse;
    add_external_corpus(AddExternalCorpusRequest) -> AddExternalCorpusResponse;
    remove_external_corpus(RemoveExternalCorpusRequest) -> RemoveExternalCorpusResponse;
    list_external_corpora(ListExternalCorporaRequest) -> ListExternalCorporaResponse;
    freeze_gc(FreezeGcRequest) -> FreezeGcResponse;
    pause_generation(PauseGenerationRequest) -> GenerationPauseResponse;
    continue_generation(ContinueGenerationRequest) -> GenerationPauseResponse;
    pd_role_switch(PdRoleSwitchRequest) -> PdRoleSwitchResponse;
    update_weight_version(UpdateWeightVersionRequest) -> UpdateWeightVersionResponse;
    open_session(OpenSessionRequest) -> OpenSessionResponse;
    close_session(CloseSessionRequest) -> CloseSessionResponse;
    get_hi_cache_l3_cache_layout(GetHiCacheL3CacheLayoutRequest) -> GetHiCacheL3CacheLayoutResponse;
    configure_logging(ConfigureLoggingRequest) -> ConfigureLoggingResponse;
    dumper_control(DumperControlRequest) -> DumperControlResponse;
    scale_elastic_ep(ScaleElasticEpRequest) -> ScaleElasticEpResponse;
    get_elastic_ep_state(ElasticEpStateRequest) -> ElasticEpStateResponse;
    ollama_tags(OllamaTagsRequest) -> OllamaTagsResponse;
    ollama_show(OllamaShowRequest) -> OllamaShowResponse;
    anthropic_messages(AnthropicMessagesRequest) -> AnthropicMessagesResponse;
    anthropic_count_tokens(AnthropicCountTokensRequest) -> AnthropicCountTokensResponse;
    open_ai_completions(OpenAiCompletionRequest) -> OpenAiCompletionResponse;
    open_ai_chat_completions(OpenAiChatCompletionRequest) -> OpenAiChatCompletionResponse;
    open_ai_embeddings(OpenAiEmbeddingRequest) -> OpenAiEmbeddingResponse;
    open_ai_classify(OpenAiClassifyRequest) -> OpenAiClassifyResponse;
    open_ai_score(OpenAiScoreRequest) -> OpenAiScoreResponse;
    open_ai_rerank(OpenAiRerankRequest) -> OpenAiRerankResponse;
    open_ai_tokenize(OpenAiTokenizeRequest) -> OpenAiTokenizeResponse;
    open_ai_detokenize(OpenAiDetokenizeRequest) -> OpenAiDetokenizeResponse;
    open_ai_responses(OpenAiResponsesRequest) -> OpenAiResponsesResponse;
    open_ai_list_models(OpenAiListModelsRequest) -> OpenAiModelList;
    open_ai_retrieve_model(OpenAiRetrieveModelRequest) -> OpenAiModelCard;

    [
        ollama_chat(OllamaChatRequest) -> OllamaChatStream<OllamaChatResponse>;
        ollama_generate(OllamaGenerateRequest) -> OllamaGenerateStream<OllamaGenerateResponse>;
        anthropic_stream_messages(AnthropicMessagesRequest) -> AnthropicStreamMessagesStream<AnthropicStreamEvent>;
        open_ai_stream_completions(OpenAiCompletionRequest) -> OpenAiStreamCompletionsStream<OpenAiCompletionStreamResponse>;
        open_ai_stream_chat_completions(OpenAiChatCompletionRequest) -> OpenAiStreamChatCompletionsStream<OpenAiChatCompletionStreamResponse>;
    ]
}
