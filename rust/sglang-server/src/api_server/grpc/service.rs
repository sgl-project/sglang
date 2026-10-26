//! The `SglangService` implementation: per-RPC translation between the
//! `api.v1` messages and the shared core.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt};
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_server::SglangService;
use tonic::{Request, Response, Status};

use super::{info, native_api};
use crate::api_server::anthropic::Anthropic;
use crate::api_server::core::HealthStatus;
use crate::api_server::handlers::{CoreHandle, HiCacheManager};
use crate::api_server::handlers::{
    CorpusSource, ToolSpec, WeightGroupOutcome, parse_function_call, separate_reasoning,
};
use crate::api_server::http::openai::{ChatFormatter, load_chat_support};
use crate::api_server::ollama::Ollama;
use crate::message::config::{PreferredSamplingParams, ServerArgs};
use crate::message::io_struct::{
    ExpertDistributionAction, PdRoleSwitch, ProfileOptions, WeightsSendGroup, WeightsUpdateGroup,
};
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
    /// The launch args the data-plane adapters (Ollama, Anthropic) read.
    server_args: Arc<ServerArgs>,
    /// The served chat template, for the chat-shaped data planes.
    chat_formatter: Option<ChatFormatter>,
}

impl GrpcService {
    pub(crate) fn new(
        core: CoreHandle,
        server_args: Arc<ServerArgs>,
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
                generation_policy: server_args.as_ref().into(),
            },
            hicache_manager,
            chat_formatter: load_chat_support(&server_args),
            server_args,
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
            server_args: Arc::new(ServerArgs {
                model_config: crate::message::config::ModelConfig {
                    context_len: 4096,
                    vocab_size: 32_000,
                    ..Default::default()
                },
                ..Default::default()
            }),
            chat_formatter: None,
        }
    }
}

fn openai_unimplemented() -> Status {
    Status::unimplemented(
        "the OpenAI API is served over HTTP only until its adapter moves onto the api.v1 schema",
    )
}

/// A group operation's merged outcome as the gRPC reply; a frontend-side
/// failure (bad request, unreachable rank) is a status instead.
fn group_response(
    outcome: Result<WeightGroupOutcome, crate::api_server::core::CoreError>,
) -> Result<Response<api::WeightGroupResponse>, Status> {
    let outcome = outcome.map_err(native_api::status)?;
    Ok(Response::new(api::WeightGroupResponse {
        success: outcome.success,
        message: outcome.message,
    }))
}

impl GrpcService {
    fn anthropic(&self) -> Anthropic<'_> {
        Anthropic {
            core: &self.core,
            chat_formatter: self.chat_formatter.as_ref(),
            server_args: &self.server_args,
        }
    }

    fn ollama(&self) -> Ollama<'_> {
        Ollama {
            core: &self.core,
            chat_formatter: self.chat_formatter.as_ref(),
            server_args: &self.server_args,
        }
    }

    async fn expert_distribution_record(
        &self,
        action: ExpertDistributionAction,
    ) -> Result<Response<api::ExpertDistributionRecordResponse>, Status> {
        self.core
            .expert_distribution_record(action)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::ExpertDistributionRecordResponse {}))
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
        info::model_info(self.core.model_info().await)
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

    async fn set_internal_state(
        &self,
        request: Request<api::SetInternalStateRequest>,
    ) -> Result<Response<api::SetInternalStateResponse>, Status> {
        let updated = self
            .core
            .set_internal_state(request.into_inner().server_args.as_deref())
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::SetInternalStateResponse { updated }))
    }

    async fn start_profile(
        &self,
        request: Request<api::StartProfileRequest>,
    ) -> Result<Response<api::ProfileResponse>, Status> {
        let request = request.into_inner();
        let options = ProfileOptions {
            output_dir: request.output_dir,
            start_step: request.start_step,
            num_steps: request.num_steps,
            activities: request.activities.map(|list| list.items),
            profile_by_stage: request.profile_by_stage,
            with_stack: request.with_stack,
            record_shapes: request.record_shapes,
            profile_id: request.profile_id,
            merge_profiles: request.merge_profiles,
            profile_prefix: request.profile_prefix,
            profile_stages: request.profile_stages.map(|list| list.items),
            detailed_annotations: request.detailed_annotations,
        };
        let outcome = self
            .core
            .start_profile(options)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::ProfileResponse {
            success: outcome.success,
            message: outcome.message,
        }))
    }

    async fn stop_profile(
        &self,
        _request: Request<api::StopProfileRequest>,
    ) -> Result<Response<api::ProfileResponse>, Status> {
        let outcome = self.core.stop_profile().await.map_err(native_api::status)?;
        Ok(Response::new(api::ProfileResponse {
            success: outcome.success,
            message: outcome.message,
        }))
    }

    async fn start_expert_distribution_record(
        &self,
        _request: Request<api::ExpertDistributionRecordRequest>,
    ) -> Result<Response<api::ExpertDistributionRecordResponse>, Status> {
        self.expert_distribution_record(ExpertDistributionAction::Start)
            .await
    }

    async fn stop_expert_distribution_record(
        &self,
        _request: Request<api::ExpertDistributionRecordRequest>,
    ) -> Result<Response<api::ExpertDistributionRecordResponse>, Status> {
        self.expert_distribution_record(ExpertDistributionAction::Stop)
            .await
    }

    async fn dump_expert_distribution_record(
        &self,
        _request: Request<api::ExpertDistributionRecordRequest>,
    ) -> Result<Response<api::ExpertDistributionRecordResponse>, Status> {
        self.expert_distribution_record(ExpertDistributionAction::Dump)
            .await
    }

    async fn slow_down(
        &self,
        request: Request<api::SlowDownRequest>,
    ) -> Result<Response<api::SlowDownResponse>, Status> {
        self.core
            .slow_down(request.into_inner().forward_sleep_time)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::SlowDownResponse {}))
    }

    async fn release_memory_occupation(
        &self,
        request: Request<api::ReleaseMemoryOccupationRequest>,
    ) -> Result<Response<api::MemoryOccupationResponse>, Status> {
        self.core
            .release_memory_occupation(request.into_inner().tags.map(|list| list.items))
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::MemoryOccupationResponse {}))
    }

    async fn resume_memory_occupation(
        &self,
        request: Request<api::ResumeMemoryOccupationRequest>,
    ) -> Result<Response<api::MemoryOccupationResponse>, Status> {
        self.core
            .resume_memory_occupation(request.into_inner().tags.map(|list| list.items))
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::MemoryOccupationResponse {}))
    }

    async fn get_weights_by_name(
        &self,
        request: Request<api::GetWeightsByNameRequest>,
    ) -> Result<Response<api::GetWeightsByNameResponse>, Status> {
        let request = request.into_inner();
        let parameters = self
            .core
            .get_weights_by_name(request.name, request.truncate_size.unwrap_or(100))
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::GetWeightsByNameResponse {
            parameters: parameters.iter().map(ToString::to_string).collect(),
        }))
    }

    async fn init_weights_update_group(
        &self,
        request: Request<api::InitWeightsUpdateGroupRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        let request = request.into_inner();
        group_response(
            self.core
                .init_weights_update_group(WeightsUpdateGroup {
                    master_address: request.master_address,
                    master_port: request.master_port,
                    rank_offset: request.rank_offset,
                    world_size: request.world_size,
                    group_name: request
                        .group_name
                        .unwrap_or_else(|| "weight_update_group".into()),
                    backend: request.backend.unwrap_or_else(|| "nccl".into()),
                })
                .await,
        )
    }

    async fn destroy_weights_update_group(
        &self,
        request: Request<api::DestroyWeightsUpdateGroupRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        let group_name = request
            .into_inner()
            .group_name
            .unwrap_or_else(|| "weight_update_group".into());
        group_response(self.core.destroy_weights_update_group(group_name).await)
    }

    async fn init_weights_send_group_for_remote_instance(
        &self,
        request: Request<api::InitWeightsSendGroupForRemoteInstanceRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        let request = request.into_inner();
        group_response(
            self.core
                .init_weights_send_group_for_remote_instance(WeightsSendGroup {
                    master_address: request.master_address,
                    ports: request.ports,
                    group_rank: request.group_rank,
                    world_size: request.world_size,
                    group_name: request
                        .group_name
                        .unwrap_or_else(|| "weight_send_group".into()),
                    backend: request.backend.unwrap_or_else(|| "nccl".into()),
                })
                .await,
        )
    }

    async fn send_weights_to_remote_instance(
        &self,
        request: Request<api::SendWeightsToRemoteInstanceRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        let request = request.into_inner();
        group_response(
            self.core
                .send_weights_to_remote_instance(
                    request.master_address,
                    request.ports,
                    request
                        .group_name
                        .unwrap_or_else(|| "weight_send_group".into()),
                )
                .await,
        )
    }

    async fn parse_function_call(
        &self,
        request: Request<api::ParseFunctionCallRequest>,
    ) -> Result<Response<api::ParseFunctionCallResponse>, Status> {
        let request = request.into_inner();
        let tools: Vec<ToolSpec> = request
            .tools
            .into_iter()
            .map(|tool| {
                let function = tool.function.unwrap_or_default();
                ToolSpec {
                    name: function.name,
                    parameters: function.parameters,
                }
            })
            .collect();
        let (normal_text, calls) =
            parse_function_call(&request.text, &tools, request.tool_call_parser.as_deref())
                .await
                .map_err(native_api::status)?;
        Ok(Response::new(api::ParseFunctionCallResponse {
            normal_text,
            calls: calls
                .into_iter()
                .map(|call| api::ParsedToolCall {
                    tool_index: call.tool_index,
                    name: call.name,
                    parameters: call.parameters,
                })
                .collect(),
        }))
    }

    async fn separate_reasoning(
        &self,
        request: Request<api::SeparateReasoningRequest>,
    ) -> Result<Response<api::SeparateReasoningResponse>, Status> {
        let request = request.into_inner();
        let (reasoning_text, text) = separate_reasoning(&request.text, &request.reasoning_parser)
            .map_err(native_api::status)?;
        let blocks = request.return_blocks;
        Ok(Response::new(api::SeparateReasoningResponse {
            reasoning_blocks: blocks.then(|| api::StringList {
                items: if reasoning_text.is_empty() {
                    Vec::new()
                } else {
                    vec![reasoning_text.clone()]
                },
            }),
            text_blocks: blocks.then(|| api::StringList {
                items: vec![text.clone()],
            }),
            reasoning_text,
            text,
        }))
    }

    async fn add_external_corpus(
        &self,
        request: Request<api::AddExternalCorpusRequest>,
    ) -> Result<Response<api::AddExternalCorpusResponse>, Status> {
        let request = request.into_inner();
        let outcome = self
            .core
            .add_external_corpus(CorpusSource {
                corpus_id: request.corpus_id,
                file_path: request.file_path,
                documents: request.documents.map(|list| list.items),
                token_chunks: request
                    .token_chunks
                    .map(|list| list.items.into_iter().map(|ids| ids.ids).collect()),
            })
            .await;
        Ok(Response::new(api::AddExternalCorpusResponse {
            success: outcome.success,
            corpus_id: outcome.corpus_id,
            message: outcome.message,
            loaded_token_count: outcome.loaded_token_count,
        }))
    }

    async fn remove_external_corpus(
        &self,
        request: Request<api::RemoveExternalCorpusRequest>,
    ) -> Result<Response<api::RemoveExternalCorpusResponse>, Status> {
        let Some(corpus_id) = request.into_inner().corpus_id.filter(|id| !id.is_empty()) else {
            return Err(Status::invalid_argument("corpus_id is required."));
        };
        let outcome = self.core.remove_external_corpus(corpus_id).await;
        Ok(Response::new(api::RemoveExternalCorpusResponse {
            success: outcome.success,
            message: outcome.message,
        }))
    }

    async fn list_external_corpora(
        &self,
        _request: Request<api::ListExternalCorporaRequest>,
    ) -> Result<Response<api::ListExternalCorporaResponse>, Status> {
        let listing = self.core.list_external_corpora().await;
        Ok(Response::new(api::ListExternalCorporaResponse {
            success: listing.success,
            corpus_token_counts: listing.corpus_token_counts.into_iter().collect(),
            message: listing.message,
        }))
    }

    async fn freeze_gc(
        &self,
        _request: Request<api::FreezeGcRequest>,
    ) -> Result<Response<api::FreezeGcResponse>, Status> {
        self.core.freeze_gc().await.map_err(native_api::status)?;
        Ok(Response::new(api::FreezeGcResponse {}))
    }

    async fn pause_generation(
        &self,
        request: Request<api::PauseGenerationRequest>,
    ) -> Result<Response<api::GenerationPauseResponse>, Status> {
        let mode = request.into_inner().mode.unwrap_or_else(|| "abort".into());
        self.core
            .pause_generation(&mode)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::GenerationPauseResponse {
            message: "Generation paused successfully.".into(),
            status: "ok".into(),
        }))
    }

    async fn continue_generation(
        &self,
        request: Request<api::ContinueGenerationRequest>,
    ) -> Result<Response<api::GenerationPauseResponse>, Status> {
        self.core
            .continue_generation(request.into_inner().torch_empty_cache.unwrap_or(true))
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::GenerationPauseResponse {
            message: "Generation continued successfully.".into(),
            status: "ok".into(),
        }))
    }

    async fn pd_role_switch(
        &self,
        request: Request<api::PdRoleSwitchRequest>,
    ) -> Result<Response<api::PdRoleSwitchResponse>, Status> {
        let request = request.into_inner();
        let outcome = self
            .core
            .pd_role_switch(PdRoleSwitch {
                new_role: request.new_role.unwrap_or_default(),
                decode_cuda_graph_bs: request.decode_cuda_graph_bs.map(|list| list.items),
                decode_cuda_graph_memory_gb: request.decode_cuda_graph_memory_gb,
                prepare_only: request.prepare_only,
            })
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::PdRoleSwitchResponse {
            success: outcome.success,
            message: outcome.message,
            old_role: outcome.old_role,
            new_role: outcome.new_role,
            safe_to_restore: outcome.safe_to_restore,
        }))
    }

    async fn update_weight_version(
        &self,
        request: Request<api::UpdateWeightVersionRequest>,
    ) -> Result<Response<api::UpdateWeightVersionResponse>, Status> {
        let request = request.into_inner();
        let new_version = request.new_version;
        match self
            .core
            .update_weight_version(
                new_version.clone(),
                request.abort_all_requests.unwrap_or(true),
            )
            .await
        {
            Ok(()) => Ok(Response::new(api::UpdateWeightVersionResponse {
                success: true,
                message: format!("Weight version updated to {new_version}"),
                new_version: Some(new_version),
            })),
            Err(error) => Ok(Response::new(api::UpdateWeightVersionResponse {
                success: false,
                message: format!("Failed to update weight version: {error}"),
                new_version: None,
            })),
        }
    }

    async fn get_remote_instance_transfer_engine_info(
        &self,
        request: Request<api::RemoteInstanceTransferEngineInfoRequest>,
    ) -> Result<Response<api::RemoteInstanceTransferEngineInfoResponse>, Status> {
        let info = self
            .core
            .remote_instance_transfer_engine_info(request.into_inner().rank)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(
            api::RemoteInstanceTransferEngineInfoResponse { info: Some(info) },
        ))
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
            models: self.core.list_models().await,
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

    type OllamaChatStream = ResponseStream<api::OllamaChatResponse>;

    /// `/api/chat` over gRPC: the adapter's frames, errors as statuses.
    async fn ollama_chat(
        &self,
        request: Request<api::OllamaChatRequest>,
    ) -> Result<Response<Self::OllamaChatStream>, Status> {
        let mut request = request.into_inner();
        request.apply_absent_defaults();
        let frames = self
            .ollama()
            .chat(request)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(Box::pin(
            frames.map(|frame| frame.map_err(native_api::status)),
        )))
    }

    type OllamaGenerateStream = ResponseStream<api::OllamaGenerateResponse>;

    async fn ollama_generate(
        &self,
        request: Request<api::OllamaGenerateRequest>,
    ) -> Result<Response<Self::OllamaGenerateStream>, Status> {
        let mut request = request.into_inner();
        request.apply_absent_defaults();
        let frames = self
            .ollama()
            .generate(request)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(Box::pin(
            frames.map(|frame| frame.map_err(native_api::status)),
        )))
    }

    async fn ollama_tags(
        &self,
        _request: Request<api::OllamaTagsRequest>,
    ) -> Result<Response<api::OllamaTagsResponse>, Status> {
        Ok(Response::new(self.ollama().tags().await))
    }

    async fn ollama_show(
        &self,
        request: Request<api::OllamaShowRequest>,
    ) -> Result<Response<api::OllamaShowResponse>, Status> {
        Ok(Response::new(self.ollama().show(request.into_inner())))
    }

    /// `/v1/messages` without `stream`: failures are statuses with the
    /// Anthropic message.
    async fn anthropic_messages(
        &self,
        request: Request<api::AnthropicMessagesRequest>,
    ) -> Result<Response<api::AnthropicMessagesResponse>, Status> {
        let mut request = request.into_inner();
        request.apply_absent_defaults();
        self.anthropic()
            .messages(request)
            .await
            .map(Response::new)
            .map_err(|failure| failure.grpc_status())
    }

    type AnthropicStreamMessagesStream = ResponseStream<api::AnthropicStreamEvent>;

    /// `/v1/messages` with `stream`: the event sequence; a failure after
    /// the first event is an `error` event, as on SSE.
    async fn anthropic_stream_messages(
        &self,
        request: Request<api::AnthropicMessagesRequest>,
    ) -> Result<Response<Self::AnthropicStreamMessagesStream>, Status> {
        let mut request = request.into_inner();
        request.apply_absent_defaults();
        let events = self
            .anthropic()
            .messages_stream(request)
            .await
            .map_err(|failure| failure.grpc_status())?;
        Ok(Response::new(Box::pin(events.map(Ok))))
    }

    async fn anthropic_count_tokens(
        &self,
        request: Request<api::AnthropicCountTokensRequest>,
    ) -> Result<Response<api::AnthropicCountTokensResponse>, Status> {
        self.anthropic()
            .count_tokens(request.into_inner())
            .await
            .map(Response::new)
            .map_err(|failure| failure.grpc_status())
    }

    // The OpenAI interface: the schema is in place for the adapter
    // refactor; the gRPC methods answer UNIMPLEMENTED until it lands.
    async fn open_ai_completions(
        &self,
        _request: Request<api::OpenAiCompletionRequest>,
    ) -> Result<Response<api::OpenAiCompletionResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_chat_completions(
        &self,
        _request: Request<api::OpenAiChatCompletionRequest>,
    ) -> Result<Response<api::OpenAiChatCompletionResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_embeddings(
        &self,
        _request: Request<api::OpenAiEmbeddingRequest>,
    ) -> Result<Response<api::OpenAiEmbeddingResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_classify(
        &self,
        _request: Request<api::OpenAiClassifyRequest>,
    ) -> Result<Response<api::OpenAiClassifyResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_score(
        &self,
        _request: Request<api::OpenAiScoreRequest>,
    ) -> Result<Response<api::OpenAiScoreResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_rerank(
        &self,
        _request: Request<api::OpenAiRerankRequest>,
    ) -> Result<Response<api::OpenAiRerankResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_tokenize(
        &self,
        _request: Request<api::OpenAiTokenizeRequest>,
    ) -> Result<Response<api::OpenAiTokenizeResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_detokenize(
        &self,
        _request: Request<api::OpenAiDetokenizeRequest>,
    ) -> Result<Response<api::OpenAiDetokenizeResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_responses(
        &self,
        _request: Request<api::OpenAiResponsesRequest>,
    ) -> Result<Response<api::OpenAiResponsesResponse>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_list_models(
        &self,
        _request: Request<api::OpenAiListModelsRequest>,
    ) -> Result<Response<api::OpenAiModelList>, Status> {
        Err(openai_unimplemented())
    }

    async fn open_ai_retrieve_model(
        &self,
        _request: Request<api::OpenAiRetrieveModelRequest>,
    ) -> Result<Response<api::OpenAiModelCard>, Status> {
        Err(openai_unimplemented())
    }

    type OpenAiStreamCompletionsStream = ResponseStream<api::OpenAiCompletionStreamResponse>;

    async fn open_ai_stream_completions(
        &self,
        _request: Request<api::OpenAiCompletionRequest>,
    ) -> Result<Response<Self::OpenAiStreamCompletionsStream>, Status> {
        Err(openai_unimplemented())
    }

    type OpenAiStreamChatCompletionsStream =
        ResponseStream<api::OpenAiChatCompletionStreamResponse>;

    async fn open_ai_stream_chat_completions(
        &self,
        _request: Request<api::OpenAiChatCompletionRequest>,
    ) -> Result<Response<Self::OpenAiStreamChatCompletionsStream>, Status> {
        Err(openai_unimplemented())
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
                    peer_endpoints: Vec::new(),
                    metadata: CoreMetadata::from(&ServerArgs {
                        served_model_name: "test-model".into(),
                        model_config: ModelConfig {
                            context_len: 4096,
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
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
                peer_endpoints: Vec::new(),
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

    /// `OllamaChat` renders the messages with the served template and streams
    /// the adapter's frames, counts on the done frame; `OllamaShow` needs no
    /// intake.
    #[tokio::test]
    async fn ollama_chat_streams_frames_and_show_is_static() {
        let mut harness = Harness::new(2, true, Duration::from_secs(1));
        harness.service.chat_formatter =
            Some(crate::api_server::http::openai::test_utils::chatml_formatter());
        let request = api::OllamaChatRequest {
            model: "m".into(),
            messages: vec![api::OllamaMessage {
                role: "user".into(),
                content: "hi".into(),
                images: None,
            }],
            ..Default::default()
        };
        let rpc = harness.service.ollama_chat(Request::new(request));
        let runtime = async {
            let intake = harness.next_generation().await;
            assert!(intake.admission.try_accept());
            assert!(intake.request.text.as_deref().unwrap().contains("hi"));
            assert_eq!(intake.request.sampling_params.max_new_tokens, Some(2048));
            intake
                .sink
                .try_send(ResponseItem::Frame(chunk(&intake.rid, "o", 7, false)))
                .unwrap();
            intake
                .sink
                .try_send(ResponseItem::Done(chunk(&intake.rid, "k", 8, true)))
                .unwrap();
        };
        let (response, ()) = tokio::join!(rpc, runtime);
        let mut stream = response.unwrap().into_inner();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.message.as_ref().unwrap().content, "o");
        assert!(!first.done && first.eval_count.is_none());
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.message.as_ref().unwrap().content, "k");
        let done = stream.next().await.unwrap().unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("stop"));
        assert_eq!(done.message.as_ref().unwrap().content, "");
        assert_eq!(done.prompt_eval_count, Some(3));
        assert_eq!(done.eval_count, Some(2));
        assert!(stream.next().await.is_none());

        let card = harness
            .service
            .ollama_show(Request::new(api::OllamaShowRequest {
                model: "org/Qwen-Chat".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(card.details.unwrap().family, "Qwen");
        assert_eq!(card.parameters, "num_ctx 4096");
    }

    /// `AnthropicMessages` renders through the chat core and shapes the
    /// reply; `AnthropicCountTokens` counts the rendered prompt's ids.
    #[tokio::test]
    async fn anthropic_messages_and_count_tokens_over_grpc() {
        let mut harness = Harness::new(2, true, Duration::from_secs(1));
        harness.service.chat_formatter =
            Some(crate::api_server::http::openai::test_utils::chatml_formatter());
        let request: api::AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let rpc = harness.service.anthropic_messages(Request::new(request));
        let runtime = async {
            let intake = harness.next_generation().await;
            assert!(intake.admission.try_accept());
            assert_eq!(intake.request.sampling_params.max_new_tokens, Some(8));
            intake
                .sink
                .try_send(ResponseItem::Done(chunk(&intake.rid, "ok", 7, true)))
                .unwrap();
        };
        let (response, ()) = tokio::join!(rpc, runtime);
        let message = response.unwrap().into_inner();
        assert_eq!(message.r#type, "message");
        assert_eq!(message.stop_reason.as_deref(), Some("end_turn"));
        let content = message.content.unwrap().items;
        assert_eq!(content.len(), 1);
        assert!(matches!(
            &content[0].block,
            Some(api::anthropic_content_block::Block::Text(text)) if text.text == "ok"
        ));
        assert_eq!(message.usage.unwrap().input_tokens, Some(3));

        let request: api::AnthropicCountTokensRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let rpc = harness
            .service
            .anthropic_count_tokens(Request::new(request));
        let runtime = async {
            let TmEvent::Intake { request, admission } =
                harness.intake_rx.recv_async().await.unwrap()
            else {
                panic!("expected intake");
            };
            assert!(admission.try_accept());
            assert!(matches!(request.kind, RequestKind::Tokenize { .. }));
            request
                .sink
                .try_send(ResponseItem::TokenIds(vec![1, 2, 3, 4]))
                .unwrap();
        };
        let (response, ()) = tokio::join!(rpc, runtime);
        assert_eq!(response.unwrap().into_inner().input_tokens, 4);
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
