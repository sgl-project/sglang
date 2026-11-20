//! The `SglangService` implementation: per-RPC translation between the
//! `api.v1` messages and the shared core.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use futures::{Stream, StreamExt};
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_server::SglangService;
use tonic::{Request, Response, Status};

use super::{info, native_api};
use crate::api_server::anthropic::Anthropic;
use crate::api_server::core::{HealthStatus, ReceivedAs};
use crate::api_server::handlers::{
    AbortEnding, CoreHandle, CorpusSource, DumperOutcome, ElasticEpState, HiCacheError,
    HiCacheManager, LORA_DISABLED, ToolSpec, WeightGroupOutcome, WeightUpdateOutcome,
    parse_function_call, separate_reasoning,
};
use crate::api_server::http::openai::{ChatFormatter, load_chat_support};
use crate::api_server::http::weights::{decode_payloads, json_object};
use crate::api_server::ollama::Ollama;
use crate::message::config::{
    DEFAULT_GRPC_RESPONSE_TIMEOUT_SECS, PreferredSamplingParams, ServerArgs,
};
use crate::message::io_struct::{
    DiskWeightUpdate, DistributedWeightUpdate, ExpertDistributionAction, IpcWeightUpdate,
    LoggingConfig, PdRoleSwitch, PluginWeightUpdate, ProfileOptions, SessionOpen,
    TensorWeightUpdate, WeightsSendGroup, WeightsUpdateGroup,
};
use crate::message::request::{GenerationPolicy, into_requests};
use crate::message::wire::{ClientMaxNewTokens, fill_preferred_sampling};
use crate::utils::environ;

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
                response_timeout: response_timeout(server_args.grpc_response_timeout_secs),
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
        // With an admin key, so the HiCache tests reach the scheduler.
        let args = ServerArgs {
            api_keys: crate::message::config::ApiKeys::new(None, Some("sk-admin".into())),
            ..Default::default()
        };
        Self {
            hicache_manager: Arc::new(HiCacheManager::new(core.clone(), &args)),
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

/// A zero timeout would expire every call at once; like the legacy gRPC
/// bridge, fall back to the default instead.
fn response_timeout(secs: u64) -> Duration {
    if secs == 0 {
        tracing::warn!(
            default = DEFAULT_GRPC_RESPONSE_TIMEOUT_SECS,
            "grpc_response_timeout_secs must be positive; using default"
        );
        return Duration::from_secs(DEFAULT_GRPC_RESPONSE_TIMEOUT_SECS);
    }
    Duration::from_secs(secs)
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

/// A weight update's merged outcome as the gRPC reply; a frontend-side
/// failure (bad request, multi-rank deployment) is a status instead.
fn update_response(
    outcome: Result<WeightUpdateOutcome, crate::api_server::core::CoreError>,
) -> Result<Response<api::WeightUpdateResponse>, Status> {
    let outcome = outcome.map_err(native_api::status)?;
    Ok(Response::new(api::WeightUpdateResponse {
        success: outcome.success,
        message: outcome.message,
    }))
}

/// Python's refusal of the LoRA routes without `--enable-lora`, which
/// Rust-server mode refuses at launch.
fn lora_disabled() -> Result<Response<api::LoraUpdateResponse>, Status> {
    Ok(Response::new(api::LoraUpdateResponse {
        success: false,
        error_message: Some(LORA_DISABLED.into()),
        loaded_adapters: Default::default(),
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
            .map_err(|error| {
                tracing::warn!(%error, "rejected request");
                Status::invalid_argument(error.to_string())
            })?;
        client_max_new_tokens.mark(&mut payloads);
        // Python starts its request clock after normalization and before
        // tokenization / multimodal preprocessing; `e2e_latency` measures from
        // the same boundary here.
        let created_at = Instant::now();
        let calls = self
            .core
            .generate_batch(payloads, ReceivedAs::without_headers())
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
            Ok(HealthStatus::Draining) => {
                Err(Status::unavailable("server is draining before shutdown"))
            }
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
        self.core
            .freeze_gc_all()
            .await
            .map_err(native_api::status)?;
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

    async fn update_weights_from_disk(
        &self,
        request: Request<api::UpdateWeightsFromDiskRequest>,
    ) -> Result<Response<api::UpdateWeightsFromDiskResponse>, Status> {
        let request = request.into_inner();
        let manifest = json_object("manifest", request.manifest.as_deref())
            .map_err(Status::invalid_argument)?;
        let outcome = self
            .core
            .update_weights_from_disk(DiskWeightUpdate {
                model_path: request.model_path,
                load_format: request.load_format,
                abort_all_requests: request.abort_all_requests,
                weight_version: request.weight_version,
                is_async: request.is_async,
                torch_empty_cache: request.torch_empty_cache,
                keep_pause: request.keep_pause,
                recapture_cuda_graph: request.recapture_cuda_graph,
                token_step: request.token_step,
                flush_cache: request.flush_cache.unwrap_or(true),
                manifest,
            })
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::UpdateWeightsFromDiskResponse {
            success: outcome.success,
            message: outcome.message,
            num_paused_requests: outcome.num_paused_requests,
        }))
    }

    async fn update_weights_from_distributed(
        &self,
        request: Request<api::UpdateWeightsFromDistributedRequest>,
    ) -> Result<Response<api::WeightUpdateResponse>, Status> {
        let request = request.into_inner();
        update_response(
            self.core
                .update_weights_from_distributed(DistributedWeightUpdate {
                    names: request.names,
                    dtypes: request.dtypes,
                    shapes: request.shapes.into_iter().map(|list| list.items).collect(),
                    group_name: request
                        .group_name
                        .unwrap_or_else(|| "weight_update_group".into()),
                    flush_cache: request.flush_cache.unwrap_or(true),
                    abort_all_requests: request.abort_all_requests,
                    weight_version: request.weight_version,
                    load_format: request.load_format,
                    selector: request.selector.unwrap_or_else(|| "all".into()),
                    torch_empty_cache: request.torch_empty_cache,
                })
                .await,
        )
    }

    async fn update_weights_from_tensor(
        &self,
        request: Request<api::UpdateWeightsFromTensorRequest>,
    ) -> Result<Response<api::WeightUpdateResponse>, Status> {
        let request = request.into_inner();
        let serialized_named_tensors =
            decode_payloads(&request.serialized_named_tensors).map_err(Status::invalid_argument)?;
        update_response(
            self.core
                .update_weights_from_tensor(TensorWeightUpdate {
                    serialized_named_tensors,
                    load_format: request.load_format,
                    flush_cache: request.flush_cache.unwrap_or(true),
                    abort_all_requests: request.abort_all_requests,
                    weight_version: request.weight_version,
                    selector: request.selector.unwrap_or_else(|| "all".into()),
                    torch_empty_cache: request.torch_empty_cache,
                })
                .await,
        )
    }

    async fn update_weights_from_ipc(
        &self,
        request: Request<api::UpdateWeightsFromIpcRequest>,
    ) -> Result<Response<api::WeightUpdateResponse>, Status> {
        let request = request.into_inner();
        update_response(
            self.core
                .update_weights_from_ipc(IpcWeightUpdate {
                    zmq_handles: request.zmq_handles.into_iter().collect(),
                    flush_cache: request.flush_cache.unwrap_or(true),
                    weight_version: request.weight_version,
                    torch_empty_cache: request.torch_empty_cache,
                })
                .await,
        )
    }

    async fn begin_weight_update(
        &self,
        request: Request<api::BeginWeightUpdateRequest>,
    ) -> Result<Response<api::WeightUpdateResponse>, Status> {
        let selector = request
            .into_inner()
            .selector
            .unwrap_or_else(|| "all".into());
        update_response(self.core.begin_weight_update(selector).await)
    }

    async fn end_weight_update(
        &self,
        _request: Request<api::EndWeightUpdateRequest>,
    ) -> Result<Response<api::WeightUpdateResponse>, Status> {
        update_response(self.core.end_weight_update().await)
    }

    async fn plugin_weight_update(
        &self,
        request: Request<api::PluginWeightUpdateRequest>,
    ) -> Result<Response<api::PluginWeightUpdateResponse>, Status> {
        let request = request.into_inner();
        let payload = json_object("payload", request.payload.as_deref())
            .map_err(Status::invalid_argument)?
            .unwrap_or_default();
        let outcome = self
            .core
            .plugin_weight_update(PluginWeightUpdate {
                handler: request.handler,
                payload,
                flush_cache: request.flush_cache.unwrap_or(true),
                abort_all_requests: request.abort_all_requests,
                weight_version: request.weight_version,
                selector: request.selector.unwrap_or_else(|| "all".into()),
                torch_empty_cache: request.torch_empty_cache,
            })
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::PluginWeightUpdateResponse {
            success: outcome.success,
            message: outcome.message,
            received_tensor_counts: outcome.received_tensor_counts,
        }))
    }

    async fn load_lora_adapter(
        &self,
        _request: Request<api::LoadLoraAdapterRequest>,
    ) -> Result<Response<api::LoraUpdateResponse>, Status> {
        lora_disabled()
    }

    async fn load_lora_adapter_from_tensors(
        &self,
        _request: Request<api::LoadLoraAdapterFromTensorsRequest>,
    ) -> Result<Response<api::LoraUpdateResponse>, Status> {
        lora_disabled()
    }

    async fn unload_lora_adapter(
        &self,
        _request: Request<api::UnloadLoraAdapterRequest>,
    ) -> Result<Response<api::LoraUpdateResponse>, Status> {
        lora_disabled()
    }

    async fn open_session(
        &self,
        request: Request<api::OpenSessionRequest>,
    ) -> Result<Response<api::OpenSessionResponse>, Status> {
        let request = request.into_inner();
        let opened = self
            .core
            .open_session(SessionOpen {
                capacity_of_str_len: request.capacity_of_str_len,
                session_id: request.session_id,
                streaming: request.streaming,
                timeout: request.timeout,
            })
            .await
            .map_err(native_api::status)?;
        match opened {
            Some(session_id) => Ok(Response::new(api::OpenSessionResponse { session_id })),
            None => Err(Status::already_exists(
                "Failed to open the session. Check if a session with the same id is still open.",
            )),
        }
    }

    async fn close_session(
        &self,
        request: Request<api::CloseSessionRequest>,
    ) -> Result<Response<api::CloseSessionResponse>, Status> {
        self.core
            .close_session(request.into_inner().session_id)
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::CloseSessionResponse {}))
    }

    async fn get_hi_cache_l3_cache_layout(
        &self,
        _request: Request<api::GetHiCacheL3CacheLayoutRequest>,
    ) -> Result<Response<api::GetHiCacheL3CacheLayoutResponse>, Status> {
        let layout = self
            .core
            .hicache_l3_cache_layout()
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::GetHiCacheL3CacheLayoutResponse {
            layout: Some(layout.to_string()),
        }))
    }

    async fn configure_logging(
        &self,
        request: Request<api::ConfigureLoggingRequest>,
    ) -> Result<Response<api::ConfigureLoggingResponse>, Status> {
        let request = request.into_inner();
        self.core
            .configure_logging(LoggingConfig {
                log_requests: request.log_requests,
                log_requests_level: request.log_requests_level,
                log_requests_format: request.log_requests_format,
                log_level: request.log_level,
                dump_requests_folder: request.dump_requests_folder,
                dump_requests_threshold: request.dump_requests_threshold,
                crash_dump_folder: request.crash_dump_folder,
                dump_requests_exclude_meta_keys: request
                    .dump_requests_exclude_meta_keys
                    .map(|list| list.items),
            })
            .await
            .map_err(native_api::status)?;
        Ok(Response::new(api::ConfigureLoggingResponse {}))
    }

    /// Served whatever `DUMPER_SERVER_PORT` says: the HTTP mount gate is
    /// Python's route-level choice, not the dumper's.
    async fn dumper_control(
        &self,
        request: Request<api::DumperControlRequest>,
    ) -> Result<Response<api::DumperControlResponse>, Status> {
        let request = request.into_inner();
        let body = json_object("body", request.body.as_deref())
            .map_err(Status::invalid_argument)?
            .unwrap_or_default();
        match self
            .core
            .dumper_control(request.method, body)
            .await
            .map_err(native_api::status)?
        {
            DumperOutcome::Responses(responses) => Ok(Response::new(api::DumperControlResponse {
                responses: Some(serde_json::Value::Array(responses).to_string()),
            })),
            DumperOutcome::Failed(errors) => Err(Status::failed_precondition(errors.join(" | "))),
        }
    }

    async fn scale_elastic_ep(
        &self,
        request: Request<api::ScaleElasticEpRequest>,
    ) -> Result<Response<api::ScaleElasticEpResponse>, Status> {
        if request.into_inner().new_ep_size <= 0 {
            return Err(Status::invalid_argument(
                "new_ep_size must be a positive integer",
            ));
        }
        if self.core.elastic_ep_state().is_none() {
            return Err(Status::not_found(
                "elastic EP is not enabled (set --elastic-ep-backend)",
            ));
        }
        Err(Status::failed_precondition(
            "elastic EP scaling is not served by the Rust frontend",
        ))
    }

    async fn get_elastic_ep_state(
        &self,
        _request: Request<api::ElasticEpStateRequest>,
    ) -> Result<Response<api::ElasticEpStateResponse>, Status> {
        match self.core.elastic_ep_state() {
            None => Err(Status::not_found(
                "elastic EP is not enabled (set --elastic-ep-backend)",
            )),
            Some(ElasticEpState { effective_ep_size }) => {
                Ok(Response::new(api::ElasticEpStateResponse {
                    is_scaling_elastic_ep: false,
                    effective_ep_size,
                    pending_ep_size: None,
                    scale_phase: "idle".into(),
                    last_error: None,
                }))
            }
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
        // Unlike the admin-only HTTP route, this RPC takes no key, so the
        // backend config (it can carry S3 keys) is not returned.
        let mut status = self.hicache_manager.status().await;
        if status.hicache_storage_backend_extra_config.is_some() {
            status.hicache_storage_backend_extra_config = Some("<redacted>".into());
        }
        Ok(Response::new(status))
    }

    async fn attach_hi_cache_storage(
        &self,
        request: Request<api::AttachHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStorageResponse>, Status> {
        let output = self
            .hicache_manager
            .attach(request.into_inner())
            .await
            .map_err(hicache_status)?;
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
            .map_err(hicache_status)?;
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
            success: self.core.abort(
                (!request.abort_all).then_some(request.rid.as_str()),
                AbortEnding::CancelledError,
            ),
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
        request: Request<api::OpenAiScoreRequest>,
    ) -> Result<Response<api::OpenAiScoreResponse>, Status> {
        let request = request.into_inner();
        let scored = crate::api_server::http::openai::score::score_request(
            &self.core,
            &self.server_args,
            ReceivedAs::without_headers,
            &request,
        )
        .await
        .map_err(|(status, message)| {
            let code = match status {
                StatusCode::BAD_REQUEST => tonic::Code::InvalidArgument,
                StatusCode::SERVICE_UNAVAILABLE => tonic::Code::Unavailable,
                status if status.is_server_error() => tonic::Code::Internal,
                _ => tonic::Code::Unknown,
            };
            Status::new(code, message)
        })?;
        Ok(Response::new(scored.into_response(
            request.model.clone().unwrap_or_else(|| "default".into()),
        )))
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

    /// The unary Responses API; a streamed response has no RPC yet.
    async fn open_ai_responses(
        &self,
        request: Request<api::OpenAiResponsesRequest>,
    ) -> Result<Response<api::OpenAiResponsesResponse>, Status> {
        let request = request.into_inner();
        let adapter = crate::api_server::responses::Responses {
            core: &self.core,
            chat_formatter: self.chat_formatter.as_ref(),
            server_args: &self.server_args,
        };
        let value = adapter
            .create(&request)
            .await
            .map_err(|failure| failure.grpc_status())?;
        serde_json::from_value::<api::OpenAiResponsesResponse>(value)
            .map(Response::new)
            .map_err(|error| Status::internal(format!("responses: {error}")))
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

/// `--grpc-port` refuses `--admin-api-key` at startup, so a gRPC server never
/// has the admin key attach and detach require; HTTP's Python message, which
/// says to restart with one, would point gRPC callers at a refused flag.
const HICACHE_GRPC_UNAVAILABLE: &str =
    "HiCache attach/detach is not available over gRPC; use the HTTP endpoint with --admin-api-key.";

fn hicache_status(error: HiCacheError) -> Status {
    match error {
        HiCacheError::AdminKeyMissing => Status::failed_precondition(HICACHE_GRPC_UNAVAILABLE),
        HiCacheError::Core(error) => native_api::status(error),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;
    use sglang_api_types::api::v1 as api;
    use sglang_api_types::api::v1::generate_stream_item::Item;
    use sglang_api_types::api::v1::sglang_service_server::SglangService;
    use tonic::{Code, Request};

    use super::GrpcService;
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::{ModelConfig, PreferredSamplingParams, ServerArgs};

    use crate::message::request::{GenerateRequest, Request as RuntimeRequest, RequestKind};
    use crate::message::response::{ResponseItem, ResponseSink};
    use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission, TmEvent};

    struct Harness {
        service: GrpcService,
        intake_rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
    }

    struct GenerationIntake {
        request: Box<GenerateRequest>,
        sink: ResponseSink,
        admission: RequestAdmission,
    }

    impl Harness {
        fn new(response_capacity: usize, incremental: bool, response_timeout: Duration) -> Self {
            Self::build(
                response_capacity,
                incremental,
                response_timeout,
                true,
                None,
                test_server_args(),
            )
        }

        fn build(
            response_capacity: usize,
            incremental: bool,
            response_timeout: Duration,
            startup_ready: bool,
            preferred: Option<PreferredSamplingParams>,
            server_args: ServerArgs,
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
                    metadata: CoreMetadata::from(&server_args),
                    metrics: None,
                    request_log: None,
                    generate_validator: None,
                    abort_hook: None,
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
                rid: _,
                sink,
                kind,
                state: _,
            } = request;
            let RequestKind::Generate(request) = kind else {
                panic!("expected generation request");
            };
            GenerationIntake {
                request,
                sink,
                admission,
            }
        }
    }

    fn test_server_args() -> ServerArgs {
        ServerArgs {
            served_model_name: "test-model".into(),
            model_config: ModelConfig {
                context_len: 4096,
                ..Default::default()
            },
            ..Default::default()
        }
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

    fn stream_error(item: api::GenerateStreamItem) -> api::GenerateStreamError {
        match item.item {
            Some(Item::Error(error)) => error,
            other => panic!("expected an error item, got {other:?}"),
        }
    }

    /// A request the tokenizer manager refuses after admission is its
    /// in-stream error item with the client's-error type, not an RPC status;
    /// the call is disarmed.
    #[tokio::test]
    async fn validation_failure_rides_the_stream_as_a_client_error_item() {
        use crate::utils::error::Error;
        for (failure, code, kind, retryable) in [(
            Error::Validation("bad sampling".into()),
            400,
            "invalid_request_error",
            false,
        )] {
            let harness = Harness::new(1, false, Duration::from_secs(1));
            let mut stream = harness
                .service
                .generate(Request::new(ids_request(&[1], None)))
                .await
                .unwrap()
                .into_inner();
            let intake = harness.next_generation().await;
            assert!(intake.admission.try_accept());
            intake.sink.try_send(ResponseItem::Error(failure)).unwrap();

            let error = stream_error(stream.next().await.unwrap().unwrap());
            assert_eq!(error.index, None);
            let body = error.error.unwrap();
            assert_eq!(body.code, code);
            assert!(body.message.contains("bad "), "{}", body.message);
            assert_eq!(body.r#type.as_deref(), Some(kind));
            assert_eq!(body.retryable, Some(retryable));
            assert!(stream.next().await.is_none());
            drop(stream);
            assert!(harness.abort_rx.try_recv().is_err());
        }
    }

    /// `--grpc-response-timeout-secs` sets the per-event wait; 0 keeps the
    /// legacy gRPC bridge's 300 s default instead of expiring every call.
    #[tokio::test]
    async fn response_timeout_follows_the_launch_arg() {
        let harness = Harness::new(1, false, Duration::from_secs(1));
        let timeout_for = |secs| {
            let server_args = ServerArgs {
                grpc_response_timeout_secs: secs,
                ..Default::default()
            };
            GrpcService::new(
                harness.service.core.clone(),
                Arc::new(server_args),
                harness.service.hicache_manager.clone(),
            )
            .config
            .response_timeout
        };
        assert_eq!(timeout_for(900), Duration::from_secs(900));
        assert_eq!(timeout_for(0), Duration::from_secs(300));
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
                request_log: None,
                generate_validator: None,
                abort_hook: None,
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
    /// status that mirrors the HTTP 400, and are logged once with the reason.
    #[tokio::test]
    async fn malformed_request_is_rejected_before_submission() {
        let harness = Harness::new(1, false, Duration::from_secs(1));
        let mut both = text_request("prompt", "both");
        both.input_ids = Some(one_token_ids(&[1]));
        let logs = crate::utils::logging::capture_logs();
        let error = harness
            .service
            .generate(Request::new(both))
            .await
            .err()
            .expect("text and input_ids together must be rejected");
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(error.message().contains("text"));
        let rejections = logs.rejections();
        assert!(
            matches!(rejections.as_slice(), [line] if line.contains("WARN") && line.contains(error.message())),
            "{rejections:?}"
        );

        let error = harness
            .service
            .generate(Request::new(ids_request(&[], None)))
            .await
            .err()
            .expect("empty input_ids must be rejected");
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(harness.intake_rx.try_recv().is_err());
    }

    /// The launch's generate validator refuses a batch whole, as on HTTP: no
    /// item reaches the scheduler, the valid one included. The refusal is
    /// logged once.
    #[tokio::test]
    async fn generate_validator_refuses_the_whole_batch_before_submission() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            flume::unbounded().0,
            CoreConfig {
                response_capacity: 2,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal: false,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::default(),
                metrics: None,
                request_log: None,
                generate_validator: Some(Arc::new(crate::api_server::core::TopLogprobsCap)),
                abort_hook: None,
            },
        );
        let service = GrpcService::for_test(core, None, false, Duration::from_secs(1));
        let request = api::GenerateRequest {
            text: Some(many_strings(&["a", "b"])),
            top_logprobs_num: Some(api::Int64OrList {
                value: Some(api::int64_or_list::Value::Many(api::Int64List {
                    items: vec![1, 2],
                })),
            }),
            ..Default::default()
        };
        let logs = crate::utils::logging::capture_logs();
        let error = service
            .generate(Request::new(request))
            .await
            .err()
            .expect("the validator must refuse the batch");
        assert_eq!(error.code(), Code::InvalidArgument);
        assert_eq!(error.message(), "top_logprobs_num 2");
        let rejections = logs.rejections();
        assert!(
            matches!(rejections.as_slice(), [line] if line.contains("WARN") && line.contains("top_logprobs_num 2")),
            "{rejections:?}"
        );
        assert!(intake_rx.is_empty());
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
        let harness = Harness::build(
            1,
            false,
            Duration::from_secs(1),
            true,
            Some(preferred),
            test_server_args(),
        );
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
        let not_ready = Harness::build(
            1,
            false,
            Duration::from_secs(1),
            false,
            None,
            test_server_args(),
        );
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
}
