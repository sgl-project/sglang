//! Pipeline stages for gRPC router request processing
//!
//! This module defines the core pipeline abstraction and individual processing stages
//! that transform a RequestContext through its lifecycle.

use std::{
    borrow::Cow,
    collections::HashMap,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use axum::response::{IntoResponse, Response};
use proto::DisaggregatedParams;
use rand::Rng;
use tokio::sync::RwLock;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::{context::*, processing, responses::BackgroundTaskInfo, streaming, utils};
use crate::{
    core::{ConnectionMode, Worker, WorkerRegistry, WorkerType},
    grpc_client::proto,
    policies::PolicyRegistry,
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        common::InputIds,
        generate::GenerateRequest,
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    tokenizer::traits::Tokenizer,
    tool_parser::ParserFactory as ToolParserFactory,
};

// ============================================================================
// Pipeline Trait
// ============================================================================

/// Trait for pipeline stages that process requests
#[async_trait]
pub trait PipelineStage: Send + Sync {
    /// Execute this stage, mutating the context
    ///
    /// Returns:
    /// - `Ok(None)` - Continue to next stage
    /// - `Ok(Some(response))` - Pipeline complete, return this response (e.g., streaming)
    /// - `Err(response)` - Error occurred, return this error response
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response>;

    /// Stage name for logging
    fn name(&self) -> &'static str;
}

// ============================================================================
// Stage 1: Preparation
// ============================================================================

/// Preparation stage: Filter tools, process messages, tokenize, build constraints
pub struct PreparationStage;

#[async_trait]
impl PipelineStage for PreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Clone Arc before match to avoid borrow checker issues
        // (matching borrows ctx, but prepare_* methods need mutable borrow)
        // Arc clone is cheap (8 bytes) - avoids full request clone (15KB-200KB)
        let is_chat = matches!(&ctx.input.request_type, RequestType::Chat(_));

        if is_chat {
            let request_arc = ctx.chat_request_arc();
            self.prepare_chat(ctx, &request_arc).await?;
        } else {
            let request_arc = ctx.generate_request_arc();
            self.prepare_generate(ctx, &request_arc).await?;
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "Preparation"
    }
}

impl PreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Step 1: Filter tools if needed
        let body_ref = utils::filter_tools_for_request(request);

        // Step 2: Process messages and apply chat template
        let processed_messages =
            match utils::process_chat_messages(&body_ref, &*ctx.components.tokenizer) {
                Ok(msgs) => msgs,
                Err(e) => {
                    return Err(utils::bad_request_error(e));
                }
            };

        // Step 3: Tokenize the processed text
        let encoding = match ctx.components.tokenizer.encode(&processed_messages.text) {
            Ok(encoding) => encoding,
            Err(e) => {
                return Err(utils::internal_error_message(format!(
                    "Tokenization failed: {}",
                    e
                )));
            }
        };

        let token_ids = encoding.token_ids().to_vec();

        // Step 4: Build tool constraints if needed
        let tool_call_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, &request.tool_choice, &request.model).map_err(
                |e| utils::bad_request_error(format!("Invalid tool configuration: {}", e)),
            )?
        } else {
            None
        };

        // Step 5: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &ctx.components.tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(processed_messages.text.clone()),
            token_ids,
            processed_messages: Some(processed_messages),
            tool_constraints: tool_call_constraint,
            filtered_request: if matches!(body_ref, Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
        });

        // Store stop decoder for reuse in response processing
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    async fn prepare_generate(
        &self,
        ctx: &mut RequestContext,
        request: &GenerateRequest,
    ) -> Result<(), Response> {
        // Resolve input (text, prompt, or input_ids)
        let (original_text, token_ids) = match self.resolve_generate_input(ctx, request) {
            Ok(res) => res,
            Err(msg) => {
                return Err(utils::bad_request_error(msg));
            }
        };

        // Create stop sequence decoder for generate requests
        let params = request.sampling_params.as_ref();
        let stop_decoder = utils::create_stop_decoder(
            &ctx.components.tokenizer,
            params.and_then(|p| p.stop.as_ref()),
            params.and_then(|p| p.stop_token_ids.as_ref()),
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            params.and_then(|p| p.no_stop_trim).unwrap_or(false),
        );

        ctx.state.preparation = Some(PreparationOutput {
            original_text,
            token_ids,
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
        });

        // Store stop decoder
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    fn resolve_generate_input(
        &self,
        ctx: &RequestContext,
        request: &GenerateRequest,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        if let Some(text) = &request.text {
            return self
                .tokenize_single_text(&ctx.components.tokenizer, text)
                .map(|(original, ids)| (Some(original), ids));
        }

        // Handle input_ids - validate and convert
        if let Some(input_ids) = &request.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| u32::try_from(id))
                    .collect::<Result<Vec<u32>, _>>()
                    .map(|converted| (None, converted))
                    .map_err(|_| "input_ids must be non-negative".to_string()),
                InputIds::Batch(_) => {
                    Err("Batch input_ids are not supported over gRPC generate yet".to_string())
                }
            };
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    fn tokenize_single_text(
        &self,
        tokenizer: &Arc<dyn Tokenizer>,
        text: &str,
    ) -> Result<(String, Vec<u32>), String> {
        let encoding = tokenizer
            .encode(text)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        Ok((text.to_string(), encoding.token_ids().to_vec()))
    }
}

// ============================================================================
// Stage 2: Worker Selection
// ============================================================================

/// Worker selection stage: Select appropriate worker(s) based on routing mode
pub struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
}

pub enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            mode,
        }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx
            .state
            .preparation
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Preparation stage not completed"))?;

        let text = prep.original_text.as_deref();

        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                match self.select_single_worker(ctx.input.model_id.as_deref(), text) {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        return Err(utils::service_unavailable_error(format!(
                            "No available workers for model: {:?}",
                            ctx.input.model_id
                        )));
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode => {
                match self.select_pd_pair(ctx.input.model_id.as_deref(), text) {
                    Some((prefill, decode)) => WorkerSelection::Dual { prefill, decode },
                    None => {
                        return Err(utils::service_unavailable_error(format!(
                            "No available PD worker pairs for model: {:?}",
                            ctx.input.model_id
                        )));
                    }
                }
            }
        };

        ctx.state.workers = Some(workers);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}

impl WorkerSelectionStage {
    fn select_single_worker(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc { port: None }),
            false, // get all workers, we'll filter by is_available() next
        );

        // Filter by availability (health + circuit breaker)
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Select worker using the policy
        let idx = policy.select_worker(&available, text)?;
        Some(available[idx].clone())
    }

    fn select_pd_pair(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
    ) -> Option<(Arc<dyn Worker>, Arc<dyn Worker>)> {
        let all_workers = self.worker_registry.get_workers_filtered(
            model_id,
            None,
            Some(ConnectionMode::Grpc { port: None }), // Match any gRPC worker
            false,
        );

        let (available_prefill, available_decode): (Vec<_>, Vec<_>) =
            all_workers
                .into_iter()
                .fold((Vec::new(), Vec::new()), |mut acc, w| {
                    if w.is_available() {
                        match w.metadata().worker_type {
                            WorkerType::Prefill { .. } => acc.0.push(w),
                            WorkerType::Decode => acc.1.push(w),
                            _ => {}
                        }
                    }
                    acc
                });

        if available_prefill.is_empty() {
            warn!("No available prefill workers");
            return None;
        }

        if available_decode.is_empty() {
            warn!("No available decode workers");
            return None;
        }

        // Select using policies
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        let prefill_idx = policy.select_worker(&available_prefill, text)?;
        let decode_idx = policy.select_worker(&available_decode, text)?;

        Some((
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
        ))
    }
}

// ============================================================================
// Stage 3: Client Acquisition
// ============================================================================

/// Client acquisition stage: Get gRPC clients from selected workers
pub struct ClientAcquisitionStage;

#[async_trait]
impl PipelineStage for ClientAcquisitionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let workers = ctx
            .state
            .workers
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Worker selection not completed"))?;

        let clients = match workers {
            WorkerSelection::Single { worker } => {
                let client = utils::get_grpc_client_from_worker(worker).await?;
                ClientSelection::Single { client }
            }
            WorkerSelection::Dual { prefill, decode } => {
                let prefill_client = utils::get_grpc_client_from_worker(prefill).await?;
                let decode_client = utils::get_grpc_client_from_worker(decode).await?;
                ClientSelection::Dual {
                    prefill: prefill_client,
                    decode: decode_client,
                }
            }
        };

        ctx.state.clients = Some(clients);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ClientAcquisition"
    }
}

// ============================================================================
// Stage 4: Request Building
// ============================================================================

/// Request building stage: Build proto GenerateRequest
pub struct RequestBuildingStage {
    inject_pd_metadata: bool,
}

impl RequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for RequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx
            .state
            .preparation
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Preparation not completed"))?;

        let clients = ctx
            .state
            .clients
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Client acquisition not completed"))?;

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        let mut proto_request = match &ctx.input.request_type {
            RequestType::Chat(request) => {
                let request_id = format!("chatcmpl-{}", Uuid::new_v4());
                let body_ref = prep.filtered_request.as_ref().unwrap_or(request);

                builder_client
                    .build_generate_request(
                        request_id,
                        body_ref,
                        prep.processed_messages.as_ref().unwrap().text.clone(),
                        prep.token_ids.clone(),
                        prep.processed_messages
                            .as_ref()
                            .unwrap()
                            .multimodal_inputs
                            .clone(),
                        prep.tool_constraints.clone(),
                    )
                    .map_err(|e| {
                        utils::bad_request_error(format!("Invalid request parameters: {}", e))
                    })?
            }
            RequestType::Generate(request) => {
                let request_id = request
                    .rid
                    .clone()
                    .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

                builder_client
                    .build_plain_generate_request(
                        request_id,
                        request,
                        prep.original_text.clone(),
                        prep.token_ids.clone(),
                    )
                    .map_err(utils::bad_request_error)?
            }
        };

        // Inject PD metadata if needed
        if self.inject_pd_metadata {
            if let WorkerSelection::Dual { prefill, .. } = ctx.state.workers.as_ref().unwrap() {
                self.inject_bootstrap_metadata(&mut proto_request, prefill);
            }
        }

        ctx.state.proto_request = Some(proto_request);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestBuilding"
    }
}

impl RequestBuildingStage {
    fn inject_bootstrap_metadata(
        &self,
        request: &mut proto::GenerateRequest,
        prefill_worker: &Arc<dyn Worker>,
    ) {
        let hostname = prefill_worker.bootstrap_host();
        let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

        // Generate room ID for bootstrap
        let room_id = rand::rng().random_range(0..i32::MAX);

        // Create DisaggregatedParams
        let disagg_params = DisaggregatedParams {
            bootstrap_host: hostname.to_string(),
            bootstrap_port: bootstrap_port as i32,
            bootstrap_room: room_id,
        };

        // Inject metadata directly into request
        request.disaggregated_params = Some(disagg_params);

        debug!(
            "Injected bootstrap metadata: host={}, port={}, room={}",
            hostname, bootstrap_port, room_id
        );
    }
}

// ============================================================================
// Stage 5: Dispatch Metadata
// ============================================================================

/// Dispatch metadata stage: Prepare metadata for dispatch
pub struct DispatchMetadataStage;

#[async_trait]
impl PipelineStage for DispatchMetadataStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let proto_request = ctx
            .state
            .proto_request
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Proto request not built"))?;

        let request_id = proto_request.request_id.clone();
        let model = match &ctx.input.request_type {
            RequestType::Chat(req) => req.model.clone(),
            RequestType::Generate(_req) => {
                // Generate requests don't have a model field
                // Use model_id from input or default
                ctx.input
                    .model_id
                    .clone()
                    .unwrap_or_else(|| "default".to_string())
            }
        };

        let weight_version = ctx
            .state
            .workers
            .as_ref()
            .map(|w| match w {
                WorkerSelection::Single { worker } => worker,
                WorkerSelection::Dual { decode, .. } => decode,
            })
            .and_then(|w| w.metadata().labels.get("weight_version").cloned())
            .unwrap_or_else(|| "default".to_string());

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ctx.state.dispatch = Some(DispatchMetadata {
            request_id,
            model,
            created,
            weight_version: Some(weight_version),
            is_streaming: ctx.is_streaming(),
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "DispatchMetadata"
    }
}

// ============================================================================
// Stage 6: Request Execution
// ============================================================================

/// Request execution stage: Execute gRPC requests (single or dual dispatch)
pub struct RequestExecutionStage {
    mode: ExecutionMode,
}

pub enum ExecutionMode {
    /// Regular mode: single worker execution
    Single,
    /// PD mode: dual dispatch to prefill + decode workers
    DualDispatch,
}

impl RequestExecutionStage {
    pub fn new(mode: ExecutionMode) -> Self {
        Self { mode }
    }
}

#[async_trait]
impl PipelineStage for RequestExecutionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let proto_request = ctx
            .state
            .proto_request
            .take()
            .ok_or_else(|| utils::internal_error_static("Proto request not built"))?;

        let clients = ctx
            .state
            .clients
            .as_mut()
            .ok_or_else(|| utils::internal_error_static("Client acquisition not completed"))?;

        let result = match self.mode {
            ExecutionMode::Single => self.execute_single(proto_request, clients).await?,
            ExecutionMode::DualDispatch => {
                self.execute_dual_dispatch(proto_request, clients).await?
            }
        };

        // Store result in context for ResponseProcessingStage
        ctx.state.response.execution_result = Some(result);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestExecution"
    }
}

impl RequestExecutionStage {
    async fn execute_single(
        &self,
        proto_request: proto::GenerateRequest,
        clients: &mut ClientSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients
            .single_mut()
            .ok_or_else(|| utils::internal_error_static("Expected single client but got dual"))?;

        let stream = client.generate(proto_request).await.map_err(|e| {
            utils::internal_error_message(format!("Failed to start generation: {}", e))
        })?;

        Ok(ExecutionResult::Single { stream })
    }

    async fn execute_dual_dispatch(
        &self,
        proto_request: proto::GenerateRequest,
        clients: &mut ClientSelection,
    ) -> Result<ExecutionResult, Response> {
        let (prefill_client, decode_client) = clients
            .dual_mut()
            .ok_or_else(|| utils::internal_error_static("Expected dual clients but got single"))?;

        let prefill_request = proto_request.clone();
        let decode_request = proto_request;

        let (prefill_result, decode_result) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Handle prefill result
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                return Err(utils::internal_error_message(format!(
                    "Prefill worker failed to start: {}",
                    e
                )));
            }
        };

        // Handle decode result
        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                return Err(utils::internal_error_message(format!(
                    "Decode worker failed to start: {}",
                    e
                )));
            }
        };

        Ok(ExecutionResult::Dual {
            prefill: prefill_stream,
            decode: Box::new(decode_stream),
        })
    }
}

// ============================================================================
// Stage 7: Response Processing
// ============================================================================

/// Response processing stage: Handles both streaming and non-streaming responses
///
/// - For streaming: Spawns background task and returns SSE response (early exit)
/// - For non-streaming: Collects all responses and builds final ChatCompletionResponse
pub struct ResponseProcessingStage {
    processor: processing::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl ResponseProcessingStage {
    pub fn new(
        processor: processing::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
        }
    }
}

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Delegate to request-type specific processing
        match &ctx.input.request_type {
            RequestType::Chat(_) => return self.process_chat_response(ctx).await,
            RequestType::Generate(_) => return self.process_generate_response(ctx).await,
        }
    }

    fn name(&self) -> &'static str {
        "ResponseProcessing"
    }
}

impl ResponseProcessingStage {
    async fn process_chat_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx
            .state
            .response
            .execution_result
            .take()
            .ok_or_else(|| utils::internal_error_static("No execution result"))?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Dispatch metadata not set"))?
            .clone();

        if is_streaming {
            // Streaming: Use StreamingProcessor and return SSE response (done)
            return Ok(Some(
                self.streaming_processor.clone().process_streaming_response(
                    execution_result,
                    ctx.chat_request_arc(), // Cheap Arc clone (8 bytes)
                    dispatch,
                ),
            ));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = match &ctx.input.request_type {
            RequestType::Chat(req) => req.logprobs,
            _ => false,
        };

        let chat_request = ctx.chat_request_arc();

        let stop_decoder = ctx
            .state
            .response
            .stop_decoder
            .as_mut()
            .ok_or_else(|| utils::internal_error_static("Stop decoder not initialized"))?;

        let response = self
            .processor
            .process_non_streaming_chat_response(
                execution_result,
                chat_request,
                dispatch,
                stop_decoder,
                request_logprobs,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Chat(response));

        Ok(None)
    }

    async fn process_generate_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let start_time = Instant::now();
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx
            .state
            .response
            .execution_result
            .take()
            .ok_or_else(|| utils::internal_error_static("No execution result"))?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Dispatch metadata not set"))?
            .clone();

        if is_streaming {
            // Streaming: Use StreamingProcessor and return SSE response (done)
            return Ok(Some(
                self.streaming_processor.clone().process_streaming_generate(
                    execution_result,
                    ctx.generate_request_arc(), // Cheap Arc clone (8 bytes)
                    dispatch,
                ),
            ));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = ctx.generate_request().return_logprob.unwrap_or(false);
        let generate_request = ctx.generate_request_arc();

        let stop_decoder = ctx
            .state
            .response
            .stop_decoder
            .as_mut()
            .ok_or_else(|| utils::internal_error_static("Stop decoder not initialized"))?;

        let result_array = self
            .processor
            .process_non_streaming_generate_response(
                execution_result,
                generate_request,
                dispatch,
                stop_decoder,
                request_logprobs,
                start_time,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Generate(result_array));

        Ok(None)
    }
}

// ============================================================================
// Pipeline Orchestrator
// ============================================================================

/// Generic request pipeline for all request types
///
/// Orchestrates all stages from request preparation to response delivery.
/// Configured differently for regular vs PD mode.
#[derive(Clone)]
pub struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
}

impl RequestPipeline {
    /// Create a regular (single-worker) pipeline
    pub fn new_regular(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tokenizer: Arc<dyn Tokenizer>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        // Create response processor
        let processor = processing::ResponseProcessor::new(
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            configured_tool_parser.clone(),
            configured_reasoning_parser.clone(),
        );

        // Create streaming processor
        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            tokenizer,
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        ));

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(PreparationStage),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::Regular,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(RequestBuildingStage::new(false)), // No PD metadata
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::Single)),
            Box::new(ResponseProcessingStage::new(processor, streaming_processor)),
        ];

        Self {
            stages: Arc::new(stages),
        }
    }

    /// Create a PD (prefill-decode) pipeline
    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tokenizer: Arc<dyn Tokenizer>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        // Create response processor
        let processor = processing::ResponseProcessor::new(
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            configured_tool_parser.clone(),
            configured_reasoning_parser.clone(),
        );

        // Create streaming processor
        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            tokenizer,
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        ));

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(PreparationStage),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::PrefillDecode,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(RequestBuildingStage::new(true)), // Inject PD metadata
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::DualDispatch)),
            Box::new(ResponseProcessingStage::new(processor, streaming_processor)),
        ];

        Self {
            stages: Arc::new(stages),
        }
    }

    /// Execute the complete pipeline for a chat request
    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);

        // Execute each stage in sequence
        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed successfully with a response (e.g., streaming)
                    return response;
                }
                Ok(None) => {
                    // Continue to next stage
                    continue;
                }
                Err(response) => {
                    // Error occurred
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        // Extract final response
        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => axum::Json(response).into_response(),
            Some(FinalResponse::Generate(_)) => {
                utils::internal_error_static("Internal error: wrong response type")
            }
            None => utils::internal_error_static("No response produced"),
        }
    }

    /// Execute the complete pipeline for a generate request
    pub async fn execute_generate(
        &self,
        request: Arc<GenerateRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let mut ctx = RequestContext::for_generate(request, headers, model_id, components);

        // Execute each stage in sequence
        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed successfully with a response (e.g., streaming)
                    return response;
                }
                Ok(None) => {
                    // Continue to next stage
                    continue;
                }
                Err(response) => {
                    // Error occurred
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        // Extract final response
        match ctx.state.response.final_response {
            Some(FinalResponse::Generate(response)) => axum::Json(response).into_response(),
            Some(FinalResponse::Chat(_)) => {
                utils::internal_error_static("Internal error: wrong response type")
            }
            None => utils::internal_error_static("No response produced"),
        }
    }

    /// Execute chat pipeline for responses endpoint (Result-based for easier composition)
    ///
    /// This is used by the responses module and returns Result instead of Response.
    /// It also supports background mode cancellation via background_tasks.
    pub async fn execute_chat_for_responses(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
        response_id: Option<String>,
        background_tasks: Option<Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>>,
    ) -> Result<ChatCompletionResponse, String> {
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);

        // Execute each stage in sequence
        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(_response)) => {
                    // Streaming not supported for responses sync mode
                    return Err("Streaming is not supported in this context".to_string());
                }
                Ok(None) => {
                    let stage_name = stage.name();

                    // After ClientAcquisitionStage, store client for background task cancellation
                    if stage_name == "ClientAcquisition" {
                        if let (Some(ref clients), Some(ref resp_id), Some(ref tasks)) =
                            (&ctx.state.clients, &response_id, &background_tasks)
                        {
                            let client_to_store = match clients {
                                ClientSelection::Single { client } => client.clone(),
                                ClientSelection::Dual { decode, .. } => decode.clone(),
                            };

                            if let Some(task_info) = tasks.write().await.get_mut(resp_id.as_str()) {
                                *task_info.client.write().await = Some(client_to_store);
                                debug!("Stored client for response_id: {}", resp_id);
                            }
                        }
                    }

                    // After DispatchMetadataStage, store grpc_request_id for background task cancellation
                    if stage_name == "DispatchMetadata" {
                        if let (Some(ref dispatch), Some(ref resp_id), Some(ref tasks)) =
                            (&ctx.state.dispatch, &response_id, &background_tasks)
                        {
                            let grpc_request_id = dispatch.request_id.clone();

                            if let Some(task_info) = tasks.write().await.get_mut(resp_id.as_str()) {
                                task_info.grpc_request_id = grpc_request_id.clone();
                                debug!("Stored grpc_request_id for response_id: {}", resp_id);
                            }
                        }
                    }

                    // Continue to next stage
                    continue;
                }
                Err(response) => {
                    // Error occurred
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(format!("Pipeline stage {} failed", stage.name()));
                }
            }
        }

        // Extract final response
        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => Ok(response),
            Some(FinalResponse::Generate(_)) => {
                Err("Internal error: wrong response type".to_string())
            }
            None => Err("No response produced".to_string()),
        }
    }
}
