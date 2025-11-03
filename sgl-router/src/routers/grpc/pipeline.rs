//! Pipeline orchestrator for gRPC router request processing
//!
//! This module defines the RequestPipeline orchestrator that coordinates
//! the execution of pipeline stages from request preparation to response delivery.

use std::{collections::HashMap, sync::Arc};

use axum::response::{IntoResponse, Response};
use tokio::sync::RwLock;
use tracing::{debug, error};

// Import all stage types from the stages module
use super::stages::*;
use super::{context::*, harmony, processing, responses::BackgroundTaskInfo, streaming, utils};
use crate::{
    core::WorkerRegistry,
    policies::PolicyRegistry,
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        generate::GenerateRequest,
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    tokenizer::traits::Tokenizer,
    tool_parser::ParserFactory as ToolParserFactory,
};

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

    /// Create a Harmony (single-worker) pipeline for Harmony-capable models
    pub fn new_harmony(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        _tokenizer: Arc<dyn Tokenizer>,
        _tool_parser_factory: ToolParserFactory,
        _reasoning_parser_factory: ReasoningParserFactory,
        _configured_tool_parser: Option<String>,
        _configured_reasoning_parser: Option<String>,
    ) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(harmony::stages::HarmonyPreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::Regular,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(harmony::stages::HarmonyRequestBuildingStage::new(false)),
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::Single)),
            Box::new(harmony::stages::HarmonyResponseProcessingStage::new()),
        ];

        Self {
            stages: Arc::new(stages),
        }
    }

    /// Create a Harmony PD (prefill-decode) pipeline
    pub fn new_harmony_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        _tokenizer: Arc<dyn Tokenizer>,
        _tool_parser_factory: ToolParserFactory,
        _reasoning_parser_factory: ReasoningParserFactory,
        _configured_tool_parser: Option<String>,
        _configured_reasoning_parser: Option<String>,
    ) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(harmony::stages::HarmonyPreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::PrefillDecode,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(harmony::stages::HarmonyRequestBuildingStage::new(true)),
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::DualDispatch)),
            Box::new(harmony::stages::HarmonyResponseProcessingStage::new()),
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

    /// Execute chat pipeline for responses endpoint
    ///
    /// TODO: The support for background tasks is not scalable. Consider replacing this with
    /// a better design in the future.
    /// Used by ALL non-streaming /v1/responses requests (both sync and background modes).
    /// Uses the same 7 pipeline stages as execute_chat(), with three differences:
    /// 1. Returns Result<ChatCompletionResponse, Response> for tool_loop composition
    /// 2. Disallows streaming (responses endpoint uses different SSE format)
    /// 3. Injects hooks for background task cancellation (only active when response_id provided)
    pub async fn execute_chat_for_responses(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
        response_id: Option<String>,
        background_tasks: Option<Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>>,
    ) -> Result<ChatCompletionResponse, Response> {
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);

        // Execute each stage in sequence
        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(_response)) => {
                    // Streaming not supported for responses sync mode
                    return Err(utils::bad_request_error(
                        "Streaming is not supported in this context".to_string(),
                    ));
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
                    // Error occurred - return the response as-is to preserve HTTP status codes
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract final response
        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => Ok(response),
            Some(FinalResponse::Generate(_)) => Err(utils::internal_error_static(
                "Internal error: wrong response type",
            )),
            None => Err(utils::internal_error_static("No response produced")),
        }
    }

    /// Execute Responses API pipeline
    ///
    /// TODO: Implement Responses API native execution
    /// This is a stub to allow compilation. The actual implementation should:
    /// 1. Support multi-turn MCP loop orchestration
    /// 2. Handle tool call execution and result injection
    /// 3. Emit proper SSE events for streaming mode
    /// 4. Store responses in data connector
    ///
    /// For now, this returns an error indicating the feature is not implemented.
    pub async fn execute_responses(
        &self,
        _request: Arc<crate::protocols::responses::ResponsesRequest>,
        _headers: Option<http::HeaderMap>,
        _model_id: Option<String>,
        _components: Arc<SharedComponents>,
    ) -> Response {
        utils::internal_error_static("Responses API execution not yet implemented")
    }

    /// Execute Harmony Responses API request through all pipeline stages
    ///
    /// This method runs a single iteration of the Responses API request,
    /// returning either ToolCallsFound (continue serving) or Completed (final response).
    ///
    /// Called by harmony::responses::serve_harmony_responses() for each iteration.
    ///
    /// # Arguments
    ///
    /// * `request` - Responses API request
    /// * `ctx` - Harmony Responses context with MCP manager and components
    ///
    /// # Returns
    ///
    /// ResponsesIterationResult indicating whether to continue iteration or return
    pub async fn execute_harmony_responses(
        &self,
        request: &crate::protocols::responses::ResponsesRequest,
        harmony_ctx: &harmony::responses::HarmonyResponsesContext,
    ) -> Result<harmony::ResponsesIterationResult, Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None, // No headers needed for internal pipeline execution
            None, // Model ID already set in request
            harmony_ctx.components.clone(),
        );

        // Execute each pipeline stage in sequence
        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage returned early response (e.g., streaming) - not expected for Responses iteration
                    error!(
                        "Stage {} ({}) returned unexpected response during Responses iteration",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => {
                    // Stage completed successfully, continue to next stage
                    continue;
                }
                Err(response) => {
                    // Stage failed
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract ResponsesIterationResult from context
        // This should have been set by HarmonyResponseProcessingStage
        ctx.state
            .response
            .responses_iteration_result
            .take()
            .ok_or_else(|| {
                utils::internal_error_static("No ResponsesIterationResult produced by pipeline")
            })
    }

    /// Execute Harmony Responses pipeline iteration with streaming support
    ///
    /// This version executes the pipeline up to the dispatch stage and returns
    /// the raw ExecutionResult (with stream) for token-level streaming processing.
    pub async fn execute_harmony_responses_streaming(
        &self,
        request: &crate::protocols::responses::ResponsesRequest,
        harmony_ctx: &harmony::responses::HarmonyResponsesContext,
    ) -> Result<ExecutionResult, Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,
            None,
            harmony_ctx.components.clone(),
        );

        // Execute pipeline stages up to dispatch (which creates the stream)
        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    error!(
                        "Stage {} ({}) returned unexpected response during streaming Responses",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => continue,
                Err(response) => {
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract execution_result (the raw stream from workers)
        ctx.state
            .response
            .execution_result
            .take()
            .ok_or_else(|| utils::internal_error_static("No ExecutionResult produced by pipeline"))
    }
}
