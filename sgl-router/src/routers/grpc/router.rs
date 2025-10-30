// gRPC Router Implementation

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::debug;

use super::{
    context::SharedComponents,
    harmony::{serve_harmony_responses, HarmonyDetector, HarmonyResponsesContext},
    pipeline::RequestPipeline,
    responses,
};
use crate::{
    app_context::AppContext,
    core::WorkerRegistry,
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
};

/// gRPC router implementation for SGLang
#[derive(Clone)]
#[allow(dead_code)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: RequestPipeline,
    harmony_pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
    // Responses context (bundles all /v1/responses dependencies: storage, MCP, background_tasks)
    responses_context: responses::ResponsesContext,
    // Harmony responses context (uses harmony pipeline)
    harmony_responses_context: responses::ResponsesContext,
}

impl GrpcRouter {
    /// Create a new gRPC router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC router requires tokenizer".to_string())?
            .clone();
        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires tool parser factory".to_string())?
            .clone();

        let worker_registry = ctx.worker_registry.clone();
        let _policy_registry = ctx.policy_registry.clone();

        // Create shared components for pipeline
        let shared_components = Arc::new(SharedComponents {
            tokenizer: tokenizer.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
        });

        // Create regular pipeline
        let pipeline = RequestPipeline::new_regular(
            worker_registry.clone(),
            _policy_registry.clone(),
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create Harmony pipelines
        let harmony_pipeline = RequestPipeline::new_harmony(
            worker_registry.clone(),
            _policy_registry.clone(),
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Extract shared dependencies for responses contexts
        let mcp_manager = ctx
            .mcp_manager
            .get()
            .ok_or_else(|| "gRPC router requires MCP manager".to_string())?
            .clone();

        // Helper closure to create responses context with a given pipeline
        let create_responses_context = |pipeline: &RequestPipeline| {
            responses::ResponsesContext::new(
                Arc::new(pipeline.clone()),
                shared_components.clone(),
                worker_registry.clone(),
                ctx.response_storage.clone(),
                ctx.conversation_storage.clone(),
                ctx.conversation_item_storage.clone(),
                mcp_manager.clone(),
            )
        };

        // Create responses contexts for both pipelines
        let responses_context = create_responses_context(&pipeline);
        let harmony_responses_context = create_responses_context(&harmony_pipeline);

        Ok(GrpcRouter {
            worker_registry,
            pipeline,
            harmony_pipeline,
            shared_components,
            responses_context,
            harmony_responses_context,
        })
    }

    /// Main route_chat implementation
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Choose Harmony pipeline if model indicates Harmony
        let is_harmony = HarmonyDetector::is_harmony_model(&body.model);

        debug!(
            "Processing chat completion request for model: {:?}, using_harmony={}",
            model_id, is_harmony
        );

        let pipeline = if is_harmony {
            &self.harmony_pipeline
        } else {
            &self.pipeline
        };

        // Use selected pipeline for ALL requests (streaming and non-streaming)
        pipeline
            .execute_chat(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    /// Main route_generate implementation
    async fn route_generate_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!("Processing generate request for model: {:?}", model_id);

        // Use pipeline for ALL requests (streaming and non-streaming)
        self.pipeline
            .execute_generate(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    /// Main route_responses implementation (pipeline-based for Harmony)
    async fn route_responses_impl(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing Harmony responses request for model: {:?}",
            model_id
        );

        // Create HarmonyResponsesContext from existing responses context
        let harmony_ctx = HarmonyResponsesContext::new(
            Arc::new(self.harmony_pipeline.clone()),
            self.shared_components.clone(),
            self.harmony_responses_context.mcp_manager.clone(),
            self.harmony_responses_context.response_storage.clone(),
        );

        // Use serve_harmony_responses for multi-turn MCP tool orchestration
        match serve_harmony_responses(&harmony_ctx, body.clone()).await {
            Ok(response) => axum::Json(response).into_response(),
            Err(error_response) => error_response,
        }
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.worker_registry.stats();
        f.debug_struct("GrpcRouter")
            .field("workers_count", &stats.total_workers)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // TODO: Implement actual generation test for gRPC
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not yet implemented for gRPC",
        )
            .into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_generate_impl(headers, body, model_id).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Choose implementation based on Harmony model detection
        let is_harmony = HarmonyDetector::is_harmony_model(&body.model);

        debug!(
            "Processing responses request for model: {:?}, using_harmony={}",
            model_id, is_harmony
        );

        if is_harmony {
            // Use pipeline-based implementation for Harmony models
            self.route_responses_impl(headers, body, model_id).await
        } else {
            // Use legacy responses module for non-Harmony models
            responses::route_responses(
                &self.responses_context,
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
            )
            .await
        }
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        responses::get_response_impl(&self.responses_context, response_id).await
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, response_id: &str) -> Response {
        responses::cancel_response_impl(&self.responses_context, response_id).await
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ClassifyRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }
}
