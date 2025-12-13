use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use tracing::debug;

use super::{
    common::responses::{
        handlers::{cancel_response_impl, get_response_impl},
        utils::validate_worker_availability,
    },
    context::SharedComponents,
    harmony::{
        serve_harmony_responses, serve_harmony_responses_stream, HarmonyDetector,
        HarmonyResponsesContext,
    },
    pipeline::RequestPipeline,
    regular::responses,
};
use crate::{
    app_context::AppContext,
    core::WorkerRegistry,
    protocols::{
        chat::ChatCompletionRequest,
        generate::GenerateRequest,
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
    responses_context: responses::ResponsesContext,
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
        // Choose Harmony pipeline if workers indicate Harmony (checks architectures, hf_model_type)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        debug!(
            "Processing chat completion request for model: {:?}, using_harmony={}",
            model_id, is_harmony
        );

        let pipeline = if is_harmony {
            &self.harmony_pipeline
        } else {
            &self.pipeline
        };

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

        self.pipeline
            .execute_generate(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    /// Main route_responses implementation
    ///
    /// Routes to either Harmony or regular responses implementation based on model detection
    async fn route_responses_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        // 0. Fast worker validation (fail-fast before expensive operations)
        let requested_model: Option<&str> = model_id.or(Some(body.model.as_str()));

        if let Some(error_response) = requested_model
            .and_then(|model| validate_worker_availability(&self.worker_registry, model))
        {
            return error_response;
        }

        // Choose implementation based on Harmony model detection (checks worker metadata)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        if is_harmony {
            debug!(
                "Processing Harmony responses request for model: {:?}, streaming: {:?}",
                model_id, body.stream
            );
            let harmony_ctx = HarmonyResponsesContext::new(
                Arc::new(self.harmony_pipeline.clone()),
                self.shared_components.clone(),
                self.harmony_responses_context.mcp_manager.clone(),
                self.harmony_responses_context.response_storage.clone(),
                self.harmony_responses_context.conversation_storage.clone(),
                self.harmony_responses_context
                    .conversation_item_storage
                    .clone(),
            );

            if body.stream.unwrap_or(false) {
                serve_harmony_responses_stream(&harmony_ctx, body.clone()).await
            } else {
                match serve_harmony_responses(&harmony_ctx, body.clone()).await {
                    Ok(response) => axum::Json(response).into_response(),
                    Err(error_response) => error_response,
                }
            }
        } else {
            responses::route_responses(
                &self.responses_context,
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
            )
            .await
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

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_responses_impl(headers, body, model_id).await
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        get_response_impl(&self.responses_context, response_id).await
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, response_id: &str) -> Response {
        cancel_response_impl(&self.responses_context, response_id).await
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }
}
