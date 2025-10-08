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

use crate::config::types::RetryConfig;
use crate::core::WorkerRegistry;
use crate::policies::PolicyRegistry;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, RerankRequest,
    ResponsesGetParams, ResponsesRequest,
};
use crate::reasoning_parser::ReasoningParserFactory;
use crate::routers::RouterTrait;
use crate::server::AppContext;
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::ToolParserFactory;

/// gRPC router implementation for SGLang
#[derive(Clone)]
#[allow(dead_code)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tokenizer: Arc<dyn Tokenizer>,
    reasoning_parser_factory: ReasoningParserFactory,
    tool_parser_factory: ToolParserFactory,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
    configured_reasoning_parser: Option<String>,
    configured_tool_parser: Option<String>,
    pipeline: super::pipeline::ChatCompletionPipeline,
    shared_components: Arc<super::context::SharedComponents>,
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
        let policy_registry = ctx.policy_registry.clone();

        // Create shared components for pipeline
        let shared_components = Arc::new(super::context::SharedComponents {
            tokenizer: tokenizer.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
        });

        // Create response processor
        let processor = super::processing::ResponseProcessor::new(
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create streaming processor
        let streaming_processor = Arc::new(super::streaming::StreamingProcessor::new(
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        ));

        // Create pipeline
        let pipeline = super::pipeline::ChatCompletionPipeline::new_regular(
            worker_registry.clone(),
            policy_registry.clone(),
            processor,
            streaming_processor,
        );

        Ok(GrpcRouter {
            worker_registry,
            policy_registry,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_factory,
            dp_aware: ctx.router_config.dp_aware,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            configured_reasoning_parser: ctx.configured_reasoning_parser.clone(),
            configured_tool_parser: ctx.configured_tool_parser.clone(),
            pipeline,
            shared_components,
        })
    }

    /// Main route_chat implementation
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {:?}",
            model_id
        );

        // Use pipeline for ALL requests (streaming and non-streaming)
        self.pipeline
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
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.worker_registry.stats();
        f.debug_struct("GrpcRouter")
            .field("workers_count", &stats.total_workers)
            .field("dp_aware", &self.dp_aware)
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
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
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
