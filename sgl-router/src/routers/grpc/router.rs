// gRPC Router Implementation

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tokio::sync::RwLock;
use tracing::debug;

use super::{
    context::SharedComponents,
    pipeline::RequestPipeline,
    responses::{self, BackgroundTaskInfo},
};
use crate::{
    config::types::RetryConfig,
    core::WorkerRegistry,
    data_connector::{
        SharedConversationItemStorage, SharedConversationStorage, SharedResponseStorage,
    },
    policies::PolicyRegistry,
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::RouterTrait,
    server::AppContext,
    tokenizer::traits::Tokenizer,
    tool_parser::ParserFactory as ToolParserFactory,
};

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
    pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
    // Storage backends for /v1/responses support
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    // Optional MCP manager for tool execution (enabled via SGLANG_MCP_CONFIG env var)
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
    // Background task handles for cancellation support (includes gRPC client for Python abort)
    background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
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

        // Extract storage backends from context
        let response_storage = ctx.response_storage.clone();
        let conversation_storage = ctx.conversation_storage.clone();
        let conversation_item_storage = ctx.conversation_item_storage.clone();

        // Optional MCP manager activation via env var path (config-driven gate)
        let mcp_manager = match std::env::var("SGLANG_MCP_CONFIG").ok() {
            Some(path) if !path.trim().is_empty() => {
                match crate::mcp::McpConfig::from_file(&path).await {
                    Ok(cfg) => match crate::mcp::McpClientManager::new(cfg).await {
                        Ok(mgr) => Some(Arc::new(mgr)),
                        Err(err) => {
                            tracing::warn!("Failed to initialize MCP manager: {}", err);
                            None
                        }
                    },
                    Err(err) => {
                        tracing::warn!("Failed to load MCP config from '{}': {}", path, err);
                        None
                    }
                }
            }
            _ => None,
        };

        // Create shared components for pipeline
        let shared_components = Arc::new(SharedComponents {
            tokenizer: tokenizer.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
        });

        // Create pipeline
        let pipeline = RequestPipeline::new_regular(
            worker_registry.clone(),
            policy_registry.clone(),
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
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
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
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
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Use responses module for ALL requests (streaming and non-streaming)
        // Responses module handles:
        // - Request validation (previous_response_id XOR conversation)
        // - Loading response chain / conversation history from storage
        // - Conversion: ResponsesRequest → ChatCompletionRequest
        // - Execution through chat pipeline stages
        // - Conversion: ChatCompletionResponse → ResponsesResponse
        // - Response persistence
        // - MCP tool loop wrapper (future)
        responses::route_responses(
            &self.pipeline,
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(|s| s.to_string()),
            self.shared_components.clone(),
            self.response_storage.clone(),
            self.conversation_storage.clone(),
            self.conversation_item_storage.clone(),
            self.background_tasks.clone(),
        )
        .await
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        responses::get_response_impl(&self.response_storage, response_id).await
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, response_id: &str) -> Response {
        responses::cancel_response_impl(&self.response_storage, &self.background_tasks, response_id)
            .await
    }

    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ClassifyRequest,
        _model_id: Option<&str>,
    ) -> Response {
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
