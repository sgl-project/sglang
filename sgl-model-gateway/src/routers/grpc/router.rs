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
        ResponsesContext,
    },
    context::SharedComponents,
    harmony::{serve_harmony_responses, serve_harmony_responses_stream, HarmonyDetector},
    pipeline::RequestPipeline,
    regular::responses,
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{is_retryable_status, RetryExecutor, WorkerRegistry, UNKNOWN_MODEL_ID},
    observability::metrics::{metrics_labels, Metrics},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
};

/// gRPC router implementation for SGLang
#[derive(Clone)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: RequestPipeline,
    harmony_pipeline: RequestPipeline,
    embedding_pipeline: RequestPipeline,
    classify_pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
    responses_context: ResponsesContext,
    harmony_responses_context: ResponsesContext,
    retry_config: RetryConfig,
}

impl GrpcRouter {
    /// Create a new gRPC router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Get tokenizer registry (no longer requires pre-loaded tokenizer)
        let tokenizer_registry = ctx.tokenizer_registry.clone();

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
            tokenizer_registry: tokenizer_registry.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
        });

        // Create regular pipeline
        let pipeline = RequestPipeline::new_regular(
            worker_registry.clone(),
            _policy_registry.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create Harmony pipelines
        let harmony_pipeline = RequestPipeline::new_harmony(
            worker_registry.clone(),
            _policy_registry.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create Embedding pipeline
        let embedding_pipeline =
            RequestPipeline::new_embeddings(worker_registry.clone(), _policy_registry.clone());

        // Create Classify pipeline
        let classify_pipeline =
            RequestPipeline::new_classify(worker_registry.clone(), _policy_registry.clone());

        // Extract shared dependencies for responses contexts
        let mcp_manager = ctx
            .mcp_manager
            .get()
            .ok_or_else(|| "gRPC router requires MCP manager".to_string())?
            .clone();

        // Helper closure to create responses context with a given pipeline
        let create_responses_context = |pipeline: &RequestPipeline| {
            ResponsesContext::new(
                Arc::new(pipeline.clone()),
                shared_components.clone(),
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
            embedding_pipeline,
            classify_pipeline,
            shared_components,
            responses_context,
            harmony_responses_context,
            retry_config: ctx.router_config.effective_retry_config(),
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
            "Processing chat completion request for model: {}, using_harmony={}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            is_harmony
        );

        let pipeline = if is_harmony {
            &self.harmony_pipeline
        } else {
            &self.pipeline
        };

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                async move {
                    pipeline
                        .execute_chat(request, headers, model_id, components)
                        .await
                }
            },
            // Should retry: check if status is retryable
            |res, _attempt| is_retryable_status(res.status()),
            // On backoff: record retry metrics
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_CHAT,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // On exhausted: record exhaustion
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_CHAT,
                );
            },
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
        debug!(
            "Processing generate request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();
        let pipeline = &self.pipeline;

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                async move {
                    pipeline
                        .execute_generate(request, headers, model_id, components)
                        .await
                }
            },
            // Should retry: check if status is retryable
            |res, _attempt| is_retryable_status(res.status()),
            // On backoff: record retry metrics
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // On exhausted: record exhaustion
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_GENERATE,
                );
            },
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
                "Processing Harmony responses request for model: {}, streaming: {}",
                model_id.unwrap_or(UNKNOWN_MODEL_ID),
                body.stream.unwrap_or(false)
            );
            let harmony_ctx = ResponsesContext::new(
                Arc::new(self.harmony_pipeline.clone()),
                self.shared_components.clone(),
                self.harmony_responses_context.response_storage.clone(),
                self.harmony_responses_context.conversation_storage.clone(),
                self.harmony_responses_context
                    .conversation_item_storage
                    .clone(),
                self.harmony_responses_context.mcp_manager.clone(),
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

    /// Main route_embeddings implementation
    async fn route_embeddings_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing embedding request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        self.embedding_pipeline
            .execute_embeddings(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    /// Main route_classify implementation
    async fn route_classify_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing classify request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        self.classify_pipeline
            .execute_classify(
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

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_embeddings_impl(headers, body, model_id).await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_classify_impl(headers, body, model_id).await
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }
}
