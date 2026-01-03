use std::sync::Arc;

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use tracing::debug;

use super::{context::SharedComponents, pipeline::RequestPipeline};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{is_retryable_status, ConnectionMode, RetryExecutor, WorkerRegistry, WorkerType},
    observability::metrics::{metrics_labels, Metrics},
    protocols::{chat::ChatCompletionRequest, generate::GenerateRequest},
    routers::RouterTrait,
};

/// gRPC PD (Prefill-Decode) router implementation for SGLang
#[derive(Clone)]
pub struct GrpcPDRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
    retry_config: RetryConfig,
}

impl GrpcPDRouter {
    /// Create a new gRPC PD router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Get registries from context
        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        // Get tokenizer registry (no longer requires pre-loaded tokenizer)
        let tokenizer_registry = ctx.tokenizer_registry.clone();

        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires tool parser factory".to_string())?
            .clone();

        // Create shared components for pipeline
        let shared_components = Arc::new(SharedComponents {
            tokenizer_registry: tokenizer_registry.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
        });

        // Create PD pipeline
        let pipeline = RequestPipeline::new_pd(
            worker_registry.clone(),
            policy_registry.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        Ok(GrpcPDRouter {
            worker_registry,
            pipeline,
            shared_components,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    /// Main route_generate implementation with PD dual dispatch
    async fn route_generate_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing generate request for model: {:?} (PD mode)",
            model_id
        );

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();
        let pipeline = &self.pipeline;

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
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
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_GENERATE,
                );
            },
        )
        .await
    }

    /// Main route_chat implementation with PD dual dispatch
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {:?} (PD mode)",
            model_id
        );

        // Clone values needed for retry closure
        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.shared_components.clone();
        let pipeline = &self.pipeline;

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
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
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_CHAT,
                );
                Metrics::record_worker_retry(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_CHAT,
                );
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_CHAT,
                );
                Metrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_CHAT,
                );
            },
        )
        .await
    }
}

impl std::fmt::Debug for GrpcPDRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefill_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            Some(ConnectionMode::Grpc { port: None }),
            None,
            false,
        );
        let decode_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Decode),
            Some(ConnectionMode::Grpc { port: None }),
            None,
            false,
        );
        f.debug_struct("GrpcPDRouter")
            .field("prefill_workers_count", &prefill_workers.len())
            .field("decode_workers_count", &decode_workers.len())
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcPDRouter {
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

    fn router_type(&self) -> &'static str {
        "grpc_pd"
    }
}
