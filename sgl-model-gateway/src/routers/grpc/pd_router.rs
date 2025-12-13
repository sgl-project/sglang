use std::sync::Arc;

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use tracing::debug;

use super::{context::SharedComponents, pipeline::RequestPipeline};
use crate::{
    app_context::AppContext,
    core::{ConnectionMode, WorkerRegistry, WorkerType},
    protocols::{chat::ChatCompletionRequest, generate::GenerateRequest},
    routers::RouterTrait,
};

/// gRPC PD (Prefill-Decode) router implementation for SGLang
#[derive(Clone)]
pub struct GrpcPDRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
}

impl GrpcPDRouter {
    /// Create a new gRPC PD router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Get registries from context
        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires tokenizer".to_string())?
            .clone();
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
            tokenizer: tokenizer.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
        });

        // Create PD pipeline
        let pipeline = RequestPipeline::new_pd(
            worker_registry.clone(),
            policy_registry.clone(),
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        Ok(GrpcPDRouter {
            worker_registry,
            pipeline,
            shared_components,
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
