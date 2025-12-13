//! Request execution stage: Execute gRPC requests (single or dual dispatch)

use async_trait::async_trait;
use axum::response::Response;
use tracing::{error, info_span, Instrument};

use super::PipelineStage;
use crate::routers::{
    error,
    grpc::{
        context::{ClientSelection, ExecutionResult, RequestContext},
        proto_wrapper::{ProtoGenerateRequest, ProtoStream},
    },
};

type StreamResult = Result<ProtoStream, Box<dyn std::error::Error + Send + Sync>>;

/// Request execution stage: Execute gRPC requests (single or dual dispatch)
pub struct RequestExecutionStage {
    mode: ExecutionMode,
}

#[derive(Debug, Clone, Copy)]
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
        let proto_request = ctx.state.proto_request.take().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Proto request not built"
            );
            error::internal_error("Proto request not built")
        })?;

        let clients = ctx.state.clients.as_mut().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error("Client acquisition not completed")
        })?;

        // Extract dispatch metadata for tracing span
        let request_id = ctx
            .state
            .dispatch
            .as_ref()
            .map(|d| d.request_id.as_str())
            .unwrap_or("unknown");
        let model = ctx
            .state
            .dispatch
            .as_ref()
            .map(|d| d.model.as_str())
            .unwrap_or("unknown");

        // Create OTEL span for gRPC request execution
        let span = info_span!(
            target: "sgl_model_gateway::otel-trace",
            "grpc_generate",
            request_id = %request_id,
            model = %model,
            mode = ?self.mode,
        );

        let result = async {
            match self.mode {
                ExecutionMode::Single => self.execute_single(proto_request, clients).await,
                ExecutionMode::DualDispatch => {
                    self.execute_dual_dispatch(proto_request, clients).await
                }
            }
        }
        .instrument(span)
        .await?;

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
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients.single_mut().ok_or_else(|| {
            error!(
                function = "execute_single",
                "Expected single client but got dual"
            );
            error::internal_error("Expected single client but got dual")
        })?;

        let stream = client.generate(proto_request).await.map_err(|e| {
            error!(
                function = "execute_single",
                error = %e,
                "Failed to start generation"
            );
            error::internal_error(format!("Failed to start generation: {}", e))
        })?;

        Ok(ExecutionResult::Single { stream })
    }

    async fn execute_dual_dispatch(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
    ) -> Result<ExecutionResult, Response> {
        let (prefill_client, decode_client) = clients.dual_mut().ok_or_else(|| {
            error!(
                function = "execute_dual_dispatch",
                "Expected dual clients but got single"
            );
            error::internal_error("Expected dual clients but got single")
        })?;

        let prefill_request = proto_request.clone_inner();
        let decode_request = proto_request;

        let (prefill_result, decode_result): (StreamResult, StreamResult) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Handle prefill result
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_dual_dispatch",
                    error = %e,
                    "Prefill worker failed to start"
                );
                return Err(error::internal_error(format!(
                    "Prefill worker failed to start: {}",
                    e
                )));
            }
        };

        // Handle decode result
        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_dual_dispatch",
                    error = %e,
                    "Decode worker failed to start"
                );
                return Err(error::internal_error(format!(
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
