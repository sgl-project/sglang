//! Request execution stage: Execute gRPC requests (single or dual dispatch)

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::{
    grpc_client::{proto, sglang_scheduler::AbortOnDropStream},
    routers::grpc::{
        context::{ClientSelection, ExecutionResult, RequestContext},
        error,
    },
};

type StreamResult = Result<AbortOnDropStream, Box<dyn std::error::Error + Send + Sync>>;

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
            .ok_or_else(|| error::internal_error("Proto request not built"))?;

        let clients = ctx
            .state
            .clients
            .as_mut()
            .ok_or_else(|| error::internal_error("Client acquisition not completed"))?;

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
            .ok_or_else(|| error::internal_error("Expected single client but got dual"))?;

        let stream = client
            .generate(proto_request)
            .await
            .map_err(|e| error::internal_error(format!("Failed to start generation: {}", e)))?;

        Ok(ExecutionResult::Single { stream })
    }

    async fn execute_dual_dispatch(
        &self,
        proto_request: proto::GenerateRequest,
        clients: &mut ClientSelection,
    ) -> Result<ExecutionResult, Response> {
        let (prefill_client, decode_client) = clients
            .dual_mut()
            .ok_or_else(|| error::internal_error("Expected dual clients but got single"))?;

        let prefill_request = proto_request.clone();
        let decode_request = proto_request;

        let (prefill_result, decode_result): (StreamResult, StreamResult) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Handle prefill result
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
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
