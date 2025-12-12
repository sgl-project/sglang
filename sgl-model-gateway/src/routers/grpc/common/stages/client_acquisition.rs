//! Client acquisition stage: Get gRPC clients from selected workers

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::PipelineStage;
use crate::routers::{
    error,
    grpc::{
        context::{ClientSelection, RequestContext, WorkerSelection},
        utils,
    },
};

/// Client acquisition stage: Get gRPC clients from selected workers
pub struct ClientAcquisitionStage;

#[async_trait]
impl PipelineStage for ClientAcquisitionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(
                function = "ClientAcquisitionStage::execute",
                "Worker selection stage not completed"
            );
            error::internal_error("Worker selection not completed")
        })?;

        let clients = match workers {
            WorkerSelection::Single { worker } => {
                let client = utils::get_grpc_client_from_worker(worker).await?;
                ClientSelection::Single { client }
            }
            WorkerSelection::Dual { prefill, decode } => {
                let prefill_client = utils::get_grpc_client_from_worker(prefill).await?;
                let decode_client = utils::get_grpc_client_from_worker(decode).await?;

                // vLLM does not support dual (PD disaggregated) mode
                if prefill_client.is_vllm() || decode_client.is_vllm() {
                    error!(
                        function = "ClientAcquisitionStage::execute",
                        "vLLM backend does not support dual (PD disaggregated) mode"
                    );
                    return Err(error::bad_request(
                        "vLLM backend does not support prefill/decode disaggregated mode. \
                         Please use runtime_type: sglang for PD mode, or use a regular (non-PD) worker configuration."
                    ));
                }

                ClientSelection::Dual {
                    prefill: prefill_client,
                    decode: decode_client,
                }
            }
        };

        ctx.state.clients = Some(clients);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ClientAcquisition"
    }
}
