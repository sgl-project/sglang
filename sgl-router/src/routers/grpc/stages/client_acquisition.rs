//! Client acquisition stage: Get gRPC clients from selected workers

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::grpc::{
    context::{ClientSelection, RequestContext, WorkerSelection},
    error, utils,
};

/// Client acquisition stage: Get gRPC clients from selected workers
pub struct ClientAcquisitionStage;

#[async_trait]
impl PipelineStage for ClientAcquisitionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let workers = ctx
            .state
            .workers
            .as_ref()
            .ok_or_else(|| error::internal_error("Worker selection not completed"))?;

        let clients = match workers {
            WorkerSelection::Single { worker } => {
                let client = utils::get_grpc_client_from_worker(worker).await?;
                ClientSelection::Single { client }
            }
            WorkerSelection::Dual { prefill, decode } => {
                let prefill_client = utils::get_grpc_client_from_worker(prefill).await?;
                let decode_client = utils::get_grpc_client_from_worker(decode).await?;
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
