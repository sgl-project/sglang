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
pub(crate) struct ClientAcquisitionStage;

#[async_trait]
impl PipelineStage for ClientAcquisitionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(
                function = "ClientAcquisitionStage::execute",
                "Worker selection stage not completed"
            );
            error::internal_error(
                "worker_selection_not_completed",
                "Worker selection not completed",
            )
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
                        "vllm_pd_mode_not_supported",
                        "vLLM backend does not support prefill/decode disaggregated mode. \
                         Please use runtime_type: sglang for PD mode, or use a regular (non-PD) worker configuration."
                    ));
                }

                ClientSelection::Dual {
                    prefill: prefill_client,
                    decode: decode_client,
                }
            }
            WorkerSelection::Triple {
                encode,
                prefill,
                decode,
            } => {
                // EPD mode: encode, prefill, and decode all use gRPC
                // Get cached encoder client from worker
                let encode_client = utils::get_encoder_client_from_worker(encode).await?;
                let prefill_client = utils::get_grpc_client_from_worker(prefill).await?;
                let decode_client = utils::get_grpc_client_from_worker(decode).await?;

                // vLLM does not support EPD disaggregated mode
                if prefill_client.is_vllm() || decode_client.is_vllm() {
                    error!(
                        function = "ClientAcquisitionStage::execute",
                        "vLLM backend does not support encode/prefill/decode disaggregated mode"
                    );
                    return Err(error::bad_request(
                        "vllm_epd_mode_not_supported",
                        "vLLM backend does not support encode/prefill/decode disaggregated mode. \
                         Please use runtime_type: sglang for EPD mode, or use a regular worker configuration."
                    ));
                }

                tracing::debug!(
                    encode_url = %encode.url(),
                    prefill_url = %prefill.url(),
                    decode_url = %decode.url(),
                    "EPD mode: all workers use gRPC"
                );

                ClientSelection::Triple {
                    encode: encode_client,
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
