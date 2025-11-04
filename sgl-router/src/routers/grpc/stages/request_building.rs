//! Request building stage: Build proto GenerateRequest

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use proto::DisaggregatedParams;
use rand::Rng;
use tracing::debug;
use uuid::Uuid;

use super::PipelineStage;
use crate::{
    core::Worker,
    grpc_client::proto,
    routers::grpc::{
        context::{ClientSelection, RequestContext, RequestType, WorkerSelection},
        error,
    },
};

/// Request building stage: Build proto GenerateRequest
pub struct RequestBuildingStage {
    inject_pd_metadata: bool,
}

impl RequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for RequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx
            .state
            .preparation
            .as_ref()
            .ok_or_else(|| error::internal_error("Preparation not completed"))?;

        let clients = ctx
            .state
            .clients
            .as_ref()
            .ok_or_else(|| error::internal_error("Client acquisition not completed"))?;

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        let mut proto_request = match &ctx.input.request_type {
            RequestType::Chat(request) => {
                let request_id = format!("chatcmpl-{}", Uuid::new_v4());
                let body_ref = prep.filtered_request.as_ref().unwrap_or(request);

                builder_client
                    .build_generate_request(
                        request_id,
                        body_ref,
                        prep.processed_messages.as_ref().unwrap().text.clone(),
                        prep.token_ids.clone(),
                        prep.processed_messages
                            .as_ref()
                            .unwrap()
                            .multimodal_inputs
                            .clone(),
                        prep.tool_constraints.clone(),
                    )
                    .map_err(|e| error::bad_request(format!("Invalid request parameters: {}", e)))?
            }
            RequestType::Generate(request) => {
                let request_id = request
                    .rid
                    .clone()
                    .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

                builder_client
                    .build_plain_generate_request(
                        request_id,
                        request,
                        prep.original_text.clone(),
                        prep.token_ids.clone(),
                    )
                    .map_err(error::bad_request)?
            }
            RequestType::Responses(_request) => {
                // Responses API builds request during the MCP loop
                // For now, create minimal request - responses handler will populate it
                let request_id = format!("resp-{}", Uuid::new_v4());

                proto::GenerateRequest {
                    request_id,
                    ..Default::default()
                }
            }
        };

        // Inject PD metadata if needed
        if self.inject_pd_metadata {
            if let WorkerSelection::Dual { prefill, .. } = ctx.state.workers.as_ref().unwrap() {
                self.inject_bootstrap_metadata(&mut proto_request, prefill);
            }
        }

        ctx.state.proto_request = Some(proto_request);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestBuilding"
    }
}

impl RequestBuildingStage {
    fn inject_bootstrap_metadata(
        &self,
        request: &mut proto::GenerateRequest,
        prefill_worker: &Arc<dyn Worker>,
    ) {
        let hostname = prefill_worker.bootstrap_host();
        let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

        // Generate room ID for bootstrap
        let room_id = rand::rng().random_range(0..i32::MAX);

        // Create DisaggregatedParams
        let disagg_params = DisaggregatedParams {
            bootstrap_host: hostname.to_string(),
            bootstrap_port: bootstrap_port as i32,
            bootstrap_room: room_id,
        };

        // Inject metadata directly into request
        request.disaggregated_params = Some(disagg_params);

        debug!(
            "Injected bootstrap metadata: host={}, port={}, room={}",
            hostname, bootstrap_port, room_id
        );
    }
}
