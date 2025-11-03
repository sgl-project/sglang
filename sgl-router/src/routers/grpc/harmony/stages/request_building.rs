//! Harmony Request Building Stage: Build gRPC request from Harmony-encoded tokens

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use rand::Rng;
use tracing::debug;
use uuid::Uuid;

use crate::{
    core::Worker,
    grpc_client::proto::{DisaggregatedParams, GenerateRequest},
    routers::grpc::{
        context::{ClientSelection, RequestContext, RequestType, WorkerSelection},
        error,
        stages::PipelineStage,
    },
};

/// Harmony Request Building stage: Convert Harmony tokens to gRPC request
///
/// Takes the Harmony-encoded input_ids from preparation and builds a proto::GenerateRequest.
/// Unlike regular request building, this uses token_ids directly (Harmony encoding handles messages).
pub struct HarmonyRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl HarmonyRequestBuildingStage {
    /// Create a new Harmony request building stage
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }

    /// Inject PD (prefill-decode) bootstrap metadata
    fn inject_bootstrap_metadata(
        &self,
        request: &mut GenerateRequest,
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
            "Injected Harmony bootstrap metadata: host={}, port={}, room={}",
            hostname, bootstrap_port, room_id
        );
    }
}

#[async_trait]
impl PipelineStage for HarmonyRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Get preparation output
        let prep = ctx
            .state
            .preparation
            .as_ref()
            .ok_or_else(|| error::internal_error("Preparation not completed"))?;

        // Get clients
        let clients = ctx
            .state
            .clients
            .as_ref()
            .ok_or_else(|| error::internal_error("Client acquisition not completed"))?;
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Generate request_id based on request type
        let request_id = match &ctx.input.request_type {
            RequestType::Chat(_) => format!("chatcmpl-{}", Uuid::new_v4()),
            RequestType::Responses(_) => format!("responses-{}", Uuid::new_v4()),
            RequestType::Generate(_) => {
                return Err(error::bad_request(
                    "Generate requests are not supported with Harmony models".to_string(),
                ));
            }
        };

        // Build gRPC request using token_ids directly (Harmony encoding already handled message rendering)
        // Use a placeholder for original_text; Harmony uses input_ids for tokenization
        let placeholder_processed_text = "[harmony]".to_string();

        let mut proto_request = match &ctx.input.request_type {
            RequestType::Chat(request) => {
                // Use filtered request if present from preparation; otherwise original
                let body = prep.filtered_request.as_ref().unwrap_or(request.as_ref());

                builder_client
                    .build_generate_request(
                        request_id,
                        body,
                        placeholder_processed_text,
                        prep.token_ids.clone(),
                        None,
                        prep.tool_constraints.clone(),
                    )
                    .map_err(|e| error::bad_request(format!("Invalid request parameters: {}", e)))?
            }
            RequestType::Responses(request) => builder_client
                .build_generate_request_from_responses(
                    request_id,
                    request.as_ref(),
                    placeholder_processed_text,
                    prep.token_ids.clone(),
                    prep.harmony_stop_ids.clone(),
                )
                .map_err(|e| error::bad_request(format!("Invalid request parameters: {}", e)))?,
            _ => unreachable!(),
        };

        // Inject Harmony stop token IDs into sampling params for ALL Harmony requests
        // These stop tokens (<|return|> and <|call|>) prevent the model from generating
        // malformed Harmony sequences
        if let Some(harmony_stops) = &prep.harmony_stop_ids {
            if let Some(params) = proto_request.sampling_params.as_mut() {
                params.stop_token_ids.extend_from_slice(harmony_stops);
                debug!(
                    stop_token_count = harmony_stops.len(),
                    "Injected Harmony stop tokens into sampling params"
                );
            }
        }

        // Inject PD metadata if needed
        if self.inject_pd_metadata {
            if let Some(WorkerSelection::Dual { prefill, .. }) = ctx.state.workers.as_ref() {
                self.inject_bootstrap_metadata(&mut proto_request, prefill);
            }
        }

        ctx.state.proto_request = Some(proto_request);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyRequestBuilding"
    }
}
