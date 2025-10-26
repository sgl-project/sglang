//! Harmony Request Building Stage: Build gRPC request from Harmony-encoded tokens

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use rand::Rng;
use tracing::debug;
use uuid::Uuid;

use crate::{
    core::Worker,
    grpc_client::proto::{DisaggregatedParams, GenerateRequest, TokenizedInput},
    routers::grpc::{
        context::{RequestContext, RequestType, WorkerSelection},
        stages::PipelineStage,
        utils,
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
            .ok_or_else(|| utils::internal_error_static("Preparation not completed"))?;

        // Get clients
        let _clients = ctx
            .state
            .clients
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Client acquisition not completed"))?;

        // Generate request ID based on request type
        let request_id = match &ctx.input.request_type {
            RequestType::Chat(_) => format!("chatcmpl-{}", Uuid::new_v4()),
            RequestType::Generate(request) => request
                .rid
                .clone()
                .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4())),
            RequestType::Responses(_) => format!("responses-{}", Uuid::new_v4()),
        };

        // Build proto::GenerateRequest from Harmony-encoded input_ids
        // The tokenized field contains the input_ids we got from Harmony encoding
        let mut proto_request = GenerateRequest {
            request_id,
            tokenized: Some(TokenizedInput {
                original_text: String::new(), // Not needed for Harmony encoded inputs
                input_ids: prep.token_ids.clone(),
            }),
            ..Default::default()
        };

        // Inject Harmony stop token IDs into sampling params
        if let Some(harmony_stops) = &prep.harmony_stop_ids {
            if let Some(params) = proto_request.sampling_params.as_mut() {
                params.stop_token_ids.extend_from_slice(harmony_stops);
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
