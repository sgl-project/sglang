//! Harmony Request Building Stage: Build gRPC request from Harmony-encoded tokens

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error};
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext, RequestType, WorkerSelection},
        proto_wrapper::ProtoGenerateRequest,
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
}

#[async_trait]
impl PipelineStage for HarmonyRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Get preparation output
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "HarmonyRequestBuildingStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error("Preparation not completed")
        })?;

        // Get clients
        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "HarmonyRequestBuildingStage::execute",
                "Client acquisition stage not completed"
            );
            error::internal_error("Client acquisition not completed")
        })?;
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Harmony model support not yet implemented for vLLM
        if builder_client.is_vllm() {
            return Err(error::not_implemented(
                "Harmony model support is not yet implemented for vLLM backend. \
                 Please use runtime_type: sglang for Harmony models.",
            ));
        }

        // Generate request_id based on request type
        let request_id = match &ctx.input.request_type {
            RequestType::Chat(_) => format!("chatcmpl-{}", Uuid::new_v4()),
            RequestType::Responses(_) => format!("responses-{}", Uuid::new_v4()),
            RequestType::Generate(_) => {
                error!(
                    function = "HarmonyRequestBuildingStage::execute",
                    "Generate request type not supported for Harmony models"
                );
                return Err(error::bad_request(
                    "Generate requests are not supported with Harmony models".to_string(),
                ));
            }
        };

        // Build gRPC request using token_ids directly (Harmony encoding already handled message rendering)
        let placeholder_processed_text = "[harmony]".to_string();

        // Harmony is SGLang-only, so we can safely unwrap as SGLang
        let sglang_client = builder_client.as_sglang();
        let proto_request_inner = match &ctx.input.request_type {
            RequestType::Chat(request) => {
                // Use filtered request if present from preparation; otherwise original
                let body = prep.filtered_request.as_ref().unwrap_or(request.as_ref());

                sglang_client
                    .build_generate_request_from_chat(
                        request_id,
                        body,
                        placeholder_processed_text,
                        prep.token_ids.clone(),
                        None,
                        prep.tool_constraints.clone(),
                    )
                    .map_err(|e| {
                        error!(
                            function = "HarmonyRequestBuildingStage::execute",
                            error = %e,
                            "Failed to build generate request from chat"
                        );
                        error::bad_request(format!("Invalid request parameters: {}", e))
                    })?
            }
            RequestType::Responses(request) => sglang_client
                .build_generate_request_from_responses(
                    request_id,
                    request.as_ref(),
                    placeholder_processed_text,
                    prep.token_ids.clone(),
                    prep.harmony_stop_ids.clone(),
                    prep.tool_constraints.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "HarmonyRequestBuildingStage::execute",
                        error = %e,
                        "Failed to build generate request from responses"
                    );
                    error::bad_request(format!("Invalid request parameters: {}", e))
                })?,
            _ => unreachable!(),
        };

        let mut proto_request = ProtoGenerateRequest::Sglang(Box::new(proto_request_inner));

        // Inject Harmony stop token IDs into sampling params for ALL Harmony requests
        // These stop tokens (<|return|> and <|call|>) prevent the model from generating
        // malformed Harmony sequences
        if let Some(harmony_stops) = &prep.harmony_stop_ids {
            let sglang_req = proto_request.as_sglang_mut();
            if let Some(params) = sglang_req.sampling_params.as_mut() {
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
                helpers::inject_bootstrap_metadata(&mut proto_request, prefill);
            }
        }

        ctx.state.proto_request = Some(proto_request);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyRequestBuilding"
    }
}
