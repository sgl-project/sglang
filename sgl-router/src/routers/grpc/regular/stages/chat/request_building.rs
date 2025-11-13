//! Chat request building stage: Build proto GenerateRequest for chat requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::grpc::{
    client::GrpcClient,
    common::stages::{helpers, PipelineStage},
    context::{ClientSelection, RequestContext, WorkerSelection},
    error,
    proto_wrapper::ProtoGenerateRequest,
};

/// Chat request building stage
///
/// Extracts chat-specific request building logic from the old unified RequestBuildingStage.
pub struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for ChatRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error("Client acquisition not completed")
        })?;

        let chat_request = ctx.chat_request_arc();

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Build chat request
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());
        let body_ref = prep.filtered_request.as_ref().unwrap_or(&chat_request);

        // Dispatch to the appropriate client based on backend type
        let mut proto_request = match builder_client {
            GrpcClient::Sglang(sglang_client) => {
                let req = sglang_client
                    .build_generate_request_from_chat(
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
                    .map_err(|e| {
                        error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to build SGLang generate request");
                        error::bad_request(format!("Invalid request parameters: {}", e))
                    })?;
                ProtoGenerateRequest::Sglang(Box::new(req))
            }
            GrpcClient::Vllm(vllm_client) => {
                let req = vllm_client
                    .build_generate_request_from_chat(
                        request_id,
                        body_ref,
                        prep.processed_messages.as_ref().unwrap().text.clone(),
                        prep.token_ids.clone(),
                        prep.tool_constraints.clone(),
                    )
                    .map_err(|e| {
                        error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to build vLLM generate request");
                        error::bad_request(format!("Invalid request parameters: {}", e))
                    })?;
                ProtoGenerateRequest::Vllm(Box::new(req))
            }
        };

        // Inject PD metadata if needed
        if self.inject_pd_metadata {
            if let WorkerSelection::Dual { prefill, .. } = ctx.state.workers.as_ref().unwrap() {
                helpers::inject_bootstrap_metadata(&mut proto_request, prefill);
            }
        }

        ctx.state.proto_request = Some(proto_request);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
