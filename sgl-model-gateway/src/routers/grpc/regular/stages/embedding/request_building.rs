//! Request building stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
        proto_wrapper::{ProtoEmbedRequest, ProtoRequest},
    },
};

/// Request building stage for embedding requests
pub(crate) struct EmbeddingRequestBuildingStage;

impl EmbeddingRequestBuildingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingRequestBuildingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for EmbeddingRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Extract log_metrics from embedding or classify request (both use same backend)
        let log_metrics = match &ctx.input.request_type {
            RequestType::Embedding(req) => req.log_metrics,
            RequestType::Classify(req) => req.log_metrics,
            _ => {
                error!(
                    function = "EmbeddingRequestBuildingStage::execute",
                    "Invalid request type: expected Embedding or Classify"
                );
                return Err(error::internal_error(
                    "invalid_request_type",
                    "Expected Embedding or Classify request",
                ));
            }
        };

        // Preparation output should have tokenized input
        let prep_output = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "EmbeddingRequestBuildingStage::execute",
                "Preparation output missing"
            );
            error::internal_error("preparation_missing", "Preparation output missing")
        })?;

        // Extract client
        let client = ctx
            .state
            .clients
            .as_ref()
            .and_then(|c| c.single())
            .ok_or_else(|| {
                error!(
                    function = "EmbeddingRequestBuildingStage::execute",
                    "Client not selected"
                );
                error::internal_error("client_missing", "Client not selected")
            })?;

        // Generate request ID with appropriate prefix based on request type
        let request_id = match &ctx.input.request_type {
            RequestType::Embedding(_) => format!("embed-{}", Uuid::new_v4()),
            RequestType::Classify(_) => format!("classify-{}", Uuid::new_v4()),
            _ => format!("embed-{}", Uuid::new_v4()), // fallback
        };

        // Extract original text
        let original_text = prep_output.original_text.clone();

        // Use backend-specific builder to create ProtoEmbedRequest
        // Currently only SGLang supports embedding via gRPC
        let sglang_client = client.as_sglang();

        let sglang_req = sglang_client.build_embed_request(
            request_id.clone(),
            original_text,
            prep_output.token_ids.clone(),
            log_metrics,
        );

        let proto_req = ProtoEmbedRequest::Sglang(Box::new(sglang_req));

        ctx.state.proto_request = Some(ProtoRequest::Embed(proto_req));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingRequestBuilding"
    }
}
