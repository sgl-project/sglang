//! Request building stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
        proto_wrapper::{ProtoEmbedRequest, ProtoRequest},
    },
};

/// Request building stage for embedding requests
pub struct EmbeddingRequestBuildingStage;

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
        // Check if the request is of type Embedding
        if let RequestType::Embedding(_) = &ctx.input.request_type {
            // Proceed as expected
        } else {
            error!(
                function = "EmbeddingRequestBuildingStage::execute",
                "Invalid request type: expected Embedding"
            );
            return Err(error::internal_error(
                "invalid_request_type",
                "Expected Embedding request",
            ));
        }

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

        // Extract request ID
        let request_id = ctx
            .state
            .dispatch
            .as_ref()
            .map(|d| d.request_id.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Extract original text
        let original_text = prep_output.original_text.clone();

        // Use backend-specific builder to create ProtoEmbedRequest
        // Currently only SGLang supports embedding via gRPC
        let sglang_client = client.as_sglang();
        let embedding_request = ctx.embedding_request();

        let sglang_req = sglang_client.build_embed_request(
            request_id.clone(),
            original_text,
            prep_output.token_ids.clone(),
            embedding_request.log_metrics,
        );

        let proto_req = ProtoEmbedRequest::Sglang(Box::new(sglang_req));

        ctx.state.proto_request = Some(ProtoRequest::Embed(proto_req));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingRequestBuilding"
    }
}
