//! Response processing stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    protocols::embedding::{EmbeddingObject, EmbeddingResponse},
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{ExecutionResult, FinalResponse, RequestContext},
            proto_wrapper::ProtoEmbedComplete,
        },
    },
};

/// Response processing stage for embedding requests
pub(crate) struct EmbeddingResponseProcessingStage;

impl EmbeddingResponseProcessingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for EmbeddingResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "EmbeddingResponseProcessingStage::execute",
                "Execution result missing"
            );
            error::internal_error("execution_result_missing", "Execution result missing")
        })?;

        // Expect Embedding result variant
        let proto_response = if let ExecutionResult::Embedding { response } = execution_result {
            response
        } else {
            error!(
                function = "EmbeddingResponseProcessingStage::execute",
                "Invalid execution result: expected Embedding"
            );
            return Err(error::internal_error(
                "invalid_execution_result",
                "Expected Embedding result",
            ));
        };

        // Convert proto response to HTTP response
        let embedding_response = self
            .convert_response(ctx, proto_response)
            .map_err(|boxed_err| *boxed_err)?;

        // Store in context for pipeline to extract
        ctx.state.response.final_response = Some(FinalResponse::Embedding(embedding_response));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingResponseProcessing"
    }
}

impl EmbeddingResponseProcessingStage {
    fn convert_response(
        &self,
        ctx: &RequestContext,
        proto: ProtoEmbedComplete,
    ) -> Result<EmbeddingResponse, Box<Response>> {
        let dispatch = ctx.state.dispatch.as_ref().ok_or_else(|| {
            error!(
                function = "EmbeddingResponseProcessingStage::convert_response",
                "Dispatch metadata missing in context"
            );
            error::internal_error("dispatch_missing", "Dispatch metadata missing")
        })?;

        let model = dispatch.model.clone();

        // Convert flat embedding vector to response
        // single input -> single embedding object

        let embedding_data = EmbeddingObject {
            object: "embedding".to_string(),
            embedding: proto.embedding().to_vec(),
            index: 0,
        };

        // Casting i32 to u32 for usage stats
        let prompt_tokens = proto.prompt_tokens().max(0) as u32;

        let usage = crate::protocols::common::UsageInfo {
            prompt_tokens,
            total_tokens: prompt_tokens, // Embedding has no completion tokens
            completion_tokens: 0,
            prompt_tokens_details: None,
            reasoning_tokens: None,
        };

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data: vec![embedding_data],
            model,
            usage,
        })
    }
}
