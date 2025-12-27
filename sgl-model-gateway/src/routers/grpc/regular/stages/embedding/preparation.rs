//! Preparation stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    protocols::common::GenerationRequest,
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{PreparationOutput, RequestContext, RequestType},
            utils,
        },
    },
};

pub struct EmbeddingPreparationStage;

impl EmbeddingPreparationStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingPreparationStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for EmbeddingPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Extract embedding request
        let request = if let RequestType::Embedding(req) = &ctx.input.request_type {
            req
        } else {
            error!(
                function = "EmbeddingPreparationStage::execute",
                "Invalid request type: expected Embedding"
            );
            return Err(error::internal_error(
                "invalid_request_type",
                "Expected Embedding request",
            ));
        };

        // Extract text from request
        let text = request.extract_text_for_routing();
        if text.is_empty() {
            return Err(error::bad_request(
                "empty_input",
                "Input text cannot be empty",
            ));
        }

        // Resolve tokenizer from registry (cached for potential reuse)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "EmbeddingPreparationStage::execute").map_err(|e| *e)?;

        // Tokenize
        let token_ids = tokenizer
            .encode(&text)
            .map_err(|e| {
                error!(
                    function = "EmbeddingPreparationStage::execute",
                    error = %e,
                    "Tokenization failed"
                );
                error::bad_request("tokenization_failed", format!("Tokenization failed: {}", e))
            })?
            .token_ids()
            .to_vec();

        // Store preparation output
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(text),
            token_ids,
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingPreparation"
    }
}
