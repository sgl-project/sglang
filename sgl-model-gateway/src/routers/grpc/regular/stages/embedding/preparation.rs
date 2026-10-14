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

pub(crate) struct EmbeddingPreparationStage;

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
        // Extract text from embedding or classify request (both use same preparation)
        let text = match &ctx.input.request_type {
            RequestType::Embedding(req) => req.extract_text_for_routing(),
            RequestType::Classify(req) => req.extract_text_for_routing(),
            _ => {
                error!(
                    function = "EmbeddingPreparationStage::execute",
                    "Invalid request type: expected Embedding or Classify"
                );
                return Err(error::internal_error(
                    "invalid_request_type",
                    "Expected Embedding or Classify request",
                ));
            }
        };
        if text.is_empty() {
            return Err(error::bad_request(
                "empty_input",
                "Input text cannot be empty",
            ));
        }

        // Resolve tokenizer from registry (cached for potential reuse)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "EmbeddingPreparationStage::execute").map_err(|e| *e)?;

        // Tokenize with special tokens (BOS/EOS) for embeddings
        // This matches Python's transformers behavior which reads add_bos_token/add_eos_token from tokenizer_config.json
        let token_ids = tokenizer
            .encode(&text, true)
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
