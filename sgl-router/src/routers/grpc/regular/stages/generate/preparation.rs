//! Generate preparation stage: Resolve input, tokenize, create stop decoder

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;

use crate::{
    protocols::{common::InputIds, generate::GenerateRequest},
    routers::grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        error, utils,
    },
    tokenizer::traits::Tokenizer,
};

/// Generate preparation stage
///
/// Extracts generate-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub struct GeneratePreparationStage;

#[async_trait]
impl PipelineStage for GeneratePreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.generate_request_arc();
        self.prepare_generate(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "GeneratePreparation"
    }
}

impl GeneratePreparationStage {
    async fn prepare_generate(
        &self,
        ctx: &mut RequestContext,
        request: &GenerateRequest,
    ) -> Result<(), Response> {
        // Resolve input (text, prompt, or input_ids)
        let (original_text, token_ids) = match self.resolve_generate_input(ctx, request) {
            Ok(res) => res,
            Err(msg) => {
                return Err(error::bad_request(msg));
            }
        };

        // Create stop sequence decoder for generate requests
        let params = request.sampling_params.as_ref();
        let stop_decoder = utils::create_stop_decoder(
            &ctx.components.tokenizer,
            params.and_then(|p| p.stop.as_ref()),
            params.and_then(|p| p.stop_token_ids.as_ref()),
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            params.and_then(|p| p.no_stop_trim).unwrap_or(false),
        );

        ctx.state.preparation = Some(PreparationOutput {
            original_text,
            token_ids,
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            // Harmony fields (not used for generate requests)
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        // Store stop decoder
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    fn resolve_generate_input(
        &self,
        ctx: &RequestContext,
        request: &GenerateRequest,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        if let Some(text) = &request.text {
            return self
                .tokenize_single_text(&ctx.components.tokenizer, text)
                .map(|(original, ids)| (Some(original), ids));
        }

        // Handle input_ids - validate and convert
        if let Some(input_ids) = &request.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| u32::try_from(id))
                    .collect::<Result<Vec<u32>, _>>()
                    .map(|converted| (None, converted))
                    .map_err(|_| "input_ids must be non-negative".to_string()),
                InputIds::Batch(_) => {
                    Err("Batch input_ids are not supported over gRPC generate yet".to_string())
                }
            };
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    fn tokenize_single_text(
        &self,
        tokenizer: &Arc<dyn Tokenizer>,
        text: &str,
    ) -> Result<(String, Vec<u32>), String> {
        let encoding = tokenizer
            .encode(text)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        Ok((text.to_string(), encoding.token_ids().to_vec()))
    }
}
