//! Preparation stage: Filter tools, process messages, tokenize, build constraints

use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::{
    protocols::{chat::ChatCompletionRequest, common::InputIds, generate::GenerateRequest},
    routers::grpc::{
        context::{PreparationOutput, RequestContext, RequestType},
        utils,
    },
    tokenizer::traits::Tokenizer,
};

/// Preparation stage: Filter tools, process messages, tokenize, build constraints
pub struct PreparationStage;

#[async_trait]
impl PipelineStage for PreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Clone Arc before match to avoid borrow checker issues
        // (matching borrows ctx, but prepare_* methods need mutable borrow)
        // Arc clone is cheap (8 bytes) - avoids full request clone (15KB-200KB)
        let is_chat = matches!(&ctx.input.request_type, RequestType::Chat(_));

        if is_chat {
            let request_arc = ctx.chat_request_arc();
            self.prepare_chat(ctx, &request_arc).await?;
        } else {
            let request_arc = ctx.generate_request_arc();
            self.prepare_generate(ctx, &request_arc).await?;
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "Preparation"
    }
}

impl PreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Step 1: Filter tools if needed
        let body_ref = utils::filter_tools_for_request(request);

        // Step 2: Process messages and apply chat template
        let processed_messages =
            match utils::process_chat_messages(&body_ref, &*ctx.components.tokenizer) {
                Ok(msgs) => msgs,
                Err(e) => {
                    return Err(utils::bad_request_error(e));
                }
            };

        // Step 3: Tokenize the processed text
        let encoding = match ctx.components.tokenizer.encode(&processed_messages.text) {
            Ok(encoding) => encoding,
            Err(e) => {
                return Err(utils::internal_error_message(format!(
                    "Tokenization failed: {}",
                    e
                )));
            }
        };

        let token_ids = encoding.token_ids().to_vec();

        // Step 4: Build tool constraints if needed
        let tool_call_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, &request.tool_choice, &request.model).map_err(
                |e| utils::bad_request_error(format!("Invalid tool configuration: {}", e)),
            )?
        } else {
            None
        };

        // Step 5: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &ctx.components.tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(processed_messages.text.clone()),
            token_ids,
            processed_messages: Some(processed_messages),
            tool_constraints: tool_call_constraint,
            filtered_request: if matches!(body_ref, Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
            // Harmony fields (not used for regular preparation)
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        // Store stop decoder for reuse in response processing
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    async fn prepare_generate(
        &self,
        ctx: &mut RequestContext,
        request: &GenerateRequest,
    ) -> Result<(), Response> {
        // Resolve input (text, prompt, or input_ids)
        let (original_text, token_ids) = match self.resolve_generate_input(ctx, request) {
            Ok(res) => res,
            Err(msg) => {
                return Err(utils::bad_request_error(msg));
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
