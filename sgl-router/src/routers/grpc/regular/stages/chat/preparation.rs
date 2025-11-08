//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use std::borrow::Cow;

use async_trait::async_trait;
use axum::response::Response;

use crate::{
    protocols::chat::ChatCompletionRequest,
    routers::grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        error, utils,
    },
};

/// Chat preparation stage
///
/// Extracts chat-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub struct ChatPreparationStage;

#[async_trait]
impl PipelineStage for ChatPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.chat_request_arc();
        self.prepare_chat(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatPreparation"
    }
}

impl ChatPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 2: Process messages and apply chat template
        let processed_messages =
            match utils::process_chat_messages(&body_ref, &*ctx.components.tokenizer) {
                Ok(msgs) => msgs,
                Err(e) => {
                    return Err(error::bad_request(e));
                }
            };

        // Step 3: Tokenize the processed text
        let encoding = match ctx.components.tokenizer.encode(&processed_messages.text) {
            Ok(encoding) => encoding,
            Err(e) => {
                return Err(error::internal_error(format!("Tokenization failed: {}", e)));
            }
        };

        let token_ids = encoding.token_ids().to_vec();

        // Step 4: Build tool constraints if needed
        let tool_call_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, &request.tool_choice, &request.model)
                .map_err(|e| error::bad_request(format!("Invalid tool configuration: {}", e)))?
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
}
