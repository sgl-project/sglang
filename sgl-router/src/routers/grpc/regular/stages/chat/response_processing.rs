//! Chat response processing stage: Handles both streaming and non-streaming responses
//!
//! - For streaming: Spawns background task and returns SSE response (early exit)
//! - For non-streaming: Collects all responses and builds final ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::grpc::{
    common::stages::PipelineStage,
    context::{FinalResponse, RequestContext},
    error,
    regular::{processor, streaming},
};

/// Chat response processing stage
///
/// Extracts chat-specific response processing logic from the old unified ResponseProcessingStage.
pub struct ChatResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl ChatResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
        }
    }
}

#[async_trait]
impl PipelineStage for ChatResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        self.process_chat_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "ChatResponseProcessing"
    }
}

impl ChatResponseProcessingStage {
    async fn process_chat_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("No execution result")
        })?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "ChatResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("Dispatch metadata not set")
            })?
            .clone();

        if is_streaming {
            // Streaming: Use StreamingProcessor and return SSE response (done)
            return Ok(Some(
                self.streaming_processor.clone().process_streaming_response(
                    execution_result,
                    ctx.chat_request_arc(), // Cheap Arc clone (8 bytes)
                    dispatch,
                ),
            ));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = ctx.chat_request().logprobs;

        let chat_request = ctx.chat_request_arc();

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error("Stop decoder not initialized")
        })?;

        let response = self
            .processor
            .process_non_streaming_chat_response(
                execution_result,
                chat_request,
                dispatch,
                stop_decoder,
                request_logprobs,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Chat(response));

        Ok(None)
    }
}
