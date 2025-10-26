//! Harmony Response Processing Stage: Parse Harmony channels to ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use uuid::Uuid;

use super::super::{HarmonyResponseProcessor, HarmonyStreamingProcessor};
use crate::routers::grpc::{context::RequestContext, stages::PipelineStage, utils};

/// Harmony Response Processing stage: Parse and format Harmony responses
///
/// Takes output tokens from execution and parses them using HarmonyParserAdapter
/// to extract analysis, tool calls, and final response text from Harmony channels.
pub struct HarmonyResponseProcessingStage {
    processor: HarmonyResponseProcessor,
    streaming_processor: Arc<HarmonyStreamingProcessor>,
}

impl HarmonyResponseProcessingStage {
    /// Create a new Harmony response processing stage
    pub fn new() -> Self {
        Self {
            processor: HarmonyResponseProcessor::new(),
            streaming_processor: Arc::new(HarmonyStreamingProcessor::new()),
        }
    }
}

impl Default for HarmonyResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for HarmonyResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Get execution result (output tokens from model)
        let execution_result = ctx
            .state
            .response
            .execution_result
            .take()
            .ok_or_else(|| utils::internal_error_static("No execution result"))?;

        let is_streaming = ctx.is_streaming();

        // For streaming, delegate to streaming processor and return SSE response
        if is_streaming {
            return Ok(Some(
                self.streaming_processor
                    .clone()
                    .process_streaming_response(execution_result, ctx.chat_request_arc()),
            ));
        }

        // For non-streaming, delegate to response processor
        let chat_request = ctx.chat_request_arc();

        // Generate or use provided request ID for response
        let request_id = Uuid::new_v4().to_string();

        self.processor
            .process_harmony_response(execution_result, chat_request, request_id)
            .await
            .map(Some)
    }

    fn name(&self) -> &'static str {
        "HarmonyResponseProcessing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response_processing_stage_creation() {
        let stage = HarmonyResponseProcessingStage::new();
        assert_eq!(stage.name(), "HarmonyResponseProcessing");
    }
}
