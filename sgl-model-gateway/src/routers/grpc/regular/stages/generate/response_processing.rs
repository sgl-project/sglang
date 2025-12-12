//! Generate response processing stage: Handles both streaming and non-streaming responses

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{FinalResponse, RequestContext},
        regular::{processor, streaming},
    },
};

/// Generate response processing stage
///
/// Extracts generate-specific response processing logic from the old unified ResponseProcessingStage.
pub struct GenerateResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl GenerateResponseProcessingStage {
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
impl PipelineStage for GenerateResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        self.process_generate_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "GenerateResponseProcessing"
    }
}

impl GenerateResponseProcessingStage {
    async fn process_generate_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let start_time = Instant::now();
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "GenerateResponseProcessingStage::execute",
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
                    function = "GenerateResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("Dispatch metadata not set")
            })?
            .clone();

        if is_streaming {
            // Streaming: Use StreamingProcessor and return SSE response (done)
            return Ok(Some(
                self.streaming_processor.clone().process_streaming_generate(
                    execution_result,
                    ctx.generate_request_arc(), // Cheap Arc clone (8 bytes)
                    dispatch,
                ),
            ));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = ctx.generate_request().return_logprob.unwrap_or(false);
        let generate_request = ctx.generate_request_arc();

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "GenerateResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error("Stop decoder not initialized")
        })?;

        let result_array = self
            .processor
            .process_non_streaming_generate_response(
                execution_result,
                generate_request,
                dispatch,
                stop_decoder,
                request_logprobs,
                start_time,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Generate(result_array));

        Ok(None)
    }
}
