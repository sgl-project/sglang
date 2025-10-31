//! Response processing stage: Handles both streaming and non-streaming responses
//!
//! - For streaming: Spawns background task and returns SSE response (early exit)
//! - For non-streaming: Collects all responses and builds final ChatCompletionResponse

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::grpc::{
    context::{FinalResponse, RequestContext, RequestType},
    processing, streaming, utils,
};

/// Response processing stage: Handles both streaming and non-streaming responses
///
/// - For streaming: Spawns background task and returns SSE response (early exit)
/// - For non-streaming: Collects all responses and builds final ChatCompletionResponse
pub struct ResponseProcessingStage {
    processor: processing::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl ResponseProcessingStage {
    pub fn new(
        processor: processing::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
        }
    }
}

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Delegate to request-type specific processing
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.process_chat_response(ctx).await,
            RequestType::Generate(_) => self.process_generate_response(ctx).await,
            RequestType::Responses(_) => Err(utils::bad_request_error(
                "Responses API processing must be handled by responses handler".to_string(),
            )),
        }
    }

    fn name(&self) -> &'static str {
        "ResponseProcessing"
    }
}

impl ResponseProcessingStage {
    async fn process_chat_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx
            .state
            .response
            .execution_result
            .take()
            .ok_or_else(|| utils::internal_error_static("No execution result"))?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Dispatch metadata not set"))?
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
        let request_logprobs = match &ctx.input.request_type {
            RequestType::Chat(req) => req.logprobs,
            _ => false,
        };

        let chat_request = ctx.chat_request_arc();

        let stop_decoder = ctx
            .state
            .response
            .stop_decoder
            .as_mut()
            .ok_or_else(|| utils::internal_error_static("Stop decoder not initialized"))?;

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

    async fn process_generate_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let start_time = Instant::now();
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx
            .state
            .response
            .execution_result
            .take()
            .ok_or_else(|| utils::internal_error_static("No execution result"))?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| utils::internal_error_static("Dispatch metadata not set"))?
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

        let stop_decoder = ctx
            .state
            .response
            .stop_decoder
            .as_mut()
            .ok_or_else(|| utils::internal_error_static("Stop decoder not initialized"))?;

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
