//! Response processing stage that delegates to endpoint-specific implementations

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;

use super::{chat::ChatResponseProcessingStage, generate::GenerateResponseProcessingStage};
use crate::routers::grpc::{
    common::stages::PipelineStage,
    context::{RequestContext, RequestType},
    error,
    regular::{processor, streaming},
};

/// Response processing stage (delegates to endpoint-specific implementations)
pub struct ResponseProcessingStage {
    chat_stage: ChatResponseProcessingStage,
    generate_stage: GenerateResponseProcessingStage,
}

impl ResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            chat_stage: ChatResponseProcessingStage::new(
                processor.clone(),
                streaming_processor.clone(),
            ),
            generate_stage: GenerateResponseProcessingStage::new(processor, streaming_processor),
        }
    }
}

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            RequestType::Responses(_) => Err(error::bad_request(
                "Responses API processing must be handled by responses handler".to_string(),
            )),
        }
    }

    fn name(&self) -> &'static str {
        "ResponseProcessing"
    }
}
