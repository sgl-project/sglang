//! Response processing stage that delegates to endpoint-specific implementations

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{
    chat::ChatResponseProcessingStage, classify::ClassifyResponseProcessingStage,
    embedding::response_processing::EmbeddingResponseProcessingStage,
    generate::GenerateResponseProcessingStage,
};
use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
        regular::{processor, streaming},
    },
};

/// Response processing stage (delegates to endpoint-specific implementations)
pub(crate) struct ResponseProcessingStage {
    chat_stage: ChatResponseProcessingStage,
    generate_stage: GenerateResponseProcessingStage,
    embedding_stage: EmbeddingResponseProcessingStage,
    classify_stage: ClassifyResponseProcessingStage,
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
            embedding_stage: EmbeddingResponseProcessingStage::new(),
            classify_stage: ClassifyResponseProcessingStage::new(),
        }
    }
}

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            RequestType::Embedding(_) => self.embedding_stage.execute(ctx).await,
            RequestType::Classify(_) => self.classify_stage.execute(ctx).await,
            RequestType::Responses(_) => {
                error!(
                    function = "ResponseProcessingStage::execute",
                    "RequestType::Responses reached regular response processing stage"
                );
                Err(error::internal_error(
                    "responses_in_wrong_pipeline",
                    "RequestType::Responses reached regular response processing stage",
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "ResponseProcessing"
    }
}
