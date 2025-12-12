//! Request building stage that delegates to endpoint-specific implementations

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatRequestBuildingStage, generate::GenerateRequestBuildingStage};
use crate::routers::{
    error as grpc_error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
    },
};

/// Request building stage (delegates to endpoint-specific implementations)
pub struct RequestBuildingStage {
    chat_stage: ChatRequestBuildingStage,
    generate_stage: GenerateRequestBuildingStage,
}

impl RequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self {
            chat_stage: ChatRequestBuildingStage::new(inject_pd_metadata),
            generate_stage: GenerateRequestBuildingStage::new(inject_pd_metadata),
        }
    }
}

#[async_trait]
impl PipelineStage for RequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            RequestType::Responses(_request) => {
                error!(
                    function = "RequestBuildingStage::execute",
                    "RequestType::Responses reached regular request building stage"
                );
                Err(grpc_error::internal_error(
                    "RequestType::Responses reached regular request building stage",
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "RequestBuilding"
    }
}
