//! Request building stage that delegates to endpoint-specific implementations

use async_trait::async_trait;
use axum::response::Response;
use uuid::Uuid;

use super::{chat::ChatRequestBuildingStage, generate::GenerateRequestBuildingStage};
use crate::{
    grpc_client::proto,
    routers::grpc::{
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
                // Responses API builds request during the MCP loop
                // For now, create minimal request - responses handler will populate it
                let request_id = format!("resp-{}", Uuid::new_v4());

                ctx.state.proto_request = Some(proto::GenerateRequest {
                    request_id,
                    ..Default::default()
                });
                Ok(None)
            }
        }
    }

    fn name(&self) -> &'static str {
        "RequestBuilding"
    }
}
