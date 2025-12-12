//! Preparation stage that delegates to endpoint-specific implementations
//!
//! This stage checks RequestType at runtime and delegates to the appropriate
//! endpoint-specific stage (ChatPreparationStage or GeneratePreparationStage).

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatPreparationStage, generate::GeneratePreparationStage};
use crate::routers::{
    error as grpc_error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
    },
};

/// Preparation stage (delegates to endpoint-specific implementations)
pub struct PreparationStage {
    chat_stage: ChatPreparationStage,
    generate_stage: GeneratePreparationStage,
}

impl PreparationStage {
    pub fn new() -> Self {
        Self {
            chat_stage: ChatPreparationStage,
            generate_stage: GeneratePreparationStage,
        }
    }
}

impl Default for PreparationStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for PreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            RequestType::Responses(_) => {
                error!(
                    function = "PreparationStage::execute",
                    "RequestType::Responses reached regular preparation stage"
                );
                Err(grpc_error::internal_error(
                    "RequestType::Responses reached regular preparation stage",
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "Preparation"
    }
}
