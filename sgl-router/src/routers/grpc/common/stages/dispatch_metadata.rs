//! Dispatch metadata stage: Prepare metadata for dispatch

use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::PipelineStage;
use crate::routers::grpc::{
    context::{DispatchMetadata, RequestContext, RequestType, WorkerSelection},
    error,
};

/// Dispatch metadata stage: Prepare metadata for dispatch
pub struct DispatchMetadataStage;

#[async_trait]
impl PipelineStage for DispatchMetadataStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let proto_request = ctx.state.proto_request.as_ref().ok_or_else(|| {
            error!(
                function = "DispatchMetadataStage::execute",
                "Proto request not built"
            );
            error::internal_error("Proto request not built")
        })?;

        let request_id = proto_request.request_id().to_string();
        let model = match &ctx.input.request_type {
            RequestType::Chat(req) => req.model.clone(),
            RequestType::Generate(_req) => {
                // Generate requests don't have a model field
                // Use model_id from input or default
                ctx.input
                    .model_id
                    .clone()
                    .unwrap_or_else(|| "default".to_string())
            }
            RequestType::Responses(req) => req.model.clone(),
        };

        let weight_version = ctx
            .state
            .workers
            .as_ref()
            .map(|w| match w {
                WorkerSelection::Single { worker } => worker,
                WorkerSelection::Dual { decode, .. } => decode,
            })
            .and_then(|w| w.metadata().labels.get("weight_version").cloned())
            .unwrap_or_else(|| "default".to_string());

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ctx.state.dispatch = Some(DispatchMetadata {
            request_id,
            model,
            created,
            weight_version: Some(weight_version),
            is_streaming: ctx.is_streaming(),
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "DispatchMetadata"
    }
}
