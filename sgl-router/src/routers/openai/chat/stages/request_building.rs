//! Request Building stage for chat pipeline
//!
//! This stage:
//! - Builds the HTTP request payload
//! - Strips SGLang-specific fields
//! - Applies provider-specific transformations
//! - Serializes to JSON

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::to_value;

use super::ChatStage;
use crate::routers::openai::{
    chat::{ChatRequestContext, PayloadOutput},
    utils::SGLANG_SPECIFIC_FIELDS,
};

/// Request building stage for chat pipeline
pub struct ChatRequestBuildingStage;

#[async_trait]
impl ChatStage for ChatRequestBuildingStage {
    async fn execute(&self, ctx: &mut ChatRequestContext) -> Result<(), Response> {
        // Get discovery output (for validation)
        ctx.state.discovery.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Model discovery stage not completed",
            )
                .into_response()
        })?;

        // Serialize chat request to JSON
        let is_streaming = ctx.request().stream;
        let mut payload = to_value(ctx.request()).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("Failed to serialize request: {}", e),
            )
                .into_response()
        })?;

        // Strip SGLang-specific fields and apply transformations
        if let Some(obj) = payload.as_object_mut() {
            obj.retain(|k, _| !SGLANG_SPECIFIC_FIELDS.contains(&k.as_str()));

            // Remove logprobs if false (Gemini compatibility)
            if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                obj.remove("logprobs");
            }
        }

        // Store payload output
        ctx.state.payload = Some(PayloadOutput {
            json_payload: payload,
            is_streaming,
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
