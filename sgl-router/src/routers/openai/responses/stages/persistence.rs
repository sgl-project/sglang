//! Persistence stage for responses pipeline
//!
//! This stage:
//! - Masks tools as MCP in response
//! - Patches response with metadata
//! - Stores response and conversation items
//! - Returns final JSON response

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use tracing::warn;

use super::ResponsesStage;
use crate::routers::openai::{
    conversations::persist_conversation_items,
    responses::{
        utils::{mask_tools_as_mcp, patch_streaming_response_json},
        ResponsesRequestContext,
    },
};

/// Persistence stage for responses pipeline
pub struct ResponsesPersistenceStage;

#[async_trait]
impl ResponsesStage for ResponsesPersistenceStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        // Get processed response
        let mut processed = ctx.state.processed.take().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Response processing stage not completed",
            )
                .into_response()
        })?;

        // Get context for previous_response_id
        let original_previous_response_id = ctx
            .state
            .context
            .as_ref()
            .and_then(|c| c.previous_response_id.as_deref());

        // Mask tools as MCP
        mask_tools_as_mcp(&mut processed.json_response, ctx.request());

        // Patch response with metadata
        patch_streaming_response_json(
            &mut processed.json_response,
            ctx.request(),
            original_previous_response_id,
        );

        // Persist conversation items and response
        if let Err(err) = persist_conversation_items(
            ctx.dependencies.conversation_storage.clone(),
            ctx.dependencies.conversation_item_storage.clone(),
            ctx.dependencies.response_storage.clone(),
            &processed.json_response,
            ctx.request(),
        )
        .await
        {
            warn!("Failed to persist conversation items: {}", err);
        }

        // Return final response
        Ok(Some(
            (StatusCode::OK, Json(processed.json_response)).into_response(),
        ))
    }

    fn name(&self) -> &'static str {
        "ResponsesPersistence"
    }
}
