//! Model Discovery stage for chat pipeline
//!
//! This stage:
//! - Determines which endpoint has the requested model
//! - Uses caching to avoid repeated endpoint probing
//! - Handles single endpoint (fast path) vs multiple endpoints

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::ChatStage;
use crate::routers::openai::{
    chat::{ChatRequestContext, DiscoveryOutput},
    utils::discover_model_endpoint,
};

/// Model discovery stage for chat pipeline
pub struct ChatModelDiscoveryStage;

impl ChatModelDiscoveryStage {
    /// Model discovery cache TTL (1 hour)
    const MODEL_CACHE_TTL_SECS: u64 = 3600;
}

#[async_trait]
impl ChatStage for ChatModelDiscoveryStage {
    async fn execute(&self, ctx: &mut ChatRequestContext) -> Result<(), Response> {
        // Get validation output
        let validation = ctx.state.validation.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Validation stage not completed",
            )
                .into_response()
        })?;

        // Use shared model discovery logic
        let discovery_output = discover_model_endpoint(
            &ctx.dependencies.http_client,
            &ctx.dependencies.worker_urls,
            &ctx.dependencies.model_cache,
            ctx.model(),
            validation.auth_header.as_deref(),
            Self::MODEL_CACHE_TTL_SECS,
        )
        .await
        .map_err(|(status, msg)| (status, msg).into_response())?;

        // Store discovery output in chat-specific format
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: discovery_output.endpoint_url,
            model: discovery_output.model,
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChatModelDiscovery"
    }
}
