//! Model Discovery stage for responses pipeline
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

use super::ResponsesStage;
use crate::routers::openai::{
    responses::{DiscoveryOutput, ResponsesRequestContext},
    utils::discover_model_endpoint,
};

/// Model discovery stage for responses pipeline
pub struct ResponsesModelDiscoveryStage;

impl ResponsesModelDiscoveryStage {
    /// Model discovery cache TTL (1 hour)
    const MODEL_CACHE_TTL_SECS: u64 = 3600;
}

#[async_trait]
impl ResponsesStage for ResponsesModelDiscoveryStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
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

        // Store discovery output in responses-specific format
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: discovery_output.endpoint_url,
            model: discovery_output.model,
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ResponsesModelDiscovery"
    }
}
