//! Model Discovery stage for responses pipeline
//!
//! This stage:
//! - Determines which endpoint has the requested model
//! - Uses caching to avoid repeated endpoint probing
//! - Handles single endpoint (fast path) vs multiple endpoints

use std::time::{Duration, Instant};

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::ResponsesStage;
use crate::routers::openai::{
    responses::{DiscoveryOutput, ResponsesRequestContext},
    router::CachedEndpoint,
    utils::probe_endpoint_for_model,
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

        // Get model name
        let model = ctx.model();

        // Fast path: single endpoint
        if ctx.dependencies.worker_urls.len() == 1 {
            ctx.state.discovery = Some(DiscoveryOutput {
                endpoint_url: ctx.dependencies.worker_urls[0].clone(),
                model: model.to_string(),
            });
            return Ok(None);
        }

        // Check cache
        if let Some(entry) = ctx.dependencies.model_cache.get(model) {
            if entry.cached_at.elapsed() < Duration::from_secs(Self::MODEL_CACHE_TTL_SECS) {
                ctx.state.discovery = Some(DiscoveryOutput {
                    endpoint_url: entry.url.clone(),
                    model: model.to_string(),
                });
                return Ok(None);
            }
        }

        // Probe all endpoints in parallel
        let mut handles = vec![];
        let model_str = model.to_string();
        let auth = validation.auth_header.clone();

        for url in &ctx.dependencies.worker_urls {
            let handle = tokio::spawn(probe_endpoint_for_model(
                ctx.dependencies.http_client.clone(),
                url.clone(),
                model_str.clone(),
                auth.clone(),
            ));
            handles.push(handle);
        }

        // Return first successful endpoint
        for handle in handles {
            if let Ok(Ok(url)) = handle.await {
                // Cache it
                ctx.dependencies.model_cache.insert(
                    model_str.clone(),
                    CachedEndpoint {
                        url: url.clone(),
                        cached_at: Instant::now(),
                    },
                );

                ctx.state.discovery = Some(DiscoveryOutput {
                    endpoint_url: url,
                    model: model_str,
                });
                return Ok(None);
            }
        }

        // Model not found on any endpoint
        Err((
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found on any endpoint", model),
        )
            .into_response())
    }

    fn name(&self) -> &'static str {
        "ResponsesModelDiscovery"
    }
}
