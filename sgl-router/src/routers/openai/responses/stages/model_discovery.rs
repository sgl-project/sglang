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

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use dashmap::DashMap;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::responses::{ResponseInput, ResponsesRequest},
        routers::openai::responses::{ResponsesDependencies, ValidationOutput},
    };

    async fn create_test_dependencies(worker_urls: Vec<String>) -> Arc<ResponsesDependencies> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());

        let mcp_config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: vec![],
            inventory: Default::default(),
        };
        let mcp_manager = Arc::new(
            McpManager::new(mcp_config, 10)
                .await
                .expect("Failed to create MCP manager"),
        );

        let response_storage = Arc::new(MemoryResponseStorage::new());
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

        Arc::new(ResponsesDependencies {
            http_client: client,
            circuit_breaker,
            model_cache,
            worker_urls,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
        })
    }

    #[tokio::test]
    async fn test_discovery_stage_single_endpoint() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies(vec!["http://localhost:8000".to_string()]).await;
        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        // Set validation output
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });

        let stage = ResponsesModelDiscoveryStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert!(ctx.state.discovery.is_some());

        let discovery = ctx.state.discovery.unwrap();
        assert_eq!(discovery.endpoint_url, "http://localhost:8000");
        assert_eq!(discovery.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_discovery_stage_no_validation() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies(vec!["http://localhost:8000".to_string()]).await;
        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        // Don't set validation output
        let stage = ResponsesModelDiscoveryStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_discovery_stage_cache_hit() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies(vec![
            "http://localhost:8000".to_string(),
            "http://localhost:8001".to_string(),
        ]).await;

        // Pre-populate cache
        dependencies.model_cache.insert(
            "gpt-4".to_string(),
            CachedEndpoint {
                url: "http://localhost:8000".to_string(),
                cached_at: Instant::now(),
            },
        );

        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });

        let stage = ResponsesModelDiscoveryStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(ctx.state.discovery.is_some());

        let discovery = ctx.state.discovery.unwrap();
        assert_eq!(discovery.endpoint_url, "http://localhost:8000");
    }
}
