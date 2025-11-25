//! Model Discovery stage
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

use super::PipelineStage;
use crate::routers::openai::{
    context::{CachedEndpoint, DiscoveryOutput, RequestContext, RequestType},
    utils::probe_endpoint_for_model,
};

/// Model discovery stage
pub struct ModelDiscoveryStage {
    worker_urls: Vec<String>,
}

impl ModelDiscoveryStage {
    pub fn new(worker_urls: Vec<String>) -> Self {
        Self { worker_urls }
    }

    /// Model discovery cache TTL (1 hour)
    const MODEL_CACHE_TTL_SECS: u64 = 3600;
}

#[async_trait]
impl PipelineStage for ModelDiscoveryStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Get validation output
        let validation = ctx.state.validation.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Validation stage not completed",
            )
                .into_response()
        })?;

        // Get model name
        let model = match &ctx.input.request_type {
            RequestType::Chat(req) => &req.model,
            RequestType::Responses(req) => ctx.input.model_id.as_deref().unwrap_or(&req.model),
        };

        // Fast path: single endpoint
        if self.worker_urls.len() == 1 {
            ctx.state.discovery = Some(DiscoveryOutput {
                endpoint_url: self.worker_urls[0].clone(),
                model: model.to_string(),
            });
            return Ok(None);
        }

        // Check cache
        if let Some(entry) = ctx.components.model_cache.get(model) {
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

        for url in &self.worker_urls {
            let handle = tokio::spawn(probe_endpoint_for_model(
                ctx.components.http_client.clone(),
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
                ctx.components.model_cache.insert(
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
        "ModelDiscovery"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dashmap::DashMap;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        routers::openai::context::{SharedComponents, ValidationOutput},
    };

    async fn create_test_components(worker_urls: Vec<String>) -> Arc<SharedComponents> {
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

        Arc::new(SharedComponents {
            http_client: client,
            circuit_breaker,
            model_cache,
            mcp_manager,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            worker_urls,
        })
    }

    #[tokio::test]
    async fn test_discovery_stage_single_endpoint() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls.clone()).await;
        let stage = ModelDiscoveryStage::new(worker_urls);

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Set validation output (prerequisite)
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Verify discovery output
        let discovery = ctx.state.discovery.as_ref().unwrap();
        assert_eq!(discovery.endpoint_url, "http://localhost:8000");
        assert_eq!(discovery.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_discovery_stage_validation_not_completed() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls.clone()).await;
        let stage = ModelDiscoveryStage::new(worker_urls);

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Don't set validation output - should error
        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());

        let response = result.unwrap_err();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_discovery_stage_cache_hit() {
        let worker_urls = vec![
            "http://localhost:8000".to_string(),
            "http://localhost:8001".to_string(),
        ];
        let components = create_test_components(worker_urls.clone()).await;

        // Pre-populate cache
        components.model_cache.insert(
            "gpt-4".to_string(),
            CachedEndpoint {
                url: "http://localhost:8001".to_string(),
                cached_at: Instant::now(),
            },
        );

        let stage = ModelDiscoveryStage::new(worker_urls);

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Set validation output
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());

        // Verify cache hit - should use cached URL
        let discovery = ctx.state.discovery.as_ref().unwrap();
        assert_eq!(discovery.endpoint_url, "http://localhost:8001");
    }
}
