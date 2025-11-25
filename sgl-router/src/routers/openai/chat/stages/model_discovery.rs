//! Model Discovery stage for chat pipeline
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

use super::ChatStage;
use crate::routers::openai::{
    chat::{ChatRequestContext, DiscoveryOutput},
    router::CachedEndpoint,
    utils::probe_endpoint_for_model,
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

        // Get model name
        let model = ctx.model();

        // Fast path: single endpoint
        if ctx.dependencies.worker_urls.len() == 1 {
            ctx.state.discovery = Some(DiscoveryOutput {
                endpoint_url: ctx.dependencies.worker_urls[0].clone(),
                model: model.to_string(),
            });
            return Ok(());
        }

        // Check cache
        if let Some(entry) = ctx.dependencies.model_cache.get(model) {
            if entry.cached_at.elapsed() < Duration::from_secs(Self::MODEL_CACHE_TTL_SECS) {
                ctx.state.discovery = Some(DiscoveryOutput {
                    endpoint_url: entry.url.clone(),
                    model: model.to_string(),
                });
                return Ok(());
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
                return Ok(());
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
        "ChatModelDiscovery"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dashmap::DashMap;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        routers::openai::chat::{ChatDependencies, ValidationOutput},
    };

    fn create_test_dependencies(worker_urls: Vec<String>) -> Arc<ChatDependencies> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());

        Arc::new(ChatDependencies {
            http_client: client,
            circuit_breaker,
            model_cache,
            worker_urls,
        })
    }

    #[tokio::test]
    async fn test_discovery_stage_single_endpoint() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let dependencies = create_test_dependencies(vec!["http://localhost:8000".to_string()]);
        let mut ctx = ChatRequestContext::new(
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

        let stage = ChatModelDiscoveryStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(ctx.state.discovery.is_some());

        let discovery = ctx.state.discovery.unwrap();
        assert_eq!(discovery.endpoint_url, "http://localhost:8000");
        assert_eq!(discovery.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_discovery_stage_no_validation() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        };

        let dependencies = create_test_dependencies(vec!["http://localhost:8000".to_string()]);
        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        // Don't set validation output
        let stage = ChatModelDiscoveryStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_discovery_stage_cache_hit() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        };

        let dependencies = create_test_dependencies(vec![
            "http://localhost:8000".to_string(),
            "http://localhost:8001".to_string(),
        ]);

        // Pre-populate cache
        dependencies.model_cache.insert(
            "gpt-4".to_string(),
            CachedEndpoint {
                url: "http://localhost:8000".to_string(),
                cached_at: Instant::now(),
            },
        );

        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });

        let stage = ChatModelDiscoveryStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(ctx.state.discovery.is_some());

        let discovery = ctx.state.discovery.unwrap();
        assert_eq!(discovery.endpoint_url, "http://localhost:8000");
    }
}
