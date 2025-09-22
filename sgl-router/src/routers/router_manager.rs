//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use crate::core::{Worker, WorkerRegistry, WorkerType};
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, RerankRequest,
    ResponsesRequest,
};
use crate::routers::RouterTrait;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use std::sync::Arc;
use tracing::info;

/// Router identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(String);

impl RouterId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Router Manager - Central coordinator for routers and workers
pub struct RouterManager {
    /// Worker registry (single source of truth in multi-router mode)
    worker_registry: Arc<WorkerRegistry>,

    /// All routers managed by this manager
    /// RouterId examples: "http-regular", "http-pd", "grpc-regular", "grpc-pd"
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,

    /// Default router for requests without specific routing
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
}

impl RouterManager {
    /// Create a new router manager with shared registries
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
        }
    }

    /// Register a router with the manager
    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        let mut default_router = self.default_router.write().unwrap();
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    /// Set the default router
    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self.default_router.write().unwrap();
        *default_router = Some(id);
    }

    /// Get the number of registered routers
    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// Get router for a specific model based on worker types
    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        let workers = self.worker_registry.get_by_model(model_id);

        if !workers.is_empty() {
            let has_pd_workers = workers.iter().any(|w| {
                matches!(
                    w.worker_type(),
                    WorkerType::Prefill { .. } | WorkerType::Decode
                )
            });

            let router_id = if has_pd_workers {
                RouterId::new("http-pd".to_string())
            } else {
                RouterId::new("http-regular".to_string())
            };

            if let Some(router) = self.routers.get(&router_id) {
                return Some(router.clone());
            }
        }

        let default_router = self.default_router.read().unwrap();
        if let Some(ref default_id) = *default_router {
            self.routers.get(default_id).map(|r| r.clone())
        } else {
            None
        }
    }

    /// Get workers for routing decision
    pub fn get_workers_for_request(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        if let Some(model) = model_id {
            self.worker_registry.get_by_model(model)
        } else {
            self.worker_registry.get_all()
        }
    }

    /// Get the appropriate router for a request based on headers and request content
    pub fn select_router_for_request(
        &self,
        headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        let _priority_threshold = headers.and_then(|h| {
            h.get("x-worker-priority")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u32>().ok())
        });

        let _max_cost = headers.and_then(|h| {
            h.get("x-max-cost")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<f32>().ok())
        });

        let prefer_pd = headers
            .and_then(|h| {
                h.get("x-prefer-pd")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s == "true" || s == "1")
            })
            .unwrap_or(false);

        let candidate_routers = if let Some(model) = model_id {
            if let Some(router) = self.get_router_for_model(model) {
                vec![router]
            } else {
                Vec::new()
            }
        } else {
            self.routers
                .iter()
                .map(|entry| entry.value().clone())
                .collect::<Vec<_>>()
        };

        if candidate_routers.is_empty() {
            return None;
        }

        let mut best_router = None;
        let mut best_score = 0.0;

        for router in candidate_routers {
            let mut score = 1.0;

            let is_pd = router.is_pd_mode();
            if prefer_pd && is_pd {
                score += 2.0;
            } else if !prefer_pd && !is_pd {
                score += 1.0;
            }

            // Get workers for this router and evaluate based on priority/cost
            // Note: This would require routers to expose their workers or stats
            // For now, we'll use a simple selection based on router type

            // TODO: Once routers expose worker stats, we can evaluate:
            // - Average worker priority vs priority_threshold
            // - Average worker cost vs max_cost
            // - Current load and health status

            if score > best_score {
                best_score = score;
                best_router = Some(router);
            }
        }

        best_router
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Health check - return 503 if no routers available
    async fn health(&self, _req: Request<Body>) -> Response {
        // Health check should succeed if RouterManager exists, even without routers
        // Individual router health can be checked via specific endpoints
        (StatusCode::OK, "RouterManager is healthy").into_response()
    }

    /// Health generate - check if any router can handle generate requests
    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Return 503 since we have no routers with workers
        // TODO: Should check if any router has healthy workers
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "No routers with healthy workers available",
        )
            .into_response()
    }

    /// Get server information - aggregate from all routers
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // TODO: Aggregate info from all routers with healthy workers
        // For now, return basic info about the RouterManager
        (
            StatusCode::OK,
            serde_json::json!({
                "router_manager": true,
                "routers_count": self.routers.len(),
                "workers_count": self.worker_registry.get_all().len()
            })
            .to_string(),
        )
            .into_response()
    }

    /// Get available models - query from worker registry
    async fn get_models(&self, _req: Request<Body>) -> Response {
        // Get models from worker registry
        let models = self.worker_registry.get_models();

        if models.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            (
                StatusCode::OK,
                serde_json::json!({
                    "models": models
                })
                .to_string(),
            )
                .into_response()
        }
    }

    /// Get model information
    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        // TODO: Extract model from request and route to appropriate router
        // For now, return not implemented
        (
            StatusCode::NOT_IMPLEMENTED,
            "Model info endpoint not yet implemented in RouterManager",
        )
            .into_response()
    }

    /// Route a generate request
    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Select router based on headers
        // GenerateRequest doesn't have a model field
        let router = self.select_router_for_request(headers, None);

        if let Some(router) = router {
            // In multi-model mode, pass None since GenerateRequest doesn't have model field
            router.route_generate(headers, body, None).await
        } else {
            // Return 404 when no router is available for the request
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(&body.model));

        if let Some(router) = router {
            // In multi-model mode, pass the model_id to the router
            router.route_chat(headers, body, Some(&body.model)).await
        } else {
            // Return 404 when the specified model is not found
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    /// Route a completion request
    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(&body.model));

        if let Some(router) = router {
            // In multi-model mode, pass the model_id to the router
            router
                .route_completion(headers, body, Some(&body.model))
                .await
        } else {
            // Return 404 when the specified model is not found
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn get_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to get response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{}'", response_id),
            )
                .into_response()
        }
    }

    /// Route embeddings request
    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(&body.model));

        if let Some(router) = router {
            router
                .route_embeddings(headers, body, Some(&body.model))
                .await
        } else {
            // Return 404 when the specified model is not found
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    /// Route rerank request
    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Try to select a router based on headers
        let router = self.select_router_for_request(headers, None);

        if let Some(router) = router {
            router.route_rerank(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for rerank request",
            )
                .into_response()
        }
    }

    /// Flush cache on all routers and workers
    async fn flush_cache(&self) -> Response {
        // TODO: Call flush_cache on all routers that have workers
        // For now, return success if we have any routers
        if self.routers.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers configured").into_response()
        } else {
            // TODO: Actually flush cache on all routers
            (StatusCode::OK, "Cache flush requested").into_response()
        }
    }

    /// Get worker loads from all routers
    async fn get_worker_loads(&self) -> Response {
        // Return worker loads from the registry
        let workers = self.worker_registry.get_all();
        let loads: Vec<serde_json::Value> = workers
            .iter()
            .map(|w| {
                serde_json::json!({
                    "url": w.url(),
                    "model": w.model_id(),
                    "load": w.load(),
                    "is_healthy": w.is_healthy()
                })
            })
            .collect();

        (
            StatusCode::OK,
            serde_json::json!({
                "workers": loads
            })
            .to_string(),
        )
            .into_response()
    }

    /// Get router type name
    fn router_type(&self) -> &'static str {
        "manager"
    }

    /// Server readiness check - check if any router is ready
    fn readiness(&self) -> Response {
        if self.routers.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers configured").into_response()
        } else {
            // TODO: Check readiness of all routers
            (StatusCode::OK, "Ready").into_response()
        }
    }
}

// Note: get_first_available_router removed - we now properly handle
// router selection based on model and worker availability

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*self.default_router.read().unwrap())
            .finish()
    }
}
