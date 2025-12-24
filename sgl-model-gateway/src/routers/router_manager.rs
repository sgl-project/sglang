//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    core::{ConnectionMode, WorkerRegistry, WorkerType},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
    server::ServerConfig,
};

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

pub struct RouterManager {
    worker_registry: Arc<WorkerRegistry>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    routers_snapshot: ArcSwap<Vec<Arc<dyn RouterTrait>>>,
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    enable_igw: bool,
}

impl RouterManager {
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
            routers_snapshot: ArcSwap::from_pointee(Vec::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            enable_igw: false, // Will be set properly in from_config
        }
    }

    pub async fn from_config(
        config: &ServerConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        use crate::routers::RouterFactory;

        let mut manager = Self::new(app_context.worker_registry.clone());
        manager.enable_igw = config.router_config.enable_igw;
        let manager = Arc::new(manager);

        if config.router_config.enable_igw {
            info!("Initializing RouterManager in multi-router mode (IGW)");

            match RouterFactory::create_regular_router(app_context).await {
                Ok(http_regular) => {
                    info!("Created HTTP Regular router");
                    manager.register_router(
                        RouterId::new("http-regular".to_string()),
                        Arc::from(http_regular),
                    );
                }
                Err(e) => {
                    warn!("Failed to create HTTP Regular router: {e}");
                }
            }

            // Always create gRPC Regular router in IGW mode
            match RouterFactory::create_grpc_router(app_context).await {
                Ok(grpc_regular) => {
                    info!("Created gRPC Regular router");
                    manager.register_router(
                        RouterId::new("grpc-regular".to_string()),
                        Arc::from(grpc_regular),
                    );
                }
                Err(e) => {
                    warn!("Failed to create gRPC Regular router: {e}");
                }
            }

            info!("PD disaggregation auto-enabled for IGW mode, creating PD routers");

            // Create HTTP PD router
            match RouterFactory::create_pd_router(
                None,
                None,
                &config.router_config.policy,
                app_context,
            )
            .await
            {
                Ok(http_pd) => {
                    info!("Created HTTP PD router");
                    manager
                        .register_router(RouterId::new("http-pd".to_string()), Arc::from(http_pd));
                }
                Err(e) => {
                    warn!("Failed to create HTTP PD router: {e}");
                }
            }

            // Create gRPC PD router
            match RouterFactory::create_grpc_pd_router(
                None,
                None,
                &config.router_config.policy,
                app_context,
            )
            .await
            {
                Ok(grpc_pd) => {
                    info!("Created gRPC PD router");
                    manager
                        .register_router(RouterId::new("grpc-pd".to_string()), Arc::from(grpc_pd));
                }
                Err(e) => {
                    warn!("Failed to create gRPC PD router: {e}");
                }
            }

            info!(
                "RouterManager initialized with {} routers for multi-router mode",
                manager.router_count(),
            );
        } else {
            info!("Initializing RouterManager in single-router mode");

            let single_router = Arc::from(RouterFactory::create_router(app_context).await?);
            let router_id = Self::determine_router_id(
                &config.router_config.mode,
                &config.router_config.connection_mode,
            );

            info!("Created single router with ID: {}", router_id.as_str());
            manager.register_router(router_id.clone(), single_router);
            manager.set_default_router(router_id);
        }

        if manager.router_count() == 0 {
            return Err("No routers could be initialized".to_string());
        }

        Ok(manager)
    }

    pub fn determine_router_id(
        routing_mode: &RoutingMode,
        connection_mode: &ConnectionMode,
    ) -> RouterId {
        match (connection_mode, routing_mode) {
            (ConnectionMode::Http, RoutingMode::Regular { .. }) => {
                RouterId::new("http-regular".to_string())
            }
            (ConnectionMode::Http, RoutingMode::PrefillDecode { .. }) => {
                RouterId::new("http-pd".to_string())
            }
            (ConnectionMode::Http, RoutingMode::OpenAI { .. }) => {
                RouterId::new("http-openai".to_string())
            }
            (ConnectionMode::Grpc { .. }, RoutingMode::Regular { .. }) => {
                RouterId::new("grpc-regular".to_string())
            }
            (ConnectionMode::Grpc { .. }, RoutingMode::PrefillDecode { .. }) => {
                RouterId::new("grpc-pd".to_string())
            }
            (ConnectionMode::Grpc { .. }, RoutingMode::OpenAI { .. }) => {
                RouterId::new("grpc-regular".to_string())
            }
        }
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        // Update the lock-free snapshot for fast per-request iteration
        let new_snapshot: Vec<_> = self.routers.iter().map(|e| e.value().clone()).collect();
        self.routers_snapshot.store(Arc::new(new_snapshot));

        let mut default_router = self.default_router.write().unwrap();
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self.default_router.write().unwrap();
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// Resolve model_id for a request, inferring from available workers if not specified
    /// - If model_id is provided, use it directly
    /// - If not provided and only one model exists, use it as implicit default
    /// - If not provided and multiple models exist, return error requiring specification
    /// - If no models exist, return service unavailable error
    fn resolve_model_id(&self, model_id: Option<&str>) -> Result<String, Box<Response>> {
        // If model_id is provided, use it
        if let Some(id) = model_id {
            return Ok(id.to_string());
        }

        // Get all available models from worker registry
        let available_models = self.worker_registry.get_models();

        match available_models.len() {
            0 => Err(Box::new(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No models available - no workers registered",
                )
                    .into_response(),
            )),
            1 => {
                // Single model: use it as implicit default
                debug!(
                    "Model not specified, using implicit default: {}",
                    available_models[0]
                );
                Ok(available_models[0].clone())
            }
            _ => {
                // Multiple models: require explicit model specification
                Err(Box::new(
                    (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Model must be specified. Available models: {}",
                            available_models.join(", ")
                        ),
                    )
                        .into_response(),
                ))
            }
        }
    }

    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        let workers = self.worker_registry.get_by_model(model_id);

        // Find the best worker type and derive router ID from it
        // Priority: grpc-pd (3) > http-pd (2) > grpc-regular (1) > http-regular (0)
        let best_score = workers
            .iter()
            .map(|w| {
                let is_pd = matches!(
                    w.worker_type(),
                    WorkerType::Prefill { .. } | WorkerType::Decode
                );
                let is_grpc = matches!(w.connection_mode(), ConnectionMode::Grpc { .. });

                match (is_grpc, is_pd) {
                    (true, true) => 3,   // grpc-pd (best)
                    (false, true) => 2,  // http-pd
                    (true, false) => 1,  // grpc-regular
                    (false, false) => 0, // http-regular
                }
            })
            .max();

        if let Some(score) = best_score {
            let router_id = match score {
                3 => "grpc-pd",
                2 => "http-pd",
                1 => "grpc-regular",
                _ => "http-regular",
            };

            if let Some(router) = self.routers.get(&RouterId::new(router_id.to_string())) {
                return Some(router.clone());
            }
        }

        // Fallback to default router
        let default_router = self.default_router.read().unwrap();
        if let Some(ref default_id) = *default_router {
            self.routers.get(default_id).map(|r| r.clone())
        } else {
            None
        }
    }

    pub fn select_router_for_request(
        &self,
        headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // In single-router mode (enable_igw=false), always use the default router
        if !self.enable_igw {
            let default_router = self.default_router.read().unwrap();
            if let Some(ref default_id) = *default_router {
                debug!(
                    "Single-router mode: using default router {} for model {:?}",
                    default_id.as_str(),
                    model_id
                );
                return self.routers.get(default_id).map(|r| r.clone());
            }
        }

        let prefer_pd = headers
            .and_then(|h| {
                h.get("x-prefer-pd")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s == "true" || s == "1")
            })
            .unwrap_or(false);

        let (num_regular_workers, num_pd_workers) = self.worker_registry.get_worker_distribution();
        let mut best_router = None;
        let mut best_score = -1.0;

        // Extract router validity check into a closure to reduce redundancy
        let is_router_valid =
            |is_pd: bool| (is_pd && num_pd_workers > 0) || (!is_pd && num_regular_workers > 0);

        if let Some(model) = model_id {
            // Efficient Single Lookup for Specific Model
            if let Some(router) = self.get_router_for_model(model) {
                if is_router_valid(router.is_pd_mode()) {
                    return Some(router);
                }
            }
        } else {
            // ZERO-ALLOCATION Snapshot Iteration (Hot Path Optimization)
            // Atomic load avoids heap allocations and DashMap shard locks per-request
            let routers_snapshot = self.routers_snapshot.load();
            for router in routers_snapshot.iter() {
                let mut score = 1.0;

                let is_pd = router.is_pd_mode();
                if prefer_pd && is_pd {
                    score += 2.0;
                } else if !prefer_pd && !is_pd {
                    score += 1.0;
                }
                // TODO: Once routers expose worker stats, we can evaluate:
                // - Average worker priority vs priority_threshold
                // - Average worker cost vs max_cost
                // - Current load and health status

                if score > best_score && is_router_valid(is_pd) {
                    best_score = score;
                    best_router = Some(Arc::clone(router));
                }
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

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // IGW readiness: return 200 if at least one router has healthy workers
        let has_healthy_workers = self
            .worker_registry
            .get_all()
            .iter()
            .any(|w| w.is_healthy());

        if has_healthy_workers {
            (StatusCode::OK, "At least one router has healthy workers").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No routers with healthy workers available",
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // TODO: Aggregate info from all routers with healthy workers
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

    async fn get_models(&self, _req: Request<Body>) -> Response {
        let model_names = self.worker_registry.get_models();

        if model_names.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            // Convert model names to OpenAI-compatible model objects
            let models: Vec<Value> = model_names
                .iter()
                .map(|name| {
                    serde_json::json!({
                        "id": name,
                        "object": "model",
                        "owned_by": "local"
                    })
                })
                .collect();

            (
                StatusCode::OK,
                serde_json::json!({
                    "object": "list",
                    "data": models
                })
                .to_string(),
            )
                .into_response()
        }
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Route to default router or first available router
        let router_id = {
            let default_router = self.default_router.read().unwrap();
            default_router.clone()
        };

        let router = if let Some(id) = router_id {
            self.routers.get(&id).map(|r| r.clone())
        } else {
            // If no default, use first available router
            self.routers.iter().next().map(|r| r.value().clone())
        };

        if let Some(router) = router {
            router.get_model_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Resolve model_id intelligently instead of falling back to "unknown"
        let resolved_model_id = match self.resolve_model_id(model_id) {
            Ok(id) => id,
            Err(err_response) => return *err_response,
        };

        let router = self.select_router_for_request(headers, Some(&resolved_model_id));

        if let Some(router) = router {
            router
                .route_generate(headers, body, Some(&resolved_model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_chat(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_completion(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let selected_model = model_id.or(Some(body.model.as_str()));
        let router = self.select_router_for_request(headers, selected_model);

        if let Some(router) = router {
            router.route_responses(headers, body, selected_model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle responses request",
            )
                .into_response()
        }
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        params: &ResponsesGetParams,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_response(headers, response_id, params).await
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

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        // Delegate to the default router (typically http-regular)
        // Response storage is shared across all routers via AppContext
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.list_response_input_items(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to list response input items",
            )
                .into_response()
        }
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_embeddings(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_classify(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

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

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*self.default_router.read().unwrap())
            .finish()
    }
}
