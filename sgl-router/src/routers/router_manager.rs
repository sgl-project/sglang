//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use std::sync::Arc;

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
    server::{AppContext, ServerConfig},
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
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    enable_igw: bool,
}

impl RouterManager {
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
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

            // TODO: Add gRPC routers once we have dynamic tokenizer loading

            info!(
                "RouterManager initialized with {} routers for multi-router mode",
                manager.router_count()
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

        // Multi-router mode logic follows
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

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // TODO: Should check if any router has healthy workers
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "No routers with healthy workers available",
        )
            .into_response()
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
        let models = self.worker_registry.get_models();

        if models.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            (
                StatusCode::OK,
                serde_json::json!({ "models": models }).to_string(),
            )
                .into_response()
        }
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        // TODO: Extract model from request and route to appropriate router
        (
            StatusCode::NOT_IMPLEMENTED,
            "Model info endpoint not yet implemented in RouterManager",
        )
            .into_response()
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);

        if let Some(router) = router {
            router.route_generate(headers, body, None).await
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
        _model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(&body.model));

        if let Some(router) = router {
            router.route_chat(headers, body, Some(&body.model)).await
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
        _model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(&body.model));

        if let Some(router) = router {
            router
                .route_completion(headers, body, Some(&body.model))
                .await
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

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, Some(&body.model));

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

    fn router_type(&self) -> &'static str {
        "manager"
    }

    // Conversations API delegates
    async fn create_conversation(&self, headers: Option<&HeaderMap>, body: &Value) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.create_conversation(headers, body).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to create conversation",
            )
                .into_response()
        }
    }

    async fn get_conversation(
        &self,
        headers: Option<&HeaderMap>,
        conversation_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_conversation(headers, conversation_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!(
                    "No router available to get conversation '{}'",
                    conversation_id
                ),
            )
                .into_response()
        }
    }

    async fn update_conversation(
        &self,
        headers: Option<&HeaderMap>,
        conversation_id: &str,
        body: &Value,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router
                .update_conversation(headers, conversation_id, body)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!(
                    "No router available to update conversation '{}'",
                    conversation_id
                ),
            )
                .into_response()
        }
    }

    async fn delete_conversation(
        &self,
        headers: Option<&HeaderMap>,
        conversation_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.delete_conversation(headers, conversation_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!(
                    "No router available to delete conversation '{}'",
                    conversation_id
                ),
            )
                .into_response()
        }
    }

    async fn list_conversation_items(
        &self,
        headers: Option<&HeaderMap>,
        conversation_id: &str,
        limit: Option<usize>,
        order: Option<String>,
        after: Option<String>,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router
                .list_conversation_items(headers, conversation_id, limit, order, after)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!(
                    "No router available to list conversation items for '{}'",
                    conversation_id
                ),
            )
                .into_response()
        }
    }

    async fn create_conversation_items(
        &self,
        headers: Option<&HeaderMap>,
        conversation_id: &str,
        body: &Value,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router
                .create_conversation_items(headers, conversation_id, body)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!(
                    "No router available to create conversation items for '{}'",
                    conversation_id
                ),
            )
                .into_response()
        }
    }

    async fn get_conversation_item(
        &self,
        headers: Option<&HeaderMap>,
        conversation_id: &str,
        item_id: &str,
        include: Option<Vec<String>>,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router
                .get_conversation_item(headers, conversation_id, item_id, include)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!(
                    "No router available to get conversation item '{}' in '{}'",
                    item_id, conversation_id
                ),
            )
                .into_response()
        }
    }

    async fn delete_conversation_item(
        &self,
        headers: Option<&HeaderMap>,
        conversation_id: &str,
        item_id: &str,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router
                .delete_conversation_item(headers, conversation_id, item_id)
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!(
                    "No router available to delete conversation item '{}' in '{}'",
                    item_id, conversation_id
                ),
            )
                .into_response()
        }
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
