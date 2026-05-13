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
use bytes::Bytes;
use dashmap::DashMap;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    core::{ConnectionMode, RuntimeType, WorkerRegistry, WorkerType},
    protocols::{
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{error, RouterTrait},
    server::ServerConfig,
};

/// Read the `model` field out of a parsed body for IGW dispatch.
/// Distinguishes four input shapes that previously all collapsed
/// to "no model":
/// - `Ok(Some(s))` — explicit non-empty string model id;
/// - `Ok(None)` — field absent, or present as JSON `null`;
/// - `Err(_)` for `model: ""` — empty string is rejected explicitly
///   so we fail fast with a structured 400 instead of routing to a
///   model id no worker can match;
/// - `Err(_)` for non-string types like `model: 42` — same reasoning.
fn extract_body_model(value: &Value) -> Result<Option<&str>, Response> {
    match value.get("model") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.is_empty() => Ok(Some(s.as_str())),
        Some(Value::String(_)) => Err(error::bad_request(
            "json_parse_error",
            "Field `model` must be a non-empty string",
        )),
        Some(_) => Err(error::bad_request(
            "json_parse_error",
            "Field `model` must be a string",
        )),
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(&'static str);

impl RouterId {
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        self.0
    }
}

/// Static router ID constants to avoid heap allocations in hot paths
pub mod router_ids {
    use super::RouterId;

    pub const HTTP_REGULAR: RouterId = RouterId::new("http-regular");
    pub const HTTP_PD: RouterId = RouterId::new("http-pd");
    pub const HTTP_OPENAI: RouterId = RouterId::new("http-openai");
    pub const GRPC_REGULAR: RouterId = RouterId::new("grpc-regular");
    pub const GRPC_PD: RouterId = RouterId::new("grpc-pd");
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
                    manager.register_router(router_ids::HTTP_REGULAR, Arc::from(http_regular));
                }
                Err(e) => {
                    warn!("Failed to create HTTP Regular router: {e}");
                }
            }

            // Always create gRPC Regular router in IGW mode
            match RouterFactory::create_grpc_router(app_context).await {
                Ok(grpc_regular) => {
                    info!("Created gRPC Regular router");
                    manager.register_router(router_ids::GRPC_REGULAR, Arc::from(grpc_regular));
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
                    manager.register_router(router_ids::HTTP_PD, Arc::from(http_pd));
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
                    manager.register_router(router_ids::GRPC_PD, Arc::from(grpc_pd));
                }
                Err(e) => {
                    warn!("Failed to create gRPC PD router: {e}");
                }
            }

            // Create OpenAI router for external OpenAI-compatible backends
            match RouterFactory::create_openai_router(app_context).await {
                Ok(openai) => {
                    info!("Created OpenAI router");
                    manager.register_router(router_ids::HTTP_OPENAI, Arc::from(openai));
                }
                Err(e) => {
                    warn!("Failed to create OpenAI router: {e}");
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
            (ConnectionMode::Http, RoutingMode::Regular { .. }) => router_ids::HTTP_REGULAR,
            (ConnectionMode::Http, RoutingMode::PrefillDecode { .. }) => router_ids::HTTP_PD,
            (ConnectionMode::Http, RoutingMode::OpenAI { .. }) => router_ids::HTTP_OPENAI,
            (ConnectionMode::Grpc { .. }, RoutingMode::Regular { .. }) => router_ids::GRPC_REGULAR,
            (ConnectionMode::Grpc { .. }, RoutingMode::PrefillDecode { .. }) => router_ids::GRPC_PD,
            (ConnectionMode::Grpc { .. }, RoutingMode::OpenAI { .. }) => router_ids::GRPC_REGULAR,
        }
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        // Update the lock-free snapshot for fast per-request iteration
        let new_snapshot: Vec<_> = self.routers.iter().map(|e| e.value().clone()).collect();
        self.routers_snapshot.store(Arc::new(new_snapshot));

        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// Resolve model_id for a request, inferring from available workers if not specified.
    ///
    /// Behavior in IGW mode (must fail fast if model not resolvable):
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
            0 => Err(Box::new(error::service_unavailable(
                "no_models_available",
                "No models available - no workers registered",
            ))),
            1 => {
                // Single model: use it as implicit default
                debug!(
                    "Model not specified, using implicit default: {}",
                    available_models[0]
                );
                Ok(available_models[0].clone())
            }
            _ => Err(Box::new(error::bad_request(
                "model_required",
                format!(
                    "Model must be specified. Available models: {}",
                    available_models.join(", ")
                ),
            ))),
        }
    }

    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        let workers = self.worker_registry.get_by_model(model_id);

        // Find the best router ID based on worker capabilities
        // Priority: external (OpenAI) > grpc-pd > http-pd > grpc-regular > http-regular
        let best_router_id = workers
            .iter()
            .map(|w| {
                let is_pd = matches!(
                    w.worker_type(),
                    WorkerType::Prefill { .. } | WorkerType::Decode
                );
                let is_grpc = matches!(w.connection_mode(), ConnectionMode::Grpc { .. });
                let is_external = matches!(w.metadata().runtime_type, RuntimeType::External);

                if is_external {
                    // External workers should be routed via OpenAI-compatible router
                    return (4, &router_ids::HTTP_OPENAI);
                }

                match (is_grpc, is_pd) {
                    (true, true) => (3, &router_ids::GRPC_PD),
                    (false, true) => (2, &router_ids::HTTP_PD),
                    (true, false) => (1, &router_ids::GRPC_REGULAR),
                    (false, false) => (0, &router_ids::HTTP_REGULAR),
                }
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, id)| id);

        if let Some(router_id) = best_router_id {
            if let Some(router) = self.routers.get(router_id) {
                return Some(router.clone());
            }
        }

        // Fallback to default router
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
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
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
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
            error::service_unavailable(
                "no_healthy_workers",
                "No routers with healthy workers available",
            )
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
            error::service_unavailable("no_models_available", "No models available")
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
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
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
            error::service_unavailable("no_routers_available", "No routers available")
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &Bytes,
        model_id: Option<&str>,
    ) -> Response {
        // Non-IGW fast path: hand body+model_id straight to the default
        // router and skip parsing entirely.
        if !self.enable_igw {
            let Some(router) = self.select_router_for_request(headers, model_id) else {
                return error::not_found(
                    "no_router_available",
                    "No router available for this request",
                );
            };
            return router.route_generate(headers, body, model_id).await;
        }

        // IGW: parse `Value` so a malformed body fails fast with a
        // structured error here, rather than being silently routed to
        // the implicit-default model and only rejected later by the
        // chosen router (which would pollute metrics/labels with a
        // model the client never asked for).
        let value: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return error::bad_request("json_parse_error", format!("Invalid JSON data: {e}"));
            }
        };
        let body_model = match extract_body_model(&value) {
            Ok(m) => m,
            Err(resp) => return resp,
        };
        let effective_model_id = match self.resolve_model_id(body_model) {
            Ok(id) => id,
            Err(err_response) => return *err_response,
        };

        let Some(router) = self.select_router_for_request(headers, Some(&effective_model_id))
        else {
            return error::not_found(
                "no_router_available",
                "No router available for this request",
            );
        };

        router
            .route_generate(headers, body, Some(&effective_model_id))
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &Bytes,
        model_id: Option<&str>,
    ) -> Response {
        if !self.enable_igw {
            let Some(router) = self.select_router_for_request(headers, model_id) else {
                return error::not_found(
                    "no_router_available",
                    "No router available for this request",
                );
            };
            return router.route_chat(headers, body, model_id).await;
        }

        // IGW: same shape as generate — hard-fail on malformed JSON
        // here so it can't masquerade as "no model field" and reach
        // an unrelated worker. Chat clients normally supply `model`;
        // when they don't and a single model is registered the IGW
        // resolver still treats it as the implicit default.
        let value: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return error::bad_request("json_parse_error", format!("Invalid JSON data: {e}"));
            }
        };
        let body_model = match extract_body_model(&value) {
            Ok(m) => m,
            Err(resp) => return resp,
        };
        let effective_model_id = match self.resolve_model_id(body_model) {
            Ok(id) => id,
            Err(err_response) => return *err_response,
        };

        let Some(router) = self.select_router_for_request(headers, Some(&effective_model_id))
        else {
            return error::not_found(
                "no_router_available",
                format!(
                    "Model '{}' not found or no router available",
                    effective_model_id
                ),
            );
        };

        router
            .route_chat(headers, body, Some(&effective_model_id))
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // In IGW mode, resolve model_id and fail fast if not resolvable
        // In non-IGW mode, pass through to router (router handles validation)
        let effective_model_id = if self.enable_igw {
            // Use provided model_id or fall back to body.model
            let model = model_id.or(Some(&body.model));
            match self.resolve_model_id(model) {
                Ok(id) => Some(id),
                Err(err_response) => return *err_response,
            }
        } else {
            None
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_completion(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            error::not_found(
                "no_router_available",
                format!("Model '{}' not found or no router available", body.model),
            )
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
            error::not_found(
                "no_router_available",
                "No router available to handle responses request",
            )
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
            error::not_found(
                "no_router_available",
                format!("No router available to get response '{}'", response_id),
            )
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            error::not_found(
                "no_router_available",
                format!("No router available to cancel response '{}'", response_id),
            )
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
            error::not_found(
                "no_router_available",
                "No router available to list response input items",
            )
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
            error::not_found(
                "no_router_available",
                format!("Model '{}' not found or no router available", body.model),
            )
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
            error::not_found(
                "no_router_available",
                format!("Model '{}' not found or no router available", body.model),
            )
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
            error::not_found(
                "no_router_available",
                "No router available for rerank request",
            )
        }
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*default_router)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex};

    use super::*;
    use crate::core::{BasicWorkerBuilder, Worker};

    /// Mock router that records the `model_id` argument it receives,
    /// so tests can assert what `RouterManager` plumbed through.
    #[derive(Debug, Default)]
    struct CaptureRouter {
        seen_generate_model: Mutex<Vec<Option<String>>>,
    }

    #[async_trait]
    impl RouterTrait for CaptureRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn route_generate(
            &self,
            _headers: Option<&HeaderMap>,
            _body: &Bytes,
            model_id: Option<&str>,
        ) -> Response {
            self.seen_generate_model
                .lock()
                .unwrap()
                .push(model_id.map(str::to_string));
            (StatusCode::OK, "captured").into_response()
        }

        async fn route_chat(
            &self,
            _headers: Option<&HeaderMap>,
            _body: &Bytes,
            _model_id: Option<&str>,
        ) -> Response {
            (StatusCode::OK, "captured").into_response()
        }

        fn router_type(&self) -> &'static str {
            "capture-mock"
        }
    }

    #[test]
    fn extract_body_model_empty_string_is_400() {
        let body = serde_json::json!({"model": ""});
        let err = extract_body_model(&body).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        // Lock the structured-error contract: clients (and our own
        // metrics labels) classify on the `X-SMG-Error-Code` header,
        // so a refactor that drops the code must not slip through.
        assert_eq!(
            error::extract_error_code_from_response(&err),
            "json_parse_error"
        );
    }

    #[test]
    fn extract_body_model_accepts_missing_and_null() {
        assert!(matches!(
            extract_body_model(&serde_json::json!({})),
            Ok(None)
        ));
        assert!(matches!(
            extract_body_model(&serde_json::json!({"model": null})),
            Ok(None)
        ));
    }

    #[test]
    fn extract_body_model_accepts_nonempty_string() {
        let body = serde_json::json!({"model": "m1"});
        assert_eq!(extract_body_model(&body).unwrap(), Some("m1"));
    }

    /// IGW (k8s-mode) implicit-default-model contract: when service
    /// discovery has registered exactly one model and the client
    /// omits `"model"`, the manager must resolve it via
    /// `resolve_model_id` and forward the resolved id to the chosen
    /// router as `model_id`. Without this, downstream metrics labels,
    /// worker selection, and cache-aware policies all lose the
    /// model dimension.
    #[tokio::test]
    async fn igw_route_generate_propagates_resolved_model() {
        let registry = Arc::new(WorkerRegistry::new());
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "test-model-a".to_string());
        let worker: Arc<dyn Worker> = Arc::from(
            BasicWorkerBuilder::new("http://capture-worker:9999")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .build(),
        );
        registry.register(worker);

        let mut manager = RouterManager::new(registry);
        manager.enable_igw = true;
        let manager = Arc::new(manager);

        let capture = Arc::new(CaptureRouter::default());
        manager.register_router(router_ids::HTTP_REGULAR, capture.clone());

        // Client omits "model" in the body — IGW must resolve the
        // single registered worker's model id and pass it down.
        let body = Bytes::from(r#"{"text":"hi"}"#);
        let response = manager.route_generate(None, &body, None).await;
        assert_eq!(response.status(), StatusCode::OK);

        let seen = capture.seen_generate_model.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![Some("test-model-a".to_string())],
            "downstream router must see the IGW-resolved model id, not None"
        );
    }
}
