//! Router implementations

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::fmt::Debug;

use crate::openai_api_types::{ChatCompletionRequest, CompletionRequest, GenerateRequest};

pub mod factory;
pub mod pd_router;
pub mod pd_types;
pub mod request_adapter;
pub mod router;

pub use factory::RouterFactory;

/// Worker management trait for administrative operations
///
/// This trait is separate from RouterTrait to allow Send futures
/// for use in service discovery and other background tasks
#[async_trait]
pub trait WorkerManagement: Send + Sync {
    /// Add a worker to the router
    async fn add_worker(&self, worker_url: &str) -> Result<String, String>;

    /// Remove a worker from the router
    fn remove_worker(&self, worker_url: &str);

    /// Get all worker URLs
    fn get_worker_urls(&self) -> Vec<String>;
}

/// Core trait for all router implementations
///
/// This trait provides a unified interface for routing requests,
/// regardless of whether it's a regular router or PD router.
#[async_trait]
pub trait RouterTrait: Send + Sync + Debug + WorkerManagement {
    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Route a health check request
    async fn health(&self, req: Request<Body>) -> Response;

    /// Route a health generate request
    async fn health_generate(&self, req: Request<Body>) -> Response;

    /// Get server information
    async fn get_server_info(&self, req: Request<Body>) -> Response;

    /// Get available models
    async fn get_models(&self, req: Request<Body>) -> Response;

    /// Get model information
    async fn get_model_info(&self, req: Request<Body>) -> Response;

    /// Route a generate request
    async fn route_generate(&self, headers: Option<&HeaderMap>, body: &GenerateRequest)
        -> Response;

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
    ) -> Response;

    /// Route a completion request
    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
    ) -> Response;

    /// Flush cache on all workers
    async fn flush_cache(&self) -> Response;

    /// Get worker loads (for monitoring)
    async fn get_worker_loads(&self) -> Response;

    /// Get router type name
    fn router_type(&self) -> &'static str;

    /// Check if this is a PD router
    fn is_pd_mode(&self) -> bool {
        self.router_type() == "pd"
    }

    /// Server liveness check - is the server process running
    fn liveness(&self) -> Response {
        // Simple liveness check - if we can respond, we're alive
        (StatusCode::OK, "OK").into_response()
    }

    /// Server readiness check - is the server ready to handle requests
    fn readiness(&self) -> Response;
}
