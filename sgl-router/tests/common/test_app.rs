use axum::Router;
use reqwest::Client;
use sglang_router_rs::{
    config::RouterConfig,
    routers::RouterTrait,
    server::{build_app, AppState},
};
use std::sync::Arc;

/// Create a test Axum application using the actual server's build_app function
pub fn create_test_app(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
) -> Router {
    // Create AppState with the test router
    let app_state = Arc::new(AppState {
        router,
        client,
        _concurrency_limiter: Arc::new(tokio::sync::Semaphore::new(
            router_config.max_concurrent_requests,
        )),
    });

    // Configure request ID headers (use defaults if not specified)
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    // Use the actual server's build_app function
    build_app(
        app_state,
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
    )
}
