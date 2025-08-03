use axum::Router;
use reqwest::Client;
use sglang_router_rs::{
    config::RouterConfig,
    routers::RouterTrait,
    server::{build_app, AppContext, AppState},
};
use std::sync::Arc;

/// Create a test Axum application using the actual server's build_app function
pub fn create_test_app(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
) -> Router {
    // Create AppContext
    let app_context = Arc::new(AppContext::new(
        router_config.clone(),
        client,
        router_config.max_concurrent_requests,
    ));

    // Create AppState with the test router and context
    let app_state = Arc::new(AppState {
        router,
        context: app_context,
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
