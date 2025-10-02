use axum::Router;
use reqwest::Client;
use sglang_router_rs::{
    config::RouterConfig,
    middleware::AuthConfig,
    routers::RouterTrait,
    server::{build_app, AppContext, AppState},
};
use std::sync::Arc;

/// Create a test Axum application using the actual server's build_app function
#[allow(dead_code)]
pub fn create_test_app(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
) -> Router {
    // Create AppContext
    let app_context = Arc::new(
        AppContext::new(
            router_config.clone(),
            client,
            router_config.max_concurrent_requests,
            router_config.rate_limit_tokens_per_second,
        )
        .expect("Failed to create AppContext in test"),
    );

    // Create AppState with the test router and context
    let app_state = Arc::new(AppState {
        router,
        context: app_context,
        concurrency_queue_tx: None,
        router_manager: None,
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

    // Create auth config from router config
    let auth_config = AuthConfig {
        api_key: router_config.api_key.clone(),
    };

    // Use the actual server's build_app function
    build_app(
        app_state,
        auth_config,
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
    )
}

/// Create a test Axum application with an existing AppContext
#[allow(dead_code)]
pub fn create_test_app_with_context(
    router: Arc<dyn RouterTrait>,
    app_context: Arc<AppContext>,
) -> Router {
    // Create AppState with the test router and context
    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
    });

    // Get config from the context
    let router_config = &app_context.router_config;

    // Configure request ID headers (use defaults if not specified)
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    // Create auth config from router config
    let auth_config = AuthConfig {
        api_key: router_config.api_key.clone(),
    };

    // Use the actual server's build_app function
    build_app(
        app_state,
        auth_config,
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
    )
}
