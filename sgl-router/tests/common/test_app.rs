use std::sync::{Arc, OnceLock};

use axum::Router;
use reqwest::Client;
use sglang_router_rs::{
    config::RouterConfig,
    core::{LoadMonitor, WorkerRegistry},
    data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    },
    middleware::{AuthConfig, TokenBucket},
    policies::PolicyRegistry,
    routers::RouterTrait,
    server::{build_app, AppContext, AppState},
};

/// Create a test Axum application using the actual server's build_app function
#[allow(dead_code)]
pub fn create_test_app(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
) -> Router {
    // Initialize rate limiter
    let rate_limiter = match router_config.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = router_config
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    // Initialize registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

    // Initialize storage backends
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    // Initialize load monitor
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        router_config.worker_startup_check_interval_secs,
    )));

    // Create empty OnceLock for worker job queue and workflow engine
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engine = Arc::new(OnceLock::new());

    // Create AppContext
    let app_context = Arc::new(AppContext::new(
        router_config.clone(),
        client,
        rate_limiter,
        None, // tokenizer
        None, // reasoning_parser_factory
        None, // tool_parser_factory
        worker_registry,
        policy_registry,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        load_monitor,
        worker_job_queue,
        workflow_engine,
    ));

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
