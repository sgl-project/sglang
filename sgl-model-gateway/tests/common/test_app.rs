use std::sync::{Arc, OnceLock};

use axum::Router;
use reqwest::Client;
use sgl_model_gateway::{
    app_context::AppContext,
    config::RouterConfig,
    core::{
        BasicWorkerBuilder, LoadMonitor, ModelCard, RuntimeType, Worker, WorkerRegistry, WorkerType,
    },
    data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    },
    mcp::{McpConfig, McpManager},
    middleware::{AuthConfig, TokenBucket},
    policies::PolicyRegistry,
    routers::RouterTrait,
    server::{build_app, AppState},
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

    // Create AppContext using builder pattern
    let app_context = Arc::new(
        AppContext::builder()
            .router_config(router_config.clone())
            .client(client)
            .rate_limiter(rate_limiter)
            .tokenizer(None) // tokenizer
            .reasoning_parser_factory(None) // reasoning_parser_factory
            .tool_parser_factory(None) // tool_parser_factory
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .load_monitor(load_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engine(workflow_engine)
            .build()
            .unwrap(),
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

/// Create a minimal test AppContext for unit tests
#[allow(dead_code)]
pub async fn create_test_app_context() -> Arc<AppContext> {
    let router_config = RouterConfig::default();
    let client = Client::new();

    // Initialize empty OnceLocks
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engine = Arc::new(OnceLock::new());

    // Initialize MCP manager with empty config
    let mcp_manager_lock = Arc::new(OnceLock::new());
    let empty_config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: vec![],
        inventory: Default::default(),
    };
    let mcp_manager = McpManager::with_defaults(empty_config)
        .await
        .expect("Failed to create MCP manager");
    mcp_manager_lock.set(Arc::new(mcp_manager)).ok();

    // Initialize registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

    // Initialize storage backends
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    Arc::new(
        AppContext::builder()
            .router_config(router_config)
            .client(client)
            .rate_limiter(None)
            .tokenizer(None)
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .load_monitor(None)
            .worker_job_queue(worker_job_queue)
            .workflow_engine(workflow_engine)
            .mcp_manager(mcp_manager_lock)
            .build()
            .unwrap(),
    )
}

/// Register an external worker (OpenAI-compatible API endpoint) in the test AppContext.
///
/// This is used by tests that need to test the OpenAI router, which expects
/// workers to be registered in the WorkerRegistry before routing requests.
///
/// # Arguments
/// * `ctx` - The AppContext to register the worker in
/// * `url` - The base URL of the external API endpoint
/// * `models` - Optional list of model IDs this worker supports. If empty, uses "gpt-3.5-turbo" as default.
#[allow(dead_code)]
pub fn register_external_worker(ctx: &Arc<AppContext>, url: &str, models: Option<Vec<&str>>) {
    let model_list: Vec<ModelCard> = models
        .unwrap_or_else(|| vec!["gpt-3.5-turbo"])
        .into_iter()
        .map(ModelCard::new)
        .collect();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .runtime_type(RuntimeType::External)
            .models(model_list)
            .build(),
    );

    ctx.worker_registry.register(worker);
}

/// Register an external worker with a custom model card that has aliases.
///
/// # Arguments
/// * `ctx` - The AppContext to register the worker in
/// * `url` - The base URL of the external API endpoint
/// * `model_card` - A fully configured ModelCard with aliases, provider, etc.
#[allow(dead_code)]
pub fn register_external_worker_with_card(ctx: &Arc<AppContext>, url: &str, model_card: ModelCard) {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .runtime_type(RuntimeType::External)
            .model(model_card)
            .build(),
    );

    ctx.worker_registry.register(worker);
}
