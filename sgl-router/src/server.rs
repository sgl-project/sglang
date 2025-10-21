use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    serve, Json, Router,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::{net::TcpListener, signal, spawn};
use tracing::{error, info, warn, Level};

use crate::{
    config::{ConnectionMode, HistoryBackend, RouterConfig, RoutingMode},
    core::{
        worker_to_info, workflow::WorkflowEngine, Job, JobQueue, JobQueueConfig, LoadMonitor,
        WorkerManager, WorkerRegistry, WorkerType,
    },
    data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        NoOpConversationStorage, NoOpResponseStorage, OracleConversationItemStorage,
        OracleConversationStorage, OracleResponseStorage, SharedConversationItemStorage,
        SharedConversationStorage, SharedResponseStorage,
    },
    logging::{self, LoggingConfig},
    metrics::{self, PrometheusConfig},
    middleware::{self, AuthConfig, QueuedRequest, TokenBucket},
    policies::PolicyRegistry,
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::{RerankRequest, V1RerankReqInput},
        responses::{ResponsesGetParams, ResponsesRequest},
        validated::ValidatedJson,
        worker_spec::{WorkerConfigRequest, WorkerErrorResponse, WorkerInfo},
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::{router_manager::RouterManager, RouterTrait},
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    tokenizer::{
        cache::{CacheConfig, CachedTokenizer},
        factory as tokenizer_factory,
        traits::Tokenizer,
    },
    tool_parser::ParserFactory as ToolParserFactory,
};

//

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    pub rate_limiter: Option<Arc<TokenBucket>>,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub response_storage: SharedResponseStorage,
    pub conversation_storage: SharedConversationStorage,
    pub conversation_item_storage: SharedConversationItemStorage,
    pub load_monitor: Option<Arc<LoadMonitor>>,
    pub configured_reasoning_parser: Option<String>,
    pub configured_tool_parser: Option<String>,
    pub worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,
    pub workflow_engine: Arc<OnceLock<Arc<WorkflowEngine>>>,
}

impl AppContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        router_config: RouterConfig,
        client: Client,
        rate_limiter: Option<Arc<TokenBucket>>,
        tokenizer: Option<Arc<dyn Tokenizer>>,
        reasoning_parser_factory: Option<ReasoningParserFactory>,
        tool_parser_factory: Option<ToolParserFactory>,
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        response_storage: SharedResponseStorage,
        conversation_storage: SharedConversationStorage,
        conversation_item_storage: SharedConversationItemStorage,
        load_monitor: Option<Arc<LoadMonitor>>,
        worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,
        workflow_engine: Arc<OnceLock<Arc<WorkflowEngine>>>,
    ) -> Self {
        let configured_reasoning_parser = router_config.reasoning_parser.clone();
        let configured_tool_parser = router_config.tool_call_parser.clone();

        Self {
            client,
            router_config,
            rate_limiter,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_factory,
            worker_registry,
            policy_registry,
            router_manager: None,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            load_monitor,
            configured_reasoning_parser,
            configured_tool_parser,
            worker_job_queue,
            workflow_engine,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
}

async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

    let is_ready = if state.context.router_config.enable_igw {
        !healthy_workers.is_empty()
    } else {
        match &state.context.router_config.mode {
            RoutingMode::PrefillDecode { .. } => {
                let has_prefill = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }));
                let has_decode = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Decode));
                has_prefill && has_decode
            }
            RoutingMode::Regular { .. } => !healthy_workers.is_empty(),
            RoutingMode::OpenAI { .. } => !healthy_workers.is_empty(),
        }
    };

    if is_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_workers.len(),
                "total_workers": workers.len()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "reason": "insufficient healthy workers"
            })),
        )
            .into_response()
    }
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

async fn generate(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<GenerateRequest>,
) -> Response {
    state
        .router
        .route_generate(Some(&headers), &body, None)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    state.router.route_chat(Some(&headers), &body, None).await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<CompletionRequest>,
) -> Response {
    state
        .router
        .route_completion(Some(&headers), &body, None)
        .await
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    ValidatedJson(body): ValidatedJson<RerankRequest>,
) -> Response {
    state.router.route_rerank(Some(&headers), &body, None).await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<V1RerankReqInput>,
) -> Response {
    state
        .router
        .route_rerank(Some(&headers), &body.into(), None)
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<ResponsesRequest>,
) -> Response {
    state
        .router
        .route_responses(Some(&headers), &body, None)
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    state
        .router
        .route_embeddings(Some(&headers), &body, None)
        .await
}

async fn v1_classify(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<ClassifyRequest>,
) -> Response {
    state
        .router
        .route_classify(Some(&headers), &body, None)
        .await
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
    Query(params): Query<ResponsesGetParams>,
) -> Response {
    state
        .router
        .get_response(Some(&headers), &response_id, &params)
        .await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .delete_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .list_response_input_items(Some(&headers), &response_id)
        .await
}

async fn v1_conversations_create(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    state
        .router
        .create_conversation(Some(&headers), &body)
        .await
}

async fn v1_conversations_get(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .get_conversation(Some(&headers), &conversation_id)
        .await
}

async fn v1_conversations_update(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    headers: http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    state
        .router
        .update_conversation(Some(&headers), &conversation_id, &body)
        .await
}

async fn v1_conversations_delete(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .delete_conversation(Some(&headers), &conversation_id)
        .await
}

#[derive(Deserialize, Default)]
struct ListItemsQuery {
    limit: Option<usize>,
    order: Option<String>,
    after: Option<String>,
}

async fn v1_conversations_list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(ListItemsQuery {
        limit,
        order,
        after,
    }): Query<ListItemsQuery>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .list_conversation_items(Some(&headers), &conversation_id, limit, order, after)
        .await
}

#[derive(Deserialize, Default)]
struct GetItemQuery {
    /// Additional fields to include in response (not yet implemented)
    include: Option<Vec<String>>,
}

async fn v1_conversations_create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    headers: http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    state
        .router
        .create_conversation_items(Some(&headers), &conversation_id, &body)
        .await
}

async fn v1_conversations_get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    Query(query): Query<GetItemQuery>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .get_conversation_item(Some(&headers), &conversation_id, &item_id, query.include)
        .await
}

async fn v1_conversations_delete_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .delete_conversation_item(Some(&headers), &conversation_id, &item_id)
        .await
}

#[derive(Deserialize)]
struct AddWorkerQuery {
    url: String,
    api_key: Option<String>,
}

async fn add_worker(
    State(state): State<Arc<AppState>>,
    Query(AddWorkerQuery { url, api_key }): Query<AddWorkerQuery>,
) -> Response {
    // Warn if router has API key but worker is being added without one
    if state.context.router_config.api_key.is_some() && api_key.is_none() {
        warn!(
            "Adding worker {} without API key while router has API key configured. \
            Worker will be accessible without authentication. \
            If the worker requires the same API key as the router, please specify it explicitly.",
            url
        );
    }

    let result = WorkerManager::add_worker(&url, &api_key, &state.context).await;

    match result {
        Ok(message) => (StatusCode::OK, message).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, error).into_response(),
    }
}

async fn list_workers(State(state): State<Arc<AppState>>) -> Response {
    let worker_list = WorkerManager::get_worker_urls(&state.context.worker_registry);
    Json(json!({ "urls": worker_list })).into_response()
}

async fn remove_worker(
    State(state): State<Arc<AppState>>,
    Query(AddWorkerQuery { url, .. }): Query<AddWorkerQuery>,
) -> Response {
    let result = WorkerManager::remove_worker(&url, &state.context);

    match result {
        Ok(message) => (StatusCode::OK, message).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, error).into_response(),
    }
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    match WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
        .await
    {
        Ok(result) => {
            if result.failed.is_empty() {
                (
                    StatusCode::OK,
                    Json(json!({
                        "status": "success",
                        "message": result.message,
                        "workers_flushed": result.successful.len(),
                        "total_http_workers": result.http_workers,
                        "total_workers": result.total_workers
                    })),
                )
                    .into_response()
            } else {
                (
                    StatusCode::PARTIAL_CONTENT,
                    Json(json!({
                        "status": "partial_success",
                        "message": result.message,
                        "successful": result.successful,
                        "failed": result.failed.into_iter().map(|(url, err)| json!({
                            "worker": url,
                            "error": err
                        })).collect::<Vec<_>>(),
                        "total_http_workers": result.http_workers,
                        "total_workers": result.total_workers
                    })),
                )
                    .into_response()
            }
        }
        Err(e) => {
            error!("Failed to flush cache: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "status": "error",
                    "message": format!("Failed to flush cache: {}", e)
                })),
            )
                .into_response()
        }
    }
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    let result =
        WorkerManager::get_all_worker_loads(&state.context.worker_registry, &state.context.client)
            .await;

    let loads: Vec<Value> = result
        .loads
        .iter()
        .map(|info| {
            json!({
                "worker": &info.worker,
                "load": info.load
            })
        })
        .collect();

    (StatusCode::OK, Json(json!({ "workers": loads }))).into_response()
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    // Warn if router has API key but worker is being added without one
    if state.context.router_config.api_key.is_some() && config.api_key.is_none() {
        warn!(
            "Adding worker {} without API key while router has API key configured. \
            Worker will be accessible without authentication. \
            If the worker requires the same API key as the router, please specify it explicitly.",
            config.url
        );
    }

    // Submit job for async processing
    let worker_url = config.url.clone();
    let job = Job::AddWorker {
        config: Box::new(config),
    };

    let job_queue = state
        .context
        .worker_job_queue
        .get()
        .expect("JobQueue not initialized");
    match job_queue.submit(job).await {
        Ok(_) => {
            let response = json!({
                "status": "accepted",
                "worker_id": worker_url,
                "message": "Worker addition queued for background processing"
            });
            (StatusCode::ACCEPTED, Json(response)).into_response()
        }
        Err(error) => {
            let error_response = WorkerErrorResponse {
                error,
                code: "INTERNAL_SERVER_ERROR".to_string(),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
        }
    }
}

async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let worker_infos: Vec<WorkerInfo> = workers.iter().map(worker_to_info).collect();

    let response = json!({
        "workers": worker_infos,
        "total": workers.len(),
        "stats": {
            "prefill_count": state.context.worker_registry.get_prefill_workers().len(),
            "decode_count": state.context.worker_registry.get_decode_workers().len(),
            "regular_count": state.context.worker_registry.get_by_type(&WorkerType::Regular).len(),
        }
    });
    Json(response).into_response()
}

async fn get_worker(State(state): State<Arc<AppState>>, Path(url): Path<String>) -> Response {
    let job_queue = state
        .context
        .worker_job_queue
        .get()
        .expect("JobQueue not initialized");

    if let Some(worker) = state.context.worker_registry.get_by_url(&url) {
        // Worker exists in registry, get its full info and attach job status if any
        let mut worker_info = worker_to_info(&worker);
        if let Some(status) = job_queue.get_status(&url) {
            worker_info.job_status = Some(status);
        }
        return Json(worker_info).into_response();
    }

    // Worker not in registry, check job queue for its status
    if let Some(status) = job_queue.get_status(&url) {
        // Create a partial WorkerInfo to report the job status
        let worker_info = WorkerInfo {
            id: url.clone(),
            url: url.clone(),
            model_id: "unknown".to_string(),
            priority: 0,
            cost: 1.0,
            worker_type: "unknown".to_string(),
            is_healthy: false,
            load: 0,
            connection_mode: "unknown".to_string(),
            tokenizer_path: None,
            reasoning_parser: None,
            tool_parser: None,
            chat_template: None,
            bootstrap_port: None,
            metadata: std::collections::HashMap::new(),
            job_status: Some(status),
        };
        return Json(worker_info).into_response();
    }

    // Worker not found in registry or job queue
    let error = WorkerErrorResponse {
        error: format!("Worker {url} not found"),
        code: "WORKER_NOT_FOUND".to_string(),
    };
    (StatusCode::NOT_FOUND, Json(error)).into_response()
}

async fn delete_worker(State(state): State<Arc<AppState>>, Path(url): Path<String>) -> Response {
    let worker_id = url.clone();
    let job = Job::RemoveWorker { url };

    let job_queue = state
        .context
        .worker_job_queue
        .get()
        .expect("JobQueue not initialized");
    match job_queue.submit(job).await {
        Ok(_) => {
            let response = json!({
                "status": "accepted",
                "worker_id": worker_id,
                "message": "Worker removal queued for background processing"
            });
            (StatusCode::ACCEPTED, Json(response)).into_response()
        }
        Err(error) => {
            let error_response = WorkerErrorResponse {
                error,
                code: "INTERNAL_SERVER_ERROR".to_string(),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
        }
    }
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub router_config: RouterConfig,
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
    pub request_id_headers: Option<Vec<String>>,
}

pub fn build_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Router {
    let protected_routes = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/rerank", post(rerank))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/responses", post(v1_responses))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/classify", post(v1_classify))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input",
            get(v1_responses_list_input_items),
        )
        .route("/v1/conversations", post(v1_conversations_create))
        .route(
            "/v1/conversations/{conversation_id}",
            get(v1_conversations_get)
                .post(v1_conversations_update)
                .delete(v1_conversations_delete),
        )
        .route(
            "/v1/conversations/{conversation_id}/items",
            get(v1_conversations_list_items).post(v1_conversations_create_items),
        )
        .route(
            "/v1/conversations/{conversation_id}/items/{item_id}",
            get(v1_conversations_get_item).delete(v1_conversations_delete_item),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/v1/models", get(v1_models))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info));

    let admin_routes = Router::new()
        .route("/add_worker", post(add_worker))
        .route("/remove_worker", post(remove_worker))
        .route("/list_workers", get(list_workers))
        .route("/flush_cache", post(flush_cache))
        .route("/get_loads", get(get_loads))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    let worker_routes = Router::new()
        .route("/workers", post(create_worker))
        .route("/workers", get(list_workers_rest))
        .route("/workers/{url}", get(get_worker))
        .route("/workers/{url}", delete(delete_worker))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state)
}

pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    let _log_guard = if !LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        Some(logging::init_logging(LoggingConfig {
            level: config
                .log_level
                .as_deref()
                .and_then(|s| match s.to_uppercase().parse::<Level>() {
                    Ok(l) => Some(l),
                    Err(_) => {
                        warn!("Invalid log level string: '{s}'. Defaulting to INFO.");
                        None
                    }
                })
                .unwrap_or(Level::INFO),
            json_format: false,
            log_dir: config.log_dir.clone(),
            colorize: true,
            log_file_name: "sgl-router".to_string(),
            log_targets: None,
        }))
    } else {
        None
    };

    if let Some(prometheus_config) = &config.prometheus_config {
        metrics::start_prometheus(prometheus_config.clone());
    }

    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | max_payload: {}MB",
        config.host,
        config.port,
        config.router_config.mode,
        config.router_config.policy,
        config.max_payload_size / (1024 * 1024)
    );

    let client = Client::builder()
        .pool_idle_timeout(Some(Duration::from_secs(50)))
        .pool_max_idle_per_host(500)
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .connect_timeout(Duration::from_secs(10))
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .build()
        .expect("Failed to create HTTP client");

    // Initialize rate limiter
    let rate_limiter = match config.router_config.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = config
                .router_config
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    // Initialize tokenizer and parser factories for gRPC mode
    let (tokenizer, reasoning_parser_factory, tool_parser_factory) = if config
        .router_config
        .connection_mode
        == ConnectionMode::Grpc
    {
        let tokenizer_path = config
            .router_config
            .tokenizer_path
            .clone()
            .or_else(|| config.router_config.model_path.clone())
            .ok_or_else(|| {
                "gRPC mode requires either --tokenizer-path or --model-path to be specified"
                    .to_string()
            })?;

        let base_tokenizer =
                tokenizer_factory::create_tokenizer_with_chat_template_blocking(
                    &tokenizer_path,
                    config.router_config.chat_template.as_deref(),
                )
                .map_err(|e| {
                    format!(
                        "Failed to create tokenizer from '{}': {}. \
                        Ensure the path is valid and points to a tokenizer file (tokenizer.json) \
                        or a HuggingFace model ID. For directories, ensure they contain tokenizer files.",
                        tokenizer_path, e
                    )
                })?;

        // Conditionally wrap with caching layer if at least one cache is enabled
        let tokenizer = if config.router_config.tokenizer_cache.enable_l0
            || config.router_config.tokenizer_cache.enable_l1
        {
            let cache_config = CacheConfig {
                enable_l0: config.router_config.tokenizer_cache.enable_l0,
                l0_max_entries: config.router_config.tokenizer_cache.l0_max_entries,
                enable_l1: config.router_config.tokenizer_cache.enable_l1,
                l1_max_memory: config.router_config.tokenizer_cache.l1_max_memory,
            };
            Some(Arc::new(CachedTokenizer::new(base_tokenizer, cache_config)) as Arc<dyn Tokenizer>)
        } else {
            // Use base tokenizer directly without caching
            Some(base_tokenizer)
        };
        let reasoning_parser_factory = Some(ReasoningParserFactory::new());
        let tool_parser_factory = Some(ToolParserFactory::new());

        (tokenizer, reasoning_parser_factory, tool_parser_factory)
    } else {
        (None, None, None)
    };

    // Initialize worker registry and policy registry
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.router_config.policy.clone()));

    // Initialize storage backends
    let (response_storage, conversation_storage): (
        SharedResponseStorage,
        SharedConversationStorage,
    ) = match config.router_config.history_backend {
        HistoryBackend::Memory => {
            info!("Initializing data connector: Memory");
            (
                Arc::new(MemoryResponseStorage::new()),
                Arc::new(MemoryConversationStorage::new()),
            )
        }
        HistoryBackend::None => {
            info!("Initializing data connector: None (no persistence)");
            (
                Arc::new(NoOpResponseStorage::new()),
                Arc::new(NoOpConversationStorage::new()),
            )
        }
        HistoryBackend::Oracle => {
            let oracle_cfg = config.router_config.oracle.clone().ok_or_else(|| {
                "oracle configuration is required when history_backend=oracle".to_string()
            })?;
            info!(
                "Initializing data connector: Oracle ATP (pool: {}-{})",
                oracle_cfg.pool_min, oracle_cfg.pool_max
            );

            let response_storage = OracleResponseStorage::new(oracle_cfg.clone())
                .map_err(|err| format!("failed to initialize Oracle response storage: {err}"))?;

            let conversation_storage =
                OracleConversationStorage::new(oracle_cfg.clone()).map_err(|err| {
                    format!("failed to initialize Oracle conversation storage: {err}")
                })?;
            info!("Data connector initialized successfully: Oracle ATP");

            (Arc::new(response_storage), Arc::new(conversation_storage))
        }
    };

    // Initialize conversation items storage
    let conversation_item_storage: SharedConversationItemStorage =
        match config.router_config.history_backend {
            HistoryBackend::Oracle => {
                let oracle_cfg = config.router_config.oracle.clone().ok_or_else(|| {
                    "oracle configuration is required when history_backend=oracle".to_string()
                })?;
                Arc::new(OracleConversationItemStorage::new(oracle_cfg).map_err(|e| {
                    format!("failed to initialize Oracle conversation item storage: {e}")
                })?)
            }
            _ => Arc::new(MemoryConversationItemStorage::new()),
        };

    // Initialize load monitor
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.router_config.worker_startup_check_interval_secs,
    )));

    // Create empty OnceLock for worker job queue and workflow engine (will be initialized below)
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engine = Arc::new(OnceLock::new());

    // Create AppContext with all initialized components
    let app_context = AppContext::new(
        config.router_config.clone(),
        client.clone(),
        rate_limiter,
        tokenizer,
        reasoning_parser_factory,
        tool_parser_factory,
        worker_registry,
        policy_registry,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        load_monitor,
        worker_job_queue,
        workflow_engine,
    );

    let app_context = Arc::new(app_context);

    let weak_context = Arc::downgrade(&app_context);
    let worker_job_queue = JobQueue::new(JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(worker_job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize workflow engine and register workflows
    let engine = Arc::new(WorkflowEngine::new());

    engine
        .event_bus()
        .subscribe(Arc::new(crate::core::workflow::LoggingSubscriber))
        .await;

    engine.register_workflow(crate::core::workflow::create_worker_registration_workflow());
    app_context
        .workflow_engine
        .set(engine)
        .expect("WorkflowEngine should only be initialized once");
    info!("Workflow engine initialized with worker registration workflow");

    info!(
        "Initializing workers for routing mode: {:?}",
        config.router_config.mode
    );

    // Submit worker initialization job to queue
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");
    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(config.router_config.clone()),
    };
    job_queue
        .submit(job)
        .await
        .map_err(|e| format!("Failed to submit worker initialization job: {}", e))?;

    let worker_stats = app_context.worker_registry.stats();
    info!(
        "Workers initialized: {} total, {} healthy",
        worker_stats.total_workers, worker_stats.healthy_workers
    );

    let router_manager = RouterManager::from_config(&config, &app_context).await?;
    let router: Arc<dyn RouterTrait> = router_manager.clone();

    let _health_checker = app_context
        .worker_registry
        .start_health_checker(config.router_config.health_check.check_interval_secs);
    info!(
        "Started health checker for workers with {}s interval",
        config.router_config.health_check.check_interval_secs
    );

    if let Some(ref load_monitor) = app_context.load_monitor {
        load_monitor.start().await;
        info!("Started LoadMonitor for PowerOfTwo policies");
    }

    let (limiter, processor) = middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.router_config.queue_size,
        Duration::from_secs(config.router_config.queue_timeout_secs),
    );

    if app_context.rate_limiter.is_none() {
        info!("Rate limiting is disabled (max_concurrent_requests = -1)");
    }

    match processor {
        Some(proc) => {
            spawn(proc.run());
            info!(
                "Started request queue (size: {}, timeout: {}s)",
                config.router_config.queue_size, config.router_config.queue_timeout_secs
            );
        }
        None => {
            info!(
                "Rate limiting enabled (max_concurrent_requests = {}, queue disabled)",
                config.router_config.max_concurrent_requests
            );
        }
    }

    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: limiter.queue_tx.clone(),
        router_manager: Some(router_manager),
    });
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            let app_context_arc = Arc::clone(&app_state.context);
            match start_service_discovery(service_discovery_config, app_context_arc).await {
                Ok(handle) => {
                    info!("Service discovery started");
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {e}");
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

    info!(
        "Router ready | workers: {:?}",
        WorkerManager::get_worker_urls(&app_state.context.worker_registry)
    );

    let request_id_headers = config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    let auth_config = AuthConfig {
        api_key: config.router_config.api_key.clone(),
    };

    let app = build_app(
        app_state,
        auth_config,
        config.max_payload_size,
        request_id_headers,
        config.router_config.cors_allowed_origins.clone(),
    );

    // TcpListener::bind accepts &str and handles IPv4/IPv6 via ToSocketAddrs
    let bind_addr = format!("{}:{}", config.host, config.port);
    info!("Starting server on {}", bind_addr);
    let listener = TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| format!("Failed to bind to {}: {}", bind_addr, e))?;
    serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([http::Method::GET, http::Method::POST, http::Method::OPTIONS])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}
