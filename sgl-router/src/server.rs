use crate::{
    config::{ConnectionMode, HistoryBackend, RouterConfig, RoutingMode},
    core::{LoadMonitor, WorkerManager, WorkerRegistry, WorkerType},
    data_connector::{
        Conversation, ConversationId, ConversationMetadata, ConversationStorageError,
        MemoryConversationStorage, MemoryResponseStorage, NewConversation, NoOpConversationStorage,
        NoOpResponseStorage, OracleConversationStorage, OracleResponseStorage,
        SharedConversationStorage, SharedResponseStorage,
    },
    logging::{self, LoggingConfig},
    metrics::{self, PrometheusConfig},
    middleware::{self, AuthConfig, QueuedRequest, TokenBucket},
    policies::PolicyRegistry,
    protocols::{
        spec::{
            ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest,
            RerankRequest, ResponsesGetParams, ResponsesRequest, V1RerankReqInput,
        },
        worker_spec::{WorkerApiResponse, WorkerConfigRequest, WorkerErrorResponse},
    },
    reasoning_parser::ReasoningParserFactory,
    routers::{router_manager::RouterManager, RouterTrait},
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    tokenizer::{factory as tokenizer_factory, traits::Tokenizer},
    tool_parser::ToolParserFactory,
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
use std::{
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
    time::Duration,
};
use tokio::{net::TcpListener, signal, spawn};
use tracing::{error, info, warn, Level};

const MAX_METADATA_PROPERTIES: usize = 16;

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    pub rate_limiter: Arc<TokenBucket>,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub response_storage: SharedResponseStorage,
    pub conversation_storage: SharedConversationStorage,
    pub load_monitor: Option<Arc<LoadMonitor>>,
}

impl AppContext {
    pub fn new(
        router_config: RouterConfig,
        client: Client,
        max_concurrent_requests: usize,
        rate_limit_tokens_per_second: Option<usize>,
    ) -> Result<Self, String> {
        let rate_limit_tokens = rate_limit_tokens_per_second.unwrap_or(max_concurrent_requests);
        let rate_limiter = Arc::new(TokenBucket::new(max_concurrent_requests, rate_limit_tokens));

        let (tokenizer, reasoning_parser_factory, tool_parser_factory) =
            if router_config.connection_mode == ConnectionMode::Grpc {
                let tokenizer_path = router_config
                    .tokenizer_path
                    .clone()
                    .or_else(|| router_config.model_path.clone())
                    .ok_or_else(|| {
                        "gRPC mode requires either --tokenizer-path or --model-path to be specified"
                            .to_string()
                    })?;

                let tokenizer = Some(
                    tokenizer_factory::create_tokenizer(&tokenizer_path)
                        .map_err(|e| format!("Failed to create tokenizer: {e}"))?,
                );
                let reasoning_parser_factory = Some(ReasoningParserFactory::new());
                let tool_parser_factory = Some(ToolParserFactory::new());

                (tokenizer, reasoning_parser_factory, tool_parser_factory)
            } else {
                (None, None, None)
            };

        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

        let router_manager = None;

        let (response_storage, conversation_storage): (
            SharedResponseStorage,
            SharedConversationStorage,
        ) = match router_config.history_backend {
            HistoryBackend::Memory => (
                Arc::new(MemoryResponseStorage::new()),
                Arc::new(MemoryConversationStorage::new()),
            ),
            HistoryBackend::None => (
                Arc::new(NoOpResponseStorage::new()),
                Arc::new(NoOpConversationStorage::new()),
            ),
            HistoryBackend::Oracle => {
                let oracle_cfg = router_config.oracle.clone().ok_or_else(|| {
                    "oracle configuration is required when history_backend=oracle".to_string()
                })?;

                let response_storage = OracleResponseStorage::new(oracle_cfg.clone()).map_err(|err| {
                    format!("failed to initialize Oracle response storage: {err}")
                })?;

                let conversation_storage = OracleConversationStorage::new(oracle_cfg).map_err(|err| {
                    format!("failed to initialize Oracle conversation storage: {err}")
                })?;

                (Arc::new(response_storage), Arc::new(conversation_storage))
            }
        };

        let load_monitor = Some(Arc::new(LoadMonitor::new(
            worker_registry.clone(),
            policy_registry.clone(),
            client.clone(),
            router_config.worker_startup_check_interval_secs,
        )));

        Ok(Self {
            client,
            router_config,
            rate_limiter,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_factory,
            worker_registry,
            policy_registry,
            router_manager,
            response_storage,
            conversation_storage,
            load_monitor,
        })
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
    Json(body): Json<ChatCompletionRequest>,
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
    Json(body): Json<RerankRequest>,
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
    _headers: http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let metadata = match extract_metadata(&body) {
        Ok(meta) => meta,
        Err(resp) => return *resp,
    };

    match state
        .context
        .conversation_storage
        .create_conversation(NewConversation { metadata })
        .await
    {
        Ok(conversation) => {
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(err) => storage_error_response(err),
    }
}

async fn v1_conversations_get(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    _headers: http::HeaderMap,
) -> Response {
    let id = ConversationId::from_string(conversation_id.clone());
    match state
        .context
        .conversation_storage
        .get_conversation(&id)
        .await
    {
        Ok(Some(conversation)) => {
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => not_found_response(&conversation_id),
        Err(err) => storage_error_response(err),
    }
}

async fn v1_conversations_update(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    _headers: http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let id = ConversationId::from_string(conversation_id.clone());
    let existing = match state
        .context
        .conversation_storage
        .get_conversation(&id)
        .await
    {
        Ok(Some(conv)) => conv,
        Ok(None) => return not_found_response(&conversation_id),
        Err(err) => return storage_error_response(err),
    };

    let patch = match parse_metadata_patch(&body) {
        Ok(patch) => patch,
        Err(resp) => return *resp,
    };

    let merged_metadata = match patch {
        MetadataPatch::NoChange => {
            return (StatusCode::OK, Json(conversation_to_json(&existing))).into_response();
        }
        other => match apply_metadata_patch(existing.metadata.clone(), other) {
            Ok(result) => result,
            Err(ValidationError { previous, updated }) => {
                return metadata_too_many_properties_error(previous, updated);
            }
        },
    };

    match state
        .context
        .conversation_storage
        .update_conversation(&id, merged_metadata)
        .await
    {
        Ok(Some(conversation)) => {
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => not_found_response(&conversation_id),
        Err(err) => storage_error_response(err),
    }
}

async fn v1_conversations_delete(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    _headers: http::HeaderMap,
) -> Response {
    let id = ConversationId::from_string(conversation_id.clone());
    match state
        .context
        .conversation_storage
        .delete_conversation(&id)
        .await
    {
        Ok(true) => (
            StatusCode::OK,
            Json(json!({
                "id": conversation_id,
                "object": "conversation.deleted",
                "deleted": true,
            })),
        )
            .into_response(),
        Ok(false) => not_found_response(conversation_id.as_str()),
        Err(err) => storage_error_response(err),
    }
}

type MetadataResult<T> = Result<T, Box<Response>>;

fn extract_metadata(payload: &Value) -> MetadataResult<Option<ConversationMetadata>> {
    parse_metadata_create(payload)
}

fn parse_metadata_create(payload: &Value) -> MetadataResult<Option<ConversationMetadata>> {
    match payload.get("metadata") {
        Some(Value::Object(map)) => {
            if map.len() > MAX_METADATA_PROPERTIES {
                return Err(Box::new(metadata_too_many_properties_error(0, map.len())));
            }
            Ok(Some(map.clone()))
        }
        Some(Value::Null) | None => Ok(None),
        Some(other) => Err(Box::new(invalid_metadata_type_response(other.clone()))),
    }
}

#[derive(Debug)]
enum MetadataPatch {
    NoChange,
    ClearAll,
    Merge(ConversationMetadata),
}

#[derive(Debug)]
struct ValidationError {
    previous: usize,
    updated: usize,
}

fn parse_metadata_patch(payload: &Value) -> MetadataResult<MetadataPatch> {
    match payload.get("metadata") {
        None => Ok(MetadataPatch::NoChange),
        Some(Value::Null) => Ok(MetadataPatch::ClearAll),
        Some(Value::Object(map)) => Ok(MetadataPatch::Merge(map.clone())),
        Some(other) => Err(Box::new(invalid_metadata_type_response(other.clone()))),
    }
}

fn apply_metadata_patch(
    existing: Option<ConversationMetadata>,
    patch: MetadataPatch,
) -> Result<Option<ConversationMetadata>, ValidationError> {
    match patch {
        MetadataPatch::NoChange => Ok(existing),
        MetadataPatch::ClearAll => Ok(None),
        MetadataPatch::Merge(updates) => {
            let mut merged = existing.unwrap_or_default();
            let previous = merged.len();

            for (key, value) in updates.into_iter() {
                if value.is_null() {
                    merged.remove(&key);
                } else {
                    merged.insert(key, value);
                }
            }

            let updated = merged.len();
            if updated > MAX_METADATA_PROPERTIES {
                return Err(ValidationError { previous, updated });
            }

            if merged.is_empty() {
                Ok(None)
            } else {
                Ok(Some(merged))
            }
        }
    }
}

fn conversation_to_json(conversation: &Conversation) -> Value {
    json!({
        "id": conversation.id.0,
        "object": "conversation",
        "created_at": conversation.created_at.timestamp(),
        "metadata": conversation
            .metadata
            .clone()
            .map(Value::Object)
            .unwrap_or(Value::Null),
    })
}

fn not_found_response(conversation_id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": format!(
                    "Conversation with id '{}' not found.",
                    conversation_id
                ),
                "type": "invalid_request_error",
                "param": Value::Null,
                "code": Value::Null
            }
        })),
    )
        .into_response()
}

fn storage_error_response(err: ConversationStorageError) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": err.to_string(),
                "type": "internal_error",
                "param": Value::Null,
                "code": Value::Null
            }
        })),
    )
        .into_response()
}

fn invalid_metadata_type_response(value: Value) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": format!(
                    "Invalid 'metadata': expected object or null but got {}",
                    value
                ),
                "type": "invalid_request_error",
                "param": "metadata",
                "code": "metadata_invalid_type"
            }
        })),
    )
        .into_response()
}

fn metadata_too_many_properties_error(previous: usize, updated: usize) -> Response {
    let message = if previous == 0 {
        format!(
            "too many properties. Metadata can have at most {} properties, but you provided {} properties.",
            MAX_METADATA_PROPERTIES, updated
        )
    } else {
        format!(
            "too many properties after update. Metadata can have at most {} properties, but your update resulted in an object with {} properties (up from {} properties before).",
            MAX_METADATA_PROPERTIES, updated, previous
        )
    };
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": format!("Invalid 'metadata': {}", message),
                "type": "invalid_request_error",
                "param": "metadata",
                "code": "metadata_max_properties_exceeded",
                "extra": {
                    "previous_property_count": previous,
                    "updated_property_count": updated
                }
            }
        })),
    )
        .into_response()
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

    (
        StatusCode::OK,
        Json(json!({
            "workers": loads
        })),
    )
        .into_response()
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

    let result = WorkerManager::add_worker_from_config(&config, &state.context).await;

    match result {
        Ok(message) => {
            let response = WorkerApiResponse {
                success: true,
                message,
                worker: None,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(error) => {
            let error_response = WorkerErrorResponse {
                error,
                code: "ADD_WORKER_FAILED".to_string(),
            };
            (StatusCode::BAD_REQUEST, Json(error_response)).into_response()
        }
    }
}

async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let response = serde_json::json!({
        "workers": workers.iter().map(|worker| {
            let mut worker_info = serde_json::json!({
                "url": worker.url(),
                "model_id": worker.model_id(),
                "worker_type": match worker.worker_type() {
                    WorkerType::Regular => "regular",
                    WorkerType::Prefill { .. } => "prefill",
                    WorkerType::Decode => "decode",
                },
                "is_healthy": worker.is_healthy(),
                "load": worker.load(),
                "connection_mode": format!("{:?}", worker.connection_mode()),
                "priority": worker.priority(),
                "cost": worker.cost(),
            });

            if let WorkerType::Prefill { bootstrap_port } = worker.worker_type() {
                worker_info["bootstrap_port"] = serde_json::json!(bootstrap_port);
            }

            worker_info
        }).collect::<Vec<_>>(),
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
    let workers = WorkerManager::get_worker_urls(&state.context.worker_registry);
    if workers.contains(&url) {
        Json(json!({
            "url": url,
            "model_id": "unknown",
            "is_healthy": true
        }))
        .into_response()
    } else {
        let error = WorkerErrorResponse {
            error: format!("Worker {url} not found"),
            code: "WORKER_NOT_FOUND".to_string(),
        };
        (StatusCode::NOT_FOUND, Json(error)).into_response()
    }
}

async fn delete_worker(State(state): State<Arc<AppState>>, Path(url): Path<String>) -> Response {
    let result = WorkerManager::remove_worker(&url, &state.context);

    match result {
        Ok(message) => {
            let response = WorkerApiResponse {
                success: true,
                message,
                worker: None,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(error) => {
            let error_response = WorkerErrorResponse {
                error,
                code: "REMOVE_WORKER_FAILED".to_string(),
            };
            (StatusCode::BAD_REQUEST, Json(error_response)).into_response()
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
            get(v1_conversations_get),
        )
        .route(
            "/v1/conversations/{conversation_id}",
            post(v1_conversations_update),
        )
        .route(
            "/v1/conversations/{conversation_id}",
            delete(v1_conversations_delete),
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

    let app_context = AppContext::new(
        config.router_config.clone(),
        client.clone(),
        config.router_config.max_concurrent_requests,
        config.router_config.rate_limit_tokens_per_second,
    )?;

    let app_context = Arc::new(app_context);

    info!(
        "Initializing workers for routing mode: {:?}",
        config.router_config.mode
    );
    WorkerManager::initialize_workers(
        &config.router_config,
        &app_context.worker_registry,
        Some(&app_context.policy_registry),
    )
    .await
    .map_err(|e| format!("Failed to initialize workers: {}", e))?;

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

    if let Some(processor) = processor {
        spawn(processor.run());
        info!(
            "Started request queue with size: {}, timeout: {}s",
            config.router_config.queue_size, config.router_config.queue_timeout_secs
        );
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
