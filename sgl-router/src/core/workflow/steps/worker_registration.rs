//! Worker registration workflow steps
//!
//! Each step is atomic and performs a single operation in the worker registration process.
//!
//! Workflow order:
//! 1. DetectConnectionMode - Probe both HTTP and gRPC to determine connection mode
//! 2. DiscoverMetadata - Fetch metadata from the worker
//! 3. DiscoverDPInfo - Fetch DP (Data Parallel) information (only for DP-aware workers)
//! 4. CreateWorker - Build worker object(s) with merged config + metadata
//! 5. RegisterWorker - Register worker(s) in registry
//! 6. UpdatePolicies - Update policy registry with worker information
//! 7. ActivateWorker - Mark worker(s) as healthy

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    core::{
        workflow::*, BasicWorkerBuilder, CircuitBreakerConfig, ConnectionMode,
        DPAwareWorkerBuilder, HealthConfig, RuntimeType, Worker, WorkerType,
    },
    protocols::worker_spec::WorkerConfigRequest,
    routers::grpc::client::GrpcClient,
};

// HTTP client for metadata fetching
static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
});

/// Server information returned from worker endpoints
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ServerInfo {
    #[serde(alias = "model")]
    model_id: Option<String>,
    model_path: Option<String>,
    served_model_name: Option<String>,
    dp_size: Option<usize>,
    version: Option<String>,
    max_batch_size: Option<usize>,
    max_total_tokens: Option<usize>,
    max_prefill_tokens: Option<usize>,
    max_running_requests: Option<usize>,
    max_num_reqs: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// Parse server info from JSON response using serde
fn parse_server_info(json: Value) -> Result<ServerInfo, String> {
    serde_json::from_value(json).map_err(|e| format!("Failed to parse server info: {}", e))
}

/// Get server info from /get_server_info endpoint
async fn get_server_info(url: &str, api_key: Option<&str>) -> Result<ServerInfo, String> {
    let base_url = url.trim_end_matches('/');
    let server_info_url = format!("{}/get_server_info", base_url);

    let mut req = HTTP_CLIENT.get(&server_info_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", server_info_url, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            server_info_url
        ));
    }

    let json = response
        .json::<Value>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", server_info_url, e))?;

    parse_server_info(json)
}

/// Get DP info for a worker URL
async fn get_dp_info(url: &str, api_key: Option<&str>) -> Result<DpInfo, String> {
    let info = get_server_info(url, api_key).await?;

    let dp_size = info
        .dp_size
        .ok_or_else(|| format!("No dp_size in response from {}", url))?;

    let model_id = info
        .model_id
        .filter(|s| !s.is_empty())
        .or(info.served_model_name.filter(|s| !s.is_empty()))
        .or_else(|| {
            info.model_path
                .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
        })
        .unwrap_or_else(|| "unknown".to_string());

    Ok(DpInfo { dp_size, model_id })
}

/// Helper: Strip protocol prefix from URL
fn strip_protocol(url: &str) -> String {
    url.trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_start_matches("grpc://")
        .to_string()
}

/// Helper: Try HTTP health check
///
/// Uses the provided client (from app_context) which supports both HTTP and HTTPS.
/// For HTTPS URLs, the client's TLS configuration (mTLS, CA certs) is used.
/// For plain HTTP URLs, the client handles them normally without TLS overhead.
async fn try_http_health_check(
    url: &str,
    timeout_secs: u64,
    client: &Client,
) -> Result<(), String> {
    // Preserve the protocol (http or https) from the original URL
    let is_https = url.starts_with("https://");
    let protocol = if is_https { "https" } else { "http" };
    let clean_url = strip_protocol(url);
    let health_url = format!("{}://{}/health", protocol, clean_url);

    // Use the AppContext client for both HTTP and HTTPS
    // The rustls backend handles both protocols correctly
    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {}", e))?;

    Ok(())
}

/// Helper: Perform gRPC health check with runtime type
async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {}", e))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {}", e))?;

    Ok(())
}

/// Helper: Try gRPC health check
///
/// If runtime_type is specified, uses the appropriate client (SGLang or vLLM).
/// If not specified, tries SGLang first, then falls back to vLLM.
async fn try_grpc_health_check(
    url: &str,
    timeout_secs: u64,
    runtime_type: Option<&str>,
) -> Result<(), String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    match runtime_type {
        Some(runtime) => do_grpc_health_check(&grpc_url, timeout_secs, runtime).await,
        None => {
            // Runtime not specified: Try SGLang first, then vLLM as fallback
            if let Ok(()) = do_grpc_health_check(&grpc_url, timeout_secs, "sglang").await {
                return Ok(());
            }

            // Try vLLM as fallback
            do_grpc_health_check(&grpc_url, timeout_secs, "vllm")
                .await
                .map_err(|e| {
                    format!(
                        "gRPC health check failed (tried both SGLang and vLLM): {}",
                        e
                    )
                })
        }
    }
}

/// Fetch metadata from gRPC server with runtime type
async fn do_fetch_grpc_metadata(
    grpc_url: &str,
    runtime_type: &str,
) -> Result<HashMap<String, String>, String> {
    let client = GrpcClient::connect(grpc_url, runtime_type)
        .await
        .map_err(|e| format!("Failed to connect to gRPC: {}", e))?;

    let model_info = client
        .get_model_info()
        .await
        .map_err(|e| format!("Failed to fetch gRPC metadata: {}", e))?;

    Ok(model_info.to_labels())
}

/// Helper: Fetch gRPC metadata
///
/// If runtime_type is specified, uses the appropriate client (SGLang or vLLM).
/// If not specified, tries SGLang first, then falls back to vLLM.
/// Returns (labels, detected_runtime_type)
async fn fetch_grpc_metadata(
    url: &str,
    runtime_type: Option<&str>,
) -> Result<(HashMap<String, String>, String), String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    match runtime_type {
        Some(runtime) => {
            let labels = do_fetch_grpc_metadata(&grpc_url, runtime).await?;
            Ok((labels, runtime.to_string()))
        }
        None => {
            // Runtime not specified: Try SGLang first, then vLLM as fallback
            if let Ok(labels) = do_fetch_grpc_metadata(&grpc_url, "sglang").await {
                return Ok((labels, "sglang".to_string()));
            }

            // Try vLLM as fallback
            let labels = do_fetch_grpc_metadata(&grpc_url, "vllm")
                .await
                .map_err(|e| {
                    format!(
                        "Failed to fetch gRPC metadata (tried both SGLang and vLLM): {}",
                        e
                    )
                })?;
            Ok((labels, "vllm".to_string()))
        }
    }
}

/// Step 1: Detect connection mode by probing both HTTP and gRPC
pub struct DetectConnectionModeStep;

#[async_trait]
impl StepExecutor for DetectConnectionModeStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        debug!(
            "Detecting connection mode for {} (timeout: {}s, max_attempts: {})",
            config.url, config.health_check_timeout_secs, config.max_connection_attempts
        );

        // Try both protocols in parallel using configured timeout
        // Use the AppContext client which has TLS configuration (CA certs, client identity)
        let url = config.url.clone();
        let timeout = config.health_check_timeout_secs;
        let client = &app_context.client;
        let runtime_type = config.runtime.as_deref();
        let (http_result, grpc_result) = tokio::join!(
            try_http_health_check(&url, timeout, client),
            try_grpc_health_check(&url, timeout, runtime_type)
        );

        let connection_mode = match (http_result, grpc_result) {
            (Ok(_), _) => {
                debug!("{} detected as HTTP", config.url);
                ConnectionMode::Http
            }
            (_, Ok(_)) => {
                debug!("{} detected as gRPC", config.url);
                ConnectionMode::Grpc { port: None }
            }
            (Err(http_err), Err(grpc_err)) => {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("detect_connection_mode"),
                    message: format!(
                        "Both HTTP and gRPC health checks failed for {}: HTTP: {}, gRPC: {}",
                        config.url, http_err, grpc_err
                    ),
                });
            }
        };

        // Store connection mode in context
        context.set("connection_mode", connection_mode);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Connection issues are retryable
    }
}

/// Step 2: Discover metadata from worker
pub struct DiscoverMetadataStep;

#[async_trait]
impl StepExecutor for DiscoverMetadataStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let connection_mode: Arc<ConnectionMode> = context
            .get("connection_mode")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("connection_mode".to_string()))?;

        debug!(
            "Discovering metadata for {} ({:?})",
            config.url, *connection_mode
        );

        let (discovered_labels, detected_runtime) = match connection_mode.as_ref() {
            ConnectionMode::Http => {
                match get_server_info(&config.url, config.api_key.as_deref()).await {
                    Ok(server_info) => {
                        let mut labels = HashMap::new();
                        if let Some(model_path) = server_info.model_path.filter(|s| !s.is_empty()) {
                            labels.insert("model_path".to_string(), model_path);
                        }
                        if let Some(served_model_name) =
                            server_info.served_model_name.filter(|s| !s.is_empty())
                        {
                            labels.insert("served_model_name".to_string(), served_model_name);
                        }

                        Ok((labels, None))
                    }
                    Err(e) => Err(e),
                }
            }
            ConnectionMode::Grpc { .. } => {
                let runtime_type = config.runtime.as_deref();
                fetch_grpc_metadata(&config.url, runtime_type)
                    .await
                    .map(|(labels, runtime)| (labels, Some(runtime)))
            }
        }
        .unwrap_or_else(|e| {
            warn!("Failed to fetch metadata for {}: {}", config.url, e);
            (HashMap::new(), None)
        });

        debug!(
            "Discovered {} metadata labels for {}",
            discovered_labels.len(),
            config.url
        );

        // Store discovered labels and detected runtime in context
        context.set("discovered_labels", discovered_labels);
        if let Some(runtime) = detected_runtime {
            debug!("Detected runtime type: {}", runtime);
            context.set("detected_runtime_type", runtime);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Metadata discovery failures are retryable
    }
}

/// Step 2.5: Discover DP (Data Parallel) information (only for DP-aware workers)
pub struct DiscoverDPInfoStep;

#[async_trait]
impl StepExecutor for DiscoverDPInfoStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;

        // Skip DP discovery if not DP-aware
        if !config.dp_aware {
            debug!(
                "Worker {} is not DP-aware, skipping DP discovery",
                config.url
            );
            return Ok(StepResult::Success);
        }

        debug!("Discovering DP info for {} (DP-aware)", config.url);

        // Get DP info from worker
        let dp_info = get_dp_info(&config.url, config.api_key.as_deref())
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("discover_dp_info"),
                message: format!("Failed to get DP info: {}", e),
            })?;

        debug!(
            "Discovered DP size {} for {} (model: {})",
            dp_info.dp_size, config.url, dp_info.model_id
        );

        // Store DP info in context
        context.set("dp_info", dp_info);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // DP info discovery failures are retryable
    }
}

/// Step 3: Create worker object with merged configuration + metadata
pub struct CreateWorkerStep;

#[async_trait]
impl StepExecutor for CreateWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let connection_mode: Arc<ConnectionMode> = context
            .get("connection_mode")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("connection_mode".to_string()))?;
        let discovered_labels: Arc<HashMap<String, String>> = context
            .get("discovered_labels")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("discovered_labels".to_string()))?;

        // Check if worker already exists
        if app_context
            .worker_registry
            .get_by_url(&config.url)
            .is_some()
        {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!("Worker {} already exists", config.url),
            });
        }

        // Build labels from config
        let mut config_labels = config.labels.clone();
        if let Some(model_id) = &config.model_id {
            config_labels.insert("model_id".to_string(), model_id.clone());
        }
        if let Some(priority) = config.priority {
            config_labels.insert("priority".to_string(), priority.to_string());
        }
        if let Some(cost) = config.cost {
            config_labels.insert("cost".to_string(), cost.to_string());
        }
        if let Some(ref tokenizer_path) = config.tokenizer_path {
            config_labels.insert("tokenizer_path".to_string(), tokenizer_path.clone());
        }
        if let Some(ref reasoning_parser) = config.reasoning_parser {
            config_labels.insert("reasoning_parser".to_string(), reasoning_parser.clone());
        }
        if let Some(ref tool_parser) = config.tool_parser {
            config_labels.insert("tool_parser".to_string(), tool_parser.clone());
        }
        if let Some(ref chat_template) = config.chat_template {
            config_labels.insert("chat_template".to_string(), chat_template.clone());
        }

        // Merge: discovered labels first, then config labels (config takes precedence)
        let mut final_labels = discovered_labels.as_ref().clone();
        for (key, value) in &config_labels {
            final_labels.insert(key.clone(), value.clone());
        }

        // Derive model_id if not already set
        if !final_labels.contains_key("model_id") {
            let derived_model_id = final_labels
                .get("served_model_name")
                .or_else(|| final_labels.get("model_path"))
                .cloned();

            if let Some(model_id) = derived_model_id {
                debug!("Derived model_id from metadata: {}", model_id);
                final_labels.insert("model_id".to_string(), model_id);
            }
        }

        debug!(
            "Creating worker {} with {} discovered + {} config = {} final labels",
            config.url,
            discovered_labels.len(),
            config_labels.len(),
            final_labels.len()
        );

        // Parse worker type
        let worker_type = config
            .worker_type
            .as_ref()
            .map(|t| match t.as_str() {
                "prefill" => WorkerType::Prefill {
                    bootstrap_port: config.bootstrap_port,
                },
                "decode" => WorkerType::Decode,
                _ => WorkerType::Regular,
            })
            .unwrap_or(WorkerType::Regular);

        // Get detected runtime type (for gRPC workers)
        let runtime_type = if matches!(connection_mode.as_ref(), ConnectionMode::Grpc { .. }) {
            // Try to get detected runtime from context, fall back to config, or default to sglang
            if let Some(detected_runtime) = context.get::<String>("detected_runtime_type") {
                match detected_runtime.as_str() {
                    "vllm" => RuntimeType::Vllm,
                    _ => RuntimeType::Sglang,
                }
            } else if let Some(ref runtime) = config.runtime {
                match runtime.as_str() {
                    "vllm" => RuntimeType::Vllm,
                    _ => RuntimeType::Sglang,
                }
            } else {
                RuntimeType::Sglang
            }
        } else {
            RuntimeType::Sglang // Default for HTTP workers
        };

        // Build circuit breaker config
        let circuit_breaker_config = {
            let cfg = app_context.router_config.effective_circuit_breaker_config();
            CircuitBreakerConfig {
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
                timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
                window_duration: Duration::from_secs(cfg.window_duration_secs),
            }
        };

        // Build health config
        let health_config = {
            let cfg = &app_context.router_config.health_check;
            HealthConfig {
                timeout_secs: cfg.timeout_secs,
                check_interval_secs: cfg.check_interval_secs,
                endpoint: cfg.endpoint.clone(),
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
            }
        };

        // Normalize URL: add protocol prefix only if missing
        let normalized_url = if config.url.starts_with("http://")
            || config.url.starts_with("https://")
            || config.url.starts_with("grpc://")
        {
            // URL already has protocol, use as-is
            config.url.clone()
        } else {
            // Bare IP:port format, add appropriate protocol based on detected mode
            match connection_mode.as_ref() {
                ConnectionMode::Http => format!("http://{}", config.url),
                ConnectionMode::Grpc { .. } => format!("grpc://{}", config.url),
            }
        };

        if normalized_url != config.url {
            debug!(
                "Normalized worker URL: {} -> {} ({:?})",
                config.url,
                normalized_url,
                connection_mode.as_ref()
            );
        }

        // Handle DP-aware vs non-DP-aware workers
        if config.dp_aware {
            // DP-aware path: Create multiple workers (one per rank)
            let dp_info: Arc<DpInfo> = context
                .get("dp_info")
                .ok_or_else(|| WorkflowError::ContextValueNotFound("dp_info".to_string()))?;

            debug!(
                "Creating {} DP-aware workers for {} (dp_size: {})",
                dp_info.dp_size, config.url, dp_info.dp_size
            );

            let mut workers = Vec::new();
            for rank in 0..dp_info.dp_size {
                let mut builder =
                    DPAwareWorkerBuilder::new(normalized_url.clone(), rank, dp_info.dp_size)
                        .worker_type(worker_type.clone())
                        .connection_mode(connection_mode.as_ref().clone())
                        .runtime_type(runtime_type.clone())
                        .circuit_breaker_config(circuit_breaker_config.clone())
                        .health_config(health_config.clone());

                if let Some(ref api_key) = config.api_key {
                    builder = builder.api_key(api_key.clone());
                }

                if !final_labels.is_empty() {
                    builder = builder.labels(final_labels.clone());
                }

                let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
                worker.set_healthy(false);
                workers.push(worker);

                debug!(
                    "Created DP-aware worker {}@{}/{} ({:?})",
                    config.url,
                    rank,
                    dp_info.dp_size,
                    connection_mode.as_ref()
                );
            }

            // Store workers (plural) and labels in context
            context.set("workers", workers);
            context.set("labels", final_labels);

            Ok(StepResult::Success)
        } else {
            // Non-DP-aware path: Create single worker
            let mut builder = BasicWorkerBuilder::new(normalized_url.clone())
                .worker_type(worker_type)
                .connection_mode(connection_mode.as_ref().clone())
                .runtime_type(runtime_type)
                .circuit_breaker_config(circuit_breaker_config)
                .health_config(health_config);

            if let Some(ref api_key) = config.api_key {
                builder = builder.api_key(api_key.clone());
            }

            if !final_labels.is_empty() {
                builder = builder.labels(final_labels.clone());
            }

            let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
            worker.set_healthy(false);

            debug!(
                "Created worker object for {} ({:?}) with {} labels",
                config.url,
                connection_mode.as_ref(),
                final_labels.len()
            );

            // Store worker (singular) and labels in context
            context.set("worker", worker);
            context.set("labels", final_labels);

            Ok(StepResult::Success)
        }
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Worker creation failures are not retryable (likely config issues)
    }
}

/// Step 4: Register worker(s) in registry
pub struct RegisterWorkerStep;

#[async_trait]
impl StepExecutor for RegisterWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        // Check if we have multiple workers (DP-aware) or single worker
        if config.dp_aware {
            // DP-aware path: Register multiple workers
            let workers: Arc<Vec<Arc<dyn Worker>>> = context
                .get("workers")
                .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

            let mut worker_ids = Vec::new();
            for worker in workers.iter() {
                let worker_id = app_context.worker_registry.register(Arc::clone(worker));
                worker_ids.push(worker_id.clone());
                debug!(
                    "Registered DP-aware worker {} with ID {:?}",
                    config.url, worker_id
                );
            }

            context.set("worker_ids", worker_ids);
            Ok(StepResult::Success)
        } else {
            // Non-DP-aware path: Register single worker
            let worker: Arc<Arc<dyn Worker>> = context
                .get("worker")
                .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;

            let worker_id = app_context
                .worker_registry
                .register(Arc::clone(worker.as_ref()));

            debug!("Registered worker {} with ID {:?}", config.url, worker_id);
            context.set("worker_id", worker_id);

            Ok(StepResult::Success)
        }
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Registration failures are not retryable
    }
}

/// Step 5: Update policy registry with worker information
pub struct UpdatePoliciesStep;

#[async_trait]
impl StepExecutor for UpdatePoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let labels: Arc<HashMap<String, String>> = context
            .get("labels")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("labels".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let policy_hint = labels.get("policy").map(|s| s.as_str());

        // Check if we have multiple workers (DP-aware) or single worker
        if config.dp_aware {
            // DP-aware path: Update policies for multiple workers
            let workers: Arc<Vec<Arc<dyn Worker>>> = context
                .get("workers")
                .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

            // Get model_id from first worker (all DP workers have same model)
            let model_id = workers[0].model_id().to_string();

            // Notify policy registry for each worker
            for _ in 0..workers.len() {
                app_context
                    .policy_registry
                    .on_worker_added(&model_id, policy_hint);
            }

            // Initialize cache-aware policy if needed
            let all_workers = app_context.worker_registry.get_by_model_fast(&model_id);
            if let Some(policy) = app_context.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(&model_id, &all_workers);
                }
            }

            debug!(
                "Updated policies for {} DP-aware workers {} (model: {})",
                workers.len(),
                config.url,
                model_id
            );
        } else {
            // Non-DP-aware path: Update policy for single worker
            let worker: Arc<Arc<dyn Worker>> = context
                .get("worker")
                .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;

            let model_id = worker.model_id().to_string();

            // Notify policy registry
            app_context
                .policy_registry
                .on_worker_added(&model_id, policy_hint);

            // Initialize cache-aware policy if needed
            let all_workers = app_context.worker_registry.get_by_model_fast(&model_id);
            if let Some(policy) = app_context.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(&model_id, &all_workers);
                }
            }
            let prefill_workers = app_context.worker_registry.get_prefill_workers();
            let policy = app_context.policy_registry.get_prefill_policy();
            if policy.name() == "bucket" {
                app_context
                    .policy_registry
                    .init_pd_bucket_policies(&prefill_workers);
            }

            debug!(
                "Updated policies for worker {} (model: {})",
                config.url, model_id
            );
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Policy update failures are not retryable
    }
}

/// Step 6: Activate worker(s) by marking them as healthy
pub struct ActivateWorkerStep;

#[async_trait]
impl StepExecutor for ActivateWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;

        // Check if we have multiple workers (DP-aware) or single worker
        if config.dp_aware {
            // DP-aware path: Activate multiple workers
            let workers: Arc<Vec<Arc<dyn Worker>>> = context
                .get("workers")
                .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

            for worker in workers.iter() {
                worker.set_healthy(true);
            }

            debug!(
                "Activated {} DP-aware workers {} (marked as healthy)",
                workers.len(),
                config.url
            );
        } else {
            // Non-DP-aware path: Activate single worker
            let worker: Arc<Arc<dyn Worker>> = context
                .get("worker")
                .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;

            worker.set_healthy(true);

            info!("Activated worker {} (marked as healthy)", config.url);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Activation is just setting a flag, not retryable
    }
}

/// Create worker registration workflow definition
///
/// Note: Actual health check timeouts and retry attempts are configured per-worker
/// via WorkerConfigRequest (populated from router config). The timeouts and retry
/// policies here serve as workflow-level bounds to prevent infinite waiting.
///
/// # Arguments
/// * `router_config` - Router configuration containing health check settings
pub fn create_worker_registration_workflow(
    router_config: &crate::config::RouterConfig,
) -> WorkflowDefinition {
    // Use startup timeout from config for worker registration
    // This is separate from health_check.timeout_secs which is for individual HTTP requests
    let detect_timeout = Duration::from_secs(router_config.worker_startup_timeout_secs);

    // Calculate max_attempts to match the startup_timeout
    // With Linear backoff (increment 1s, max 5s):
    // - Attempts 1-5: 0s, 1s, 2s, 3s, 4s = 10s total
    // - Attempts 6+: 5s each
    // max_attempts = 5 + (timeout_seconds - 10) / 5
    // Use 90% of timeout to leave buffer for actual connection attempts
    let timeout_secs = detect_timeout.as_secs() as f64;
    let effective_timeout = timeout_secs * 0.9;
    let max_attempts = if effective_timeout > 10.0 {
        (5 + ((effective_timeout - 10.0) / 5.0).ceil() as u32).max(3)
    } else {
        3
    };

    WorkflowDefinition::new("worker_registration", "Worker Registration")
        .add_step(
            StepDefinition::new(
                "detect_connection_mode",
                "Detect Connection Mode",
                Arc::new(DetectConnectionModeStep),
            )
            .with_retry(RetryPolicy {
                max_attempts,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            // Workflow-level timeout uses configured health check timeout + buffer
            .with_timeout(detect_timeout)
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "discover_metadata",
                "Discover Metadata",
                Arc::new(DiscoverMetadataStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep), // Metadata discovery is optional
        )
        .add_step(
            StepDefinition::new(
                "discover_dp_info",
                "Discover DP Info",
                Arc::new(DiscoverDPInfoStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::FailWorkflow), // DP info is required for DP-aware workers
        )
        .add_step(
            StepDefinition::new("create_worker", "Create Worker", Arc::new(CreateWorkerStep))
                .with_timeout(Duration::from_secs(5))
                .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "register_worker",
                "Register Worker",
                Arc::new(RegisterWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "update_policies",
                "Update Policies",
                Arc::new(UpdatePoliciesStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep), // Policy updates are optional
        )
        .add_step(
            StepDefinition::new(
                "activate_worker",
                "Activate Worker",
                Arc::new(ActivateWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
}
