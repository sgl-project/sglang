//! Local worker registration workflow steps
//!
//! This workflow handles registration of local inference workers (SGLang, vLLM).
//! For external API endpoints (OpenAI, xAI, etc.), see external_worker_registration.rs.
//!
//! Workflow order:
//! 1. DetectConnectionMode - Probe HTTP and gRPC to determine connection mode
//! 2. DiscoverMetadata - Fetch metadata from /server_info or gRPC
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
        DPAwareWorkerBuilder, HealthConfig, ModelCard, RuntimeType, Worker, WorkerType,
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

/// Server information returned from /server_info endpoint
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

/// Model information returned from /model_info endpoint
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelInfo {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    is_generation: Option<bool>,
    /// HuggingFace model type string (e.g., "llama", "qwen2", "gpt_oss")
    model_type: Option<String>,
    /// Model architectures from HuggingFace config (e.g., ["LlamaForCausalLM"])
    architectures: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// Parse server info from JSON response
fn parse_server_info(json: Value) -> Result<ServerInfo, String> {
    serde_json::from_value(json).map_err(|e| format!("Failed to parse server info: {}", e))
}

/// Get server info from /server_info endpoint
async fn get_server_info(url: &str, api_key: Option<&str>) -> Result<ServerInfo, String> {
    let base_url = url.trim_end_matches('/');
    let server_info_url = format!("{}/server_info", base_url);

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

/// Get model info from /model_info endpoint
async fn get_model_info(url: &str, api_key: Option<&str>) -> Result<ModelInfo, String> {
    let base_url = url.trim_end_matches('/');
    let model_info_url = format!("{}/model_info", base_url);

    let mut req = HTTP_CLIENT.get(&model_info_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", model_info_url, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            model_info_url
        ));
    }

    response
        .json::<ModelInfo>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", model_info_url, e))
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

/// Strip protocol prefix from URL
fn strip_protocol(url: &str) -> String {
    url.trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_start_matches("grpc://")
        .to_string()
}

/// Try HTTP health check
async fn try_http_health_check(
    url: &str,
    timeout_secs: u64,
    client: &Client,
) -> Result<(), String> {
    let is_https = url.starts_with("https://");
    let protocol = if is_https { "https" } else { "http" };
    let clean_url = strip_protocol(url);
    let health_url = format!("{}://{}/health", protocol, clean_url);

    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {}", e))?;

    Ok(())
}

/// Perform gRPC health check with runtime type
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

/// Try gRPC health check (tries SGLang first, then vLLM if not specified)
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
            // Try SGLang first, then vLLM as fallback
            if let Ok(()) = do_grpc_health_check(&grpc_url, timeout_secs, "sglang").await {
                return Ok(());
            }
            do_grpc_health_check(&grpc_url, timeout_secs, "vllm")
                .await
                .map_err(|e| format!("gRPC failed (tried SGLang and vLLM): {}", e))
        }
    }
}

/// Fetch metadata from gRPC server
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

/// Fetch gRPC metadata (returns labels and detected runtime type)
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
            // Try SGLang first, then vLLM as fallback
            if let Ok(labels) = do_fetch_grpc_metadata(&grpc_url, "sglang").await {
                return Ok((labels, "sglang".to_string()));
            }
            let labels = do_fetch_grpc_metadata(&grpc_url, "vllm")
                .await
                .map_err(|e| format!("gRPC metadata failed (tried SGLang and vLLM): {}", e))?;
            Ok((labels, "vllm".to_string()))
        }
    }
}

/// Step 1: Detect connection mode by probing HTTP and gRPC
pub struct DetectConnectionModeStep;

#[async_trait]
impl StepExecutor for DetectConnectionModeStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

        debug!(
            "Detecting connection mode for {} (timeout: {}s, max_attempts: {})",
            config.url, config.health_check_timeout_secs, config.max_connection_attempts
        );

        // Try both protocols in parallel
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

        context.set("connection_mode", connection_mode);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

/// Step 2: Discover metadata from worker
pub struct DiscoverMetadataStep;

#[async_trait]
impl StepExecutor for DiscoverMetadataStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let connection_mode: Arc<ConnectionMode> = context.get_or_err("connection_mode")?;

        debug!(
            "Discovering metadata for {} ({:?})",
            config.url, *connection_mode
        );

        let (discovered_labels, detected_runtime) = match connection_mode.as_ref() {
            ConnectionMode::Http => {
                let mut labels = HashMap::new();

                // Fetch from /server_info for server-related metadata
                if let Ok(server_info) =
                    get_server_info(&config.url, config.api_key.as_deref()).await
                {
                    if let Some(model_path) = server_info.model_path.filter(|s| !s.is_empty()) {
                        labels.insert("model_path".to_string(), model_path);
                    }
                    if let Some(served_model_name) =
                        server_info.served_model_name.filter(|s| !s.is_empty())
                    {
                        labels.insert("served_model_name".to_string(), served_model_name);
                    }
                }

                // Fetch from /model_info for model-related metadata (model_type, architectures)
                if let Ok(model_info) = get_model_info(&config.url, config.api_key.as_deref()).await
                {
                    if let Some(model_type) = model_info.model_type.filter(|s| !s.is_empty()) {
                        labels.insert("model_type".to_string(), model_type);
                    }
                    if let Some(architectures) = model_info.architectures.filter(|a| !a.is_empty())
                    {
                        if let Ok(json_str) = serde_json::to_string(&architectures) {
                            labels.insert("architectures".to_string(), json_str);
                        }
                    }
                }

                Ok((labels, None))
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

        context.set("discovered_labels", discovered_labels);
        if let Some(runtime) = detected_runtime {
            debug!("Detected runtime type: {}", runtime);
            context.set("detected_runtime_type", runtime);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

/// Step 2.5: Discover DP (Data Parallel) information (only for DP-aware workers)
pub struct DiscoverDPInfoStep;

#[async_trait]
impl StepExecutor for DiscoverDPInfoStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;

        if !config.dp_aware {
            debug!(
                "Worker {} is not DP-aware, skipping DP discovery",
                config.url
            );
            return Ok(StepResult::Success);
        }

        debug!("Discovering DP info for {} (DP-aware)", config.url);

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

        context.set("dp_info", dp_info);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

/// Step 3: Create worker object with merged configuration + metadata
pub struct CreateWorkerStep;

#[async_trait]
impl StepExecutor for CreateWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let connection_mode: Arc<ConnectionMode> = context.get_or_err("connection_mode")?;
        let discovered_labels: Arc<HashMap<String, String>> =
            context.get_or_err("discovered_labels")?;

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
        if let Some(priority) = config.priority {
            config_labels.insert("priority".to_string(), priority.to_string());
        }
        if let Some(cost) = config.cost {
            config_labels.insert("cost".to_string(), cost.to_string());
        }

        // Merge: discovered labels first, then config labels (config takes precedence)
        let mut final_labels = discovered_labels.as_ref().clone();
        for (key, value) in &config_labels {
            final_labels.insert(key.clone(), value.clone());
        }

        // Determine model_id: config > served_model_name > model_path > "unknown"
        let model_id = config
            .model_id
            .clone()
            .or_else(|| final_labels.get("served_model_name").cloned())
            .or_else(|| final_labels.get("model_path").cloned())
            .unwrap_or_else(|| "unknown".to_string());

        if model_id != "unknown" {
            debug!("Using model_id: {}", model_id);
        }

        // Create ModelCard
        let model_card = {
            let mut card = ModelCard::new(&model_id);
            if let Some(ref tokenizer_path) = config.tokenizer_path {
                card = card.with_tokenizer_path(tokenizer_path.clone());
            }
            if let Some(ref reasoning_parser) = config.reasoning_parser {
                card = card.with_reasoning_parser(reasoning_parser.clone());
            }
            if let Some(ref tool_parser) = config.tool_parser {
                card = card.with_tool_parser(tool_parser.clone());
            }
            if let Some(ref chat_template) = config.chat_template {
                card = card.with_chat_template(chat_template.clone());
            }
            // Set HuggingFace model type from discovered labels
            if let Some(model_type_str) = final_labels.get("model_type") {
                card = card.with_hf_model_type(model_type_str.clone());
            }
            // Set architectures from discovered labels (JSON array string)
            if let Some(architectures_json) = final_labels.get("architectures") {
                if let Ok(architectures) = serde_json::from_str::<Vec<String>>(architectures_json) {
                    card = card.with_architectures(architectures);
                }
            }
            card
        };

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

        // Get runtime type (for gRPC workers)
        let runtime_type = if matches!(connection_mode.as_ref(), ConnectionMode::Grpc { .. }) {
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
            RuntimeType::Sglang
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
            config.url.clone()
        } else {
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
            let dp_info: Arc<DpInfo> = context.get_or_err("dp_info")?;

            debug!(
                "Creating {} DP-aware workers for {} (dp_size: {})",
                dp_info.dp_size, config.url, dp_info.dp_size
            );

            let mut workers = Vec::new();
            for rank in 0..dp_info.dp_size {
                let mut builder =
                    DPAwareWorkerBuilder::new(normalized_url.clone(), rank, dp_info.dp_size)
                        .model(model_card.clone())
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

            context.set("workers", workers);
            context.set("labels", final_labels);
        } else {
            let mut builder = BasicWorkerBuilder::new(normalized_url.clone())
                .model(model_card)
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

            context.set("worker", worker);
            context.set("labels", final_labels);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Step 4: Register worker(s) in registry
pub struct RegisterWorkerStep;

#[async_trait]
impl StepExecutor for RegisterWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

        if config.dp_aware {
            let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

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
        } else {
            let worker: Arc<Arc<dyn Worker>> = context.get_or_err("worker")?;

            let worker_id = app_context
                .worker_registry
                .register(Arc::clone(worker.as_ref()));
            debug!("Registered worker {} with ID {:?}", config.url, worker_id);
            context.set("worker_id", worker_id);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Step 5: Update policy registry with worker information
pub struct UpdatePoliciesStep;

#[async_trait]
impl StepExecutor for UpdatePoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let labels: Arc<HashMap<String, String>> = context.get_or_err("labels")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

        let policy_hint = labels.get("policy").map(|s| s.as_str());

        if config.dp_aware {
            let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

            let model_id = workers[0].model_id().to_string();

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
            let worker: Arc<Arc<dyn Worker>> = context.get_or_err("worker")?;

            let model_id = worker.model_id().to_string();

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

            // Initialize bucket policies for prefill workers
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
        false
    }
}

/// Step 6: Activate worker(s) by marking them as healthy
pub struct ActivateWorkerStep;

#[async_trait]
impl StepExecutor for ActivateWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;

        if config.dp_aware {
            let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

            for worker in workers.iter() {
                worker.set_healthy(true);
            }

            info!(
                "Activated {} DP-aware workers from {} (marked as healthy)",
                workers.len(),
                config.url
            );
        } else {
            let worker: Arc<Arc<dyn Worker>> = context.get_or_err("worker")?;

            worker.set_healthy(true);

            info!("Activated worker {} (marked as healthy)", config.url);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Create local worker registration workflow definition
pub fn create_worker_registration_workflow(
    router_config: &crate::config::RouterConfig,
) -> WorkflowDefinition {
    let detect_timeout = Duration::from_secs(router_config.worker_startup_timeout_secs);

    // Calculate max_attempts based on timeout
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
            .with_failure_action(FailureAction::ContinueNextStep),
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
            .with_failure_action(FailureAction::FailWorkflow),
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
            .with_failure_action(FailureAction::ContinueNextStep),
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
