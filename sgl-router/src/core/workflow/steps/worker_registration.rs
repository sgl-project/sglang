//! Worker registration workflow steps
//!
//! Each step is atomic and performs a single operation in the worker registration process.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde_json::Value;
use tracing::{info, warn};

use crate::{
    core::{
        workflow::*, BasicWorkerBuilder, CircuitBreakerConfig, ConnectionMode, HealthConfig,
        Worker, WorkerType,
    },
    grpc_client::SglangSchedulerClient,
    protocols::worker_spec::WorkerConfigRequest,
    server::AppContext,
};

/// Step 1: Create worker object from configuration
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

        // Build labels from config (matches WorkerManager::add_worker_from_config lines 590-612)
        let mut labels = config.labels.clone();
        if let Some(model_id) = &config.model_id {
            labels.insert("model_id".to_string(), model_id.clone());
        }
        if let Some(priority) = config.priority {
            labels.insert("priority".to_string(), priority.to_string());
        }
        if let Some(cost) = config.cost {
            labels.insert("cost".to_string(), cost.to_string());
        }
        if let Some(ref tokenizer_path) = config.tokenizer_path {
            labels.insert("tokenizer_path".to_string(), tokenizer_path.clone());
        }
        if let Some(ref reasoning_parser) = config.reasoning_parser {
            labels.insert("reasoning_parser".to_string(), reasoning_parser.clone());
        }
        if let Some(ref tool_parser) = config.tool_parser {
            labels.insert("tool_parser".to_string(), tool_parser.clone());
        }
        if let Some(ref chat_template) = config.chat_template {
            labels.insert("chat_template".to_string(), chat_template.clone());
        }

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

        // Parse connection mode
        let connection_mode = if config.url.starts_with("grpc://") {
            ConnectionMode::Grpc { port: None }
        } else {
            ConnectionMode::Http
        };

        // Convert circuit breaker config
        let circuit_breaker_config = {
            let cfg = app_context.router_config.effective_circuit_breaker_config();
            CircuitBreakerConfig {
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
                timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
                window_duration: Duration::from_secs(cfg.window_duration_secs),
            }
        };

        // Convert health config
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

        // Build worker with config labels
        let mut builder = BasicWorkerBuilder::new(config.url.clone())
            .worker_type(worker_type)
            .connection_mode(connection_mode)
            .circuit_breaker_config(circuit_breaker_config)
            .health_config(health_config);

        if let Some(ref api_key) = config.api_key {
            builder = builder.api_key(api_key.clone());
        }

        // Add config labels to worker
        if !labels.is_empty() {
            builder = builder.labels(labels.clone());
        }

        // Build and wrap in Arc
        let worker = Arc::new(builder.build()) as Arc<dyn Worker>;

        // Mark as unhealthy initially
        worker.set_healthy(false);

        info!(
            "Created worker object for {} with {} config labels",
            config.url,
            labels.len()
        );

        // Store worker and config labels in context
        context.set("worker", worker);
        context.set("config_labels", labels);

        Ok(StepResult::Success)
    }
}

/// Step 4: Discover metadata from worker endpoints
pub struct DiscoverMetadataStep;

#[async_trait]
impl StepExecutor for DiscoverMetadataStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let worker: Arc<Arc<dyn Worker>> = context
            .get("worker")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;
        let config_labels: Arc<HashMap<String, String>> = context
            .get("config_labels")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("config_labels".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        // Discover metadata from worker endpoints (HTTP and gRPC)
        let mut discovered_labels = HashMap::new();

        match worker.connection_mode() {
            ConnectionMode::Http => {
                // Try to get model_info via HTTP
                if let Ok(model_info) = get_model_info(&config.url, config.api_key.as_deref()).await
                {
                    if let Some(model_path) = model_info.get("model_path").and_then(|v| v.as_str())
                    {
                        if !model_path.is_empty() {
                            discovered_labels
                                .insert("model_path".to_string(), model_path.to_string());
                        }
                    }
                    if let Some(tokenizer_path) =
                        model_info.get("tokenizer_path").and_then(|v| v.as_str())
                    {
                        if !tokenizer_path.is_empty() {
                            discovered_labels
                                .insert("tokenizer_path".to_string(), tokenizer_path.to_string());
                        }
                    }
                    if let Some(served_model_name) =
                        model_info.get("served_model_name").and_then(|v| v.as_str())
                    {
                        if !served_model_name.is_empty() {
                            discovered_labels.insert(
                                "served_model_name".to_string(),
                                served_model_name.to_string(),
                            );
                        }
                    }
                }
            }
            ConnectionMode::Grpc { .. } => {
                // Try to get model_info via gRPC
                match SglangSchedulerClient::connect(&config.url).await {
                    Ok(client) => match client.get_model_info().await {
                        Ok(model_info) => {
                            if !model_info.model_path.is_empty() {
                                discovered_labels.insert(
                                    "model_path".to_string(),
                                    model_info.model_path.clone(),
                                );
                            }
                            if !model_info.tokenizer_path.is_empty() {
                                discovered_labels.insert(
                                    "tokenizer_path".to_string(),
                                    model_info.tokenizer_path.clone(),
                                );
                            }
                            if !model_info.served_model_name.is_empty() {
                                discovered_labels.insert(
                                    "served_model_name".to_string(),
                                    model_info.served_model_name.clone(),
                                );
                            }
                            if !model_info.weight_version.is_empty() {
                                discovered_labels.insert(
                                    "weight_version".to_string(),
                                    model_info.weight_version.clone(),
                                );
                            }
                            if !model_info.model_type.is_empty() {
                                discovered_labels.insert(
                                    "model_type".to_string(),
                                    model_info.model_type.clone(),
                                );
                            }
                            if !model_info.preferred_sampling_params.is_empty() {
                                discovered_labels.insert(
                                    "preferred_sampling_params".to_string(),
                                    model_info.preferred_sampling_params.clone(),
                                );
                            }
                            discovered_labels.insert(
                                "is_generation".to_string(),
                                model_info.is_generation.to_string(),
                            );
                            if model_info.max_context_length > 0 {
                                discovered_labels.insert(
                                    "max_context_length".to_string(),
                                    model_info.max_context_length.to_string(),
                                );
                            }
                            if model_info.max_req_input_len > 0 {
                                discovered_labels.insert(
                                    "max_req_input_len".to_string(),
                                    model_info.max_req_input_len.to_string(),
                                );
                            }
                            if model_info.vocab_size > 0 {
                                discovered_labels.insert(
                                    "vocab_size".to_string(),
                                    model_info.vocab_size.to_string(),
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Failed to fetch gRPC model info from {}: {}", config.url, e);
                        }
                    },
                    Err(e) => {
                        warn!("Failed to connect to gRPC worker {}: {}", config.url, e);
                    }
                }
            }
        }

        // Merge: discovered labels first, then config labels (config takes precedence)
        // This matches WorkerManager::create_basic_worker lines 973-978
        let mut final_labels = discovered_labels.clone();
        for (key, value) in config_labels.as_ref() {
            final_labels.insert(key.clone(), value.clone());
        }

        info!(
            "Discovered {} metadata labels, merged with {} config labels = {} final labels for {}",
            discovered_labels.len(),
            config_labels.len(),
            final_labels.len(),
            config.url
        );

        // Recreate worker with merged labels
        let circuit_breaker_config = {
            let cfg = app_context.router_config.effective_circuit_breaker_config();
            CircuitBreakerConfig {
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
                timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
                window_duration: Duration::from_secs(cfg.window_duration_secs),
            }
        };

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

        let mut builder = BasicWorkerBuilder::new(config.url.clone())
            .worker_type(worker.worker_type().clone())
            .connection_mode(worker.connection_mode().clone())
            .circuit_breaker_config(circuit_breaker_config)
            .health_config(health_config);

        if let Some(ref api_key) = config.api_key {
            builder = builder.api_key(api_key.clone());
        }

        // Add merged labels
        if !final_labels.is_empty() {
            builder = builder.labels(final_labels.clone());
        }

        // Rebuild worker
        let new_worker = Arc::new(builder.build()) as Arc<dyn Worker>;

        // Preserve healthy status
        new_worker.set_healthy(worker.is_healthy());

        info!("Recreated worker {} with merged labels", config.url);

        // Replace worker in context with new one
        context.set("worker", new_worker);
        context.set("labels", final_labels);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        // Metadata discovery failures are retryable
        true
    }
}

/// Step 3: Register worker in registry (after health check passes)
pub struct RegisterWorkerStep;

#[async_trait]
impl StepExecutor for RegisterWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let worker: Arc<Arc<dyn Worker>> = context
            .get("worker")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        // Register worker in registry (clone the inner Arc<dyn Worker>)
        let worker_id = app_context
            .worker_registry
            .register(Arc::clone(worker.as_ref()));

        info!("Registered worker {} with ID {:?}", config.url, worker_id);

        // Store worker_id in context
        context.set("worker_id", worker_id);

        Ok(StepResult::Success)
    }
}

/// Step 2: Perform health check validation (optional)
pub struct HealthCheckStep;

#[async_trait]
impl StepExecutor for HealthCheckStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        // Check if health check should be skipped
        let skip_health_check = context
            .get::<bool>("skip_health_check")
            .map(|v| *v)
            .unwrap_or(false);

        if skip_health_check {
            info!("Skipping health check as requested");
            return Ok(StepResult::Skip);
        }

        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let worker: Arc<Arc<dyn Worker>> = context
            .get("worker")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;

        // Use worker's check_health_async which handles both HTTP and gRPC
        let timeout_duration =
            Duration::from_secs(app_context.router_config.worker_startup_timeout_secs);
        let start_time = std::time::Instant::now();

        loop {
            if start_time.elapsed() > timeout_duration {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("health_check"),
                    message: format!(
                        "Health check timeout after {}s for {}",
                        timeout_duration.as_secs(),
                        config.url
                    ),
                });
            }

            match worker.check_health_async().await {
                Ok(()) => {
                    let mode = match worker.connection_mode() {
                        ConnectionMode::Http => "HTTP",
                        ConnectionMode::Grpc { .. } => "gRPC",
                    };
                    info!("Health check passed for {} ({})", config.url, mode);
                    return Ok(StepResult::Success);
                }
                Err(e) => {
                    warn!("Health check attempt failed for {}: {}", config.url, e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        // Health checks should be retried
        true
    }
}

/// Step 6: Activate worker by marking it healthy
pub struct ActivateWorkerStep;

#[async_trait]
impl StepExecutor for ActivateWorkerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let worker: Arc<Arc<dyn Worker>> = context
            .get("worker")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;

        // Mark worker as healthy
        worker.set_healthy(true);

        info!("Activated worker {} (marked as healthy)", config.url);

        Ok(StepResult::Success)
    }
}

/// Step 5: Update policy registry
pub struct UpdatePoliciesStep;

#[async_trait]
impl StepExecutor for UpdatePoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context
            .get("worker_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker_config".to_string()))?;
        let worker: Arc<Arc<dyn Worker>> = context
            .get("worker")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("worker".to_string()))?;
        let labels: Arc<HashMap<String, String>> = context
            .get("labels")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("labels".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let model_id = worker.model_id().to_string();
        let policy_hint = labels.get("policy").map(|s| s.as_str());

        // Notify policy registry
        app_context
            .policy_registry
            .on_worker_added(&model_id, policy_hint);

        // Initialize cache-aware policy if needed
        let workers = app_context.worker_registry.get_by_model_fast(&model_id);
        if let Some(policy) = app_context.policy_registry.get_policy(&model_id) {
            if policy.name() == "cache_aware" {
                app_context
                    .policy_registry
                    .init_cache_aware_policy(&model_id, &workers);
            }
        }

        info!(
            "Updated policies for worker {} (model: {})",
            config.url, model_id
        );

        Ok(StepResult::Success)
    }
}

/// Helper function to get model info from worker
async fn get_model_info(url: &str, api_key: Option<&str>) -> Result<Value, String> {
    static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
        Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client")
    });

    let base_url = url.trim_end_matches('/');
    let model_info_url = format!("{}/get_model_info", base_url);

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

    let json = response
        .json::<Value>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", model_info_url, e))?;

    Ok(json)
}

/// Create the worker registration workflow definition
pub fn create_worker_registration_workflow() -> WorkflowDefinition {
    let default_retry = RetryPolicy {
        max_attempts: 3,
        backoff: BackoffStrategy::Exponential {
            base: Duration::from_secs(1),
            max: Duration::from_secs(10),
        },
    };

    let health_check_retry = RetryPolicy {
        max_attempts: 5,
        backoff: BackoffStrategy::Exponential {
            base: Duration::from_secs(2),
            max: Duration::from_secs(30),
        },
    };

    WorkflowDefinition::new("worker_registration", "Worker Registration and Activation")
        .with_default_timeout(Duration::from_secs(60))
        .add_step(
            StepDefinition::new(
                "create_worker",
                "Create worker object from configuration",
                Arc::new(CreateWorkerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "health_check",
                "Validate worker health (optional)",
                Arc::new(HealthCheckStep),
            )
            .with_retry(health_check_retry)
            .with_timeout(Duration::from_secs(30))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "register_worker",
                "Register worker in registry",
                Arc::new(RegisterWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "discover_metadata",
                "Discover worker metadata from endpoints",
                Arc::new(DiscoverMetadataStep),
            )
            .with_retry(default_retry.clone())
            .with_timeout(Duration::from_secs(15))
            .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(
            StepDefinition::new(
                "update_policies",
                "Update policy registry",
                Arc::new(UpdatePoliciesStep),
            )
            .with_retry(default_retry)
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(
            StepDefinition::new(
                "activate_worker",
                "Mark worker as healthy",
                Arc::new(ActivateWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
}
