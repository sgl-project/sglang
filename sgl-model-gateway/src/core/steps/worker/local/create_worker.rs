//! Local worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::debug;

use super::discover_dp::DpInfo;
use crate::{
    app_context::AppContext,
    core::{
        circuit_breaker::CircuitBreakerConfig,
        model_card::ModelCard,
        worker::{HealthConfig, RuntimeType, WorkerType},
        BasicWorkerBuilder, ConnectionMode, DPAwareWorkerBuilder, Worker, UNKNOWN_MODEL_ID,
    },
    protocols::worker_spec::WorkerConfigRequest,
    workflow::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step 3: Create worker object(s) with merged configuration + metadata.
///
/// This step:
/// 1. Merges discovered labels with config labels
/// 2. Determines the model ID from various sources
/// 3. Creates ModelCard with metadata
/// 4. Builds worker(s) - either single worker or multiple DP-aware workers
/// 5. Outputs unified `workers: Vec<Arc<dyn Worker>>` for downstream steps
pub struct CreateLocalWorkerStep;

#[async_trait]
impl StepExecutor for CreateLocalWorkerStep {
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

        // Determine model_id: config > served_model_name > model_path > UNKNOWN_MODEL_ID
        let model_id = config
            .model_id
            .clone()
            .or_else(|| final_labels.get("served_model_name").cloned())
            .or_else(|| final_labels.get("model_path").cloned())
            .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

        if model_id != UNKNOWN_MODEL_ID {
            debug!("Using model_id: {}", model_id);
        }

        // Create ModelCard
        let model_card = build_model_card(&model_id, &config, &final_labels);

        debug!(
            "Creating worker {} with {} discovered + {} config = {} final labels",
            config.url,
            discovered_labels.len(),
            config_labels.len(),
            final_labels.len()
        );

        // Parse worker type
        let worker_type = parse_worker_type(&config);

        // Get runtime type (for gRPC workers)
        let runtime_type = determine_runtime_type(&connection_mode, context, &config);

        // Build circuit breaker config
        let circuit_breaker_config = build_circuit_breaker_config(&app_context);

        // Build health config
        let health_config = build_health_config(&app_context);

        // Normalize URL
        let normalized_url = normalize_url(&config.url, &connection_mode);

        if normalized_url != config.url {
            debug!(
                "Normalized worker URL: {} -> {} ({:?})",
                config.url,
                normalized_url,
                connection_mode.as_ref()
            );
        }

        // Create workers - always output as Vec for unified downstream handling
        let workers = if config.dp_aware {
            create_dp_aware_workers(
                context,
                &normalized_url,
                model_card,
                worker_type,
                &connection_mode,
                runtime_type,
                circuit_breaker_config,
                health_config,
                &config,
                &final_labels,
            )?
        } else {
            create_single_worker(
                &normalized_url,
                model_card,
                worker_type,
                &connection_mode,
                runtime_type,
                circuit_breaker_config,
                health_config,
                &config,
                &final_labels,
            )
        };

        context.set("workers", workers);
        context.set("labels", final_labels);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

fn build_model_card(
    model_id: &str,
    config: &WorkerConfigRequest,
    labels: &HashMap<String, String>,
) -> ModelCard {
    let mut card = ModelCard::new(model_id);

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
    if let Some(model_type_str) = labels.get("model_type") {
        card = card.with_hf_model_type(model_type_str.clone());
    }
    if let Some(architectures_json) = labels.get("architectures") {
        if let Ok(architectures) = serde_json::from_str::<Vec<String>>(architectures_json) {
            card = card.with_architectures(architectures);
        }
    }

    // Parse classification model id2label mapping
    // The proto field is id2label_json: JSON string like {"0": "negative", "1": "positive"}
    if let Some(id2label_json) = labels.get("id2label_json") {
        if !id2label_json.is_empty() {
            // Parse JSON: keys are string indices, values are label names
            if let Ok(string_map) = serde_json::from_str::<HashMap<String, String>>(id2label_json) {
                // Convert string keys ("0", "1") to u32 keys (0, 1)
                let id2label: HashMap<u32, String> = string_map
                    .into_iter()
                    .filter_map(|(k, v)| k.parse::<u32>().ok().map(|idx| (idx, v)))
                    .collect();

                if !id2label.is_empty() {
                    card = card.with_id2label(id2label);
                    debug!("Parsed id2label with {} classes", card.num_labels);
                }
            }
        }
    }
    // Fallback: if num_labels is set but id2label wasn't parsed, create default labels
    // Match logic in serving_classify.py::_get_id2label_mapping
    else if let Some(num_labels_str) = labels.get("num_labels") {
        if let Ok(num_labels) = num_labels_str.parse::<u32>() {
            if num_labels > 0 {
                // Create default mapping: {0: "LABEL_0", 1: "LABEL_1", ...}
                let id2label: HashMap<u32, String> = (0..num_labels)
                    .map(|i| (i, format!("LABEL_{}", i)))
                    .collect();
                card = card.with_id2label(id2label);
                debug!("Created default id2label with {} classes", num_labels);
            }
        }
    }

    card
}

fn parse_worker_type(config: &WorkerConfigRequest) -> WorkerType {
    config
        .worker_type
        .as_ref()
        .map(|t| match t.as_str() {
            "prefill" => WorkerType::Prefill {
                bootstrap_port: config.bootstrap_port,
            },
            "decode" => WorkerType::Decode,
            _ => WorkerType::Regular,
        })
        .unwrap_or(WorkerType::Regular)
}

fn determine_runtime_type(
    connection_mode: &ConnectionMode,
    context: &WorkflowContext,
    config: &WorkerConfigRequest,
) -> RuntimeType {
    if !matches!(connection_mode, ConnectionMode::Grpc { .. }) {
        return RuntimeType::Sglang;
    }

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
}

fn build_circuit_breaker_config(app_context: &AppContext) -> CircuitBreakerConfig {
    let cfg = app_context.router_config.effective_circuit_breaker_config();
    CircuitBreakerConfig {
        failure_threshold: cfg.failure_threshold,
        success_threshold: cfg.success_threshold,
        timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
        window_duration: Duration::from_secs(cfg.window_duration_secs),
    }
}

fn build_health_config(app_context: &AppContext) -> HealthConfig {
    let cfg = &app_context.router_config.health_check;
    HealthConfig {
        timeout_secs: cfg.timeout_secs,
        check_interval_secs: cfg.check_interval_secs,
        endpoint: cfg.endpoint.clone(),
        failure_threshold: cfg.failure_threshold,
        success_threshold: cfg.success_threshold,
    }
}

fn normalize_url(url: &str, connection_mode: &ConnectionMode) -> String {
    if url.starts_with("http://") || url.starts_with("https://") || url.starts_with("grpc://") {
        url.to_string()
    } else {
        match connection_mode {
            ConnectionMode::Http => format!("http://{}", url),
            ConnectionMode::Grpc { .. } => format!("grpc://{}", url),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn create_dp_aware_workers(
    context: &WorkflowContext,
    normalized_url: &str,
    model_card: ModelCard,
    worker_type: WorkerType,
    connection_mode: &ConnectionMode,
    runtime_type: RuntimeType,
    circuit_breaker_config: CircuitBreakerConfig,
    health_config: HealthConfig,
    config: &WorkerConfigRequest,
    final_labels: &HashMap<String, String>,
) -> Result<Vec<Arc<dyn Worker>>, WorkflowError> {
    let dp_info: Arc<DpInfo> = context.get_or_err("dp_info")?;

    debug!(
        "Creating {} DP-aware workers for {} (dp_size: {})",
        dp_info.dp_size, normalized_url, dp_info.dp_size
    );

    let mut workers = Vec::with_capacity(dp_info.dp_size);
    for rank in 0..dp_info.dp_size {
        let mut builder =
            DPAwareWorkerBuilder::new(normalized_url.to_string(), rank, dp_info.dp_size)
                .model(model_card.clone())
                .worker_type(worker_type.clone())
                .connection_mode(connection_mode.clone())
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
            normalized_url, rank, dp_info.dp_size, connection_mode
        );
    }

    Ok(workers)
}

#[allow(clippy::too_many_arguments)]
fn create_single_worker(
    normalized_url: &str,
    model_card: ModelCard,
    worker_type: WorkerType,
    connection_mode: &ConnectionMode,
    runtime_type: RuntimeType,
    circuit_breaker_config: CircuitBreakerConfig,
    health_config: HealthConfig,
    config: &WorkerConfigRequest,
    final_labels: &HashMap<String, String>,
) -> Vec<Arc<dyn Worker>> {
    let mut builder = BasicWorkerBuilder::new(normalized_url.to_string())
        .model(model_card)
        .worker_type(worker_type)
        .connection_mode(connection_mode.clone())
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
        normalized_url,
        connection_mode,
        final_labels.len()
    );

    vec![worker]
}
