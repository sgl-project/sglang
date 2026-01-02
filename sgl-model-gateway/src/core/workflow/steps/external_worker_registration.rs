//! External worker registration workflow steps
//!
//! This workflow handles registration of external API endpoints (OpenAI, xAI, Anthropic, etc.)
//!
//! Key features:
//! - Fetches models from /v1/models endpoint
//! - Groups dated model variants under base model names (e.g., gpt-4o, gpt-4o-2024-08-06)
//! - Infers ModelType from model ID patterns (LLM, embedding, image gen, audio, etc.)
//!
//! Workflow order:
//! 1. DiscoverModels - Fetch available models from /v1/models endpoint
//! 2. CreateExternalWorkers - Build worker objects for each discovered model
//! 3. RegisterWorkers - Register workers in registry
//! 4. UpdatePolicies - Update policy registry
//! 5. ActivateWorkers - Mark workers as healthy

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, info};

use crate::{
    app_context::AppContext,
    core::{
        model_card::{ModelCard, ProviderType},
        model_type::ModelType,
        workflow::*,
        BasicWorkerBuilder, CircuitBreakerConfig, ConnectionMode, HealthConfig, RuntimeType,
        Worker, WorkerType,
    },
    protocols::worker_spec::WorkerConfigRequest,
};

// HTTP client for API calls
static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
});

// Regex to strip date suffix: -YYYY-MM-DD or -YYYY-MM
static DATE_SUFFIX_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"-\d{4}-\d{2}(-\d{2})?$").expect("Invalid date regex"));

// ============================================================================
// Model Discovery Types
// ============================================================================

/// OpenAI /v1/models response format
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
    #[serde(default)]
    pub object: String,
}

/// Individual model information from /v1/models
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub owned_by: Option<String>,
}

// ============================================================================
// Model Discovery Functions
// ============================================================================

/// Group models by base name (stripping date suffixes) and create ModelCards with aliases.
///
/// # Example
/// Input:  `["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"]`
/// Output: `ModelCard { id: "gpt-4o", aliases: ["gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"] }`
pub fn group_models_into_cards(models: Vec<ModelInfo>) -> Vec<ModelCard> {
    // Group model IDs by base name (with date stripped)
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for model in &models {
        let base = DATE_SUFFIX_PATTERN.replace(&model.id, "").to_string();
        groups.entry(base).or_default().push(model.id.clone());
    }

    // Create ModelCard for each group
    groups
        .into_values()
        .map(|mut variants| {
            // Sort: shortest first (base name), then alphabetically
            variants.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

            let primary_id = variants.remove(0); // shortest = primary ID
            let aliases = variants; // rest = aliases

            let model_type = infer_model_type_from_id(&primary_id);
            let provider = infer_provider_from_id(&primary_id);

            let mut card = ModelCard::new(&primary_id)
                .with_aliases(aliases)
                .with_model_type(model_type);

            if let Some(p) = provider {
                card = card.with_provider(p);
            }

            card
        })
        .collect()
}

/// Infer ModelType from model ID string.
///
/// This function analyzes the model ID to determine what type of model it is
/// (LLM, embedding, image generation, audio, etc.).
pub fn infer_model_type_from_id(id: &str) -> ModelType {
    let id_lower = id.to_lowercase();

    // Embedding models
    if id_lower.contains("embed") || id_lower.contains("ada-002") {
        return ModelType::EMBED_MODEL;
    }

    // Rerank models
    if id_lower.contains("rerank") {
        return ModelType::RERANK_MODEL;
    }

    // Image generation models (DALL-E, Sora, gpt-image)
    if id_lower.starts_with("dall-e")
        || id_lower.starts_with("sora")
        || (id_lower.contains("image") && !id_lower.contains("vision"))
    {
        return ModelType::IMAGE_MODEL;
    }

    // Audio models (TTS, Whisper, realtime, audio)
    if id_lower.starts_with("tts")
        || id_lower.starts_with("whisper")
        || id_lower.contains("audio")
        || id_lower.contains("realtime")
        || id_lower.contains("transcribe")
    {
        return ModelType::AUDIO_MODEL;
    }

    // Moderation models
    if id_lower.contains("moderation") {
        return ModelType::MODERATION_MODEL;
    }

    // Vision LLM (models with vision capability)
    // gpt-4o, gpt-4-vision, etc.
    if id_lower.contains("vision") || id_lower.contains("4o") {
        return ModelType::VISION_LLM;
    }

    // Reasoning models (o1, o3, etc.)
    if id_lower.starts_with("o1") || id_lower.starts_with("o3") {
        return ModelType::REASONING_LLM;
    }

    // Default to standard LLM
    ModelType::LLM
}

/// Infer provider type from model ID string.
///
/// Returns `None` for models that don't clearly indicate a provider.
fn infer_provider_from_id(id: &str) -> Option<ProviderType> {
    let id_lower = id.to_lowercase();

    // OpenAI models
    if id_lower.starts_with("gpt")
        || id_lower.starts_with("o1")
        || id_lower.starts_with("o3")
        || id_lower.starts_with("dall-e")
        || id_lower.starts_with("whisper")
        || id_lower.starts_with("tts")
        || id_lower.starts_with("text-embedding")
        || id_lower.starts_with("babbage")
        || id_lower.starts_with("davinci")
        || id_lower.contains("omni")
    {
        return Some(ProviderType::OpenAI);
    }

    // xAI/Grok models
    if id_lower.starts_with("grok") {
        return Some(ProviderType::XAI);
    }

    // Anthropic Claude models
    if id_lower.starts_with("claude") {
        return Some(ProviderType::Anthropic);
    }

    // Google Gemini models
    if id_lower.starts_with("gemini") {
        return Some(ProviderType::Gemini);
    }

    None
}

// ============================================================================
// Workflow Steps
// ============================================================================

/// Step 1: Discover models from external /v1/models endpoint
pub struct DiscoverModelsStep;

#[async_trait]
impl StepExecutor for DiscoverModelsStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;

        // If no API key is provided, skip model discovery and use wildcard mode.
        if config.api_key.as_ref().is_none_or(|k| k.is_empty()) {
            info!(
                "No API key provided for {} - using wildcard mode (accepts any model). \
                 User's Authorization header will be forwarded to backend.",
                config.url
            );
            context.set::<Vec<ModelCard>>("model_cards", vec![]);
            return Ok(StepResult::Success);
        }

        debug!("Discovering models from external endpoint {}", config.url);

        let model_cards = fetch_models(&config.url, config.api_key.as_deref())
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("discover_models"),
                message: format!("Failed to discover models from {}: {}", config.url, e),
            })?;

        if model_cards.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("discover_models"),
                message: format!("No models discovered from {}", config.url),
            });
        }

        info!(
            "Discovered {} models from {}: {:?}",
            model_cards.len(),
            config.url,
            model_cards.iter().map(|c| &c.id).collect::<Vec<_>>()
        );

        context.set("model_cards", model_cards);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Network issues are retryable
    }
}

/// Fetch models from /v1/models endpoint
async fn fetch_models(url: &str, api_key: Option<&str>) -> Result<Vec<ModelCard>, String> {
    let base_url = url.trim_end_matches('/');
    let models_url = format!("{}/v1/models", base_url);

    let mut req = HTTP_CLIENT.get(&models_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", models_url, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            models_url
        ));
    }

    let models_response: ModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse models response: {}", e))?;

    debug!(
        "Fetched {} raw models from {}",
        models_response.data.len(),
        url
    );

    // Group models into cards (e.g., gpt-4o-2024-08-06 → gpt-4o with aliases)
    let model_cards = group_models_into_cards(models_response.data);

    debug!(
        "Grouped into {} model cards with aliases",
        model_cards.len()
    );

    Ok(model_cards)
}

/// Step 2: Create worker objects for each discovered model
pub struct CreateExternalWorkersStep;

#[async_trait]
impl StepExecutor for CreateExternalWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let model_cards: Arc<Vec<ModelCard>> = context.get_or_err("model_cards")?;

        // Build configs from router settings
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

        // Build labels
        let mut labels = config.labels.clone();
        if let Some(priority) = config.priority {
            labels.insert("priority".to_string(), priority.to_string());
        }
        if let Some(cost) = config.cost {
            labels.insert("cost".to_string(), cost.to_string());
        }

        // Normalize URL (ensure https:// for external APIs)
        let normalized_url = normalize_external_url(&config.url);

        let mut workers = Vec::new();

        // Handle wildcard mode: create a single worker with empty models list
        if model_cards.is_empty() {
            debug!("Creating wildcard worker (no models) for {}", config.url);

            let mut builder = BasicWorkerBuilder::new(normalized_url.clone())
                .models(vec![]) // Empty models = accepts any model
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::External)
                .circuit_breaker_config(circuit_breaker_config.clone())
                .health_config(health_config.clone());

            if let Some(ref api_key) = config.api_key {
                builder = builder.api_key(api_key.clone());
            }

            if !labels.is_empty() {
                builder = builder.labels(labels.clone());
            }

            let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
            worker.set_healthy(false);

            info!(
                "Created wildcard worker at {} (accepts any model, user auth forwarded)",
                normalized_url
            );

            workers.push(worker);
        } else {
            debug!(
                "Creating {} external workers for {}",
                model_cards.len(),
                config.url
            );

            // Create a worker for each model
            for model_card in model_cards.iter() {
                let mut builder = BasicWorkerBuilder::new(normalized_url.clone())
                    .model(model_card.clone())
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Http)
                    .runtime_type(RuntimeType::External)
                    .circuit_breaker_config(circuit_breaker_config.clone())
                    .health_config(health_config.clone());

                if let Some(ref api_key) = config.api_key {
                    builder = builder.api_key(api_key.clone());
                }

                if !labels.is_empty() {
                    builder = builder.labels(labels.clone());
                }

                let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
                worker.set_healthy(false);

                debug!(
                    "Created external worker for model {} at {}",
                    model_card.id, normalized_url
                );

                workers.push(worker);
            }

            info!(
                "Created {} external workers from {}",
                workers.len(),
                config.url
            );
        }

        context.set("workers", workers);
        context.set("labels", labels);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Creation failures are config issues
    }
}

/// Normalize URL for external APIs (ensure https://)
fn normalize_external_url(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{}", url)
    }
}

/// Step 3: Register workers in registry
pub struct RegisterExternalWorkersStep;

#[async_trait]
impl StepExecutor for RegisterExternalWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

        let mut worker_ids = Vec::new();
        for worker in workers.iter() {
            let worker_id = app_context.worker_registry.register(Arc::clone(worker));
            worker_ids.push(worker_id.clone());
            debug!(
                "Registered external worker {} (model: {}) with ID {:?}",
                config.url,
                worker.model_id(),
                worker_id
            );
        }

        context.set("worker_ids", worker_ids);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Step 4: Update policy registry
pub struct UpdateExternalPoliciesStep;

#[async_trait]
impl StepExecutor for UpdateExternalPoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let labels: Arc<HashMap<String, String>> = context.get_or_err("labels")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

        let policy_hint = labels.get("policy").map(|s| s.as_str());

        // Each external worker has a different model_id
        for worker in workers.iter() {
            let model_id = worker.model_id().to_string();
            app_context
                .policy_registry
                .on_worker_added(&model_id, policy_hint);
        }

        debug!(
            "Updated policies for {} external workers from {}",
            workers.len(),
            config.url
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Step 5: Activate workers by marking them healthy
pub struct ActivateExternalWorkersStep;

#[async_trait]
impl StepExecutor for ActivateExternalWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<WorkerConfigRequest> = context.get_or_err("worker_config")?;
        let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

        for worker in workers.iter() {
            worker.set_healthy(true);
        }

        info!(
            "Activated {} external workers from {} (marked as healthy)",
            workers.len(),
            config.url
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

// ============================================================================
// Workflow Definition
// ============================================================================

/// Create external worker registration workflow definition
pub fn create_external_worker_registration_workflow() -> WorkflowDefinition {
    WorkflowDefinition::new(
        "external_worker_registration",
        "External Worker Registration",
    )
    .add_step(
        StepDefinition::new(
            "discover_models",
            "Discover Models",
            Arc::new(DiscoverModelsStep),
        )
        .with_retry(RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(1),
                max: Duration::from_secs(10),
            },
        })
        .with_timeout(Duration::from_secs(30))
        .with_failure_action(FailureAction::FailWorkflow),
    )
    .add_step(
        StepDefinition::new(
            "create_workers",
            "Create Workers",
            Arc::new(CreateExternalWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow),
    )
    .add_step(
        StepDefinition::new(
            "register_workers",
            "Register Workers",
            Arc::new(RegisterExternalWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow),
    )
    .add_step(
        StepDefinition::new(
            "update_policies",
            "Update Policies",
            Arc::new(UpdateExternalPoliciesStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::ContinueNextStep),
    )
    .add_step(
        StepDefinition::new(
            "activate_workers",
            "Activate Workers",
            Arc::new(ActivateExternalWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow),
    )
}
