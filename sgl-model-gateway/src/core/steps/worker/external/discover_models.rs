//! Model discovery step for external API endpoints.

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::core::{
    model_card::{ModelCard, ProviderType},
    model_type::ModelType,
    steps::workflow_data::ExternalWorkerWorkflowData,
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

/// OpenAI /v1/models response format.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
    #[serde(default)]
    pub object: String,
}

/// Individual model information from /v1/models.
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

    // Image generation models
    if id_lower.starts_with("dall-e")
        || id_lower.starts_with("sora")
        || (id_lower.contains("image") && !id_lower.contains("vision"))
    {
        return ModelType::IMAGE_MODEL;
    }

    // Audio models
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

    // Vision LLM
    if id_lower.contains("vision") || id_lower.contains("4o") {
        return ModelType::VISION_LLM;
    }

    // Reasoning models
    if id_lower.starts_with("o1") || id_lower.starts_with("o3") {
        return ModelType::REASONING_LLM;
    }

    // Default to standard LLM
    ModelType::LLM
}

/// Infer provider type from model ID string.
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

/// Fetch models from /v1/models endpoint.
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

    let model_cards = group_models_into_cards(models_response.data);

    debug!(
        "Grouped into {} model cards with aliases",
        model_cards.len()
    );

    Ok(model_cards)
}

/// Step 1: Discover models from external /v1/models endpoint.
pub struct DiscoverModelsStep;

#[async_trait]
impl StepExecutor<ExternalWorkerWorkflowData> for DiscoverModelsStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<ExternalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let has_api_key = config.api_key.as_ref().is_some_and(|key| !key.is_empty());

        debug!("Discovering models from external endpoint {}", config.url);

        match fetch_models(&config.url, config.api_key.as_deref()).await {
            Ok(model_cards) if !model_cards.is_empty() => {
                info!(
                    "Discovered {} models from {}: {:?}",
                    model_cards.len(),
                    config.url,
                    model_cards.iter().map(|c| &c.id).collect::<Vec<_>>()
                );
                context.data.model_cards = model_cards;
            }
            // A keyed upstream that advertises nothing is a misconfiguration, as
            // before.
            Ok(_) if has_api_key => {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("discover_models"),
                    message: format!("No models discovered from {}", config.url),
                });
            }
            // Keyless upstreams stay usable when the endpoint is reachable but
            // publishes an empty list; see the fallback below.
            Ok(_) => {
                info!(
                    "No models advertised by {} - using wildcard mode (accepts any model). \
                     User's Authorization header will be forwarded to backend.",
                    config.url
                );
            }
            Err(e) if has_api_key => {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("discover_models"),
                    message: format!("Failed to discover models from {}: {}", config.url, e),
                });
            }
            // Without a key there is nothing to authenticate discovery with, so a
            // failure is not fatal: the endpoint may simply not implement
            // /v1/models. Registering a wildcard worker keeps the previous
            // behavior (accept any model, forward the caller's Authorization).
            Err(e) => {
                info!(
                    "Could not read models from {} ({}) - using wildcard mode (accepts any \
                     model). User's Authorization header will be forwarded to backend.",
                    config.url, e
                );
            }
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::protocols::worker_spec::WorkerConfigRequest;

    const TWO_MODELS: &str = r#"{"object":"list","data":[{"id":"gpt-4o"},{"id":"gpt-4o-2024-05-13"},{"id":"whisper-1"}]}"#;
    const NO_MODELS: &str = r#"{"object":"list","data":[]}"#;

    /// Stand-in for an upstream's `/v1/models` endpoint.
    async fn spawn_models_endpoint(status: u16, body: &'static str) -> String {
        use axum::{http::StatusCode, routing::get, Router};

        let app = Router::new().route(
            "/v1/models",
            get(move || async move { (StatusCode::from_u16(status).unwrap(), body) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{}", addr)
    }

    fn workflow_data(
        url: &str,
        api_key: Option<&str>,
    ) -> WorkflowContext<ExternalWorkerWorkflowData> {
        WorkflowContext::new(
            WorkflowInstanceId::new(),
            ExternalWorkerWorkflowData {
                config: WorkerConfigRequest {
                    url: url.to_string(),
                    api_key: api_key.map(str::to_string),
                    model_id: None,
                    worker_type: None,
                    priority: None,
                    cost: None,
                    runtime: None,
                    labels: HashMap::new(),
                    bootstrap_port: None,
                    tokenizer_path: None,
                    reasoning_parser: None,
                    tool_parser: None,
                    chat_template: None,
                    health_check_timeout_secs: 30,
                    health_check_interval_secs: 60,
                    health_success_threshold: 2,
                    health_failure_threshold: 3,
                    disable_health_check: true,
                    max_connection_attempts: 20,
                    dp_aware: false,
                },
                model_cards: Vec::new(),
                workers: None,
                labels: HashMap::new(),
                app_context: None,
                actual_workers: None,
            },
        )
    }

    fn discovered_ids(context: &WorkflowContext<ExternalWorkerWorkflowData>) -> Vec<&str> {
        let mut ids: Vec<&str> = context
            .data
            .model_cards
            .iter()
            .map(|card| card.id.as_str())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// A keyless upstream that publishes its list is discovered, instead of being
    /// registered as a single wildcard worker.
    #[tokio::test]
    async fn keyless_upstream_is_discovered() {
        let url = spawn_models_endpoint(200, TWO_MODELS).await;
        let mut context = workflow_data(&url, None);

        let result = DiscoverModelsStep.execute(&mut context).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert_eq!(discovered_ids(&context), vec!["gpt-4o", "whisper-1"]);
        let gpt4o = context
            .data
            .model_cards
            .iter()
            .find(|card| card.id == "gpt-4o")
            .expect("gpt-4o card");
        assert_eq!(gpt4o.aliases, vec!["gpt-4o-2024-05-13"]);
    }

    /// Reachable endpoint, empty list: keep the previous wildcard behavior.
    #[tokio::test]
    async fn keyless_upstream_with_empty_list_stays_wildcard() {
        let url = spawn_models_endpoint(200, NO_MODELS).await;
        let mut context = workflow_data(&url, None);

        let result = DiscoverModelsStep.execute(&mut context).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert!(discovered_ids(&context).is_empty(), "wildcard worker");
    }

    /// An upstream that does not implement `/v1/models` must not break a keyless
    /// registration.
    #[tokio::test]
    async fn keyless_upstream_without_v1_models_stays_wildcard() {
        let url = spawn_models_endpoint(404, "not found").await;
        let mut context = workflow_data(&url, None);

        let result = DiscoverModelsStep.execute(&mut context).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert!(discovered_ids(&context).is_empty(), "wildcard worker");
    }

    /// Keyed behavior is unchanged: a failed discovery is still fatal.
    #[tokio::test]
    async fn keyed_upstream_still_fails_when_discovery_fails() {
        let url = spawn_models_endpoint(404, "not found").await;
        let mut context = workflow_data(&url, Some("secret"));

        let result = DiscoverModelsStep.execute(&mut context).await;

        assert!(
            result.is_err(),
            "keyed discovery failure stays a step error"
        );
    }

    /// Keyed upstreams that advertise nothing still fail, as before.
    #[tokio::test]
    async fn keyed_upstream_with_empty_list_still_fails() {
        let url = spawn_models_endpoint(200, NO_MODELS).await;
        let mut context = workflow_data(&url, Some("secret"));

        let result = DiscoverModelsStep.execute(&mut context).await;

        assert!(result.is_err(), "keyed empty discovery stays a step error");
    }
}
