//! External worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::{debug, info};

use crate::{
    app_context::AppContext,
    core::{
        circuit_breaker::CircuitBreakerConfig,
        model_card::ModelCard,
        worker::{HealthConfig, RuntimeType, WorkerType},
        BasicWorkerBuilder, ConnectionMode, Worker,
    },
    protocols::worker_spec::WorkerConfigRequest,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Normalize URL for external APIs (ensure https://).
fn normalize_external_url(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{}", url)
    }
}

/// Step 2: Create worker objects for each discovered model.
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

        // Build labels from config
        let mut labels: HashMap<String, String> = config.labels.clone();
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
        false
    }
}
