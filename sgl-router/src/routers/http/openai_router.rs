//! OpenAI router implementation

use crate::config::CircuitBreakerConfig;
use crate::core::{CircuitBreaker, CircuitBreakerConfig as CoreCircuitBreakerConfig};
use crate::protocols::{
    generate::GenerateRequest,
    openai::{chat::ChatCompletionRequest, completions::CompletionRequest},
};
use async_openai::{config::OpenAIConfig, Client};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::{
    any::Any,
    sync::atomic::{AtomicBool, Ordering},
};

/// Router for OpenAI backend
#[derive(Debug)]
pub struct OpenAIRouter {
    /// OpenAI client for direct API calls
    openai_client: Client<OpenAIConfig>,
    /// Model name
    model: String,
    /// Base URL for identification
    base_url: String,
    /// Circuit breaker
    circuit_breaker: CircuitBreaker,
    /// Health status
    healthy: AtomicBool,
}

impl OpenAIRouter {
    /// Create a new OpenAI router
    pub async fn new(
        api_key: Option<String>,
        model: String,
        base_url: String,
        circuit_breaker_config: Option<CircuitBreakerConfig>,
    ) -> Result<Self, String> {
        // Configure OpenAI client
        let final_api_key =
            api_key.unwrap_or_else(|| std::env::var("OPENAI_API_KEY").unwrap_or_default());

        if final_api_key.is_empty() {
            return Err("No OpenAI API key provided. Use --api-key or set OPENAI_API_KEY environment variable.".to_string());
        }

        // Append /v1 to base URL for async-openai
        let api_base_url = format!("{}/v1", base_url.trim_end_matches('/'));

        let config = OpenAIConfig::new()
            .with_api_base(&api_base_url)
            .with_api_key(final_api_key.clone());

        let openai_client = Client::with_config(config);

        // Convert circuit breaker config
        let core_cb_config = circuit_breaker_config
            .map(|cb| CoreCircuitBreakerConfig {
                failure_threshold: cb.failure_threshold,
                success_threshold: cb.success_threshold,
                timeout_duration: std::time::Duration::from_secs(cb.timeout_duration_secs),
                window_duration: std::time::Duration::from_secs(cb.window_duration_secs),
            })
            .unwrap_or_default();

        let circuit_breaker = CircuitBreaker::with_config(core_cb_config);

        Ok(Self {
            openai_client,
            model,
            base_url,
            circuit_breaker,
            healthy: AtomicBool::new(true),
        })
    }
}

#[async_trait]
impl super::super::WorkerManagement for OpenAIRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("Cannot add workers to OpenAI router".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {
        // No-op for OpenAI router
    }

    fn get_worker_urls(&self) -> Vec<String> {
        vec![self.base_url.clone()]
    }
}

#[async_trait]
impl super::super::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        if self.healthy.load(Ordering::Acquire) && self.circuit_breaker.can_execute() {
            (StatusCode::OK, "OK").into_response()
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "Not healthy").into_response()
        }
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // For OpenAI, health_generate is the same as health
        self.health(_req).await
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let info = serde_json::json!({
            "router_type": "openai",
            "model": &self.model,
            "workers": 1,
            "base_url": &self.base_url
        });
        (StatusCode::OK, info.to_string()).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        let models = serde_json::json!({
            "object": "list",
            "data": [{
                "id": &self.model,
                "object": "model",
                "created": 0,
                "owned_by": "openai"
            }]
        });
        (StatusCode::OK, models.to_string()).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        let info = serde_json::json!({
            "id": &self.model,
            "object": "model",
            "created": 0,
            "owned_by": "openai"
        });
        (StatusCode::OK, info.to_string()).into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
    ) -> Response {
        // Generate endpoint is SGLang-specific, not supported for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not supported for OpenAI backend",
        )
            .into_response()
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
    ) -> Response {
        if !self.circuit_breaker.can_execute() {
            return (StatusCode::SERVICE_UNAVAILABLE, "Circuit breaker open").into_response();
        }

        // Use the model from the request or fall back to router's default model
        let model_name = if body.model.is_empty() {
            self.model.as_str()
        } else {
            body.model.as_str()
        };

        // For now, just extract text from the first message
        // TODO: Properly convert all messages to OpenAI format
        let user_content = match body.messages.first() {
            Some(crate::protocols::openai::chat::types::ChatMessage::User {
                content: crate::protocols::openai::chat::types::UserMessageContent::Text(text),
                ..
            }) => text.clone(),
            Some(crate::protocols::openai::chat::types::ChatMessage::System {
                content, ..
            }) => content.clone(),
            _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    "First message must be a text user or system message",
                )
                    .into_response();
            }
        };

        // Use the content for OpenAI request
        let user_content_for_openai = user_content;

        // Create a simple user message for OpenAI
        let openai_messages = vec![async_openai::types::ChatCompletionRequestMessage::User(
            async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                .content(user_content_for_openai)
                .build()
                .unwrap(),
        )];

        // Build OpenAI request - start simple and add essential parameters
        let openai_request = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model(model_name)
            .messages(openai_messages)
            .max_tokens(body.max_tokens.unwrap_or(100)) // Default to 100 if not specified
            .temperature(body.temperature.unwrap_or(1.0)) // Default to 1.0 if not specified
            .build();

        match openai_request {
            Ok(req) => match self.openai_client.chat().create(req).await {
                Ok(response) => {
                    self.circuit_breaker.record_success();
                    (
                        StatusCode::OK,
                        serde_json::to_string(&response).unwrap_or_default(),
                    )
                        .into_response()
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    let error_msg = format!("OpenAI API error: {}", e);
                    (StatusCode::INTERNAL_SERVER_ERROR, error_msg).into_response()
                }
            },
            Err(e) => {
                let error_msg = format!("Invalid OpenAI request: {}", e);
                (StatusCode::BAD_REQUEST, error_msg).into_response()
            }
        }
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
    ) -> Response {
        // Completion endpoint not implemented for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented for OpenAI backend",
        )
            .into_response()
    }

    async fn flush_cache(&self) -> Response {
        (StatusCode::OK, "Cache flushed").into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        let loads = serde_json::json!({
            "loads": {
                "openai": 0
            }
        });
        (StatusCode::OK, loads.to_string()).into_response()
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }

    fn readiness(&self) -> Response {
        if self.healthy.load(Ordering::Acquire) && self.circuit_breaker.can_execute() {
            (StatusCode::OK, "Ready").into_response()
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "Not ready").into_response()
        }
    }
}
