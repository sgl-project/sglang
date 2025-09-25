// gRPC Router Implementation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::{debug, error, info, warn};

use crate::config::types::RetryConfig;
use crate::core::{
    BasicWorkerBuilder, CircuitBreakerConfig, HealthConfig, WorkerRegistry, WorkerType,
};
use crate::grpc::{proto, SglangSchedulerClient};
use crate::metrics::RouterMetrics;
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::protocols::spec::{ChatCompletionRequest, ResponseFormat, StringOrArray};
use crate::reasoning_parser::ParserFactory;
use crate::routers::RouterTrait;
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::ParserRegistry;
use uuid::Uuid;

use crate::tokenizer::chat_template::{ChatTemplateContentFormat, ChatTemplateParams};
use serde_json::Value;

// Data structures for processing
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<proto::MultimodalInputs>,
    pub stop_sequences: Option<StringOrArray>,
}

/// gRPC router implementation for SGLang
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcRouter {
    /// Centralized worker registry
    worker_registry: Arc<WorkerRegistry>,
    /// Centralized policy registry
    policy_registry: Arc<PolicyRegistry>,
    /// Load balancing policy
    policy: Arc<dyn LoadBalancingPolicy>,
    /// Tokenizer for handling text encoding/decoding
    tokenizer: Arc<dyn Tokenizer>,
    /// Reasoning parser factory for structured reasoning outputs
    reasoning_parser_factory: ParserFactory,
    /// Tool parser registry for function/tool calls
    tool_parser_registry: &'static ParserRegistry,
    /// Configuration
    timeout_secs: u64,
    interval_secs: u64,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
}

impl GrpcRouter {
    /// Create a new gRPC router
    pub async fn new(
        worker_urls: Vec<String>,
        policy: Arc<dyn LoadBalancingPolicy>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Update metrics
        RouterMetrics::set_active_workers(worker_urls.len());

        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC router requires tokenizer".to_string())?
            .clone();
        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_registry = ctx
            .tool_parser_registry
            .ok_or_else(|| "gRPC router requires tool parser registry".to_string())?;

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Create gRPC clients for each worker
        let mut grpc_clients = HashMap::new();
        for url in &worker_urls {
            match SglangSchedulerClient::connect(url).await {
                Ok(client) => {
                    grpc_clients.insert(url.clone(), client);
                    info!("Connected to gRPC worker at {}", url);
                }
                Err(e) => {
                    warn!("Failed to connect to gRPC worker at {}: {}", url, e);
                    // Continue with other workers
                }
            }
        }

        if grpc_clients.is_empty() {
            return Err("Failed to connect to any gRPC workers".to_string());
        }

        // Get registries from context
        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        // Create Worker trait objects with gRPC connection mode and register them
        for url in &worker_urls {
            if let Some(client) = grpc_clients.remove(url) {
                let worker = BasicWorkerBuilder::new(url.clone())
                    .worker_type(WorkerType::Regular)
                    .connection_mode(crate::core::ConnectionMode::Grpc { port: None })
                    .circuit_breaker_config(core_cb_config.clone())
                    .health_config(HealthConfig {
                        timeout_secs: ctx.router_config.health_check.timeout_secs,
                        check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                        endpoint: ctx.router_config.health_check.endpoint.clone(),
                        failure_threshold: ctx.router_config.health_check.failure_threshold,
                        success_threshold: ctx.router_config.health_check.success_threshold,
                    })
                    .grpc_client(client)
                    .build();

                // Register worker in the centralized registry
                worker_registry.register(Arc::new(worker));
            } else {
                warn!("No gRPC client for worker {}, skipping", url);
            }
        }

        // Get only gRPC workers from registry for policy initialization
        let workers = worker_registry.get_workers_filtered(
            None, // any model
            Some(WorkerType::Regular),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false, // include unhealthy workers during initialization
        );

        // Initialize policy with workers if needed
        if let Some(cache_aware) = policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&workers);
        }

        // No need for local health checkers - WorkerRegistry handles health checking

        Ok(GrpcRouter {
            worker_registry,
            policy_registry,
            policy,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_registry,
            timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            interval_secs: ctx.router_config.worker_startup_check_interval_secs,
            dp_aware: ctx.router_config.dp_aware,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
        })
    }

    // ============ Chat Implementation ============

    /// Main route_chat implementation
    async fn route_chat_impl(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {:?}",
            model_id
        );

        // Step 1: Select worker (fail fast if no workers available)
        let worker = match self.select_worker_for_request(model_id, None) {
            Some(w) => w,
            None => {
                warn!("No available workers for model: {:?}", model_id);
                return (StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response();
            }
        };

        debug!("Selected worker: {}", worker.url());

        // Step 2: Get gRPC client for worker (fail fast if can't connect)
        let client = match self.get_or_create_grpc_client(worker.url()).await {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to get gRPC client: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to get gRPC client: {}", e),
                )
                    .into_response();
            }
        };

        // Step 3: Process messages and apply chat template
        let processed_messages = match self.process_chat_messages(body) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!("Failed to process chat messages: {}", e);
                return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
            }
        };

        // Step 4: Tokenize the processed text
        let encoding = match self.tokenizer.encode(&processed_messages.text) {
            Ok(encoding) => encoding,
            Err(e) => {
                error!("Tokenization failed: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Tokenization failed: {}", e),
                )
                    .into_response();
            }
        };

        let token_ids = encoding.token_ids().to_vec();
        debug!("Tokenized {} tokens from input", token_ids.len());

        // Step 5: Build tool constraints if needed
        let tool_call_constraint = if let Some(tools) = &body.tools {
            self.generate_tool_constraints(tools, &body.tool_choice, &body.model)
        } else {
            None
        };

        // Step 6: Build SamplingParams for gRPC
        let sampling_params = match self.build_grpc_sampling_params(body, tool_call_constraint) {
            Ok(params) => params,
            Err(e) => {
                error!("Failed to build sampling parameters: {}", e);
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid sampling parameters: {}", e),
                )
                    .into_response();
            }
        };

        // Step 7: Create GenerateRequest
        let grpc_request = proto::GenerateRequest {
            request_id: format!("chatcmpl-{}", Uuid::new_v4()),
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_messages.text.clone(),
                input_ids: token_ids.into_iter().map(|id| id as i32).collect(),
            }),
            mm_inputs: processed_messages.multimodal_inputs,
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: -1,
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            return_hidden_states: body.return_hidden_states,
            ..Default::default()
        };

        // Step 8: Handle streaming vs non-streaming
        if body.stream {
            self.handle_streaming_chat(client, grpc_request, body).await
        } else {
            self.handle_non_streaming_chat(client, grpc_request, body)
                .await
        }
    }

    // ============ Helper Methods ============
    /// Select a worker for the request
    fn select_worker_for_request(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
    ) -> Option<Arc<dyn crate::core::Worker>> {
        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_id,
            Some(WorkerType::Regular),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false, // get all workers, we'll filter by is_available() next
        );

        // Filter by availability (health + circuit breaker)
        let available: Vec<Arc<dyn crate::core::Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Select worker using the policy
        let idx = policy.select_worker(&available, text)?;
        Some(available[idx].clone())
    }

    /// Process chat messages and apply template
    fn process_chat_messages(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ProcessedMessages, String> {
        // Use the tokenizer's chat template - we require HuggingFace tokenizer for gRPC
        let formatted_text = if let Some(hf_tokenizer) =
            self.tokenizer
                .as_any()
                .downcast_ref::<crate::tokenizer::HuggingFaceTokenizer>()
        {
            // Get content format and transform messages accordingly
            let content_format = hf_tokenizer.chat_template_content_format();
            let mut transformed_messages =
                Self::process_content_format(&request.messages, content_format)?;

            // Process tool call arguments in assistant messages
            Self::process_tool_call_arguments(&mut transformed_messages)?;

            // Convert tools to JSON values for template processing
            let tools_json: Option<Vec<serde_json::Value>> = request
                .tools
                .as_ref()
                .map(|tools| {
                    tools
                        .iter()
                        .map(serde_json::to_value)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()
                .map_err(|e| format!("Failed to serialize tools: {}", e))?;

            // Build template kwargs, merging reasoning_effort if present
            let mut combined_template_kwargs = std::collections::HashMap::new();

            // Add reasoning_effort if present (like Python does)
            if let Some(reasoning_effort) = &request.reasoning_effort {
                combined_template_kwargs.insert(
                    "reasoning_effort".to_string(),
                    serde_json::Value::String(reasoning_effort.clone()),
                );
            }

            // Add any additional template kwargs from request
            if let Some(template_kwargs) = &request.chat_template_kwargs {
                for (key, value) in template_kwargs {
                    combined_template_kwargs.insert(key.clone(), value.clone());
                }
            }

            let final_template_kwargs = if combined_template_kwargs.is_empty() {
                None
            } else {
                Some(&combined_template_kwargs)
            };

            let params = ChatTemplateParams {
                add_generation_prompt: true,
                continue_final_message: request.continue_final_message,
                tools: tools_json.as_deref(),
                template_kwargs: final_template_kwargs,
                ..Default::default()
            };

            // Handle assistant prefix for continue_final_message
            let assistant_prefix = if request.continue_final_message
                && !transformed_messages.is_empty()
                && transformed_messages
                    .last()
                    .and_then(|msg| msg.get("role"))
                    .and_then(|v| v.as_str())
                    == Some("assistant")
            {
                // Pop the last message to handle it separately
                let last_msg = transformed_messages.pop().unwrap();
                last_msg
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            };

            // Apply chat template with the (now possibly shorter) list of messages
            let rendered = hf_tokenizer
                .apply_chat_template(&transformed_messages, params)
                .map_err(|e| format!("Failed to apply chat template: {}", e))?;

            // Append assistant prefix if we have one
            if let Some(prefix) = assistant_prefix {
                format!("{}{}", rendered, prefix)
            } else {
                rendered
            }
        } else {
            return Err(
                "gRPC router requires HuggingFace tokenizer with chat template support".to_string(),
            );
        };

        // Placeholder for multimodal inputs
        let multimodal_inputs = None;

        Ok(ProcessedMessages {
            text: formatted_text,
            multimodal_inputs,
            stop_sequences: request.stop.clone(),
        })
    }

    /// Process messages based on content format for ANY message type
    fn process_content_format(
        messages: &[crate::protocols::spec::ChatMessage],
        content_format: crate::tokenizer::chat_template::ChatTemplateContentFormat,
    ) -> Result<Vec<serde_json::Value>, String> {
        messages
            .iter()
            .map(|message| {
                let mut message_json = serde_json::to_value(message)
                    .map_err(|e| format!("Failed to serialize message: {}", e))?;

                if let Some(obj) = message_json.as_object_mut() {
                    if let Some(content_value) = obj.get_mut("content") {
                        Self::transform_content_field(content_value, content_format);
                    }
                }

                Ok(message_json)
            })
            .collect()
    }

    /// Transform a single content field based on content format
    fn transform_content_field(
        content_value: &mut Value,
        content_format: ChatTemplateContentFormat,
    ) {
        let Some(content_array) = content_value.as_array() else {
            return; // Not multimodal, keep as-is
        };

        match content_format {
            ChatTemplateContentFormat::String => {
                // Extract and join text parts only
                let text_parts: Vec<String> = content_array
                    .iter()
                    .filter_map(|part| {
                        part.as_object()?
                            .get("type")?
                            .as_str()
                            .filter(|&t| t == "text")
                            .and_then(|_| part.as_object()?.get("text")?.as_str())
                            .map(String::from)
                    })
                    .collect();

                if !text_parts.is_empty() {
                    *content_value = Value::String(text_parts.join(" "));
                }
            }
            ChatTemplateContentFormat::OpenAI => {
                // Replace media URLs with simple type placeholders
                let processed_parts: Vec<Value> = content_array
                    .iter()
                    .map(|part| {
                        part.as_object()
                            .and_then(|obj| obj.get("type")?.as_str())
                            .and_then(|type_str| match type_str {
                                "image_url" => Some(serde_json::json!({"type": "image"})),
                                "video_url" => Some(serde_json::json!({"type": "video"})),
                                "audio_url" => Some(serde_json::json!({"type": "audio"})),
                                _ => None,
                            })
                            .unwrap_or_else(|| part.clone())
                    })
                    .collect();

                *content_value = Value::Array(processed_parts);
            }
        }
    }

    /// Process tool call arguments in messages
    /// Per Transformers docs, tool call arguments in assistant messages should be dicts
    fn process_tool_call_arguments(messages: &mut [serde_json::Value]) -> Result<(), String> {
        for msg in messages {
            // Early return if not assistant message
            let role = msg.get("role").and_then(|v| v.as_str());
            if role != Some("assistant") {
                continue;
            }

            // Early return if no tool_calls
            let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut())
            else {
                continue;
            };

            // Process each tool call's arguments
            for call in tool_calls {
                let Some(function) = call.get_mut("function") else {
                    continue;
                };
                let Some(args) = function.get_mut("arguments") else {
                    continue;
                };
                let Some(args_str) = args.as_str() else {
                    continue;
                };

                // Parse JSON string to object (like Python json.loads)
                match serde_json::from_str::<serde_json::Value>(args_str) {
                    Ok(parsed) => *args = parsed,
                    Err(e) => {
                        return Err(format!(
                            "Failed to parse tool call arguments as JSON: '{}'. Error: {}",
                            args_str, e
                        ))
                    }
                }
            }
        }
        Ok(())
    }

    /// Build gRPC SamplingParams from OpenAI request
    fn build_grpc_sampling_params(
        &self,
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = self.extract_stop_strings(request);

        // Handle max tokens: prefer max_completion_tokens (new) over max_tokens (deprecated)
        // If neither is specified, use None to let the backend decide the default
        #[allow(deprecated)]
        let max_new_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .map(|v| v as i32);

        // Handle skip_special_tokens: set to false if tools are present and tool_choice is not "none"
        let skip_special_tokens = if request.tools.is_some() {
            match &request.tool_choice {
                Some(crate::protocols::spec::ToolChoice::Value(
                    crate::protocols::spec::ToolChoiceValue::None,
                )) => request.skip_special_tokens,
                Some(_) => false, // tool_choice is not "none"
                None => false, // TODO: this assumes tool_choice defaults to "auto" when tools present
            }
        } else {
            request.skip_special_tokens
        };

        #[allow(deprecated)]
        Ok(proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(-1),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_new_tokens,
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens,
            n: request.n.unwrap_or(1) as i32,
            constraint: self.build_constraint(request, tool_call_constraint)?,
            ..Default::default()
        })
    }

    /// Extract stop strings from request
    fn extract_stop_strings(&self, request: &ChatCompletionRequest) -> Vec<String> {
        match &request.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        }
    }

    /// Build constraint for structured generation
    fn build_constraint(
        &self,
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();

        if let Some(ResponseFormat::JsonSchema { json_schema }) = &request.response_format {
            let schema_str = serde_json::to_string(&json_schema.schema)
                .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
            constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
        }

        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
                ebnf.clone(),
            ));
        }

        if let Some(regex) = &request.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }

        // Handle tool call constraint
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if !constraints.is_empty() {
                return Err("Constrained decoding is not compatible with tool calls.".to_string());
            }
            let tool_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {}", constraint_type)),
            };
            constraints.push(tool_constraint);
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple constraints are not allowed.".to_string()),
        }
    }

    /// Generate tool constraints for structured generation
    fn generate_tool_constraints(
        &self,
        _tools: &[crate::protocols::spec::Tool],
        _tool_choice: &Option<crate::protocols::spec::ToolChoice>,
        model: &str,
    ) -> Option<(String, String)> {
        let _parser = self.tool_parser_registry.get_parser(model)?;
        // TODO: Implement actual constraint generation logic
        // For now, return None as this is placeholder implementation
        None
    }

    /// Get or create a gRPC client for the worker
    async fn get_or_create_grpc_client(
        &self,
        worker_url: &str,
    ) -> Result<SglangSchedulerClient, String> {
        // TODO: move to worker
        debug!("Creating new gRPC client for worker: {}", worker_url);
        SglangSchedulerClient::connect(worker_url)
            .await
            .map_err(|e| format!("Failed to connect to gRPC server: {}", e))
    }

    /// Placeholder for streaming handler (to be implemented in Phase 2)
    async fn handle_streaming_chat(
        &self,
        _client: SglangSchedulerClient,
        _request: proto::GenerateRequest,
        _original_request: &ChatCompletionRequest,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Streaming not yet implemented").into_response()
    }

    /// Placeholder for non-streaming handler (to be implemented in Phase 3)
    async fn handle_non_streaming_chat(
        &self,
        _client: SglangSchedulerClient,
        _request: proto::GenerateRequest,
        _original_request: &ChatCompletionRequest,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Non-streaming not yet implemented",
        )
            .into_response()
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.worker_registry.stats();
        f.debug_struct("GrpcRouter")
            .field("workers_count", &stats.total_workers)
            .field("timeout_secs", &self.timeout_secs)
            .field("interval_secs", &self.interval_secs)
            .field("dp_aware", &self.dp_aware)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // TODO: Implement actual generation test for gRPC
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not yet implemented for gRPC",
        )
            .into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &crate::protocols::spec::ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::spec::{ChatMessage, ContentPart, ImageUrl, UserMessageContent};
    use crate::tokenizer::chat_template::ChatTemplateContentFormat;
    use serde_json::json;

    #[test]
    fn test_transform_messages_string_format() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Hello".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "World".to_string(),
                },
            ]),
            name: None,
        }];

        let result =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::String)
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should flatten multimodal content to text only
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Hello World"
        );
        assert_eq!(transformed_message["role"].as_str().unwrap(), "user");
    }

    #[test]
    fn test_transform_messages_openai_format() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this image:".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let result =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::OpenAI)
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should replace media URLs with simple type placeholders
        let content_array = transformed_message["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);

        // Text part should remain unchanged
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[0]["text"], "Describe this image:");

        // Image part should be replaced with simple type placeholder
        assert_eq!(content_array[1], json!({"type": "image"}));
    }

    #[test]
    fn test_transform_messages_simple_string_content() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text("Simple text message".to_string()),
            name: None,
        }];

        let result =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::String)
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Simple string content should remain unchanged
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Simple text message"
        );
    }

    #[test]
    fn test_transform_messages_assistant_message() {
        let messages = vec![ChatMessage::Assistant {
            role: "assistant".to_string(),
            content: Some("Assistant response".to_string()),
            name: None,
            tool_calls: None,
            function_call: None,
            reasoning_content: None,
        }];

        let result =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::String)
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        assert_eq!(transformed_message["role"].as_str().unwrap(), "assistant");
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Assistant response"
        );
    }

    #[test]
    fn test_transform_messages_multiple_messages() {
        let messages = vec![
            ChatMessage::System {
                role: "system".to_string(),
                content: "System prompt".to_string(),
                name: None,
            },
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "User message".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: None,
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::String)
                .unwrap();

        assert_eq!(result.len(), 2);

        // System message should remain unchanged
        assert_eq!(result[0]["role"].as_str().unwrap(), "system");
        assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");

        // User message should be flattened to text only
        assert_eq!(result[1]["role"].as_str().unwrap(), "user");
        assert_eq!(result[1]["content"].as_str().unwrap(), "User message");
    }

    #[test]
    fn test_transform_messages_empty_text_parts() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            }]),
            name: None,
        }];

        let result =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::String)
                .unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should keep original multimodal content when no text parts exist
        assert!(transformed_message["content"].is_array());
    }

    #[test]
    fn test_transform_messages_mixed_content_types() {
        // Test with both text and multimodal content
        let messages = vec![
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Text("Plain text".to_string()),
                name: None,
            },
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "With image".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        // Test String format
        let result_string =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::String)
                .unwrap();

        assert_eq!(result_string.len(), 2);
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(result_string[1]["content"].as_str().unwrap(), "With image");

        // Test OpenAI format
        let result_openai =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::OpenAI)
                .unwrap();

        assert_eq!(result_openai.len(), 2);
        assert_eq!(result_openai[0]["content"].as_str().unwrap(), "Plain text");

        let content_array = result_openai[1]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1], json!({"type": "image"}));
    }
}
