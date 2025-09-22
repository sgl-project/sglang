// gRPC Router Implementation

use crate::config::types::RetryConfig;
use crate::core::{
    BasicWorkerBuilder, CircuitBreakerConfig, HealthConfig, WorkerRegistry, WorkerType,
};
use crate::grpc::{proto, SglangSchedulerClient};
use crate::metrics::RouterMetrics;
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::protocols::spec::{ChatCompletionRequest, ChatMessage, StringOrArray, UserMessageContent, ContentPart, ResponseFormat};
use crate::reasoning_parser::ParserFactory;
use crate::routers::RouterTrait;
use crate::tokenizer::{traits::Tokenizer, chat_template::ChatMessage as TokenizerChatMessage};
use crate::tool_parser::ParserRegistry;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

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
        debug!("Processing chat completion request for model: {:?}", model_id);


        // Step 1: Process messages and apply chat template
        let processed_messages = match self.process_chat_messages(body) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!("Failed to process chat messages: {}", e);
                return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
            }
        };

        // Step 2: Tokenize the processed text
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

        // Step 3: Build tool constraints if needed
        let structural_tag = if let Some(tools) = &body.tools {
            self.generate_tool_constraints(tools, &body.tool_choice, &body.model)
        } else {
            None
        };

        // Step 4: Build SamplingParams for gRPC
        let sampling_params = self.build_grpc_sampling_params(body, structural_tag);

        // Step 5: Create GenerateRequest
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

        // Step 6: Select worker
        let worker = match self.select_worker_for_request(model_id, &processed_messages.text) {
            Some(w) => w,
            None => {
                warn!("No available workers for model: {:?}", model_id);
                return (StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response();
            }
        };

        debug!("Selected worker: {}", worker.url());

        // Step 7: Get gRPC client for worker
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

        // Step 8: Handle streaming vs non-streaming
        if body.stream {
            self.handle_streaming_chat(client, grpc_request, body).await
        } else {
            self.handle_non_streaming_chat(client, grpc_request, body).await
        }
    }

    // ============ Helper Methods ============


    /// Process chat messages and apply template
    fn process_chat_messages(&self, request: &ChatCompletionRequest) -> Result<ProcessedMessages, String> {
        let tokenizer_messages = self.convert_messages_for_tokenizer(&request.messages)?;
        let formatted_text = self.format_messages_simple(&tokenizer_messages);
        let multimodal_inputs = self.extract_multimodal_inputs(&request.messages)?;

        Ok(ProcessedMessages {
            text: formatted_text,
            multimodal_inputs,
            stop_sequences: request.stop.clone(),
        })
    }

    /// Convert spec ChatMessage enum to tokenizer ChatMessage struct
    fn convert_messages_for_tokenizer(&self, messages: &[ChatMessage]) -> Result<Vec<TokenizerChatMessage>, String> {
        let mut converted = Vec::new();

        for message in messages {
            let tokenizer_msg = match message {
                ChatMessage::System { content, .. } => TokenizerChatMessage::new("system", content),
                ChatMessage::User { content, .. } => {
                    let text_content = match content {
                        UserMessageContent::Text(text) => text.clone(),
                        UserMessageContent::Parts(parts) => {
                            let mut combined_text = String::new();
                            for part in parts {
                                match part {
                                    ContentPart::Text { text } => {
                                        combined_text.push_str(text);
                                        combined_text.push('\n');
                                    }
                                    ContentPart::ImageUrl { image_url } => {
                                        combined_text.push_str(&format!("image://{}\n", image_url.url));
                                    }
                                }
                            }
                            combined_text
                        }
                    };
                    TokenizerChatMessage::new("user", text_content)
                }
                ChatMessage::Assistant { content, tool_calls, reasoning_content, .. } => {
                    let mut assistant_content = String::new();
                    if let Some(reasoning) = reasoning_content {
                        assistant_content.push_str(&format!("<think>{}</think>\n", reasoning));
                    }
                    if let Some(content) = content {
                        assistant_content.push_str(content);
                    }
                    if let Some(calls) = tool_calls {
                        for call in calls {
                            assistant_content.push_str(&format!("\n<tool_call>{}</tool_call>",
                                serde_json::to_string(call).unwrap_or_default()));
                        }
                    }
                    TokenizerChatMessage::new("assistant", assistant_content)
                }
                ChatMessage::Tool { content, tool_call_id, .. } => {
                    TokenizerChatMessage::new("tool", format!("tool_call_id: {}\n{}", tool_call_id, content))
                }
                ChatMessage::Function { content, name, .. } => {
                    TokenizerChatMessage::new("function", format!("function: {}\n{}", name, content))
                }
            };
            converted.push(tokenizer_msg);
        }

        Ok(converted)
    }

    /// Simple message formatting fallback
    fn format_messages_simple(&self, messages: &[TokenizerChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            result.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        result.push_str("<|im_start|>assistant\n");
        result
    }

    /// Extract multimodal inputs from messages
    fn extract_multimodal_inputs(&self, messages: &[ChatMessage]) -> Result<Option<proto::MultimodalInputs>, String> {
        let mut image_urls = Vec::new();
        let video_urls = Vec::new();
        let audio_urls = Vec::new();

        for message in messages {
            if let ChatMessage::User { content, .. } = message {
                if let UserMessageContent::Parts(parts) = content {
                    for part in parts {
                        if let ContentPart::ImageUrl { image_url } = part {
                            image_urls.push(image_url.url.clone());
                        }
                    }
                }
            }
        }

        if image_urls.is_empty() && video_urls.is_empty() && audio_urls.is_empty() {
            Ok(None)
        } else {
            Ok(Some(proto::MultimodalInputs {
                image_urls,
                video_urls,
                audio_urls,
                ..Default::default()
            }))
        }
    }

    /// Build gRPC SamplingParams from OpenAI request
    fn build_grpc_sampling_params(&self, request: &ChatCompletionRequest, structural_tag: Option<String>) -> proto::SamplingParams {
        let stop_sequences = self.extract_stop_strings(request);

        #[allow(deprecated)]
        proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(-1) as i32,
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_new_tokens: request.max_tokens.unwrap_or(128) as i32,
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens: request.skip_special_tokens,
            n: request.n.unwrap_or(1) as i32,
            structural_tag: structural_tag.unwrap_or_default(),
            constraint: self.build_constraint(request),
            ..Default::default()
        }
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
    fn build_constraint(&self, request: &ChatCompletionRequest) -> Option<proto::sampling_params::Constraint> {
        if let Some(format) = &request.response_format {
            if let ResponseFormat::JsonSchema { json_schema } = format {
                return Some(proto::sampling_params::Constraint::JsonSchema(
                    serde_json::to_string(&json_schema.schema).unwrap_or_default()
                ));
            }
        }

        if let Some(ebnf) = &request.ebnf {
            return Some(proto::sampling_params::Constraint::EbnfGrammar(ebnf.clone()));
        }

        if let Some(regex) = &request.regex {
            return Some(proto::sampling_params::Constraint::Regex(regex.clone()));
        }

        None
    }

    /// Generate tool constraints for structured generation
    fn generate_tool_constraints(&self, _tools: &[crate::protocols::spec::Tool], _tool_choice: &Option<crate::protocols::spec::ToolChoice>, model: &str) -> Option<String> {
        let _parser = self.tool_parser_registry.get_parser(model)?;
        None
    }

    /// Select a worker for the request
    fn select_worker_for_request(&self, model_id: Option<&str>, text: &str) -> Option<Arc<dyn crate::core::Worker>> {
        let workers = self.worker_registry.get_workers_filtered(
            model_id,
            Some(WorkerType::Regular),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            true,
        );

        if workers.is_empty() {
            return None;
        }

        // Policy returns an index, get the worker at that index
        if let Some(index) = self.policy.select_worker(&workers, Some(text)) {
            workers.get(index).cloned()
        } else {
            None
        }
    }

    /// Get or create a gRPC client for the worker
    async fn get_or_create_grpc_client(&self, worker_url: &str) -> Result<SglangSchedulerClient, String> {
        debug!("Creating new gRPC client for worker: {}", worker_url);
        SglangSchedulerClient::connect(worker_url)
            .await
            .map_err(|e| format!("Failed to connect to gRPC server: {}", e))
    }

    /// Placeholder for streaming handler (to be implemented in Phase 2)
    async fn handle_streaming_chat(&self, _client: SglangSchedulerClient, _request: proto::GenerateRequest, _original_request: &ChatCompletionRequest) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Streaming not yet implemented").into_response()
    }

    /// Placeholder for non-streaming handler (to be implemented in Phase 3)
    async fn handle_non_streaming_chat(&self, _client: SglangSchedulerClient, _request: proto::GenerateRequest, _original_request: &ChatCompletionRequest) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Non-streaming not yet implemented").into_response()
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

    async fn health(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
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

    async fn flush_cache(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }

    fn readiness(&self) -> Response {
        (StatusCode::SERVICE_UNAVAILABLE).into_response()
    }
}
