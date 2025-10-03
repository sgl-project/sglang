// gRPC Router Implementation

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use std::io;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

use crate::config::types::RetryConfig;
use crate::core::{ConnectionMode, Worker, WorkerRegistry, WorkerType};
use crate::grpc_client::{proto, SglangSchedulerClient};
use crate::metrics::RouterMetrics;
use crate::policies::PolicyRegistry;
use crate::protocols::spec::ChatMessage;
use crate::protocols::spec::{
    ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice, CompletionRequest,
    EmbeddingRequest, FunctionCallDelta, FunctionCallResponse, GenerateRequest, RerankRequest,
    ResponsesGetParams, ResponsesRequest, StringOrArray, Tool, ToolCall, ToolCallDelta, ToolChoice,
    ToolChoiceValue, Usage,
};
use crate::reasoning_parser::{ParserResult, ReasoningParserFactory};
use crate::routers::RouterTrait;
use crate::server::AppContext;
use crate::tokenizer::chat_template::{ChatTemplateContentFormat, ChatTemplateParams};
use crate::tokenizer::stop::{
    SequenceDecoderOutput, StopSequenceDecoder, StopSequenceDecoderBuilder,
};
use crate::tokenizer::traits::Tokenizer;
use crate::tokenizer::HuggingFaceTokenizer;
use crate::tool_parser::{StreamingParseResult, ToolParserFactory};
use proto::generate_response::Response::{Chunk, Complete, Error};
use serde_json::{json, Map, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;
use uuid::Uuid;

// Data structures for processing
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<proto::MultimodalInputs>,
    pub stop_sequences: Option<StringOrArray>,
}

/// gRPC router implementation for SGLang
#[derive(Clone)]
#[allow(dead_code)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tokenizer: Arc<dyn Tokenizer>,
    reasoning_parser_factory: ReasoningParserFactory,
    tool_parser_factory: ToolParserFactory,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
}

impl GrpcRouter {
    /// Create a new gRPC router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
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
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires tool parser factory".to_string())?
            .clone();

        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        let workers = worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc { port: None }),
            false,
        );

        RouterMetrics::set_active_workers(workers.len());
        info!("gRPC router found {} workers in registry", workers.len());

        Ok(GrpcRouter {
            worker_registry,
            policy_registry,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_factory,
            dp_aware: ctx.router_config.dp_aware,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

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

        // Step 2: Get gRPC client from worker
        let client = match Self::get_grpc_client_from_worker(&worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        // Step 3: Filter tools if needed for allowed_tools or specific function
        // Only clone body if we need to modify tools
        let mut body_with_filtered_tools;
        let body_ref = match &body.tool_choice {
            Some(ToolChoice::AllowedTools { tools: allowed, .. }) if body.tools.is_some() => {
                body_with_filtered_tools = body.clone();
                let all_tools = body_with_filtered_tools.tools.as_ref().unwrap();
                let allowed_names: std::collections::HashSet<&str> =
                    allowed.iter().map(|t| t.name.as_str()).collect();
                let filtered_tools: Vec<Tool> = all_tools
                    .iter()
                    .filter(|t| allowed_names.contains(t.function.name.as_str()))
                    .cloned()
                    .collect();
                body_with_filtered_tools.tools = Some(filtered_tools);
                &body_with_filtered_tools
            }
            Some(ToolChoice::Function { function, .. }) if body.tools.is_some() => {
                body_with_filtered_tools = body.clone();
                let all_tools = body_with_filtered_tools.tools.as_ref().unwrap();
                let filtered_tools: Vec<Tool> = all_tools
                    .iter()
                    .filter(|t| t.function.name == function.name)
                    .cloned()
                    .collect();
                body_with_filtered_tools.tools = Some(filtered_tools);
                &body_with_filtered_tools
            }
            _ => body, // No filtering needed, use original
        };

        // Step 4: Process messages and apply chat template
        let processed_messages = match self.process_chat_messages(body_ref) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!("Failed to process chat messages: {}", e);
                return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
            }
        };

        // Step 5: Tokenize the processed text
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

        // Step 6: Build tool constraints if needed
        // body_ref already has filtered tools if needed
        let tool_call_constraint = body_ref.tools.as_ref().and_then(|tools| {
            self.generate_tool_constraints(tools, &body.tool_choice, &body.model)
        });

        // Step 7: Build the base gRPC request (use body_ref with filtered tools if applicable)
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());
        let request = match client.build_generate_request(
            request_id,
            body_ref,
            processed_messages.text.clone(),
            token_ids,
            processed_messages.multimodal_inputs,
            tool_call_constraint, // Pass the full tuple (type, value)
        ) {
            Ok(request) => request,
            Err(e) => {
                error!("Failed to build gRPC request: {}", e);
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid request parameters: {}", e),
                )
                    .into_response();
            }
        };

        // Step 7: Handle streaming vs non-streaming
        if body.stream {
            self.handle_streaming_chat(client, request, body).await
        } else {
            self.handle_non_streaming_chat(client, request, body).await
        }
    }

    /// Main route_generate implementation
    async fn route_generate_impl(
        &self,
        _headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!("Processing generate request for model: {:?}", model_id);

        // Step 1: Resolve input (text, prompt, or input_ids)
        let (original_text, token_ids) = match self.resolve_generate_input(body) {
            Ok(res) => res,
            Err(msg) => {
                error!("Invalid generate request: {}", msg);
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        debug!("Resolved input with {} tokens", token_ids.len());

        // Step 2: Select worker (fail fast if no workers available)
        let worker = match self.select_worker_for_request(model_id, original_text.as_deref()) {
            Some(w) => w,
            None => {
                warn!("No available workers for model: {:?}", model_id);
                return (StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response();
            }
        };

        debug!("Selected worker: {}", worker.url());

        // Step 3: Get gRPC client from worker
        let client = match Self::get_grpc_client_from_worker(&worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        // Step 4: Build the gRPC request
        let request_id = body
            .rid
            .clone()
            .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

        let request = match client.build_plain_generate_request(
            request_id.clone(),
            body,
            original_text.clone(),
            token_ids,
        ) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to build generate request: {}", e);
                return (StatusCode::BAD_REQUEST, e).into_response();
            }
        };

        // Step 5: Get weight version for response metadata
        let weight_version = worker
            .metadata()
            .labels
            .get("weight_version")
            .cloned()
            .unwrap_or_else(|| "default".to_string());

        // Step 6: Handle streaming vs non-streaming
        if body.stream {
            // TODO: Implement streaming support for generate endpoint
            return (
                StatusCode::NOT_IMPLEMENTED,
                "Streaming generate over gRPC is not supported yet",
            )
                .into_response();
        }

        self.handle_non_streaming_generate(client, request, body, request_id, weight_version)
            .await
    }

    /// Get gRPC client from worker, returning appropriate error response on failure
    async fn get_grpc_client_from_worker(
        worker: &Arc<dyn Worker>,
    ) -> Result<SglangSchedulerClient, Response> {
        let client_arc = worker
            .get_grpc_client()
            .await
            .map_err(|e| {
                error!("Failed to get gRPC client from worker: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to get gRPC client: {}", e),
                )
                    .into_response()
            })?
            .ok_or_else(|| {
                error!("Selected worker is not a gRPC worker");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Selected worker is not configured for gRPC",
                )
                    .into_response()
            })?;

        let client = client_arc.lock().await.clone();
        Ok(client)
    }

    /// Select a worker for the request
    fn select_worker_for_request(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc { port: None }),
            false, // get all workers, we'll filter by is_available() next
        );

        // Filter by availability (health + circuit breaker)
        let available: Vec<Arc<dyn Worker>> = workers
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
        let formatted_text = if let Some(hf_tokenizer) = self
            .tokenizer
            .as_any()
            .downcast_ref::<HuggingFaceTokenizer>()
        {
            // Get content format and transform messages accordingly
            let content_format = hf_tokenizer.chat_template_content_format();
            let mut transformed_messages =
                Self::process_content_format(&request.messages, content_format)?;

            // Process tool call arguments in assistant messages
            Self::process_tool_call_arguments(&mut transformed_messages)?;

            // Convert tools to JSON values for template processing
            let tools_json: Option<Vec<Value>> = request
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
                    Value::String(reasoning_effort.clone()),
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
        messages: &[ChatMessage],
        content_format: ChatTemplateContentFormat,
    ) -> Result<Vec<Value>, String> {
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
                                "image_url" => Some(json!({"type": "image"})),
                                "video_url" => Some(json!({"type": "video"})),
                                "audio_url" => Some(json!({"type": "audio"})),
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
    fn process_tool_call_arguments(messages: &mut [Value]) -> Result<(), String> {
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
                match serde_json::from_str::<Value>(args_str) {
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

    /// Generate tool constraints for structured generation
    /// Note: tools should already be filtered if needed (by allowed_tools or specific function)
    fn generate_tool_constraints(
        &self,
        tools: &[Tool],
        tool_choice: &Option<ToolChoice>,
        _model: &str,
    ) -> Option<(String, String)> {
        let choice = tool_choice.as_ref()?;

        match choice {
            // Specific function: Return parameters schema directly
            // tools should already be filtered to contain only the specific function
            ToolChoice::Function { .. } => {
                if tools.is_empty() {
                    return None;
                }
                let tool = &tools[0];

                // Return the tool's parameters schema directly (not wrapped in array)
                let params_schema = serde_json::to_string(&tool.function.parameters).ok()?;
                Some(("json_schema".to_string(), params_schema))
            }

            // Required: Array of tool calls with minItems: 1
            ToolChoice::Value(ToolChoiceValue::Required) => {
                let schema = self.build_required_array_schema(tools)?;
                Some(("json_schema".to_string(), schema))
            }

            // AllowedTools with required mode: tools are already filtered
            ToolChoice::AllowedTools { mode, .. } => {
                if mode == "required" {
                    if tools.is_empty() {
                        return None;
                    }
                    let schema = self.build_required_array_schema(tools)?;
                    Some(("json_schema".to_string(), schema))
                } else {
                    // "auto" mode - no constraint needed
                    None
                }
            }

            // "auto" or "none" - no constraint
            _ => None,
        }
    }

    /// Build JSON schema for required tool calls (array with minItems: 1)
    /// Includes $defs consolidation from all tools (matching Python's behavior)
    fn build_required_array_schema(&self, tools: &[Tool]) -> Option<String> {
        // Build anyOf schemas for each tool
        let mut any_of_schemas = Vec::new();
        for tool in tools {
            let tool_schema = json!({
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [tool.function.name]
                    },
                    "parameters": tool.function.parameters
                },
                "required": ["name", "parameters"]
            });
            any_of_schemas.push(tool_schema);
        }

        // Consolidate $defs from all tools (matching Python's _get_tool_schema_defs)
        let mut all_defs: HashMap<String, Value> = HashMap::new();
        for tool in tools {
            if let Value::Object(params) = &tool.function.parameters {
                if let Some(Value::Object(defs)) = params.get("$defs") {
                    for (def_name, def_schema) in defs {
                        if let Some(existing) = all_defs.get(def_name) {
                            // Check for conflicts
                            if existing != def_schema {
                                error!(
                                    "Tool definition '{}' has multiple schemas, which is not supported",
                                    def_name
                                );
                                return None;
                            }
                        } else {
                            all_defs.insert(def_name.clone(), def_schema.clone());
                        }
                    }
                }
            }
        }

        // Build the full array schema
        let mut array_schema = json!({
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": any_of_schemas
            }
        });

        // Add $defs if any were found (matching Python's behavior)
        if !all_defs.is_empty() {
            if let Value::Object(ref mut schema_obj) = array_schema {
                let defs_value =
                    Value::Object(all_defs.into_iter().collect::<Map<String, Value>>());
                schema_obj.insert("$defs".to_string(), defs_value);
            }
        }

        serde_json::to_string(&array_schema).ok()
    }

    /// Parse tool calls from JSON schema constrained response
    fn parse_json_schema_response(
        &self,
        processed_text: &str,
        tool_choice: &Option<ToolChoice>,
    ) -> (Option<Vec<ToolCall>>, String) {
        match tool_choice {
            Some(ToolChoice::Function { function, .. }) => {
                // Specific function: Parse parameters directly
                match serde_json::from_str::<Value>(processed_text) {
                    Ok(params) => {
                        let tool_call = ToolCall {
                            id: format!("call_{}", uuid::Uuid::new_v4()),
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: function.name.clone(),
                                arguments: Some(
                                    serde_json::to_string(&params)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                ),
                            },
                        };
                        (Some(vec![tool_call]), String::new())
                    }
                    Err(e) => {
                        error!("Failed to parse specific function parameters: {}", e);
                        (None, processed_text.to_string())
                    }
                }
            }
            Some(ToolChoice::Value(ToolChoiceValue::Required))
            | Some(ToolChoice::AllowedTools { .. }) => {
                // Required mode: Parse array of tool calls
                match serde_json::from_str::<Vec<Value>>(processed_text) {
                    Ok(parsed_array) => {
                        let spec_tool_calls: Vec<ToolCall> = parsed_array
                            .into_iter()
                            .enumerate()
                            .filter_map(|(i, item)| {
                                let obj = item.as_object()?;
                                let name = obj.get("name")?.as_str()?.to_string();
                                let parameters = obj.get("parameters")?;

                                Some(ToolCall {
                                    id: format!("call_{}_{}", i, uuid::Uuid::new_v4()),
                                    tool_type: "function".to_string(),
                                    function: FunctionCallResponse {
                                        name,
                                        arguments: Some(
                                            serde_json::to_string(parameters)
                                                .unwrap_or_else(|_| "{}".to_string()),
                                        ),
                                    },
                                })
                            })
                            .collect();
                        (Some(spec_tool_calls), String::new())
                    }
                    Err(e) => {
                        error!("Failed to parse required tool call array: {}", e);
                        (None, processed_text.to_string())
                    }
                }
            }
            _ => (None, processed_text.to_string()),
        }
    }

    /// Parse tool calls using model-specific parser
    async fn parse_tool_calls(
        &self,
        processed_text: &str,
        model: &str,
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = self.tool_parser_factory.get_pooled(model);

        // Check format detection first
        let can_parse = {
            let parser = pooled_parser.lock().await;
            parser.detect_format(processed_text)
            // Lock is dropped here
        };

        if !can_parse {
            return (None, processed_text.to_string());
        }

        // Lock again for async parsing
        let result = {
            let parser = pooled_parser.lock().await;
            parser.parse_complete(processed_text).await
            // Lock is dropped here
        };

        match result {
            Ok((normal_text, parsed_tool_calls)) => {
                if parsed_tool_calls.is_empty() {
                    return (None, normal_text);
                }

                let spec_tool_calls = parsed_tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, tc)| {
                        // Generate ID for this tool call
                        let id = Self::generate_tool_call_id(
                            model,
                            &tc.function.name,
                            index,
                            history_tool_calls_count,
                        );
                        ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: tc.function.name,
                                arguments: Some(
                                    serde_json::to_string(&tc.function.arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                ),
                            },
                        }
                    })
                    .collect();
                (Some(spec_tool_calls), normal_text)
            }
            Err(e) => {
                error!("Tool call parsing error: {}", e);
                (None, processed_text.to_string())
            }
        }
    }

    /// Resolve the generate input into optional original text and token IDs
    fn resolve_generate_input(
        &self,
        request: &GenerateRequest,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        if let Some(text) = &request.text {
            return self
                .tokenize_single_text(text)
                .map(|(original, ids)| (Some(original), ids));
        }

        // Handle input_ids - validate and convert
        if let Some(input_ids) = &request.input_ids {
            return match input_ids {
                crate::protocols::spec::InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| u32::try_from(id))
                    .collect::<Result<Vec<u32>, _>>()
                    .map(|converted| (None, converted))
                    .map_err(|_| "input_ids must be non-negative".to_string()),
                crate::protocols::spec::InputIds::Batch(_) => {
                    Err("Batch input_ids are not supported over gRPC generate yet".to_string())
                }
            };
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    fn tokenize_single_text(&self, text: &str) -> Result<(String, Vec<u32>), String> {
        let encoding = self
            .tokenizer
            .encode(text)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        Ok((text.to_string(), encoding.token_ids().to_vec()))
    }

    fn internal_error_static(msg: &'static str) -> Response {
        error!("{}", msg);
        (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
    }

    fn internal_error_message(message: String) -> Response {
        error!("{}", message);
        (StatusCode::INTERNAL_SERVER_ERROR, message).into_response()
    }

    /// Create a StopSequenceDecoder from stop parameters
    fn create_stop_decoder(
        &self,
        stop: Option<&StringOrArray>,
        stop_token_ids: Option<&Vec<u32>>,
        skip_special_tokens: bool,
        no_stop_trim: bool,
    ) -> StopSequenceDecoder {
        // Extract stop sequences
        let stop_sequences: Vec<String> = match stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        };

        // Build stop sequence decoder
        let mut builder = StopSequenceDecoderBuilder::new(self.tokenizer.clone())
            .skip_special_tokens(skip_special_tokens);

        // Add stop sequences (visible if no_stop_trim is true, hidden otherwise)
        for seq in stop_sequences {
            builder = if no_stop_trim {
                builder.visible_stop_sequence(seq)
            } else {
                builder.stop_sequence(seq)
            };
        }

        // Add stop token IDs (visible if no_stop_trim is true, hidden otherwise)
        if let Some(token_ids) = stop_token_ids {
            for &token_id in token_ids {
                builder = if no_stop_trim {
                    builder.visible_stop_token(token_id)
                } else {
                    builder.stop_token(token_id)
                };
            }
        }

        builder.build()
    }

    /// Count the number of tool calls in the request message history
    /// This is used for KimiK2 format which needs globally unique indices
    fn get_history_tool_calls_count(request: &ChatCompletionRequest) -> usize {
        request
            .messages
            .iter()
            .filter_map(|msg| {
                if let ChatMessage::Assistant { tool_calls, .. } = msg {
                    tool_calls.as_ref().map(|calls| calls.len())
                } else {
                    None
                }
            })
            .sum()
    }

    /// Generate a tool call ID based on model format
    ///
    /// # Arguments
    /// * `model` - Model name to determine ID format
    /// * `tool_name` - Name of the tool being called
    /// * `tool_index` - Index of this tool call within the current message
    /// * `history_count` - Number of tool calls in previous messages
    ///
    /// # Returns
    /// A unique ID string. KimiK2 uses `functions.{name}:{global_index}`, others use `call_{uuid}`
    fn generate_tool_call_id(
        model: &str,
        tool_name: &str,
        tool_index: usize,
        history_count: usize,
    ) -> String {
        if model.to_lowercase().contains("kimi") {
            // KimiK2 format: functions.{name}:{global_index}
            format!("functions.{}:{}", tool_name, history_count + tool_index)
        } else {
            // Standard OpenAI format: call_{24-char-uuid}
            format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
        }
    }

    /// Process a chunk of tokens through the stop decoder
    fn process_chunk_tokens(
        stop_decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
    ) -> (String, bool) {
        let mut chunk_text = String::new();

        for &token_id in token_ids {
            match stop_decoder.process_token(token_id).unwrap_or_else(|e| {
                debug!(
                    "Error processing token {}: {}. Treating as Held.",
                    token_id, e
                );
                SequenceDecoderOutput::Held
            }) {
                SequenceDecoderOutput::Text(text) => {
                    chunk_text.push_str(&text);
                }
                SequenceDecoderOutput::StoppedWithText(text) => {
                    chunk_text.push_str(&text);
                    return (chunk_text, true); // Return text and signal to stop
                }
                SequenceDecoderOutput::Stopped => {
                    return (chunk_text, true); // Return text and signal to stop
                }
                SequenceDecoderOutput::Held => {
                    // Text held for potential stop sequence match
                }
            }
        }
        (chunk_text, false) // Return text and continue processing
    }

    /// Helper: Process reasoning content in streaming mode
    /// Returns (modified_delta, optional_reasoning_chunk)
    fn process_reasoning_stream(
        &self,
        delta: &str,
        index: u32,
        reasoning_parsers: &mut HashMap<
            u32,
            Arc<std::sync::Mutex<Box<dyn crate::reasoning_parser::ReasoningParser>>>,
        >,
        request_id: &str,
        model: &str,
        created: u64,
    ) -> (String, Option<ChatCompletionStreamResponse>) {
        // Get or create parser for this index
        reasoning_parsers
            .entry(index)
            .or_insert_with(|| self.reasoning_parser_factory.get_pooled(model));

        if let Some(pooled_parser) = reasoning_parsers.get(&index) {
            let parse_result = {
                let mut parser = pooled_parser.lock().unwrap();
                parser.parse_reasoning_streaming_incremental(delta)
            };

            match parse_result {
                Ok(ParserResult {
                    reasoning_text,
                    normal_text,
                }) => {
                    let chunk = if !reasoning_text.is_empty() {
                        Some(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    tool_calls: None,
                                    reasoning_content: Some(reasoning_text),
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        })
                    } else {
                        None
                    };
                    return (normal_text, chunk);
                }
                Err(e) => {
                    warn!("Reasoning parsing error: {}", e);
                }
            }
        }

        (delta.to_string(), None)
    }

    /// Helper: Process tool calls in streaming mode
    /// Returns (should_skip_content, chunks_to_emit)
    #[allow(clippy::too_many_arguments)]
    async fn process_tool_calls_stream(
        &self,
        delta: &str,
        index: u32,
        tool_parsers: &mut HashMap<
            u32,
            Arc<tokio::sync::Mutex<Box<dyn crate::tool_parser::ToolParser>>>,
        >,
        has_tool_calls: &mut HashMap<u32, bool>,
        tools: &[crate::protocols::spec::Tool],
        request_id: &str,
        model: &str,
        created: u64,
        history_tool_calls_count: usize,
    ) -> (bool, Vec<ChatCompletionStreamResponse>) {
        let mut chunks = Vec::new();

        // Get or create parser for this index
        tool_parsers
            .entry(index)
            .or_insert_with(|| self.tool_parser_factory.get_pooled(model));

        if let Some(pooled_parser) = tool_parsers.get(&index) {
            let mut parser = pooled_parser.lock().await;
            match parser.parse_incremental(delta, tools).await {
                Ok(StreamingParseResult { normal_text, calls }) => {
                    // Emit normal text if present
                    if !normal_text.is_empty() {
                        chunks.push(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: Some(normal_text),
                                    tool_calls: None,
                                    reasoning_content: None,
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        });
                    }

                    // Emit tool call chunks
                    for tool_call_item in calls {
                        has_tool_calls.insert(index, true);

                        let tool_call_id = if let Some(ref name) = tool_call_item.name {
                            Some(Self::generate_tool_call_id(
                                model,
                                name,
                                tool_call_item.tool_index,
                                history_tool_calls_count,
                            ))
                        } else {
                            None
                        };

                        let tool_call_delta = ToolCallDelta {
                            index: tool_call_item.tool_index as u32,
                            id: tool_call_id,
                            tool_type: if tool_call_item.name.is_some() {
                                Some("function".to_string())
                            } else {
                                None
                            },
                            function: Some(FunctionCallDelta {
                                name: tool_call_item.name,
                                arguments: if !tool_call_item.parameters.is_empty() {
                                    Some(tool_call_item.parameters)
                                } else {
                                    None
                                },
                            }),
                        };

                        chunks.push(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    tool_calls: Some(vec![tool_call_delta]),
                                    reasoning_content: None,
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        });
                    }

                    // If we emitted chunks, skip regular content
                    return (!chunks.is_empty(), chunks);
                }
                Err(e) => {
                    warn!("Tool call parsing error: {}", e);
                }
            }
        }

        (false, chunks)
    }

    /// Helper: Create content chunk
    fn create_content_chunk(
        content: String,
        index: u32,
        request_id: &str,
        model: &str,
        created: u64,
        logprobs: Option<crate::protocols::spec::ChatLogProbs>,
    ) -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: request_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            system_fingerprint: None,
            choices: vec![ChatStreamChoice {
                index,
                delta: ChatMessageDelta {
                    role: Some("assistant".to_string()),
                    content: Some(content),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs,
                finish_reason: None,
                matched_stop: None,
            }],
            usage: None,
        }
    }

    /// Helper: Format response as SSE chunk
    fn format_sse_chunk(response: &ChatCompletionStreamResponse) -> String {
        format!(
            "data: {}\n\n",
            serde_json::to_string(response).unwrap_or_default()
        )
    }

    /// Submit request and handle streaming response for chat completions route
    async fn handle_streaming_chat(
        &self,
        mut client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &ChatCompletionRequest,
    ) -> Response {
        let request_id = request.request_id.clone();
        let model = original_request.model.clone();

        // Create channel for SSE streaming
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Start the gRPC stream
        let mut grpc_stream = match client.generate(request).await {
            Ok(stream) => stream,
            Err(e) => {
                error!("Failed to start generation: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Generation failed: {}", e),
                )
                    .into_response();
            }
        };

        let stop_params = (
            original_request.stop.clone(),
            original_request.stop_token_ids.clone(),
            original_request.skip_special_tokens,
            original_request.no_stop_trim,
        );

        // Spawn processing task
        let self_clone = self.clone();
        let original_request_clone = original_request.clone();
        tokio::spawn(async move {
            let result = Self::process_streaming_chunks(
                &self_clone,
                &mut grpc_stream,
                request_id,
                model,
                stop_params,
                original_request_clone,
                &tx,
            )
            .await;

            if let Err(e) = result {
                let error_chunk = format!(
                    "data: {}\n\n",
                    json!({
                        "error": {
                            "message": e,
                            "type": "internal_error"
                        }
                    })
                );
                let _ = tx.send(Ok(Bytes::from(error_chunk)));
            }

            // Send DONE marker
            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
        });

        // Create response with SSE headers
        let stream = UnboundedReceiverStream::new(rx);
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        response
            .headers_mut()
            .insert("Cache-Control", HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert("Connection", HeaderValue::from_static("keep-alive"));
        response
    }

    /// Process streaming chunks and send SSE events
    async fn process_streaming_chunks(
        router: &GrpcRouter,
        grpc_stream: &mut (impl tokio_stream::Stream<Item = Result<proto::GenerateResponse, tonic::Status>>
                  + Unpin),
        request_id: String,
        model: String,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        original_request: ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Extract request parameters
        let separate_reasoning = original_request.separate_reasoning;
        let tool_choice = &original_request.tool_choice;
        let tools = &original_request.tools;
        let history_tool_calls_count = Self::get_history_tool_calls_count(&original_request);
        let stream_options = &original_request.stream_options;

        // Phase 1: Initialize state tracking (per-index for n>1 support)
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut stream_buffers: HashMap<u32, String> = HashMap::new();
        let mut finish_reasons: HashMap<u32, String> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<Value>> = HashMap::new();
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut completion_tokens: HashMap<u32, u32> = HashMap::new();
        let mut cached_tokens: HashMap<u32, u32> = HashMap::new();

        // Parser state (lazy initialization per index)
        type PooledReasoningParser =
            Arc<std::sync::Mutex<Box<dyn crate::reasoning_parser::ReasoningParser>>>;
        let mut reasoning_parsers: HashMap<u32, PooledReasoningParser> = HashMap::new();

        type PooledToolParser = Arc<tokio::sync::Mutex<Box<dyn crate::tool_parser::ToolParser>>>;
        let mut tool_parsers: HashMap<u32, PooledToolParser> = HashMap::new();
        let mut has_tool_calls: HashMap<u32, bool> = HashMap::new();

        // Create stop decoder
        let (stop, stop_token_ids, skip_special_tokens, no_stop_trim) = stop_params;
        let mut stop_decoder = router.create_stop_decoder(
            stop.as_ref(),
            stop_token_ids.as_ref(),
            skip_special_tokens,
            no_stop_trim,
        );

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Phase 2: Main streaming loop
        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e))?;

            match gen_response.response {
                Some(Chunk(chunk)) => {
                    let index = chunk.index;

                    // Process tokens through stop decoder
                    let (chunk_text, _should_stop) =
                        Self::process_chunk_tokens(&mut stop_decoder, &chunk.token_ids);

                    if chunk_text.is_empty() {
                        continue;
                    }

                    // Process logprobs if present
                    let choice_logprobs = if let Some(ref proto_logprobs) = chunk.output_logprobs {
                        match router.convert_proto_to_openai_logprobs(proto_logprobs) {
                            Ok(logprobs) => Some(logprobs),
                            Err(e) => {
                                warn!("Failed to process logprobs: {}", e);
                                None
                            }
                        }
                    } else {
                        None
                    };

                    // Initialize stream buffer if first time
                    let stream_buffer = stream_buffers.entry(index).or_default();

                    // Send first chunk with role
                    if is_firsts.get(&index).copied().unwrap_or(true) {
                        let first_chunk = ChatCompletionStreamResponse {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    tool_calls: None,
                                    reasoning_content: None,
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        };
                        tx.send(Ok(Bytes::from(Self::format_sse_chunk(&first_chunk))))
                            .map_err(|_| "Failed to send first chunk".to_string())?;
                        is_firsts.insert(index, false);
                    }

                    // Calculate delta
                    let mut delta = chunk_text;
                    stream_buffer.push_str(&delta);

                    // Reasoning content handling
                    if separate_reasoning {
                        let (normal_text, reasoning_chunk) = router.process_reasoning_stream(
                            &delta,
                            index,
                            &mut reasoning_parsers,
                            &request_id,
                            &model,
                            created,
                        );
                        if let Some(chunk) = reasoning_chunk {
                            tx.send(Ok(Bytes::from(Self::format_sse_chunk(&chunk))))
                                .map_err(|_| "Failed to send reasoning chunk".to_string())?;
                        }
                        delta = normal_text;
                    }

                    // Tool call handling
                    let tool_choice_enabled =
                        !matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));

                    if tool_choice_enabled && tools.is_some() {
                        let (should_skip, tool_chunks) = router
                            .process_tool_calls_stream(
                                &delta,
                                index,
                                &mut tool_parsers,
                                &mut has_tool_calls,
                                tools.as_ref().unwrap(),
                                &request_id,
                                &model,
                                created,
                                history_tool_calls_count,
                            )
                            .await;

                        for chunk in tool_chunks {
                            tx.send(Ok(Bytes::from(Self::format_sse_chunk(&chunk))))
                                .map_err(|_| "Failed to send tool call chunk".to_string())?;
                        }

                        if should_skip {
                            continue;
                        }
                    }

                    // Regular content emission
                    if !delta.is_empty() {
                        let content_chunk = Self::create_content_chunk(
                            delta,
                            index,
                            &request_id,
                            &model,
                            created,
                            choice_logprobs,
                        );
                        tx.send(Ok(Bytes::from(Self::format_sse_chunk(&content_chunk))))
                            .map_err(|_| "Failed to send content chunk".to_string())?;
                    }
                }
                Some(Complete(complete)) => {
                    // Flush any remaining text
                    if let SequenceDecoderOutput::Text(text) = stop_decoder.flush() {
                        if !text.is_empty() {
                            let index = complete.index;
                            let stream_buffer = stream_buffers.entry(index).or_default();
                            stream_buffer.push_str(&text);

                            let content_chunk = ChatCompletionStreamResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model.clone(),
                                system_fingerprint: None,
                                choices: vec![ChatStreamChoice {
                                    index,
                                    delta: ChatMessageDelta {
                                        role: Some("assistant".to_string()),
                                        content: Some(text),
                                        tool_calls: None,
                                        reasoning_content: None,
                                    },
                                    logprobs: None,
                                    finish_reason: None,
                                    matched_stop: None,
                                }],
                                usage: None,
                            };

                            let sse_chunk = serde_json::to_string(&content_chunk)
                                .map_err(|e| format!("Failed to serialize content chunk: {}", e))?;
                            tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                                .map_err(|_| "Failed to send flushed content".to_string())?;
                        }
                    }

                    // Store metadata
                    let index = complete.index;
                    prompt_tokens.insert(index, complete.prompt_tokens as u32);
                    completion_tokens.insert(index, complete.completion_tokens as u32);
                    cached_tokens.insert(index, complete.cached_tokens as u32);
                    finish_reasons.insert(index, complete.finish_reason.clone());

                    // Extract matched_stop
                    let matched_stop_value = match &complete.matched_stop {
                        Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                            Some(Value::Number(serde_json::Number::from(*token_id)))
                        }
                        Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                            Some(Value::String(stop_str.clone()))
                        }
                        None => None,
                    };
                    matched_stops.insert(index, matched_stop_value);

                    break;
                }
                Some(Error(error)) => {
                    return Err(error.message);
                }
                None => continue,
            }
        }

        // Phase 3: Check unstreamed tool args
        // Check if parsers have any remaining arguments that haven't been streamed yet
        for (index, parser) in &tool_parsers {
            let parser_guard = parser.lock().await;
            if let Some(unstreamed_items) = parser_guard.get_unstreamed_tool_args() {
                for tool_call_item in unstreamed_items {
                    let tool_call_delta = ToolCallDelta {
                        index: tool_call_item.tool_index as u32,
                        id: None,
                        tool_type: None, // No type for argument deltas
                        function: Some(FunctionCallDelta {
                            name: None, // No name for argument deltas
                            arguments: if !tool_call_item.parameters.is_empty() {
                                Some(tool_call_item.parameters)
                            } else {
                                None
                            },
                        }),
                    };

                    let tool_chunk = ChatCompletionStreamResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        system_fingerprint: None,
                        choices: vec![ChatStreamChoice {
                            index: *index,
                            delta: ChatMessageDelta {
                                role: Some("assistant".to_string()),
                                content: None,
                                tool_calls: Some(vec![tool_call_delta]),
                                reasoning_content: None,
                            },
                            logprobs: None,
                            finish_reason: None,
                            matched_stop: None,
                        }],
                        usage: None,
                    };

                    let sse_chunk = serde_json::to_string(&tool_chunk)
                        .map_err(|e| format!("Failed to serialize tool chunk: {}", e))?;
                    tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                        .map_err(|_| "Failed to send unstreamed tool args".to_string())?;
                }
            }
        }

        // Phase 4: Finish reason chunks
        for (index, finish_reason) in finish_reasons.iter() {
            let final_finish_reason =
                if has_tool_calls.get(index).copied().unwrap_or(false) && finish_reason == "stop" {
                    "tool_calls".to_string()
                } else {
                    finish_reason.clone()
                };

            let matched_stop_value = matched_stops.get(index).and_then(|v| v.clone());

            let finish_chunk = ChatCompletionStreamResponse {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                system_fingerprint: None,
                choices: vec![ChatStreamChoice {
                    index: *index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: Some(final_finish_reason),
                    matched_stop: matched_stop_value,
                }],
                usage: None,
            };

            let sse_chunk = serde_json::to_string(&finish_chunk)
                .map_err(|e| format!("Failed to serialize finish chunk: {}", e))?;
            tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                .map_err(|_| "Failed to send finish chunk".to_string())?;
        }

        // Phase 5: Usage chunk
        if let Some(stream_opts) = stream_options {
            if stream_opts.include_usage.unwrap_or(false) {
                let total_prompt: u32 = prompt_tokens.values().sum();
                let total_completion: u32 = completion_tokens.values().sum();

                let usage_chunk = ChatCompletionStreamResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    system_fingerprint: None,
                    choices: vec![],
                    usage: Some(Usage {
                        prompt_tokens: total_prompt,
                        completion_tokens: total_completion,
                        total_tokens: total_prompt + total_completion,
                        completion_tokens_details: None,
                    }),
                };

                let sse_chunk = serde_json::to_string(&usage_chunk)
                    .map_err(|e| format!("Failed to serialize usage chunk: {}", e))?;
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                    .map_err(|_| "Failed to send usage chunk".to_string())?;
            }
        }

        Ok(())
    }

    /// Submit request and handle non-streaming response for chat completions route
    async fn handle_non_streaming_chat(
        &self,
        mut client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &ChatCompletionRequest,
    ) -> Response {
        let mut stop_decoder = self.create_stop_decoder(
            original_request.stop.as_ref(),
            original_request.stop_token_ids.as_ref(),
            original_request.skip_special_tokens,
            original_request.no_stop_trim,
        );

        // Start generation
        let mut stream = match client.generate(request).await {
            Ok(s) => s,
            Err(e) => {
                return Self::internal_error_message(format!("Failed to start generation: {}", e))
            }
        };

        // Collect all responses (for n>1 support)
        let mut all_responses = Vec::new();
        while let Some(response) = stream.next().await {
            match response {
                Ok(gen_response) => match gen_response.response {
                    Some(Complete(complete)) => {
                        all_responses.push(complete);
                    }
                    Some(Error(err)) => {
                        return Self::internal_error_message(format!(
                            "Generation failed: {}",
                            err.message
                        ));
                    }
                    Some(Chunk(_)) => {
                        return Self::internal_error_static(
                            "Unexpected chunk response for non-streaming request",
                        )
                    }
                    None => return Self::internal_error_static("Empty response from server"),
                },
                Err(e) => {
                    return Self::internal_error_message(format!(
                        "Failed to get GenerateResponse: {}",
                        e
                    ))
                }
            }
        }

        if all_responses.is_empty() {
            return Self::internal_error_static("No responses from server");
        }

        // Process each response into a ChatChoice
        let history_tool_calls_count = Self::get_history_tool_calls_count(original_request);
        let mut choices = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            match self
                .process_single_choice(
                    complete,
                    index,
                    original_request,
                    &mut stop_decoder,
                    history_tool_calls_count,
                )
                .await
            {
                Ok(choice) => choices.push(choice),
                Err(e) => {
                    return Self::internal_error_message(format!(
                        "Failed to process choice {}: {}",
                        index, e
                    ));
                }
            }
        }

        // Aggregate usage information from all responses
        let total_prompt_tokens: u32 = all_responses.iter().map(|r| r.prompt_tokens as u32).sum();
        let total_completion_tokens: u32 = all_responses
            .iter()
            .map(|r| r.completion_tokens as u32)
            .sum();
        let usage = Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens_details: None,
        };

        // Build final ChatCompletionResponse
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: original_request.model.clone(),
            choices,
            usage: Some(usage),
            system_fingerprint: None,
        };

        // Serialize and return JSON response
        Json(response).into_response()
    }

    /// Submit request and handle non-streaming response for the `/generate` endpoint
    async fn handle_non_streaming_generate(
        &self,
        mut client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &GenerateRequest,
        request_id: String,
        weight_version: String,
    ) -> Response {
        let start_time = Instant::now();

        let mut stream = match client.generate(request).await {
            Ok(stream) => stream,
            Err(e) => {
                return Self::internal_error_message(format!("Failed to start generation: {}", e))
            }
        };

        let mut final_completion: Option<proto::GenerateComplete> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(gen_response) => match gen_response.response {
                    Some(Complete(complete)) => {
                        final_completion = Some(complete);
                        break;
                    }
                    Some(Error(err)) => {
                        return Self::internal_error_message(format!(
                            "Generation failed: {}",
                            err.message
                        ));
                    }
                    Some(Chunk(_)) | None => continue,
                },
                Err(e) => {
                    return Self::internal_error_message(format!(
                        "Failed to receive generate response: {}",
                        e
                    ))
                }
            }
        }

        let mut complete = match final_completion {
            Some(c) => c,
            None => {
                return Self::internal_error_static("No completion received from scheduler");
            }
        };

        // Create stop decoder from sampling params
        let params = original_request.sampling_params.as_ref();
        let mut stop_decoder = self.create_stop_decoder(
            params.and_then(|p| p.stop.as_ref()),
            params.and_then(|p| p.stop_token_ids.as_ref()),
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            params.and_then(|p| p.no_stop_trim).unwrap_or(false),
        );

        // Process tokens through stop decoder
        let outputs = match stop_decoder.process_tokens(&complete.output_ids) {
            Ok(outputs) => outputs,
            Err(e) => {
                return Self::internal_error_message(format!("Failed to process tokens: {}", e))
            }
        };

        // Accumulate text with early breaks
        let mut decoded_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    decoded_text.push_str(&t);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }

        // Flush remaining text
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            decoded_text.push_str(&t);
        }

        let output_ids = std::mem::take(&mut complete.output_ids);
        let finish_reason = std::mem::take(&mut complete.finish_reason);

        // Build base meta_info using json! macro
        let mut meta_info = json!({
            "finish_reason": finish_reason,
            "prompt_tokens": complete.prompt_tokens,
            "completion_tokens": complete.completion_tokens,
            "cached_tokens": complete.cached_tokens,
            "id": request_id,
            "weight_version": weight_version,
            "e2e_latency": start_time.elapsed().as_secs_f64(),
        });

        let meta_obj = meta_info.as_object_mut().unwrap();

        // Add matched_stop if present
        if let Some(matched) = complete.matched_stop.take() {
            use proto::generate_complete::MatchedStop;
            let matched_value = match matched {
                MatchedStop::MatchedTokenId(id) => json!(id),
                MatchedStop::MatchedStopStr(s) => json!(s),
            };
            meta_obj.insert("matched_stop".to_string(), matched_value);
        }

        let response_body = json!({
            "text": decoded_text,
            "output_ids": output_ids,
            "meta_info": meta_info,
        });

        Json(response_body).into_response()
    }

    /// Convert proto LogProbs to OpenAI ChatLogProbs format
    /// Note: Always decodes with skip_special_tokens=false to show actual tokens generated
    fn convert_proto_to_openai_logprobs(
        &self,
        proto_logprobs: &proto::OutputLogProbs,
    ) -> Result<crate::protocols::spec::ChatLogProbs, String> {
        let mut content_items = Vec::new();

        // Decode token IDs to text (always with skip_special_tokens=false for logprobs)
        let token_texts: Vec<String> = proto_logprobs
            .token_ids
            .iter()
            .map(|&token_id| {
                self.tokenizer
                    .decode(&[token_id as u32], false)
                    .unwrap_or_else(|_| format!("<token_{}>", token_id))
            })
            .collect();

        // Build ChatLogProbsContent for each token (consume iterator to avoid clones)
        for (i, (&logprob, token_text)) in proto_logprobs
            .token_logprobs
            .iter()
            .zip(token_texts.into_iter())
            .enumerate()
        {
            let bytes = Some(token_text.as_bytes().to_vec());

            // Build top_logprobs for this position
            let mut top_logprobs = Vec::new();
            if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
                // Decode top token IDs (always with skip_special_tokens=false)
                let top_token_texts: Vec<String> = top_logprobs_entry
                    .token_ids
                    .iter()
                    .map(|&tid| {
                        self.tokenizer
                            .decode(&[tid as u32], false)
                            .unwrap_or_else(|_| format!("<token_{}>", tid))
                    })
                    .collect();

                for (j, (&top_logprob, &_top_token_id)) in top_logprobs_entry
                    .values
                    .iter()
                    .zip(top_logprobs_entry.token_ids.iter())
                    .enumerate()
                {
                    if let Some(top_token_text) = top_token_texts.get(j) {
                        top_logprobs.push(crate::protocols::spec::TopLogProb {
                            token: top_token_text.clone(),
                            logprob: top_logprob,
                            bytes: Some(top_token_text.as_bytes().to_vec()),
                        });
                    }
                }
            }

            content_items.push(crate::protocols::spec::ChatLogProbsContent {
                token: token_text,
                logprob,
                bytes,
                top_logprobs,
            });
        }

        Ok(crate::protocols::spec::ChatLogProbs::Detailed {
            content: (!content_items.is_empty()).then_some(content_items),
        })
    }

    /// Process a single GenerateComplete response into a ChatChoice
    async fn process_single_choice(
        &self,
        complete: &proto::GenerateComplete,
        index: usize,
        original_request: &ChatCompletionRequest,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
    ) -> Result<ChatChoice, String> {
        stop_decoder.reset();
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(&complete.output_ids)
            .map_err(|e| format!("Failed to process tokens: {}", e))?;

        // Accumulate text with early breaks
        let mut final_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }

        // Flush remaining text
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            final_text.push_str(&t);
        }

        // Step 1: Handle reasoning content parsing
        let mut reasoning_text: Option<String> = None;
        let mut processed_text = final_text;

        // Check if reasoning parsing is enabled and separate_reasoning is requested
        if original_request.separate_reasoning {
            let pooled_parser = self
                .reasoning_parser_factory
                .get_pooled(&original_request.model);

            let mut parser = pooled_parser
                .lock()
                .map_err(|e| format!("Failed to acquire reasoning parser lock: {}", e))?;
            match parser.detect_and_parse_reasoning(&processed_text) {
                Ok(result) => {
                    if !result.reasoning_text.is_empty() {
                        reasoning_text = Some(result.reasoning_text);
                    }
                    processed_text = result.normal_text;
                }
                Err(e) => {
                    return Err(format!("Reasoning parsing error: {}", e));
                }
            }
        }

        // Step 2: Handle tool call parsing
        let mut tool_calls: Option<Vec<crate::protocols::spec::ToolCall>> = None;

        // Check if tool calls should be processed
        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(ToolChoice::Value(
                crate::protocols::spec::ToolChoiceValue::None
            ))
        );

        if tool_choice_enabled && original_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific function or required mode)
            let used_json_schema = match &original_request.tool_choice {
                Some(ToolChoice::Function { .. }) => true,
                Some(ToolChoice::Value(crate::protocols::spec::ToolChoiceValue::Required)) => true,
                Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                _ => false,
            };

            if used_json_schema {
                (tool_calls, processed_text) =
                    self.parse_json_schema_response(&processed_text, &original_request.tool_choice);
            } else {
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        &original_request.model,
                        history_tool_calls_count,
                    )
                    .await;
            }
        }

        // Step 3: Use finish reason directly from proto (already OpenAI-compatible string)
        let finish_reason_str = &complete.finish_reason;

        // Override finish reason if we have tool calls
        let final_finish_reason_str = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason_str
        };

        // Extract matched_stop information from proto
        let matched_stop = match &complete.matched_stop {
            Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => Some(
                serde_json::Value::Number(serde_json::Number::from(*token_id)),
            ),
            Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                Some(serde_json::Value::String(stop_str.clone()))
            }
            None => None,
        };

        // Step 4: Convert output logprobs if present
        // Note: complete.input_logprobs exists in proto but is not used for chat completions
        //       (input logprobs are only used in /v1/completions endpoint with echo=true)
        let logprobs = if let Some(proto_logprobs) = &complete.output_logprobs {
            match self.convert_proto_to_openai_logprobs(proto_logprobs) {
                Ok(logprobs) => Some(logprobs),
                Err(e) => {
                    error!("Failed to convert logprobs: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 5: Build ChatCompletionMessage (proper response message type)
        let chat_message = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: if processed_text.is_empty() {
                None
            } else {
                Some(processed_text)
            },
            tool_calls,
            reasoning_content: reasoning_text,
        };

        // Step 6: Build ChatChoice
        let choice = ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        };

        Ok(choice)
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.worker_registry.stats();
        f.debug_struct("GrpcRouter")
            .field("workers_count", &stats.total_workers)
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
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_generate_impl(headers, body, model_id).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
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

        let result_string =
            GrpcRouter::process_content_format(&messages, ChatTemplateContentFormat::String)
                .unwrap();

        assert_eq!(result_string.len(), 2);
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(result_string[1]["content"].as_str().unwrap(), "With image");

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
