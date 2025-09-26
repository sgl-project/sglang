// gRPC Router Implementation

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::{debug, error, info, warn};

use crate::config::types::RetryConfig;
use crate::core::{ConnectionMode, Worker, WorkerRegistry, WorkerType};
use crate::grpc_client::{proto, SglangSchedulerClient};
use crate::metrics::RouterMetrics;
use crate::policies::PolicyRegistry;
use crate::protocols::spec::ChatMessage;
use crate::protocols::spec::{ChatCompletionRequest, StringOrArray};
use crate::protocols::spec::{
    CompletionRequest, EmbeddingRequest, GenerateRequest, RerankRequest, ResponsesGetParams,
    ResponsesRequest, Tool, ToolChoice,
};
use crate::reasoning_parser::ParserFactory;
use crate::routers::RouterTrait;
use crate::server::AppContext;
use crate::tokenizer::chat_template::{ChatTemplateContentFormat, ChatTemplateParams};
use crate::tokenizer::traits::Tokenizer;
use crate::tokenizer::HuggingFaceTokenizer;
use crate::tool_parser::ParserRegistry;
use serde_json::Value;
use uuid::Uuid;

// Data structures for processing
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<proto::MultimodalInputs>,
    pub stop_sequences: Option<StringOrArray>,
}

/// gRPC router implementation for SGLang
#[allow(dead_code)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tokenizer: Arc<dyn Tokenizer>,
    reasoning_parser_factory: ParserFactory,
    tool_parser_registry: &'static ParserRegistry,
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
        let tool_parser_registry = ctx
            .tool_parser_registry
            .ok_or_else(|| "gRPC router requires tool parser registry".to_string())?;

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
            tool_parser_registry,
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
        let client = match worker.get_grpc_client().await {
            Ok(Some(client_arc)) => {
                // Clone the client from inside the Arc<Mutex<>>
                let client = client_arc.lock().await.clone();
                client
            }
            Ok(None) => {
                error!("Selected worker is not a gRPC worker");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Selected worker is not configured for gRPC",
                )
                    .into_response();
            }
            Err(e) => {
                error!("Failed to get gRPC client from worker: {}", e);
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

        // Step 6: Build the base gRPC request
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());
        let request = match client.build_generate_request(
            request_id,
            body,
            processed_messages.text.clone(),
            token_ids.into_iter().map(|id| id as i32).collect(),
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

    /// Generate tool constraints for structured generation
    fn generate_tool_constraints(
        &self,
        _tools: &[Tool],
        _tool_choice: &Option<ToolChoice>,
        model: &str,
    ) -> Option<(String, String)> {
        let _parser = self.tool_parser_registry.get_parser(model)?;
        // TODO: Implement actual constraint generation logic
        // For now, return None as this is placeholder implementation
        None
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
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
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
