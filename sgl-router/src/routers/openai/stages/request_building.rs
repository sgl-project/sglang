//! Request Building stage
//!
//! This stage:
//! - Builds the HTTP request payload
//! - Injects conversation items from context loading
//! - Strips SGLang-specific fields (conversation, previous_response_id, store)
//! - Applies provider-specific transformations (xAI/Grok)
//! - Serializes to JSON

use std::collections::HashSet;

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use once_cell::sync::Lazy;
use serde_json::{to_value, Value};

use super::PipelineStage;
use crate::{
    protocols::responses::{ResponseContentPart, ResponseInput, ResponseInputOutputItem},
    routers::openai::context::{PayloadOutput, RequestContext, RequestType},
};

/// Fields specific to SGLang that should be stripped when forwarding to OpenAI-compatible endpoints
static SGLANG_FIELDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "request_id",
        "priority",
        "top_k",
        "min_p",
        "min_tokens",
        "regex",
        "ebnf",
        "stop_token_ids",
        "no_stop_trim",
        "ignore_eos",
        "continue_final_message",
        "skip_special_tokens",
        "lora_path",
        "session_params",
        "separate_reasoning",
        "stream_reasoning",
        "chat_template_kwargs",
        "return_hidden_states",
        "repetition_penalty",
        "sampling_seed",
        "conversation",
        "previous_response_id",
    ])
});

/// Request building stage
pub struct RequestBuildingStage;

#[async_trait]
impl PipelineStage for RequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Build payload based on request type
        let (payload, is_streaming) = match &ctx.input.request_type {
            RequestType::Chat(req) => {
                let is_streaming = req.stream;
                let mut payload = to_value(&**req).map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to serialize request: {}", e),
                    )
                        .into_response()
                })?;

                // Strip SGLang-specific fields and apply transformations
                if let Some(obj) = payload.as_object_mut() {
                    obj.retain(|k, _| !SGLANG_FIELDS.contains(&k.as_str()));

                    // Remove logprobs if false (Gemini compatibility)
                    if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                        obj.remove("logprobs");
                    }
                }

                (payload, is_streaming)
            }
            RequestType::Responses(req) => {
                // Clone request for modification
                let mut request_body = (**req).clone();

                // Apply model_id if provided
                if let Some(model) = &ctx.input.model_id {
                    request_body.model = model.clone();
                }

                // Inject conversation items from context loading
                if let Some(context) = &ctx.state.context {
                    if let Some(conversation_items) = &context.conversation_items {
                        let mut items = conversation_items.clone();

                        // Append current request input
                        match &request_body.input {
                            ResponseInput::Text(text) => {
                                items.push(ResponseInputOutputItem::Message {
                                    id: format!(
                                        "msg_u_{}",
                                        context
                                            .previous_response_id
                                            .as_ref()
                                            .or(context.conversation_id.as_ref())
                                            .unwrap_or(&"new".to_string())
                                    ),
                                    role: "user".to_string(),
                                    content: vec![ResponseContentPart::InputText {
                                        text: text.clone(),
                                    }],
                                    status: Some("completed".to_string()),
                                });
                            }
                            ResponseInput::Items(current_items) => {
                                // Process all items, normalizing SimpleInputMessage to Message
                                for item in current_items.iter() {
                                    let normalized =
                                        crate::protocols::responses::normalize_input_item(item);
                                    items.push(normalized);
                                }
                            }
                        }

                        request_body.input = ResponseInput::Items(items);
                    }
                }

                // Always set store=false for upstream (we store internally)
                request_body.store = Some(false);

                // Remove conversation and previous_response_id (SGLang-specific, not forwarded)
                request_body.conversation = None;
                request_body.previous_response_id = None;

                // Filter out reasoning items from input - they're internal processing details
                if let ResponseInput::Items(ref mut items) = request_body.input {
                    items.retain(|item| !matches!(item, ResponseInputOutputItem::Reasoning { .. }));
                }

                let is_streaming = request_body.stream.unwrap_or(false);

                // Convert to JSON
                let mut payload = to_value(&request_body).map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to serialize request: {}", e),
                    )
                        .into_response()
                })?;

                // Strip SGLang-specific fields
                if let Some(obj) = payload.as_object_mut() {
                    obj.retain(|k, _| !SGLANG_FIELDS.contains(&k.as_str()));

                    // XAI (Grok models) special handling
                    let is_grok_model = obj
                        .get("model")
                        .and_then(|v| v.as_str())
                        .map(|m| m.starts_with("grok"))
                        .unwrap_or(false);

                    if is_grok_model {
                        // XAI doesn't support the full OpenAI item type input
                        // Strip extra fields from input messages (id, status)
                        // Normalize content types: output_text -> input_text
                        if let Some(input_arr) = obj.get_mut("input").and_then(Value::as_array_mut)
                        {
                            for item_obj in input_arr.iter_mut().filter_map(Value::as_object_mut) {
                                // Remove fields not universally supported
                                item_obj.remove("id");
                                item_obj.remove("status");

                                // Normalize content types to input_text (xAI compatibility)
                                if let Some(content_arr) =
                                    item_obj.get_mut("content").and_then(Value::as_array_mut)
                                {
                                    for content_obj in
                                        content_arr.iter_mut().filter_map(Value::as_object_mut)
                                    {
                                        // Change output_text to input_text
                                        if content_obj.get("type").and_then(Value::as_str)
                                            == Some("output_text")
                                        {
                                            content_obj.insert(
                                                "type".to_string(),
                                                Value::String("input_text".to_string()),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                (payload, is_streaming)
            }
        };

        // Store payload output
        ctx.state.payload = Some(PayloadOutput {
            json_payload: payload,
            is_streaming,
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestBuilding"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dashmap::DashMap;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::{
            chat::{ChatCompletionRequest, ChatMessage, MessageContent},
            responses::ResponsesRequest,
        },
        routers::openai::context::{ContextOutput, RequestInput, SharedComponents},
    };

    async fn create_test_components(worker_urls: Vec<String>) -> Arc<SharedComponents> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());

        let mcp_config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: vec![],
            inventory: Default::default(),
        };
        let mcp_manager = Arc::new(
            McpManager::new(mcp_config, 10)
                .await
                .expect("Failed to create MCP manager"),
        );

        let response_storage = Arc::new(MemoryResponseStorage::new());
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

        Arc::new(SharedComponents {
            http_client: client,
            circuit_breaker,
            model_cache,
            mcp_manager,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            worker_urls,
        })
    }

    #[tokio::test]
    async fn test_request_building_stage_chat_request() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = RequestBuildingStage;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            stream: true,
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Verify payload output
        let payload = ctx.state.payload.as_ref().unwrap();
        assert!(payload.is_streaming);
        assert!(payload.json_payload.is_object());

        let obj = payload.json_payload.as_object().unwrap();
        assert_eq!(obj.get("model").and_then(|v| v.as_str()), Some("gpt-4"));
        assert!(obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(false));

        // Verify SGLang-specific fields are stripped
        assert!(!obj.contains_key("conversation"));
        assert!(!obj.contains_key("previous_response_id"));
    }

    #[tokio::test]
    async fn test_request_building_stage_responses_no_context() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = RequestBuildingStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            conversation: Some("conv_123".to_string()),
            previous_response_id: Some("resp_456".to_string()),
            store: Some(true),
            stream: Some(false),
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Responses(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Verify payload output
        let payload = ctx.state.payload.as_ref().unwrap();
        assert!(!payload.is_streaming);

        let obj = payload.json_payload.as_object().unwrap();

        // Verify SGLang-specific fields are stripped
        assert!(!obj.contains_key("conversation"));
        assert!(!obj.contains_key("previous_response_id"));

        // Verify store is set to false (we store internally)
        assert_eq!(obj.get("store").and_then(|v| v.as_bool()), Some(false));
    }

    #[tokio::test]
    async fn test_request_building_stage_responses_with_context() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = RequestBuildingStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("World".to_string()),
            conversation: Some("conv_123".to_string()),
            stream: Some(false),
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Responses(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Simulate context loading output
        ctx.state.context = Some(ContextOutput {
            conversation_items: Some(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello".to_string(),
                }],
                status: Some("completed".to_string()),
            }]),
            conversation_id: Some("conv_123".to_string()),
            previous_response_id: None,
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Verify payload output
        let payload = ctx.state.payload.as_ref().unwrap();
        let obj = payload.json_payload.as_object().unwrap();

        // Verify input is now an Items array with conversation items
        let input = obj.get("input").unwrap();
        assert!(input.is_array());

        let input_arr = input.as_array().unwrap();
        assert_eq!(input_arr.len(), 2); // Previous "Hello" + Current "World"

        // First item should be the conversation history
        let first_item = input_arr[0].as_object().unwrap();
        assert_eq!(
            first_item.get("role").and_then(|v| v.as_str()),
            Some("user")
        );

        // Second item should be current request
        let second_item = input_arr[1].as_object().unwrap();
        assert_eq!(
            second_item.get("role").and_then(|v| v.as_str()),
            Some("user")
        );
    }

    #[tokio::test]
    async fn test_request_building_stage_grok_transformations() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = RequestBuildingStage;

        let request = ResponsesRequest {
            model: "grok-4-fast".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "Hello".to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: Some("completed".to_string()),
            }]),
            stream: Some(false),
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Responses(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());

        // Verify payload output
        let payload = ctx.state.payload.as_ref().unwrap();
        let obj = payload.json_payload.as_object().unwrap();

        // Verify xAI transformations were applied
        let input = obj.get("input").unwrap().as_array().unwrap();
        let item = input[0].as_object().unwrap();

        // id and status should be removed for Grok
        assert!(!item.contains_key("id"));
        assert!(!item.contains_key("status"));

        // output_text should be normalized to input_text
        let content = item.get("content").unwrap().as_array().unwrap();
        let content_obj = content[0].as_object().unwrap();
        assert_eq!(
            content_obj.get("type").and_then(|v| v.as_str()),
            Some("input_text")
        );
    }
}
