//! Context Loading stage
//!
//! This stage:
//! - Loads previous response chain (if previous_response_id provided)
//! - Loads conversation history (if conversation ID provided)
//! - Builds conversation context for the request

use async_trait::async_trait;
use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;
use tracing::warn;

use super::PipelineStage;
use crate::{
    data_connector::{ConversationId, ListParams, ResponseId, SortOrder},
    protocols::responses::{ResponseContentPart, ResponseInputOutputItem},
    routers::openai::context::{ContextOutput, RequestContext, RequestType},
};

/// Context loading stage
pub struct ContextLoadingStage;

impl ContextLoadingStage {
    /// Maximum conversation history items to load
    const MAX_CONVERSATION_HISTORY_ITEMS: usize = 1000;
}

#[async_trait]
impl PipelineStage for ContextLoadingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Only applicable to Responses requests
        let responses_req = match &ctx.input.request_type {
            RequestType::Responses(req) => req,
            RequestType::Chat(_) => {
                // Skip for chat requests
                return Ok(None);
            }
        };

        let mut conversation_items: Vec<ResponseInputOutputItem> = Vec::new();

        // Load previous response chain
        if let Some(prev_id_str) = &responses_req.previous_response_id {
            let prev_id = ResponseId::from(prev_id_str.as_str());
            match ctx
                .components
                .response_storage
                .get_response_chain(&prev_id, None)
                .await
            {
                Ok(chain) => {
                    for stored in chain.responses.iter() {
                        // Convert input items from stored input (JSON array)
                        if let Some(input_arr) = stored.input.as_array() {
                            for item in input_arr {
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.clone(),
                                ) {
                                    Ok(input_item) => {
                                        conversation_items.push(input_item);
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize stored input item: {}. Item: {}",
                                            e, item
                                        );
                                    }
                                }
                            }
                        }

                        // Convert output items from stored output (JSON array)
                        if let Some(output_arr) = stored.output.as_array() {
                            for item in output_arr {
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.clone(),
                                ) {
                                    Ok(output_item) => {
                                        conversation_items.push(output_item);
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize stored output item: {}. Item: {}",
                                            e, item
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to load previous response chain for {}: {}",
                        prev_id_str, e
                    );
                }
            }
        }

        // Load conversation history
        if let Some(conv_id_str) = &responses_req.conversation {
            let conv_id = ConversationId::from(conv_id_str.as_str());

            // Verify conversation exists
            if let Ok(None) = ctx
                .components
                .conversation_storage
                .get_conversation(&conv_id)
                .await
            {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(json!({"error": "Conversation not found"})),
                )
                    .into_response());
            }

            // Load conversation history (ascending order for chronological context)
            let params = ListParams {
                limit: Self::MAX_CONVERSATION_HISTORY_ITEMS,
                order: SortOrder::Asc,
                after: None,
            };

            match ctx
                .components
                .conversation_item_storage
                .list_items(&conv_id, params)
                .await
            {
                Ok(stored_items) => {
                    for item in stored_items.into_iter() {
                        // Include messages, function calls, and function call outputs
                        // Skip reasoning items as they're internal processing details
                        match item.item_type.as_str() {
                            "message" => {
                                match serde_json::from_value::<Vec<ResponseContentPart>>(
                                    item.content.clone(),
                                ) {
                                    Ok(content_parts) => {
                                        conversation_items.push(ResponseInputOutputItem::Message {
                                            id: item.id.0.clone(),
                                            role: item
                                                .role
                                                .clone()
                                                .unwrap_or_else(|| "user".to_string()),
                                            content: content_parts,
                                            status: item.status.clone(),
                                        });
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize message content: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            "function_call" => {
                                // The entire function_call item is stored in content field
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.content.clone(),
                                ) {
                                    Ok(func_call) => conversation_items.push(func_call),
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize function_call: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            "function_call_output" => {
                                // The entire function_call_output item is stored in content field
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.content.clone(),
                                ) {
                                    Ok(func_output) => {
                                        conversation_items.push(func_output);
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize function_call_output: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            "reasoning" => {
                                // Skip reasoning items - they're internal processing details
                            }
                            _ => {
                                // Skip unknown item types
                                warn!("Unknown item type in conversation: {}", item.item_type);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to load conversation history for {}: {}",
                        conv_id_str, e
                    );
                }
            }
        }

        // Store context output
        ctx.state.context = Some(ContextOutput {
            conversation_items: if conversation_items.is_empty() {
                None
            } else {
                Some(conversation_items)
            },
            conversation_id: responses_req.conversation.clone(),
            previous_response_id: responses_req.previous_response_id.clone(),
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ContextLoading"
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

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
        routers::openai::context::{SharedComponents, ValidationOutput},
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
    async fn test_context_loading_stage_chat_request() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = ContextLoadingStage;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
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

        // Context should not be set for chat requests
        assert!(ctx.state.context.is_none());
    }

    #[tokio::test]
    async fn test_context_loading_stage_responses_no_context() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = ContextLoadingStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: crate::protocols::responses::ResponseInput::Text("Hello".to_string()),
            conversation: None,
            previous_response_id: None,
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Responses(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Set validation output (prerequisite)
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Context should be set but with no items
        let context = ctx.state.context.as_ref().unwrap();
        assert!(context.conversation_items.is_none());
        assert!(context.conversation_id.is_none());
        assert!(context.previous_response_id.is_none());
    }

    #[tokio::test]
    async fn test_context_loading_stage_conversation_not_found() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = ContextLoadingStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: crate::protocols::responses::ResponseInput::Text("Hello".to_string()),
            conversation: Some("non-existent-conv".to_string()),
            previous_response_id: None,
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Responses(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());

        let response = result.unwrap_err();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
