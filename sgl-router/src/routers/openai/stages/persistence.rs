//! Persistence stage
//!
//! This stage:
//! - Masks tools as MCP in response
//! - Patches response with metadata
//! - Stores response and conversation items
//! - Returns final JSON response

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use tracing::warn;

use super::PipelineStage;
use crate::routers::openai::{
    context::{RequestContext, RequestType},
    conversations::persist_conversation_items,
    responses::{mask_tools_as_mcp, patch_streaming_response_json},
};

/// Persistence stage
pub struct PersistenceStage;

#[async_trait]
impl PipelineStage for PersistenceStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Get processed response
        let mut processed = ctx.state.processed.take().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Response processing stage not completed",
            )
                .into_response()
        })?;

        // Only Responses API needs persistence
        let responses_req = match &ctx.input.request_type {
            RequestType::Responses(req) => req,
            RequestType::Chat(_) => {
                // For chat requests, just return the JSON response
                return Ok(Some(
                    (StatusCode::OK, Json(processed.json_response)).into_response(),
                ));
            }
        };

        // Get context for previous_response_id
        let original_previous_response_id = ctx
            .state
            .context
            .as_ref()
            .and_then(|c| c.previous_response_id.as_deref());

        // Mask tools as MCP
        mask_tools_as_mcp(&mut processed.json_response, responses_req);

        // Patch response with metadata
        patch_streaming_response_json(
            &mut processed.json_response,
            responses_req,
            original_previous_response_id,
        );

        // Persist conversation items and response
        if let Err(err) = persist_conversation_items(
            ctx.components.conversation_storage.clone(),
            ctx.components.conversation_item_storage.clone(),
            ctx.components.response_storage.clone(),
            &processed.json_response,
            responses_req,
        )
        .await
        {
            warn!("Failed to persist conversation items: {}", err);
        }

        // Return final response
        Ok(Some(
            (StatusCode::OK, Json(processed.json_response)).into_response(),
        ))
    }

    fn name(&self) -> &'static str {
        "Persistence"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dashmap::DashMap;
    use serde_json::json;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::{
            chat::{ChatCompletionRequest, ChatMessage, MessageContent},
            responses::{ResponseInput, ResponsesRequest},
        },
        routers::openai::context::{ProcessedResponse, RequestInput, SharedComponents},
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
    async fn test_persistence_stage_chat_request() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = PersistenceStage;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
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

        // Set processed response
        ctx.state.processed = Some(ProcessedResponse {
            json_response: json!({
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "Hi there!"}}]
            }),
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.is_some()); // Should return response
        let resp = response.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_persistence_stage_responses_request() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = PersistenceStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
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

        // Set processed response
        ctx.state.processed = Some(ProcessedResponse {
            json_response: json!({
                "id": "resp_123",
                "output": [{"type": "message", "content": [{"type": "text", "text": "Hi!"}]}]
            }),
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.is_some());
        let resp = response.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_persistence_stage_no_processed_response() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = PersistenceStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
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

        // No processed response set
        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
