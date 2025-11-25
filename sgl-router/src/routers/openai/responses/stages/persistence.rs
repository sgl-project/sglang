//! Persistence stage for responses pipeline
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

use super::ResponsesStage;
use crate::routers::openai::{
    conversations::persist_conversation_items,
    responses::{
        utils::{mask_tools_as_mcp, patch_streaming_response_json},
        ResponsesRequestContext,
    },
};

/// Persistence stage for responses pipeline
pub struct ResponsesPersistenceStage;

#[async_trait]
impl ResponsesStage for ResponsesPersistenceStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        // Get processed response
        let mut processed = ctx.state.processed.take().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Response processing stage not completed",
            )
                .into_response()
        })?;

        // Get context for previous_response_id
        let original_previous_response_id = ctx
            .state
            .context
            .as_ref()
            .and_then(|c| c.previous_response_id.as_deref());

        // Mask tools as MCP
        mask_tools_as_mcp(&mut processed.json_response, ctx.request());

        // Patch response with metadata
        patch_streaming_response_json(
            &mut processed.json_response,
            ctx.request(),
            original_previous_response_id,
        );

        // Persist conversation items and response
        if let Err(err) = persist_conversation_items(
            ctx.dependencies.conversation_storage.clone(),
            ctx.dependencies.conversation_item_storage.clone(),
            ctx.dependencies.response_storage.clone(),
            &processed.json_response,
            ctx.request(),
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
        "ResponsesPersistence"
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use dashmap::DashMap;
    use serde_json::json;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::responses::{ResponseInput, ResponsesRequest},
        routers::openai::responses::{
            mcp::ToolLoopState, ContextOutput, DiscoveryOutput, McpOutput, PayloadOutput,
            ProcessedResponse, ResponsesDependencies, ValidationOutput,
        },
    };

    async fn create_test_dependencies() -> Arc<ResponsesDependencies> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());
        let worker_urls = vec!["http://localhost:8000".to_string()];

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

        Arc::new(ResponsesDependencies {
            http_client: client,
            circuit_breaker,
            model_cache,
            worker_urls,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
        })
    }

    #[tokio::test]
    async fn test_persistence_stage_no_processed_response() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(Arc::new(request), None, None, dependencies);

        // No processed response set
        let stage = ResponsesPersistenceStage;
        let result = stage.execute(&mut ctx).await;

        // Should fail due to missing processed response
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_persistence_stage_success() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(Arc::new(request), None, None, dependencies);

        // Set all prerequisites
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://localhost:8000".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.context = Some(ContextOutput {
            conversation_items: None,
            conversation_id: None,
            previous_response_id: None,
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: json!({"model": "gpt-4", "input": "Hello"}),
            is_streaming: false,
        });
        ctx.state.mcp = Some(McpOutput {
            active: false,
            tool_loop_state: ToolLoopState {
                iteration: 0,
                total_calls: 0,
                conversation_history: vec![],
                original_input: ResponseInput::Text("Hello".to_string()),
            },
            max_iterations: 0,
        });
        ctx.state.processed = Some(ProcessedResponse {
            json_response: json!({
                "id": "resp_123",
                "model": "gpt-4",
                "output": []
            }),
        });

        let stage = ResponsesPersistenceStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_some()); // Should return final response
    }
}
