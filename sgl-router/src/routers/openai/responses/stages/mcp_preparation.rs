//! MCP Preparation stage for responses pipeline
//!
//! This stage:
//! - Detects MCP tools in request
//! - Ensures dynamic MCP client exists
//! - Transforms MCP tools to function format
//! - Initializes tool loop state

use async_trait::async_trait;
use axum::response::Response;
use serde_json::Value;

use super::ResponsesStage;
use crate::{
    protocols::responses::ResponseInput,
    routers::openai::{
        mcp::{ensure_request_mcp_client, ToolLoopState},
        responses::{McpOutput, ResponsesRequestContext},
        utils::event_types,
    },
};

/// MCP preparation stage for responses pipeline
pub struct ResponsesMcpPreparationStage;

/// Maximum iterations for MCP tool loop (safety limit)
const MAX_MCP_ITERATIONS: usize = 10;

impl ResponsesMcpPreparationStage {
    /// Transform MCP tools to function format in the payload
    fn prepare_mcp_tools(payload: &mut Value, tools: &[rmcp::model::Tool]) {
        if let Some(obj) = payload.as_object_mut() {
            // Remove any non-function tools from outgoing payload
            if let Some(v) = obj.get_mut("tools") {
                if let Some(arr) = v.as_array_mut() {
                    arr.retain(|item| {
                        item.get("type")
                            .and_then(|v| v.as_str())
                            .map(|s| s == event_types::ITEM_TYPE_FUNCTION)
                            .unwrap_or(false)
                    });
                }
            }

            // Build function tools for all discovered MCP tools
            let mut tools_json = Vec::new();
            for t in tools {
                let parameters = Value::Object((*t.input_schema).clone());
                let tool = serde_json::json!({
                    "type": event_types::ITEM_TYPE_FUNCTION,
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters
                });
                tools_json.push(tool);
            }

            if !tools_json.is_empty() {
                obj.insert("tools".to_string(), Value::Array(tools_json));
                obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
            }
        }
    }
}

#[async_trait]
impl ResponsesStage for ResponsesMcpPreparationStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        // Check if request has MCP tools - if yes, ensure dynamic MCP client exists
        if let Some(ref tools) = ctx.request().tools {
            ensure_request_mcp_client(&ctx.dependencies.mcp_manager, tools.as_slice()).await;
        }

        // Check if MCP manager has any tools available (static or dynamic)
        let has_mcp_tools = !ctx.dependencies.mcp_manager.list_tools().is_empty();

        if has_mcp_tools {
            // MCP is active - prepare payload and initialize tool loop state

            // Get tools before modifying payload to avoid borrow checker issues
            let tools = ctx.dependencies.mcp_manager.list_tools();

            // Get or create payload
            let payload_output = ctx.state.payload.as_mut().expect(
                "PayloadOutput must exist before MCP preparation (RequestBuilding stage should run first)",
            );

            // Transform MCP tools to function format
            Self::prepare_mcp_tools(&mut payload_output.json_payload, &tools);

            // Initialize tool loop state
            let tool_loop_state = ToolLoopState {
                iteration: 0,
                total_calls: 0,
                conversation_history: vec![],
                original_input: ctx.request().input.clone(),
            };

            ctx.state.mcp = Some(McpOutput {
                active: true,
                tool_loop_state,
                max_iterations: MAX_MCP_ITERATIONS,
            });
        } else {
            // No MCP tools - mark as inactive
            ctx.state.mcp = Some(McpOutput {
                active: false,
                tool_loop_state: ToolLoopState {
                    iteration: 0,
                    total_calls: 0,
                    conversation_history: vec![],
                    original_input: ctx.request().input.clone(),
                },
                max_iterations: 0,
            });
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ResponsesMcpPreparation"
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
            ContextOutput, DiscoveryOutput, PayloadOutput, ResponsesDependencies,
            ValidationOutput,
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
    async fn test_mcp_preparation_stage_no_tools() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            tools: None,
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        // Set prerequisites
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
            json_payload: json!({
                "model": "gpt-4",
                "input": "Hello"
            }),
            is_streaming: false,
        });

        let stage = ResponsesMcpPreparationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert!(ctx.state.mcp.is_some());

        let mcp = ctx.state.mcp.unwrap();
        assert_eq!(mcp.active, false);
        assert_eq!(mcp.max_iterations, 0);
    }

    #[tokio::test]
    async fn test_mcp_preparation_stage_tool_loop_state() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        // Set prerequisites
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
            json_payload: json!({
                "model": "gpt-4",
                "input": "Hello"
            }),
            is_streaming: false,
        });

        let stage = ResponsesMcpPreparationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(ctx.state.mcp.is_some());

        let mcp = ctx.state.mcp.unwrap();
        assert_eq!(mcp.tool_loop_state.iteration, 0);
        assert_eq!(mcp.tool_loop_state.total_calls, 0);
        assert!(mcp.tool_loop_state.conversation_history.is_empty());
    }
}
