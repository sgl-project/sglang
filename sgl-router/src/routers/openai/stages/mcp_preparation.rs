//! MCP Preparation stage
//!
//! This stage:
//! - Detects MCP tools in request
//! - Ensures dynamic MCP client exists
//! - Transforms MCP tools to function format
//! - Initializes tool loop state

use async_trait::async_trait;
use axum::response::Response;
use serde_json::Value;

use super::PipelineStage;
use crate::{
    protocols::responses::ResponseInput,
    routers::openai::{
        context::{McpOutput, RequestContext, RequestType},
        mcp::{ensure_request_mcp_client, ToolLoopState},
        utils::event_types,
    },
};

/// MCP preparation stage
pub struct McpPreparationStage;

/// Maximum iterations for MCP tool loop (safety limit)
const MAX_MCP_ITERATIONS: usize = 10;

impl McpPreparationStage {
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
impl PipelineStage for McpPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Only applicable to Responses requests
        let responses_req = match &ctx.input.request_type {
            RequestType::Responses(req) => req,
            RequestType::Chat(_) => {
                // Skip for chat requests - mark as inactive
                ctx.state.mcp = Some(McpOutput {
                    active: false,
                    tool_loop_state: ToolLoopState {
                        iteration: 0,
                        total_calls: 0,
                        conversation_history: vec![],
                        original_input: ResponseInput::Text(String::new()),
                    },
                    max_iterations: 0,
                });
                return Ok(None);
            }
        };

        // Check if request has MCP tools - if yes, ensure dynamic MCP client exists
        if let Some(ref tools) = responses_req.tools {
            ensure_request_mcp_client(&ctx.components.mcp_manager, tools.as_slice()).await;
        }

        // Check if MCP manager has any tools available (static or dynamic)
        let has_mcp_tools = !ctx.components.mcp_manager.list_tools().is_empty();

        if has_mcp_tools {
            // MCP is active - prepare payload and initialize tool loop state

            // Get tools before modifying payload to avoid borrow checker issues
            let tools = ctx.components.mcp_manager.list_tools();

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
                original_input: responses_req.input.clone(),
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
                    original_input: responses_req.input.clone(),
                },
                max_iterations: 0,
            });
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "McpPreparation"
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
            responses::{ResponseInput, ResponsesRequest},
        },
        routers::openai::context::{PayloadOutput, RequestInput, SharedComponents},
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
    async fn test_mcp_preparation_stage_chat_request() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = McpPreparationStage;

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

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // For chat requests, MCP should be inactive
        let mcp = ctx.state.mcp.as_ref().unwrap();
        assert!(!mcp.active);
    }

    #[tokio::test]
    async fn test_mcp_preparation_stage_responses_no_tools() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = McpPreparationStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            tools: None,
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

        // Set up prerequisite payload output
        ctx.state.payload = Some(PayloadOutput {
            json_payload: serde_json::json!({
                "model": "gpt-4",
                "input": "Hello"
            }),
            is_streaming: false,
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Without tools, MCP should be inactive
        let mcp = ctx.state.mcp.as_ref().unwrap();
        assert!(!mcp.active);
        assert_eq!(mcp.max_iterations, 0);
    }

    #[tokio::test]
    async fn test_mcp_preparation_stage_tool_loop_state() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = McpPreparationStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Use the calculator".to_string()),
            tools: None,
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

        // Set up prerequisite payload output
        ctx.state.payload = Some(PayloadOutput {
            json_payload: serde_json::json!({
                "model": "gpt-4",
                "input": "Use the calculator"
            }),
            is_streaming: false,
        });

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());

        // Verify tool loop state initialization
        let mcp = ctx.state.mcp.as_ref().unwrap();
        assert_eq!(mcp.tool_loop_state.iteration, 0);
        assert_eq!(mcp.tool_loop_state.total_calls, 0);
        assert!(mcp.tool_loop_state.conversation_history.is_empty());

        // Verify original input is preserved
        match &mcp.tool_loop_state.original_input {
            ResponseInput::Text(text) => assert_eq!(text, "Use the calculator"),
            _ => panic!("Expected Text input"),
        }
    }
}
