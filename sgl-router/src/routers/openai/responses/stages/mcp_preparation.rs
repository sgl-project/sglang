//! MCP Preparation stage for responses pipeline
//!
//! This stage:
//! - Detects MCP tools in request
//! - Ensures dynamic MCP client exists
//! - Transforms MCP tools to function format
//! - Initializes tool loop state

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::Value;

use super::ResponsesStage;
use crate::routers::openai::{
    responses::{
        mcp::{ensure_request_mcp_client, ToolLoopState},
        McpOutput, ResponsesRequestContext,
    },
    utils::event_types,
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
            let payload_output = ctx.state.payload.as_mut().ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Payload building stage not completed before MCP preparation",
                )
                    .into_response()
            })?;

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
