//! MCP Loop Backend implementation for OpenAI router
//!
//! This module implements [`McpLoopBackend`] for the OpenAI router,
//! enabling use of the shared MCP tool loop executor.

use async_trait::async_trait;
use axum::http::HeaderMap;
use serde_json::{json, to_value, Value};
use tracing::debug;

use super::mcp::{build_mcp_call_item, build_mcp_list_tools_item, extract_function_call};
use crate::{
    mcp::{McpManager, Tool as McpTool},
    protocols::responses::{ResponseInput, ResponsesRequest},
    routers::{
        header_utils::apply_request_headers,
        mcp::tool_loop::{
            extract_server_label, IterationOutcome, McpLoopBackend, ParsedToolCall,
            ToolExecutionResult,
        },
    },
};

// ============================================================================
// Backend State
// ============================================================================

/// Backend implementation for OpenAI router MCP tool loop
///
/// This struct holds all the state and dependencies needed to execute
/// the MCP tool loop for the OpenAI router using HTTP requests.
pub struct OpenAIMcpBackend<'a> {
    /// HTTP client for upstream requests
    client: &'a reqwest::Client,

    /// Upstream URL
    url: &'a str,

    /// HTTP headers for requests
    headers: Option<&'a HeaderMap>,

    /// Base payload template (with cleaned fields)
    base_payload: Value,

    /// Tools JSON for resume payloads
    tools_json: Value,

    /// Model name (for metrics)
    model_name: String,

    /// Server label for MCP metadata
    server_label: String,

    /// Tool execution results (converted on-demand for resume/metadata)
    tool_results: Vec<ToolExecutionResult>,

    /// Original user input (for resume request building)
    original_input: ResponseInput,

    /// Current iteration
    iteration: usize,
}

impl<'a> OpenAIMcpBackend<'a> {
    /// Create a new backend instance
    pub fn new(
        client: &'a reqwest::Client,
        url: &'a str,
        headers: Option<&'a HeaderMap>,
        initial_payload: Value,
        original_body: &'a ResponsesRequest,
    ) -> Self {
        let server_label = extract_server_label(original_body);

        // Keep initial_payload as base template (already has fields cleaned)
        let tools_json = initial_payload.get("tools").cloned().unwrap_or(json!([]));

        Self {
            client,
            url,
            headers,
            base_payload: initial_payload.clone(),
            tools_json,
            model_name: original_body.model.clone(),
            server_label,
            tool_results: Vec::new(),
            original_input: original_body.input.clone(),
            iteration: 0,
        }
    }
}

// ============================================================================
// Trait Implementation
// ============================================================================

#[async_trait]
impl McpLoopBackend for OpenAIMcpBackend<'_> {
    type Request = Value;
    type Response = Value;
    type Error = String;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn server_label(&self) -> &str {
        &self.server_label
    }

    fn prepare_request_with_mcp_tools(
        &mut self,
        _request: &mut Self::Request,
        _mcp_tools: &[McpTool],
        iteration: usize,
    ) {
        // OpenAI payload is already prepared with tools in the caller
        // We just track the iteration
        self.iteration = iteration;
    }

    async fn execute_iteration(
        &mut self,
        request: &Self::Request,
    ) -> Result<IterationOutcome<Self::Response>, Self::Error> {
        // Make request to upstream
        let request_builder = self.client.post(self.url).json(request);
        let request_builder = if let Some(headers) = self.headers {
            apply_request_headers(headers, request_builder, true)
        } else {
            request_builder
        };

        let response = request_builder
            .send()
            .await
            .map_err(|e| format!("upstream request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("upstream error {}: {}", status, body));
        }

        let response_json = response
            .json::<Value>()
            .await
            .map_err(|e| format!("parse response: {}", e))?;

        // Check for function call
        if let Some((call_id, tool_name, args_json_str)) = extract_function_call(&response_json) {
            debug!(
                iteration = self.iteration,
                tool_name = %tool_name,
                call_id = %call_id,
                "Tool call found in response"
            );

            Ok(IterationOutcome::ToolCallsFound(vec![ParsedToolCall {
                call_id,
                name: tool_name,
                arguments: args_json_str,
            }]))
        } else {
            // No more tool calls
            Ok(IterationOutcome::Completed(response_json))
        }
    }

    fn record_tool_execution(&mut self, result: &ToolExecutionResult) {
        self.tool_results.push(result.clone());
    }

    fn build_resume_request(
        &self,
        _base_request: &Self::Request,
    ) -> Result<Self::Request, Self::Error> {
        // Clone the base payload which already has cleaned fields
        let mut payload = self.base_payload.clone();

        let obj = payload
            .as_object_mut()
            .ok_or_else(|| "payload not an object".to_string())?;

        // Build input array: start with original user input
        let mut input_array = Vec::new();

        // Add original user message
        match &self.original_input {
            ResponseInput::Text(text) => {
                let user_item = json!({
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": text }]
                });
                input_array.push(user_item);
            }
            ResponseInput::Items(items) => {
                if let Ok(items_value) = to_value(items) {
                    if let Some(items_arr) = items_value.as_array() {
                        input_array.extend_from_slice(items_arr);
                    }
                }
            }
        }

        // Convert tool results to conversation history on-demand
        for result in &self.tool_results {
            super::mcp::record_tool_call(
                &mut input_array,
                &result.call_id,
                &result.name,
                &result.arguments,
                &result.output,
            );
        }

        obj.insert("input".to_string(), Value::Array(input_array));

        // Use the transformed tools (function tools, not MCP tools)
        if let Some(tools_arr) = self.tools_json.as_array() {
            if !tools_arr.is_empty() {
                obj.insert("tools".to_string(), self.tools_json.clone());
            }
        }

        // Set streaming mode to false for non-streaming loop
        obj.insert("stream".to_string(), Value::Bool(false));
        obj.insert("store".to_string(), Value::Bool(false));

        Ok(payload)
    }

    fn inject_mcp_metadata(&self, response: &mut Self::Response, mcp_manager: &McpManager) {
        if self.tool_results.is_empty() {
            return;
        }

        // Build mcp_list_tools item
        let list_tools_item = build_mcp_list_tools_item(mcp_manager, &self.server_label);

        // Insert at beginning of output array
        if let Some(output_array) = response.get_mut("output").and_then(|v| v.as_array_mut()) {
            output_array.insert(0, list_tools_item);

            // Build mcp_call items on-demand from tool results
            let mut insert_pos = 1;
            for result in &self.tool_results {
                let mcp_call_item = build_mcp_call_item(
                    &result.name,
                    &result.arguments,
                    &result.output,
                    &self.server_label,
                    result.success,
                    result.error.as_deref(),
                );
                output_array.insert(insert_pos, mcp_call_item);
                insert_pos += 1;
            }
        }
    }

    fn mark_incomplete(&self, response: &mut Self::Response) {
        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));
            obj.insert(
                "incomplete_details".to_string(),
                json!({ "reason": "max_tool_calls" }),
            );
        }
    }
}
