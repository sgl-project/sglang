//! MCP Loop Backend implementation for gRPC Harmony router
//!
//! This module implements [`McpLoopBackend`] for the gRPC Harmony router,
//! enabling use of the shared MCP tool loop executor.

use async_trait::async_trait;
use axum::response::Response;
use serde_json::json;
use tracing::debug;

use super::responses::{
    build_next_request_with_tools, convert_mcp_tools_to_response_tools, HarmonyResponsesContext,
};
use crate::{
    mcp::{McpManager, Tool as McpTool},
    protocols::{
        common::{FunctionCallResponse, ToolCall},
        responses::{ResponseStatus, ResponsesRequest, ResponsesResponse},
    },
    routers::{
        grpc::{
            common::responses::{build_mcp_call_item, build_mcp_list_tools_item},
            harmony::processor::ResponsesIterationResult,
        },
        mcp::tool_loop::{
            extract_server_label, IterationOutcome, McpLoopBackend, ParsedToolCall,
            ToolExecutionResult,
        },
    },
};

// ============================================================================
// Backend State
// ============================================================================

/// Backend implementation for gRPC Harmony router MCP tool loop
///
/// This struct holds all the state and dependencies needed to execute
/// the MCP tool loop for the gRPC Harmony router.
pub struct GrpcHarmonyMcpBackend<'a> {
    /// Harmony Responses context with pipeline and components
    ctx: &'a HarmonyResponsesContext,

    /// Model name (for metrics)
    model_name: String,

    /// Server label for MCP metadata
    server_label: String,

    /// Tool execution results (converted on-demand for resume/metadata)
    tool_results: Vec<ToolExecutionResult>,

    /// Analysis from last iteration (for building resume request)
    last_analysis: Option<String>,

    /// Partial text from last iteration (for building resume request)
    last_partial_text: String,

    /// Current iteration (for request preparation)
    iteration: usize,
}

impl<'a> GrpcHarmonyMcpBackend<'a> {
    /// Create a new backend instance
    pub fn new(ctx: &'a HarmonyResponsesContext, request: &ResponsesRequest) -> Self {
        let server_label = extract_server_label(request);

        Self {
            ctx,
            model_name: request.model.clone(),
            server_label,
            tool_results: Vec::new(),
            last_analysis: None,
            last_partial_text: String::new(),
            iteration: 0,
        }
    }
}

// ============================================================================
// Trait Implementation
// ============================================================================

#[async_trait]
impl McpLoopBackend for GrpcHarmonyMcpBackend<'_> {
    type Request = ResponsesRequest;
    type Response = ResponsesResponse;
    type Error = Response;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn server_label(&self) -> &str {
        &self.server_label
    }

    fn prepare_request_with_mcp_tools(
        &mut self,
        request: &mut Self::Request,
        mcp_tools: &[McpTool],
        iteration: usize,
    ) {
        self.iteration = iteration;

        // Only add MCP tools on first iteration
        if iteration == 1 && !mcp_tools.is_empty() {
            let mcp_response_tools = convert_mcp_tools_to_response_tools(mcp_tools);

            let mut all_tools = request.tools.clone().unwrap_or_default();
            all_tools.extend(mcp_response_tools);
            request.tools = Some(all_tools);

            debug!(
                mcp_tool_count = mcp_tools.len(),
                total_tool_count = request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                "Added MCP tools to Harmony Responses request"
            );
        }
    }

    async fn execute_iteration(
        &mut self,
        request: &Self::Request,
    ) -> Result<IterationOutcome<Self::Response>, Self::Error> {
        // Execute through full Harmony pipeline
        let iteration_result = self
            .ctx
            .pipeline
            .execute_harmony_responses(request, self.ctx)
            .await?;

        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                usage: _,
                request_id: _,
            } => {
                // Convert ToolCall to ParsedToolCall
                let parsed_calls: Vec<ParsedToolCall> = tool_calls
                    .iter()
                    .map(|tc| ParsedToolCall {
                        call_id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone().unwrap_or_default(),
                    })
                    .collect();

                // Store iteration data for resume request building
                self.last_analysis = analysis;
                self.last_partial_text = partial_text;
                // Clear state for new iteration - tool calls will be recorded via record_tool_execution
                self.tool_results.clear();

                debug!(
                    tool_call_count = parsed_calls.len(),
                    has_analysis = self.last_analysis.is_some(),
                    partial_text_len = self.last_partial_text.len(),
                    "Harmony iteration found tool calls"
                );

                Ok(IterationOutcome::ToolCallsFound(parsed_calls))
            }
            ResponsesIterationResult::Completed { response, usage: _ } => {
                debug!(
                    output_items = response.output.len(),
                    "Harmony iteration completed"
                );
                Ok(IterationOutcome::Completed(*response))
            }
        }
    }

    fn record_tool_execution(&mut self, result: &ToolExecutionResult) {
        self.tool_results.push(result.clone());
    }

    fn build_resume_request(
        &self,
        base_request: &Self::Request,
    ) -> Result<Self::Request, Self::Error> {
        // Convert tool results to ToolCall format for the request
        let tool_calls: Vec<ToolCall> = self
            .tool_results
            .iter()
            .map(|r| ToolCall {
                id: r.call_id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: r.name.clone(),
                    arguments: Some(r.arguments.clone()),
                },
            })
            .collect();

        build_next_request_with_tools(
            base_request.clone(),
            tool_calls,
            &self.tool_results,
            self.last_analysis.clone(),
            self.last_partial_text.clone(),
        )
        .map_err(|e| *e)
    }

    fn inject_mcp_metadata(&self, response: &mut Self::Response, mcp_manager: &McpManager) {
        if self.tool_results.is_empty() {
            return;
        }

        // Prepend mcp_list_tools item
        let mcp_list_tools = build_mcp_list_tools_item(mcp_manager, &self.server_label);
        response.output.insert(0, mcp_list_tools);

        // Append all mcp_call items (converted on-demand)
        for result in &self.tool_results {
            let mcp_call = build_mcp_call_item(
                &result.name,
                &result.arguments,
                &result.output,
                &self.server_label,
                result.success,
                result.error.as_deref(),
            );
            response.output.push(mcp_call);
        }

        debug!(
            mcp_calls = self.tool_results.len(),
            "Injected MCP metadata into Harmony response"
        );
    }

    fn mark_incomplete(&self, response: &mut Self::Response) {
        response.status = ResponseStatus::Completed;
        response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));
    }
}
