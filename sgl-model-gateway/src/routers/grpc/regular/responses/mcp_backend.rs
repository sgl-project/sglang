//! MCP Loop Backend implementation for gRPC Regular router
//!
//! This module implements [`McpLoopBackend`] for the gRPC Regular router,
//! enabling use of the shared MCP tool loop executor.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use serde_json::{json, Value};
use tracing::error;

use super::{context::ResponsesContext, conversions};
use crate::{
    mcp::{McpManager, Tool as McpTool},
    protocols::{
        chat::ChatCompletionRequest,
        common::{Function, Tool, ToolChoice, ToolChoiceValue},
        responses::{
            self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseStatus,
            ResponsesRequest, ResponsesResponse,
        },
    },
    routers::{
        error as router_error,
        grpc::common::responses::{build_mcp_call_item, build_mcp_list_tools_item},
        mcp::tool_loop::{
            extract_server_label, IterationOutcome, McpLoopBackend, ParsedToolCall,
            ToolExecutionResult,
        },
    },
};

// ============================================================================
// Backend State
// ============================================================================

/// Backend implementation for gRPC Regular router MCP tool loop
///
/// This struct holds all the state and dependencies needed to execute
/// the MCP tool loop for the gRPC Regular router.
pub struct GrpcRegularMcpBackend<'a> {
    /// Responses context with pipeline and components
    ctx: &'a ResponsesContext,

    /// HTTP headers for backend requests
    headers: Option<http::HeaderMap>,

    /// Model ID for routing
    model_id: Option<String>,

    /// Response ID for the final response
    response_id: Option<String>,

    /// Original request (for conversion context)
    original_request: &'a ResponsesRequest,

    /// Model name (cached for metrics)
    model_name: String,

    /// Server label for MCP metadata
    server_label: String,

    /// MCP tools converted to Chat format (cached)
    mcp_chat_tools: Vec<Tool>,

    /// Tool execution results (converted on-demand for resume/metadata)
    tool_results: Vec<ToolExecutionResult>,

    /// Original user input (for resume request building)
    original_input: ResponseInput,

    /// Current iteration number
    iteration: usize,
}

impl<'a> GrpcRegularMcpBackend<'a> {
    /// Create a new backend instance
    pub fn new(
        ctx: &'a ResponsesContext,
        original_request: &'a ResponsesRequest,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        response_id: Option<String>,
    ) -> Self {
        let server_label = extract_server_label(original_request);

        // Get MCP tools and convert to chat format
        let mcp_tools = ctx.mcp_manager.list_tools();
        let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&mcp_tools);

        Self {
            ctx,
            headers,
            model_id,
            response_id,
            original_request,
            model_name: original_request.model.clone(),
            server_label,
            mcp_chat_tools,
            tool_results: Vec::new(),
            original_input: original_request.input.clone(),
            iteration: 0,
        }
    }
}

// ============================================================================
// Trait Implementation
// ============================================================================

#[async_trait]
impl McpLoopBackend for GrpcRegularMcpBackend<'_> {
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
        _request: &mut Self::Request,
        _mcp_tools: &[McpTool],
        iteration: usize,
    ) {
        // We handle tool preparation in execute_iteration since we convert
        // ResponsesRequest -> ChatCompletionRequest there
        self.iteration = iteration;
    }

    async fn execute_iteration(
        &mut self,
        request: &Self::Request,
    ) -> Result<IterationOutcome<Self::Response>, Self::Error> {
        // Convert ResponsesRequest to ChatCompletionRequest
        let mut chat_request = conversions::responses_to_chat(request).map_err(|e| {
            error!(
                function = "execute_iteration",
                iteration = self.iteration,
                error = %e,
                "Failed to convert ResponsesRequest to ChatCompletionRequest"
            );
            router_error::bad_request(
                "convert_request_failed",
                format!("Failed to convert request: {}", e),
            )
        })?;

        // Prepare tools and tool_choice for this iteration
        prepare_chat_tools_and_choice(&mut chat_request, &self.mcp_chat_tools, self.iteration);

        // Execute chat pipeline
        let chat_response = self
            .ctx
            .pipeline
            .execute_chat_for_responses(
                Arc::new(chat_request),
                self.headers.clone(),
                self.model_id.clone(),
                self.ctx.components.clone(),
            )
            .await?;

        // Extract tool calls from response
        let tool_calls = extract_tool_calls_from_chat(&chat_response);

        if tool_calls.is_empty() {
            // No tool calls - convert to final response
            let response = conversions::chat_to_responses(
                &chat_response,
                self.original_request,
                self.response_id.clone(),
            )
            .map_err(|e| {
                error!(
                    function = "execute_iteration",
                    iteration = self.iteration,
                    error = %e,
                    "Failed to convert ChatCompletionResponse to ResponsesResponse"
                );
                router_error::internal_error(
                    "convert_to_responses_format_failed",
                    format!("Failed to convert to responses format: {}", e),
                )
            })?;

            Ok(IterationOutcome::Completed(response))
        } else {
            Ok(IterationOutcome::ToolCallsFound(tool_calls))
        }
    }

    fn record_tool_execution(&mut self, result: &ToolExecutionResult) {
        self.tool_results.push(result.clone());
    }

    fn build_resume_request(
        &self,
        base_request: &Self::Request,
    ) -> Result<Self::Request, Self::Error> {
        // Start with original input
        let mut input_items = match &self.original_input {
            ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
                id: format!("msg_u_{}", self.iteration),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text: text.clone() }],
                status: Some("completed".to_string()),
            }],
            ResponseInput::Items(items) => {
                items.iter().map(responses::normalize_input_item).collect()
            }
        };

        // Convert tool results to conversation history on-demand
        for result in &self.tool_results {
            input_items.push(ResponseInputOutputItem::FunctionToolCall {
                id: result.call_id.clone(),
                call_id: result.call_id.clone(),
                name: result.name.clone(),
                arguments: result.arguments.clone(),
                output: Some(result.output.clone()),
                status: Some("completed".to_string()),
            });
        }

        // Build new request
        Ok(ResponsesRequest {
            input: ResponseInput::Items(input_items),
            model: base_request.model.clone(),
            instructions: base_request.instructions.clone(),
            tools: base_request.tools.clone(),
            max_output_tokens: base_request.max_output_tokens,
            temperature: base_request.temperature,
            top_p: base_request.top_p,
            stream: Some(false),
            store: Some(false),
            background: Some(false),
            max_tool_calls: base_request.max_tool_calls,
            tool_choice: base_request.tool_choice.clone(),
            parallel_tool_calls: base_request.parallel_tool_calls,
            previous_response_id: None,
            conversation: None,
            user: base_request.user.clone(),
            metadata: base_request.metadata.clone(),
            include: base_request.include.clone(),
            reasoning: base_request.reasoning.clone(),
            service_tier: base_request.service_tier.clone(),
            top_logprobs: base_request.top_logprobs,
            truncation: base_request.truncation.clone(),
            text: base_request.text.clone(),
            request_id: None,
            priority: base_request.priority,
            frequency_penalty: base_request.frequency_penalty,
            presence_penalty: base_request.presence_penalty,
            stop: base_request.stop.clone(),
            top_k: base_request.top_k,
            min_p: base_request.min_p,
            repetition_penalty: base_request.repetition_penalty,
        })
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
    }

    fn mark_incomplete(&self, response: &mut Self::Response) {
        response.status = ResponseStatus::Completed;
        response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert MCP tools to Chat API tool format
fn convert_mcp_tools_to_chat_tools(mcp_tools: &[McpTool]) -> Vec<Tool> {
    mcp_tools
        .iter()
        .map(|tool_info| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: tool_info.name.to_string(),
                description: tool_info.description.as_ref().map(|d| d.to_string()),
                parameters: Value::Object((*tool_info.input_schema).clone()),
                strict: None,
            },
        })
        .collect()
}

/// Merge MCP tools with request tools and set tool_choice
fn prepare_chat_tools_and_choice(
    chat_request: &mut ChatCompletionRequest,
    mcp_chat_tools: &[Tool],
    iteration: usize,
) {
    // Merge function tools from request with MCP tools
    let mut all_tools = chat_request.tools.clone().unwrap_or_default();
    all_tools.extend(mcp_chat_tools.iter().cloned());
    chat_request.tools = Some(all_tools);

    // Set tool_choice based on iteration
    // - Iteration 1: Use user's tool_choice or default to auto
    // - Iteration 2+: Always use auto to avoid infinite loops
    chat_request.tool_choice = if iteration <= 1 {
        chat_request
            .tool_choice
            .clone()
            .or(Some(ToolChoice::Value(ToolChoiceValue::Auto)))
    } else {
        Some(ToolChoice::Value(ToolChoiceValue::Auto))
    };
}

/// Extract tool calls from chat response
fn extract_tool_calls_from_chat(
    response: &crate::protocols::chat::ChatCompletionResponse,
) -> Vec<ParsedToolCall> {
    let Some(choice) = response.choices.first() else {
        return Vec::new();
    };

    let Some(tool_calls) = &choice.message.tool_calls else {
        return Vec::new();
    };

    tool_calls
        .iter()
        .map(|tc| ParsedToolCall {
            call_id: tc.id.clone(),
            name: tc.function.name.clone(),
            arguments: tc
                .function
                .arguments
                .clone()
                .unwrap_or_else(|| "{}".to_string()),
        })
        .collect()
}
