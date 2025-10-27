//! Harmony Responses API serving layer with multi-turn MCP tool support
//!
//! This module implements the Harmony Responses API serving logic, orchestrating
//! full pipeline execution with MCP tool support for multi-turn conversations.
//!
//! ## Architecture
//!
//! Multi-turn pipeline orchestration (NOT just a tool loop):
//! - Serves Harmony Responses API requests end-to-end
//! - Each iteration executes FULL pipeline (worker selection + client acquisition + execution + parsing)
//! - Handles MCP tool execution and history building between iterations
//! - Clean separation: serving orchestration (this file) vs. pipeline stages (stages/)
//!
//! ## Flow
//!
//! ```text
//! loop {
//!     // Execute through FULL pipeline
//!     let result = pipeline.execute_harmony_responses(&request, &ctx).await?;
//!
//!     match result {
//!         ToolCallsFound { tool_calls, .. } => {
//!             // Execute MCP tools
//!             // Build next request with tool results
//!             // Continue loop
//!         }
//!         Completed { response, .. } => {
//!             return Ok(response);
//!         }
//!     }
//! }
//! ```
//!
//! ## Design Reference
//!
//! See `/Users/simolin/workspace/sglang/.claude/docs/harmony_pipeline/tool_loop_design.md`
//! for complete architecture, rationale, and implementation details.

use std::sync::Arc;

use axum::response::Response;
use serde_json::Value as JsonValue;

use crate::{
    mcp::McpManager,
    protocols::{
        common::ToolCall,
        responses::{ResponseInput, ResponsesRequest, ResponsesResponse},
    },
    routers::grpc::{
        context::SharedComponents, harmony::processor::ResponsesIterationResult,
        pipeline::RequestPipeline, utils,
    },
};

/// Maximum number of tool execution iterations to prevent infinite loops
const MAX_TOOL_ITERATIONS: usize = 10;

/// Context for Harmony Responses execution with MCP tool support
///
/// Contains all dependencies needed for multi-turn Responses API execution.
/// Cheap to clone (all Arc references).
#[derive(Clone)]
pub struct HarmonyResponsesContext {
    /// Pipeline for executing Harmony requests
    pub pipeline: Arc<RequestPipeline>,

    /// Shared components (tokenizer, parsers)
    pub components: Arc<SharedComponents>,

    /// MCP manager for tool execution
    pub mcp_manager: Arc<McpManager>,

    /// Optional streaming sender (for future streaming support)
    pub stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Result<String, String>>>,
}

impl HarmonyResponsesContext {
    /// Create a new Harmony Responses context
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        mcp_manager: Arc<McpManager>,
    ) -> Self {
        Self {
            pipeline,
            components,
            mcp_manager,
            stream_tx: None,
        }
    }

    /// Create with streaming support
    pub fn with_streaming(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        mcp_manager: Arc<McpManager>,
        stream_tx: tokio::sync::mpsc::UnboundedSender<Result<String, String>>,
    ) -> Self {
        Self {
            pipeline,
            components,
            mcp_manager,
            stream_tx: Some(stream_tx),
        }
    }
}

/// Serve Harmony Responses API request with multi-turn MCP tool support
///
/// This function orchestrates the multi-turn conversation flow:
/// 1. Execute request through full pipeline
/// 2. Check for tool calls in commentary channel
/// 3. If tool calls found:
///    - Execute MCP tools
///    - Build next request with tool results
///    - Repeat from step 1 (full pipeline re-execution)
/// 4. If no tool calls, return final response
///
/// # Architecture
///
/// Uses **external loop pattern**: wraps full pipeline execution rather than
/// implementing loop inside pipeline. Each iteration goes through:
/// - Worker selection (fresh selection based on current context)
/// - Client acquisition (new gRPC client if worker changed)
/// - Request building (Harmony prefill with complete history)
/// - Execution (model generation)
/// - Response processing (parse channels, detect tool calls)
///
/// # Arguments
///
/// * `ctx` - Harmony Responses context with pipeline, components, MCP manager
/// * `request` - Initial Responses API request
///
/// # Returns
///
/// Final ResponsesResponse after all tool iterations complete
///
/// # Errors
///
/// Returns error if:
/// - Max iterations exceeded (10 iterations)
/// - Pipeline execution fails
/// - MCP tool execution fails
/// - Response building fails
pub async fn serve_harmony_responses(
    ctx: &HarmonyResponsesContext,
    request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
    let mut current_request = request;
    let mut iteration_count = 0;

    loop {
        iteration_count += 1;

        // Safety check: prevent infinite loops
        if iteration_count > MAX_TOOL_ITERATIONS {
            return Err(utils::internal_error_message(format!(
                "Maximum tool iterations ({}) exceeded",
                MAX_TOOL_ITERATIONS
            )));
        }

        tracing::debug!(
            iteration = iteration_count,
            "Harmony Responses serving iteration"
        );

        // Execute through full pipeline
        // This includes:
        // - HarmonyPreparationStage (builder.rs: construct_input_messages_with_harmony)
        // - WorkerSelectionStage (FRESH selection based on current context)
        // - ClientAcquisitionStage (NEW gRPC client if needed)
        // - HarmonyRequestBuildingStage (encode to token_ids)
        // - RequestExecutionStage (model generation)
        // - HarmonyResponseProcessingStage (processor.rs: process_responses_iteration)
        let iteration_result = ctx
            .pipeline
            .execute_harmony_responses(&current_request, ctx)
            .await?;

        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
            } => {
                tracing::debug!(
                    tool_call_count = tool_calls.len(),
                    has_analysis = analysis.is_some(),
                    partial_text_len = partial_text.len(),
                    "Tool calls found in commentary channel"
                );

                // TODO: Streaming support - emit intermediate chunks
                // if let Some(tx) = &ctx.stream_tx {
                //     emit_intermediate_chunks(tx, &analysis, &partial_text, iteration_count).await?;
                // }

                // Execute MCP tools via MCP manager
                let tool_results = execute_mcp_tools(&ctx.mcp_manager, &tool_calls).await?;

                // Build next request with appended history
                current_request = build_next_request_with_tools(
                    current_request,
                    tool_calls,
                    tool_results,
                    analysis,
                    partial_text,
                )
                .map_err(|e| *e)?;

                // Continue loop - next iteration will select workers and execute
            }
            ResponsesIterationResult::Completed { response, usage } => {
                tracing::debug!(
                    output_items = response.output.len(),
                    input_tokens = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    "Harmony Responses serving completed - no more tool calls"
                );

                // No tool calls - this is the final response
                // TODO: Accumulate usage across all iterations if needed
                return Ok(*response);
            }
        }
    }
}

/// Execute MCP tools and collect results
///
/// Executes each tool call sequentially via the MCP manager.
/// Tool execution errors are returned as error results to the model
/// (allows model to handle gracefully).
///
/// # Arguments
///
/// * `mcp_manager` - MCP manager for tool execution
/// * `tool_calls` - Tool calls from commentary channel
///
/// # Returns
///
/// Vector of tool results (one per tool call)
async fn execute_mcp_tools(
    mcp_manager: &Arc<McpManager>,
    tool_calls: &[ToolCall],
) -> Result<Vec<ToolResult>, Response> {
    let mut results = Vec::new();

    for tool_call in tool_calls {
        tracing::debug!(
            tool_name = %tool_call.function.name,
            call_id = %tool_call.id,
            "Executing MCP tool"
        );

        // Parse tool arguments from JSON string
        let args_str = tool_call.function.arguments.as_deref().unwrap_or("{}");
        let args: JsonValue = serde_json::from_str(args_str).map_err(|e| {
            utils::internal_error_message(format!(
                "Invalid tool arguments JSON for tool '{}': {}",
                tool_call.function.name, e
            ))
        })?;

        // Execute tool via MCP manager
        // Convert JsonValue to ToolArgs via Option<Map> (MCP manager expects this)
        let args_map = if let JsonValue::Object(map) = args {
            Some(map)
        } else {
            None
        };

        match mcp_manager
            .call_tool(&tool_call.function.name, args_map)
            .await
        {
            Ok(mcp_result) => {
                tracing::debug!(
                    tool_name = %tool_call.function.name,
                    call_id = %tool_call.id,
                    "Tool execution succeeded"
                );

                // Extract content from MCP result
                let output = if let Some(content) = mcp_result.content.first() {
                    // TODO: Handle different content types (text, image, resource)
                    // For now, serialize the entire content item
                    serde_json::to_value(content).unwrap_or_else(
                        |_| serde_json::json!({"error": "Failed to serialize tool result"}),
                    )
                } else {
                    serde_json::json!({"result": "success"})
                };

                results.push(ToolResult {
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    output,
                    is_error: mcp_result.is_error.unwrap_or(false),
                });
            }
            Err(e) => {
                tracing::warn!(
                    tool_name = %tool_call.function.name,
                    call_id = %tool_call.id,
                    error = %e,
                    "Tool execution failed"
                );

                // Return error result to model (let it handle gracefully)
                results.push(ToolResult {
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    output: serde_json::json!({
                        "error": format!("Tool execution failed: {}", e)
                    }),
                    is_error: true,
                });
            }
        }
    }

    Ok(results)
}

/// Build next request with tool results appended to history
///
/// Constructs a new ResponsesRequest with:
/// 1. Original input items (preserved)
/// 2. Assistant message with analysis (reasoning) + partial_text + tool_calls
/// 3. Tool result messages for each tool execution
///
/// # Arguments
///
/// * `request` - Current request (contains original input)
/// * `tool_calls` - Tool calls from commentary channel
/// * `tool_results` - Results from MCP tool execution
/// * `analysis` - Analysis channel content (becomes reasoning_content)
/// * `partial_text` - Final channel content (becomes message content)
///
/// # Returns
///
/// New ResponsesRequest with appended history
fn build_next_request_with_tools(
    mut request: ResponsesRequest,
    _tool_calls: Vec<ToolCall>,
    _tool_results: Vec<ToolResult>,
    _analysis: Option<String>,
    _partial_text: String,
) -> Result<ResponsesRequest, Box<Response>> {
    // Get current input items (or empty vec if Text variant)
    let items = match request.input {
        ResponseInput::Items(items) => items,
        ResponseInput::Text(_) => {
            // Convert text to items format
            // TODO: Handle text variant properly
            vec![]
        }
    };

    // Append assistant message with tool calls
    // TODO: Build proper ResponseInputOutputItem::Message with:
    // - role: "assistant"
    // - reasoning_content (from analysis)
    // - content (from partial_text)
    // - tool_calls

    // Append tool results
    // TODO: Build ResponseInputOutputItem::FunctionCallOutput for each result
    // - call_id
    // - output (serialized result)

    // For now, placeholder implementation
    // This will be completed when we have proper ResponseInputOutputItem builders

    // Update request with new items
    request.input = ResponseInput::Items(items);

    Ok(request)
}

/// Tool execution result
///
/// Contains the result of executing a single MCP tool.
#[allow(dead_code)] // TODO: Remove when build_next_request_with_tools is implemented
struct ToolResult {
    /// Tool call ID (for matching with request)
    call_id: String,

    /// Tool name
    tool_name: String,

    /// Tool output (JSON value)
    output: JsonValue,

    /// Whether this is an error result
    is_error: bool,
}

// TODO: Implement streaming support
// /// Emit intermediate streaming chunks for analysis and partial text
// ///
// /// Emits SSE chunks for Responses API streaming:
// /// - Reasoning chunks for analysis channel
// /// - Message chunks for partial text from final channel
// ///
// /// # Arguments
// ///
// /// * `tx` - Streaming sender
// /// * `analysis` - Analysis channel content
// /// * `partial_text` - Final channel content
// /// * `iteration` - Current iteration number
// async fn emit_intermediate_chunks(
//     tx: &tokio::sync::mpsc::UnboundedSender<Result<String, String>>,
//     analysis: &Option<String>,
//     partial_text: &str,
//     iteration: usize,
// ) -> Result<(), Response> {
//     // TODO: Implement streaming emission
//     // - Emit reasoning chunks for analysis
//     // - Emit message chunks for partial_text
//     // - Follow OpenAI Responses streaming format (14 SSE event types)
//     Ok(())
// }
