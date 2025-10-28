//! Harmony Responses API implementation with multi-turn MCP tool support
//!
//! This module implements the Harmony Responses API orchestration logic,
//! coordinating full pipeline execution with MCP tool support for multi-turn conversations.
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
        common::{Function, ToolCall},
        responses::{
            ResponseInput, ResponseTool, ResponseToolType, ResponsesRequest, ResponsesResponse,
            StringOrContentParts,
        },
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

/// Execute Harmony Responses API request with multi-turn MCP tool support
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

    // Get MCP tools and add to request (do this once before loop)
    // This ensures the model knows about available MCP tools
    let mcp_tools = ctx.mcp_manager.list_tools();
    let mcp_response_tools = convert_mcp_tools_to_response_tools(&mcp_tools);

    // Merge with user-provided tools
    let mut all_tools = current_request.tools.clone().unwrap_or_default();
    all_tools.extend(mcp_response_tools);
    current_request.tools = Some(all_tools);

    tracing::debug!(
        mcp_tool_count = mcp_tools.len(),
        total_tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        "Added MCP tools to Harmony Responses request"
    );

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

                // Check if ALL tools exist in MCP inventory before executing
                // If ANY tool is not found, treat this as a completed response (single turn)
                let all_tools_available = tool_calls
                    .iter()
                    .all(|call| ctx.mcp_manager.get_tool(&call.function.name).is_some());

                if !all_tools_available {
                    // At least one tool not found - return final response immediately
                    let unavailable_tools: Vec<_> = tool_calls
                        .iter()
                        .filter(|call| ctx.mcp_manager.get_tool(&call.function.name).is_none())
                        .map(|call| call.function.name.as_str())
                        .collect();

                    tracing::info!(
                        unavailable_tools = ?unavailable_tools,
                        "Tool calls requested but tools not available - returning single-turn response"
                    );

                    // Build final response with reasoning, partial text, and failed tool calls
                    // This matches the vLLM/OpenAI behavior: if tools aren't available,
                    // return what the model generated including tool calls marked as "failed"
                    return build_final_response_without_tools(
                        &current_request,
                        tool_calls,
                        analysis,
                        partial_text,
                    )
                    .map_err(|e| *e);
                }

                // TODO: Streaming support - emit intermediate chunks
                // if let Some(tx) = &ctx.stream_tx {
                //     emit_intermediate_chunks(tx, &analysis, &partial_text, iteration_count).await?;
                // }

                // All tools available - execute MCP tools via MCP manager
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

/// Build final response without executing tools
///
/// When tool calls are detected but the tools are not available in MCP,
/// we return a single-turn response containing:
/// - Reasoning (if present from analysis channel)
/// - Message (if present from final channel)
/// - FunctionToolCall items with status "failed" (tool not available)
///
/// This matches the vLLM/OpenAI behavior: if tools aren't available,
/// return what the model generated including the tool calls, but mark them
/// as failed without entering the MCP loop.
///
/// # Arguments
///
/// * `request` - Current Responses API request
/// * `tool_calls` - Tool calls from commentary channel
/// * `analysis` - Analysis channel content (reasoning)
/// * `partial_text` - Final channel content (message text)
///
/// # Returns
///
/// ResponsesResponse with reasoning, message, and failed tool call output items
fn build_final_response_without_tools(
    request: &ResponsesRequest,
    tool_calls: Vec<ToolCall>,
    analysis: Option<String>,
    partial_text: String,
) -> Result<ResponsesResponse, Box<Response>> {
    use uuid::Uuid;

    use crate::protocols::responses::{
        ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
        ResponseUsage, ResponsesUsage,
    };

    let mut output: Vec<ResponseOutputItem> = Vec::new();
    let response_id = format!("responses-{}", Uuid::new_v4());

    // Add reasoning output if analysis present
    if let Some(analysis_text) = analysis {
        output.push(ResponseOutputItem::Reasoning {
            id: format!("reasoning_{}", response_id),
            summary: vec![],
            content: vec![ResponseReasoningContent::ReasoningText {
                text: analysis_text,
            }],
            status: Some("completed".to_string()),
        });
    }

    // Add message output if partial text present
    if !partial_text.is_empty() {
        output.push(ResponseOutputItem::Message {
            id: format!("msg_{}", response_id),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: "completed".to_string(),
        });
    }

    // Add tool calls with "failed" status (tools not available in MCP)
    for tool_call in tool_calls {
        output.push(ResponseOutputItem::FunctionToolCall {
            id: tool_call.id,
            name: tool_call.function.name,
            arguments: tool_call
                .function
                .arguments
                .unwrap_or_else(|| "{}".to_string()),
            output: Some(
                serde_json::json!({
                    "error": "Tool not available in MCP inventory"
                })
                .to_string(),
            ),
            status: "failed".to_string(),
        });
    }

    // Build ResponsesResponse
    // Note: We don't have usage info available here since we're in the serving loop
    // and don't have access to the execution result. This is acceptable as this is
    // a fallback path for unavailable tools.

    // Serialize tool_choice to string (matching response format)
    let tool_choice = request
        .tool_choice
        .as_ref()
        .and_then(|tc| serde_json::to_value(tc).ok())
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "auto".to_string());

    // Serialize truncation to string if present
    let truncation = request
        .truncation
        .as_ref()
        .and_then(|t| serde_json::to_string(t).ok());

    let response = ResponsesResponse {
        id: response_id,
        object: "response".to_string(),
        created_at: chrono::Utc::now().timestamp(),
        status: ResponseStatus::Completed,
        error: None,
        incomplete_details: None,
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        model: request.model.clone(),
        output,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        previous_response_id: request.previous_response_id.clone(),
        reasoning: None,
        store: request.store.unwrap_or(false),
        temperature: request.temperature,
        text: None, // ResponsesResponse doesn't have text field in request
        tool_choice,
        tools: request.tools.clone().unwrap_or_default(),
        top_p: request.top_p,
        truncation,
        user: request.user.clone(),
        usage: Some(ResponsesUsage::Modern(ResponseUsage {
            input_tokens: 0,             // Not available in serving loop
            output_tokens: 0,            // Not available in serving loop
            total_tokens: 0,             // Not available in serving loop
            input_tokens_details: None,  // Not available
            output_tokens_details: None, // Not available
        })),
        metadata: request.metadata.clone().unwrap_or_default(),
    };

    Ok(response)
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
/// * `analysis` - Analysis channel content (becomes reasoning content)
/// * `partial_text` - Final channel content (becomes message content)
///
/// # Returns
///
/// New ResponsesRequest with appended history
fn build_next_request_with_tools(
    mut request: ResponsesRequest,
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<ToolResult>,
    analysis: Option<String>,
    partial_text: String,
) -> Result<ResponsesRequest, Box<Response>> {
    use uuid::Uuid;

    use crate::protocols::responses::{
        ResponseContentPart, ResponseInputOutputItem, ResponseReasoningContent,
    };

    // Get current input items (or empty vec if Text variant)
    let mut items = match request.input {
        ResponseInput::Items(items) => items,
        ResponseInput::Text(text) => {
            // Convert text to items format
            vec![ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String(text),
                role: "user".to_string(),
                r#type: None,
            }]
        }
    };

    // Build assistant response item with reasoning + content + tool calls
    // This represents what the model generated in this iteration
    let assistant_id = format!("msg_{}", Uuid::new_v4());

    // Add reasoning if present (from analysis channel)
    if let Some(analysis_text) = analysis {
        items.push(ResponseInputOutputItem::Reasoning {
            id: format!("reasoning_{}", assistant_id),
            summary: vec![],
            content: vec![ResponseReasoningContent::ReasoningText {
                text: analysis_text,
            }],
            status: Some("completed".to_string()),
        });
    }

    // Add message content if present (from final channel)
    if !partial_text.is_empty() {
        items.push(ResponseInputOutputItem::Message {
            id: assistant_id.clone(),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
        });
    }

    // Add function tool calls (from commentary channel)
    for tool_call in tool_calls {
        items.push(ResponseInputOutputItem::FunctionToolCall {
            id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            arguments: tool_call
                .function
                .arguments
                .unwrap_or_else(|| "{}".to_string()),
            output: None, // Output will be added next
            status: Some("in_progress".to_string()),
        });
    }

    // Add tool results
    for tool_result in tool_results {
        // Serialize tool output to string
        let output_str = serde_json::to_string(&tool_result.output).unwrap_or_else(|e| {
            format!("{{\"error\": \"Failed to serialize tool output: {}\"}}", e)
        });

        // Update the corresponding tool call with output and completed status
        // Find and update the matching FunctionToolCall
        if let Some(ResponseInputOutputItem::FunctionToolCall {
            output,
            status,
            ..
        }) = items
            .iter_mut()
            .find(|item| matches!(item, ResponseInputOutputItem::FunctionToolCall { id, .. } if id == &tool_result.call_id))
        {
            *output = Some(output_str);
            *status = if tool_result.is_error {
                Some("failed".to_string())
            } else {
                Some("completed".to_string())
            };
        }
    }

    // Update request with new items
    request.input = ResponseInput::Items(items);

    Ok(request)
}

/// Tool execution result
///
/// Contains the result of executing a single MCP tool.
struct ToolResult {
    /// Tool call ID (for matching with request)
    call_id: String,

    /// Tool name
    #[allow(dead_code)] // Kept for documentation and future use
    tool_name: String,

    /// Tool output (JSON value)
    output: JsonValue,

    /// Whether this is an error result
    is_error: bool,
}

/// Convert MCP tools to Responses API tool format
///
/// Converts MCP ToolInfo entries to ResponseTool format so the model
/// knows about available MCP tools when making tool calls.
///
/// # Arguments
///
/// * `mcp_tools` - MCP tools from the MCP manager inventory
///
/// # Returns
///
/// Vector of ResponseTool entries in function format
fn convert_mcp_tools_to_response_tools(mcp_tools: &[crate::mcp::ToolInfo]) -> Vec<ResponseTool> {
    mcp_tools
        .iter()
        .map(|tool_info| ResponseTool {
            r#type: ResponseToolType::Function,
            function: Some(Function {
                name: tool_info.name.clone(),
                description: Some(tool_info.description.clone()),
                parameters: tool_info.parameters.clone().unwrap_or_else(|| {
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }),
                strict: None,
            }),
            server_url: Some(tool_info.server.clone()),
            authorization: None,
            server_label: None,
            server_description: Some(tool_info.description.clone()),
            require_approval: None,
            allowed_tools: None,
        })
        .collect()
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
