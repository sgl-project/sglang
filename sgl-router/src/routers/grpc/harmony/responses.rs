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
    data_connector::{ResponseId, ResponseStorage},
    mcp::McpManager,
    protocols::{
        common::{Function, ToolCall},
        responses::{
            ResponseInput, ResponseInputOutputItem, ResponseTool, ResponsesRequest,
            ResponsesResponse, StringOrContentParts,
        },
    },
    routers::grpc::{
        context::SharedComponents, harmony::processor::ResponsesIterationResult,
        pipeline::RequestPipeline, utils,
    },
};

/// Maximum number of tool execution iterations to prevent infinite loops
const MAX_TOOL_ITERATIONS: usize = 10;

/// Record of a single MCP tool call execution
///
/// Stores metadata needed to build mcp_call output items for Responses API format
#[derive(Debug, Clone)]
struct McpCallRecord {
    /// Tool call ID (stored for potential future use, currently generate new IDs)
    #[allow(dead_code)]
    call_id: String,
    /// Tool name
    tool_name: String,
    /// JSON-encoded arguments
    arguments: String,
    /// JSON-encoded output/result
    output: String,
    /// Whether execution succeeded
    success: bool,
    /// Error message if execution failed
    error: Option<String>,
}

/// Tracking structure for MCP tool calls across iterations
///
/// Accumulates all MCP tool call metadata during multi-turn conversation
/// so we can build proper mcp_list_tools and mcp_call output items.
#[derive(Debug, Clone)]
struct McpCallTracking {
    /// MCP server label (e.g., "sglang-mcp")
    server_label: String,
    /// All tool call records across all iterations
    tool_calls: Vec<McpCallRecord>,
}

impl McpCallTracking {
    fn new(server_label: String) -> Self {
        Self {
            server_label,
            tool_calls: Vec::new(),
        }
    }

    fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        arguments: String,
        output: String,
        success: bool,
        error: Option<String>,
    ) {
        self.tool_calls.push(McpCallRecord {
            call_id,
            tool_name,
            arguments,
            output,
            success,
            error,
        });
    }

    fn total_calls(&self) -> usize {
        self.tool_calls.len()
    }
}

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

    /// Response storage for loading conversation history
    pub response_storage: Arc<dyn ResponseStorage>,

    /// Optional streaming sender (for future streaming support)
    pub stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Result<String, String>>>,
}

impl HarmonyResponsesContext {
    /// Create a new Harmony Responses context
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        mcp_manager: Arc<McpManager>,
        response_storage: Arc<dyn ResponseStorage>,
    ) -> Self {
        Self {
            pipeline,
            components,
            mcp_manager,
            response_storage,
            stream_tx: None,
        }
    }

    /// Create with streaming support
    pub fn with_streaming(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        mcp_manager: Arc<McpManager>,
        response_storage: Arc<dyn ResponseStorage>,
        stream_tx: tokio::sync::mpsc::UnboundedSender<Result<String, String>>,
    ) -> Self {
        Self {
            pipeline,
            components,
            mcp_manager,
            response_storage,
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
    // Load previous conversation history if previous_response_id is set
    let mut current_request = load_previous_messages(ctx, request).await?;
    let mut iteration_count = 0;

    // Check if request has MCP tools - if so, ensure dynamic client is registered
    // and add static MCP tools to the request
    use crate::{
        protocols::responses::ResponseToolType, routers::openai::mcp::ensure_request_mcp_client,
    };

    let has_mcp_tools = current_request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .any(|t| matches!(t.r#type, ResponseToolType::Mcp))
        })
        .unwrap_or(false);

    // Initialize MCP call tracking (will be passed to processor for final response)
    let mut mcp_tracking = if has_mcp_tools {
        Some(McpCallTracking::new("sglang-mcp".to_string()))
    } else {
        None
    };

    if has_mcp_tools {
        // Ensure dynamic MCP client is registered for request-scoped tools
        if let Some(tools) = &current_request.tools {
            ensure_request_mcp_client(&ctx.mcp_manager, tools).await;
        }

        // Add static MCP tools from inventory to the request
        // (similar to non-Harmony pipeline pattern)
        let mcp_tools = ctx.mcp_manager.list_tools();
        if !mcp_tools.is_empty() {
            let mcp_response_tools = convert_mcp_tools_to_response_tools(&mcp_tools);

            let mut all_tools = current_request.tools.clone().unwrap_or_default();
            all_tools.extend(mcp_response_tools);
            current_request.tools = Some(all_tools);

            tracing::debug!(
                mcp_tool_count = mcp_tools.len(),
                total_tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                "Request has MCP tools - added static MCP tools to Harmony Responses request"
            );
        }
    }

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
                // If tools don't exist, call_tool() will return error naturally
                let tool_results = if let Some(ref mut tracking) = mcp_tracking {
                    execute_mcp_tools(&ctx.mcp_manager, &tool_calls, tracking).await?
                } else {
                    // Should never happen (we only get tool_calls when has_mcp_tools=true)
                    return Err(utils::internal_error_static(
                        "Tool calls found but MCP tracking not initialized",
                    ));
                };

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
            ResponsesIterationResult::Completed {
                mut response,
                usage,
            } => {
                tracing::debug!(
                    output_items = response.output.len(),
                    input_tokens = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    has_mcp_tracking = mcp_tracking.is_some(),
                    "Harmony Responses serving completed - no more tool calls"
                );

                // Inject MCP output items if MCP tools were available
                // (even if no tools were called, we still list available tools)
                if let Some(tracking) = mcp_tracking {
                    inject_mcp_metadata(&mut response, &tracking, &ctx.mcp_manager);

                    tracing::debug!(
                        mcp_calls = tracking.total_calls(),
                        output_items_after = response.output.len(),
                        "Injected MCP metadata into final response"
                    );
                }

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
    tracking: &mut McpCallTracking,
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

                let is_error = mcp_result.is_error.unwrap_or(false);
                let output_str = serde_json::to_string(&output)
                    .unwrap_or_else(|_| r#"{"error": "Failed to serialize output"}"#.to_string());

                // Record this call in tracking
                tracking.record_call(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    args_str.to_string(),
                    output_str.clone(),
                    !is_error,
                    if is_error {
                        Some(output_str.clone())
                    } else {
                        None
                    },
                );

                results.push(ToolResult {
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    output,
                    is_error,
                });
            }
            Err(e) => {
                tracing::warn!(
                    tool_name = %tool_call.function.name,
                    call_id = %tool_call.id,
                    error = %e,
                    "Tool execution failed"
                );

                let error_msg = format!("Tool execution failed: {}", e);
                let error_output = serde_json::json!({
                    "error": error_msg.clone()
                });
                let error_output_str = serde_json::to_string(&error_output)
                    .unwrap_or_else(|_| format!(r#"{{"error": "{}"}}"#, error_msg));

                // Record failed call in tracking
                tracking.record_call(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    args_str.to_string(),
                    error_output_str.clone(),
                    false,
                    Some(error_msg),
                );

                // Return error result to model (let it handle gracefully)
                results.push(ToolResult {
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    output: error_output,
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
/// Converts MCP Tool entries (from rmcp SDK) to ResponseTool format so the model
/// knows about available MCP tools when making tool calls.
///
/// # Arguments
///
/// * `mcp_tools` - MCP tools from the MCP manager inventory (rmcp::model::Tool)
///
/// # Returns
///
/// Vector of ResponseTool entries in MCP format
fn convert_mcp_tools_to_response_tools(mcp_tools: &[crate::mcp::Tool]) -> Vec<ResponseTool> {
    use serde_json::Value;

    use crate::protocols::responses::ResponseToolType;

    mcp_tools
        .iter()
        .map(|tool_info| ResponseTool {
            r#type: ResponseToolType::Mcp,
            function: Some(Function {
                name: tool_info.name.to_string(),
                description: tool_info.description.as_ref().map(|d| d.to_string()),
                parameters: Value::Object((*tool_info.input_schema).clone()),
                strict: None,
            }),
            server_url: None, // MCP tools from inventory don't have individual server URLs
            authorization: None,
            server_label: None,
            server_description: tool_info.description.as_ref().map(|d| d.to_string()),
            require_approval: None,
            allowed_tools: None,
        })
        .collect()
}

/// Inject MCP metadata into final response
///
/// Adds mcp_list_tools and mcp_call output items to the response output array.
/// Following non-Harmony pipeline pattern:
/// 1. Prepend mcp_list_tools at the beginning
/// 2. Append all mcp_call items at the end
///
/// # Arguments
///
/// * `response` - Final response to modify
/// * `tracking` - MCP call tracking data
/// * `mcp_manager` - MCP manager for listing tools
fn inject_mcp_metadata(
    response: &mut ResponsesResponse,
    tracking: &McpCallTracking,
    mcp_manager: &Arc<McpManager>,
) {
    use serde_json::{json, Value};
    use uuid::Uuid;

    use crate::protocols::responses::{McpToolInfo, ResponseOutputItem};

    // Build mcp_list_tools item
    let tools = mcp_manager.list_tools();
    let tools_info: Vec<McpToolInfo> = tools
        .iter()
        .map(|t| McpToolInfo {
            name: t.name.to_string(),
            description: t.description.as_ref().map(|d| d.to_string()),
            input_schema: Value::Object((*t.input_schema).clone()),
            annotations: Some(json!({
                "read_only": false
            })),
        })
        .collect();

    let mcp_list_tools = ResponseOutputItem::McpListTools {
        id: format!("mcpl_{}", Uuid::new_v4()),
        server_label: tracking.server_label.clone(),
        tools: tools_info,
    };

    // Build mcp_call items for each tracked call
    let mcp_call_items: Vec<ResponseOutputItem> = tracking
        .tool_calls
        .iter()
        .map(|record| ResponseOutputItem::McpCall {
            id: format!("mcp_{}", Uuid::new_v4()),
            status: if record.success {
                "completed"
            } else {
                "failed"
            }
            .to_string(),
            approval_request_id: None,
            arguments: record.arguments.clone(),
            error: record.error.clone(),
            name: record.tool_name.clone(),
            output: record.output.clone(),
            server_label: tracking.server_label.clone(),
        })
        .collect();

    // Inject into response output:
    // 1. Prepend mcp_list_tools at the beginning
    response.output.insert(0, mcp_list_tools);

    // 2. Append all mcp_call items at the end
    response.output.extend(mcp_call_items);
}

/// Load previous conversation messages from storage
///
/// If the request has `previous_response_id`, loads the response chain from storage
/// and prepends the conversation history to the request input items.
///
/// # Arguments
///
/// * `ctx` - Harmony Responses context with response_storage
/// * `request` - Current request (may have previous_response_id set)
///
/// # Returns
///
/// Modified request with conversation history prepended to input items
async fn load_previous_messages(
    ctx: &HarmonyResponsesContext,
    request: ResponsesRequest,
) -> Result<ResponsesRequest, Response> {
    let Some(ref prev_id_str) = request.previous_response_id else {
        // No previous_response_id, return request as-is
        return Ok(request);
    };

    let prev_id = ResponseId::from(prev_id_str.as_str());

    // Load response chain from storage
    let chain = ctx
        .response_storage
        .get_response_chain(&prev_id, None)
        .await
        .map_err(|e| {
            utils::internal_error_message(format!(
                "Failed to load previous response chain for {}: {}",
                prev_id_str, e
            ))
        })?;

    // Build conversation history from stored responses
    let mut history_items = Vec::new();

    // Helper to deserialize and collect items from a JSON array
    let deserialize_items =
        |arr: &serde_json::Value, item_type: &str| -> Vec<ResponseInputOutputItem> {
            arr.as_array()
                .into_iter()
                .flat_map(|items| items.iter())
                .filter_map(|item| {
                    serde_json::from_value::<ResponseInputOutputItem>(item.clone())
                        .map_err(|e| {
                            tracing::warn!(
                                "Failed to deserialize stored {} item: {}. Item: {}",
                                item_type,
                                e,
                                item
                            );
                        })
                        .ok()
                })
                .collect()
        };

    for stored in chain.responses.iter() {
        history_items.extend(deserialize_items(&stored.input, "input"));
        history_items.extend(deserialize_items(&stored.output, "output"));
    }

    tracing::debug!(
        previous_response_id = %prev_id_str,
        history_items_count = history_items.len(),
        "Loaded conversation history from previous response"
    );

    // Build modified request with history prepended
    let mut modified_request = request;

    // Convert current input to items format
    let all_items = match modified_request.input {
        ResponseInput::Items(items) => {
            // Prepend history to existing items
            let mut combined = history_items;
            combined.extend(items);
            combined
        }
        ResponseInput::Text(text) => {
            // Convert text to item and prepend history
            history_items.push(ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String(text),
                role: "user".to_string(),
                r#type: None,
            });
            history_items
        }
    };

    // Update request with combined items and clear previous_response_id
    modified_request.input = ResponseInput::Items(all_items);
    modified_request.previous_response_id = None;

    Ok(modified_request)
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
