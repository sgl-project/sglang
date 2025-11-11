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
//!             // Separate MCP tools from function tools
//!             // Execute MCP tools, return if function tools found
//!             // Continue loop with MCP results if only MCP tools
//!         }
//!         Completed { response, .. } => {
//!             return Ok(response);
//!         }
//!     }
//! }
//! ```
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::response::Response;
use bytes::Bytes;
use serde_json::{from_str, from_value, json, to_string, to_value, Value};
use tokio::sync::mpsc;
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::{
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseId, ResponseStorage},
    mcp::{self, McpManager},
    protocols::{
        common::{Function, ToolCall, ToolChoice, ToolChoiceValue, Usage},
        responses::{
            McpToolInfo, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponseOutputItem, ResponseReasoningContent, ResponseStatus, ResponseTool,
            ResponseToolType, ResponseUsage, ResponsesRequest, ResponsesResponse, ResponsesUsage,
            StringOrContentParts,
        },
    },
    routers::grpc::{
        common::responses::{
            build_sse_response, ensure_mcp_connection, persist_response_if_needed,
            streaming::{OutputItemType, ResponseStreamEventEmitter},
        },
        context::SharedComponents,
        error,
        harmony::{processor::ResponsesIterationResult, streaming::HarmonyStreamingProcessor},
        pipeline::RequestPipeline,
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
    pub fn new(server_label: String) -> Self {
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

    /// Conversation storage for persisting conversations
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// Conversation item storage for persisting conversation items
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,

    /// Optional streaming sender (for future streaming support)
    pub stream_tx: Option<mpsc::UnboundedSender<Result<String, String>>>,
}

impl HarmonyResponsesContext {
    /// Create a new Harmony Responses context
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        mcp_manager: Arc<McpManager>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
    ) -> Self {
        Self {
            pipeline,
            components,
            mcp_manager,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            stream_tx: None,
        }
    }

    /// Create with streaming support
    pub fn with_streaming(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        mcp_manager: Arc<McpManager>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
        stream_tx: mpsc::UnboundedSender<Result<String, String>>,
    ) -> Self {
        Self {
            pipeline,
            components,
            mcp_manager,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            stream_tx: Some(stream_tx),
        }
    }
}

/// Build a HashSet of MCP tool names for O(1) lookup
///
/// Creates a HashSet containing the names of all MCP tools in the request,
/// allowing for efficient O(1) lookups when partitioning tool calls.
fn build_mcp_tool_names_set(request_tools: &[ResponseTool]) -> std::collections::HashSet<&str> {
    request_tools
        .iter()
        .filter(|t| t.r#type == ResponseToolType::Mcp)
        .filter_map(|t| t.function.as_ref().map(|f| f.name.as_str()))
        .collect()
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
    // Clone request for persistence
    let original_request = request.clone();

    // Load previous conversation history if previous_response_id is set
    let current_request = load_previous_messages(ctx, request).await?;

    // Check MCP connection and get whether MCP tools are present
    let has_mcp_tools =
        ensure_mcp_connection(&ctx.mcp_manager, current_request.tools.as_deref()).await?;

    let response = if has_mcp_tools {
        execute_with_mcp_loop(ctx, current_request).await?
    } else {
        // No MCP tools - execute pipeline once (may have function tools or no tools)
        execute_without_mcp_loop(ctx, current_request).await?
    };

    // Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &response,
        &original_request,
    )
    .await;

    Ok(response)
}

/// Execute Harmony Responses with MCP tool loop
///
/// Automatically executes MCP tools in a loop until no more tool calls or max iterations
async fn execute_with_mcp_loop(
    ctx: &HarmonyResponsesContext,
    mut current_request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
    let mut iteration_count = 0;

    // Extract server_label from request tools
    let server_label = extract_mcp_server_label(current_request.tools.as_deref());
    let mut mcp_tracking = McpCallTracking::new(server_label.clone());

    // Extract user's max_tool_calls limit (if set)
    let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);

    // Add static MCP tools from inventory to the request
    let mcp_tools = ctx.mcp_manager.list_tools();
    if !mcp_tools.is_empty() {
        let mcp_response_tools = convert_mcp_tools_to_response_tools(&mcp_tools);

        let mut all_tools = current_request.tools.clone().unwrap_or_default();
        all_tools.extend(mcp_response_tools);
        current_request.tools = Some(all_tools);

        debug!(
            mcp_tool_count = mcp_tools.len(),
            total_tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            "MCP client available - added static MCP tools to Harmony Responses request"
        );
    }

    loop {
        iteration_count += 1;

        // Safety check: prevent infinite loops
        if iteration_count > MAX_TOOL_ITERATIONS {
            error!(
                function = "execute_with_mcp_loop",
                iteration_count = iteration_count,
                max_iterations = MAX_TOOL_ITERATIONS,
                "Maximum tool iterations exceeded"
            );
            return Err(error::internal_error(format!(
                "Maximum tool iterations ({}) exceeded",
                MAX_TOOL_ITERATIONS
            )));
        }

        debug!(
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
                usage,
                request_id,
            } => {
                debug!(
                    tool_call_count = tool_calls.len(),
                    has_analysis = analysis.is_some(),
                    partial_text_len = partial_text.len(),
                    "Tool calls found - separating MCP and function tools"
                );

                // Separate MCP and function tool calls based on tool type
                let request_tools = current_request.tools.as_deref().unwrap_or(&[]);
                let mcp_tool_names = build_mcp_tool_names_set(request_tools);
                let (mcp_tool_calls, function_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                    .into_iter()
                    .partition(|tc| mcp_tool_names.contains(tc.function.name.as_str()));

                debug!(
                    mcp_calls = mcp_tool_calls.len(),
                    function_calls = function_tool_calls.len(),
                    "Tool calls separated by type"
                );

                // Check combined limit (user's max_tool_calls vs safety limit)
                let effective_limit = match max_tool_calls {
                    Some(user_max) => user_max.min(MAX_TOOL_ITERATIONS),
                    None => MAX_TOOL_ITERATIONS,
                };

                // Check if we would exceed the limit with these new MCP tool calls
                let total_calls_after = mcp_tracking.total_calls() + mcp_tool_calls.len();
                if total_calls_after > effective_limit {
                    warn!(
                        current_calls = mcp_tracking.total_calls(),
                        new_calls = mcp_tool_calls.len() + function_tool_calls.len(),
                        total_after = total_calls_after,
                        effective_limit = effective_limit,
                        user_max = ?max_tool_calls,
                        "Reached tool call limit - returning incomplete response"
                    );

                    // Combine back for response
                    let all_tool_calls: Vec<_> = mcp_tool_calls
                        .into_iter()
                        .chain(function_tool_calls)
                        .collect();

                    // Build response with incomplete status - no tools executed due to limit
                    let mut response = build_tool_response(
                        vec![],         // No MCP tools executed
                        vec![],         // No MCP results
                        all_tool_calls, // All tools returned as function calls (not executed)
                        analysis,
                        partial_text,
                        usage,
                        request_id,
                        Arc::new(current_request),
                    );

                    // Mark as completed with incomplete_details
                    response.status = ResponseStatus::Completed;
                    response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                    // Inject MCP metadata if any calls were executed
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(&mut response, &mcp_tracking, &ctx.mcp_manager);
                    }

                    return Ok(response);
                }

                // Execute MCP tools (if any)
                let mcp_results = if !mcp_tool_calls.is_empty() {
                    execute_mcp_tools(&ctx.mcp_manager, &mcp_tool_calls, &mut mcp_tracking).await?
                } else {
                    Vec::new()
                };

                // If there are function tools, exit MCP loop and return response
                if !function_tool_calls.is_empty() {
                    debug!(
                        "Function tool calls present - exiting MCP loop and returning to caller"
                    );

                    // Build response that includes:
                    // 1. Reasoning/message from this iteration
                    // 2. MCP tools as completed (with output) - these were executed
                    // 3. Function tools as completed (without output) - need caller execution
                    let mut response = build_tool_response(
                        mcp_tool_calls,
                        mcp_results,
                        function_tool_calls,
                        analysis,
                        partial_text,
                        usage,
                        request_id,
                        Arc::new(current_request),
                    );

                    // Inject MCP metadata for all executed calls
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(&mut response, &mcp_tracking, &ctx.mcp_manager);
                    }

                    return Ok(response);
                }

                // Only MCP tools - continue loop with their results
                debug!("Only MCP tools - continuing loop with results");

                // Build next request with appended history
                current_request = build_next_request_with_tools(
                    current_request,
                    mcp_tool_calls,
                    mcp_results,
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
                debug!(
                    output_items = response.output.len(),
                    input_tokens = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    "MCP loop completed - no more tool calls"
                );

                // Inject MCP metadata into final response
                inject_mcp_metadata(&mut response, &mcp_tracking, &ctx.mcp_manager);

                debug!(
                    mcp_calls = mcp_tracking.total_calls(),
                    output_items_after = response.output.len(),
                    "Injected MCP metadata into final response"
                );

                // No tool calls - this is the final response
                return Ok(*response);
            }
        }
    }
}

/// Execute Harmony Responses without MCP loop (single execution)
///
/// For function tools or no tools - executes pipeline once and returns
async fn execute_without_mcp_loop(
    ctx: &HarmonyResponsesContext,
    current_request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
    debug!("Executing Harmony Responses without MCP loop");

    // Execute pipeline once
    let iteration_result = ctx
        .pipeline
        .execute_harmony_responses(&current_request, ctx)
        .await?;

    match iteration_result {
        ResponsesIterationResult::ToolCallsFound {
            tool_calls,
            analysis,
            partial_text,
            usage,
            request_id,
        } => {
            // Function tool calls found - return to caller for execution
            debug!(
                tool_call_count = tool_calls.len(),
                "Function tool calls found - returning to caller"
            );

            Ok(build_tool_response(
                vec![],
                vec![],
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id,
                Arc::new(current_request),
            ))
        }
        ResponsesIterationResult::Completed { response, usage: _ } => {
            // No tool calls - return completed response
            debug!("No tool calls - returning completed response");
            Ok(*response)
        }
    }
}

/// Serve Harmony Responses API with streaming (SSE)
///
/// This is the streaming equivalent of `serve_harmony_responses()`.
/// Emits SSE events for lifecycle, MCP list_tools, and per-iteration streaming.
///
/// # Architecture
///
/// - Emits `response.created` and `response.in_progress` at start
/// - Emits `mcp_list_tools` events on first iteration (if MCP tools available)
/// - Loops through tool execution iterations (max 10)
/// - Calls `streaming::process_responses_iteration_stream()` for per-iteration events
/// - Emits `response.completed` at end
/// - Handles errors with `response.failed`
pub async fn serve_harmony_responses_stream(
    ctx: &HarmonyResponsesContext,
    request: ResponsesRequest,
) -> Response {
    // Load previous conversation history if previous_response_id is set
    let current_request = match load_previous_messages(ctx, request.clone()).await {
        Ok(req) => req,
        Err(err_response) => return err_response,
    };

    // Check MCP connection BEFORE starting stream and get whether MCP tools are present
    let has_mcp_tools =
        match ensure_mcp_connection(&ctx.mcp_manager, current_request.tools.as_deref()).await {
            Ok(has_mcp) => has_mcp,
            Err(response) => return response,
        };

    // Create SSE channel
    let (tx, rx) = mpsc::unbounded_channel();

    // Create response event emitter
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = current_request.model.clone();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id.clone(), model, created_at);

    // Set original request for complete response fields
    emitter.set_original_request(current_request.clone());

    // Clone context for spawned task
    let ctx_clone = ctx.clone();

    // Spawn async task to handle streaming
    tokio::spawn(async move {
        let ctx = &ctx_clone;

        // Emit initial response.created and response.in_progress events
        let event = emitter.emit_created();
        if emitter.send_event(&event, &tx).is_err() {
            return;
        }
        let event = emitter.emit_in_progress();
        if emitter.send_event(&event, &tx).is_err() {
            return;
        }

        if has_mcp_tools {
            execute_mcp_tool_loop_streaming(ctx, current_request, &request, &mut emitter, &tx)
                .await;
        } else {
            execute_without_mcp_streaming(ctx, &current_request, &request, &mut emitter, &tx).await;
        }
    });

    // Return SSE stream response
    build_sse_response(rx)
}

// Execute MCP tool loop with streaming
///
/// Handles the full MCP workflow:
/// - Adds static MCP tools to request
/// - Emits mcp_list_tools events
/// - Loops through tool execution iterations
/// - Emits final response.completed event
/// - Persists response internally
async fn execute_mcp_tool_loop_streaming(
    ctx: &HarmonyResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) {
    // Extract server_label from request tools
    let server_label = extract_mcp_server_label(current_request.tools.as_deref());

    // Set server label in emitter for MCP call items
    emitter.set_mcp_server_label(server_label.clone());

    // Initialize MCP call tracking
    let mut mcp_tracking = McpCallTracking::new(server_label.clone());

    // Extract user's max_tool_calls limit (if set)
    let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);

    // Add static MCP tools from inventory
    let mcp_tools = ctx.mcp_manager.list_tools();
    if !mcp_tools.is_empty() {
        let mcp_response_tools = convert_mcp_tools_to_response_tools(&mcp_tools);
        let mut all_tools = current_request.tools.clone().unwrap_or_default();
        all_tools.extend(mcp_response_tools);
        current_request.tools = Some(all_tools);

        debug!(
            mcp_tool_count = mcp_tools.len(),
            total_tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            "MCP client available - added static MCP tools to Harmony Responses streaming request"
        );
    }

    // Build HashSet of MCP tool names for O(1) lookup during streaming
    // Clone tool names to owned strings to avoid borrowing current_request
    let mcp_tool_names: std::collections::HashSet<String> = current_request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .filter(|t| t.r#type == ResponseToolType::Mcp)
                .filter_map(|t| t.function.as_ref().map(|f| f.name.clone()))
                .collect()
        })
        .unwrap_or_default();

    // Emit mcp_list_tools on first iteration
    let (output_index, item_id) = emitter.allocate_output_index(OutputItemType::McpListTools);

    // Build tools list for item structure
    let tool_items: Vec<_> = mcp_tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": Value::Object((*t.input_schema).clone())
            })
        })
        .collect();

    // Build final item with completed status and tools
    let item_done = json!({
        "id": item_id,
        "type": "mcp_list_tools",
        "server_label": server_label,
        "status": "completed",
        "tools": tool_items
    });

    // Store the completed item data and mark as completed FIRST
    // This ensures it appears in final response even if event sending fails
    emitter.emit_output_item_done(output_index, &item_done);
    emitter.complete_output_item(output_index);

    // Now emit all the events (failures won't affect the stored data)
    // Emit output_item.added
    let item = json!({
        "id": item_id,
        "type": "mcp_list_tools",
        "server_label": server_label,
        "status": "in_progress",
        "tools": []
    });
    let event = emitter.emit_output_item_added(output_index, &item);
    if emitter.send_event(&event, tx).is_err() {
        return;
    }

    // Emit mcp_list_tools.in_progress
    let event = emitter.emit_mcp_list_tools_in_progress(output_index);
    if emitter.send_event(&event, tx).is_err() {
        return;
    }

    // Emit mcp_list_tools.completed
    let event = emitter.emit_mcp_list_tools_completed(output_index, &mcp_tools);
    if emitter.send_event(&event, tx).is_err() {
        return;
    }

    // Emit output_item.done
    let event = emitter.emit_output_item_done(output_index, &item_done);
    if emitter.send_event(&event, tx).is_err() {
        return;
    }

    debug!(
        tool_count = mcp_tools.len(),
        "Emitted mcp_list_tools on first iteration"
    );

    // MCP tool loop (max 10 iterations)
    let mut iteration_count = 0;
    loop {
        iteration_count += 1;

        // Safety check: prevent infinite loops
        if iteration_count > MAX_TOOL_ITERATIONS {
            emitter.emit_error(
                &format!("Maximum tool iterations ({}) exceeded", MAX_TOOL_ITERATIONS),
                Some("max_iterations_exceeded"),
                tx,
            );
            return;
        }

        debug!(
            iteration = iteration_count,
            "Harmony Responses streaming iteration"
        );

        // Execute pipeline and get stream
        let execution_result = match ctx
            .pipeline
            .execute_harmony_responses_streaming(&current_request, ctx)
            .await
        {
            Ok(result) => result,
            Err(err_response) => {
                emitter.emit_error(
                    &format!("Pipeline execution failed: {:?}", err_response),
                    Some("pipeline_error"),
                    tx,
                );
                return;
            }
        };

        // Process stream with token-level streaming (mixed tools - emits correct events per tool type)
        let iteration_result = match HarmonyStreamingProcessor::process_responses_iteration_stream(
            execution_result,
            emitter,
            tx,
            &mcp_tool_names,
        )
        .await
        {
            Ok(result) => result,
            Err(err_msg) => {
                emitter.emit_error(&err_msg, Some("processing_error"), tx);
                return;
            }
        };

        // Handle iteration result (tool calls or completion)
        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id: _,
            } => {
                debug!(
                    tool_call_count = tool_calls.len(),
                    has_analysis = analysis.is_some(),
                    partial_text_len = partial_text.len(),
                    "Tool calls found - separating MCP and function tools"
                );

                // Separate MCP and function tool calls based on tool type
                let request_tools = current_request.tools.as_deref().unwrap_or(&[]);
                let mcp_tool_names = build_mcp_tool_names_set(request_tools);
                let (mcp_tool_calls, function_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                    .into_iter()
                    .partition(|tc| mcp_tool_names.contains(tc.function.name.as_str()));

                debug!(
                    mcp_calls = mcp_tool_calls.len(),
                    function_calls = function_tool_calls.len(),
                    "Tool calls separated by type in streaming"
                );

                // Check combined limit (user's max_tool_calls vs safety limit)
                let effective_limit = match max_tool_calls {
                    Some(user_max) => user_max.min(MAX_TOOL_ITERATIONS),
                    None => MAX_TOOL_ITERATIONS,
                };

                // Check if we would exceed the limit with these new MCP tool calls
                let total_calls_after = mcp_tracking.total_calls() + mcp_tool_calls.len();
                if total_calls_after > effective_limit {
                    warn!(
                        current_calls = mcp_tracking.total_calls(),
                        new_calls = mcp_tool_calls.len() + function_tool_calls.len(),
                        total_after = total_calls_after,
                        effective_limit = effective_limit,
                        user_max = ?max_tool_calls,
                        "Reached tool call limit in streaming - emitting completion with incomplete_details"
                    );

                    // Emit response.completed with incomplete_details and usage
                    let incomplete_details = json!({ "reason": "max_tool_calls" });
                    let usage_json = json!({
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "incomplete_details": incomplete_details,
                    });
                    let event = emitter.emit_completed(Some(&usage_json));
                    emitter.send_event_best_effort(&event, tx);
                    return;
                }

                // Execute MCP tools (if any)
                let mcp_results = if !mcp_tool_calls.is_empty() {
                    match execute_mcp_tools(&ctx.mcp_manager, &mcp_tool_calls, &mut mcp_tracking)
                        .await
                    {
                        Ok(results) => results,
                        Err(err_response) => {
                            emitter.emit_error(
                                &format!("MCP tool execution failed: {:?}", err_response),
                                Some("mcp_tool_error"),
                                tx,
                            );
                            return;
                        }
                    }
                } else {
                    Vec::new()
                };

                // Update mcp_call output items with execution results (if any MCP tools were executed)
                if !mcp_results.is_empty() {
                    emitter.update_mcp_call_outputs(&mcp_results);
                }

                // If there are function tools, exit MCP loop and emit completion
                if !function_tool_calls.is_empty() {
                    debug!(
                        "Function tool calls present - exiting MCP loop and emitting completion"
                    );

                    // Function tool calls were already emitted during streaming processing
                    // Just emit response.completed with usage
                    let usage_json = json!({
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    });
                    let event = emitter.emit_completed(Some(&usage_json));
                    emitter.send_event_best_effort(&event, tx);
                    return;
                }

                // Only MCP tools - continue loop with their results
                debug!("Only MCP tools - continuing loop with results");

                // Build next request with appended history
                current_request = match build_next_request_with_tools(
                    current_request,
                    mcp_tool_calls,
                    mcp_results,
                    analysis,
                    partial_text,
                ) {
                    Ok(req) => req,
                    Err(e) => {
                        emitter.emit_error(
                            &format!("Failed to build next request: {:?}", e),
                            Some("request_building_error"),
                            tx,
                        );
                        return;
                    }
                };

                // Continue loop
            }
            ResponsesIterationResult::Completed { response, usage } => {
                debug!(
                    output_items = response.output.len(),
                    input_tokens = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    "Harmony Responses streaming completed - no more tool calls"
                );

                // Finalize response from emitter's accumulated data
                let final_response = emitter.finalize(Some(usage.clone()));

                // Persist response to storage if store=true
                persist_response_if_needed(
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    ctx.response_storage.clone(),
                    &final_response,
                    original_request,
                )
                .await;

                // Emit response.completed with usage
                let usage_json = json!({
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                });
                let event = emitter.emit_completed(Some(&usage_json));
                emitter.send_event_best_effort(&event, tx);
                return;
            }
        }
    }
}

/// Execute without MCP tool loop (single execution with streaming)
///
/// For function tools or no tools - executes pipeline once and emits completion.
/// The streaming processor handles all output items (reasoning, message, function tool calls).
async fn execute_without_mcp_streaming(
    ctx: &HarmonyResponsesContext,
    current_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) {
    debug!("No MCP tools - executing single iteration");

    // Execute pipeline and get stream
    let execution_result = match ctx
        .pipeline
        .execute_harmony_responses_streaming(current_request, ctx)
        .await
    {
        Ok(result) => result,
        Err(err_response) => {
            emitter.emit_error(
                &format!("Pipeline execution failed: {:?}", err_response),
                Some("pipeline_error"),
                tx,
            );
            return;
        }
    };

    // Process stream (emits all output items during streaming - function tool path emits function_call_arguments.* events)
    // Pass empty HashSet so all tools are treated as function tools (per-tool detection)
    let empty_mcp_tools = std::collections::HashSet::new();
    let iteration_result = match HarmonyStreamingProcessor::process_responses_iteration_stream(
        execution_result,
        emitter,
        tx,
        &empty_mcp_tools,
    )
    .await
    {
        Ok(result) => result,
        Err(err_msg) => {
            emitter.emit_error(&err_msg, Some("processing_error"), tx);
            return;
        }
    };

    // Extract usage from iteration result
    let usage = match iteration_result {
        ResponsesIterationResult::ToolCallsFound { usage, .. } => usage,
        ResponsesIterationResult::Completed { usage, .. } => usage,
    };

    // Finalize response from emitter's accumulated data
    let final_response = emitter.finalize(Some(usage.clone()));

    // Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &final_response,
        original_request,
    )
    .await;

    // Emit response.completed with usage
    let usage_json = json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    });
    let event = emitter.emit_completed(Some(&usage_json));
    emitter.send_event_best_effort(&event, tx);
}

/// Build ResponsesResponse with tool calls (MCP and/or function tools)
///
/// ResponsesResponse with tool calls
#[allow(clippy::too_many_arguments)]
fn build_tool_response(
    mcp_tool_calls: Vec<ToolCall>,
    mcp_results: Vec<ToolResult>,
    function_tool_calls: Vec<ToolCall>,
    analysis: Option<String>, // Analysis channel content (reasoning)
    partial_text: String,     // Final channel content (message)
    usage: Usage,
    request_id: String,
    responses_request: Arc<ResponsesRequest>,
) -> ResponsesResponse {
    let mut output: Vec<ResponseOutputItem> = Vec::new();

    // Add reasoning output item if analysis exists
    if let Some(analysis_text) = analysis {
        output.push(ResponseOutputItem::Reasoning {
            id: format!("reasoning_{}", request_id),
            summary: vec![],
            content: vec![ResponseReasoningContent::ReasoningText {
                text: analysis_text,
            }],
            status: Some("completed".to_string()),
        });
    }

    // Add message output item if partial text exists
    if !partial_text.is_empty() {
        output.push(ResponseOutputItem::Message {
            id: format!("msg_{}", request_id),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: "completed".to_string(),
        });
    }

    // Add MCP tool calls WITH output (these were executed)
    for (tool_call, result) in mcp_tool_calls.iter().zip(mcp_results.iter()) {
        let output_str = to_string(&result.output).unwrap_or_else(|e| {
            format!("{{\"error\": \"Failed to serialize tool output: {}\"}}", e)
        });

        output.push(ResponseOutputItem::FunctionToolCall {
            id: tool_call.id.clone(),
            call_id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            arguments: tool_call.function.arguments.clone().unwrap_or_default(),
            output: Some(output_str),
            status: if result.is_error {
                "failed"
            } else {
                "completed"
            }
            .to_string(),
        });
    }

    // Add function tool calls WITHOUT output (need caller execution)
    for tool_call in function_tool_calls {
        output.push(ResponseOutputItem::FunctionToolCall {
            id: tool_call.id.clone(),
            call_id: tool_call.id.clone(),
            name: tool_call.function.name.clone(),
            arguments: tool_call.function.arguments.clone().unwrap_or_default(),
            output: None, // No output = needs execution
            status: "completed".to_string(),
        });
    }

    // Build ResponsesResponse with Completed status
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    ResponsesResponse::builder(&request_id, &responses_request.model)
        .copy_from_request(&responses_request)
        .created_at(created_at)
        .status(ResponseStatus::Completed)
        .output(output)
        .usage(ResponsesUsage::Modern(ResponseUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: None,
            output_tokens_details: None,
        }))
        .build()
}

/// Execute MCP tools and collect results
///
/// Executes each tool call sequentially via the MCP manager.
/// Tool execution errors are returned as error results to the model
/// (allows model to handle gracefully).
///
/// Vector of tool results (one per tool call)
async fn execute_mcp_tools(
    mcp_manager: &Arc<McpManager>,
    tool_calls: &[ToolCall],
    tracking: &mut McpCallTracking,
) -> Result<Vec<ToolResult>, Response> {
    let mut results = Vec::new();

    for tool_call in tool_calls {
        debug!(
            tool_name = %tool_call.function.name,
            call_id = %tool_call.id,
            "Executing MCP tool"
        );

        // Parse tool arguments from JSON string
        let args_str = tool_call.function.arguments.as_deref().unwrap_or("{}");
        let args: Value = from_str(args_str).map_err(|e| {
            error!(
                function = "execute_mcp_tools",
                tool_name = %tool_call.function.name,
                call_id = %tool_call.id,
                error = %e,
                "Failed to parse tool arguments JSON"
            );
            error::internal_error(format!(
                "Invalid tool arguments JSON for tool '{}': {}",
                tool_call.function.name, e
            ))
        })?;

        // Execute tool via MCP manager
        let args_map = if let Value::Object(map) = args {
            Some(map)
        } else {
            None
        };

        match mcp_manager
            .call_tool(&tool_call.function.name, args_map)
            .await
        {
            Ok(mcp_result) => {
                debug!(
                    tool_name = %tool_call.function.name,
                    call_id = %tool_call.id,
                    "Tool execution succeeded"
                );

                // Extract content from MCP result
                let output = if let Some(content) = mcp_result.content.first() {
                    // Serialize the entire content item
                    to_value(content)
                        .unwrap_or_else(|_| json!({"error": "Failed to serialize tool result"}))
                } else {
                    json!({"result": "success"})
                };

                let is_error = mcp_result.is_error.unwrap_or(false);
                let output_str = to_string(&output)
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
                warn!(
                    tool_name = %tool_call.function.name,
                    call_id = %tool_call.id,
                    error = %e,
                    "Tool execution failed"
                );

                let error_msg = format!("Tool execution failed: {}", e);
                let error_output = json!({
                    "error": error_msg.clone()
                });
                let error_output_str = to_string(&error_output)
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
fn build_next_request_with_tools(
    mut request: ResponsesRequest,
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<ToolResult>,
    analysis: Option<String>, // Analysis channel content (becomes reasoning content)
    partial_text: String,     // Final channel content (becomes message content)
) -> Result<ResponsesRequest, Box<Response>> {
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
            call_id: tool_call.id.clone(),
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
        let output_str = to_string(&tool_result.output).unwrap_or_else(|e| {
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

    // Switch tool_choice to "auto" for subsequent iterations
    // This prevents infinite loops when original tool_choice was "required" or specific function
    // After receiving tool results, the model should be free to decide whether to call more tools or finish
    request.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));

    Ok(request)
}

/// Tool execution result
///
/// Contains the result of executing a single MCP tool.
pub(crate) struct ToolResult {
    /// Tool call ID (for matching with request)
    pub(crate) call_id: String,

    /// Tool name
    #[allow(dead_code)] // Kept for documentation and future use
    pub(crate) tool_name: String,

    /// Tool output (JSON value)
    pub(crate) output: Value,

    /// Whether this is an error result
    pub(crate) is_error: bool,
}

/// Convert MCP tools to Responses API tool format
///
/// Converts MCP Tool entries (from rmcp SDK) to ResponseTool format so the model
/// knows about available MCP tools when making tool calls.
pub fn convert_mcp_tools_to_response_tools(mcp_tools: &[mcp::Tool]) -> Vec<ResponseTool> {
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

/// Extract MCP server label from request tools
///
/// Searches for the first MCP tool in the tools array and returns its server_label.
/// Falls back to "sglang-mcp" if no MCP tool with server_label is found.
fn extract_mcp_server_label(tools: Option<&[ResponseTool]>) -> String {
    tools
        .and_then(|tools| {
            tools.iter().find_map(|tool| {
                if matches!(tool.r#type, ResponseToolType::Mcp) {
                    tool.server_label.clone()
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| "sglang-mcp".to_string())
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
            error!(
                function = "load_previous_messages",
                prev_id = %prev_id_str,
                error = %e,
                "Failed to load previous response chain from storage"
            );
            error::internal_error(format!(
                "Failed to load previous response chain for {}: {}",
                prev_id_str, e
            ))
        })?;

    // Build conversation history from stored responses
    let mut history_items = Vec::new();

    // Helper to deserialize and collect items from a JSON array
    let deserialize_items = |arr: &Value, item_type: &str| -> Vec<ResponseInputOutputItem> {
        arr.as_array()
            .into_iter()
            .flat_map(|items| items.iter())
            .filter_map(|item| {
                from_value::<ResponseInputOutputItem>(item.clone())
                    .map_err(|e| {
                        warn!(
                            "Failed to deserialize stored {} item: {}. Item: {}",
                            item_type, e, item
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

    debug!(
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
