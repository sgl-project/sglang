//! Streaming Harmony Responses API implementation

use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::Response;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{debug, warn};
use uuid::Uuid;

use super::{
    common::{
        build_mcp_tool_names_set, build_next_request_with_tools, build_server_label_map,
        build_tool_label_map, list_tools_by_server, load_previous_messages, McpCallTracking,
        McpToolLookup,
    },
    execution::{convert_mcp_tools_to_response_tools, execute_mcp_tools},
};
use crate::{
    observability::metrics::Metrics,
    protocols::responses::{ResponseToolType, ResponsesRequest},
    routers::{
        grpc::{
            common::responses::{
                build_sse_response, ensure_mcp_connection, persist_response_if_needed,
                streaming::{OutputItemType, ResponseStreamEventEmitter},
                ResponsesContext,
            },
            harmony::{processor::ResponsesIterationResult, streaming::HarmonyStreamingProcessor},
        },
        mcp_utils::{
            build_allowed_tools_map, filter_tools_for_server, resolve_server_label,
            DEFAULT_MAX_ITERATIONS,
        },
    },
};

/// Serve Harmony Responses API with streaming (SSE)
///
/// This is the streaming equivalent of `serve_harmony_responses()`.
/// Emits SSE events for lifecycle, MCP list_tools, and per-iteration streaming.
pub(crate) async fn serve_harmony_responses_stream(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
) -> Response {
    // Load previous conversation history if previous_response_id is set
    let current_request = match load_previous_messages(ctx, request.clone()).await {
        Ok(req) => req,
        Err(err_response) => return err_response,
    };

    // Check MCP connection BEFORE starting stream and get whether MCP tools are present
    let (has_mcp_tools, server_keys) =
        match ensure_mcp_connection(&ctx.mcp_manager, current_request.tools.as_deref()).await {
            Ok(result) => result,
            Err(response) => return response,
        };

    // Set the server keys in the context
    {
        let mut servers = ctx.requested_servers.write().unwrap();
        *servers = server_keys;
    }

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

/// Execute MCP tool loop with streaming
///
/// Handles the full MCP workflow:
/// - Adds static MCP tools to request
/// - Emits mcp_list_tools events
/// - Loops through tool execution iterations
/// - Emits final response.completed event
/// - Persists response internally
async fn execute_mcp_tool_loop_streaming(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) {
    let servers = ctx.requested_servers.read().unwrap().clone();
    let server_labels = build_server_label_map(current_request.tools.as_deref());
    let allowed_tools = build_allowed_tools_map(current_request.tools.as_deref());
    let tool_labels = build_tool_label_map(&ctx.mcp_manager, &servers, &server_labels);

    // Set tool labels in emitter for MCP call items
    emitter.set_mcp_tool_labels(tool_labels.clone());

    // Initialize MCP call tracking
    let mut mcp_tracking = McpCallTracking::new(McpToolLookup::default());
    // Extract user's max_tool_calls limit (if set)
    let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);

    // Add filtered MCP tools (static + requested dynamic) to the request
    let (mcp_response_tools, tool_lookup) = convert_mcp_tools_to_response_tools(
        &ctx.mcp_manager,
        &servers,
        &server_labels,
        current_request.tools.as_deref(),
    );
    let mcp_tool_count = mcp_response_tools.len();
    if !mcp_response_tools.is_empty() {
        mcp_tracking.tool_lookup = tool_lookup;

        let mut all_tools = current_request.tools.clone().unwrap_or_default();
        all_tools.extend(mcp_response_tools);
        current_request.tools = Some(all_tools);

        debug!(
            mcp_tool_count,
            total_tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            "Added MCP tools to request"
        );
    }

    let tools_by_server = list_tools_by_server(&ctx.mcp_manager, &servers);

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
    for (server_key, tools) in tools_by_server.iter() {
        let server_label = resolve_server_label(server_key, &server_labels);
        let filtered_tools = filter_tools_for_server(tools, &server_label, &allowed_tools);

        if filtered_tools.is_empty() {
            continue;
        }

        let (output_index, item_id) = emitter.allocate_output_index(OutputItemType::McpListTools);

        // Build tools list for item structure
        let tool_items: Vec<_> = filtered_tools
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
        let event = emitter.emit_mcp_list_tools_completed(output_index, &filtered_tools);
        if emitter.send_event(&event, tx).is_err() {
            return;
        }

        // Emit output_item.done
        let event = emitter.emit_output_item_done(output_index, &item_done);
        if emitter.send_event(&event, tx).is_err() {
            return;
        }
    }

    debug!(
        tool_count = mcp_tool_count,
        "Emitted mcp_list_tools on first iteration"
    );

    // MCP tool loop (max 10 iterations)
    let mut iteration_count = 0;
    loop {
        iteration_count += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&current_request.model);

        // Safety check: prevent infinite loops
        if iteration_count > DEFAULT_MAX_ITERATIONS {
            emitter.emit_error(
                &format!(
                    "Maximum tool iterations ({}) exceeded",
                    DEFAULT_MAX_ITERATIONS
                ),
                Some("max_iterations_exceeded"),
                tx,
            );
            return;
        }

        debug!(
            iteration = iteration_count,
            "Harmony Responses streaming iteration"
        );

        // Execute pipeline and get stream + load guards
        let (execution_result, _load_guards) = match ctx
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
        // Load guards are held during processing and dropped when iteration completes
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
                    Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                    None => DEFAULT_MAX_ITERATIONS,
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
                    match execute_mcp_tools(
                        &ctx.mcp_manager,
                        &mcp_tool_calls,
                        &mut mcp_tracking,
                        &current_request.model,
                    )
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
    ctx: &ResponsesContext,
    current_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) {
    debug!("No MCP tools - executing single iteration");

    // Execute pipeline and get stream + load guards
    let (execution_result, _load_guards) = match ctx
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
    // Load guards are held during processing and dropped when iteration completes
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
    // _load_guards dropped here after iteration completes

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
