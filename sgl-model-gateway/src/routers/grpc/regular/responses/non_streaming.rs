//! Non-streaming execution for Regular Responses API
//!
//! This module handles non-streaming request execution:
//! - `route_responses_internal` - Core execution orchestration
//! - `execute_tool_loop` - MCP tool loop execution
//! - `execute_without_mcp` - Simple pipeline execution without MCP

use std::{sync::Arc, time::Instant};

use axum::response::Response;
use serde_json::json;
use tracing::{debug, error, trace, warn};

use super::{
    common::{
        build_mcp_list_tools_items_for_request, build_next_request, build_server_label_map,
        convert_mcp_tools_to_chat_tools, decode_mcp_function_name,
        extract_all_tool_calls_from_chat, load_conversation_history, prepare_chat_tools_and_choice,
        ExtractedToolCall, McpToolLookup, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    protocols::responses::{
        ResponseOutputItem, ResponseStatus, ResponsesRequest, ResponsesResponse,
    },
    routers::{
        error,
        grpc::common::responses::{
            ensure_mcp_connection, persist_response_if_needed, ResponsesContext,
        },
        mcp_utils::DEFAULT_MAX_ITERATIONS,
    },
};

/// Internal implementation for non-streaming responses
///
/// This is the core execution path that:
/// 1. Loads conversation history / response chain
/// 2. Checks for MCP tools
/// 3. Executes with or without MCP tool loop
/// 4. Persists to storage
pub(super) async fn route_responses_internal(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    // 1. Load conversation history and build modified request
    let modified_request = load_conversation_history(ctx, &request).await?;

    // 2. Check MCP connection and get whether MCP tools are present
    let (has_mcp_tools, server_keys) =
        ensure_mcp_connection(&ctx.mcp_manager, request.tools.as_deref()).await?;

    // Set the server keys in the context
    {
        let mut servers = ctx.requested_servers.write().unwrap();
        *servers = server_keys;
    }

    let responses_response = if has_mcp_tools {
        debug!("MCP tools detected, using tool loop");

        // Execute with MCP tool loop
        execute_tool_loop(
            ctx,
            modified_request,
            &request,
            headers,
            model_id,
            response_id.clone(),
        )
        .await?
    } else {
        // No MCP tools - execute without MCP (may have function tools or no tools)
        execute_without_mcp(
            ctx,
            &modified_request,
            &request,
            headers,
            model_id,
            response_id.clone(),
        )
        .await?
    };

    // 5. Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &responses_response,
        &request,
    )
    .await;

    Ok(responses_response)
}

/// Execute request without MCP tool loop (simple pipeline execution)
pub(super) async fn execute_without_mcp(
    ctx: &ResponsesContext,
    modified_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    // Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(modified_request).map_err(|e| {
        error!(
            function = "execute_without_mcp",
            error = %e,
            "Failed to convert ResponsesRequest to ChatCompletionRequest"
        );
        error::bad_request(
            "convert_request_failed",
            format!("Failed to convert request: {}", e),
        )
    })?;

    // Execute chat pipeline (errors already have proper HTTP status codes)
    let chat_response = ctx
        .pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            ctx.components.clone(),
        )
        .await?; // Preserve the Response error as-is

    // Convert ChatCompletionResponse → ResponsesResponse
    conversions::chat_to_responses(&chat_response, original_request, response_id).map_err(|e| {
        error!(
            function = "execute_without_mcp",
            error = %e,
            "Failed to convert ChatCompletionResponse to ResponsesResponse"
        );
        error::internal_error(
            "convert_to_responses_format_failed",
            format!("Failed to convert to responses format: {}", e),
        )
    })
}

/// Execute the MCP tool calling loop
///
/// This wraps pipeline.execute_chat_for_responses() in a loop that:
/// 1. Executes the chat pipeline
/// 2. Checks if response has tool calls
/// 3. If yes, executes MCP tools and builds resume request
/// 4. Repeats until no more tool calls or limit reached
pub(super) async fn execute_tool_loop(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    let servers = ctx.requested_servers.read().unwrap().clone();
    let server_labels = build_server_label_map(original_request.tools.as_deref());

    let mut state = ToolLoopState::new(original_request.input.clone(), McpToolLookup::default());

    // Configuration: max iterations as safety limit
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    trace!(
        "Starting MCP tool loop: max_tool_calls={:?}, max_iterations={}",
        max_tool_calls,
        DEFAULT_MAX_ITERATIONS
    );

    // Get MCP tools and convert to chat format (do this once before loop)
    let (mcp_chat_tools, tool_lookup) = convert_mcp_tools_to_chat_tools(
        &ctx.mcp_manager,
        &servers,
        &server_labels,
        original_request.tools.as_deref(),
    );
    state.tool_lookup = tool_lookup;
    trace!(
        "Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    loop {
        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request).map_err(|e| {
            error!(
                function = "tool_loop",
                iteration = state.iteration,
                error = %e,
                "Failed to convert ResponsesRequest to ChatCompletionRequest in tool loop"
            );
            error::bad_request(
                "convert_request_failed",
                format!("Failed to convert request: {}", e),
            )
        })?;

        // Prepare tools and tool_choice for this iteration
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // Execute chat pipeline (errors already have proper HTTP status codes)
        let chat_response = ctx
            .pipeline
            .execute_chat_for_responses(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                ctx.components.clone(),
            )
            .await?;

        // Check for function calls (extract all for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&chat_response);

        if !tool_calls.is_empty() {
            state.iteration += 1;

            // Record tool loop iteration metric
            Metrics::record_mcp_tool_iteration(&current_request.model);

            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls
            let mcp_tool_names: std::collections::HashSet<&str> = mcp_chat_tools
                .iter()
                .map(|t| t.function.name.as_str())
                .collect();
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| mcp_tool_names.contains(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // If ANY tool call is a function tool, return to caller immediately
            if !function_tool_calls.is_empty() {
                // Convert chat response to responses format (includes all tool calls)
                let responses_response = conversions::chat_to_responses(
                    &chat_response,
                    original_request,
                    response_id.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "tool_loop",
                        iteration = state.iteration,
                        error = %e,
                        context = "function_tool_calls",
                        "Failed to convert ChatCompletionResponse to ResponsesResponse"
                    );
                    error::internal_error(
                        "convert_to_responses_format_failed",
                        format!("Failed to convert to responses format: {}", e),
                    )
                })?;

                // Return response with function tool calls to caller
                return Ok(responses_response);
            }

            // All MCP tools - check combined limit BEFORE executing
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );

                // Convert chat response to responses format and mark as incomplete
                let mut responses_response = conversions::chat_to_responses(
                    &chat_response,
                    original_request,
                    response_id.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "tool_loop",
                        iteration = state.iteration,
                        error = %e,
                        context = "max_tool_calls_limit",
                        "Failed to convert ChatCompletionResponse to ResponsesResponse"
                    );
                    error::internal_error(
                        "convert_to_responses_format_failed",
                        format!("Failed to convert to responses format: {}", e),
                    )
                })?;

                // Mark as completed but with incomplete details
                responses_response.status = ResponseStatus::Completed;
                responses_response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                return Ok(responses_response);
            }

            // Execute all MCP tools
            for tool_call in mcp_tool_calls {
                trace!(
                    "Calling MCP tool '{}' (call_id: {}) with args: {}",
                    tool_call.name,
                    tool_call.call_id,
                    tool_call.arguments
                );

                let tool_start = Instant::now();
                let (_resolved_label, raw_tool_name) = decode_mcp_function_name(&tool_call.name)
                    .unwrap_or_else(|| ("mcp".to_string(), tool_call.name.clone()));

                let (output_str, success, error) =
                    match state.tool_lookup.tool_servers.get(&tool_call.name).cloned() {
                        Some(server_key) => {
                            let schema = state.tool_lookup.tool_schemas.get(&tool_call.name);
                            let args_map = match crate::mcp::tool_args::ToolArgs::from(
                                tool_call.arguments.as_str(),
                            )
                            .into_map(schema)
                            {
                                Ok(map) => Ok(map),
                                Err(e) => Err(format!("Invalid tool args: {}", e)),
                            };

                            match args_map {
                                Ok(args_map) => match ctx
                                    .mcp_manager
                                    .call_tool_on_server(&server_key, &raw_tool_name, args_map)
                                    .await
                                {
                                    Ok(result) => match serde_json::to_string(&result) {
                                        Ok(output) => (output, true, None),
                                        Err(e) => {
                                            let err =
                                                format!("Failed to serialize tool result: {}", e);
                                            warn!("{}", err);
                                            let error_json = json!({ "error": &err }).to_string();
                                            (error_json, false, Some(err))
                                        }
                                    },
                                    Err(err) => {
                                        let err_str = format!("tool call failed: {}", err);
                                        warn!("Tool execution failed: {}", err_str);
                                        let error_json = json!({ "error": &err_str }).to_string();
                                        (error_json, false, Some(err_str))
                                    }
                                },
                                Err(err) => {
                                    let error_json = json!({ "error": &err }).to_string();
                                    (error_json, false, Some(err))
                                }
                            }
                        }
                        None => {
                            let err = format!("Unknown MCP tool '{}'", tool_call.name);
                            let error_json = json!({ "error": &err }).to_string();
                            (error_json, false, Some(err))
                        }
                    };
                let tool_duration = tool_start.elapsed();

                // Record MCP tool metrics
                Metrics::record_mcp_tool_duration(
                    &current_request.model,
                    &raw_tool_name,
                    tool_duration,
                );
                Metrics::record_mcp_tool_call(
                    &current_request.model,
                    &raw_tool_name,
                    if success {
                        metrics_labels::RESULT_SUCCESS
                    } else {
                        metrics_labels::RESULT_ERROR
                    },
                );

                // Record the call in state
                state.record_call(
                    tool_call.call_id,
                    tool_call.name,
                    tool_call.arguments,
                    output_str,
                    success,
                    error,
                );

                // Increment total calls counter
                state.total_calls += 1;
            }

            // Build resume request with conversation history
            current_request = build_next_request(&state, &current_request);

            // Continue to next iteration
        } else {
            // No more tool calls, we're done
            trace!(
                "Tool loop completed: {} iterations, {} total calls",
                state.iteration,
                state.total_calls
            );

            // Convert final chat response to responses format
            let mut responses_response = conversions::chat_to_responses(
                &chat_response,
                original_request,
                response_id.clone(),
            )
            .map_err(|e| {
                error!(
                    function = "tool_loop",
                    iteration = state.iteration,
                    error = %e,
                    context = "final_response",
                    "Failed to convert ChatCompletionResponse to ResponsesResponse"
                );
                error::internal_error(
                    "convert_to_responses_format_failed",
                    format!("Failed to convert to responses format: {}", e),
                )
            })?;

            // Inject MCP metadata into output
            if state.total_calls > 0 {
                responses_response.output.retain(|item| {
                    !matches!(
                        item,
                        ResponseOutputItem::McpListTools { .. }
                            | ResponseOutputItem::McpCall { .. }
                    )
                });

                // Prepend mcp_list_tools items (one per server)
                let mcp_list_tools_items = build_mcp_list_tools_items_for_request(
                    &ctx.mcp_manager,
                    &servers,
                    original_request.tools.as_deref(),
                );
                let list_tools_count = mcp_list_tools_items.len();
                let mut existing = Vec::new();
                std::mem::swap(&mut existing, &mut responses_response.output);
                responses_response.output.extend(mcp_list_tools_items);
                responses_response.output.extend(state.mcp_call_items);
                responses_response.output.extend(existing);

                trace!(
                    "Injected MCP metadata: {} mcp_list_tools + {} mcp_call items",
                    list_tools_count,
                    state.total_calls
                );
            }

            return Ok(responses_response);
        }
    }
}
