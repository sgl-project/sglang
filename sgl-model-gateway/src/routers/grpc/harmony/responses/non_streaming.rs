//! Non-streaming Harmony Responses API implementation

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::response::Response;
use serde_json::{json, to_string};
use tracing::{debug, error, warn};

use super::{
    common::{
        build_mcp_tool_names_set, build_next_request_with_tools, build_server_label_map,
        inject_mcp_metadata, load_previous_messages, McpCallTracking, McpToolLookup,
    },
    execution::{convert_mcp_tools_to_response_tools, execute_mcp_tools, ToolResult},
};
use crate::{
    observability::metrics::Metrics,
    protocols::{
        common::{ToolCall, Usage},
        responses::{
            OutputTokensDetails, ResponseContentPart, ResponseOutputItem, ResponseReasoningContent,
            ResponseStatus, ResponseUsage, ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::{
        error,
        grpc::{
            common::responses::{
                ensure_mcp_connection, persist_response_if_needed, ResponsesContext,
            },
            harmony::processor::ResponsesIterationResult,
        },
        mcp_utils::DEFAULT_MAX_ITERATIONS,
    },
};

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
pub(crate) async fn serve_harmony_responses(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
    // Clone request for persistence
    let original_request = request.clone();

    // Load previous conversation history if previous_response_id is set
    let current_request = load_previous_messages(ctx, request).await?;

    // Check MCP connection and get whether MCP tools are present
    let (has_mcp_tools, server_keys) =
        ensure_mcp_connection(&ctx.mcp_manager, current_request.tools.as_deref()).await?;

    // Set the server keys in the context
    {
        let mut servers = ctx.requested_servers.write().unwrap();
        *servers = server_keys;
    }

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
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
) -> Result<ResponsesResponse, Response> {
    let mut iteration_count = 0;

    let request_tools = current_request.tools.clone();
    let response_request = current_request.clone();

    let servers = ctx.requested_servers.read().unwrap().clone();
    let server_labels = build_server_label_map(request_tools.as_deref());
    let mut mcp_tracking = McpCallTracking::new(McpToolLookup::default());

    // Extract user's max_tool_calls limit (if set)
    let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);

    // Add filtered MCP tools (static + requested dynamic) to the request
    let (mcp_response_tools, tool_lookup) = convert_mcp_tools_to_response_tools(
        &ctx.mcp_manager,
        &servers,
        &server_labels,
        request_tools.as_deref(),
    );
    if !mcp_response_tools.is_empty() {
        let mcp_tool_count = mcp_response_tools.len();
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

    loop {
        iteration_count += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&current_request.model);

        // Safety check: prevent infinite loops
        if iteration_count > DEFAULT_MAX_ITERATIONS {
            error!(
                function = "execute_with_mcp_loop",
                iteration_count = iteration_count,
                max_iterations = DEFAULT_MAX_ITERATIONS,
                "Maximum tool iterations exceeded"
            );
            return Err(error::internal_error(
                "tool_iterations_exceeded",
                format!(
                    "Maximum tool iterations ({}) exceeded",
                    DEFAULT_MAX_ITERATIONS
                ),
            ));
        }

        debug!(
            iteration = iteration_count,
            "Harmony Responses serving iteration"
        );

        // Execute through full pipeline
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
                let request_tools_slice = current_request.tools.as_deref().unwrap_or(&[]);
                let mcp_tool_names = build_mcp_tool_names_set(request_tools_slice);
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
                        "Reached tool call limit - returning incomplete response"
                    );

                    // Combine back for response
                    let all_tool_calls: Vec<_> = mcp_tool_calls
                        .into_iter()
                        .chain(function_tool_calls)
                        .collect();

                    // Build response with incomplete status - no tools executed due to limit
                    let request_tools = current_request.tools.clone();
                    let mut response = build_tool_response(
                        vec![],         // No MCP tools executed
                        vec![],         // No MCP results
                        all_tool_calls, // All tools returned as function calls (not executed)
                        analysis,
                        partial_text,
                        usage,
                        request_id,
                        Arc::new(response_request.clone()),
                    );

                    // Mark as completed with incomplete_details
                    response.status = ResponseStatus::Completed;
                    response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                    // Inject MCP metadata if any calls were executed
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(
                            &mut response,
                            &mcp_tracking,
                            &ctx.mcp_manager,
                            &servers,
                            &server_labels,
                            request_tools.as_deref(),
                        );
                    }

                    return Ok(response);
                }

                // Execute MCP tools (if any)
                let mcp_results = if !mcp_tool_calls.is_empty() {
                    execute_mcp_tools(
                        &ctx.mcp_manager,
                        &mcp_tool_calls,
                        &mut mcp_tracking,
                        &current_request.model,
                    )
                    .await?
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
                        Arc::new(response_request.clone()),
                    );

                    // Inject MCP metadata for all executed calls
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(
                            &mut response,
                            &mcp_tracking,
                            &ctx.mcp_manager,
                            &servers,
                            &server_labels,
                            request_tools.as_deref(),
                        );
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

                response.tools = response_request.tools.clone().unwrap_or_default();
                response.tool_choice = if let Some(ref tool_choice) = response_request.tool_choice {
                    to_string(tool_choice).unwrap_or_else(|_| "auto".to_string())
                } else {
                    "auto".to_string()
                };

                // Inject MCP metadata into final response
                inject_mcp_metadata(
                    &mut response,
                    &mcp_tracking,
                    &ctx.mcp_manager,
                    &servers,
                    &server_labels,
                    request_tools.as_deref(),
                );

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
    ctx: &ResponsesContext,
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

/// Build ResponsesResponse with tool calls (MCP and/or function tools)
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
            output_tokens_details: usage.completion_tokens_details.as_ref().and_then(|d| {
                d.reasoning_tokens.map(|tokens| OutputTokensDetails {
                    reasoning_tokens: tokens,
                })
            }),
        }))
        .build()
}
