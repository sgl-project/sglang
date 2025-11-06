//! MCP tool loop execution for /v1/responses endpoint

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, warn};
use uuid::Uuid;

use super::conversions;
use crate::{
    mcp::{self, McpManager},
    protocols::{
        chat::{
            ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionStreamResponse,
        },
        common::{Function, FunctionCallResponse, Tool, ToolCall, ToolChoice, ToolChoiceValue},
        responses::{
            self, McpToolInfo, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponseOutputItem, ResponseStatus, ResponseToolType, ResponsesRequest,
            ResponsesResponse,
        },
    },
    routers::grpc::{
        common::responses::streaming::{OutputItemType, ResponseStreamEventEmitter},
        error,
    },
};

/// Merge function tools from request with MCP tools and set tool_choice based on iteration
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
    // - Iteration 0: Use user's tool_choice or default to auto
    // - Iteration 1+: Always use auto to avoid infinite loops
    chat_request.tool_choice = if iteration == 0 {
        chat_request
            .tool_choice
            .clone()
            .or(Some(ToolChoice::Value(ToolChoiceValue::Auto)))
    } else {
        Some(ToolChoice::Value(ToolChoiceValue::Auto))
    };
}

/// Extract all tool calls from chat response (for parallel tool call support)
fn extract_all_tool_calls_from_chat(
    response: &ChatCompletionResponse,
) -> Vec<(String, String, String)> {
    // Check if response has choices with tool calls
    let Some(choice) = response.choices.first() else {
        return Vec::new();
    };
    let message = &choice.message;

    // Look for tool_calls in the message
    if let Some(tool_calls) = &message.tool_calls {
        tool_calls
            .iter()
            .map(|tool_call| {
                (
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    tool_call
                        .function
                        .arguments
                        .clone()
                        .unwrap_or_else(|| "{}".to_string()),
                )
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// State for tracking multi-turn tool calling loop
struct ToolLoopState {
    iteration: usize,
    total_calls: usize,
    conversation_history: Vec<ResponseInputOutputItem>,
    original_input: ResponseInput,
    mcp_call_items: Vec<ResponseOutputItem>,
    server_label: String,
}

impl ToolLoopState {
    fn new(original_input: ResponseInput, server_label: String) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
            mcp_call_items: Vec::new(),
            server_label,
        }
    }

    fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        success: bool,
        error: Option<String>,
    ) {
        // Add function_tool_call item with both arguments and output
        self.conversation_history
            .push(ResponseInputOutputItem::FunctionToolCall {
                id: call_id.clone(),
                call_id: call_id.clone(),
                name: tool_name.clone(),
                arguments: args_json_str.clone(),
                output: Some(output_str.clone()),
                status: Some("completed".to_string()),
            });

        // Add mcp_call output item for metadata
        let mcp_call = build_mcp_call_item(
            &tool_name,
            &args_json_str,
            &output_str,
            &self.server_label,
            success,
            error.as_deref(),
        );
        self.mcp_call_items.push(mcp_call);
    }
}

// ============================================================================
// MCP Metadata Builders
// ============================================================================

/// Generate unique ID for MCP items
fn generate_mcp_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

/// Build mcp_list_tools output item
fn build_mcp_list_tools_item(mcp: &Arc<McpManager>, server_label: &str) -> ResponseOutputItem {
    let tools = mcp.list_tools();
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

    ResponseOutputItem::McpListTools {
        id: generate_mcp_id("mcpl"),
        server_label: server_label.to_string(),
        tools: tools_info,
    }
}

/// Build mcp_call output item
fn build_mcp_call_item(
    tool_name: &str,
    arguments: &str,
    output: &str,
    server_label: &str,
    success: bool,
    error: Option<&str>,
) -> ResponseOutputItem {
    ResponseOutputItem::McpCall {
        id: generate_mcp_id("mcp"),
        status: if success { "completed" } else { "failed" }.to_string(),
        approval_request_id: None,
        arguments: arguments.to_string(),
        error: error.map(|e| e.to_string()),
        name: tool_name.to_string(),
        output: output.to_string(),
        server_label: server_label.to_string(),
    }
}

/// Execute the MCP tool calling loop
///
/// This wraps pipeline.execute_chat_for_responses() in a loop that:
/// 1. Executes the chat pipeline
/// 2. Checks if response has tool calls
/// 3. If yes, executes MCP tools and builds resume request
/// 4. Repeats until no more tool calls or limit reached
pub(super) async fn execute_tool_loop(
    ctx: &super::context::ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    // Get server label from original request tools
    let server_label = original_request
        .tools
        .as_ref()
        .and_then(|tools| {
            tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.clone())
        })
        .unwrap_or_else(|| "request-mcp".to_string());

    let mut state = ToolLoopState::new(original_request.input.clone(), server_label.clone());

    // Configuration: max iterations as safety limit
    const MAX_ITERATIONS: usize = 10;
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    debug!(
        "Starting MCP tool loop: server_label={}, max_tool_calls={:?}, max_iterations={}",
        server_label, max_tool_calls, MAX_ITERATIONS
    );

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_tools = ctx.mcp_manager.list_tools();
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&mcp_tools);
    debug!(
        "Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    loop {
        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| error::bad_request(format!("Failed to convert request: {}", e)))?;

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

            debug!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls
            let mcp_tool_names: std::collections::HashSet<&str> =
                mcp_tools.iter().map(|t| t.name.as_ref()).collect();
            let (mcp_tool_calls, function_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                .into_iter()
                .partition(|(_, tool_name, _)| mcp_tool_names.contains(tool_name.as_str()));

            debug!(
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
                    error::internal_error(format!("Failed to convert to responses format: {}", e))
                })?;

                // Return response with function tool calls to caller
                return Ok(responses_response);
            }

            // All MCP tools - check combined limit BEFORE executing
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(MAX_ITERATIONS),
                None => MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    MAX_ITERATIONS
                );

                // Convert chat response to responses format and mark as incomplete
                let mut responses_response = conversions::chat_to_responses(
                    &chat_response,
                    original_request,
                    response_id.clone(),
                )
                .map_err(|e| {
                    error::internal_error(format!("Failed to convert to responses format: {}", e))
                })?;

                // Mark as completed but with incomplete details
                responses_response.status = ResponseStatus::Completed;
                responses_response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                return Ok(responses_response);
            }

            // Execute all MCP tools
            for (call_id, tool_name, args_json_str) in mcp_tool_calls {
                debug!(
                    "Calling MCP tool '{}' (call_id: {}) with args: {}",
                    tool_name, call_id, args_json_str
                );

                let (output_str, success, error) = match ctx
                    .mcp_manager
                    .call_tool(tool_name.as_str(), args_json_str.as_str())
                    .await
                {
                    Ok(result) => match serde_json::to_string(&result) {
                        Ok(output) => (output, true, None),
                        Err(e) => {
                            let err = format!("Failed to serialize tool result: {}", e);
                            warn!("{}", err);
                            let error_json = json!({ "error": &err }).to_string();
                            (error_json, false, Some(err))
                        }
                    },
                    Err(err) => {
                        let err_str = format!("tool call failed: {}", err);
                        warn!("Tool execution failed: {}", err_str);
                        // Return error as output, let model decide how to proceed
                        let error_json = json!({ "error": &err_str }).to_string();
                        (error_json, false, Some(err_str))
                    }
                };

                // Record the call in state
                state.record_call(
                    call_id,
                    tool_name,
                    args_json_str,
                    output_str,
                    success,
                    error,
                );

                // Increment total calls counter
                state.total_calls += 1;
            }

            // Build resume request with conversation history
            // Start with original input
            let mut input_items = match &state.original_input {
                ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
                    id: format!("msg_u_{}", state.iteration),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                }],
                ResponseInput::Items(items) => {
                    items.iter().map(responses::normalize_input_item).collect()
                }
            };

            // Append all conversation history (function calls and outputs)
            input_items.extend_from_slice(&state.conversation_history);

            // Build new request for next iteration
            current_request = ResponsesRequest {
                input: ResponseInput::Items(input_items),
                model: current_request.model.clone(),
                instructions: current_request.instructions.clone(),
                tools: current_request.tools.clone(),
                max_output_tokens: current_request.max_output_tokens,
                temperature: current_request.temperature,
                top_p: current_request.top_p,
                stream: Some(false), // Always non-streaming in tool loop
                store: Some(false),  // Don't store intermediate responses
                background: Some(false),
                max_tool_calls: current_request.max_tool_calls,
                tool_choice: current_request.tool_choice.clone(),
                parallel_tool_calls: current_request.parallel_tool_calls,
                previous_response_id: None,
                conversation: None,
                user: current_request.user.clone(),
                metadata: current_request.metadata.clone(),
                // Additional fields from ResponsesRequest
                include: current_request.include.clone(),
                reasoning: current_request.reasoning.clone(),
                service_tier: current_request.service_tier.clone(),
                top_logprobs: current_request.top_logprobs,
                truncation: current_request.truncation.clone(),
                request_id: None,
                priority: current_request.priority,
                frequency_penalty: current_request.frequency_penalty,
                presence_penalty: current_request.presence_penalty,
                stop: current_request.stop.clone(),
                top_k: current_request.top_k,
                min_p: current_request.min_p,
                repetition_penalty: current_request.repetition_penalty,
            };

            // Continue to next iteration
        } else {
            // No more tool calls, we're done
            debug!(
                "Tool loop completed: {} iterations, {} total calls",
                state.iteration, state.total_calls
            );

            // Convert final chat response to responses format
            let mut responses_response = conversions::chat_to_responses(
                &chat_response,
                original_request,
                response_id.clone(),
            )
            .map_err(|e| {
                error::internal_error(format!("Failed to convert to responses format: {}", e))
            })?;

            // Inject MCP metadata into output
            if state.total_calls > 0 {
                // Prepend mcp_list_tools item
                let mcp_list_tools = build_mcp_list_tools_item(&ctx.mcp_manager, &server_label);
                responses_response.output.insert(0, mcp_list_tools);

                // Append all mcp_call items at the end
                responses_response.output.extend(state.mcp_call_items);

                debug!(
                    "Injected MCP metadata: 1 mcp_list_tools + {} mcp_call items",
                    state.total_calls
                );
            }

            return Ok(responses_response);
        }
    }
}

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
pub(super) async fn execute_tool_loop_streaming(
    ctx: &super::context::ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    // Create SSE channel for client
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Clone data for background task
    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();

    // Spawn background task for tool loop
    tokio::spawn(async move {
        let result = execute_tool_loop_streaming_internal(
            &ctx_clone,
            current_request,
            &original_request_clone,
            headers,
            model_id,
            tx.clone(),
        )
        .await;

        if let Err(e) = result {
            warn!("Streaming tool loop error: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "tool_loop_error"
                }
            });
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_event))));
        }

        // Send [DONE]
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Build SSE response
    let stream = UnboundedReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(body)
        .unwrap();

    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );

    response
}

/// Internal streaming tool loop implementation
async fn execute_tool_loop_streaming_internal(
    ctx: &super::context::ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    // Extract server label from original request tools
    let server_label = original_request
        .tools
        .as_ref()
        .and_then(|tools| {
            tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.clone())
        })
        .unwrap_or_else(|| "request-mcp".to_string());

    const MAX_ITERATIONS: usize = 10;
    let mut state = ToolLoopState::new(original_request.input.clone(), server_label.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Create response event emitter
    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = current_request.model.clone();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = emitter.emit_created();
    emitter.send_event(&event, &tx)?;
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, &tx)?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_tools = ctx.mcp_manager.list_tools();
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&mcp_tools);
    debug!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    loop {
        state.iteration += 1;
        if state.iteration > MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({})",
                MAX_ITERATIONS
            ));
        }

        debug!("Streaming MCP tool loop iteration {}", state.iteration);

        // Emit mcp_list_tools as first output item (only once, on first iteration)
        if !mcp_list_tools_emitted {
            let (output_index, item_id) =
                emitter.allocate_output_index(OutputItemType::McpListTools);

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

            // Build mcp_list_tools item
            let item = json!({
                "id": item_id,
                "type": "mcp_list_tools",
                "server_label": state.server_label,
                "status": "in_progress",
                "tools": []
            });

            // Emit output_item.added
            let event = emitter.emit_output_item_added(output_index, &item);
            emitter.send_event(&event, &tx)?;

            // Emit mcp_list_tools.in_progress
            let event = emitter.emit_mcp_list_tools_in_progress(output_index);
            emitter.send_event(&event, &tx)?;

            // Emit mcp_list_tools.completed
            let event = emitter.emit_mcp_list_tools_completed(output_index, &mcp_tools);
            emitter.send_event(&event, &tx)?;

            // Build complete item with tools
            let item_done = json!({
                "id": item_id,
                "type": "mcp_list_tools",
                "server_label": state.server_label,
                "status": "completed",
                "tools": tool_items
            });

            // Emit output_item.done
            let event = emitter.emit_output_item_done(output_index, &item_done);
            emitter.send_event(&event, &tx)?;

            emitter.complete_output_item(output_index);
            mcp_list_tools_emitted = true;
        }

        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

        // Prepare tools and tool_choice for this iteration (same logic as non-streaming)
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // Execute chat streaming
        let response = ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                ctx.components.clone(),
            )
            .await;

        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            convert_and_accumulate_stream(response.into_body(), &mut emitter, &tx).await?;

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);

        if !tool_calls.is_empty() {
            debug!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls
            let mcp_tool_names: std::collections::HashSet<&str> =
                mcp_tools.iter().map(|t| t.name.as_ref()).collect();
            let (mcp_tool_calls, function_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                .into_iter()
                .partition(|(_, tool_name, _)| mcp_tool_names.contains(tool_name.as_str()));

            debug!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // Check combined limit (only count MCP tools since function tools will be returned)
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(MAX_ITERATIONS),
                None => MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    MAX_ITERATIONS
                );
                break;
            }

            // Process each MCP tool call
            for (call_id, tool_name, args_json_str) in mcp_tool_calls {
                state.total_calls += 1;

                debug!(
                    "Executing tool call {}/{}: {} (call_id: {})",
                    state.total_calls, state.total_calls, tool_name, call_id
                );

                // Allocate output_index for this mcp_call item
                let (output_index, item_id) =
                    emitter.allocate_output_index(OutputItemType::McpCall);

                // Build initial mcp_call item
                let item = json!({
                    "id": item_id,
                    "type": "mcp_call",
                    "name": tool_name,
                    "server_label": state.server_label,
                    "status": "in_progress",
                    "arguments": ""
                });

                // Emit output_item.added
                let event = emitter.emit_output_item_added(output_index, &item);
                emitter.send_event(&event, &tx)?;

                // Emit mcp_call.in_progress
                let event = emitter.emit_mcp_call_in_progress(output_index, &item_id);
                emitter.send_event(&event, &tx)?;

                // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
                let event =
                    emitter.emit_mcp_call_arguments_delta(output_index, &item_id, &args_json_str);
                emitter.send_event(&event, &tx)?;

                // Emit mcp_call_arguments.done
                let event =
                    emitter.emit_mcp_call_arguments_done(output_index, &item_id, &args_json_str);
                emitter.send_event(&event, &tx)?;

                // Execute the MCP tool - manager handles parsing and type coercion
                debug!(
                    "Calling MCP tool '{}' with args: {}",
                    tool_name, args_json_str
                );
                let (output_str, success, error) = match ctx
                    .mcp_manager
                    .call_tool(tool_name.as_str(), args_json_str.as_str())
                    .await
                {
                    Ok(result) => match serde_json::to_string(&result) {
                        Ok(output) => {
                            // Emit mcp_call.completed
                            let event = emitter.emit_mcp_call_completed(output_index, &item_id);
                            emitter.send_event(&event, &tx)?;

                            // Build complete item with output
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_name,
                                "server_label": state.server_label,
                                "status": "completed",
                                "arguments": args_json_str,
                                "output": output
                            });

                            // Emit output_item.done
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, &tx)?;

                            emitter.complete_output_item(output_index);
                            (output, true, None)
                        }
                        Err(e) => {
                            let err = format!("Failed to serialize tool result: {}", e);
                            warn!("{}", err);
                            // Emit mcp_call.failed
                            let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err);
                            emitter.send_event(&event, &tx)?;

                            // Build failed item
                            let item_done = json!({
                                "id": item_id,
                                "type": "mcp_call",
                                "name": tool_name,
                                "server_label": state.server_label,
                                "status": "failed",
                                "arguments": args_json_str,
                                "error": &err
                            });

                            // Emit output_item.done
                            let event = emitter.emit_output_item_done(output_index, &item_done);
                            emitter.send_event(&event, &tx)?;

                            emitter.complete_output_item(output_index);
                            let error_json = json!({ "error": &err }).to_string();
                            (error_json, false, Some(err))
                        }
                    },
                    Err(err) => {
                        let err_str = format!("tool call failed: {}", err);
                        warn!("Tool execution failed: {}", err_str);
                        // Emit mcp_call.failed
                        let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err_str);
                        emitter.send_event(&event, &tx)?;

                        // Build failed item
                        let item_done = json!({
                            "id": item_id,
                            "type": "mcp_call",
                            "name": tool_name,
                            "server_label": state.server_label,
                            "status": "failed",
                            "arguments": args_json_str,
                            "error": &err_str
                        });

                        // Emit output_item.done
                        let event = emitter.emit_output_item_done(output_index, &item_done);
                        emitter.send_event(&event, &tx)?;

                        emitter.complete_output_item(output_index);
                        let error_json = json!({ "error": &err_str }).to_string();
                        (error_json, false, Some(err_str))
                    }
                };

                // Record the call in state
                state.record_call(
                    call_id,
                    tool_name,
                    args_json_str,
                    output_str,
                    success,
                    error,
                );
            }

            // If there are function tool calls, emit events and exit MCP loop
            if !function_tool_calls.is_empty() {
                debug!(
                    "Found {} function tool call(s) - emitting events and exiting MCP loop",
                    function_tool_calls.len()
                );

                // Emit function_tool_call events for each function tool
                for (call_id, tool_name, args_json_str) in function_tool_calls {
                    // Allocate output_index for this function_tool_call item
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemType::FunctionCall);

                    // Build initial function_tool_call item
                    let item = json!({
                        "id": item_id,
                        "type": "function_tool_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "status": "in_progress",
                        "arguments": ""
                    });

                    // Emit output_item.added
                    let event = emitter.emit_output_item_added(output_index, &item);
                    emitter.send_event(&event, &tx)?;

                    // Emit function_call_arguments.delta
                    let event = emitter.emit_function_call_arguments_delta(
                        output_index,
                        &item_id,
                        &args_json_str,
                    );
                    emitter.send_event(&event, &tx)?;

                    // Emit function_call_arguments.done
                    let event = emitter.emit_function_call_arguments_done(
                        output_index,
                        &item_id,
                        &args_json_str,
                    );
                    emitter.send_event(&event, &tx)?;

                    // Build complete item
                    let item_complete = json!({
                        "id": item_id,
                        "type": "function_tool_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "status": "completed",
                        "arguments": args_json_str
                    });

                    // Emit output_item.done
                    let event = emitter.emit_output_item_done(output_index, &item_complete);
                    emitter.send_event(&event, &tx)?;

                    emitter.complete_output_item(output_index);
                }

                // Break loop to return response to caller
                break;
            }

            // Build next request with conversation history
            let mut input_items = match &state.original_input {
                ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
                    id: format!("msg_u_{}", state.iteration),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                }],
                ResponseInput::Items(items) => {
                    items.iter().map(responses::normalize_input_item).collect()
                }
            };

            input_items.extend_from_slice(&state.conversation_history);

            current_request = ResponsesRequest {
                input: ResponseInput::Items(input_items),
                model: current_request.model.clone(),
                instructions: current_request.instructions.clone(),
                tools: current_request.tools.clone(),
                max_output_tokens: current_request.max_output_tokens,
                temperature: current_request.temperature,
                top_p: current_request.top_p,
                stream: Some(true),
                store: Some(false),
                background: Some(false),
                max_tool_calls: current_request.max_tool_calls,
                tool_choice: current_request.tool_choice.clone(),
                parallel_tool_calls: current_request.parallel_tool_calls,
                previous_response_id: None,
                conversation: None,
                user: current_request.user.clone(),
                metadata: current_request.metadata.clone(),
                include: current_request.include.clone(),
                reasoning: current_request.reasoning.clone(),
                service_tier: current_request.service_tier.clone(),
                top_logprobs: current_request.top_logprobs,
                truncation: current_request.truncation.clone(),
                request_id: None,
                priority: current_request.priority,
                frequency_penalty: current_request.frequency_penalty,
                presence_penalty: current_request.presence_penalty,
                stop: current_request.stop.clone(),
                top_k: current_request.top_k,
                min_p: current_request.min_p,
                repetition_penalty: current_request.repetition_penalty,
            };

            continue;
        }

        // No tool calls, this is the final response
        debug!("No tool calls found, ending streaming MCP loop");

        // Check for reasoning content
        let reasoning_content = accumulated_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone());

        // Emit reasoning item if present
        if let Some(reasoning) = reasoning_content {
            if !reasoning.is_empty() {
                emitter.emit_reasoning_item(&tx, Some(reasoning))?;
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        // Emit final response.completed event
        let usage_json = accumulated_response.usage.as_ref().map(|u| {
            json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens
            })
        });
        let event = emitter.emit_completed(usage_json.as_ref());
        emitter.send_event(&event, &tx)?;

        break;
    }

    Ok(())
}

/// Convert MCP tools to Chat API tool format
fn convert_mcp_tools_to_chat_tools(mcp_tools: &[mcp::Tool]) -> Vec<Tool> {
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

/// Convert chat stream to Responses API events while accumulating for tool call detection
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;

        // Parse chunk
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        if event == "data: [DONE]" {
            break;
        }

        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();
            if let Ok(chat_chunk) = serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                // Convert chat chunk to Responses API events and emit
                emitter.process_chunk(&chat_chunk, tx)?;

                // Accumulate for tool call detection
                accumulator.process_chunk(&chat_chunk);
            }
        }
    }

    Ok(accumulator.finalize())
}

/// Accumulates chat streaming chunks into complete ChatCompletionResponse
struct ChatResponseAccumulator {
    id: String,
    model: String,
    content: String,
    tool_calls: HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            tool_calls: HashMap::new(),
            finish_reason: None,
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if !chunk.id.is_empty() {
            self.id = chunk.id.clone();
        }
        if !chunk.model.is_empty() {
            self.model = chunk.model.clone();
        }

        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content.push_str(content);
            }

            // Accumulate tool calls
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let index = delta.index as usize;
                    let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                        id: String::new(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: String::new(),
                            arguments: Some(String::new()),
                        },
                    });

                    if let Some(id) = &delta.id {
                        entry.id = id.clone();
                    }
                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            entry.function.name = name.clone();
                        }
                        if let Some(args) = &function.arguments {
                            if let Some(ref mut existing_args) = entry.function.arguments {
                                existing_args.push_str(args);
                            }
                        }
                    }
                }
            }

            // Capture finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let mut tool_calls_vec: Vec<_> = self.tool_calls.into_iter().collect();
        tool_calls_vec.sort_by_key(|(index, _)| *index);
        let tool_calls: Vec<_> = tool_calls_vec.into_iter().map(|(_, call)| call).collect();

        ChatCompletionResponse {
            id: self.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: self.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: if self.content.is_empty() {
                        None
                    } else {
                        Some(self.content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    reasoning_content: None,
                },
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }],
            usage: None,
            system_fingerprint: None,
        }
    }
}
