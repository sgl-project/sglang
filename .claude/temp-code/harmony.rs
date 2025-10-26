//! Harmony (GPT-OSS) model support for /v1/responses endpoint
//!
//! This module implements direct Harmony processing for gpt-oss models.
//! Unlike non-Harmony models which convert to chat format, Harmony models
//! are processed directly using token-level parsing with separate output items.
//!
//! Architecture:
//! 1. Initialize StreamableParser from openai-harmony
//! 2. Process tokens directly from gRPC backend
//! 3. Route messages to separate output items (reasoning, tool calls, messages)
//! 4. Return ResponsesResponse with structured output array
//!
//! Token-level structured conversation format for gpt-oss models.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::{to_bytes, Body},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use openai_harmony::{
    chat::{
        Author, ChannelConfig, Content, DeveloperContent, Message as HarmonyMessage,
        ReasoningEffort::{High, Low, Medium},
        Role, SystemContent, TextContent, ToolDescription, ToolNamespaceConfig,
    },
    HarmonyEncoding, HarmonyEncodingName, StreamableParser,
};
use serde_json::json;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::types::BackgroundTaskInfo;
/// Re-export for MCP manager creation
pub(super) use crate::routers::openai::mcp::mcp_manager_from_request_tools as create_mcp_manager_from_request;
use crate::{
    data_connector::{
        ResponseId, SharedConversationItemStorage, SharedConversationStorage, SharedResponseStorage,
    },
    mcp::McpClientManager,
    protocols::{
        common::InputIds,
        generate::{GenerateFinishReason, GenerateRequest, GenerateResponse},
        responses::{
            ReasoningEffort as ProtocolEffort, ResponseContentPart, ResponseInput,
            ResponseInputOutputItem, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
            ResponseTool, ResponseToolType, ResponseUsage, ResponsesRequest, ResponsesResponse,
            ResponsesUsage::Modern, WebSearchAction,
        },
        sampling_params::SamplingParams,
    },
    routers::{
        grpc::{context::SharedComponents, pipeline::RequestPipeline},
        openai::conversations::persist_conversation_items,
    },
};

/// Main entry point for Harmony responses
///
/// This function handles Harmony (gpt-oss) model requests with direct token processing.
/// It does NOT convert to chat format - instead processes tokens directly through
/// Harmony's StreamableParser and routes output to separate ResponseOutputItems.
///
/// # Arguments
/// * `pipeline` - Request pipeline for low-level execution (tokenization, worker selection, etc.)
/// * `body` - The ResponsesRequest from the client
/// * `headers` - HTTP headers from the request
/// * `model_id` - Optional model ID override
/// * `components` - Shared components (tokenizer, parser factories)
/// * `response_storage` - Storage backend for responses
/// * `conversation_storage` - Storage backend for conversations
/// * `conversation_item_storage` - Storage backend for conversation items
/// * `background_tasks` - Shared map for background task cancellation
///
/// # Returns
/// HTTP Response with ResponsesResponse body (streaming or non-streaming)
#[allow(clippy::too_many_arguments)]
pub async fn route_harmony_responses(
    pipeline: &RequestPipeline,
    body: Arc<ResponsesRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
) -> Response {
    debug!(
        "Processing Harmony responses request for model: {:?}",
        model_id.as_deref().unwrap_or(&body.model)
    );

    // 1. Validate request parameters (same as non-Harmony)
    if body.previous_response_id.is_some() && body.conversation.is_some() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": {
                    "message": "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
                    "type": "invalid_request_error",
                    "code": "mutually_exclusive_parameters"
                }
            })),
        )
            .into_response();
    }

    // 2. Check execution mode
    let is_streaming = body.stream.unwrap_or(false);
    let is_background = body.background.unwrap_or(false);

    if is_streaming && is_background {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": {
                    "message": "Cannot use streaming with background mode.",
                    "type": "invalid_request_error",
                    "code": "incompatible_parameters"
                }
            })),
        )
            .into_response();
    }

    // 3. Route based on execution mode
    if is_streaming {
        execute_harmony_streaming(
            pipeline,
            body,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
        )
        .await
    } else if is_background {
        execute_harmony_background(
            pipeline,
            body,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            background_tasks,
        )
        .await
    } else {
        execute_harmony_sync(
            pipeline,
            body,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
        )
        .await
    }
}

// ============================================================================
// Synchronous Execution
// ============================================================================

/// Execute synchronous Harmony responses request
#[allow(clippy::too_many_arguments)]
async fn execute_harmony_sync(
    pipeline: &RequestPipeline,
    body: Arc<ResponsesRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
) -> Response {
    match execute_harmony_internal(
        pipeline,
        body,
        headers,
        model_id,
        components,
        response_storage,
        conversation_storage,
        conversation_item_storage,
    )
    .await
    {
        Ok(responses_response) => axum::Json(responses_response).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "error": {
                    "message": e,
                    "type": "internal_error"
                }
            })),
        )
            .into_response(),
    }
}

/// Internal implementation of Harmony execution with MCP support
///
/// This function checks if the request has MCP tools and routes accordingly:
/// - If MCP tools present: Execute with tool loop
/// - If no MCP tools: Execute single generation
#[allow(clippy::too_many_arguments)]
async fn execute_harmony_internal(
    pipeline: &RequestPipeline,
    body: Arc<ResponsesRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
) -> Result<ResponsesResponse, String> {
    // Check if request has MCP tools
    if let Some(tools) = &body.tools {
        if let Some(mcp_manager) = create_mcp_manager_from_request(tools).await {
            debug!("MCP tools detected for Harmony, using tool loop");

            // Load conversation history to get modified request
            let modified_request = super::load_conversation_history(
                &body,
                &response_storage,
                &conversation_storage,
                &conversation_item_storage,
            )
            .await?;

            // Execute with MCP tool loop
            return execute_harmony_with_mcp_loop(
                pipeline,
                modified_request,
                &body,
                headers,
                model_id,
                components,
                response_storage,
                conversation_storage,
                conversation_item_storage,
                mcp_manager,
            )
            .await;
        }
    }

    // No MCP tools, execute single generation (without tool descriptions)
    execute_harmony_internal_single(
        pipeline,
        body,
        headers,
        model_id,
        components,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        None, // No MCP manager available
    )
    .await
}

/// Execute Harmony with MCP tool loop for built-in tools
///
/// This wraps the Harmony execution in a loop that:
/// 1. Executes generation
/// 2. Checks for pending built-in tool calls (browser.*, python)
/// 3. If found, executes via MCP and continues
/// 4. If not found, returns final response
#[allow(clippy::too_many_arguments)]
async fn execute_harmony_with_mcp_loop(
    pipeline: &RequestPipeline,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    mcp_manager: Arc<McpClientManager>,
) -> Result<ResponsesResponse, String> {
    let mut iteration = 0;

    loop {
        if iteration >= MAX_MCP_ITERATIONS {
            warn!(
                "Harmony MCP tool loop reached max iterations ({})",
                MAX_MCP_ITERATIONS
            );
            break;
        }
        iteration += 1;

        debug!("Harmony MCP loop iteration {}", iteration);

        // Execute generation (with MCP manager for tool descriptions)
        let responses_response = execute_harmony_internal_single(
            pipeline,
            Arc::new(current_request.clone()),
            headers.clone(),
            model_id.clone(),
            components.clone(),
            response_storage.clone(),
            conversation_storage.clone(),
            conversation_item_storage.clone(),
            Some(&mcp_manager),
        )
        .await?;

        // Check for pending built-in tool calls
        if let Some((tool_name, arguments)) =
            find_pending_builtin_tool_call(&responses_response.output)
        {
            debug!("Found pending tool call: {}", tool_name);

            // Execute tool via MCP
            let tool_result_message =
                execute_builtin_tool_via_mcp(&mcp_manager, &tool_name, arguments).await?;

            // Convert current conversation to Harmony messages
            let mut conversation_messages = match &current_request.input {
                ResponseInput::Items(items) => {
                    // Convert items to Harmony messages
                    items
                        .iter()
                        .filter_map(|item| item_to_harmony_message(item).ok())
                        .collect::<Vec<_>>()
                }
                ResponseInput::Text(_) => {
                    // Already handled in previous iteration
                    vec![]
                }
            };

            // Add model's output (with tool call)
            for item in &responses_response.output {
                if let Some(msg) = output_item_to_harmony_message(item) {
                    conversation_messages.push(msg);
                }
            }

            // Add tool result message
            conversation_messages.push(tool_result_message);

            // Build new request with tool result
            current_request = ResponsesRequest {
                input: ResponseInput::Items(
                    conversation_messages
                        .iter()
                        .map(harmony_message_to_input_item)
                        .collect(),
                ),
                model: original_request.model.clone(),
                instructions: original_request.instructions.clone(),
                temperature: original_request.temperature,
                top_p: original_request.top_p,
                max_output_tokens: original_request.max_output_tokens,
                tools: original_request.tools.clone(),
                tool_choice: original_request.tool_choice.clone(),
                parallel_tool_calls: original_request.parallel_tool_calls,
                reasoning: original_request.reasoning.clone(),
                metadata: original_request.metadata.clone(),
                store: original_request.store,
                conversation: None, // Clear conversation ID for continuation
                previous_response_id: Some(responses_response.id.clone()),
                stream: original_request.stream,
                background: original_request.background,
                include: original_request.include.clone(),
                max_tool_calls: original_request.max_tool_calls,
                service_tier: original_request.service_tier.clone(),
                top_logprobs: original_request.top_logprobs,
                truncation: original_request.truncation.clone(),
                user: original_request.user.clone(),
                request_id: original_request.request_id.clone(),
                priority: original_request.priority,
                frequency_penalty: original_request.frequency_penalty,
                presence_penalty: original_request.presence_penalty,
                stop: original_request.stop.clone(),
                top_k: original_request.top_k,
                min_p: original_request.min_p,
                repetition_penalty: original_request.repetition_penalty,
            };

            // Continue loop
            continue;
        } else {
            // No pending tool calls, return final response
            debug!("No pending tool calls, returning final response");
            return Ok(responses_response);
        }
    }

    // If we exit loop due to max iterations, return last response
    execute_harmony_internal_single(
        pipeline,
        Arc::new(current_request),
        headers,
        model_id,
        components,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        Some(&mcp_manager), // Still have MCP manager available
    )
    .await
}

/// Internal implementation of single Harmony execution (no MCP loop)
#[allow(clippy::too_many_arguments)]
async fn execute_harmony_internal_single(
    pipeline: &RequestPipeline,
    body: Arc<ResponsesRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    mcp_manager: Option<&McpClientManager>,
) -> Result<ResponsesResponse, String> {
    // 1. Load conversation history
    let modified_request = super::load_conversation_history(
        &body,
        &response_storage,
        &conversation_storage,
        &conversation_item_storage,
    )
    .await?;

    // 2. Convert input to Harmony messages (with tool descriptions if MCP available)
    let harmony_messages = convert_input_to_harmony_messages(&modified_request, mcp_manager)?;

    // 3. Encode to tokens
    let encoding = get_harmony_encoding();
    let token_ids = encode_harmony_conversation(encoding, &harmony_messages)?;

    debug!("Encoded Harmony conversation to {} tokens", token_ids.len());

    // 4. Create GenerateRequest with token_ids
    let token_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();

    let mut sampling_params = SamplingParams::default();
    if let Some(temp) = body.temperature {
        sampling_params.temperature = Some(temp);
    }
    if let Some(top_p) = body.top_p {
        sampling_params.top_p = Some(top_p);
    }
    if let Some(max_tokens) = body.max_output_tokens {
        sampling_params.max_new_tokens = Some(max_tokens);
    }

    let generate_request = Arc::new(GenerateRequest {
        text: None,
        input_ids: Some(InputIds::Single(token_ids_i32)),
        input_embeds: None,
        image_data: None,
        video_data: None,
        audio_data: None,
        sampling_params: Some(sampling_params),
        return_logprob: None,
        logprob_start_len: None,
        top_logprobs_num: None,
        token_ids_logprob: None,
        return_text_in_logprobs: false,
        stream: false, // Non-streaming for now
        log_metrics: true,
        return_hidden_states: false,
        modalities: None,
        session_params: None,
        lora_path: None,
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        data_parallel_rank: None,
        background: false,
        conversation_id: None,
        priority: None,
        extra_key: None,
        no_logs: false,
        custom_labels: None,
        return_bytes: false,
        return_entropy: false,
        rid: None,
    });

    // 5. Execute through pipeline
    let generate_response = pipeline
        .execute_generate(generate_request, headers, model_id, components)
        .await;

    // 6. Extract response body
    let (_, body_data) = generate_response.into_parts();
    let body_bytes = to_bytes(body_data, usize::MAX)
        .await
        .map_err(|e| format!("Failed to read response body: {}", e))?;

    // 7. Parse GenerateResponse
    let generate_resp: GenerateResponse = serde_json::from_slice(&body_bytes)
        .map_err(|e| format!("Failed to parse generate response: {}", e))?;

    // 8. Process output tokens through Harmony parser
    let mut parser = StreamableParser::new(encoding.clone(), Some(Role::Assistant))
        .map_err(|e| format!("Failed to create Harmony parser: {}", e))?;

    // Extract output token IDs directly from backend response
    let output_token_ids = &generate_resp.output_ids;

    debug!(
        "Processing {} output tokens through Harmony parser",
        output_token_ids.len()
    );

    // Feed tokens to parser
    for &token_id in output_token_ids {
        parser
            .process(token_id)
            .map_err(|e| format!("Failed to process token: {}", e))?;
    }

    // 9. Extract completed messages and convert to ResponseOutputItems
    let mut output_items = parse_all_messages(&parser)?;

    // Parse remaining state for incomplete generation
    let remaining_items = parse_remaining_state(&parser);
    if !remaining_items.is_empty() {
        debug!(
            "Parser has {} incomplete item(s) in current state",
            remaining_items.len()
        );
        output_items.extend(remaining_items);
    }

    debug!(
        "Harmony parser extracted {} output items from {} tokens",
        output_items.len(),
        output_token_ids.len()
    );

    // 10. Build ResponsesResponse
    let response_id = Uuid::new_v4().simple().to_string();
    let response_id_full = format!("resp_{}", response_id);

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // - "length" → "incomplete" (hit max tokens)
    // - "abort" → "cancelled" (request was aborted)
    // - empty output → "incomplete"
    // - otherwise → "completed"
    // Map finish_reason to status:
    // - "length" → "incomplete" (hit max tokens)
    // - "abort" → "cancelled" (request was aborted)
    // - empty output → "incomplete"
    // - otherwise → "completed"
    let status = if output_items.is_empty() {
        ResponseStatus::Incomplete
    } else {
        match &generate_resp.meta_info.finish_reason {
            GenerateFinishReason::Length { .. } => ResponseStatus::Incomplete,
            GenerateFinishReason::Other(val) => {
                // Check if it's an abort
                if val
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("abort"))
                {
                    ResponseStatus::Cancelled
                } else {
                    ResponseStatus::Completed
                }
            }
            GenerateFinishReason::Stop => ResponseStatus::Completed,
        }
    };

    let responses_response = ResponsesResponse {
        id: response_id_full,
        object: "response".to_string(),
        created_at: created,
        status,
        error: None,
        incomplete_details: None,
        instructions: body.instructions.clone(),
        max_output_tokens: body.max_output_tokens,
        model: body.model.clone(),
        output: output_items,
        parallel_tool_calls: body.parallel_tool_calls.unwrap_or(true),
        previous_response_id: body.previous_response_id.clone(),
        reasoning: None,
        store: body.store.unwrap_or(true),
        temperature: body.temperature,
        text: None,
        tool_choice: "auto".to_string(),
        tools: Vec::new(),
        top_p: body.top_p,
        truncation: None,
        usage: Some(Modern(ResponseUsage {
            input_tokens: token_ids.len() as u32,
            output_tokens: output_token_ids.len() as u32,
            total_tokens: (token_ids.len() + output_token_ids.len()) as u32,
            input_tokens_details: None,
            output_tokens_details: None,
        })),
        user: None,
        metadata: body.metadata.clone().unwrap_or_default(),
    };

    // 11. Persist response to storage if store=true
    // Check if response is already cancelled before updating
    if body.store.unwrap_or(true) {
        if let Ok(response_json) = serde_json::to_value(&responses_response) {
            // Check if response is already cancelled (don't overwrite)
            let resp_id = ResponseId::from(responses_response.id.as_str());
            let should_store = match response_storage.get_response(&resp_id).await {
                Ok(Some(stored)) => {
                    // Check status from raw_response
                    let current_status = stored
                        .raw_response
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    // Don't update if already cancelled
                    current_status != "cancelled"
                }
                Ok(None) => true, // Not stored yet, store it
                Err(_) => true,   // Error reading, try to store anyway
            };

            if should_store {
                if let Err(e) = persist_conversation_items(
                    conversation_storage,
                    conversation_item_storage,
                    response_storage,
                    &response_json,
                    &body,
                )
                .await
                {
                    warn!("Failed to persist Harmony response: {}", e);
                } else {
                    debug!("Persisted Harmony response: {}", responses_response.id);
                }
            } else {
                debug!(
                    "Skipping storage update for cancelled response: {}",
                    responses_response.id
                );
            }
        }
    }

    Ok(responses_response)
}

/// Create system message for Harmony models
///
/// Create system message for Harmony models:
/// - Sets reasoning effort if provided
/// - Sets conversation start date (current date for determinism warning)
/// - Adds instructions to model identity if provided
/// - Adds built-in tool descriptions (browser, python, container) if MCP manager provided
/// - Disables commentary channel if no custom tools
fn create_harmony_system_message(
    request: &ResponsesRequest,
    mcp_manager: Option<&McpClientManager>,
) -> Result<HarmonyMessage, String> {
    use chrono::Utc;

    let mut sys_content = SystemContent::new();

    // Set reasoning effort if provided
    // Convert from protocol ReasoningEffort to openai_harmony ReasoningEffort
    if let Some(ref reasoning) = request.reasoning {
        if let Some(ref protocol_effort) = reasoning.effort {
            let harmony_effort = match protocol_effort {
                ProtocolEffort::High => High,
                ProtocolEffort::Medium => Medium,
                ProtocolEffort::Low => Low,
            };
            sys_content = sys_content.with_reasoning_effort(harmony_effort);
        }
    }

    // Append instructions to model_identity if VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS is true
    // Otherwise, instructions go to developer message
    let use_system_instructions = std::env::var("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS")
        .ok()
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(true); // Default to true (put in system message)

    if use_system_instructions {
        if let Some(ref instructions) = request.instructions {
            let new_identity = if let Some(identity) = &sys_content.model_identity {
                format!("{}\n{}", identity, instructions)
            } else {
                instructions.clone()
            };
            sys_content = sys_content.with_model_identity(&new_identity);
        }
    }

    // Set conversation start date (current date)
    // Note: Using current date brings non-determinism to responses
    let start_date = Utc::now().format("%Y-%m-%d").to_string();
    sys_content = sys_content.with_conversation_start_date(&start_date);

    // Add built-in tool descriptions if MCP manager is available
    if let Some(mcp) = mcp_manager {
        if let Some(tools) = &request.tools {
            // Collect tool types to enable
            let mut tool_types_to_enable = Vec::new();

            // First, collect tool types from regular tools
            for tool in tools {
                let tool_type = match tool.r#type {
                    ResponseToolType::WebSearchPreview => Some("browser"),
                    ResponseToolType::CodeInterpreter => Some("python"),
                    ResponseToolType::Container => Some("container"),
                    ResponseToolType::Mcp => None, // Will be handled by allowlist below
                    ResponseToolType::Function => None, // Function tools go in developer message
                };

                if let Some(t) = tool_type {
                    tool_types_to_enable.push(t);
                }
            }

            // Check GPT_OSS_SYSTEM_TOOL_MCP_LABELS env var for MCP tool allowlist
            // Allows MCP tools to enable built-in tools (browser, python, container)
            // Example: GPT_OSS_SYSTEM_TOOL_MCP_LABELS=web_search_preview,code_interpreter
            if let Ok(allowlist_str) = std::env::var("GPT_OSS_SYSTEM_TOOL_MCP_LABELS") {
                let allowlist: Vec<&str> = allowlist_str.split(',').map(|s| s.trim()).collect();

                // For each MCP tool, check if server_label is in allowlist
                for tool in tools {
                    if matches!(tool.r#type, ResponseToolType::Mcp) {
                        if let Some(ref label) = tool.server_label {
                            if allowlist.contains(&label.as_str()) {
                                // Map server_label to built-in tool type
                                let builtin_tool = match label.as_str() {
                                    "web_search_preview" => Some("browser"),
                                    "code_interpreter" => Some("python"),
                                    "container" => Some("container"),
                                    _ => None,
                                };

                                if let Some(t) = builtin_tool {
                                    if !tool_types_to_enable.contains(&t) {
                                        tool_types_to_enable.push(t);
                                        debug!(
                                            "MCP tool with label '{}' enabled built-in tool '{}'",
                                            label, t
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Now enable all collected tool types
            for tool_type in tool_types_to_enable {
                if let Some(config) = get_builtin_tool_config(mcp, tool_type) {
                    sys_content = sys_content.with_tools(config);
                    debug!("Added {} tool configuration to system message", tool_type);
                }
            }
        }
    }

    // Disable commentary channel if no custom function tools
    // Built-in tools (browser, python, container) don't use commentary channel
    // Custom function tools DO use commentary channel for function calls
    let has_custom_tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .any(|tool| matches!(tool.r#type, ResponseToolType::Function))
        })
        .unwrap_or(false);

    if !has_custom_tools {
        if let Some(ref channel_config) = sys_content.channel_config {
            let valid_channels: Vec<_> = channel_config
                .valid_channels
                .iter()
                .filter(|&ch| ch != "commentary")
                .cloned()
                .collect();
            let new_config = ChannelConfig::require_channels(&valid_channels);
            sys_content = sys_content.with_channel_config(new_config);
        }
    } else {
        debug!("Commentary channel enabled for custom function tools");
    }

    Ok(HarmonyMessage::from_role_and_content(
        Role::System,
        Content::SystemContent(sys_content),
    ))
}

/// Get tool namespace config for built-in tools if available in MCP manager
///
/// Returns the appropriate ToolNamespaceConfig from openai-harmony if the
/// corresponding MCP tool is available.
fn get_builtin_tool_config(
    mcp_manager: &McpClientManager,
    tool_type: &str,
) -> Option<ToolNamespaceConfig> {
    match tool_type {
        "python" => {
            if mcp_manager.has_tool("python") {
                Some(ToolNamespaceConfig::python())
            } else {
                None
            }
        }
        "browser" => {
            // Check if any browser tools are available
            let has_browser = mcp_manager.has_tool("browser.search")
                || mcp_manager.has_tool("browser.open")
                || mcp_manager.has_tool("browser.find");

            if has_browser {
                Some(ToolNamespaceConfig::browser())
            } else {
                None
            }
        }
        "container" => {
            // Container tool support: openai-harmony doesn't have a built-in container config yet
            // For now, we skip it - this would need to be added to openai-harmony library
            // TODO: Add container tool configuration when openai-harmony adds support
            debug!("Container tool requested but not yet supported in openai-harmony");
            None
        }
        _ => None,
    }
}

/// Create developer message for Harmony models with custom tools
///
/// Create developer message for Harmony models:
/// - Adds instructions if not already in system message
/// - Adds function tool definitions for custom tools
fn create_harmony_developer_message(
    request: &ResponsesRequest,
) -> Result<Option<HarmonyMessage>, String> {
    // Check if instructions should go to developer message
    let use_system_instructions = std::env::var("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS")
        .ok()
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(true); // Default to true (put in system message)

    // Extract function tools from request
    let function_tools: Vec<&ResponseTool> = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .filter(|tool| matches!(tool.r#type, ResponseToolType::Function))
                .collect()
        })
        .unwrap_or_default();

    // Create developer message if:
    // 1. There are function tools, OR
    // 2. Instructions should go to developer message (not system)
    let need_dev_message =
        !function_tools.is_empty() || (!use_system_instructions && request.instructions.is_some());

    if !need_dev_message {
        return Ok(None);
    }

    let mut dev_content = DeveloperContent::new();

    // Add instructions to developer message if not in system message
    if !use_system_instructions {
        if let Some(ref instructions) = request.instructions {
            dev_content = dev_content.with_instructions(instructions);
        }
    }

    // Convert function tools to ToolDescription
    let tool_descriptions: Vec<ToolDescription> = function_tools
        .iter()
        .filter_map(|tool| {
            tool.function.as_ref().map(|func| {
                ToolDescription::new(
                    &func.name,
                    func.description.as_deref().unwrap_or(""),
                    func.parameters.clone(),
                )
            })
        })
        .collect();

    if tool_descriptions.is_empty() {
        return Ok(None);
    }

    // Add function tools to developer content
    dev_content = dev_content.with_function_tools(tool_descriptions);

    debug!(
        "Created developer message with {} function tools",
        function_tools.len()
    );

    Ok(Some(HarmonyMessage::from_role_and_content(
        Role::Developer,
        Content::DeveloperContent(dev_content),
    )))
}

/// Convert ResponsesRequest input to Harmony Messages
///
/// **UPDATED**: Now injects system/developer messages for new conversations
/// Message construction pattern for Harmony models:
///
/// For new conversations (no previous_response_id):
/// 1. Add system message with reasoning effort, instructions, date, channel config
/// 2. Add developer message if custom tools are present
/// 3. Add user input
///
/// For continuing conversations (with previous_response_id):
/// 1. Previous messages should already be loaded via load_conversation_history
/// 2. Just append new user input
fn convert_input_to_harmony_messages(
    request: &ResponsesRequest,
    mcp_manager: Option<&McpClientManager>,
) -> Result<Vec<HarmonyMessage>, String> {
    let mut messages = Vec::new();

    // Check if this is a new conversation
    let is_new_conversation = request.previous_response_id.is_none();

    // For new conversations, inject system and developer messages
    if is_new_conversation {
        // 1. Add system message (with tool descriptions if MCP manager available)
        let system_message = create_harmony_system_message(request, mcp_manager)?;
        messages.push(system_message);

        // 2. Add developer message if custom tools present
        if let Some(developer_message) = create_harmony_developer_message(request)? {
            messages.push(developer_message);
        }

        debug!(
            "Injected system/developer messages for new Harmony conversation (reasoning: {:?}, tools: {})",
            request.reasoning.as_ref().and_then(|r| r.effort.as_ref()),
            request.tools.as_ref().map(|t| t.len()).unwrap_or(0)
        );
    }

    // Get input items based on ResponseInput type
    let input_items: Vec<ResponseInputOutputItem> = match &request.input {
        ResponseInput::Text(text) => {
            // Convert text to a user message
            vec![ResponseInputOutputItem::Message {
                id: "user_input".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text: text.clone() }],
                status: None,
            }]
        }
        ResponseInput::Items(items) => items.clone(),
    };

    // Track whether we're continuing from a previous response (for chain-of-thought removal)
    let is_continuing_conversation = !is_new_conversation;

    // Convert each input item to Harmony Message
    // Support for tool call messages in input
    for item in input_items {
        match item {
            ResponseInputOutputItem::Message { role, content, .. } => {
                // Extract text from content parts
                let text = content
                    .iter()
                    .filter_map(|c| match c {
                        ResponseContentPart::InputText { text } => Some(text.as_str()),
                        ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");

                // Convert role to Harmony Role
                let (harmony_role, text_prefix) = match role.as_str() {
                    "user" => (Role::User, ""),
                    "assistant" => (Role::Assistant, ""),
                    "system" => {
                        // System messages from user become developer messages
                        // with "Instructions:\n" prefix
                        (Role::Developer, "Instructions:\n")
                    }
                    "developer" => (Role::Developer, ""),
                    _ => (Role::User, ""), // Default to user
                };

                // Create Harmony Message
                let mut msg = HarmonyMessage {
                    author: Author {
                        role: harmony_role,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent {
                        text: format!("{}{}", text_prefix, text),
                    })],
                    channel: None,
                    content_type: None,
                };

                // Set channel="final" for assistant messages
                if role == "assistant" {
                    msg.channel = Some("final".to_string());
                }

                messages.push(msg);
            }

            ResponseInputOutputItem::Reasoning { content, .. } => {
                // Reasoning input: Create assistant message with reasoning text
                // Channel will be set by model during generation
                let text = content
                    .iter()
                    .map(|c| match c {
                        ResponseReasoningContent::ReasoningText { text } => text.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join("");

                messages.push(HarmonyMessage {
                    author: Author {
                        role: Role::Assistant,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text })],
                    channel: Some("analysis".to_string()), // Reasoning uses analysis channel
                    content_type: None,
                });
            }

            ResponseInputOutputItem::FunctionToolCall {
                name,
                arguments,
                output,
                ..
            } => {
                // Function tool call has two cases:
                // 1. Without output: This is the tool call request (assistant → function)
                // 2. With output: This is the tool result (function → assistant)

                if let Some(output_text) = output {
                    // Case 2: Tool result (function → assistant)
                    // Create tool message with author Role::TOOL and name "functions.{name}"
                    messages.push(HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(format!("functions.{}", name)),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent { text: output_text })],
                        channel: Some("commentary".to_string()),
                        content_type: None,
                    });
                } else {
                    // Case 1: Tool call request (assistant → function)
                    // Create assistant message with:
                    // - channel = "commentary"
                    // - recipient = "functions.{name}"
                    // - content_type = "json"
                    messages.push(HarmonyMessage {
                        author: Author {
                            role: Role::Assistant,
                            name: None,
                        },
                        recipient: Some(format!("functions.{}", name)),
                        content: vec![Content::Text(TextContent { text: arguments })],
                        channel: Some("commentary".to_string()),
                        content_type: Some("json".to_string()),
                    });
                }
            }

            ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                // Simple input message: Convert to Harmony message
                use crate::protocols::responses::StringOrContentParts;

                // Extract text from content (string or array of parts)
                let text = match content {
                    StringOrContentParts::String(s) => s.clone(),
                    StringOrContentParts::Array(parts) => {
                        parts
                            .iter()
                            .filter_map(|c| match c {
                                ResponseContentPart::InputText { text } => Some(text.as_str()),
                                ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    }
                };

                // Convert role to Harmony Role
                let (harmony_role, text_prefix) = match role.as_str() {
                    "user" => (Role::User, ""),
                    "assistant" => (Role::Assistant, ""),
                    "system" => {
                        // System messages from user become developer messages
                        // with "Instructions:\n" prefix
                        (Role::Developer, "Instructions:\n")
                    }
                    "developer" => (Role::Developer, ""),
                    _ => (Role::User, ""), // Default to user
                };

                // Create Harmony Message
                let mut msg = HarmonyMessage {
                    author: Author {
                        role: harmony_role,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent {
                        text: format!("{}{}", text_prefix, text),
                    })],
                    channel: None,
                    content_type: None,
                };

                // Set channel="final" for assistant messages
                if role == "assistant" {
                    msg.channel = Some("final".to_string());
                }

                messages.push(msg);
            }
        }
    }

    // Remove chain-of-thought (analysis channel messages) from most recent turn
    // when continuing a conversation (lines 944-968 in serving_responses.py)
    if is_continuing_conversation && !messages.is_empty() {
        // Check if last message has channel="final"
        if let Some(last_msg) = messages.last() {
            if last_msg.channel.as_deref() == Some("final") {
                // Find the previous "final" message index (working backwards from second-to-last)
                let mut prev_final_idx: Option<usize> = None;
                for i in (0..messages.len() - 1).rev() {
                    if messages[i].channel.as_deref() == Some("final") {
                        prev_final_idx = Some(i);
                        break;
                    }
                }

                // If we found a previous "final" message, remove analysis messages in between
                if let Some(prev_idx) = prev_final_idx {
                    // Collect messages from the most recent turn (prev_final + 1 to end)
                    // Keep only non-analysis messages
                    let recent_turn_start = prev_idx + 1;
                    let filtered_recent: Vec<_> = messages
                        .iter()
                        .skip(recent_turn_start)
                        .filter(|msg| msg.channel.as_deref() != Some("analysis"))
                        .cloned()
                        .collect();

                    // Replace recent turn with filtered version
                    messages.truncate(recent_turn_start);
                    messages.extend(filtered_recent);

                    debug!(
                        "Removed analysis messages from most recent turn (after message {})",
                        prev_idx
                    );
                }
            }
        }
    }

    if messages.is_empty() {
        return Err("No input messages found in request".to_string());
    }

    Ok(messages)
}

/// Encode Harmony conversation to token IDs
fn encode_harmony_conversation(
    encoding: &HarmonyEncoding,
    messages: &[HarmonyMessage],
) -> Result<Vec<u32>, String> {
    let mut token_ids = Vec::new();

    // Use Harmony's render_conversation_into to encode messages
    // Third parameter is config - we pass None for default configuration
    encoding
        .render_conversation_into(messages.iter(), &mut token_ids, None)
        .map_err(|e| format!("Failed to encode Harmony conversation: {}", e))?;

    Ok(token_ids)
}

// ============================================================================
// Streaming Execution
// ============================================================================

/// Execute streaming Harmony responses request
///
/// Implements real-time streaming of Harmony model output with incremental message parsing.
/// Parse all completed messages from the Harmony parser:
/// 1. Load conversation history and encode to tokens
/// 2. Execute generate request to get token stream
/// 3. Process tokens incrementally through StreamableParser
/// 4. Emit SSE events as new messages complete
/// 5. Persist final response when done
#[allow(clippy::too_many_arguments)]
async fn execute_harmony_streaming(
    pipeline: &RequestPipeline,
    body: Arc<ResponsesRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
) -> Response {
    debug!("Starting Harmony streaming execution");

    // 1. Load conversation history
    let modified_request = match super::load_conversation_history(
        &body,
        &response_storage,
        &conversation_storage,
        &conversation_item_storage,
    )
    .await
    {
        Ok(req) => req,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({
                    "error": {
                        "message": e,
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 2. Convert to Harmony messages and encode (streaming doesn't support MCP yet)
    let (harmony_messages, token_ids) = match convert_and_encode_harmony(&modified_request, None) {
        Ok(result) => result,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({
                    "error": {
                        "message": e,
                        "type": "encoding_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 3. Create GenerateRequest for backend execution
    // Clone token_ids for later use in usage calculation
    let token_ids_for_usage = token_ids.clone();
    let generate_request = match create_generate_request_streaming(&body, token_ids) {
        Ok(req) => Arc::new(req),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({
                    "error": {
                        "message": e,
                        "type": "request_creation_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 4. Execute streaming generate request
    let generate_response = pipeline
        .execute_generate(
            generate_request,
            headers.clone(),
            model_id.clone(),
            components.clone(),
        )
        .await;

    // 5. Extract body and create SSE channel
    let (parts, generate_body) = generate_response.into_parts();
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // 6. Spawn background task to process stream
    let body_clone = body.clone();
    let response_storage_clone = response_storage.clone();
    let conversation_storage_clone = conversation_storage.clone();
    let conversation_item_storage_clone = conversation_item_storage.clone();

    tokio::spawn(async move {
        if let Err(e) = process_harmony_stream(
            generate_body,
            body_clone,
            harmony_messages,
            token_ids_for_usage, // Pass input token IDs for usage calculation
            response_storage_clone,
            conversation_storage_clone,
            conversation_item_storage_clone,
            tx.clone(),
        )
        .await
        {
            error!("Error processing Harmony stream: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "stream_error"
                }
            });
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_event))));
        }

        // Send final [DONE] event
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // 7. Build SSE response
    let stream = UnboundedReceiverStream::new(rx);
    let response_body = Body::from_stream(stream);

    let mut response = Response::builder()
        .status(parts.status)
        .body(response_body)
        .unwrap();

    // Set SSE headers
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

/// Helper function to convert request to Harmony messages and encode
fn convert_and_encode_harmony(
    request: &ResponsesRequest,
    mcp_manager: Option<&McpClientManager>,
) -> Result<(Vec<HarmonyMessage>, Vec<u32>), String> {
    let harmony_messages = convert_input_to_harmony_messages(request, mcp_manager)?;
    let encoding = get_harmony_encoding();
    let token_ids = encode_harmony_conversation(encoding, &harmony_messages)?;
    Ok((harmony_messages, token_ids))
}

/// Create streaming GenerateRequest
fn create_generate_request_streaming(
    body: &ResponsesRequest,
    token_ids: Vec<u32>,
) -> Result<GenerateRequest, String> {
    let token_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();

    let mut sampling_params = SamplingParams::default();
    if let Some(temp) = body.temperature {
        sampling_params.temperature = Some(temp);
    }
    if let Some(top_p) = body.top_p {
        sampling_params.top_p = Some(top_p);
    }
    if let Some(max_tokens) = body.max_output_tokens {
        sampling_params.max_new_tokens = Some(max_tokens);
    }

    Ok(GenerateRequest {
        text: None,
        input_ids: Some(InputIds::Single(token_ids_i32)),
        input_embeds: None,
        image_data: None,
        video_data: None,
        audio_data: None,
        sampling_params: Some(sampling_params),
        return_logprob: None,
        logprob_start_len: None,
        top_logprobs_num: None,
        token_ids_logprob: None,
        return_text_in_logprobs: false,
        stream: true, // Enable streaming
        log_metrics: true,
        return_hidden_states: false,
        modalities: None,
        session_params: None,
        lora_path: None,
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        data_parallel_rank: None,
        background: false,
        conversation_id: None,
        priority: None,
        extra_key: None,
        no_logs: false,
        custom_labels: None,
        return_bytes: false,
        return_entropy: false,
        rid: None,
    })
}

/// Process Harmony token stream and emit SSE events
#[allow(clippy::too_many_arguments)]
async fn process_harmony_stream(
    body: Body,
    original_request: Arc<ResponsesRequest>,
    _harmony_messages: Vec<HarmonyMessage>,
    input_token_ids: Vec<u32>, // Input token IDs for usage calculation
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    use futures_util::stream::StreamExt;

    // Create parser for incremental processing
    let encoding = get_harmony_encoding();
    let mut parser = StreamableParser::new(encoding.clone(), Some(Role::Assistant))
        .map_err(|e| format!("Failed to create parser: {}", e))?;

    // Track streaming state for granular events
    let mut processed_message_count = 0;
    let mut all_output_items: Vec<ResponseOutputItem> = Vec::new();
    let mut current_item_id: String = String::new();
    let mut sent_output_item_added = false;
    let mut current_content_index: usize = 0;
    let mut accumulated_text = String::new();
    let mut previous_message_count = 0;

    // Track finish_reason from streaming events
    let mut finish_reason: Option<GenerateFinishReason> = None;

    // Track output token count for usage calculation
    let mut output_token_count: u32 = 0;

    // Convert body to data stream
    let mut stream = body.into_data_stream();

    // Process stream chunks (each chunk may contain SSE events from generate)
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream error: {}", e))?;

        // Parse the SSE event to extract token IDs
        // The generate stream sends JSON events with token_ids field
        let chunk_str = String::from_utf8_lossy(&chunk);

        for line in chunk_str.lines() {
            if line.starts_with("data: ") {
                let data = line.trim_start_matches("data: ").trim();
                if data == "[DONE]" {
                    break;
                }

                // Parse JSON to extract output_ids and meta_info
                if let Ok(json_data) = serde_json::from_str::<serde_json::Value>(data) {
                    // Extract finish_reason from meta_info
                    if let Some(meta_info) = json_data.get("meta_info") {
                        if let Ok(fr) = serde_json::from_value::<GenerateFinishReason>(
                            meta_info
                                .get("finish_reason")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null),
                        ) {
                            finish_reason = Some(fr);
                        }
                    }

                    if let Some(output_ids) = json_data.get("output_ids").and_then(|v| v.as_array())
                    {
                        // Process each token through parser
                        for token_id in output_ids {
                            if let Some(id) = token_id.as_u64() {
                                parser
                                    .process(id as u32)
                                    .map_err(|e| format!("Parser error: {}", e))?;

                                // Track output tokens for usage calculation
                                output_token_count += 1;

                                // Detect message boundary (new message started)
                                let current_messages = parser.messages();
                                if current_messages.len() > previous_message_count {
                                    // Previous message completed - emit done events
                                    if sent_output_item_added && !current_item_id.is_empty() {
                                        let previous_message =
                                            &current_messages[previous_message_count];
                                        let channel =
                                            previous_message.channel.as_deref().unwrap_or("");

                                        if channel == "analysis" {
                                            // Emit reasoning_text.done
                                            let event = json!({
                                                "type": "response.reasoning_text.done",
                                                "item_id": current_item_id,
                                                "text": accumulated_text
                                            });
                                            tx.send(Ok(Bytes::from(format!(
                                                "data: {}\n\n",
                                                event
                                            ))))
                                            .map_err(|_| "Channel closed".to_string())?;
                                        } else if channel == "final" {
                                            // Emit output_text.done
                                            let event = json!({
                                                "type": "response.output_text.done",
                                                "item_id": &current_item_id,
                                                "content_index": current_content_index,
                                                "text": &accumulated_text
                                            });
                                            tx.send(Ok(Bytes::from(format!(
                                                "data: {}\n\n",
                                                event
                                            ))))
                                            .map_err(|_| "Channel closed".to_string())?;

                                            // Emit content_part.done
                                            let event = json!({
                                                "type": "response.content_part.done",
                                                "item_id": &current_item_id,
                                                "content_index": current_content_index,
                                                "part": {
                                                    "type": "output_text",
                                                    "text": &accumulated_text,
                                                    "annotations": []
                                                }
                                            });
                                            tx.send(Ok(Bytes::from(format!(
                                                "data: {}\n\n",
                                                event
                                            ))))
                                            .map_err(|_| "Channel closed".to_string())?;
                                        }

                                        // Emit output_item.done
                                        let completed_item =
                                            parse_output_message(previous_message)?;
                                        let event = json!({
                                            "type": "response.output_item.done",
                                            "item": completed_item
                                        });
                                        tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                                            .map_err(|_| "Channel closed".to_string())?;
                                    }

                                    // New message started - reset state
                                    previous_message_count = current_messages.len();
                                    sent_output_item_added = false;
                                    accumulated_text.clear();
                                }

                                // Check for content delta
                                if let Ok(Some(delta)) = parser.last_content_delta() {
                                    let channel = parser.current_channel();
                                    let channel_str = channel.as_deref().unwrap_or("");

                                    // Emit output_item.added if not sent yet
                                    if !sent_output_item_added {
                                        current_item_id =
                                            format!("item_{}", Uuid::new_v4().simple());
                                        current_content_index = 0;

                                        // Create in-progress item based on channel
                                        let in_progress_item = if channel_str == "analysis" {
                                            ResponseOutputItem::Reasoning {
                                                id: current_item_id.clone(),
                                                summary: vec![],
                                                content: vec![],
                                                status: Some("in_progress".to_string()),
                                            }
                                        } else if channel_str == "final" {
                                            ResponseOutputItem::Message {
                                                id: current_item_id.clone(),
                                                role: "assistant".to_string(),
                                                content: vec![],
                                                status: "in_progress".to_string(),
                                            }
                                        } else {
                                            // Default to message for commentary/other channels
                                            ResponseOutputItem::Message {
                                                id: current_item_id.clone(),
                                                role: "assistant".to_string(),
                                                content: vec![],
                                                status: "in_progress".to_string(),
                                            }
                                        };

                                        // Emit output_item.added
                                        let event = json!({
                                            "type": "response.output_item.added",
                                            "item": in_progress_item
                                        });
                                        tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                                            .map_err(|_| "Channel closed".to_string())?;

                                        // Emit content_part.added or reasoning_part.added
                                        if channel_str == "analysis" {
                                            let event = json!({
                                                "type": "response.reasoning_part.added",
                                                "item_id": &current_item_id,
                                                "content_index": current_content_index,
                                                "part": {
                                                    "type": "reasoning_text",
                                                    "text": ""
                                                }
                                            });
                                            tx.send(Ok(Bytes::from(format!(
                                                "data: {}\n\n",
                                                event
                                            ))))
                                            .map_err(|_| "Channel closed".to_string())?;
                                        } else {
                                            let event = json!({
                                                "type": "response.content_part.added",
                                                "item_id": &current_item_id,
                                                "content_index": current_content_index,
                                                "part": {
                                                    "type": "output_text",
                                                    "text": "",
                                                    "annotations": []
                                                }
                                            });
                                            tx.send(Ok(Bytes::from(format!(
                                                "data: {}\n\n",
                                                event
                                            ))))
                                            .map_err(|_| "Channel closed".to_string())?;
                                        }

                                        sent_output_item_added = true;
                                    }

                                    // Accumulate text for done events
                                    accumulated_text.push_str(&delta);

                                    // Emit delta event
                                    if channel_str == "analysis" {
                                        let event = json!({
                                            "type": "response.reasoning_text.delta",
                                            "item_id": &current_item_id,
                                            "content_index": current_content_index,
                                            "delta": delta
                                        });
                                        tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                                            .map_err(|_| "Channel closed".to_string())?;
                                    } else {
                                        // Extract logprobs from json_data if available and top_logprobs requested
                                        let logprobs_opt =
                                            if original_request.top_logprobs.unwrap_or(0) > 0 {
                                                json_data.get("logprobs").cloned()
                                            } else {
                                                None
                                            };

                                        let mut event = json!({
                                            "type": "response.output_text.delta",
                                            "item_id": &current_item_id,
                                            "content_index": current_content_index,
                                            "delta": delta
                                        });

                                        // Add logprobs if available
                                        if let Some(logprobs) = logprobs_opt {
                                            event["logprobs"] = logprobs;
                                        }

                                        tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                                            .map_err(|_| "Channel closed".to_string())?;
                                    }
                                }
                            }
                        }

                        // Check if all messages are now complete (for final done event)
                        let current_messages = parser.messages();
                        if current_messages.len() > processed_message_count {
                            for message in &current_messages[processed_message_count..] {
                                let output_item = parse_output_message(message)?;
                                all_output_items.push(output_item.clone());
                            }
                            processed_message_count = current_messages.len();
                        }
                    }
                }
            }
        }
    }

    // Handle final message completion if streaming ended
    if sent_output_item_added && !current_item_id.is_empty() {
        let current_messages = parser.messages();
        if !current_messages.is_empty() {
            let last_message = &current_messages[current_messages.len() - 1];
            let channel = last_message.channel.as_deref().unwrap_or("");

            if channel == "analysis" {
                let event = json!({
                    "type": "response.reasoning_text.done",
                    "item_id": current_item_id,
                    "text": accumulated_text
                });
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                    .map_err(|_| "Channel closed".to_string())?;
            } else if channel == "final" {
                let event = json!({
                    "type": "response.output_text.done",
                    "item_id": &current_item_id,
                    "content_index": current_content_index,
                    "text": &accumulated_text
                });
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                    .map_err(|_| "Channel closed".to_string())?;

                let event = json!({
                    "type": "response.content_part.done",
                    "item_id": &current_item_id,
                    "content_index": current_content_index,
                    "part": {
                        "type": "output_text",
                        "text": &accumulated_text,
                        "annotations": []
                    }
                });
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                    .map_err(|_| "Channel closed".to_string())?;
            }

            let completed_item = parse_output_message(last_message)?;
            let event = json!({
                "type": "response.output_item.done",
                "item": completed_item
            });
            tx.send(Ok(Bytes::from(format!("data: {}\n\n", event))))
                .map_err(|_| "Channel closed".to_string())?;
        }
    }

    // Parse remaining state for incomplete generation
    let remaining_items = parse_remaining_state(&parser);
    if !remaining_items.is_empty() {
        debug!(
            "Streaming parser has {} incomplete item(s) in current state",
            remaining_items.len()
        );
        all_output_items.extend(remaining_items);
    }

    // Map finish_reason to status:
    // - "length" → "incomplete" (hit max tokens)
    // - "abort" → "cancelled" (request was aborted)
    // - empty output → "incomplete"
    // - otherwise → "completed"
    let status = if all_output_items.is_empty() {
        ResponseStatus::Incomplete
    } else if let Some(ref fr) = finish_reason {
        match fr {
            GenerateFinishReason::Length { .. } => ResponseStatus::Incomplete,
            GenerateFinishReason::Other(val) => {
                // Check if it's an abort
                if val
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("abort"))
                {
                    ResponseStatus::Cancelled
                } else {
                    ResponseStatus::Completed
                }
            }
            GenerateFinishReason::Stop => ResponseStatus::Completed,
        }
    } else {
        // No finish_reason available, default to Completed
        ResponseStatus::Completed
    };

    // Build final response
    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let final_response = ResponsesResponse {
        id: response_id.clone(),
        object: "response".to_string(),
        created_at: created,
        status, // Use computed status instead of hardcoded "Completed"
        error: None,
        incomplete_details: None,
        instructions: original_request.instructions.clone(),
        max_output_tokens: original_request.max_output_tokens,
        model: original_request
            .model
            .clone(),
        output: all_output_items,
        parallel_tool_calls: original_request.parallel_tool_calls.unwrap_or(true),
        previous_response_id: original_request.previous_response_id.clone(),
        reasoning: None,
        store: original_request.store.unwrap_or(true),
        temperature: original_request.temperature,
        text: None,
        tool_choice: "auto".to_string(),
        tools: Vec::new(),
        top_p: original_request.top_p,
        truncation: None,
        usage: Some(Modern(ResponseUsage {
            input_tokens: input_token_ids.len() as u32,
            output_tokens: output_token_count,
            total_tokens: (input_token_ids.len() as u32) + output_token_count,
            input_tokens_details: None,
            output_tokens_details: None,
        })),
        user: None,
        metadata: original_request.metadata.clone().unwrap_or_default(),
    };

    // Persist final response if store=true
    if original_request.store.unwrap_or(true) {
        if let Ok(response_json) = serde_json::to_value(&final_response) {
            if let Err(e) = persist_conversation_items(
                conversation_storage,
                conversation_item_storage,
                response_storage,
                &response_json,
                &original_request,
            )
            .await
            {
                warn!("Failed to persist streamed Harmony response: {}", e);
            } else {
                debug!("Persisted streamed Harmony response: {}", response_id);
            }
        }
    }

    // Emit final completion event
    let completion_event = json!({
        "type": "response.done",
        "response": final_response
    });
    tx.send(Ok(Bytes::from(format!("data: {}\n\n", completion_event))))
        .map_err(|_| "Channel closed".to_string())?;

    Ok(())
}

// ============================================================================
// Background Execution
// ============================================================================

/// Execute background Harmony responses request
#[allow(clippy::too_many_arguments)]
async fn execute_harmony_background(
    _pipeline: &RequestPipeline,
    _body: Arc<ResponsesRequest>,
    _headers: Option<HeaderMap>,
    _model_id: Option<String>,
    _components: Arc<SharedComponents>,
    _response_storage: SharedResponseStorage,
    _conversation_storage: SharedConversationStorage,
    _conversation_item_storage: SharedConversationItemStorage,
    _background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
) -> Response {
    // TODO: Implement background Harmony execution
    (
        StatusCode::NOT_IMPLEMENTED,
        axum::Json(serde_json::json!({
            "error": {
                "message": "Harmony background execution not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
        .into_response()
}

// ============================================================================
// Harmony Encoding Singleton
// ============================================================================

use std::sync::OnceLock;

/// Global Harmony encoding (loaded once, reused across all requests)
static HARMONY_ENCODING: OnceLock<HarmonyEncoding> = OnceLock::new();

/// Get or initialize the global Harmony encoding
///
/// This is loaded once on first use and reused for all subsequent requests.
/// Uses HarmonyGptOss encoding which supports the gpt-oss model family.
pub fn get_harmony_encoding() -> &'static HarmonyEncoding {
    HARMONY_ENCODING.get_or_init(|| {
        openai_harmony::load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .expect("Failed to load Harmony encoding")
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate unique ID for output items
///
/// Format: `{prefix}_{uuid_without_hyphens}`
/// Examples: "msg_a1b2c3d4", "reasoning_e5f6g7h8", "tool_i9j0k1l2"
fn generate_item_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().simple())
}

// ============================================================================
// Core Routing Logic
// ============================================================================

// ============================================================================
// MCP Tool Execution Helpers
// ============================================================================

/// Maximum iterations for MCP tool execution loop
const MAX_MCP_ITERATIONS: usize = 100;

/// Check if output items contain pending built-in tool calls
///
/// Returns (tool_name, arguments) for the first pending tool call found
fn find_pending_builtin_tool_call(
    output_items: &[ResponseOutputItem],
) -> Option<(String, serde_json::Map<String, serde_json::Value>)> {
    for item in output_items {
        match item {
            ResponseOutputItem::WebSearchCall { action, status, .. } => {
                if status == "pending" || status == "in_progress" {
                    let (tool_name, args) = match action {
                        WebSearchAction::Search { query } => {
                            let mut args = serde_json::Map::new();
                            args.insert("query".to_string(), json!(query));
                            ("search".to_string(), args)
                        }
                        WebSearchAction::OpenPage { url } => {
                            let mut args = serde_json::Map::new();
                            args.insert("url".to_string(), json!(url));
                            ("open".to_string(), args)
                        }
                        WebSearchAction::Find { pattern, url } => {
                            let mut args = serde_json::Map::new();
                            args.insert("pattern".to_string(), json!(pattern));
                            args.insert("url".to_string(), json!(url));
                            ("find".to_string(), args)
                        }
                    };
                    return Some((tool_name, args));
                }
            }
            ResponseOutputItem::CodeInterpreterCall { code, status, .. } => {
                if status == "pending" || status == "in_progress" {
                    let mut args = serde_json::Map::new();
                    args.insert("code".to_string(), json!(code));
                    return Some(("execute".to_string(), args));
                }
            }
            _ => continue,
        }
    }
    None
}

/// Execute MCP tool and convert result to Harmony tool result message
///
/// This calls the MCP manager to execute the tool and converts the result
/// into a Harmony message that can be added to the conversation.
async fn execute_builtin_tool_via_mcp(
    mcp_manager: &McpClientManager,
    tool_name: &str,
    arguments: serde_json::Map<String, serde_json::Value>,
) -> Result<HarmonyMessage, String> {
    // Execute MCP tool
    let result = mcp_manager
        .call_tool(tool_name, Some(arguments))
        .await
        .map_err(|e| format!("MCP tool execution failed: {}", e))?;

    // Convert result to Harmony message
    // The tool result message should have:
    // - author: system
    // - recipient: assistant (returning control to assistant)
    // - channel: commentary
    // - content: tool result as text

    let result_text = result
        .content
        .iter()
        .filter_map(|c| {
            // rmcp Content has as_text() that returns &RawTextContent with a text field
            c.as_text().map(|text_content| text_content.text.clone())
        })
        .collect::<Vec<_>>()
        .join("\n");

    use openai_harmony::chat::{Author, Content, TextContent};

    Ok(HarmonyMessage {
        author: Author {
            role: Role::System,
            name: Some("tool_result".to_string()),
        },
        recipient: Some("assistant".to_string()),
        content: vec![Content::Text(TextContent { text: result_text })],
        channel: Some("commentary".to_string()),
        content_type: None,
    })
}

/// Parse browser tool call (web search) from commentary channel
///
/// Maps browser actions to WebSearchAction variants:
/// - browser.search → Search { query }
/// - browser.open → OpenPage { url }
/// - browser.find → Find { pattern, url }
fn parse_browser_tool_call(recipient: &str, text: &str) -> Result<ResponseOutputItem, String> {
    // Parse arguments as JSON
    let args: serde_json::Map<String, serde_json::Value> = serde_json::from_str(text)
        .map_err(|e| format!("Failed to parse browser tool arguments: {}", e))?;

    // Determine action based on recipient
    let action = if recipient == "browser.search" {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        WebSearchAction::Search { query }
    } else if recipient == "browser.open" {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        WebSearchAction::OpenPage { url }
    } else if recipient == "browser.find" {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        WebSearchAction::Find { pattern, url }
    } else {
        return Err(format!("Unknown browser action: {}", recipient));
    };

    Ok(ResponseOutputItem::WebSearchCall {
        id: generate_item_id("ws"),
        action,
        status: "pending".to_string(), // Will be executed via MCP
        output: None,
    })
}

/// Parse python tool call (code interpreter) from commentary channel
///
/// Python tool calls contain code to execute in the text content.
fn parse_python_tool_call(text: &str) -> Result<ResponseOutputItem, String> {
    Ok(ResponseOutputItem::CodeInterpreterCall {
        id: generate_item_id("ci"),
        code: text.to_string(),
        status: "pending".to_string(), // Will be executed via MCP
        outputs: None,
    })
}

/// Extract text content from a Harmony message
fn extract_text_from_message(message: &HarmonyMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            Content::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Route a Harmony message to a ResponseOutputItem
///
/// This is the core routing function that mirrors Python's parse_output_message().
/// It examines the message's channel field and creates the appropriate output item:
///
/// - "analysis" → ReasoningItem (chain-of-thought reasoning)
/// - "commentary" → FunctionToolCall (tool invocation)
/// - "final" or None → Message (normal assistant text)
///
/// # Arguments
/// * `message` - Harmony message from StreamableParser
///
/// # Returns
/// ResponseOutputItem variant based on channel
pub fn parse_output_message(message: &HarmonyMessage) -> Result<ResponseOutputItem, String> {
    // Only process assistant messages
    if message.author.role != Role::Assistant {
        return Err("Only assistant messages are supported".to_string());
    }

    // Extract text from message
    let text = extract_text_from_message(message);

    // Route based on channel field (Harmony's channel system)
    match message.channel.as_deref() {
        Some("analysis") => {
            // Reasoning channel - chain-of-thought thinking
            Ok(ResponseOutputItem::Reasoning {
                id: generate_item_id("reasoning"),
                summary: vec![], // Harmony doesn't provide summaries
                content: vec![ResponseReasoningContent::ReasoningText { text }],
                status: Some("completed".to_string()),
            })
        }
        Some("commentary") => {
            // Tool call channel - parse recipient for tool type
            // Commentary channel format: message has recipient="tool_type.action"
            // - browser.* → WebSearchCall (built-in)
            // - python → CodeInterpreterCall (built-in)
            // - functions.* → FunctionToolCall (user-defined)

            if let Some(ref recipient) = message.recipient {
                // Specific browser tool calls (browser.search, browser.open, browser.find)
                // Parse tool call from commentary channel
                if recipient.starts_with("browser.") {
                    return parse_browser_tool_call(recipient, &text);
                }

                // Python code interpreter tool call
                if recipient == "python" {
                    return parse_python_tool_call(&text);
                }

                // User-defined function call
                if recipient.starts_with("functions.") {
                    // Extract function name (everything after "functions.")
                    let function_name = recipient
                        .strip_prefix("functions.")
                        .unwrap_or("unknown")
                        .to_string();

                    // Generate unique ID for this tool call
                    let fc_id = generate_item_id("fc");

                    // Create FunctionToolCall item
                    return Ok(ResponseOutputItem::FunctionToolCall {
                        id: fc_id,
                        name: function_name,
                        arguments: text, // The text content is the JSON arguments
                        output: None,
                        status: "completed".to_string(),
                    });
                }

                // Built-in tools (browser, container) in commentary channel
                // Output analysis channel as reasoning items
                // Note: "python" is now handled as CodeInterpreterCall above
                if recipient.starts_with("browser") || recipient.starts_with("container") {
                    return Ok(ResponseOutputItem::Reasoning {
                        id: generate_item_id("rs"),
                        summary: Vec::new(),
                        content: vec![ResponseReasoningContent::ReasoningText { text }],
                        status: None,
                    });
                }
            }

            // If not a recognized tool call, treat as regular message (fallback)
            warn!(
                "Commentary channel message with unrecognized recipient: {:?}",
                message.recipient
            );
            Ok(ResponseOutputItem::Message {
                id: generate_item_id("msg"),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            })
        }
        Some("final") | None => {
            // Normal text channel - assistant message
            Ok(ResponseOutputItem::Message {
                id: generate_item_id("msg"),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            })
        }
        Some(other) => {
            // Unknown channel - treat as normal text with warning
            error!("Unknown Harmony channel: {}", other);
            Ok(ResponseOutputItem::Message {
                id: generate_item_id("msg"),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            })
        }
    }
}

/// Parse all messages from a Harmony parser
///
/// Processes all completed messages from the parser and converts them
/// to ResponseOutputItems. This is used for non-streaming responses
/// or when finalizing a streaming response.
///
/// # Arguments
/// * `parser` - StreamableParser with processed tokens
///
/// # Returns
/// Vector of ResponseOutputItems (in order of completion)
pub fn parse_all_messages(parser: &StreamableParser) -> Result<Vec<ResponseOutputItem>, String> {
    let messages = parser.messages();
    let mut items = Vec::new();

    for message in messages {
        let item = parse_output_message(message)?;
        items.push(item);
    }

    Ok(items)
}

/// Parse remaining state from StreamableParser for incomplete generation
///
/// Parse incomplete messages from the parser state:
/// Handles cases where generation stops mid-message by extracting partial content
/// from the parser's current state.
fn parse_remaining_state(parser: &StreamableParser) -> Vec<ResponseOutputItem> {
    use uuid::Uuid;

    // Get current content - if empty or error, return empty list
    let current_content = match parser.current_content() {
        Ok(content) if !content.is_empty() => content,
        _ => return Vec::new(),
    };

    // Only process assistant role messages
    if parser.current_role() != Some(Role::Assistant) {
        return Vec::new();
    }

    // Skip browser tool calls
    if let Some(recipient) = parser.current_recipient() {
        if recipient.starts_with("browser.") {
            return Vec::new();
        }
    }

    // Handle based on current channel
    match parser.current_channel().as_deref() {
        Some("analysis") => {
            // Create reasoning item for incomplete analysis
            vec![ResponseOutputItem::Reasoning {
                id: format!("rs_{}", Uuid::new_v4().simple()),
                status: None,
                content: vec![ResponseReasoningContent::ReasoningText {
                    text: current_content,
                }],
                summary: Vec::new(),
            }]
        }
        Some("final") => {
            // Create message item for incomplete final message
            vec![ResponseOutputItem::Message {
                id: format!("msg_{}", Uuid::new_v4().simple()),
                role: "assistant".to_string(),
                status: "incomplete".to_string(), // Mark as incomplete
                content: vec![ResponseContentPart::OutputText {
                    text: current_content,
                    annotations: Vec::new(),
                    logprobs: None,
                }],
            }]
        }
        _ => Vec::new(),
    }
}

// ============================================================================
// Conversion Helpers for MCP Loop
// ============================================================================

/// Convert ResponseInputOutputItem to HarmonyMessage
fn item_to_harmony_message(item: &ResponseInputOutputItem) -> Result<HarmonyMessage, String> {
    use openai_harmony::chat::{Author, Content, TextContent};

    match item {
        ResponseInputOutputItem::Message { role, content, .. } => {
            let text = content_parts_to_text(content);
            let (harmony_role, recipient, channel) = match role.as_str() {
                "user" => (
                    Role::User,
                    Some("assistant".to_string()),
                    Some("final".to_string()),
                ),
                "assistant" => (Role::Assistant, None, Some("final".to_string())),
                "system" => (
                    Role::System,
                    Some("assistant".to_string()),
                    Some("final".to_string()),
                ),
                _ => return Err(format!("Unknown role: {}", role)),
            };
            Ok(HarmonyMessage {
                author: Author {
                    role: harmony_role,
                    name: None,
                },
                recipient,
                content: vec![Content::Text(TextContent { text })],
                channel,
                content_type: None,
            })
        }
        _ => Err("Can only convert Message items to Harmony messages".to_string()),
    }
}

/// Convert ResponseOutputItem to HarmonyMessage (for adding to conversation)
fn output_item_to_harmony_message(item: &ResponseOutputItem) -> Option<HarmonyMessage> {
    use openai_harmony::chat::{Author, Content, TextContent};

    match item {
        ResponseOutputItem::Message { content, .. } => {
            let text = content_parts_to_text(content);
            Some(HarmonyMessage {
                author: Author {
                    role: Role::Assistant,
                    name: None,
                },
                recipient: None,
                content: vec![Content::Text(TextContent { text })],
                channel: Some("final".to_string()),
                content_type: None,
            })
        }
        ResponseOutputItem::Reasoning { content, .. } => {
            let text = content
                .iter()
                .map(|c| match c {
                    ResponseReasoningContent::ReasoningText { text } => text.clone(),
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(HarmonyMessage {
                author: Author {
                    role: Role::Assistant,
                    name: None,
                },
                recipient: None,
                content: vec![Content::Text(TextContent { text })],
                channel: Some("analysis".to_string()),
                content_type: None,
            })
        }
        ResponseOutputItem::WebSearchCall { action, .. } => {
            let (recipient, args_json) = match action {
                WebSearchAction::Search { query } => {
                    ("browser.search", json!({"query": query}).to_string())
                }
                WebSearchAction::OpenPage { url } => {
                    ("browser.open", json!({"url": url}).to_string())
                }
                WebSearchAction::Find { pattern, url } => (
                    "browser.find",
                    json!({"pattern": pattern, "url": url}).to_string(),
                ),
            };
            Some(HarmonyMessage {
                author: Author {
                    role: Role::Assistant,
                    name: None,
                },
                recipient: Some(recipient.to_string()),
                content: vec![Content::Text(TextContent { text: args_json })],
                channel: Some("commentary".to_string()),
                content_type: None,
            })
        }
        ResponseOutputItem::CodeInterpreterCall { code, .. } => Some(HarmonyMessage {
            author: Author {
                role: Role::Assistant,
                name: None,
            },
            recipient: Some("python".to_string()),
            content: vec![Content::Text(TextContent { text: code.clone() })],
            channel: Some("commentary".to_string()),
            content_type: None,
        }),
        ResponseOutputItem::FunctionToolCall {
            name, arguments, ..
        } => Some(HarmonyMessage {
            author: Author {
                role: Role::Assistant,
                name: None,
            },
            recipient: Some(format!("functions.{}", name)),
            content: vec![Content::Text(TextContent {
                text: arguments.clone(),
            })],
            channel: Some("commentary".to_string()),
            content_type: None,
        }),
        // Other types don't need to be added to conversation
        _ => None,
    }
}

/// Convert HarmonyMessage to ResponseInputOutputItem
fn harmony_message_to_input_item(message: &HarmonyMessage) -> ResponseInputOutputItem {
    let text = extract_text_from_message(message);
    let role_str = match message.author.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        _ => "assistant", // Default for Developer, Tool, or future roles
    };

    ResponseInputOutputItem::Message {
        id: generate_item_id("msg"),
        role: role_str.to_string(),
        content: vec![ResponseContentPart::OutputText {
            text,
            annotations: vec![],
            logprobs: None,
        }],
        status: None,
    }
}

/// Helper to convert content parts to text
fn content_parts_to_text(parts: &[ResponseContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            ResponseContentPart::InputText { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use openai_harmony::chat::{Author, Content, TextContent};

    use super::*;

    #[test]
    fn test_generate_item_id() {
        let id = generate_item_id("msg");
        assert!(id.starts_with("msg_"));
        assert_eq!(id.len(), 4 + 32); // prefix + underscore + 32 hex chars
    }

    #[test]
    fn test_get_harmony_encoding() {
        // Should not panic and return valid encoding
        let encoding = get_harmony_encoding();
        assert_eq!(encoding.name(), "HarmonyGptOss");
    }

    #[test]
    fn test_parse_output_message_tool_call() {
        // Create a commentary channel message with functions.get_weather recipient
        let message = HarmonyMessage {
            author: Author {
                role: Role::Assistant,
                name: None,
            },
            recipient: Some("functions.get_weather".to_string()),
            content: vec![Content::Text(TextContent {
                text: r#"{"location": "San Francisco", "unit": "celsius"}"#.to_string(),
            })],
            channel: Some("commentary".to_string()),
            content_type: None,
        };

        let result = parse_output_message(&message);
        assert!(result.is_ok());

        match result.unwrap() {
            ResponseOutputItem::FunctionToolCall {
                name,
                arguments,
                status,
                ..
            } => {
                assert_eq!(name, "get_weather");
                assert_eq!(
                    arguments,
                    r#"{"location": "San Francisco", "unit": "celsius"}"#
                );
                assert_eq!(status, "completed");
            }
            other => panic!("Expected FunctionToolCall, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_output_message_reasoning() {
        // Create an analysis channel message
        let message = HarmonyMessage {
            author: Author {
                role: Role::Assistant,
                name: None,
            },
            recipient: None,
            content: vec![Content::Text(TextContent {
                text: "Let me think about this...".to_string(),
            })],
            channel: Some("analysis".to_string()),
            content_type: None,
        };

        let result = parse_output_message(&message);
        assert!(result.is_ok());

        match result.unwrap() {
            ResponseOutputItem::Reasoning { content, .. } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    ResponseReasoningContent::ReasoningText { text } => {
                        assert_eq!(text, "Let me think about this...");
                    }
                }
            }
            other => panic!("Expected Reasoning, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_output_message_final() {
        // Create a final channel message
        let message = HarmonyMessage {
            author: Author {
                role: Role::Assistant,
                name: None,
            },
            recipient: None,
            content: vec![Content::Text(TextContent {
                text: "Here is the answer".to_string(),
            })],
            channel: Some("final".to_string()),
            content_type: None,
        };

        let result = parse_output_message(&message);
        assert!(result.is_ok());

        match result.unwrap() {
            ResponseOutputItem::Message {
                role,
                content,
                status,
                ..
            } => {
                assert_eq!(role, "assistant");
                assert_eq!(status, "completed");
                assert_eq!(content.len(), 1);
                match &content[0] {
                    ResponseContentPart::OutputText { text, .. } => {
                        assert_eq!(text, "Here is the answer");
                    }
                    _ => panic!("Expected OutputText"),
                }
            }
            other => panic!("Expected Message, got {:?}", other),
        }
    }
}
