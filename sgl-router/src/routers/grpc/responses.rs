//! gRPC Router `/v1/responses` endpoint implementation
//!
//! This module handles all responses-specific logic including:
//! - Request validation
//! - Conversation history and response chain loading
//! - Background mode execution
//! - Streaming support
//! - MCP tool loop wrapper (future)
//! - Response persistence

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    http::{self, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::json;
use tokio::sync::RwLock;
use tracing::{debug, warn};
use uuid::Uuid;

use super::{
    context::SharedComponents,
    conversions,
    pipeline::RequestPipeline,
    router::BackgroundTaskInfo,
};
use crate::{
    data_connector::{
        ConversationId, ResponseId, SharedConversationItemStorage, SharedConversationStorage,
        SharedResponseStorage,
    },
    mcp::McpClientManager,
    protocols::{
        chat::ChatCompletionResponse,
        responses::{
            ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseStatus,
            ResponseTool, ResponseToolType, ResponsesRequest, ResponsesResponse,
        },
    },
};

// ============================================================================
// Main Request Handler
// ============================================================================

/// Main handler for POST /v1/responses
///
/// Validates request, determines execution mode (sync/async/streaming), and delegates
pub async fn route_responses(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    background_tasks: Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>,
) -> Response {
    // 1. Validate mutually exclusive parameters
    if request.previous_response_id.is_some() && request.conversation.is_some() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": {
                    "message": "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
                    "type": "invalid_request_error",
                    "param": serde_json::Value::Null,
                    "code": "mutually_exclusive_parameters"
                }
            })),
        )
            .into_response();
    }

    // 2. Check for incompatible parameter combinations
    let is_streaming = request.stream.unwrap_or(false);
    let is_background = request.background.unwrap_or(false);

    if is_streaming && is_background {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": {
                    "message": "Cannot use streaming with background mode. Please set either 'stream' or 'background' to false.",
                    "type": "invalid_request_error",
                    "param": serde_json::Value::Null,
                    "code": "incompatible_parameters"
                }
            })),
        )
            .into_response();
    }

    // 3. Route based on execution mode
    if is_streaming {
        route_responses_streaming(
            pipeline,
            request,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
        )
        .await
    } else if is_background {
        route_responses_background(
            pipeline,
            request,
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
        route_responses_sync(
            pipeline,
            request,
            headers,
            model_id,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            None, // No response_id for sync
            None, // No background_tasks for sync
        )
        .await
    }
}

// ============================================================================
// Synchronous Execution
// ============================================================================

/// Execute synchronous responses request
///
/// This is the core execution path that:
/// 1. Loads conversation history / response chain
/// 2. Converts to ChatCompletionRequest
/// 3. Executes chat pipeline
/// 4. Converts back to ResponsesResponse
/// 5. Persists to storage
async fn route_responses_sync(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Response {
    match route_responses_internal(
        pipeline,
        request,
        headers,
        model_id,
        components,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        response_id,
        background_tasks,
    )
    .await
    {
        Ok(responses_response) => axum::Json(responses_response).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({
                "error": {
                    "message": e,
                    "type": "internal_error"
                }
            })),
        )
            .into_response(),
    }
}

/// Internal implementation that returns Result for background task compatibility
async fn route_responses_internal(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Result<ResponsesResponse, String> {
    // 1. Load conversation history and build modified request
    let modified_request = load_conversation_history(
        &request,
        &response_storage,
        &conversation_storage,
        &conversation_item_storage,
    )
    .await?;

    // 2. Check if request has MCP tools - if so, use tool loop
    let responses_response = if let Some(tools) = &request.tools {
        if let Some(mcp_manager) = create_mcp_manager_from_request(tools).await {
            debug!("MCP tools detected, using tool loop");

            // Execute with MCP tool loop
            execute_tool_loop(
                pipeline,
                modified_request,
                &request,
                headers,
                model_id,
                components,
                mcp_manager,
                response_id.clone(),
                background_tasks,
            )
            .await?
        } else {
            // No MCP manager, execute normally
            execute_without_mcp(
                pipeline,
                &modified_request,
                &request,
                headers,
                model_id,
                components,
                response_id.clone(),
                background_tasks,
            )
            .await?
        }
    } else {
        // No tools, execute normally
        execute_without_mcp(
            pipeline,
            &modified_request,
            &request,
            headers,
            model_id,
            components,
            response_id.clone(),
            background_tasks,
        )
        .await?
    };

    // 5. Persist response to storage if store=true
    if request.store.unwrap_or(true) {
        if let Ok(response_json) = serde_json::to_value(&responses_response) {
            if let Err(e) = crate::routers::openai::conversations::persist_conversation_items(
                conversation_storage,
                conversation_item_storage,
                response_storage,
                &response_json,
                &request,
            )
            .await
            {
                warn!("Failed to persist response: {}", e);
            }
        }
    }

    Ok(responses_response)
}

// ============================================================================
// Background Mode Execution
// ============================================================================

/// Execute responses request in background mode
async fn route_responses_background(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    background_tasks: Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>,
) -> Response {
    // Generate response_id for background tracking
    let response_id = format!("resp_{}", Uuid::new_v4());

    // Get current timestamp
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Create queued response
    let queued_response = ResponsesResponse {
        id: response_id.clone(),
        object: "response".to_string(),
        created_at,
        status: ResponseStatus::Queued,
        error: None,
        incomplete_details: None,
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        model: request
            .model
            .clone()
            .unwrap_or_else(|| "default".to_string()),
        output: Vec::new(),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        previous_response_id: request.previous_response_id.clone(),
        reasoning: None,
        store: request.store.unwrap_or(true),
        temperature: request.temperature,
        text: None,
        tool_choice: "auto".to_string(),
        tools: request.tools.clone().unwrap_or_default(),
        top_p: request.top_p,
        truncation: None,
        usage: None,
        user: request.user.clone(),
        metadata: request.metadata.clone().unwrap_or_default(),
    };

    // Persist queued response to storage
    if let Ok(response_json) = serde_json::to_value(&queued_response) {
        if let Err(e) = crate::routers::openai::conversations::persist_conversation_items(
            conversation_storage.clone(),
            conversation_item_storage.clone(),
            response_storage.clone(),
            &response_json,
            &request,
        )
        .await
        {
            warn!("Failed to persist queued response: {}", e);
        }
    }

    // Spawn background task
    let pipeline = pipeline.clone();
    let request_clone = request.clone();
    let headers_clone = headers.clone();
    let model_id_clone = model_id.clone();
    let components_clone = components.clone();
    let response_storage_clone = response_storage.clone();
    let conversation_storage_clone = conversation_storage.clone();
    let conversation_item_storage_clone = conversation_item_storage.clone();
    let response_id_clone = response_id.clone();
    let background_tasks_clone = background_tasks.clone();

    let handle = tokio::task::spawn(async move {
        // Execute synchronously (set background=false to prevent recursion)
        let mut background_request = (*request_clone).clone();
        background_request.background = Some(false);

        match route_responses_internal(
            &pipeline,
            Arc::new(background_request),
            headers_clone,
            model_id_clone,
            components_clone,
            response_storage_clone,
            conversation_storage_clone,
            conversation_item_storage_clone,
            Some(response_id_clone.clone()),
            Some(background_tasks_clone.clone()),
        )
        .await
        {
            Ok(_) => {
                debug!(
                    "Background response {} completed successfully",
                    response_id_clone
                );
            }
            Err(e) => {
                warn!("Background response {} failed: {}", response_id_clone, e);
            }
        }

        // Clean up task handle when done
        background_tasks_clone
            .write()
            .await
            .remove(&response_id_clone);
    });

    // Store task info for cancellation support
    background_tasks.write().await.insert(
        response_id.clone(),
        BackgroundTaskInfo {
            handle,
            grpc_request_id: String::new(), // Will be populated by pipeline at DispatchMetadataStage
            client: Arc::new(RwLock::new(None)),
        },
    );

    // Return queued response immediately
    axum::Json(queued_response).into_response()
}

// ============================================================================
// Streaming Mode Execution
// ============================================================================

/// Execute streaming responses request
async fn route_responses_streaming(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
) -> Response {
    // 1. Load conversation history
    let modified_request = match load_conversation_history(
        &request,
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

    // 2. Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = match conversions::responses_to_chat(&modified_request) {
        Ok(req) => Arc::new(req),
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(json!({
                    "error": {
                        "message": format!("Failed to convert request: {}", e),
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 3. Execute chat pipeline and convert streaming format
    convert_chat_stream_to_responses_stream(
        pipeline,
        chat_request,
        headers,
        model_id,
        components,
        &request,
        response_storage,
        conversation_storage,
        conversation_item_storage,
    )
    .await
}

/// Convert chat streaming response to responses streaming format
///
/// NOTE: Currently returns chat SSE format directly (functional passthrough).
/// Full responses-formatted SSE transformation is deferred because:
/// 1. OpenAI has not published a streaming spec for /v1/responses endpoint
/// 2. Chat SSE format is functionally compatible with most clients
/// 3. Complex body stream transformation requires additional dependencies
///
/// Future enhancement when spec is available:
/// - Intercept and parse chat SSE events
/// - Transform to ResponsesResponse delta format
/// - Accumulate and persist final response
async fn convert_chat_stream_to_responses_stream(
    pipeline: &RequestPipeline,
    chat_request: Arc<crate::protocols::chat::ChatCompletionRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    _original_request: &ResponsesRequest,
    _response_storage: SharedResponseStorage,
    _conversation_storage: SharedConversationStorage,
    _conversation_item_storage: SharedConversationItemStorage,
) -> Response {
    debug!("Streaming responses (using chat SSE format - responses streaming spec not yet defined)");

    // Return chat streaming response directly (functional passthrough)
    pipeline
        .execute_chat(chat_request, headers, model_id, components)
        .await
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Load conversation history and response chains, returning modified request
async fn load_conversation_history(
    request: &ResponsesRequest,
    response_storage: &SharedResponseStorage,
    conversation_storage: &SharedConversationStorage,
    conversation_item_storage: &SharedConversationItemStorage,
) -> Result<ResponsesRequest, String> {
    let mut modified_request = request.clone();
    let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;

    // Handle previous_response_id by loading response chain
    if let Some(ref prev_id_str) = modified_request.previous_response_id {
        let prev_id = ResponseId::from(prev_id_str.as_str());
        match response_storage.get_response_chain(&prev_id, None).await {
            Ok(chain) => {
                let mut items = Vec::new();
                for stored in chain.responses.iter() {
                    // Convert input to conversation item
                    items.push(ResponseInputOutputItem::Message {
                        id: format!("msg_u_{}", stored.id.0.trim_start_matches("resp_")),
                        role: "user".to_string(),
                        content: vec![ResponseContentPart::InputText {
                            text: stored.input.clone(),
                        }],
                        status: Some("completed".to_string()),
                    });

                    // Convert output to conversation items
                    if let Some(output_arr) =
                        stored.raw_response.get("output").and_then(|v| v.as_array())
                    {
                        for item in output_arr {
                            if let Ok(output_item) =
                                serde_json::from_value::<ResponseInputOutputItem>(item.clone())
                            {
                                items.push(output_item);
                            }
                        }
                    }
                }
                conversation_items = Some(items);
                modified_request.previous_response_id = None;
            }
            Err(e) => {
                warn!(
                    "Failed to load previous response chain for {}: {}",
                    prev_id_str, e
                );
            }
        }
    }

    // Handle conversation by loading conversation history
    if let Some(ref conv_id_str) = request.conversation {
        let conv_id = ConversationId::from(conv_id_str.as_str());

        // Verify conversation exists
        if let Ok(None) = conversation_storage.get_conversation(&conv_id).await {
            return Err("Conversation not found".to_string());
        }

        // Load conversation history
        const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;
        let params = crate::data_connector::conversation_items::ListParams {
            limit: MAX_CONVERSATION_HISTORY_ITEMS,
            order: crate::data_connector::conversation_items::SortOrder::Asc,
            after: None,
        };

        match conversation_item_storage.list_items(&conv_id, params).await {
            Ok(stored_items) => {
                let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                for item in stored_items.into_iter() {
                    if item.item_type == "message" {
                        if let Ok(content_parts) =
                            serde_json::from_value::<Vec<ResponseContentPart>>(item.content.clone())
                        {
                            items.push(ResponseInputOutputItem::Message {
                                id: item.id.0.clone(),
                                role: item.role.clone().unwrap_or_else(|| "user".to_string()),
                                content: content_parts,
                                status: item.status.clone(),
                            });
                        }
                    }
                }

                // Append current request
                match &modified_request.input {
                    ResponseInput::Text(text) => {
                        items.push(ResponseInputOutputItem::Message {
                            id: format!("msg_u_{}", conv_id.0),
                            role: "user".to_string(),
                            content: vec![ResponseContentPart::InputText {
                                text: text.clone(),
                            }],
                            status: Some("completed".to_string()),
                        });
                    }
                    ResponseInput::Items(current_items) => {
                        items.extend_from_slice(current_items);
                    }
                }

                modified_request.input = ResponseInput::Items(items);
            }
            Err(e) => {
                warn!("Failed to load conversation history: {}", e);
            }
        }
    }

    // If we have conversation_items from previous_response_id, merge them
    if let Some(mut items) = conversation_items {
        // Append current request
        match &modified_request.input {
            ResponseInput::Text(text) => {
                items.push(ResponseInputOutputItem::Message {
                    id: format!(
                        "msg_u_{}",
                        request
                            .previous_response_id
                            .as_ref()
                            .unwrap_or(&"new".to_string())
                    ),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: text.clone(),
                    }],
                    status: Some("completed".to_string()),
                });
            }
            ResponseInput::Items(current_items) => {
                items.extend_from_slice(current_items);
            }
        }

        modified_request.input = ResponseInput::Items(items);
    }

    Ok(modified_request)
}

// ============================================================================
// MCP Tool Support
// ============================================================================

/// Build a request-scoped MCP manager from request tools, if present
async fn create_mcp_manager_from_request(
    tools: &[ResponseTool],
) -> Option<Arc<McpClientManager>> {
    let tool = tools
        .iter()
        .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())?;

    let server_url = tool.server_url.as_ref()?.trim().to_string();
    if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
        warn!("Ignoring MCP server_url with unsupported scheme: {}", server_url);
        return None;
    }

    let name = tool
        .server_label
        .clone()
        .unwrap_or_else(|| "request-mcp".to_string());
    let token = tool.authorization.clone();

    let transport = if server_url.contains("/sse") {
        crate::mcp::McpTransport::Sse {
            url: server_url,
            token,
        }
    } else {
        crate::mcp::McpTransport::Streamable {
            url: server_url,
            token,
        }
    };

    let cfg = crate::mcp::McpConfig {
        servers: vec![crate::mcp::McpServerConfig { name, transport }],
    };

    match McpClientManager::new(cfg).await {
        Ok(mgr) => Some(Arc::new(mgr)),
        Err(err) => {
            warn!("Failed to initialize request-scoped MCP manager: {}", err);
            None
        }
    }
}

/// Extract function call from a chat completion response
/// Returns (call_id, tool_name, arguments_json_str) if found
fn extract_function_call_from_chat(response: &ChatCompletionResponse) -> Option<(String, String, String)> {
    // Check if response has choices with tool calls
    let choice = response.choices.first()?;
    let message = &choice.message;

    // Look for tool_calls in the message
    if let Some(tool_calls) = &message.tool_calls {
        if let Some(tool_call) = tool_calls.first() {
            return Some((
                tool_call.id.clone(),
                tool_call.function.name.clone(),
                tool_call.function.arguments.clone().unwrap_or_else(|| "{}".to_string()),
            ));
        }
    }

    None
}

/// Execute an MCP tool call
async fn execute_mcp_call(
    mcp_mgr: &Arc<McpClientManager>,
    tool_name: &str,
    args_json_str: &str,
) -> Result<String, String> {
    let args_value: serde_json::Value =
        serde_json::from_str(args_json_str).map_err(|e| format!("parse tool args: {}", e))?;
    let args_obj = args_value.as_object().cloned();

    let _server_name = mcp_mgr
        .get_tool(tool_name)
        .map(|t| t.server)
        .ok_or_else(|| format!("tool not found: {}", tool_name))?;

    let result = mcp_mgr
        .call_tool(tool_name, args_obj)
        .await
        .map_err(|e| format!("tool call failed: {}", e))?;

    let output_str = serde_json::to_string(&result)
        .map_err(|e| format!("Failed to serialize tool result: {}", e))?;
    Ok(output_str)
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
        self.conversation_history.push(ResponseInputOutputItem::FunctionToolCall {
            id: call_id.clone(),
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

use crate::protocols::responses::{McpToolInfo, ResponseOutputItem};

/// Generate unique ID for MCP items
fn generate_mcp_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

/// Build mcp_list_tools output item
fn build_mcp_list_tools_item(
    mcp: &Arc<McpClientManager>,
    server_label: &str,
) -> ResponseOutputItem {
    let tools = mcp.list_tools();
    let tools_info: Vec<McpToolInfo> = tools
        .iter()
        .map(|t| McpToolInfo {
            name: t.name.clone(),
            description: Some(t.description.clone()),
            input_schema: t.parameters.clone().unwrap_or_else(|| {
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })
            }),
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

/// Execute request without MCP tool loop (simple pipeline execution)
async fn execute_without_mcp(
    pipeline: &RequestPipeline,
    modified_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Result<ResponsesResponse, String> {
    // Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(modified_request)
        .map_err(|e| format!("Failed to convert request: {}", e))?;

    // Execute chat pipeline
    let chat_response = pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            components,
            response_id,
            background_tasks,
        )
        .await
        .map_err(|e| format!("Pipeline execution failed: {}", e))?;

    // Convert ChatCompletionResponse → ResponsesResponse
    conversions::chat_to_responses(&chat_response, original_request)
        .map_err(|e| format!("Failed to convert to responses format: {}", e))
}

/// Execute the MCP tool calling loop
///
/// This wraps pipeline.execute_chat_for_responses() in a loop that:
/// 1. Executes the chat pipeline
/// 2. Checks if response has tool calls
/// 3. If yes, executes MCP tools and builds resume request
/// 4. Repeats until no more tool calls or limit reached
async fn execute_tool_loop(
    pipeline: &RequestPipeline,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    mcp_manager: Arc<McpClientManager>,
    response_id: Option<String>,
    background_tasks: Option<Arc<RwLock<std::collections::HashMap<String, BackgroundTaskInfo>>>>,
) -> Result<ResponsesResponse, String> {
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

    loop {
        // Convert to chat request
        let chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {}", e))?;

        // Execute chat pipeline
        let chat_response = pipeline
            .execute_chat_for_responses(
                Arc::new(chat_request),
                headers.clone(),
                model_id.clone(),
                components.clone(),
                response_id.clone(),
                background_tasks.clone(),
            )
            .await
            .map_err(|e| format!("Pipeline execution failed: {}", e))?;

        // Check for function calls
        if let Some((call_id, tool_name, args_json_str)) = extract_function_call_from_chat(&chat_response) {
            state.iteration += 1;
            state.total_calls += 1;

            debug!(
                "Tool loop iteration {}: calling {} (call_id: {})",
                state.iteration, tool_name, call_id
            );

            // Check combined limit
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(MAX_ITERATIONS),
                None => MAX_ITERATIONS,
            };

            if state.total_calls > effective_limit {
                warn!(
                    "Reached tool call limit: {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls, max_tool_calls, MAX_ITERATIONS
                );

                // Convert chat response to responses format and mark as incomplete
                let mut responses_response = conversions::chat_to_responses(&chat_response, original_request)
                    .map_err(|e| format!("Failed to convert to responses format: {}", e))?;

                // Mark as completed but with incomplete details
                responses_response.status = ResponseStatus::Completed;
                responses_response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                return Ok(responses_response);
            }

            // Execute the MCP tool
            let (output_str, success, error) = match execute_mcp_call(&mcp_manager, &tool_name, &args_json_str).await {
                Ok(output) => (output, true, None),
                Err(err) => {
                    warn!("Tool execution failed: {}", err);
                    let error_msg = err.clone();
                    // Return error as output, let model decide how to proceed
                    let error_json = json!({ "error": err }).to_string();
                    (error_json, false, Some(error_msg))
                }
            };

            // Record the call in state (includes MCP metadata)
            state.record_call(call_id, tool_name, args_json_str, output_str, success, error);

            // Build resume request with conversation history
            // Start with original input
            let mut input_items = match &state.original_input {
                ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
                    id: format!("msg_u_{}", state.iteration),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                }],
                ResponseInput::Items(items) => items.clone(),
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
            let mut responses_response = conversions::chat_to_responses(&chat_response, original_request)
                .map_err(|e| format!("Failed to convert to responses format: {}", e))?;

            // Inject MCP metadata into output
            if state.total_calls > 0 {
                // Prepend mcp_list_tools item
                let mcp_list_tools = build_mcp_list_tools_item(&mcp_manager, &server_label);
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
