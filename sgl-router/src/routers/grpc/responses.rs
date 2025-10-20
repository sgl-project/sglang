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
    protocols::responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseStatus,
        ResponsesRequest, ResponsesResponse,
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
            None, // No grpc_request_id for sync
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
    grpc_request_id: Option<String>,
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
        grpc_request_id,
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
    grpc_request_id: Option<String>,
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

    // 2. Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(&modified_request)
        .map_err(|e| format!("Failed to convert request: {}", e))?;

    // 3. Execute chat pipeline
    let chat_response = pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            components,
            grpc_request_id,
            response_id.clone(),
            background_tasks,
        )
        .await
        .map_err(|e| format!("Pipeline execution failed: {}", e))?;

    // 4. Convert ChatCompletionResponse → ResponsesResponse
    let responses_response = conversions::chat_to_responses(&chat_response, &request)
        .map_err(|e| format!("Failed to convert to responses format: {}", e))?;

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
    // Generate both response_id and grpc_request_id for Option 2
    let response_id = format!("resp_{}", Uuid::new_v4());
    let grpc_request_id = format!("chatcmpl-{}", Uuid::new_v4());

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
    let grpc_request_id_clone = grpc_request_id.clone();
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
            Some(grpc_request_id_clone),
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
            grpc_request_id: grpc_request_id.clone(),
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

    // 3. Execute chat pipeline - returns streaming SSE Response
    let chat_stream_response = pipeline
        .execute_chat(chat_request, headers, model_id, components)
        .await;

    // TODO: Wrap the streaming response to convert ChatCompletionChunk SSE → ResponsesResponse SSE
    // For now, return the chat stream directly (it's valid SSE, just wrong format)
    warn!("Streaming responses not fully implemented yet - returning chat stream format");
    chat_stream_response
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
