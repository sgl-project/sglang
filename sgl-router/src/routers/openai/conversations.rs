//! Conversation CRUD operations and persistence

use crate::data_connector::{
    conversation_items::ListParams, conversation_items::SortOrder, Conversation, ConversationId,
    ConversationItemStorage, ConversationStorage, NewConversation, NewConversationItem, ResponseId,
    ResponseStorage, SharedConversationItemStorage, SharedConversationStorage,
};
use crate::protocols::spec::{ResponseInput, ResponsesRequest};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

use super::responses::build_stored_response;

/// Maximum number of properties allowed in conversation metadata
pub(crate) const MAX_METADATA_PROPERTIES: usize = 16;

// ============================================================================
// Conversation CRUD Operations
// ============================================================================

/// Create a new conversation
pub(super) async fn create_conversation(
    conversation_storage: &SharedConversationStorage,
    body: Value,
) -> Response {
    // TODO: The validation should be done in the right place
    let metadata = match body.get("metadata") {
        Some(Value::Object(map)) => {
            if map.len() > MAX_METADATA_PROPERTIES {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": format!(
                            "metadata cannot have more than {} properties",
                            MAX_METADATA_PROPERTIES
                        )
                    })),
                )
                    .into_response();
            }
            Some(map.clone())
        }
        Some(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "metadata must be an object"})),
            )
                .into_response();
        }
        None => None,
    };

    let new_conv = NewConversation { metadata };

    match conversation_storage.create_conversation(new_conv).await {
        Ok(conversation) => {
            info!(conversation_id = %conversation.id.0, "Created conversation");
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to create conversation: {}", e)})),
        )
            .into_response(),
    }
}

/// Get a conversation by ID
pub(super) async fn get_conversation(
    conversation_storage: &SharedConversationStorage,
    conv_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    match conversation_storage
        .get_conversation(&conversation_id)
        .await
    {
        Ok(Some(conversation)) => {
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Conversation not found"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to get conversation: {}", e)})),
        )
            .into_response(),
    }
}

/// Update a conversation's metadata
pub(super) async fn update_conversation(
    conversation_storage: &SharedConversationStorage,
    conv_id: &str,
    body: Value,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    let current_meta = match conversation_storage
        .get_conversation(&conversation_id)
        .await
    {
        Ok(Some(meta)) => meta,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Conversation not found"})),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Failed to get conversation: {}", e)})),
            )
                .into_response();
        }
    };

    #[derive(Debug)]
    enum Patch {
        Set(String, Value),
        Delete(String),
    }

    let mut patches: Vec<Patch> = Vec::new();

    if let Some(metadata_val) = body.get("metadata") {
        if let Some(map) = metadata_val.as_object() {
            for (k, v) in map {
                if v.is_null() {
                    patches.push(Patch::Delete(k.clone()));
                } else {
                    patches.push(Patch::Set(k.clone(), v.clone()));
                }
            }
        } else {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "metadata must be an object"})),
            )
                .into_response();
        }
    }

    let mut new_metadata = current_meta.metadata.clone().unwrap_or_default();
    for patch in patches {
        match patch {
            Patch::Set(k, v) => {
                new_metadata.insert(k, v);
            }
            Patch::Delete(k) => {
                new_metadata.remove(&k);
            }
        }
    }

    if new_metadata.len() > MAX_METADATA_PROPERTIES {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!(
                    "metadata cannot have more than {} properties",
                    MAX_METADATA_PROPERTIES
                )
            })),
        )
            .into_response();
    }

    let final_metadata = if new_metadata.is_empty() {
        None
    } else {
        Some(new_metadata)
    };

    match conversation_storage
        .update_conversation(&conversation_id, final_metadata)
        .await
    {
        Ok(Some(conversation)) => {
            info!(conversation_id = %conversation_id.0, "Updated conversation");
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Conversation not found"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to update conversation: {}", e)})),
        )
            .into_response(),
    }
}

/// Delete a conversation
pub(super) async fn delete_conversation(
    conversation_storage: &SharedConversationStorage,
    conv_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    match conversation_storage
        .get_conversation(&conversation_id)
        .await
    {
        Ok(Some(_)) => {}
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Conversation not found"})),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Failed to get conversation: {}", e)})),
            )
                .into_response();
        }
    }

    match conversation_storage
        .delete_conversation(&conversation_id)
        .await
    {
        Ok(_) => {
            info!(conversation_id = %conversation_id.0, "Deleted conversation");
            (
                StatusCode::OK,
                Json(json!({
                    "id": conversation_id.0,
                    "object": "conversation.deleted",
                    "deleted": true
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to delete conversation: {}", e)})),
        )
            .into_response(),
    }
}

/// List items in a conversation with pagination
pub(super) async fn list_conversation_items(
    conversation_storage: &SharedConversationStorage,
    item_storage: &SharedConversationItemStorage,
    conv_id: &str,
    query_params: HashMap<String, String>,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    match conversation_storage
        .get_conversation(&conversation_id)
        .await
    {
        Ok(Some(_)) => {}
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Conversation not found"})),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Failed to get conversation: {}", e)})),
            )
                .into_response();
        }
    }

    let limit: usize = query_params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let after = query_params.get("after").map(|s| s.to_string());

    // Default to descending order (most recent first)
    let order = query_params
        .get("order")
        .and_then(|s| match s.as_str() {
            "asc" => Some(SortOrder::Asc),
            "desc" => Some(SortOrder::Desc),
            _ => None,
        })
        .unwrap_or(SortOrder::Desc);

    let params = ListParams {
        limit,
        order,
        after,
    };

    match item_storage.list_items(&conversation_id, params).await {
        Ok(items) => {
            let item_values: Vec<Value> = items
                .iter()
                .map(|item| {
                    let mut obj = serde_json::Map::new();
                    obj.insert("id".to_string(), json!(item.id.0));
                    obj.insert("type".to_string(), json!(item.item_type));
                    obj.insert("created_at".to_string(), json!(item.created_at));

                    obj.insert("content".to_string(), item.content.clone());
                    if let Some(status) = &item.status {
                        obj.insert("status".to_string(), json!(status));
                    }

                    Value::Object(obj)
                })
                .collect();

            let has_more = items.len() == limit;
            let last_id = items.last().map(|item| item.id.0.clone());

            (
                StatusCode::OK,
                Json(json!({
                    "object": "list",
                    "data": item_values,
                    "has_more": has_more,
                    "first_id": items.first().map(|item| &item.id.0),
                    "last_id": last_id,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to list items: {}", e)})),
        )
            .into_response(),
    }
}

// ============================================================================
// Persistence Operations
// ============================================================================

/// Persist conversation items (delegates to persist_items_with_storages)
pub(super) async fn persist_conversation_items(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> Result<(), String> {
    persist_items_with_storages(
        conversation_storage,
        item_storage,
        response_storage,
        response_json,
        original_body,
    )
    .await
}

/// Helper function to create and link a conversation item (two-step API)
async fn create_and_link_item(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &ConversationId,
    mut new_item: NewConversationItem,
) -> Result<(), String> {
    // Set default status if not provided
    if new_item.status.is_none() {
        new_item.status = Some("completed".to_string());
    }

    // Step 1: Create the item
    let created = item_storage
        .create_item(new_item)
        .await
        .map_err(|e| format!("Failed to create item: {}", e))?;

    // Step 2: Link it to the conversation
    item_storage
        .link_item(conv_id, &created.id, Utc::now())
        .await
        .map_err(|e| format!("Failed to link item: {}", e))?;

    info!(
        conversation_id = %conv_id.0,
        item_id = %created.id.0,
        item_type = %created.item_type,
        "Persisted conversation item and link"
    );

    Ok(())
}

/// Persist conversation items with all storages
async fn persist_items_with_storages(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> Result<(), String> {
    let conv_id = match &original_body.conversation {
        Some(id) => ConversationId::from(id.as_str()),
        None => return Ok(()),
    };

    if conversation_storage
        .get_conversation(&conv_id)
        .await
        .map_err(|e| format!("Failed to get conversation: {}", e))?
        .is_none()
    {
        warn!(conversation_id = %conv_id.0, "Conversation not found, skipping item persistence");
        return Ok(());
    }

    let response_id_str = response_json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Response missing id field".to_string())?;
    let response_id = ResponseId::from(response_id_str);

    let response_id_opt = Some(response_id_str.to_string());

    // Persist input items
    match &original_body.input {
        ResponseInput::Text(text) => {
            let new_item = NewConversationItem {
                id: None, // Let storage generate ID
                response_id: response_id_opt.clone(),
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([{ "type": "input_text", "text": text }]),
                status: Some("completed".to_string()),
            };
            create_and_link_item(&item_storage, &conv_id, new_item).await?;
        }
        ResponseInput::Items(items_array) => {
            for input_item in items_array {
                match input_item {
                    crate::protocols::spec::ResponseInputOutputItem::Message {
                        role,
                        content,
                        status,
                        ..
                    } => {
                        let content_v = serde_json::to_value(content)
                            .map_err(|e| format!("Failed to serialize content: {}", e))?;
                        let new_item = NewConversationItem {
                            id: None,
                            response_id: response_id_opt.clone(),
                            item_type: "message".to_string(),
                            role: Some(role.clone()),
                            content: content_v,
                            status: status.clone(),
                        };
                        create_and_link_item(&item_storage, &conv_id, new_item).await?;
                    }
                    _ => {
                        // For other types (FunctionToolCall, etc.), serialize the whole item
                        let item_val = serde_json::to_value(input_item)
                            .map_err(|e| format!("Failed to serialize item: {}", e))?;
                        let new_item = NewConversationItem {
                            id: None,
                            response_id: response_id_opt.clone(),
                            item_type: "unknown".to_string(),
                            role: None,
                            content: item_val,
                            status: Some("completed".to_string()),
                        };
                        create_and_link_item(&item_storage, &conv_id, new_item).await?;
                    }
                }
            }
        }
    }

    // Persist output items
    if let Some(output_arr) = response_json.get("output").and_then(|v| v.as_array()) {
        for output_item in output_arr {
            if let Some(obj) = output_item.as_object() {
                let item_type = obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("message");

                let role = obj.get("role").and_then(|v| v.as_str()).map(String::from);
                let status = obj.get("status").and_then(|v| v.as_str()).map(String::from);

                let content = if item_type == "message" {
                    obj.get("content").cloned().unwrap_or(json!([]))
                } else if item_type == "function_call" || item_type == "function_tool_call" {
                    json!({
                        "type": "function_call",
                        "name": obj.get("name"),
                        "call_id": obj.get("call_id").or_else(|| obj.get("id")),
                        "arguments": obj.get("arguments")
                    })
                } else if item_type == "function_call_output" {
                    json!({
                        "type": "function_call_output",
                        "call_id": obj.get("call_id"),
                        "output": obj.get("output")
                    })
                } else {
                    output_item.clone()
                };

                let new_item = NewConversationItem {
                    id: None,
                    response_id: response_id_opt.clone(),
                    item_type: item_type.to_string(),
                    role,
                    content,
                    status,
                };
                create_and_link_item(&item_storage, &conv_id, new_item).await?;
            }
        }
    }

    // Store the full response using the shared helper
    let mut stored_response = build_stored_response(response_json, original_body);
    stored_response.id = response_id;
    let final_response_id = stored_response.id.clone();

    response_storage
        .store_response(stored_response)
        .await
        .map_err(|e| format!("Failed to store response in conversation: {}", e))?;

    info!(conversation_id = %conv_id.0, response_id = %final_response_id.0, "Persisted conversation items and response");

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert conversation to JSON response
fn conversation_to_json(conversation: &Conversation) -> Value {
    let mut response = json!({
        "id": conversation.id.0,
        "object": "conversation",
        "created_at": conversation.created_at.timestamp()
    });

    if let Some(metadata) = &conversation.metadata {
        if !metadata.is_empty() {
            if let Some(obj) = response.as_object_mut() {
                obj.insert("metadata".to_string(), Value::Object(metadata.clone()));
            }
        }
    }

    response
}
