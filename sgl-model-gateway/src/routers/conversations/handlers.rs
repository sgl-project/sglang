//! Conversation CRUD handlers - shared across routers

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::{
    data_connector::{
        Conversation, ConversationId, ConversationItem, ConversationItemId,
        ConversationItemStorage, ConversationStorage, ListParams, NewConversation,
        NewConversationItem, ResponseId, ResponseStorage, SortOrder, StoredResponse,
    },
    protocols::responses::{
        generate_id, ResponseInput, ResponseInputOutputItem, ResponsesRequest, StringOrContentParts,
    },
};

// ============================================================================
// Constants
// ============================================================================

pub const MAX_METADATA_PROPERTIES: usize = 16;
const MAX_ITEMS_PER_REQUEST: usize = 20;

const SUPPORTED_ITEM_TYPES: &[&str] = &[
    "message",
    "reasoning",
    "mcp_list_tools",
    "mcp_call",
    "item_reference",
    "function_call",
    "function_call_output",
    "file_search_call",
    "computer_call",
    "computer_call_output",
    "web_search_call",
    "image_generation_call",
    "code_interpreter_call",
    "local_shell_call",
    "local_shell_call_output",
    "mcp_approval_request",
    "mcp_approval_response",
    "custom_tool_call",
    "custom_tool_call_output",
];

const IMPLEMENTED_ITEM_TYPES: &[&str] = &[
    "message",
    "reasoning",
    "mcp_list_tools",
    "mcp_call",
    "item_reference",
];

// ============================================================================
// Error Response Helpers
// ============================================================================

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": message.into()})),
    )
        .into_response()
}

fn not_found(message: impl Into<String>) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": message.into()})),
    )
        .into_response()
}

fn internal_error(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": message.into()})),
    )
        .into_response()
}

fn bad_request_structured(error_obj: Value) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": error_obj}))).into_response()
}

// ============================================================================
// Storage Helpers
// ============================================================================

async fn ensure_conversation_exists(
    storage: &Arc<dyn ConversationStorage>,
    conv_id: &ConversationId,
) -> Result<Conversation, Response> {
    match storage.get_conversation(conv_id).await {
        Ok(Some(conv)) => Ok(conv),
        Ok(None) => Err(not_found("Conversation not found")),
        Err(e) => Err(internal_error(format!("Failed to get conversation: {e}"))),
    }
}

// ============================================================================
// Metadata Operations
// ============================================================================

fn validate_metadata(value: &Value) -> Result<Option<serde_json::Map<String, Value>>, String> {
    match value.get("metadata") {
        Some(Value::Object(map)) => {
            if map.len() > MAX_METADATA_PROPERTIES {
                Err(format!(
                    "metadata cannot have more than {MAX_METADATA_PROPERTIES} properties"
                ))
            } else {
                Ok(Some(map.clone()))
            }
        }
        Some(_) => Err("metadata must be an object".to_string()),
        None => Ok(None),
    }
}

fn apply_metadata_patches(
    current: Option<serde_json::Map<String, Value>>,
    body: &Value,
) -> Result<Option<serde_json::Map<String, Value>>, String> {
    let patch_map = match body.get("metadata") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("metadata must be an object".to_string()),
        None => return Ok(current),
    };

    let mut result = current.unwrap_or_default();
    for (k, v) in patch_map {
        if v.is_null() {
            result.remove(k);
        } else {
            result.insert(k.clone(), v.clone());
        }
    }

    if result.len() > MAX_METADATA_PROPERTIES {
        return Err(format!(
            "metadata cannot have more than {MAX_METADATA_PROPERTIES} properties"
        ));
    }

    Ok(if result.is_empty() {
        None
    } else {
        Some(result)
    })
}

// ============================================================================
// Conversation CRUD Handlers
// ============================================================================

pub async fn create_conversation(storage: &Arc<dyn ConversationStorage>, body: Value) -> Response {
    let metadata = match validate_metadata(&body) {
        Ok(m) => m,
        Err(msg) => return bad_request(msg),
    };

    let new_conv = NewConversation { id: None, metadata };

    match storage.create_conversation(new_conv).await {
        Ok(conversation) => {
            info!(conversation_id = %conversation.id.0, "Created conversation");
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(e) => internal_error(format!("Failed to create conversation: {e}")),
    }
}

pub async fn get_conversation(storage: &Arc<dyn ConversationStorage>, conv_id: &str) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    match storage.get_conversation(&conversation_id).await {
        Ok(Some(conversation)) => {
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => not_found("Conversation not found"),
        Err(e) => internal_error(format!("Failed to get conversation: {e}")),
    }
}

pub async fn update_conversation(
    storage: &Arc<dyn ConversationStorage>,
    conv_id: &str,
    body: Value,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    let current = match ensure_conversation_exists(storage, &conversation_id).await {
        Ok(c) => c,
        Err(response) => return response,
    };

    let final_metadata = match apply_metadata_patches(current.metadata.clone(), &body) {
        Ok(m) => m,
        Err(msg) => return bad_request(msg),
    };

    match storage
        .update_conversation(&conversation_id, final_metadata)
        .await
    {
        Ok(Some(conversation)) => {
            info!(conversation_id = %conversation_id.0, "Updated conversation");
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => not_found("Conversation not found"),
        Err(e) => internal_error(format!("Failed to update conversation: {e}")),
    }
}

pub async fn delete_conversation(
    storage: &Arc<dyn ConversationStorage>,
    conv_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) = ensure_conversation_exists(storage, &conversation_id).await {
        return response;
    }

    match storage.delete_conversation(&conversation_id).await {
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
        Err(e) => internal_error(format!("Failed to delete conversation: {e}")),
    }
}

// ============================================================================
// Conversation Item Handlers
// ============================================================================

pub async fn list_conversation_items(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    limit: Option<usize>,
    order: Option<&str>,
    after: Option<&str>,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) = ensure_conversation_exists(conversation_storage, &conversation_id).await
    {
        return response;
    }

    let limit = limit.unwrap_or(100);
    let order = match order {
        Some("asc") => SortOrder::Asc,
        _ => SortOrder::Desc,
    };

    let params = ListParams {
        limit,
        order,
        after: after.map(String::from),
    };

    match item_storage.list_items(&conversation_id, params).await {
        Ok(items) => {
            let item_values: Vec<Value> = items
                .iter()
                .map(|item| {
                    let mut item_json = item_to_json(item);
                    if let Some(obj) = item_json.as_object_mut() {
                        obj.insert("created_at".to_string(), json!(item.created_at));
                    }
                    item_json
                })
                .collect();

            (
                StatusCode::OK,
                Json(json!({
                    "object": "list",
                    "data": item_values,
                    "has_more": items.len() == limit,
                    "first_id": items.first().map(|item| &item.id.0),
                    "last_id": items.last().map(|item| &item.id.0),
                })),
            )
                .into_response()
        }
        Err(e) => internal_error(format!("Failed to list items: {e}")),
    }
}

pub async fn create_conversation_items(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    body: Value,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) = ensure_conversation_exists(conversation_storage, &conversation_id).await
    {
        return response;
    }

    let items_array = match body.get("items").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return bad_request("Missing or invalid 'items' field"),
    };

    if items_array.len() > MAX_ITEMS_PER_REQUEST {
        return bad_request(format!(
            "Cannot add more than {MAX_ITEMS_PER_REQUEST} items at a time"
        ));
    }

    let mut created_items = Vec::new();
    let mut warnings = Vec::new();
    let added_at = Utc::now();

    for item_val in items_array {
        match process_item(item_storage, &conversation_id, item_val, added_at).await {
            Ok((item_json, warning)) => {
                created_items.push(item_json);
                if let Some(w) = warning {
                    warnings.push(w);
                }
            }
            Err(response) => return response,
        }
    }

    let mut response = json!({
        "object": "list",
        "data": created_items,
        "first_id": created_items.first().and_then(|v| v.get("id")),
        "last_id": created_items.last().and_then(|v| v.get("id")),
        "has_more": false
    });

    if !warnings.is_empty() {
        if let Some(obj) = response.as_object_mut() {
            obj.insert("warnings".to_string(), json!(warnings));
        }
    }

    (StatusCode::OK, Json(response)).into_response()
}

/// Process a single item for creation/linking
async fn process_item(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: &ConversationId,
    item_val: &Value,
    added_at: chrono::DateTime<Utc>,
) -> Result<(Value, Option<String>), Response> {
    let item_type = item_val
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    // Handle item_reference specially - just link existing item
    if item_type == "item_reference" {
        return process_item_reference(item_storage, conversation_id, item_val, added_at).await;
    }

    let user_provided_id = item_val.get("id").and_then(|v| v.as_str());

    let (item, warning) = if let Some(id_str) = user_provided_id {
        process_item_with_id(item_storage, conversation_id, item_val, id_str).await?
    } else {
        process_new_item(item_storage, item_val).await?
    };

    // Link item to conversation
    if let Err(e) = item_storage
        .link_item(conversation_id, &item.id, added_at)
        .await
    {
        warn!("Failed to link item {}: {}", item.id.0, e);
    }

    Ok((item_to_json(&item), warning))
}

/// Process an item_reference - link an existing item to the conversation
async fn process_item_reference(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: &ConversationId,
    item_val: &Value,
    added_at: chrono::DateTime<Utc>,
) -> Result<(Value, Option<String>), Response> {
    let ref_id = item_val
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| bad_request("item_reference requires 'id' field"))?;

    let item_id = ConversationItemId::from(ref_id);

    let existing_item = match item_storage.get_item(&item_id).await {
        Ok(Some(item)) => item,
        Ok(None) => return Err(not_found(format!("Referenced item '{ref_id}' not found"))),
        Err(e) => {
            return Err(internal_error(format!(
                "Failed to get referenced item: {e}"
            )))
        }
    };

    if let Err(e) = item_storage
        .link_item(conversation_id, &existing_item.id, added_at)
        .await
    {
        warn!("Failed to link item {}: {}", existing_item.id.0, e);
    }

    Ok((item_to_json(&existing_item), None))
}

/// Process an item with a user-provided ID
async fn process_item_with_id(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: &ConversationId,
    item_val: &Value,
    id_str: &str,
) -> Result<(ConversationItem, Option<String>), Response> {
    let item_id = ConversationItemId::from(id_str);

    // Check if already linked
    let is_linked = item_storage
        .is_item_linked(conversation_id, &item_id)
        .await
        .map_err(|e| internal_error(format!("Failed to check item link: {e}")))?;

    if is_linked {
        return Err(bad_request_structured(json!({
            "message": "Item already in conversation",
            "type": "invalid_request_error",
            "param": "items",
            "code": "item_already_in_conversation"
        })));
    }

    // Check if item exists globally
    match item_storage.get_item(&item_id).await {
        Ok(Some(existing)) => Ok((existing, None)),
        Ok(None) => {
            // Create new item with the provided ID
            let (mut new_item, warning) = parse_item_from_value(item_val).map_err(bad_request)?;
            new_item.id = Some(item_id);

            let created = item_storage
                .create_item(new_item)
                .await
                .map_err(|e| internal_error(format!("Failed to create item: {e}")))?;

            Ok((created, warning))
        }
        Err(e) => Err(internal_error(format!(
            "Failed to check item existence: {e}"
        ))),
    }
}

/// Process a new item without a user-provided ID
async fn process_new_item(
    item_storage: &Arc<dyn ConversationItemStorage>,
    item_val: &Value,
) -> Result<(ConversationItem, Option<String>), Response> {
    let (new_item, warning) = parse_item_from_value(item_val).map_err(bad_request)?;

    let created = item_storage
        .create_item(new_item)
        .await
        .map_err(|e| internal_error(format!("Failed to create item: {e}")))?;

    Ok((created, warning))
}

pub async fn get_conversation_item(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    item_id: &str,
    _include: Option<Vec<String>>,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);
    let item_id = ConversationItemId::from(item_id);

    if let Err(response) = ensure_conversation_exists(conversation_storage, &conversation_id).await
    {
        return response;
    }

    let is_linked = match item_storage
        .is_item_linked(&conversation_id, &item_id)
        .await
    {
        Ok(linked) => linked,
        Err(e) => return internal_error(format!("Failed to check item link: {e}")),
    };

    if !is_linked {
        return not_found("Item not found in this conversation");
    }

    match item_storage.get_item(&item_id).await {
        Ok(Some(item)) => (StatusCode::OK, Json(item_to_json(&item))).into_response(),
        Ok(None) => not_found("Item not found"),
        Err(e) => internal_error(format!("Failed to get item: {e}")),
    }
}

pub async fn delete_conversation_item(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    item_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);
    let item_id = ConversationItemId::from(item_id);

    let conversation =
        match ensure_conversation_exists(conversation_storage, &conversation_id).await {
            Ok(conv) => conv,
            Err(response) => return response,
        };

    match item_storage.delete_item(&conversation_id, &item_id).await {
        Ok(_) => {
            info!(
                conversation_id = %conversation_id.0,
                item_id = %item_id.0,
                "Deleted conversation item"
            );
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(e) => internal_error(format!("Failed to delete item: {e}")),
    }
}

// ============================================================================
// Item Creation Helper
// ============================================================================

pub async fn create_and_link_item(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id_opt: Option<&ConversationId>,
    mut new_item: NewConversationItem,
) -> Result<(), String> {
    if new_item.status.is_none() {
        new_item.status = Some("completed".to_string());
    }

    let created = item_storage
        .create_item(new_item)
        .await
        .map_err(|e| format!("Failed to create item: {e}"))?;

    if let Some(conv_id) = conv_id_opt {
        item_storage
            .link_item(conv_id, &created.id, Utc::now())
            .await
            .map_err(|e| format!("Failed to link item: {e}"))?;

        debug!(
            conversation_id = %conv_id.0,
            item_id = %created.id.0,
            item_type = %created.item_type,
            "Persisted conversation item and link"
        );
    } else {
        debug!(
            item_id = %created.id.0,
            item_type = %created.item_type,
            "Persisted conversation item (no conversation link)"
        );
    }

    Ok(())
}

// ============================================================================
// Parsing and Serialization
// ============================================================================

fn parse_item_from_value(
    item_val: &Value,
) -> Result<(NewConversationItem, Option<String>), String> {
    let item_type = item_val
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    if !SUPPORTED_ITEM_TYPES.contains(&item_type) {
        return Err(format!(
            "Unsupported item type '{}'. Supported types: {}",
            item_type,
            SUPPORTED_ITEM_TYPES.join(", ")
        ));
    }

    let warning = if !IMPLEMENTED_ITEM_TYPES.contains(&item_type) {
        Some(format!(
            "Item type '{}' is accepted but not yet implemented. \
             The item will be stored but may not function as expected.",
            item_type
        ))
    } else {
        None
    };

    let role = item_val
        .get("role")
        .and_then(|v| v.as_str())
        .map(String::from);

    if item_type == "message" && role.is_none() {
        return Err("Message items require 'role' field".to_string());
    }

    let status = item_val
        .get("status")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| Some("completed".to_string()));

    let content = if item_type == "message" || item_type == "reasoning" {
        item_val.get("content").cloned().unwrap_or(json!([]))
    } else {
        item_val.clone()
    };

    Ok((
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: item_type.to_string(),
            role,
            content,
            status,
        },
        warning,
    ))
}

/// Field mappings for item types that store data in content
const ITEM_TYPE_FIELDS: &[(&str, &[&str])] = &[
    (
        "mcp_call",
        &[
            "name",
            "arguments",
            "output",
            "server_label",
            "approval_request_id",
            "error",
        ],
    ),
    ("mcp_list_tools", &["tools", "server_label"]),
    ("function_call", &["call_id", "name", "arguments", "output"]),
    ("function_call_output", &["call_id", "output"]),
];

pub fn item_to_json(item: &ConversationItem) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("id".to_string(), json!(item.id.0));
    obj.insert("type".to_string(), json!(item.item_type));

    if let Some(role) = &item.role {
        obj.insert("role".to_string(), json!(role));
    }

    // Find field mappings for this item type
    let fields = ITEM_TYPE_FIELDS
        .iter()
        .find(|(t, _)| *t == item.item_type)
        .map(|(_, fields)| *fields);

    if let Some(fields) = fields {
        // Extract specific fields from content
        if let Some(content_obj) = item.content.as_object() {
            for field in fields {
                if let Some(value) = content_obj.get(*field) {
                    obj.insert((*field).to_string(), value.clone());
                }
            }
        }
    } else {
        // Default: include content as-is
        obj.insert("content".to_string(), item.content.clone());
    }

    if let Some(status) = &item.status {
        obj.insert("status".to_string(), json!(status));
    }

    Value::Object(obj)
}

pub fn conversation_to_json(conversation: &Conversation) -> Value {
    let mut obj = json!({
        "id": conversation.id.0,
        "object": "conversation",
        "created_at": conversation.created_at.timestamp()
    });

    if let Some(metadata) = &conversation.metadata {
        if !metadata.is_empty() {
            obj["metadata"] = Value::Object(metadata.clone());
        }
    }

    obj
}

// ============================================================================
// Response Persistence
// ============================================================================

/// Extract a string field from JSON, returning owned String
fn get_string(json: &Value, key: &str) -> Option<String> {
    json.get(key).and_then(|v| v.as_str()).map(String::from)
}

/// Build a StoredResponse from response JSON and original request
pub fn build_stored_response(
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> StoredResponse {
    let mut stored = StoredResponse::new(None);

    // Initialize empty arrays - will be populated by persist_conversation_items
    stored.input = Value::Array(vec![]);
    stored.output = Value::Array(vec![]);

    stored.instructions =
        get_string(response_json, "instructions").or_else(|| original_body.instructions.clone());

    stored.model = get_string(response_json, "model").or_else(|| Some(original_body.model.clone()));

    stored.safety_identifier = original_body.user.clone();
    stored.conversation_id = original_body.conversation.clone();

    stored.metadata = response_json
        .get("metadata")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_else(|| original_body.metadata.clone().unwrap_or_default());

    stored.previous_response_id = get_string(response_json, "previous_response_id")
        .map(|s| ResponseId::from(s.as_str()))
        .or_else(|| {
            original_body
                .previous_response_id
                .as_deref()
                .map(ResponseId::from)
        });

    if let Some(id_str) = get_string(response_json, "id") {
        stored.id = ResponseId::from(id_str.as_str());
    }

    stored.raw_response = response_json.clone();
    stored
}

/// Extract and normalize input items from ResponseInput
fn extract_input_items(input: &ResponseInput) -> Result<Vec<Value>, String> {
    let items = match input {
        ResponseInput::Text(text) => {
            // Convert simple text to message item
            vec![json!({
                "id": generate_id("msg"),
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
                "status": "completed"
            })]
        }
        ResponseInput::Items(items) => {
            // Process all item types and ensure IDs
            items
                .iter()
                .map(|item| {
                    match item {
                        ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                            // Convert SimpleInputMessage to standard message format with ID
                            let content_json = match content {
                                StringOrContentParts::String(s) => {
                                    json!([{"type": "input_text", "text": s}])
                                }
                                StringOrContentParts::Array(parts) => serde_json::to_value(parts)
                                    .map_err(|e| {
                                    format!("Failed to serialize content: {}", e)
                                })?,
                            };

                            Ok(json!({
                                "id": generate_id("msg"),
                                "type": "message",
                                "role": role,
                                "content": content_json,
                                "status": "completed"
                            }))
                        }
                        _ => {
                            // For other item types, serialize and ensure ID
                            let mut value = serde_json::to_value(item)
                                .map_err(|e| format!("Failed to serialize item: {}", e))?;

                            // Ensure ID exists - generate if missing
                            if let Some(obj) = value.as_object_mut() {
                                if !obj.contains_key("id")
                                    || obj
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.is_empty())
                                        .unwrap_or(true)
                                {
                                    // Generate ID with appropriate prefix based on type
                                    let item_type =
                                        obj.get("type").and_then(|v| v.as_str()).unwrap_or("item");
                                    let prefix = match item_type {
                                        "function_call" | "function_call_output" => "fc",
                                        "message" => "msg",
                                        _ => "item",
                                    };
                                    obj.insert("id".to_string(), json!(generate_id(prefix)));
                                }
                            }

                            Ok(value)
                        }
                    }
                })
                .collect::<Result<Vec<_>, String>>()?
        }
    };

    Ok(items)
}

/// Convert a JSON item to NewConversationItem
///
/// For input items: function_call/function_call_output store whole item as content
/// For output items: message extracts content field, others store whole item
fn item_to_new_conversation_item(
    item_value: &Value,
    response_id: Option<String>,
    is_input: bool,
) -> NewConversationItem {
    let item_type = item_value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    // Determine if we should store the whole item or just the content field
    let store_whole_item = if is_input {
        item_type == "function_call" || item_type == "function_call_output"
    } else {
        item_type != "message"
    };

    let content = if store_whole_item {
        item_value.clone()
    } else {
        item_value.get("content").cloned().unwrap_or(json!([]))
    };

    NewConversationItem {
        id: item_value
            .get("id")
            .and_then(|v| v.as_str())
            .map(ConversationItemId::from),
        response_id,
        item_type: item_type.to_string(),
        role: item_value
            .get("role")
            .and_then(|v| v.as_str())
            .map(String::from),
        content,
        status: item_value
            .get("status")
            .and_then(|v| v.as_str())
            .map(String::from),
    }
}

/// Link all input and output items to a conversation
async fn link_items_to_conversation(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &ConversationId,
    input_items: &[Value],
    output_items: &[Value],
    response_id: &str,
) -> Result<(), String> {
    let response_id_opt = Some(response_id.to_string());

    for item in input_items {
        let new_item = item_to_new_conversation_item(item, response_id_opt.clone(), true);
        create_and_link_item(item_storage, Some(conv_id), new_item).await?;
    }

    for item in output_items {
        let new_item = item_to_new_conversation_item(item, response_id_opt.clone(), false);
        create_and_link_item(item_storage, Some(conv_id), new_item).await?;
    }

    Ok(())
}

/// Persist conversation items to storage
///
/// This function:
/// 1. Extracts and normalizes input items from the request
/// 2. Extracts output items from the response
/// 3. Stores ALL items in response storage (always)
/// 4. If conversation provided, also links items to conversation
pub async fn persist_conversation_items(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> Result<(), String> {
    // Extract response ID
    let response_id_str = response_json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Response missing id field".to_string())?;
    let response_id = ResponseId::from(response_id_str);

    // Parse and normalize input items from request
    let input_items = extract_input_items(&original_body.input)?;

    // Parse output items from response
    let output_items = response_json
        .get("output")
        .and_then(|v| v.as_array())
        .cloned()
        .ok_or_else(|| "No output array in response".to_string())?;

    // Build and store response
    let mut stored_response = build_stored_response(response_json, original_body);
    stored_response.id = response_id.clone();
    stored_response.input = Value::Array(input_items.clone());
    stored_response.output = Value::Array(output_items.clone());

    response_storage
        .store_response(stored_response)
        .await
        .map_err(|e| format!("Failed to store response: {}", e))?;

    // Check if conversation is provided and validate it exists
    let conv_id_opt = if let Some(id) = &original_body.conversation {
        let conv_id = ConversationId::from(id.as_str());
        match conversation_storage.get_conversation(&conv_id).await {
            Ok(Some(_)) => Some(conv_id),
            Ok(None) => {
                warn!(conversation_id = %conv_id.0, "Conversation not found, skipping item linking");
                None
            }
            Err(e) => return Err(format!("Failed to get conversation: {}", e)),
        }
    } else {
        None
    };

    // If conversation exists, link items to it
    if let Some(conv_id) = conv_id_opt {
        link_items_to_conversation(
            &item_storage,
            &conv_id,
            &input_items,
            &output_items,
            response_id_str,
        )
        .await?;
        info!(
            conversation_id = %conv_id.0,
            response_id = %response_id.0,
            input_count = input_items.len(),
            output_count = output_items.len(),
            "Persisted response and linked items to conversation"
        );
    } else {
        info!(
            response_id = %response_id.0,
            input_count = input_items.len(),
            output_count = output_items.len(),
            "Persisted response without conversation linking"
        );
    }

    Ok(())
}
