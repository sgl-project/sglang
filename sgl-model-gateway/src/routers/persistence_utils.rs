//! Utilities for persisting responses and conversation items across router implementations.

use std::sync::Arc;

use chrono::Utc;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::{
    data_connector::{
        ConversationId, ConversationItem, ConversationItemId, ConversationItemStorage,
        ConversationStorage, NewConversationItem, ResponseId, ResponseStorage, StoredResponse,
    },
    protocols::responses::{
        generate_id, ResponseInput, ResponseInputOutputItem, ResponsesRequest, StringOrContentParts,
    },
};

// ============================================================================
// Constants
// ============================================================================

/// Field mappings for item types that store data in content
pub const ITEM_TYPE_FIELDS: &[(&str, &[&str])] = &[
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

// ============================================================================
// JSON Serialization
// ============================================================================

/// Convert a ConversationItem to JSON, extracting specified fields based on item type
/// or including content as-is for standard message types.
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

// ============================================================================
// Item Creation Helper
// ============================================================================

/// Create a conversation item and optionally link it to a conversation.
/// Sets default "completed" status if not provided.
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
