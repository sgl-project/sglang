//! Conversation operations for OpenAI router
//!
//! Re-exports shared CRUD handlers and provides OpenAI-specific persistence logic.

use std::sync::Arc;

use serde_json::{json, Value};
use tracing::{info, warn};

use super::responses::build_stored_response;
// Re-export shared conversation handlers for backward compatibility
pub use crate::routers::conversations::{
    conversation_to_json, create_and_link_item, create_conversation, create_conversation_items,
    delete_conversation, delete_conversation_item, get_conversation, get_conversation_item,
    item_to_json, list_conversation_items, update_conversation, MAX_METADATA_PROPERTIES,
};
use crate::{
    data_connector::{
        ConversationId, ConversationItemId, ConversationItemStorage, ConversationStorage,
        NewConversationItem, ResponseId, ResponseStorage,
    },
    protocols::responses::{generate_id, ResponseInput, ResponsesRequest},
};

// ============================================================================
// Persistence Operations (OpenAI-specific)
// ============================================================================

/// Persist conversation items (delegates to persist_items_with_storages)
pub async fn persist_conversation_items(
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

/// Persist conversation items with all storages
///
/// This function:
/// 1. Extracts and normalizes input items from the request
/// 2. Extracts output items from the response
/// 3. Stores ALL items in response storage (always)
/// 4. If conversation provided, also links items to conversation
async fn persist_items_with_storages(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> Result<(), String> {
    // Step 1: Extract response ID
    let response_id_str = response_json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Response missing id field".to_string())?;
    let response_id = ResponseId::from(response_id_str);

    // Step 2: Parse and normalize input items from request
    let input_items = extract_input_items(&original_body.input)?;

    // Step 3: Parse output items from response
    let output_items = extract_output_items(response_json)?;

    // Step 4: Build StoredResponse with input and output as JSON arrays
    let mut stored_response = build_stored_response(response_json, original_body);
    stored_response.id = response_id.clone();
    stored_response.input = Value::Array(input_items.clone());
    stored_response.output = Value::Array(output_items.clone());

    // Step 5: Store response (ALWAYS, regardless of conversation)
    response_storage
        .store_response(stored_response)
        .await
        .map_err(|e| format!("Failed to store response: {}", e))?;

    // Step 6: Check if conversation is provided and validate it
    let conv_id_opt = match &original_body.conversation {
        Some(id) => {
            let conv_id = ConversationId::from(id.as_str());
            // Verify conversation exists
            if conversation_storage
                .get_conversation(&conv_id)
                .await
                .map_err(|e| format!("Failed to get conversation: {}", e))?
                .is_none()
            {
                warn!(conversation_id = %conv_id.0, "Conversation not found, skipping item linking");
                None // Conversation doesn't exist, items already stored in response
            } else {
                Some(conv_id)
            }
        }
        None => None, // No conversation provided, items already stored in response
    };

    // Step 7: If conversation exists, link items to it
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

/// Extract and normalize input items from ResponseInput
fn extract_input_items(input: &ResponseInput) -> Result<Vec<Value>, String> {
    use crate::protocols::responses::{ResponseInputOutputItem, StringOrContentParts};

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
                            // For other item types (Message, Reasoning, FunctionToolCall, FunctionCallOutput), serialize and ensure ID
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

/// Extract ALL output items from response JSON
fn extract_output_items(response_json: &Value) -> Result<Vec<Value>, String> {
    response_json
        .get("output")
        .and_then(|v| v.as_array())
        .cloned()
        .ok_or_else(|| "No output array in response".to_string())
}

/// Link ALL input and output items to a conversation
async fn link_items_to_conversation(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &ConversationId,
    input_items: &[Value],
    output_items: &[Value],
    response_id: &str,
) -> Result<(), String> {
    let response_id_opt = Some(response_id.to_string());

    // Link ALL input items (no filtering by type)
    for input_item_value in input_items {
        let item_type = input_item_value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");
        let role = input_item_value
            .get("role")
            .and_then(|v| v.as_str())
            .map(String::from);

        // For function_call and function_call_output, store the entire item as content
        // For message types, extract just the content field
        let content = if item_type == "function_call" || item_type == "function_call_output" {
            input_item_value.clone()
        } else {
            input_item_value
                .get("content")
                .cloned()
                .unwrap_or(json!([]))
        };

        let status = input_item_value
            .get("status")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Extract the original item ID from input if present
        let item_id = input_item_value
            .get("id")
            .and_then(|v| v.as_str())
            .map(ConversationItemId::from);

        let new_item = NewConversationItem {
            id: item_id, // Preserve ID if present
            response_id: response_id_opt.clone(),
            item_type: item_type.to_string(),
            role,
            content,
            status,
        };

        create_and_link_item(item_storage, Some(conv_id), new_item).await?;
    }

    // Link ALL output items (no filtering by type)
    // Store reasoning, function_tool_call, mcp_call, and any other types
    for output_item_value in output_items {
        let item_type = output_item_value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");
        let role = output_item_value
            .get("role")
            .and_then(|v| v.as_str())
            .map(String::from);
        let status = output_item_value
            .get("status")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Extract the original item ID from the response
        let item_id = output_item_value
            .get("id")
            .and_then(|v| v.as_str())
            .map(ConversationItemId::from);

        // For non-message types, store the entire item as content
        // For message types, extract just the content field
        let content = if item_type == "message" {
            output_item_value
                .get("content")
                .cloned()
                .unwrap_or(json!([]))
        } else {
            // For other types (reasoning, function_call, function_call_output, mcp_call, etc.)
            // store the entire item structure
            output_item_value.clone()
        };

        let new_item = NewConversationItem {
            id: item_id, // Preserve ID if present
            response_id: response_id_opt.clone(),
            item_type: item_type.to_string(),
            role,
            content,
            status,
        };

        create_and_link_item(item_storage, Some(conv_id), new_item).await?;
    }

    Ok(())
}
