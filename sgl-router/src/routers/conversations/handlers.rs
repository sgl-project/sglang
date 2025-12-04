//! Conversation CRUD handlers - shared across routers

use std::{collections::HashMap, sync::Arc};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::data_connector::{
    Conversation, ConversationId, ConversationItem, ConversationItemId, ConversationItemStorage,
    ConversationStorage, ListParams, NewConversation, NewConversationItem, SortOrder,
};

pub const MAX_METADATA_PROPERTIES: usize = 16;

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

pub async fn create_conversation(
    conversation_storage: &Arc<dyn ConversationStorage>,
    body: Value,
) -> Response {
    let metadata = match body.get("metadata") {
        Some(Value::Object(map)) => {
            if map.len() > MAX_METADATA_PROPERTIES {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error":
                            format!(
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

    let new_conv = NewConversation { id: None, metadata };

    match conversation_storage.create_conversation(new_conv).await {
        Ok(conversation) => {
            info!(conversation_id = %conversation.id.0, "Created conversation");
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": format!("Failed to create conversation: {}", e)
            })),
        )
            .into_response(),
    }
}

pub async fn get_conversation(
    conversation_storage: &Arc<dyn ConversationStorage>,
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
            Json(json!({
                "error": format!("Failed to get conversation: {}", e)
            })),
        )
            .into_response(),
    }
}

pub async fn update_conversation(
    conversation_storage: &Arc<dyn ConversationStorage>,
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
                Json(json!({
                    "error": format!("Failed to get conversation: {}", e)
                })),
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
                "error":
                    format!(
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
            Json(json!({
                "error": format!("Failed to update conversation: {}", e)
            })),
        )
            .into_response(),
    }
}

pub async fn delete_conversation(
    conversation_storage: &Arc<dyn ConversationStorage>,
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
                Json(json!({
                    "error": format!("Failed to get conversation: {}", e)
                })),
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
            Json(json!({
                "error": format!("Failed to delete conversation: {}", e)
            })),
        )
            .into_response(),
    }
}

pub async fn list_conversation_items(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
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
                Json(json!({
                    "error": format!("Failed to get conversation: {}", e)
                })),
            )
                .into_response();
        }
    }

    let limit: usize = query_params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let after = query_params.get("after").map(|s| s.to_string());

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
                    let mut item_json = item_to_json(item);
                    if let Some(obj) = item_json.as_object_mut() {
                        obj.insert("created_at".to_string(), json!(item.created_at));
                    }
                    item_json
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
            Json(json!({ "error": format!("Failed to list items: {}", e) })),
        )
            .into_response(),
    }
}

pub async fn create_conversation_items(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    body: Value,
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
                Json(json!({
                    "error": format!("Failed to get conversation: {}", e)
                })),
            )
                .into_response();
        }
    }

    let items_array = match body.get("items").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Missing or invalid 'items' field"})),
            )
                .into_response();
        }
    };

    if items_array.len() > 20 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Cannot add more than 20 items at a time"})),
        )
            .into_response();
    }

    let mut created_items = Vec::new();
    let mut warnings = Vec::new();
    let added_at = Utc::now();

    for item_val in items_array {
        let item_type = item_val
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        if item_type == "item_reference" {
            let ref_id = match item_val.get("id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"error": "item_reference requires 'id' field"})),
                    )
                        .into_response();
                }
            };

            let existing_item_id = ConversationItemId::from(ref_id);

            let existing_item = match item_storage.get_item(&existing_item_id).await {
                Ok(Some(item)) => item,
                Ok(None) => {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(json!({
                            "error": format!("Referenced item '{}' not found", ref_id)
                        })),
                    )
                        .into_response();
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({
                            "error": format!("Failed to get referenced item: {}", e)
                        })),
                    )
                        .into_response();
                }
            };

            if let Err(e) = item_storage
                .link_item(&conversation_id, &existing_item.id, added_at)
                .await
            {
                warn!("Failed to link item {}: {}", existing_item.id.0, e);
            }

            created_items.push(item_to_json(&existing_item));
            continue;
        }

        let user_provided_id = item_val.get("id").and_then(|v| v.as_str());

        let item = if let Some(id_str) = user_provided_id {
            let item_id = ConversationItemId::from(id_str);

            let is_already_linked = match item_storage
                .is_item_linked(&conversation_id, &item_id)
                .await
            {
                Ok(linked) => linked,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({
                            "error": format!("Failed to check item link: {}", e)
                        })),
                    )
                        .into_response();
                }
            };

            if is_already_linked {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": "Item already in conversation",
                            "type": "invalid_request_error",
                            "param": "items",
                            "code": "item_already_in_conversation"
                        }
                    })),
                )
                    .into_response();
            }

            let existing_item = match item_storage.get_item(&item_id).await {
                Ok(Some(item)) => item,
                Ok(None) => {
                    let (new_item, warning) = match parse_item_from_value(item_val) {
                        Ok((mut item, warn)) => {
                            item.id = Some(item_id.clone());
                            (item, warn)
                        }
                        Err(e) => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(json!({ "error": format!("Invalid item: {}", e) })),
                            )
                                .into_response();
                        }
                    };

                    if let Some(w) = warning {
                        warnings.push(w);
                    }

                    match item_storage.create_item(new_item).await {
                        Ok(item) => item,
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": format!("Failed to create item: {}", e) })),
                            )
                                .into_response();
                        }
                    }
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({
                            "error": format!("Failed to check item existence: {}", e)
                        })),
                    )
                        .into_response();
                }
            };

            existing_item
        } else {
            let (new_item, warning) = match parse_item_from_value(item_val) {
                Ok((item, warn)) => (item, warn),
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({ "error": format!("Invalid item: {}", e) })),
                    )
                        .into_response();
                }
            };

            if let Some(w) = warning {
                warnings.push(w);
            }

            match item_storage.create_item(new_item).await {
                Ok(item) => item,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": format!("Failed to create item: {}", e) })),
                    )
                        .into_response();
                }
            }
        };

        if let Err(e) = item_storage
            .link_item(&conversation_id, &item.id, added_at)
            .await
        {
            warn!("Failed to link item {}: {}", item.id.0, e);
        }

        created_items.push(item_to_json(&item));
    }

    let first_id = created_items.first().and_then(|v| v.get("id"));
    let last_id = created_items.last().and_then(|v| v.get("id"));

    let mut response = json!({
        "object": "list",
        "data": created_items,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": false
    });

    if !warnings.is_empty() {
        if let Some(obj) = response.as_object_mut() {
            obj.insert("warnings".to_string(), json!(warnings));
        }
    }

    (StatusCode::OK, Json(response)).into_response()
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
                Json(json!({
                    "error": format!("Failed to get conversation: {}", e)
                })),
            )
                .into_response();
        }
    }

    let is_linked = match item_storage
        .is_item_linked(&conversation_id, &item_id)
        .await
    {
        Ok(linked) => linked,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": format!("Failed to check item link: {}", e)
                })),
            )
                .into_response();
        }
    };

    if !is_linked {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Item not found in this conversation"})),
        )
            .into_response();
    }

    match item_storage.get_item(&item_id).await {
        Ok(Some(item)) => (StatusCode::OK, Json(item_to_json(&item))).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Item not found"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to get item: {}", e) })),
        )
            .into_response(),
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

    let conversation = match conversation_storage
        .get_conversation(&conversation_id)
        .await
    {
        Ok(Some(conv)) => conv,
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
                Json(json!({
                    "error": format!("Failed to get conversation: {}", e)
                })),
            )
                .into_response();
        }
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
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to delete item: {}", e) })),
        )
            .into_response(),
    }
}

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
        .map_err(|e| format!("Failed to create item: {}", e))?;

    if let Some(conv_id) = conv_id_opt {
        item_storage
            .link_item(conv_id, &created.id, Utc::now())
            .await
            .map_err(|e| format!("Failed to link item: {}", e))?;

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
    let status = item_val
        .get("status")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| Some("completed".to_string()));

    if item_type == "message" && role.is_none() {
        return Err("Message items require 'role' field".to_string());
    }

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

pub fn item_to_json(item: &ConversationItem) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("id".to_string(), json!(item.id.0));
    obj.insert("type".to_string(), json!(item.item_type));

    if let Some(role) = &item.role {
        obj.insert("role".to_string(), json!(role));
    }

    match item.item_type.as_str() {
        "mcp_call" => {
            if let Some(content_obj) = item.content.as_object() {
                if let Some(name) = content_obj.get("name") {
                    obj.insert("name".to_string(), name.clone());
                }
                if let Some(arguments) = content_obj.get("arguments") {
                    obj.insert("arguments".to_string(), arguments.clone());
                }
                if let Some(output) = content_obj.get("output") {
                    obj.insert("output".to_string(), output.clone());
                }
                if let Some(server_label) = content_obj.get("server_label") {
                    obj.insert("server_label".to_string(), server_label.clone());
                }
                if let Some(approval_request_id) = content_obj.get("approval_request_id") {
                    obj.insert(
                        "approval_request_id".to_string(),
                        approval_request_id.clone(),
                    );
                }
                if let Some(error) = content_obj.get("error") {
                    obj.insert("error".to_string(), error.clone());
                }
            }
        }
        "mcp_list_tools" => {
            if let Some(content_obj) = item.content.as_object() {
                if let Some(tools) = content_obj.get("tools") {
                    obj.insert("tools".to_string(), tools.clone());
                }
                if let Some(server_label) = content_obj.get("server_label") {
                    obj.insert("server_label".to_string(), server_label.clone());
                }
            }
        }
        "function_call" => {
            if let Some(content_obj) = item.content.as_object() {
                for field in ["call_id", "name", "arguments", "output"] {
                    if let Some(value) = content_obj.get(field) {
                        obj.insert(field.to_string(), value.clone());
                    }
                }
            }
        }
        "function_call_output" => {
            if let Some(content_obj) = item.content.as_object() {
                for field in ["call_id", "output"] {
                    if let Some(value) = content_obj.get(field) {
                        obj.insert(field.to_string(), value.clone());
                    }
                }
            }
        }
        _ => {
            obj.insert("content".to_string(), item.content.clone());
        }
    }

    if let Some(status) = &item.status {
        obj.insert("status".to_string(), json!(status));
    }

    Value::Object(obj)
}

pub fn conversation_to_json(conversation: &Conversation) -> Value {
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
