//! Response storage, patching, and extraction utilities

use std::collections::HashMap;

use serde_json::{json, Value};
use tracing::warn;

use super::utils::event_types;
use crate::{
    data_connector::{ResponseId, StoredResponse},
    protocols::responses::{ResponseToolType, ResponsesRequest},
};

// ============================================================================
// Response Storage Operations
// ============================================================================

/// Build a StoredResponse from response JSON and original request
pub(super) fn build_stored_response(
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> StoredResponse {
    let mut stored_response = StoredResponse::new(None);

    // Initialize empty arrays - will be populated by persist_items_with_storages
    stored_response.input = Value::Array(vec![]);
    stored_response.output = Value::Array(vec![]);

    stored_response.instructions = response_json
        .get("instructions")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| original_body.instructions.clone());

    stored_response.model = response_json
        .get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| Some(original_body.model.clone()));

    if let Some(safety_identifier) = original_body.user.clone() {
        stored_response.safety_identifier = Some(safety_identifier);
    }

    // Set conversation id from request if provided
    if let Some(conv_id) = original_body.conversation.clone() {
        stored_response.conversation_id = Some(conv_id);
    }

    stored_response.metadata = response_json
        .get("metadata")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_else(|| original_body.metadata.clone().unwrap_or_default());

    stored_response.previous_response_id = response_json
        .get("previous_response_id")
        .and_then(|v| v.as_str())
        .map(ResponseId::from)
        .or_else(|| {
            original_body
                .previous_response_id
                .as_ref()
                .map(|id| ResponseId::from(id.as_str()))
        });

    if let Some(id_str) = response_json.get("id").and_then(|v| v.as_str()) {
        stored_response.id = ResponseId::from(id_str);
    }

    stored_response.raw_response = response_json.clone();

    stored_response
}

// ============================================================================
// Response JSON Patching
// ============================================================================

/// Patch streaming response JSON with metadata from original request
pub(super) fn patch_streaming_response_json(
    response_json: &mut Value,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<&str>,
) {
    if let Some(obj) = response_json.as_object_mut() {
        if let Some(prev_id) = original_previous_response_id {
            let should_insert = obj
                .get("previous_response_id")
                .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                .unwrap_or(true);
            if should_insert {
                obj.insert(
                    "previous_response_id".to_string(),
                    Value::String(prev_id.to_string()),
                );
            }
        }

        if !obj.contains_key("instructions")
            || obj
                .get("instructions")
                .map(|v| v.is_null())
                .unwrap_or(false)
        {
            if let Some(instructions) = &original_body.instructions {
                obj.insert(
                    "instructions".to_string(),
                    Value::String(instructions.clone()),
                );
            }
        }

        if !obj.contains_key("metadata")
            || obj.get("metadata").map(|v| v.is_null()).unwrap_or(false)
        {
            if let Some(metadata) = &original_body.metadata {
                let metadata_map: serde_json::Map<String, Value> = metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                obj.insert("metadata".to_string(), Value::Object(metadata_map));
            }
        }

        obj.insert(
            "store".to_string(),
            Value::Bool(original_body.store.unwrap_or(false)),
        );

        if obj
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.is_empty())
            .unwrap_or(true)
        {
            obj.insert(
                "model".to_string(),
                Value::String(original_body.model.clone()),
            );
        }

        if obj
            .get("safety_identifier")
            .map(|v| v.is_null())
            .unwrap_or(false)
        {
            if let Some(safety_identifier) = &original_body.user {
                obj.insert(
                    "safety_identifier".to_string(),
                    Value::String(safety_identifier.clone()),
                );
            }
        }

        // Attach conversation id for client response if present (final aggregated JSON)
        if let Some(conv_id) = original_body.conversation.clone() {
            obj.insert("conversation".to_string(), json!({ "id": conv_id }));
        }
    }
}

/// Rewrite streaming SSE block to include metadata from original request
pub(super) fn rewrite_streaming_block(
    block: &str,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<&str>,
) -> Option<String> {
    let trimmed = block.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut data_lines: Vec<String> = Vec::new();

    for line in trimmed.lines() {
        if line.starts_with("data:") {
            data_lines.push(line.trim_start_matches("data:").trim_start().to_string());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    let payload = data_lines.join("\n");
    let mut parsed: Value = match serde_json::from_str(&payload) {
        Ok(value) => value,
        Err(err) => {
            warn!("Failed to parse streaming JSON payload: {}", err);
            return None;
        }
    };

    let event_type = parsed
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    let should_patch = matches!(
        event_type,
        event_types::RESPONSE_CREATED
            | event_types::RESPONSE_IN_PROGRESS
            | event_types::RESPONSE_COMPLETED
    );

    if !should_patch {
        return None;
    }

    let mut changed = false;
    if let Some(response_obj) = parsed.get_mut("response").and_then(|v| v.as_object_mut()) {
        let desired_store = Value::Bool(original_body.store.unwrap_or(false));
        if response_obj.get("store") != Some(&desired_store) {
            response_obj.insert("store".to_string(), desired_store);
            changed = true;
        }

        if let Some(prev_id) = original_previous_response_id {
            let needs_previous = response_obj
                .get("previous_response_id")
                .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                .unwrap_or(true);

            if needs_previous {
                response_obj.insert(
                    "previous_response_id".to_string(),
                    Value::String(prev_id.to_string()),
                );
                changed = true;
            }
        }

        // Attach conversation id into streaming event response content with ordering
        if let Some(conv_id) = original_body.conversation.clone() {
            response_obj.insert("conversation".to_string(), json!({ "id": conv_id }));
            changed = true;
        }
    }

    if !changed {
        return None;
    }

    let new_payload = match serde_json::to_string(&parsed) {
        Ok(json) => json,
        Err(err) => {
            warn!("Failed to serialize modified streaming payload: {}", err);
            return None;
        }
    };

    let mut rebuilt_lines = Vec::new();
    let mut data_written = false;
    for line in trimmed.lines() {
        if line.starts_with("data:") {
            if !data_written {
                rebuilt_lines.push(format!("data: {}", new_payload));
                data_written = true;
            }
        } else {
            rebuilt_lines.push(line.to_string());
        }
    }

    if !data_written {
        rebuilt_lines.push(format!("data: {}", new_payload));
    }

    Some(rebuilt_lines.join("\n"))
}

/// Mask function tools as MCP tools in response for client
pub(super) fn mask_tools_as_mcp(resp: &mut Value, original_body: &ResponsesRequest) {
    let mcp_tool = original_body.tools.as_ref().and_then(|tools| {
        tools
            .iter()
            .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())
    });
    let Some(t) = mcp_tool else {
        return;
    };

    let mut m = serde_json::Map::new();
    m.insert("type".to_string(), Value::String("mcp".to_string()));
    if let Some(label) = &t.server_label {
        m.insert("server_label".to_string(), Value::String(label.clone()));
    }
    if let Some(url) = &t.server_url {
        m.insert("server_url".to_string(), Value::String(url.clone()));
    }
    if let Some(desc) = &t.server_description {
        m.insert(
            "server_description".to_string(),
            Value::String(desc.clone()),
        );
    }
    if let Some(req) = &t.require_approval {
        m.insert("require_approval".to_string(), Value::String(req.clone()));
    }
    if let Some(allowed) = &t.allowed_tools {
        m.insert(
            "allowed_tools".to_string(),
            Value::Array(allowed.iter().map(|s| Value::String(s.clone())).collect()),
        );
    }

    if let Some(obj) = resp.as_object_mut() {
        obj.insert("tools".to_string(), Value::Array(vec![Value::Object(m)]));
        obj.entry("tool_choice")
            .or_insert(Value::String("auto".to_string()));
    }
}
