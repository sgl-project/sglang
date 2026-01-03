use serde_json::{json, Map, Value};
use tracing::warn;

use crate::{
    data_connector::{ResponseId, StoredResponse},
    protocols::{
        event_types::is_response_event,
        responses::{ResponseToolType, ResponsesRequest},
    },
};

/// Extract a string field from JSON, returning owned String
fn get_string(json: &Value, key: &str) -> Option<String> {
    json.get(key).and_then(|v| v.as_str()).map(String::from)
}

/// Check if a JSON value is missing, null, or an empty string
fn is_missing_or_empty(value: Option<&Value>) -> bool {
    match value {
        None => true,
        Some(v) => v.is_null() || v.as_str().is_some_and(|s| s.is_empty()),
    }
}

/// Insert a string value into a JSON object if the condition is met
fn insert_if<F>(obj: &mut Map<String, Value>, key: &str, value: &str, condition: F)
where
    F: FnOnce(&Map<String, Value>) -> bool,
{
    if condition(obj) {
        obj.insert(key.to_string(), Value::String(value.to_string()));
    }
}

/// Build a StoredResponse from response JSON and original request
pub(super) fn build_stored_response(
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> StoredResponse {
    let mut stored = StoredResponse::new(None);

    // Initialize empty arrays - will be populated by persist_items_with_storages
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

/// Patch streaming response JSON with metadata from original request
pub(super) fn patch_streaming_response_json(
    response_json: &mut Value,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<&str>,
) {
    let Some(obj) = response_json.as_object_mut() else {
        return;
    };

    // Set previous_response_id if missing/empty
    if let Some(prev_id) = original_previous_response_id {
        insert_if(obj, "previous_response_id", prev_id, |o| {
            is_missing_or_empty(o.get("previous_response_id"))
        });
    }

    // Set instructions if missing/null
    if let Some(instructions) = &original_body.instructions {
        insert_if(obj, "instructions", instructions, |o| {
            is_missing_or_empty(o.get("instructions"))
        });
    }

    // Set metadata if missing/null
    if is_missing_or_empty(obj.get("metadata")) {
        if let Some(metadata) = &original_body.metadata {
            let metadata_map: Map<String, Value> = metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            obj.insert("metadata".to_string(), Value::Object(metadata_map));
        }
    }

    // Always set store
    obj.insert(
        "store".to_string(),
        Value::Bool(original_body.store.unwrap_or(false)),
    );

    // Set model if missing/empty
    insert_if(obj, "model", &original_body.model, |o| {
        is_missing_or_empty(o.get("model"))
    });

    // Set safety_identifier if null (but key exists)
    if let Some(user) = &original_body.user {
        if obj
            .get("safety_identifier")
            .is_some_and(|v: &Value| v.is_null())
        {
            obj.insert("safety_identifier".to_string(), Value::String(user.clone()));
        }
    }

    // Attach conversation id for client response
    if let Some(conv_id) = &original_body.conversation {
        obj.insert("conversation".to_string(), json!({ "id": conv_id }));
    }
}

/// Extract data payload from SSE block lines
fn extract_sse_data(block: &str) -> Option<String> {
    let data_lines: Vec<_> = block
        .lines()
        .filter(|line| line.starts_with("data:"))
        .map(|line| line.trim_start_matches("data:").trim_start())
        .collect();

    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
    }
}

/// Rebuild SSE block with new data payload
fn rebuild_sse_block(block: &str, new_payload: &str) -> String {
    let mut rebuilt_lines = Vec::new();
    let mut data_written = false;

    for line in block.lines() {
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

    rebuilt_lines.join("\n")
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

    let payload = extract_sse_data(trimmed)?;
    let mut parsed: Value = serde_json::from_str(&payload)
        .map_err(|e| warn!("Failed to parse streaming JSON payload: {}", e))
        .ok()?;

    let event_type = parsed
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    if !is_response_event(event_type) {
        return None;
    }

    let response_obj = parsed.get_mut("response").and_then(|v| v.as_object_mut())?;
    let mut changed = false;

    // Update store value if different
    let desired_store = Value::Bool(original_body.store.unwrap_or(false));
    if response_obj.get("store") != Some(&desired_store) {
        response_obj.insert("store".to_string(), desired_store);
        changed = true;
    }

    // Set previous_response_id if missing/empty
    if let Some(prev_id) = original_previous_response_id {
        if is_missing_or_empty(response_obj.get("previous_response_id")) {
            response_obj.insert("previous_response_id".to_string(), json!(prev_id));
            changed = true;
        }
    }

    // Attach conversation id
    if let Some(conv_id) = &original_body.conversation {
        response_obj.insert("conversation".to_string(), json!({ "id": conv_id }));
        changed = true;
    }

    if !changed {
        return None;
    }

    let new_payload = serde_json::to_string(&parsed)
        .map_err(|e| warn!("Failed to serialize modified streaming payload: {}", e))
        .ok()?;

    Some(rebuild_sse_block(trimmed, &new_payload))
}

/// Helper to insert an optional string field into a JSON map
fn insert_optional_string(map: &mut Map<String, Value>, key: &str, value: &Option<String>) {
    if let Some(v) = value {
        map.insert(key.to_string(), Value::String(v.clone()));
    }
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

    let mut m = Map::new();
    m.insert("type".to_string(), json!("mcp"));
    insert_optional_string(&mut m, "server_label", &t.server_label);
    insert_optional_string(&mut m, "server_url", &t.server_url);
    insert_optional_string(&mut m, "server_description", &t.server_description);
    insert_optional_string(&mut m, "require_approval", &t.require_approval);

    if let Some(allowed) = &t.allowed_tools {
        m.insert(
            "allowed_tools".to_string(),
            Value::Array(allowed.iter().map(|s| json!(s)).collect()),
        );
    }

    if let Some(obj) = resp.as_object_mut() {
        obj.insert("tools".to_string(), json!([Value::Object(m)]));
        obj.entry("tool_choice").or_insert(json!("auto"));
    }
}
