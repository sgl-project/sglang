//! Response patching and transformation utilities for OpenAI responses

use serde_json::{json, Map, Value};
use tracing::warn;

use crate::protocols::{
    event_types::is_response_event,
    responses::{ResponseToolType, ResponsesRequest},
};

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

/// Patch response JSON with metadata from original request
///
/// The upstream response may be missing fields that were in the original request.
/// This function ensures these fields are preserved in the final response:
/// - `previous_response_id` - conversation threading
/// - `instructions` - system instructions
/// - `metadata` - user-provided metadata
/// - `store` - whether to persist the response
/// - `model` - model identifier
/// - `safety_identifier` - user identifier for safety
pub(super) fn patch_response_with_request_metadata(
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
    let tools = original_body.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())
            .map(|t| {
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

                if t.authorization.is_some() {
                    m.insert("authorization".to_string(), json!("<redacted>"));
                }

                if let Some(headers) = &t.headers {
                    let mut redacted = Map::new();
                    for key in headers.keys() {
                        redacted.insert(key.clone(), json!("<redacted>"));
                    }
                    m.insert("headers".to_string(), Value::Object(redacted));
                }

                Value::Object(m)
            })
            .collect::<Vec<Value>>()
    });

    let Some(tools) = tools.filter(|tools| !tools.is_empty()) else {
        return;
    };

    if let Some(obj) = resp.as_object_mut() {
        obj.insert("tools".to_string(), Value::Array(tools));
        obj.entry("tool_choice").or_insert(json!("auto"));
    }
}
