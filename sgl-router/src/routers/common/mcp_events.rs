//! MCP Event Builders
//!
//! Shared builders for MCP-related SSE events.
//! Used by both OpenAI router (transformation) and gRPC router (generation).

use serde_json::{json, Value};

use super::event_types;

// ============================================================================
// MCP List Tools Events
// ============================================================================

/// Build mcp_list_tools.in_progress event
pub fn build_mcp_list_tools_in_progress(
    output_index: usize,
    item_id: &str,
    sequence_number: u64,
) -> Value {
    json!({
        "type": event_types::MCP_LIST_TOOLS_IN_PROGRESS,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id
    })
}

/// Build mcp_list_tools.completed event with tools list
pub fn build_mcp_list_tools_completed(
    output_index: usize,
    item_id: &str,
    sequence_number: u64,
    tools: &[Value],
) -> Value {
    json!({
        "type": event_types::MCP_LIST_TOOLS_COMPLETED,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "tools": tools
    })
}

// ============================================================================
// MCP Call Events
// ============================================================================

/// Build mcp_call.in_progress event
pub fn build_mcp_call_in_progress(
    output_index: usize,
    item_id: &str,
    sequence_number: u64,
) -> Value {
    json!({
        "type": event_types::MCP_CALL_IN_PROGRESS,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id
    })
}

/// Build mcp_call_arguments.delta event
pub fn build_mcp_call_arguments_delta(
    output_index: usize,
    item_id: &str,
    sequence_number: u64,
    delta: &str,
) -> Value {
    json!({
        "type": event_types::MCP_CALL_ARGUMENTS_DELTA,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "delta": delta
    })
}

/// Build mcp_call_arguments.done event
pub fn build_mcp_call_arguments_done(
    output_index: usize,
    item_id: &str,
    sequence_number: u64,
    arguments: &str,
) -> Value {
    json!({
        "type": event_types::MCP_CALL_ARGUMENTS_DONE,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "arguments": arguments
    })
}

/// Build mcp_call.completed event
pub fn build_mcp_call_completed(output_index: usize, item_id: &str, sequence_number: u64) -> Value {
    json!({
        "type": event_types::MCP_CALL_COMPLETED,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id
    })
}

/// Build mcp_call.failed event
pub fn build_mcp_call_failed(
    output_index: usize,
    item_id: &str,
    sequence_number: u64,
    error: &str,
) -> Value {
    json!({
        "type": event_types::MCP_CALL_FAILED,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "error": error
    })
}

// ============================================================================
// Output Item Wrapper Events
// ============================================================================

/// Build output_item.added event
pub fn build_output_item_added(output_index: usize, sequence_number: u64, item: &Value) -> Value {
    json!({
        "type": event_types::OUTPUT_ITEM_ADDED,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item": item
    })
}

/// Build output_item.done event
pub fn build_output_item_done(output_index: usize, sequence_number: u64, item: &Value) -> Value {
    json!({
        "type": event_types::OUTPUT_ITEM_DONE,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item": item
    })
}

// ============================================================================
// Content Events
// ============================================================================

/// Build content_part.added event
pub fn build_content_part_added(
    output_index: usize,
    item_id: &str,
    content_index: usize,
    sequence_number: u64,
) -> Value {
    json!({
        "type": event_types::CONTENT_PART_ADDED,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "part": {
            "type": "text",
            "text": ""
        }
    })
}

/// Build output_text.delta event
pub fn build_output_text_delta(
    output_index: usize,
    item_id: &str,
    content_index: usize,
    sequence_number: u64,
    delta: &str,
) -> Value {
    json!({
        "type": event_types::OUTPUT_TEXT_DELTA,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "delta": delta
    })
}

/// Build output_text.done event
pub fn build_output_text_done(
    output_index: usize,
    item_id: &str,
    content_index: usize,
    sequence_number: u64,
    text: &str,
) -> Value {
    json!({
        "type": event_types::OUTPUT_TEXT_DONE,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "text": text
    })
}

/// Build content_part.done event
pub fn build_content_part_done(
    output_index: usize,
    item_id: &str,
    content_index: usize,
    sequence_number: u64,
    text: &str,
) -> Value {
    json!({
        "type": event_types::CONTENT_PART_DONE,
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "part": {
            "type": "text",
            "text": text
        }
    })
}

// ============================================================================
// Response Lifecycle Events
// ============================================================================

/// Build response.created event
pub fn build_response_created(
    response_id: &str,
    model: &str,
    created_at: u64,
    sequence_number: u64,
) -> Value {
    json!({
        "type": event_types::RESPONSE_CREATED,
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": model,
            "output": []
        }
    })
}

/// Build response.in_progress event
pub fn build_response_in_progress(response_id: &str, sequence_number: u64) -> Value {
    json!({
        "type": event_types::RESPONSE_IN_PROGRESS,
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "status": "in_progress"
        }
    })
}

/// Build response.completed event
pub fn build_response_completed(
    response_id: &str,
    model: &str,
    created_at: u64,
    sequence_number: u64,
    output: Vec<Value>,
    usage: Option<&Value>,
) -> Value {
    let mut response = json!({
        "type": event_types::RESPONSE_COMPLETED,
        "sequence_number": sequence_number,
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": model,
            "output": output
        }
    });

    if let Some(usage_val) = usage {
        response["response"]["usage"] = usage_val.clone();
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_mcp_call_in_progress() {
        let event = build_mcp_call_in_progress(2, "mcp_123", 5);
        assert_eq!(event["type"], "response.mcp_call.in_progress");
        assert_eq!(event["output_index"], 2);
        assert_eq!(event["item_id"], "mcp_123");
        assert_eq!(event["sequence_number"], 5);
    }

    #[test]
    fn test_build_mcp_call_arguments_delta() {
        let event = build_mcp_call_arguments_delta(2, "mcp_123", 6, "{\"foo\":");
        assert_eq!(event["type"], "response.mcp_call_arguments.delta");
        assert_eq!(event["delta"], "{\"foo\":");
    }

    #[test]
    fn test_build_output_item_added() {
        let item = json!({"id": "mcp_123", "type": "mcp_call"});
        let event = build_output_item_added(2, 4, &item);
        assert_eq!(event["type"], "response.output_item.added");
        assert_eq!(event["output_index"], 2);
        assert_eq!(event["sequence_number"], 4);
        assert_eq!(event["item"]["id"], "mcp_123");
    }

    #[test]
    fn test_build_response_created() {
        let event = build_response_created("resp_123", "gpt-4", 1234567890, 0);
        assert_eq!(event["type"], "response.created");
        assert_eq!(event["response"]["id"], "resp_123");
        assert_eq!(event["response"]["model"], "gpt-4");
        assert_eq!(event["response"]["status"], "in_progress");
    }
}
