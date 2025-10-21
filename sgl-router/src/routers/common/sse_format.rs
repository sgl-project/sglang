//! SSE Formatting Utilities
//!
//! Helpers for formatting Server-Sent Events (SSE) blocks.
//! Used by both OpenAI and gRPC routers.

use bytes::Bytes;
use serde_json::Value;

// ============================================================================
// SSE Formatting
// ============================================================================

/// Format a value as an SSE event block
///
/// # Arguments
/// * `event_type` - Optional event type (e.g., "response.created")
/// * `data` - JSON data to include
///
/// # Returns
/// Formatted SSE block: "event: {type}\ndata: {json}\n\n"
///
/// # Examples
/// ```
/// let event = json!({"type": "response.created", "sequence_number": 0});
/// let sse = format_sse_event(Some("response.created"), &event);
/// // Returns: "event: response.created\ndata: {...}\n\n"
/// ```
pub fn format_sse_event(event_type: Option<&str>, data: &Value) -> Result<Bytes, String> {
    let json_str =
        serde_json::to_string(data).map_err(|e| format!("Failed to serialize SSE data: {}", e))?;

    let mut block = String::new();

    if let Some(evt) = event_type {
        block.push_str("event: ");
        block.push_str(evt);
        block.push('\n');
    }

    block.push_str("data: ");
    block.push_str(&json_str);
    block.push_str("\n\n");

    Ok(Bytes::from(block))
}

/// Format a value as an SSE event block without event type
///
/// Convenience wrapper for `format_sse_event(None, data)`
pub fn format_sse_data(data: &Value) -> Result<Bytes, String> {
    format_sse_event(None, data)
}

/// Format the SSE "[DONE]" marker
pub fn format_sse_done() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_format_sse_event_with_type() {
        let data = json!({"foo": "bar"});
        let result = format_sse_event(Some("test.event"), &data).unwrap();
        let s = String::from_utf8(result.to_vec()).unwrap();

        assert!(s.starts_with("event: test.event\n"));
        assert!(s.contains("data: {"));
        assert!(s.ends_with("\n\n"));
    }

    #[test]
    fn test_format_sse_data_without_type() {
        let data = json!({"foo": "bar"});
        let result = format_sse_data(&data).unwrap();
        let s = String::from_utf8(result.to_vec()).unwrap();

        assert!(!s.contains("event:"));
        assert!(s.starts_with("data: {"));
        assert!(s.ends_with("\n\n"));
    }

    #[test]
    fn test_format_sse_done() {
        let result = format_sse_done();
        assert_eq!(result, "data: [DONE]\n\n");
    }
}
