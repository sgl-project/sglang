//! OpenAI error payload schema.

pub(crate) fn error_payload(
    code: u16,
    message: impl Into<String>,
    error_type: &str,
) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "object": "error", "message": message.into(), "type": error_type,
            "param": null, "code": code,
        }
    })
}
