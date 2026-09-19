//! Transport-neutral pieces of the OpenAI-compatible frontend: chat-template
//! rendering, tool-call constraint + parsing, and reasoning-content splitting.
//! The HTTP handlers stay in `api_server::openai`.

pub(crate) mod chat;
pub(crate) mod completions;
pub(crate) mod models;
pub(crate) mod reasoning;
pub(crate) mod template;
pub(crate) mod tools;

pub(crate) fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn unix_seconds_u32() -> u32 {
    u32::try_from(unix_seconds()).unwrap_or(u32::MAX)
}

/// The OpenAI error payload — the body shape every OpenAI-compatible surface
/// answers errors with, regardless of transport framing.
pub(crate) fn error_payload_value(code: u16, message: &str) -> serde_json::Value {
    let error_type = if code == 401 {
        "AuthenticationError"
    } else if (500..600).contains(&code) {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message,
            "type": error_type,
            "param": null,
            "code": code,
        }
    })
}
