//! Minimal Anthropic protocol types used for request routing.
//!
//! The SGLang inference engine natively supports the Anthropic Messages API
//! at `/v1/messages`, so the gateway does **not** need to convert between
//! protocols. These types exist only to extract the fields required for
//! backend worker selection (`model`) and response handling (`stream`).
//! The complete request body is forwarded to the backend unchanged.

use serde::Deserialize;

/// Routing-relevant fields extracted from an Anthropic Messages API request.
///
/// Only `model` and `stream` are needed by the gateway:
/// - `model` → used to select the target backend worker.
/// - `stream` → used to set the correct `Content-Type` on the response.
///
/// All other fields are preserved in the raw request body and forwarded
/// verbatim to the backend.
#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    /// Model identifier, e.g. `"claude-3-5-sonnet-20241022"`.
    pub model: String,

    /// Whether the client wants a streaming SSE response.
    #[serde(default)]
    pub stream: bool,
}

/// Routing-relevant fields extracted from an Anthropic token-count request.
///
/// Only `model` is needed for worker selection; the full body is forwarded
/// verbatim to `POST /v1/messages/count_tokens` on the backend.
#[derive(Debug, Deserialize)]
pub struct AnthropicCountTokensRequest {
    pub model: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AnthropicMessagesRequest ──────────────────────────────────────────

    #[test]
    fn messages_request_parses_model_and_stream() {
        let json = r#"{"model":"claude-3-5-sonnet-20241022","max_tokens":1024,"stream":true,"messages":[]}"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-3-5-sonnet-20241022");
        assert!(req.stream);
    }

    #[test]
    fn messages_request_stream_defaults_to_false() {
        let json = r#"{"model":"claude-3-haiku-20240307","max_tokens":256,"messages":[]}"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-3-haiku-20240307");
        assert!(!req.stream);
    }

    #[test]
    fn messages_request_missing_model_fails() {
        let json = r#"{"max_tokens":100,"messages":[]}"#;
        assert!(serde_json::from_str::<AnthropicMessagesRequest>(json).is_err());
    }

    #[test]
    fn messages_request_ignores_unknown_fields() {
        // Gateway only cares about `model` and `stream`; all other fields
        // are forwarded verbatim in the raw body.
        let json = r#"{"model":"claude-3-5-sonnet-20241022","max_tokens":512,
                       "temperature":0.7,"tools":[],"messages":[]}"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-3-5-sonnet-20241022");
    }

    // ── AnthropicCountTokensRequest ───────────────────────────────────────

    #[test]
    fn count_tokens_request_parses_model() {
        let json = r#"{"model":"claude-3-5-sonnet-20241022","messages":[]}"#;
        let req: AnthropicCountTokensRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn count_tokens_request_missing_model_fails() {
        let json = r#"{"messages":[]}"#;
        assert!(serde_json::from_str::<AnthropicCountTokensRequest>(json).is_err());
    }

    #[test]
    fn count_tokens_request_ignores_extra_fields() {
        let json = r#"{"model":"claude-3-haiku-20240307","system":"Be concise.","messages":[],"tools":[]}"#;
        let req: AnthropicCountTokensRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-3-haiku-20240307");
    }
}
