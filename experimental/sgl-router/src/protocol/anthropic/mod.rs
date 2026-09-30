// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API (`/v1/messages`) over chat completions.

pub mod request;
pub mod response;
pub mod stream;

use serde_json::{json, Value};

pub use request::{to_chat, Converted, EchoContext};

/// `input_tokens` excludes cache reads, as Anthropic defines it.
pub(crate) fn usage_from_chat(usage: Option<&Value>) -> Value {
    let int = |p: &str| {
        usage
            .and_then(|u| u.pointer(p))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let prompt = int("/prompt_tokens");
    let cached = int("/prompt_tokens_details/cached_tokens").min(prompt);
    json!({
        "input_tokens": prompt - cached,
        "output_tokens": int("/completion_tokens"),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    })
}

/// `finish_reason` + sglang's `matched_stop` → (`stop_reason`, `stop_sequence`).
pub(crate) fn stop_reason(
    finish_reason: Option<&str>,
    matched_stop: Option<&Value>,
    echo: &EchoContext,
) -> (&'static str, Value) {
    match finish_reason {
        Some("length") => ("max_tokens", Value::Null),
        Some("tool_calls") => ("tool_use", Value::Null),
        Some("content_filter") => ("refusal", Value::Null),
        _ => match matched_stop.and_then(Value::as_str) {
            Some(s) if echo.stop_sequences.iter().any(|q| q == s) => {
                ("stop_sequence", Value::String(s.to_owned()))
            }
            _ => ("end_turn", Value::Null),
        },
    }
}

pub fn error_type(status: u16) -> &'static str {
    match status {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        402 => "billing_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        504 => "timeout_error",
        _ => "api_error",
    }
}

pub fn error_body(status: u16, message: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "type": "error",
        "error": {"type": error_type(status), "message": message},
    }))
    .expect("serialize error")
}

/// Re-wrap any chat-path error body into the Anthropic envelope.
pub fn wrap_error_body(body: &[u8], status: u16) -> Vec<u8> {
    let v: Option<Value> = serde_json::from_slice(body).ok();
    if let Some(v) = &v {
        if v.get("type").and_then(Value::as_str) == Some("error")
            && v.get("error").is_some_and(Value::is_object)
        {
            return body.to_vec();
        }
    }
    let message = v
        .as_ref()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("message"))
                .or_else(|| v.get("detail"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            let text = String::from_utf8_lossy(body).trim().to_owned();
            (!text.is_empty()).then_some(text)
        })
        .unwrap_or_else(|| format!("upstream returned HTTP {status}"));
    error_body(status, &message)
}

pub(crate) fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

pub(crate) fn tool_input(arguments: &str) -> Value {
    match serde_json::from_str::<Value>(arguments) {
        Ok(v @ Value::Object(_)) => v,
        _ if arguments.trim().is_empty() => json!({}),
        _ => {
            tracing::warn!("messages: tool call arguments are not a JSON object");
            json!({})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo(stops: &[&str]) -> EchoContext {
        EchoContext {
            model: "m".into(),
            stop_sequences: stops.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn usage_subtracts_cache_reads() {
        let u = usage_from_chat(Some(&json!({"prompt_tokens": 10, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 4}})));
        assert_eq!(
            u,
            json!({"input_tokens": 6, "output_tokens": 3, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 4})
        );
        assert_eq!(usage_from_chat(None)["cache_read_input_tokens"], 0);
    }

    #[test]
    fn stop_reasons() {
        let e = echo(&["<END>"]);
        assert_eq!(stop_reason(Some("stop"), None, &e).0, "end_turn");
        assert_eq!(
            stop_reason(Some("stop"), Some(&json!("<END>")), &e),
            ("stop_sequence", json!("<END>"))
        );
        assert_eq!(stop_reason(Some("stop"), Some(&json!(2)), &e).0, "end_turn");
        assert_eq!(stop_reason(Some("length"), None, &e).0, "max_tokens");
        assert_eq!(stop_reason(Some("tool_calls"), None, &e).0, "tool_use");
    }

    #[test]
    fn error_wrapping() {
        let v: Value = serde_json::from_slice(&wrap_error_body(
            br#"{"error":{"type":"invalid_request_error","code":"bad_request","message":"bad"}}"#,
            400,
        ))
        .unwrap();
        assert_eq!(
            v,
            json!({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})
        );
        let v: Value = serde_json::from_slice(&wrap_error_body(
            br#"{"object":"error","message":"schema","type":"BadRequestError","code":400}"#,
            400,
        ))
        .unwrap();
        assert_eq!(v["error"]["message"], "schema");
        let v: Value = serde_json::from_slice(&wrap_error_body(b"", 503)).unwrap();
        assert_eq!(v["error"]["type"], "overloaded_error");
        assert!(!v["error"]["message"].as_str().unwrap().is_empty());
    }
}
