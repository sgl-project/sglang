// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Client protocols served by converting to and from chat completions.

pub(crate) mod anthropic;
pub(crate) mod responses;
pub(crate) mod sse;

use serde_json::Value;

/// 5xx text can carry stack frames, paths or prompt fragments, so it never
/// reaches the client; 4xx keeps the validation message.
pub(crate) fn scrub_message(message: &str, status: u16) -> String {
    const MAX_CHARS: usize = 500;
    if status >= 500 {
        return "Internal server error".into();
    }
    let kept: Vec<&str> = message
        .lines()
        .filter(|l| !l.starts_with("Traceback") && !l.contains("File \"/"))
        .collect();
    let cleaned = kept.join("\n");
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return "Request failed".into();
    }
    match cleaned.char_indices().nth(MAX_CHARS) {
        Some((at, _)) => format!("{}\u{2026}", &cleaned[..at]),
        None => cleaned.to_owned(),
    }
}

pub(crate) fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// OpenAI error `type` for an HTTP status.
pub(crate) fn openai_error_type(status: u16) -> &'static str {
    if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    }
}

/// An in-band error on a chat stream chunk: its status (500 without a
/// `code`) and a client-safe message.
pub(crate) fn upstream_error(chunk: &Value) -> Option<(u16, String)> {
    let err = chunk.get("error").filter(|e| !e.is_null())?;
    let message = err
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| err.as_str())
        .unwrap_or("upstream error");
    let status = err
        .get("code")
        .and_then(Value::as_u64)
        .and_then(|c| u16::try_from(c).ok())
        .unwrap_or(500);
    Some((status, scrub_message(message, status)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scrubs_5xx_and_tracebacks() {
        assert_eq!(scrub_message("secret path", 502), "Internal server error");
        let traced = "bad input\nTraceback (most recent call last):\n  File \"/a.py\", line 1";
        assert_eq!(scrub_message(traced, 400), "bad input");
        let long = scrub_message(&"x".repeat(600), 400);
        assert_eq!(long.chars().count(), 501);
        assert!(long.ends_with('\u{2026}'));
        assert_eq!(scrub_message("", 400), "Request failed");
    }

    #[test]
    fn upstream_error_shapes() {
        let e = |v: Value| upstream_error(&json!({ "error": v }));
        assert_eq!(
            e(json!({"message": "too long", "code": 400})),
            Some((400, "too long".into()))
        );
        assert_eq!(
            e(json!("queue full")),
            Some((500, "Internal server error".into()))
        );
        assert_eq!(upstream_error(&json!({"choices": []})), None);
    }
}
