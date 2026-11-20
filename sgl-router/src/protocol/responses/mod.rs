// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses API (`/v1/responses`) over chat completions.

pub mod request;
pub mod response;
pub mod stream;

use serde_json::{json, Map, Value};

pub(crate) use request::{to_chat, EchoContext, ToolMap};

pub(crate) use crate::protocol::new_id;
use crate::protocol::{openai_error_type, scrub_message};

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Chat `usage` → Responses `usage`; detail fields are always present.
pub(crate) fn usage_from_chat(usage: Option<&Value>) -> Value {
    let int = |v: Option<&Value>| v.and_then(Value::as_u64).unwrap_or(0);
    let u = usage.filter(|u| u.is_object());
    let input = int(u.and_then(|u| u.get("prompt_tokens")));
    let output = int(u.and_then(|u| u.get("completion_tokens")));
    let cached = int(u.and_then(|u| u.pointer("/prompt_tokens_details/cached_tokens")));
    // sglang reports `reasoning_tokens` at the top level; OpenAI nests it.
    let reasoning = u
        .and_then(|u| {
            u.get("reasoning_tokens")
                .filter(|v| !v.is_null())
                .or_else(|| u.pointer("/completion_tokens_details/reasoning_tokens"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens": input,
        // Required by the OpenAI SDK; the engine does not report it.
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input + output,
    })
}

/// How a finished generation ends the Response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finish {
    Completed,
    Incomplete(&'static str),
    /// The engine aborted the request (`finish_reason: "abort"`).
    Aborted,
}

impl Finish {
    pub(crate) fn from_chat(finish_reason: Option<&str>) -> Self {
        match finish_reason {
            Some("length") => Finish::Incomplete("max_output_tokens"),
            Some("content_filter") => Finish::Incomplete("content_filter"),
            Some("abort") => Finish::Aborted,
            _ => Finish::Completed,
        }
    }

    pub(crate) fn status(self) -> &'static str {
        match self {
            Finish::Completed => "completed",
            Finish::Incomplete(_) => "incomplete",
            Finish::Aborted => "failed",
        }
    }

    pub(crate) fn item_status(self) -> &'static str {
        match self {
            Finish::Completed => "completed",
            _ => "incomplete",
        }
    }

    /// The terminal stream event.
    pub(crate) fn event(self) -> &'static str {
        match self {
            Finish::Completed => "response.completed",
            Finish::Incomplete(_) => "response.incomplete",
            Finish::Aborted => "response.failed",
        }
    }

    /// Sets `incomplete_details` or `error` on a finished Response.
    pub(crate) fn annotate(self, resp: &mut Value) {
        match self {
            Finish::Completed => {}
            Finish::Incomplete(reason) => resp["incomplete_details"] = json!({ "reason": reason }),
            Finish::Aborted => resp["error"] = failed_error("generation was aborted"),
        }
    }
}

pub(crate) fn failed_error(message: &str) -> Value {
    json!({"code": "server_error", "message": message})
}

pub(crate) fn response_object(
    echo: &EchoContext,
    id: &str,
    created_at: u64,
    status: &str,
    output: Vec<Value>,
    usage: Value,
) -> Value {
    let mut m = Map::new();
    m.insert("id".into(), id.into());
    m.insert("object".into(), "response".into());
    m.insert("created_at".into(), created_at.into());
    m.insert(
        "completed_at".into(),
        if status == "completed" {
            now_secs().max(created_at).into()
        } else {
            Value::Null
        },
    );
    m.insert("status".into(), status.into());
    m.insert("error".into(), Value::Null);
    m.insert("incomplete_details".into(), Value::Null);
    m.insert("model".into(), echo.model.clone().into());
    m.insert("output".into(), Value::Array(output));
    m.insert("usage".into(), usage);
    for (k, v) in &echo.fields {
        m.insert(k.clone(), v.clone());
    }
    Value::Object(m)
}

/// The trace goes in `content`; `reasoning.summary` mirrors it into
/// `summary`, and `include: ["reasoning.encrypted_content"]` packs it into
/// `encrypted_content`, both as the engine's own `/v1/responses` does.
pub(crate) fn reasoning_item(id: &str, text: &str, status: &str, echo: &EchoContext) -> Value {
    let summary = if echo.reasoning_summary {
        json!([{"type": "summary_text", "text": text}])
    } else {
        json!([])
    };
    let encrypted = if echo.encrypted_reasoning {
        Value::String(encode_reasoning_state(text))
    } else {
        Value::Null
    };
    json!({
        "id": id,
        "type": "reasoning",
        "summary": summary,
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": encrypted,
        "status": status,
    })
}

/// The engine's `reasoning.encrypted_content` blob: a prefix plus the
/// zlib-compressed, URL-safe base64 trace. Encoded, not encrypted: it lets a
/// `store: false` client replay reasoning without server-side state, and it
/// is interchangeable with the engine's own.
const REASONING_STATE_PREFIX: &str = "sglang-reasoning-v1.";

pub(crate) fn encode_reasoning_state(text: &str) -> String {
    use base64::Engine;
    use std::io::Write;
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = z.write_all(text.as_bytes());
    let packed = z.finish().unwrap_or_default();
    format!(
        "{REASONING_STATE_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE.encode(packed)
    )
}

/// `None` for a blob this engine family did not produce, or a damaged one.
pub(crate) fn decode_reasoning_state(blob: &str) -> Option<String> {
    use base64::Engine;
    use std::io::Read;
    let packed = base64::engine::general_purpose::URL_SAFE
        .decode(blob.strip_prefix(REASONING_STATE_PREFIX)?)
        .ok()?;
    let mut text = String::new();
    flate2::read::ZlibDecoder::new(&packed[..])
        .read_to_string(&mut text)
        .ok()?;
    Some(text)
}

pub(crate) fn output_text_part(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
}

pub(crate) fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [output_text_part(text)],
    })
}

/// A tool call output item; `name` is the chat function name, mapped back to
/// the declared tool.
pub(crate) fn call_item(
    id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
    status: &str,
    tools: &ToolMap,
) -> Value {
    let Some(t) = tools.get(name) else {
        return json!({"id": id, "type": "function_call", "call_id": call_id, "name": name,
                      "arguments": arguments, "status": status});
    };
    let mut item = if t.custom {
        json!({"id": id, "type": "custom_tool_call", "call_id": call_id, "name": t.name,
               "input": custom_input(arguments), "status": status})
    } else {
        json!({"id": id, "type": "function_call", "call_id": call_id, "name": t.name,
               "arguments": arguments, "status": status})
    };
    if let Some(ns) = &t.namespace {
        item["namespace"] = Value::String(ns.clone());
    }
    item
}

/// A custom tool's raw input, unwrapped from `{"input": ...}`.
pub(crate) fn custom_input(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|v| v.get("input").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| arguments.to_owned())
}

pub(crate) fn error_body(status: u16, message: &str) -> Vec<u8> {
    let error = json!({"message": message, "type": openai_error_type(status),
                       "param": null, "code": status});
    serde_json::to_vec(&json!({ "error": error })).expect("serialize error")
}

/// Any non-2xx body in the `{"error": {...}}` envelope. The router's own
/// envelope is kept; engine text is scrubbed.
pub(crate) fn wrap_error_body(body: &[u8], status: u16) -> Vec<u8> {
    let v: Option<Value> = serde_json::from_slice(body).ok();
    if v.as_ref()
        .is_some_and(|v| v.get("error").is_some_and(Value::is_object))
    {
        return body.to_vec();
    }
    let field = |k: &str| v.as_ref().and_then(|v| v.get(k)).filter(|f| !f.is_null());
    let message = field("message")
        .and_then(Value::as_str)
        .map_or_else(|| String::from_utf8_lossy(body).into_owned(), str::to_owned);
    let error = json!({
        "message": scrub_message(&message, status),
        // The engine's own `/v1/responses` reports the OpenAI type, not the
        // chat route's `BadRequestError`.
        "type": openai_error_type(status),
        "param": field("param").cloned().unwrap_or(Value::Null),
        "code": field("code").cloned().unwrap_or_else(|| status.into()),
    });
    serde_json::to_vec(&json!({ "error": error })).expect("serialize error")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_state_round_trips_and_rejects_foreign_blobs() {
        let blob = encode_reasoning_state("why: 42 ✓");
        assert!(blob.starts_with("sglang-reasoning-v1."));
        assert_eq!(decode_reasoning_state(&blob).as_deref(), Some("why: 42 ✓"));
        assert_eq!(decode_reasoning_state("gAAAAA"), None);
        assert_eq!(decode_reasoning_state("sglang-reasoning-v1.@@"), None);
        // The engine's own encoding (Python zlib.compress + urlsafe_b64encode).
        assert_eq!(
            decode_reasoning_state("sglang-reasoning-v1.eJzLyAQAATsA0g==").as_deref(),
            Some("hi")
        );
        assert_eq!(
            encode_reasoning_state("why: 42 ✓"),
            "sglang-reasoning-v1.eJwrz6i0UjAxUng0ZzIAF0wESg=="
        );
    }

    #[test]
    fn reasoning_item_honours_summary_and_encrypted_flags() {
        let mut echo = to_chat(json!({"model": "m", "input": "x"}), false)
            .unwrap()
            .echo;
        let plain = reasoning_item("rs_1", "t", "completed", &echo);
        assert_eq!(plain["summary"], json!([]));
        assert_eq!(plain["encrypted_content"], Value::Null);
        echo.reasoning_summary = true;
        echo.encrypted_reasoning = true;
        let full = reasoning_item("rs_1", "t", "completed", &echo);
        assert_eq!(
            full["summary"],
            json!([{"type": "summary_text", "text": "t"}])
        );
        assert_eq!(
            decode_reasoning_state(full["encrypted_content"].as_str().unwrap()).as_deref(),
            Some("t")
        );
    }

    #[test]
    fn usage_maps_and_defaults_details() {
        let u = usage_from_chat(Some(&json!({
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4}, "reasoning_tokens": 3,
        })));
        assert_eq!(
            u,
            json!({"input_tokens": 10,
                   "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 0},
                   "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 3},
                   "total_tokens": 15})
        );
        let u = usage_from_chat(Some(&json!({"prompt_tokens": 2, "completion_tokens": 1})));
        assert_eq!(u["input_tokens_details"]["cached_tokens"], 0);
        assert_eq!(u["output_tokens_details"]["reasoning_tokens"], 0);
        assert_eq!(u["total_tokens"], 3);
    }

    #[test]
    fn wraps_flat_sglang_errors_only() {
        let flat = br#"{"object":"error","message":"bad schema","type":"BadRequestError","param":null,"code":400}"#;
        let w: Value = serde_json::from_slice(&wrap_error_body(flat, 400)).unwrap();
        assert_eq!(w["error"]["message"], "bad schema");
        assert_eq!(w["error"]["code"], 400);
        assert_eq!(w["error"]["type"], "invalid_request_error");
        let own = br#"{"error":{"message":"x"}}"#;
        assert_eq!(wrap_error_body(own, 503), own);
        // Non-JSON bodies are wrapped too; 5xx text never reaches the client.
        let w: Value = serde_json::from_slice(&wrap_error_body(b"Traceback: /srv", 500)).unwrap();
        assert_eq!(w["error"]["message"], "Internal server error");
        assert_eq!(w["error"]["type"], "server_error");
    }
}
