// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses API (`/v1/responses`) over chat completions.

pub mod request;
pub mod response;
pub mod stream;

use serde_json::{json, Map, Value};

pub(crate) use request::{to_chat, EchoContext, ToolMap};

pub(crate) fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

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

pub(crate) fn reasoning_item(id: &str, text: &str, status: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": null,
        "status": status,
    })
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

/// Re-wrap sglang's flat error body into `{"error": {...}}`; `None` = keep.
pub(crate) fn wrap_error_body(body: &[u8], status: u16) -> Option<Vec<u8>> {
    let v: Value = serde_json::from_slice(body).ok()?;
    if v.get("error").is_some_and(Value::is_object) {
        return None;
    }
    let message = v.get("message").and_then(Value::as_str)?;
    let typ = v
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_else(|| crate::server::error::openai_error_type(status));
    let code = v.get("code").cloned().unwrap_or(Value::Null);
    let param = v.get("param").cloned().unwrap_or(Value::Null);
    serde_json::to_vec(&json!({
        "error": {"message": message, "type": typ, "param": param, "code": code}
    }))
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let w: Value = serde_json::from_slice(&wrap_error_body(flat, 400).unwrap()).unwrap();
        assert_eq!(w["error"]["message"], "bad schema");
        assert_eq!(w["error"]["code"], 400);
        assert!(wrap_error_body(br#"{"error":{"message":"x"}}"#, 400).is_none());
        assert!(wrap_error_body(b"not json", 500).is_none());
    }
}
