//! Response pieces shared by the OpenAI endpoints, shaped as Python serializes them.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::py::python_json;

/// A complete HTTP response the host sends instead of a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    pub body: String,
}

/// `protocol.openai_error_type`: OpenAI's `error.type` for an HTTP status.
fn openai_error_type(code: u16) -> &'static str {
    match code {
        401 => "authentication_error",
        403 => "permission_error",
        429 => "rate_limit_error",
        503 => "service_unavailable_error",
        500.. => "server_error",
        _ => "invalid_request_error",
    }
}

/// `protocol.openai_error_body`.
pub(super) fn error_body(message: &str, code: u16) -> Value {
    json!({"error": {"message": message, "type": openai_error_type(code), "param": null, "code": null}})
}

/// `OpenAIServingBase.create_error_response`.
pub(super) fn error_reply(message: &str, code: u16) -> Reply {
    Reply {
        status: code,
        body: error_body(message, code).to_string(),
    }
}

/// `OpenAIServingBase.create_streaming_error_response`, as an SSE event.
pub(super) fn stream_error_event(message: &str, code: u16) -> String {
    format!(
        "data: {}\n\n",
        python_json(&error_body(message, code), true)
    )
}

/// The OpenAI reply for a `/generate` error response. `None` passes the
/// engine's reply through.
pub(super) fn engine_error_reply(body: &[u8]) -> Option<Reply> {
    let body: Value = serde_json::from_slice(body).ok()?;
    // A ValueError: `/generate` answers `{"error": {"message"}}`, OpenAI a 400.
    if let Some(message) = body.pointer("/error/message").and_then(Value::as_str) {
        return Some(error_reply(message, 400));
    }
    // An `HTTPException`: `/generate` answers the flat `ErrorResponse`, the
    // OpenAI servings the envelope with the same status.
    let code = body["code"]
        .as_u64()
        .and_then(|code| u16::try_from(code).ok())?;
    (body["object"] == "error" && (400..600).contains(&code))
        .then(|| error_reply(body["message"].as_str().unwrap_or_default(), code))
}

/// Python's catch-all 500, when the engine output lacks what the OpenAI layer reads.
pub(super) fn malformed_output_reply() -> Reply {
    error_reply("Internal server error", 500)
}

/// Whether `code` is one of the `HTTPStatus` codes SGLang aborts with.
pub(super) fn is_abort_status(code: u64) -> bool {
    matches!(
        code,
        400 | 401 | 403 | 404 | 408 | 413 | 422 | 429 | 500 | 501 | 502 | 503 | 504
    )
}

/// `UsageProcessor.calculate_token_usage`, dumped as `UsageInfo`.
pub(super) fn usage(prompt: u64, completion: u64, reasoning: u64, cached: Option<u64>) -> Value {
    json!({
        "prompt_tokens": prompt,
        "total_tokens": prompt + completion,
        "completion_tokens": completion,
        "prompt_tokens_details": cached.map(|n| json!({"cached_tokens": n})),
        "reasoning_tokens": reasoning,
    })
}

pub(super) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The non-streaming response over `ret`, the `/generate` items for `choices`.
/// Prompt and cached tokens count once per prompt, i.e. every `n`-th item.
pub(super) fn unary_reply(
    ret: &[Value],
    n: usize,
    cache_report: bool,
    object: &str,
    model: &str,
    choices: Vec<Value>,
) -> Option<Reply> {
    let first = &ret.first()?["meta_info"];
    let stride = |key: &str| -> u64 {
        ret.iter()
            .step_by(n)
            .map(|r| meta_u64(&r["meta_info"], key))
            .sum()
    };
    let all = |key: &str| -> u64 { ret.iter().map(|r| meta_u64(&r["meta_info"], key)).sum() };
    let cached = cache_report.then(|| stride("cached_tokens"));
    let mut metadata = json!({"weight_version": first["weight_version"]});
    if let Some(versions) = first.get("weight_versions") {
        metadata["weight_versions"] = versions.clone();
    }
    let response = json!({
        "id": first["id"],
        "object": object,
        "created": now(),
        "model": model,
        "choices": choices,
        "usage": usage(stride("prompt_tokens"), all("completion_tokens"), all("reasoning_tokens"), cached),
        "metadata": metadata,
    });
    Some(Reply {
        status: 200,
        body: response.to_string(),
    })
}

/// What a stream does with one `/generate` SSE payload.
pub(super) enum Payload {
    Done,
    Frame(Value),
    Events(Vec<String>),
}

/// Stream state both endpoints keep: whether it started, stopped or failed,
/// and each choice's latest token counts.
#[derive(Default)]
pub(super) struct StreamState {
    pub started: bool,
    pub stopped: bool,
    pub failed: bool,
    pub done: bool,
    pub last_id: Value,
    /// Per choice: characters of cumulative text and logprobs already sent.
    pub text_offsets: HashMap<u64, usize>,
    pub logprob_counts: HashMap<u64, usize>,
    prompt: HashMap<u64, u64>,
    completion: HashMap<u64, u64>,
    reasoning: HashMap<u64, u64>,
    cached: HashMap<u64, u64>,
}

impl StreamState {
    /// The checks Python makes before reading a frame. `Err` replaces the
    /// whole stream, before anything was sent.
    pub(super) fn payload(&mut self, data: &[u8]) -> Result<Payload, Reply> {
        if self.done {
            return Ok(Payload::Events(Vec::new()));
        }
        if data == b"[DONE]" {
            return Ok(Payload::Done);
        }
        let Ok(content) = serde_json::from_slice::<Value>(data) else {
            return Ok(Payload::Events(Vec::new()));
        };
        if let Some(message) = content.pointer("/error/message").and_then(Value::as_str) {
            if !self.started {
                return Err(error_reply(message, 400));
            }
            self.failed = true;
            return Ok(Payload::Events(vec![stream_error_event(message, 400)]));
        }
        self.started = true;
        Ok(Payload::Frame(content))
    }

    /// An abort or error just ended the stream; Python stops there, without
    /// waiting for the other choices.
    pub(super) fn ending(&self) -> bool {
        !self.done && (self.stopped || self.failed)
    }

    /// Record the frame's id and choice `index`'s token counts.
    pub(super) fn record(&mut self, index: u64, meta: &Value) {
        self.last_id = meta["id"].clone();
        self.prompt.insert(index, meta_u64(meta, "prompt_tokens"));
        self.completion
            .insert(index, meta_u64(meta, "completion_tokens"));
        self.reasoning
            .insert(index, meta_u64(meta, "reasoning_tokens"));
        self.cached.insert(index, meta_u64(meta, "cached_tokens"));
    }

    /// Choice `index`'s usage, for `continuous_usage_stats`.
    pub(super) fn choice_usage(&self, index: u64, cache_report: bool) -> Value {
        let cached = cache_report.then(|| self.cached[&index]);
        usage(
            self.prompt[&index],
            self.completion[&index],
            self.reasoning[&index],
            cached,
        )
    }

    /// The final usage chunk's; prompt and cached tokens count once per prompt.
    pub(super) fn total_usage(&self, n: usize, cache_report: bool) -> Value {
        let per_prompt = |map: &HashMap<u64, u64>| -> u64 {
            map.iter()
                .filter(|(i, _)| *i % n as u64 == 0)
                .map(|(_, v)| v)
                .sum()
        };
        let cached = cache_report.then(|| per_prompt(&self.cached));
        usage(
            per_prompt(&self.prompt),
            self.completion.values().sum(),
            self.reasoning.values().sum(),
            cached,
        )
    }
}

/// `model_dump_json(exclude_none=True)`: drop `null` fields, recursively.
pub(super) fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, without_nulls(v)))
                .collect::<Map<_, _>>(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

pub(super) fn meta_u64(meta: &Value, key: &str) -> u64 {
    meta.get(key).and_then(Value::as_u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_become_the_envelope_with_their_status() {
        let reply = |body: Value| engine_error_reply(body.to_string().as_bytes());
        let envelope = |message: &str, error_type: &str| json!({"error": {"message": message, "type": error_type, "param": null, "code": null}});
        // A ValueError from `/generate`.
        let value_error = reply(json!({"error": {"message": "bad n"}})).unwrap();
        assert_eq!(value_error.status, 400);
        let body: Value = serde_json::from_str(&value_error.body).unwrap();
        assert_eq!(body, envelope("bad n", "invalid_request_error"));
        // An HTTPException's flat ErrorResponse keeps its status.
        let flat = json!({"object": "error", "message": "The request queue is full.", "type": "503", "param": null, "code": 503});
        let http_exception = reply(flat).unwrap();
        assert_eq!(http_exception.status, 503);
        let body: Value = serde_json::from_str(&http_exception.body).unwrap();
        assert_eq!(
            body,
            envelope("The request queue is full.", "service_unavailable_error")
        );
        // Anything else passes through.
        assert!(reply(json!({"detail": "not found"})).is_none());

        let malformed = malformed_output_reply();
        assert_eq!(malformed.status, 500);
        let body: Value = serde_json::from_str(&malformed.body).unwrap();
        assert_eq!(body, envelope("Internal server error", "server_error"));
    }

    #[test]
    fn cache_report_keeps_zero_cached_tokens() {
        let details = |cached| usage(3, 2, 0, cached)["prompt_tokens_details"].clone();
        assert_eq!(details(Some(0)), json!({"cached_tokens": 0}));
        assert_eq!(details(None), Value::Null);
    }
}
