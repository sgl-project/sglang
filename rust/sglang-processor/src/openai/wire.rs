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

/// `ErrorResponse(...).model_dump()`.
pub(super) fn error_body(message: &str, err_type: &str, code: u16) -> Value {
    json!({"object": "error", "message": message, "type": err_type, "param": null, "code": code})
}

/// `OpenAIServingBase.create_error_response`.
pub(super) fn error_reply(message: &str, err_type: &str, code: u16) -> Reply {
    Reply {
        status: code,
        body: error_body(message, err_type, code).to_string(),
    }
}

/// `OpenAIServingBase.create_streaming_error_response`, as an SSE event.
pub(super) fn stream_error_event(message: &str, err_type: &str, code: u16) -> String {
    let error = json!({"error": error_body(message, err_type, code)});
    format!("data: {}\n\n", python_json(&error, true))
}

/// The OpenAI reply for a `/generate` error response. `None` passes the
/// engine's reply through: SGLang answers its other errors, such as an
/// `HTTPException`, in the OpenAI error shape on every route.
pub(super) fn engine_error_reply(body: &[u8]) -> Option<Reply> {
    let body: Value = serde_json::from_slice(body).ok()?;
    // A ValueError: `/generate` answers `{"error": {"message"}}`, OpenAI a 400.
    let message = body.pointer("/error/message")?.as_str()?;
    Some(error_reply(message, "BadRequestError", 400))
}

/// Python's answer when the engine output lacks what the OpenAI layer reads.
pub(super) fn malformed_output_reply(missing: &str) -> Reply {
    let message = format!("Internal server error: '{missing}'");
    error_reply(&message, "InternalServerError", 500)
}

/// `HTTPStatus(code).name` for the codes SGLang aborts with.
pub(super) fn http_status_name(code: u64) -> Option<&'static str> {
    Some(match code {
        400 => "BAD_REQUEST",
        401 => "UNAUTHORIZED",
        403 => "FORBIDDEN",
        404 => "NOT_FOUND",
        408 => "REQUEST_TIMEOUT",
        413 => "REQUEST_ENTITY_TOO_LARGE",
        422 => "UNPROCESSABLE_ENTITY",
        429 => "TOO_MANY_REQUESTS",
        500 => "INTERNAL_SERVER_ERROR",
        501 => "NOT_IMPLEMENTED",
        502 => "BAD_GATEWAY",
        503 => "SERVICE_UNAVAILABLE",
        504 => "GATEWAY_TIMEOUT",
        _ => return None,
    })
}

/// `UsageProcessor.calculate_token_usage`, dumped as `UsageInfo`.
pub(super) fn usage(prompt: u64, completion: u64, reasoning: u64, cached: Option<u64>) -> Value {
    json!({
        "prompt_tokens": prompt,
        "total_tokens": prompt + completion,
        "completion_tokens": completion,
        "prompt_tokens_details": cached.filter(|&n| n > 0).map(|n| json!({"cached_tokens": n})),
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
                return Err(error_reply(message, "BadRequestError", 400));
            }
            self.failed = true;
            return Ok(Payload::Events(vec![stream_error_event(
                message,
                "BadRequestError",
                400,
            )]));
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
