//! Response pieces shared by the OpenAI endpoints, shaped as Python serializes them.

use std::collections::HashMap;

use serde_json::{Map, Value, json};
use sglang_api_types::api::v1::{
    FinishReason, GenerateMetaInfo, GenerateResponse, GenerateStreamError, finish_reason,
};

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

/// A native `/generate` error's message, unary and streamed OpenAI types, and
/// status. Python's `/generate` errors are 400s; another status a host names
/// stays, as Python's `HTTPException` keeps it.
fn engine_error_parts(error: GenerateStreamError) -> (String, String, &'static str, u16) {
    let error = error.error.unwrap_or_default();
    let status = Some(u64::from(error.code))
        .filter(|&code| code != 400)
        .and_then(|code| Some((code as u16, http_status_name(code)?)));
    match status {
        Some((code, name)) => (error.message, code.to_string(), name, code),
        None => (error.message, "BadRequestError".to_owned(), "BadRequestError", 400),
    }
}

/// The OpenAI reply for a native `/generate` error.
pub(super) fn engine_error_reply(error: GenerateStreamError) -> Reply {
    let (message, unary_type, _, code) = engine_error_parts(error);
    error_reply(&message, &unary_type, code)
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
    ret: &[GenerateResponse],
    n: usize,
    cache_report: bool,
    object: &str,
    model: &str,
    choices: Vec<Value>,
) -> Option<Reply> {
    let first = ret.first()?.meta_info.as_ref();
    let sum = |stride: usize, field: fn(&GenerateMetaInfo) -> u64| -> u64 {
        ret.iter()
            .step_by(stride)
            .filter_map(|r| r.meta_info.as_ref())
            .map(field)
            .sum()
    };
    let cached = cache_report.then(|| sum(n, |m| u64::from(m.cached_tokens)));
    let mut metadata = json!({"weight_version": first.and_then(|m| m.weight_version.as_deref())});
    if let Some(meta) = first.filter(|m| !m.weight_versions.is_empty()) {
        metadata["weight_versions"] = json!(meta.weight_versions);
    }
    let response = json!({
        "id": first.map(|m| &m.id),
        "object": object,
        "created": now(),
        "model": model,
        "choices": choices,
        "usage": usage(sum(n, |m| u64::from(m.prompt_tokens)), sum(1, |m| m.completion_tokens), sum(1, |m| m.reasoning_tokens), cached),
        "metadata": metadata,
    });
    Some(Reply {
        status: 200,
        body: response.to_string(),
    })
}

/// Stream state both endpoints keep: whether it started, stopped or failed,
/// and each choice's latest token counts.
#[derive(Default)]
pub(super) struct StreamState {
    pub started: bool,
    pub stopped: bool,
    pub failed: bool,
    pub done: bool,
    pub last_id: String,
    /// Per choice: characters of cumulative text and logprobs already sent.
    pub text_offsets: HashMap<u64, usize>,
    pub logprob_counts: HashMap<u64, usize>,
    prompt: HashMap<u64, u64>,
    completion: HashMap<u64, u64>,
    reasoning: HashMap<u64, u64>,
    cached: HashMap<u64, u64>,
}

impl StreamState {
    /// An engine error replaces the reply until the first chunk has been sent.
    pub(super) fn error(&mut self, error: GenerateStreamError) -> Result<Vec<String>, Reply> {
        let (message, unary_type, event_type, code) = engine_error_parts(error);
        if !self.started {
            // Python's `/generate` streams a scheduler abort as a frame and
            // sends only 400 error frames, so a 500 is a host's own failure:
            // `handle_request`'s generic `InternalServerError`.
            if code == 500 {
                let message = format!("Internal server error: {message}");
                return Err(error_reply(&message, "InternalServerError", code));
            }
            return Err(error_reply(&message, &unary_type, code));
        }
        self.failed = true;
        Ok(vec![stream_error_event(&message, event_type, code)])
    }

    /// An abort or error just ended the stream; Python stops there, without
    /// waiting for the other choices.
    pub(super) fn ending(&self) -> bool {
        !self.done && (self.stopped || self.failed)
    }

    /// Record the frame's id and choice `index`'s token counts.
    pub(super) fn record(&mut self, index: u64, meta: &GenerateMetaInfo) {
        self.last_id.clone_from(&meta.id);
        self.prompt.insert(index, u64::from(meta.prompt_tokens));
        self.completion.insert(index, meta.completion_tokens);
        self.reasoning.insert(index, meta.reasoning_tokens);
        self.cached.insert(index, u64::from(meta.cached_tokens));
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

pub(super) fn finish_fields(reason: Option<&FinishReason>) -> (Option<String>, Value) {
    let Some(reason) = reason else {
        return (None, Value::Null);
    };
    let matched = match reason.kind.as_ref() {
        Some(finish_reason::Kind::Stop(stop)) => json!(stop.matched),
        Some(finish_reason::Kind::Unknown(raw)) => {
            // The shared API retains unknown finish reasons as raw JSON.
            let value: Value = serde_json::from_str(raw).unwrap_or_default();
            return (
                value.get("type").and_then(Value::as_str).map(str::to_owned),
                value.get("matched").cloned().unwrap_or(Value::Null),
            );
        }
        _ => Value::Null,
    };
    (reason.kind_name().map(|kind| kind.into_owned()), matched)
}

pub(super) fn abort_event(reason: Option<&FinishReason>) -> Option<String> {
    let finish_reason::Kind::Abort(abort) = reason?.kind.as_ref()? else {
        return None;
    };
    let code = u64::from(abort.status_code?);
    Some(stream_error_event(
        abort.message.as_deref().unwrap_or("Generation aborted."),
        http_status_name(code)?,
        code as u16,
    ))
}

pub(super) fn logprob_slice<T>(items: &[T], prev: usize, total: usize, incremental: bool) -> &[T] {
    if incremental {
        items
    } else {
        let start = prev.min(items.len());
        &items[start..start + total.saturating_sub(prev).min(items.len() - start)]
    }
}

/// `output_token_logprobs_length`, or, for frames without it, the count the
/// frame's list implies.
pub(super) fn logprobs_length(meta: &GenerateMetaInfo, prev: usize, incremental: bool) -> usize {
    if let Some(length) = meta.output_token_logprobs_length {
        return length as usize;
    }
    let listed = meta
        .output_token_logprobs
        .as_ref()
        .map_or(0, |logprobs| logprobs.entries.len());
    if incremental { prev + listed } else { listed }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_report_keeps_zero_cached_tokens() {
        let details = |cached| usage(3, 2, 0, cached)["prompt_tokens_details"].clone();
        assert_eq!(details(Some(0)), json!({"cached_tokens": 0}));
        assert_eq!(details(None), Value::Null);
    }
}
