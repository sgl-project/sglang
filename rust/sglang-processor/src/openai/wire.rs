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

/// A native `/generate` error's message and status. Python's `/generate`
/// sends no code (a ValueError, so a 400); an engine that sends one keeps its
/// status.
fn engine_error_parts(error: GenerateStreamError) -> (String, u16) {
    let error = error.error.unwrap_or_default();
    let code = u16::try_from(error.code)
        .ok()
        .filter(|code| (400..600).contains(code))
        .unwrap_or(400);
    (error.message, code)
}

/// The OpenAI reply for a native `/generate` error.
pub(super) fn engine_error_reply(error: GenerateStreamError) -> Reply {
    let (message, code) = engine_error_parts(error);
    error_reply(&message, code)
}

/// Python's catch-all 500, when the engine output lacks what the OpenAI layer reads.
pub(super) fn malformed_output_reply() -> Reply {
    error_reply("Internal server error", 500)
}

/// Whether `code` is an `HTTPStatus` error code, as Python 3.12 defines them.
pub(super) fn is_http_error_status(code: u64) -> bool {
    matches!(
        code,
        400..=418 | 421..=426 | 428 | 429 | 431 | 451 | 500..=508 | 510 | 511
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
        // OpenAI's place for the reasoning count, set only when there is some.
        "completion_tokens_details": (reasoning > 0).then(|| json!({"reasoning_tokens": reasoning})),
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
        let (message, code) = engine_error_parts(error);
        if !self.started {
            // Python's `/generate` streams a scheduler abort as a frame and
            // sends only 400 error frames, so a 500 is a host's own failure:
            // `handle_request`'s generic 500, which sends no exception text.
            if code == 500 {
                return Err(error_reply("Internal server error", code));
            }
            return Err(error_reply(&message, code));
        }
        self.failed = true;
        Ok(vec![stream_error_event(&message, code)])
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
    is_http_error_status(code).then(|| {
        stream_error_event(
            abort.message.as_deref().unwrap_or("Generation aborted."),
            code as u16,
        )
    })
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
    fn engine_errors_become_the_envelope_with_their_status() {
        let error = |message: &str, code: u32| GenerateStreamError {
            error: Some(sglang_api_types::api::v1::ErrorBody {
                message: message.into(),
                code,
            }),
            index: None,
        };
        let envelope = |message: &str, error_type: &str| json!({"error": {"message": message, "type": error_type, "param": null, "code": null}});
        let body = |reply: &Reply| serde_json::from_str::<Value>(&reply.body).unwrap();
        // A ValueError from `/generate` carries no code: a 400.
        let value_error = engine_error_reply(error("bad n", 0));
        assert_eq!(value_error.status, 400);
        assert_eq!(
            body(&value_error),
            envelope("bad n", "invalid_request_error")
        );
        // An engine status is kept, and typed by it.
        let unavailable = engine_error_reply(error("The request queue is full.", 503));
        assert_eq!(unavailable.status, 503);
        assert_eq!(
            body(&unavailable),
            envelope("The request queue is full.", "service_unavailable_error")
        );

        // Before the first chunk, a 500 is the generic one, without its text.
        let mut stream = StreamState::default();
        let failed = stream.error(error("boom", 500)).unwrap_err();
        assert_eq!(failed.status, 500);
        assert_eq!(
            body(&failed),
            envelope("Internal server error", "server_error")
        );
        // Mid-stream, the error is an envelope frame typed by its status.
        stream.started = true;
        let events = stream.error(error("Out of memory.", 503)).unwrap();
        assert_eq!(
            events,
            [format!(
                "data: {}\n\n",
                python_json(
                    &envelope("Out of memory.", "service_unavailable_error"),
                    true
                )
            )]
        );

        let malformed = malformed_output_reply();
        assert_eq!(malformed.status, 500);
        assert_eq!(
            body(&malformed),
            envelope("Internal server error", "server_error")
        );
    }

    #[test]
    fn cache_report_keeps_zero_cached_tokens() {
        let details = |cached| usage(3, 2, 0, cached)["prompt_tokens_details"].clone();
        assert_eq!(details(Some(0)), json!({"cached_tokens": 0}));
        assert_eq!(details(None), Value::Null);
    }
}
