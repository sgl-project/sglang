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

/// The native finish reason's `matched` value as OpenAI wire JSON: a token id
/// or a string; a multi-token match (native-only) is carried as its id list.
pub(crate) fn matched_stop_value(
    reason: &crate::message::finish_reason::FinishReason,
) -> Option<serde_json::Value> {
    use crate::message::finish_reason::Matched;

    reason.matched().map(|matched| match matched {
        Matched::Token(id) => serde_json::json!(id),
        Matched::Str(value) => serde_json::json!(value),
        Matched::Tokens(ids) => serde_json::json!(ids.ids),
    })
}

/// Usage details Dynamo's `CompletionUsage` cannot express on the wire.
#[derive(Clone, Copy, Default)]
pub(crate) struct UsageDetails {
    pub reasoning_tokens: u32,
    /// Already gated by `--enable-cache-report`; `None` omits the details block.
    pub cached_tokens: Option<u32>,
}

/// Serialize one `CompletionUsage` in Python's `UsageInfo` shape:
/// `reasoning_tokens` is a top-level key (always emitted, 0 allowed) and
/// `prompt_tokens_details.cached_tokens` is present only when reporting is
/// enabled and positive. Dynamo's type nests `reasoning_tokens` and its
/// details block serializes extra null keys, so the keys are injected here.
pub(crate) fn usage_value(
    usage: dynamo_protocols::types::CompletionUsage,
    details: UsageDetails,
) -> serde_json::Value {
    let mut value = serde_json::to_value(usage).expect("usage always serializes");
    let Some(object) = value.as_object_mut() else {
        return value;
    };
    object.insert(
        "reasoning_tokens".into(),
        serde_json::json!(details.reasoning_tokens),
    );
    if let Some(cached_tokens) = details.cached_tokens.filter(|&cached| cached > 0) {
        object.insert(
            "prompt_tokens_details".into(),
            serde_json::json!({"cached_tokens": cached_tokens}),
        );
    }
    value
}

/// The endpoint-visible weight metadata Python builds from `meta_info`
/// (`utils/weight_versions.build_endpoint_weight_version_metadata`): the last
/// span's version, falling back to the launch-time scalar. The span list is
/// present only when the producer sent one.
pub(crate) fn weight_metadata_value(
    spans: Option<&[crate::message::response::WeightVersionSpan]>,
    fallback_version: Option<&str>,
) -> serde_json::Value {
    let version = spans
        .and_then(|spans| spans.last())
        .map(|span| span.version.clone())
        .or_else(|| fallback_version.map(str::to_owned));
    let mut metadata = serde_json::json!({"weight_version": version});
    if let Some(spans) = spans.filter(|spans| !spans.is_empty()) {
        metadata["weight_versions"] =
            serde_json::to_value(spans).expect("weight spans always serialize");
    }
    metadata
}

/// The machine-readable `type` Python's OpenAI errors carry for a status:
/// serving failures default to `BadRequestError`, 5xx to
/// `InternalServerError`, 401 to `AuthenticationError`.
pub(crate) fn error_type(code: u16) -> &'static str {
    if code == 401 {
        "AuthenticationError"
    } else if (500..600).contains(&code) {
        "InternalServerError"
    } else {
        "BadRequestError"
    }
}

/// Python's unary `ErrorResponse` body: a FLAT object whose `object` field is
/// `"error"` (`serving_base.create_error_response` dumps the model directly —
/// only the SSE frame nests under `error`).
pub(crate) fn unary_error_value(code: u16, message: &str, error_type: &str) -> serde_json::Value {
    serde_json::json!({
        "object": "error",
        "message": message,
        "type": error_type,
        "param": null,
        "code": code,
    })
}

/// The OpenAI error payload — the in-band SSE frame shape every
/// OpenAI-compatible surface answers streamed errors with (Python
/// `create_streaming_error_response` nests under `error`).
pub(crate) fn error_payload_value(code: u16, message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message,
            "type": error_type(code),
            "param": null,
            "code": code,
        }
    })
}
