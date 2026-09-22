// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat request validation, optional tokenization, and outgoing body preparation.

use crate::config::{ConflictPolicy, ParamSpec, SamplingField, SamplingOverrides};
use crate::discovery::ModelId;
use crate::policies::{has_caller_input_ids, request_tokens_for, RequestTokens};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::MetricsRegistry;
use bytes::Bytes;
use serde::de::IgnoredAny;
use serde::Deserialize;
use serde_json::{json, Number, Value};

/// SGLang upstream's coarse bytes-per-token estimate; only relative load ordering matters.
const BYTES_PER_TOKEN_ESTIMATE: usize = 4;

/// Validated routing inputs and the original body, ready for worker selection.
pub(super) struct PreparedChatRequest {
    pub(super) model: ModelId,
    pub(super) streaming: bool,
    pub(super) max_output_tokens: Option<u64>,
    pub(super) body: Bytes,
    pub(super) tokens: Option<RequestTokens>,
    /// Token count for routing/load accounting; estimated from body size when unavailable.
    pub(super) input_token_count: usize,
    caller_set_rid: bool,
    fans_out: bool,
    can_forward_input_ids: bool,
    parsed_body: Option<Value>,
    sampling_defaults: Vec<(SamplingField, Number)>,
}

impl PreparedChatRequest {
    pub(super) fn prepare(
        ctx: &AppContext,
        model: ModelId,
        fields: RoutingFields,
        body: Bytes,
        policy_needs_request_tokens: bool,
    ) -> Result<Self, ApiError> {
        // Validate configured sampling rules and collect missing defaults for forwarding.
        let sampling_defaults =
            resolve_sampling_defaults(&ctx.config.model.sampling_overrides, &fields, &ctx.metrics)?;
        let can_forward_input_ids = !ctx.config.model.disable_input_ids_forwarding
            && ctx.tokenizers.has_chat_formatter(&model.0);
        let needs_tokens = should_tokenize_request(
            can_forward_input_ids,
            policy_needs_request_tokens,
            ctx.bucket_selector.is_enabled(),
        );
        // Parse the full body only when rendering or routing needs tokens.
        let parsed_body = needs_tokens
            .then(|| serde_json::from_slice(&body))
            .transpose()
            .map_err(|_| invalid_request())?;
        let tokens = parsed_body
            .as_ref()
            .and_then(|parsed_body| request_tokens_for(&ctx.tokenizers, &model, parsed_body));
        // Keep load accounting available even when tokenization is unavailable.
        let input_token_count = tokens
            .as_ref()
            .map(|tokens| tokens.ids.len().max(1))
            .unwrap_or_else(|| estimate_prefill_tokens(&body));
        Ok(Self {
            model,
            streaming: fields.stream.unwrap_or(false),
            max_output_tokens: fields.requested_max_output_tokens(),
            body,
            tokens,
            input_token_count,
            caller_set_rid: fields.caller_set_rid,
            fans_out: requests_multiple_samples(&fields, &sampling_defaults),
            can_forward_input_ids,
            parsed_body,
            sampling_defaults,
        })
    }

    pub(super) fn engine_rid(&self, pd_mode: bool) -> Option<String> {
        // Caller IDs are unsafe for prefix aborts; fan-out regenerates IDs; PD must finish KV transfer.
        if self.caller_set_rid || self.fans_out || pd_mode {
            return None;
        }
        Some(uuid::Uuid::new_v4().simple().to_string())
    }

    pub(super) fn into_outgoing_body(
        self,
        ctx: &AppContext,
        bootstrap: Option<&BootstrapFields>,
        engine_rid: Option<&str>,
    ) -> Result<Bytes, ApiError> {
        // Routing tokens can replace engine tokenization only for supported chat templates.
        let input_ids = match (self.tokens.as_ref(), self.parsed_body.as_ref()) {
            (Some(tokens), Some(parsed_body))
                if self.can_forward_input_ids
                    && tokens.rendered_from_chat
                    && can_forward_chat_tokens(parsed_body) =>
            {
                Some(tokens.ids.as_slice())
            }
            _ => None,
        };
        if chat_tokenization_failed(
            self.can_forward_input_ids,
            self.parsed_body.as_ref(),
            self.tokens.as_ref(),
        ) {
            ctx.metrics.record_ingress_tokenize_error(&self.model.0);
        }
        build_outgoing_body(
            &self.body,
            self.parsed_body,
            input_ids,
            bootstrap,
            &self.sampling_defaults,
            engine_rid,
        )
    }
}

/// Routing and sampling fields retained by the lightweight request parser.
#[derive(Debug, Default)]
pub(super) struct RoutingFields {
    stream: Option<bool>,
    pub(super) model: Option<String>,
    max_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    sampling: [SamplingValue; SamplingField::ALL.len()],
    // Preserve both string and list IDs without retaining their contents.
    caller_set_rid: bool,
}

/// Null is absent; unrepresentable values are rejected only under a sampling contract.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum SamplingValue {
    #[default]
    Absent,
    Number(f64),
    Unusable,
}

/// Cap numeric-string parsing and its stack buffer at 64 bytes.
const MAX_SAMPLING_NUMERIC_LEN: usize = 64;

/// Match engine coercion: trim ordinary numbers, but preserve whitespace for underscored ones.
fn parse_engine_numeric_string(input: &str) -> Option<f64> {
    let trimmed = input.trim();
    if trimmed.len() > MAX_SAMPLING_NUMERIC_LEN {
        return None;
    }
    if let Ok(number) = trimmed.parse::<f64>() {
        return Some(number);
    }
    if input.len() > MAX_SAMPLING_NUMERIC_LEN
        || !input.contains('_')
        || input.starts_with('_')
        || input.ends_with('_')
        || input.contains("__")
    {
        return None;
    }
    // Bound numeric parsing and keep underscore removal on the stack.
    let mut normalized = [0u8; MAX_SAMPLING_NUMERIC_LEN];
    let mut length = 0;
    for &byte in input.as_bytes() {
        if byte != b'_' {
            normalized[length] = byte;
            length += 1;
        }
    }
    std::str::from_utf8(&normalized[..length])
        .ok()?
        .parse()
        .ok()
}

impl<'de> Deserialize<'de> for SamplingValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> serde::de::Visitor<'de> for ValueVisitor {
            type Value = SamplingValue;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sampling parameter value")
            }

            fn visit_i64<E>(self, v: i64) -> Result<SamplingValue, E> {
                Ok(SamplingValue::Number(v as f64))
            }

            fn visit_u64<E>(self, v: u64) -> Result<SamplingValue, E> {
                Ok(SamplingValue::Number(v as f64))
            }

            fn visit_f64<E>(self, v: f64) -> Result<SamplingValue, E> {
                Ok(SamplingValue::Number(v))
            }

            fn visit_str<E>(self, v: &str) -> Result<SamplingValue, E> {
                Ok(parse_engine_numeric_string(v)
                    .map_or(SamplingValue::Unusable, SamplingValue::Number))
            }

            fn visit_unit<E>(self) -> Result<SamplingValue, E> {
                Ok(SamplingValue::Absent)
            }

            fn visit_bool<E>(self, v: bool) -> Result<SamplingValue, E> {
                Ok(SamplingValue::Number(if v { 1.0 } else { 0.0 }))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<SamplingValue, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(SamplingValue::Unusable)
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<SamplingValue, M::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(SamplingValue::Unusable)
            }
        }
        d.deserialize_any(ValueVisitor)
    }
}

#[derive(Debug, Clone, Copy)]
enum RoutingKey {
    Stream,
    Model,
    MaxTokens,
    MaxCompletionTokens,
}

impl RoutingKey {
    const fn bit(self) -> u8 {
        match self {
            Self::Stream => 1 << 0,
            Self::Model => 1 << 1,
            Self::MaxTokens => 1 << 2,
            Self::MaxCompletionTokens => 1 << 3,
        }
    }

    const fn wire_name(self) -> &'static str {
        match self {
            Self::Stream => "stream",
            Self::Model => "model",
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

enum RequestKey {
    Routing(RoutingKey),
    Sampling(SamplingField),
    Rid,
    Other,
}

impl<'de> Deserialize<'de> for RequestKey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl serde::de::Visitor<'_> for KeyVisitor {
            type Value = RequestKey;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a request field name")
            }

            fn visit_str<E>(self, v: &str) -> Result<RequestKey, E> {
                Ok(match v {
                    "stream" => RequestKey::Routing(RoutingKey::Stream),
                    "model" => RequestKey::Routing(RoutingKey::Model),
                    "max_tokens" => RequestKey::Routing(RoutingKey::MaxTokens),
                    "max_completion_tokens" => RequestKey::Routing(RoutingKey::MaxCompletionTokens),
                    "rid" => RequestKey::Rid,
                    other => match SamplingField::from_wire_name(other) {
                        Some(field) => RequestKey::Sampling(field),
                        None => RequestKey::Other,
                    },
                })
            }
        }
        d.deserialize_str(KeyVisitor)
    }
}

struct RoutingFieldsVisitor {
    read_sampling: bool,
}

impl<'de> serde::de::Visitor<'de> for RoutingFieldsVisitor {
    type Value = RoutingFields;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON object")
    }

    fn visit_map<M: serde::de::MapAccess<'de>>(
        self,
        mut map: M,
    ) -> Result<RoutingFields, M::Error> {
        let mut fields = RoutingFields::default();
        let mut seen_routing_keys = 0u8;
        while let Some(key) = map.next_key::<RequestKey>()? {
            match key {
                RequestKey::Routing(field) => {
                    if seen_routing_keys & field.bit() != 0 {
                        return Err(serde::de::Error::custom(format_args!(
                            "duplicate field `{}`",
                            field.wire_name()
                        )));
                    }
                    // Track keys separately so even a repeated null is rejected.
                    seen_routing_keys |= field.bit();
                    match field {
                        RoutingKey::Stream => fields.stream = map.next_value()?,
                        RoutingKey::Model => fields.model = map.next_value()?,
                        RoutingKey::MaxTokens => fields.max_tokens = map.next_value()?,
                        RoutingKey::MaxCompletionTokens => {
                            fields.max_completion_tokens = map.next_value()?
                        }
                    }
                }
                RequestKey::Sampling(field) => {
                    // Match the engine's last-value-wins behavior for sampling fields.
                    fields.sampling[field.index()] = if self.read_sampling {
                        map.next_value()?
                    } else {
                        map.next_value::<IgnoredAny>()?;
                        SamplingValue::Unusable
                    };
                }
                RequestKey::Rid => {
                    fields.caller_set_rid = map.next_value::<Option<IgnoredAny>>()?.is_some();
                }
                RequestKey::Other => {
                    // Validate unrelated JSON without retaining its contents.
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(fields)
    }
}

impl<'de> Deserialize<'de> for RoutingFields {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(RoutingFieldsVisitor {
            read_sampling: true,
        })
    }
}

/// Preserve valid JSON with sampling numbers outside f64 range (for example, 1e400).
fn parse_without_sampling_values(body: &[u8]) -> Result<RoutingFields, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_slice(body);
    let fields = serde::Deserializer::deserialize_map(
        &mut de,
        RoutingFieldsVisitor {
            read_sampling: false,
        },
    )?;
    de.end()?;
    Ok(fields)
}

impl RoutingFields {
    fn requested_max_output_tokens(&self) -> Option<u64> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    fn sampling_field(&self, field: SamplingField) -> SamplingValue {
        self.sampling[field.index()]
    }
}

/// Tokens support engine offload, cache-aware policies, and bucket size checks.
fn should_tokenize_request(
    can_forward_input_ids: bool,
    policy_needs_request_tokens: bool,
    bucket_routing_enabled: bool,
) -> bool {
    can_forward_input_ids || policy_needs_request_tokens || bucket_routing_enabled
}

// Use the effective n, including injected defaults; unreadable values opt out.
fn requests_multiple_samples(
    fields: &RoutingFields,
    sampling_defaults: &[(SamplingField, Number)],
) -> bool {
    match fields.sampling_field(SamplingField::N) {
        SamplingValue::Number(n) => n > 1.0,
        SamplingValue::Unusable => true,
        SamplingValue::Absent => sampling_defaults
            .iter()
            .find(|(field, _)| *field == SamplingField::N)
            .and_then(|(_, value)| value.as_f64())
            .is_some_and(|n| n > 1.0),
    }
}

fn estimate_prefill_tokens(body: &Bytes) -> usize {
    // Never 0: a zero-load entry is invisible to the cache-aware imbalance fast path.
    (body.len() / BYTES_PER_TOKEN_ESTIMATE).max(1)
}

/// The engine stores bootstrap rooms as signed int64 values.
pub(super) fn generate_room_id() -> u64 {
    rand::random::<u64>() & (i64::MAX as u64)
}

pub(super) struct BootstrapFields {
    pub(super) host: String,
    pub(super) port: Option<u16>,
    pub(super) room: u64,
}

/// Append before the closing brace so injected values win over explicit nulls.
fn append_top_level_fields(
    body: &Bytes,
    sampling_defaults: &[(SamplingField, Number)],
    rid: Option<&str>,
) -> Option<Bytes> {
    use std::io::Write as _;

    let open = body.iter().position(|&b| b == b'{')?;
    let close = body.iter().rposition(|&b| b == b'}')?;
    if close <= open {
        return None;
    }
    let has_members = body[open + 1..close]
        .iter()
        .any(|b| !b.is_ascii_whitespace());
    let rid_budget = rid.map_or(0, |rid| rid.len() + ",\"rid\":\"\"".len());
    let mut output = Vec::with_capacity(body.len() + 24 * sampling_defaults.len() + rid_budget + 1);
    output.extend_from_slice(&body[..close]);
    let mut wrote_any = has_members;
    for (field, value) in sampling_defaults {
        if wrote_any {
            output.push(b',');
        }
        write!(output, "\"{}\":{}", field.wire_name(), value).ok()?;
        wrote_any = true;
    }
    if let Some(rid) = rid {
        if wrote_any {
            output.push(b',');
        }
        output.extend_from_slice(b"\"rid\":");
        serde_json::to_writer(&mut output, rid).ok()?;
    }
    output.extend_from_slice(&body[close..]);
    Some(Bytes::from(output))
}

/// Preserve original bytes where possible; reuse parsed JSON for token or bootstrap injection.
fn build_outgoing_body(
    body: &Bytes,
    parsed_body: Option<Value>,
    input_ids: Option<&[u32]>,
    bootstrap: Option<&BootstrapFields>,
    sampling_defaults: &[(SamplingField, Number)],
    rid: Option<&str>,
) -> Result<Bytes, ApiError> {
    let needs_parse = input_ids.is_some() || bootstrap.is_some();
    if !needs_parse && sampling_defaults.is_empty() && rid.is_none() {
        // Cloning Bytes shares the original allocation when no injection is needed.
        return Ok(body.clone());
    }
    if !needs_parse {
        if let Some(spliced) = append_top_level_fields(body, sampling_defaults, rid) {
            return Ok(spliced);
        }
    }
    let parsed = match parsed_body {
        Some(cached_body) => cached_body,
        None => serde_json::from_slice(body).map_err(|_| invalid_request())?,
    };
    let mut body_fields = match parsed {
        Value::Object(map) => map,
        _ => {
            return Err(invalid_request());
        }
    };
    if let Some(rid) = rid {
        body_fields.insert("rid".into(), Value::String(rid.to_owned()));
    }
    for (field, default) in sampling_defaults {
        body_fields.insert(field.wire_name().into(), default.clone().into());
    }
    if let Some(token_ids) = input_ids {
        body_fields.insert("input_ids".into(), json!(token_ids));
    }
    if let Some(bootstrap) = bootstrap {
        body_fields.insert("bootstrap_host".into(), json!(bootstrap.host));
        // A missing port must be JSON null, not omitted; the engine's validator distinguishes them.
        body_fields.insert("bootstrap_port".into(), json!(bootstrap.port));
        body_fields.insert("bootstrap_room".into(), json!(bootstrap.room));
    }
    let bytes = serde_json::to_vec(&body_fields).map_err(|e| {
        ApiError::Internal(anyhow::Error::new(e).context("re-serialize injected request body"))
    })?;
    Ok(Bytes::from(bytes))
}

/// Forward generated IDs only for request shapes verified against the engine.
/// The engine uses `input_ids` verbatim, bypassing its chat-template processing.
///
/// Preserve caller-provided IDs. Exclude requests that may render differently
/// with dynamo-render:
/// - Non-leading system turns or consecutive users, which strict templates rewrite.
/// - Historical `reasoning_content`, which may be injected into message content.
/// - Tools and tool-call history, which the engine merges and normalizes
///   before rendering.
/// - Non-string or missing content, which the engine flattens or blanks.
/// - Template overrides, kwargs, reasoning controls, or task selection.
/// - Assistant continuations, whose final turn the engine handles separately.
///
/// Matching model files and engine defaults are still required. Worker template
/// overrides and default kwargs cannot be inferred from the request.
/// `--disable-input-ids-forwarding` gates forwarding separately for such fleets.
fn can_forward_chat_tokens(value: &Value) -> bool {
    if has_caller_input_ids(value)
        || request_has_tools(value)
        || request_has_non_text_content(value)
        || request_has_reasoning_content(value)
        || request_has_role_rewrites(value)
    {
        return false;
    }
    // Request controls whose rendering has not been verified against the engine.
    for key in [
        "chat_template",
        "chat_template_kwargs",
        "reasoning",
        "reasoning_effort",
        "task",
    ] {
        if value.get(key).is_some_and(|v| !v.is_null()) {
            return false;
        }
    }
    if value
        .get("continue_final_message")
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return false;
    }
    !last_message_is_assistant(value)
}

/// Whether to increment `sgl_router_ingress_tokenize_errors_total`.
///
/// Count chats with forwarding enabled that pass the forwarding guard
/// but lack chat-rendered tokens. Excluded requests are expected fallbacks,
/// even when rendering fails.
fn chat_tokenization_failed(
    can_forward_input_ids: bool,
    request_value: Option<&Value>,
    request_tokens: Option<&RequestTokens>,
) -> bool {
    if !can_forward_input_ids {
        return false;
    }
    let chat_request = request_value.is_some_and(|v| {
        v.get("messages").is_some_and(|m| m.is_array()) && can_forward_chat_tokens(v)
    });
    if !chat_request {
        return false;
    }
    !request_tokens.is_some_and(|t| t.rendered_from_chat)
}

/// Whether the final chat message has `role: "assistant"` (a prefix /
/// continuation turn the engine's template path special-cases).
fn last_message_is_assistant(value: &Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        == Some("assistant")
}

/// Tool schemas and tool-call history require engine normalization before
/// rendering: the engine merges message-level `tools` into the template's tools
/// and parses `tool_calls` arguments; dynamo-render does neither the same way.
fn request_has_tools(value: &Value) -> bool {
    let nonempty = |v: &Value| match v {
        Value::Array(a) => !a.is_empty(),
        Value::Null => false,
        _ => true,
    };
    if ["tools", "functions"]
        .iter()
        .any(|key| value.get(key).is_some_and(nonempty))
    {
        return true;
    }
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message["role"] == "tool"
                    || ["tools", "tool_calls", "function_call"]
                        .iter()
                        .any(|key| message.get(key).is_some_and(nonempty))
            })
        })
}

/// dynamo-render may inject historical reasoning into content the engine leaves unchanged.
fn request_has_reasoning_content(value: &Value) -> bool {
    value
        .get("messages")
        .and_then(|messages| messages.as_array())
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message
                    .get("reasoning_content")
                    .is_some_and(|v| !v.is_null())
            })
        })
}

/// Message orders dynamo-render may rewrite for strict templates.
fn request_has_role_rewrites(value: &Value) -> bool {
    let Some(messages) = value.get("messages").and_then(|v| v.as_array()) else {
        return false;
    };
    messages.iter().skip(1).any(|m| m["role"] == "system")
        || messages
            .windows(2)
            .any(|pair| pair[0]["role"] == "user" && pair[1]["role"] == "user")
}

/// Detect non-string or missing content, which requires engine tokenization:
/// the engine normalizes arrays and nulls differently from dynamo-render.
fn request_has_non_text_content(value: &Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|msgs| {
            msgs.iter()
                .any(|m| !matches!(m.get("content"), Some(Value::String(_))))
        })
}

/// Validate supplied values and return exact defaults for missing or null parameters.
fn resolve_sampling_defaults(
    overrides: &SamplingOverrides,
    fields: &RoutingFields,
    metrics: &MetricsRegistry,
) -> Result<Vec<(SamplingField, Number)>, ApiError> {
    let mut defaults = Vec::with_capacity(overrides.params.len());
    // Count every violated parameter, but report only the first to the client.
    let mut first_violation: Option<ApiError> = None;
    for (&field, spec) in &overrides.params {
        let provided = fields.sampling_field(field);
        if provided == SamplingValue::Absent {
            if let ParamSpec::Exact(value) = spec {
                defaults.push((field, value.clone()));
            }
            continue;
        }
        if overrides.conflict == ConflictPolicy::Allow {
            continue;
        }
        if let Some(detail) = sampling_violation(spec, provided) {
            let param = field.wire_name();
            metrics.record_sampling_contract_rejection(param);
            first_violation.get_or_insert(ApiError::SamplingContract { param, detail });
        }
    }
    match first_violation {
        Some(err) => Err(err),
        None => Ok(defaults),
    }
}

fn sampling_violation(spec: &ParamSpec, provided: SamplingValue) -> Option<String> {
    match (spec, provided) {
        (ParamSpec::Exact(expected), SamplingValue::Unusable) => Some(format!(
            "expected {expected} (or omit the field), got a non-numeric value"
        )),
        (ParamSpec::Range { lo, hi }, SamplingValue::Unusable) => Some(format!(
            "must be a number between {lo} and {hi}, got a non-numeric value"
        )),
        (ParamSpec::Exact(expected), SamplingValue::Number(provided))
            if Some(provided) != expected.as_f64() =>
        {
            Some(format!(
                "got {provided}, expected {expected} (or omit the field)"
            ))
        }
        (&ParamSpec::Range { lo, hi }, SamplingValue::Number(provided))
            if !(lo..=hi).contains(&provided) =>
        {
            Some(format!("must be between {lo} and {hi}, got {provided}"))
        }
        _ => None,
    }
}

pub(super) fn parse_routing_fields(body: &Bytes) -> Result<RoutingFields, ApiError> {
    let err = match serde_json::from_slice::<RoutingFields>(body) {
        Ok(fields) => return Ok(fields),
        Err(e) => e,
    };
    // Retry without numeric conversion so sampling rules can handle values such as 1e400.
    if let Ok(fields) = parse_without_sampling_values(body) {
        return Ok(fields);
    }
    // Keep deserialization details out of the client-visible error.
    tracing::debug!(error = %err, "chat-completions routing-fields deserialize failed");
    Err(invalid_request())
}

fn invalid_request() -> ApiError {
    ApiError::BadRequest("invalid request: body must be a JSON object".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn bucket_routing_requests_tokens_even_for_a_non_token_policy() {
        assert!(should_tokenize_request(false, false, true));
        assert!(!should_tokenize_request(false, false, false));
    }

    #[test]
    fn generate_room_id_stays_in_63_bit_range() {
        for _ in 0..10_000 {
            let r = generate_room_id();
            assert!(
                r <= i64::MAX as u64,
                "generate_room_id() returned {r} > i64::MAX; would wrap negative as torch.int64",
            );
        }
    }

    #[test]
    fn outgoing_body_injects_tokens_and_bootstrap_without_losing_messages() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let original: Value = serde_json::from_slice(&body).unwrap();
        let bootstrap = |port| BootstrapFields {
            host: "h".into(),
            port,
            room: 42,
        };
        for (ids, bootstrap, fields) in [
            (Some(&[1, 2, 3][..]), None, json!({"input_ids": [1, 2, 3]})),
            (
                None,
                Some(bootstrap(None)),
                json!({"bootstrap_host": "h", "bootstrap_port": null, "bootstrap_room": 42}),
            ),
            (
                None,
                Some(bootstrap(Some(9))),
                json!({"bootstrap_host": "h", "bootstrap_port": 9, "bootstrap_room": 42}),
            ),
            (
                Some(&[7, 8][..]),
                Some(bootstrap(Some(9))),
                json!({"input_ids": [7, 8], "bootstrap_host": "h", "bootstrap_port": 9, "bootstrap_room": 42}),
            ),
        ] {
            let mut expected = original.clone();
            expected
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            for value in [None, Some(original.clone())] {
                let out =
                    build_outgoing_body(&body, value, ids, bootstrap.as_ref(), &[], None).unwrap();
                assert_eq!(serde_json::from_slice::<Value>(&out).unwrap(), expected);
            }
        }
    }

    #[test]
    fn outgoing_body_without_injection_reuses_original_bytes() {
        for raw in [r#"{"model":"x"}"#, r#"{"model":"x","messages":[]}"#] {
            let body = Bytes::copy_from_slice(raw.as_bytes());
            for value in [None, Some(serde_json::from_slice(&body).unwrap())] {
                let out = build_outgoing_body(&body, value, None, None, &[], None).unwrap();
                assert_eq!(out, body);
                assert_eq!(out.as_ptr(), body.as_ptr());
            }
        }
    }

    #[test]
    fn request_has_tools_detects_tools_and_functions() {
        assert!(request_has_tools(&json!({"tools":[{"type":"function"}]})));
        assert!(request_has_tools(&json!({"functions":[{"name":"f"}]})));
        assert!(!request_has_tools(&json!({"tools":[]})));
        assert!(!request_has_tools(&json!({"messages":[]})));
        for message in [
            json!({"role":"system","content":"s","tools":[{"type":"function"}]}),
            json!({"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"{}"}}]}),
        ] {
            assert!(request_has_tools(&json!({"messages":[message]})));
        }
    }

    #[test]
    fn request_has_non_text_content_detects_non_string_content() {
        for content in [
            json!([{"type":"image_url","image_url":"x"}]),
            json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]),
            Value::Null,
        ] {
            assert!(
                request_has_non_text_content(&json!({
                    "messages":[{"role":"user","content":"hi"},{"role":"assistant","content":content}]
                })),
                "content {content} must block"
            );
        }
        assert!(request_has_non_text_content(&json!({
            "messages":[{"role":"assistant","tool_calls":[]}]
        })));
        assert!(!request_has_non_text_content(&json!({
            "messages":[{"role":"user","content":"hello"}]
        })));
    }

    #[test]
    fn reasoning_history_is_an_expected_forwarding_omission() {
        let mut value = json!({"messages": [
            {"role":"user", "content":"hi"},
            {"role":"assistant", "content":"answer", "reasoning_content":"prior reasoning"},
            {"role":"user", "content":"next"}
        ]});
        assert!(!can_forward_chat_tokens(&value));
        assert!(!chat_tokenization_failed(true, Some(&value), None));
        value["messages"][1]["reasoning_content"] = Value::Null;
        assert!(can_forward_chat_tokens(&value));
        value["messages"][1]
            .as_object_mut()
            .unwrap()
            .remove("reasoning_content");
        assert!(can_forward_chat_tokens(&value));
    }

    #[test]
    fn role_rewrites_are_expected_forwarding_omissions() {
        for roles in [
            vec!["user", "user"],
            vec!["system", "system", "user"],
            vec!["user", "assistant", "system", "user"],
        ] {
            let messages: Vec<_> = roles
                .iter()
                .map(|role| json!({"role": role, "content": "text"}))
                .collect();
            let value = json!({"messages": messages});
            assert!(!can_forward_chat_tokens(&value), "{roles:?}");
            assert!(!chat_tokenization_failed(true, Some(&value), None));
        }
        assert!(can_forward_chat_tokens(&json!({"messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "next"}
        ]})));
    }

    #[test]
    fn can_forward_chat_tokens_allows_plain_text_chat() {
        assert!(can_forward_chat_tokens(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        })));
    }

    #[test]
    fn can_forward_chat_tokens_blocks_unreplicated_signals() {
        let blockers = [
            json!({"messages":[{"role":"user","content":"hi"}],"input_ids":[7, 8]}),
            json!({"messages":[{"role":"user","content":"hi"}],"input_ids":"bad"}),
            json!({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}),
            json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}),
            json!({"messages":[{"role":"user","content":"hi"}],"chat_template":"{{ custom }}"}),
            json!({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":true}}),
            json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}),
            json!({"messages":[{"role":"user","content":"hi"}],"reasoning":{"enabled":true}}),
            json!({"messages":[{"role":"user","content":"hi"}],"task":"generate"}),
            json!({"messages":[{"role":"user","content":"hi"}],"continue_final_message":true}),
            json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"partial"}]}),
        ];
        for b in blockers {
            assert!(
                !can_forward_chat_tokens(&b),
                "must NOT forward input_ids for: {b}"
            );
        }
    }

    #[test]
    fn can_forward_chat_tokens_ignores_null_and_false_fields() {
        assert!(can_forward_chat_tokens(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "input_ids": null,
            "chat_template": null,
            "reasoning_effort": null,
            "chat_template_kwargs": null,
            "continue_final_message": false
        })));
    }

    #[test]
    fn offload_failures_only_count_forwardable_chats_missing_rendered_tokens() {
        let chat = json!({"messages":[{"role":"user","content":"hi"}]});
        let tools = json!({"messages":[{"role":"user","content":"hi"}], "tools":[{"type":"function","function":{"name":"f"}}]});
        let prompt = json!({"prompt":"hi"});
        for (formatter, value, rendered, failed) in [
            (true, Some(&chat), Some(true), false),
            (true, Some(&chat), Some(false), true),
            (true, Some(&chat), None, true),
            (false, Some(&chat), None, false),
            (true, Some(&tools), None, false),
            (true, Some(&prompt), None, false),
            (true, None, None, false),
        ] {
            let tokens = rendered.map(|rendered_from_chat| RequestTokens {
                ids: vec![1, 2, 3],
                rendered_from_chat,
            });
            assert_eq!(
                chat_tokenization_failed(formatter, value, tokens.as_ref()),
                failed,
                "formatter={formatter}, request={value:?}, rendered={rendered:?}"
            );
        }
    }

    #[test]
    fn routing_fields_parse_with_nested_messages() {
        for stream in [true, false] {
            for messages in [
                json!([]),
                json!([{"role":"user","content":[{"type":"text","text":"hi"}]}]),
            ] {
                let body =
                    json!({"model":"tiny", "stream":stream, "messages":messages}).to_string();
                let fields = fields_of(&body);
                assert_eq!(fields.stream, Some(stream));
                assert_eq!(fields.model.as_deref(), Some("tiny"));
                assert_eq!(fields.requested_max_output_tokens(), None);
            }
        }
        let fields = fields_of(r#"{"model":"tiny","messages":[]}"#);
        assert_eq!(fields.stream, None);
        assert_eq!(fields.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn routing_fields_prefer_modern_output_budget() {
        for body in [
            r#"{"model":"tiny","messages":[],"max_completion_tokens":256}"#,
            r#"{"model":"tiny","max_tokens":128,"max_completion_tokens":256}"#,
        ] {
            assert_eq!(fields_of(body).requested_max_output_tokens(), Some(256));
        }
    }

    #[test]
    fn routing_fields_reject_invalid_requests_without_exposing_parser_details() {
        for body in [
            "null",
            "[]",
            r#""hi""#,
            "42",
            "{not json}",
            r#"{"stream":"not-a-bool"}"#,
            r#"{"stream":true,"stream":false}"#,
        ] {
            let error = parse_routing_fields(&Bytes::copy_from_slice(body.as_bytes())).unwrap_err();
            assert!(
                matches!(error, ApiError::BadRequest(ref message)
                if message == "invalid request: body must be a JSON object"),
                "{body}: {error:?}"
            );
        }
    }

    fn fields_of(body: &str) -> RoutingFields {
        parse_routing_fields(&Bytes::copy_from_slice(body.as_bytes())).unwrap()
    }

    fn overrides_of(conflict: ConflictPolicy, json: &str) -> SamplingOverrides {
        crate::config::parse_sampling_overrides(json, conflict).expect("test config must parse")
    }

    fn metrics() -> Arc<MetricsRegistry> {
        MetricsRegistry::new()
    }

    fn assert_rejections(metrics: &MetricsRegistry, param: &str, count: usize) {
        let rendered = metrics.render();
        let expected =
            format!(r#"sgl_router_sampling_contract_rejections_total{{param="{param}"}} {count}"#);
        assert!(
            rendered.contains(&expected),
            "missing {expected}:\n{rendered}"
        );
    }

    #[test]
    fn unconfigured_sampling_overrides_inject_nothing() {
        let overrides = SamplingOverrides::default();
        let p = fields_of(r#"{"model":"x","temperature":0.7,"n":4}"#);
        assert_eq!(
            resolve_sampling_defaults(&overrides, &p, &metrics()).unwrap(),
            vec![]
        );
    }

    fn reject_overrides() -> SamplingOverrides {
        overrides_of(
            ConflictPolicy::Reject,
            r#"{"top_p": 0.95, "frequency_penalty": 0.0, "presence_penalty": 0.0,
                "n": 1, "temperature": {"min": 0, "max": 1}}"#,
        )
    }

    #[test]
    fn reject_mode_injects_missing_exact_values_but_not_ranges() {
        let inject = resolve_sampling_defaults(
            &reject_overrides(),
            &fields_of(r#"{"model":"x","messages":[]}"#),
            &metrics(),
        )
        .unwrap();
        assert_eq!(
            inject
                .iter()
                .map(|(f, v)| (f.wire_name(), v.to_string()))
                .collect::<Vec<_>>(),
            vec![
                ("top_p", "0.95".to_string()),
                ("frequency_penalty", "0.0".to_string()),
                ("presence_penalty", "0.0".to_string()),
                ("n", "1".to_string()),
            ]
        );
    }

    #[test]
    fn reject_mode_pins_configured_values_and_admits_a_band() {
        let overrides = reject_overrides();
        for (field, value, accepted) in [
            ("temperature", "0.0", true),
            ("temperature", "0.6", true),
            ("temperature", "1.0", true),
            ("temperature", "1.1", false),
            ("temperature", "2.0", false),
            ("temperature", "-0.1", false),
            ("top_p", "0.95", true),
            ("top_p", r#""0.95""#, true),
            ("top_p", "0.8", false),
            ("top_p", r#""0.8""#, false),
            ("top_p", r#""hot""#, false),
            ("presence_penalty", "0", true),
            ("presence_penalty", "0.5", false),
            ("frequency_penalty", "0", true),
            ("frequency_penalty", "0.5", false),
            ("n", "1", true),
            ("n", "true", true),
            ("n", "2", false),
        ] {
            let body = format!(r#"{{"model":"x","{field}":{value}}}"#);
            let result = resolve_sampling_defaults(&overrides, &fields_of(&body), &metrics());
            if accepted {
                let inject = result.unwrap_or_else(|error| panic!("{body}: {error:?}"));
                assert!(
                    !inject.iter().any(|(f, _)| f.wire_name() == field),
                    "{body}"
                );
            } else {
                assert!(
                    matches!(result, Err(ApiError::SamplingContract { .. })),
                    "{body}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn exact_temperature_rejects_conflicts_and_injects_when_absent() {
        let exact = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#);
        let p = fields_of(r#"{"model":"x","temperature":0.6}"#);
        assert!(resolve_sampling_defaults(&exact, &p, &metrics()).is_err());
        assert_eq!(
            resolve_sampling_defaults(&exact, &fields_of(r#"{"model":"x"}"#), &metrics()).unwrap(),
            vec![(SamplingField::Temperature, Number::from_f64(1.0).unwrap())]
        );
    }

    #[test]
    fn allow_mode_only_injects_missing_or_null_parameters() {
        let overrides = overrides_of(
            ConflictPolicy::Allow,
            r#"{"temperature":1,"top_p":0.95,"n":1}"#,
        );
        for (body, fields) in [
            (r#"{"model":"x"}"#, vec!["temperature", "top_p", "n"]),
            (r#"{"temperature":0.6,"top_p":0.8,"n":4}"#, vec![]),
            (r#"{"temperature":0.6}"#, vec!["top_p", "n"]),
            (
                r#"{"temperature":null,"top_p":0.95,"n":1}"#,
                vec!["temperature"],
            ),
            (r#"{"temperature":"abc","top_p":[1],"n":1}"#, vec![]),
        ] {
            let inject =
                resolve_sampling_defaults(&overrides, &fields_of(body), &metrics()).unwrap();
            assert_eq!(
                inject
                    .iter()
                    .map(|(f, _)| f.wire_name())
                    .collect::<Vec<_>>(),
                fields,
                "{body}"
            );
        }
    }

    #[test]
    fn outgoing_body_combines_sampling_defaults_and_input_ids() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let overrides = overrides_of(
            ConflictPolicy::Reject,
            r#"{"top_p":0.95,"top_k":1000,"frequency_penalty":0.0,"presence_penalty":0.0,"n":1}"#,
        );
        let inject = resolve_sampling_defaults(
            &overrides,
            &parse_routing_fields(&body).unwrap(),
            &metrics(),
        )
        .unwrap();
        let out = build_outgoing_body(&body, None, Some(&[1, 2, 3]), None, &inject, None).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&out).unwrap(),
            json!({
                "model":"x", "messages":[{"role":"user","content":"hi"}],
                "input_ids":[1,2,3], "top_p":0.95, "top_k":1000,
                "frequency_penalty":0.0, "presence_penalty":0.0, "n":1
            })
        );
    }

    #[test]
    fn duplicate_sampling_key_takes_the_last_value_like_the_engine() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);

        let p = fields_of(r#"{"model":"x","temperature":0.5,"temperature":1}"#);
        assert!(resolve_sampling_defaults(&overrides, &p, &metrics()).is_ok());

        let p = fields_of(r#"{"model":"x","temperature":1,"temperature":0.5}"#);
        let err = resolve_sampling_defaults(&overrides, &p, &metrics()).unwrap_err();
        assert!(
            format!("{err}").contains("got 0.5"),
            "must judge the last value; got {err}"
        );
    }

    #[test]
    fn duplicate_routing_key_still_rejects() {
        for body in [
            r#"{"model":"a","model":"b"}"#,
            r#"{"stream":true,"stream":false}"#,
            r#"{"max_tokens":1,"max_tokens":2}"#,
            r#"{"max_completion_tokens":1,"max_completion_tokens":2}"#,
            r#"{"stream":null,"stream":true}"#,
        ] {
            let b = Bytes::copy_from_slice(body.as_bytes());
            assert!(
                parse_routing_fields(&b).is_err(),
                "{body} must be rejected as ambiguous"
            );
        }
    }

    #[test]
    fn oversized_non_numeric_sampling_value_is_drained_not_materialized() {
        let big_array = format!("[{}]", "1,".repeat(50_000) + "1");
        let big_string = format!("\"{}\"", "x".repeat(200_000));
        let big_object = format!("{{{}\"k\":1}}", "\"j\":[[[1]]],".repeat(10_000));
        for value in [&big_array, &big_string, &big_object] {
            let body = format!(r#"{{"model":"x","temperature":{value}}}"#);
            let fields = fields_of(&body);
            assert_eq!(
                fields.sampling_field(SamplingField::Temperature),
                SamplingValue::Unusable,
                "a non-numeric value must collapse to Unusable"
            );
            let allow = overrides_of(ConflictPolicy::Allow, r#"{"temperature": 1}"#);
            let inject = resolve_sampling_defaults(&allow, &fields, &metrics()).unwrap();
            assert!(inject.is_empty(), "must not inject over a client value");
            let reject = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
            assert!(
                resolve_sampling_defaults(&reject, &fields, &metrics()).is_err(),
                "reject must refuse a value it cannot read as a number"
            );
        }
    }

    #[test]
    fn sampling_values_normalize_to_numbers() {
        let p = fields_of(r#"{"model":"x","temperature":" 0.7 ","top_k":40,"min_p":0.05}"#);
        assert_eq!(
            p.sampling_field(SamplingField::Temperature),
            SamplingValue::Number(0.7)
        );
        assert_eq!(
            p.sampling_field(SamplingField::TopK),
            SamplingValue::Number(40.0)
        );
        assert_eq!(
            p.sampling_field(SamplingField::MinP),
            SamplingValue::Number(0.05)
        );
        assert_eq!(p.sampling_field(SamplingField::N), SamplingValue::Absent);
    }

    #[test]
    fn contract_rejection_has_its_own_error_code_and_counter() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"top_p": 0.95}"#);
        let metrics = metrics();
        let p = fields_of(r#"{"model":"x","top_p":0.5}"#);

        let err = resolve_sampling_defaults(&overrides, &p, &metrics).unwrap_err();
        let ApiError::SamplingContract { param, .. } = &err else {
            panic!("expected SamplingContract, got {err:?}");
        };
        assert_eq!(*param, "top_p");
        let msg = format!("{err}");
        assert!(msg.contains("top_p") && msg.contains("0.5"), "got {msg}");

        resolve_sampling_defaults(&overrides, &p, &metrics).unwrap_err();
        assert_rejections(&metrics, "top_p", 2);
    }

    #[test]
    fn sampling_splice_preserves_bytes_and_overrides_null_with_or_without_cached_json() {
        let config = overrides_of(ConflictPolicy::Reject, r#"{"temperature":1.0,"n":1}"#);
        for (raw, expected) in [
            (r#"{}"#, r#"{"temperature":1.0,"n":1}"#),
            (r#"{ }"#, r#"{ "temperature":1.0,"n":1}"#),
            (
                "\n\t {\"a\":1}",
                "\n\t {\"a\":1,\"temperature\":1.0,\"n\":1}",
            ),
            (r#"{"a":"}"}"#, r#"{"a":"}","temperature":1.0,"n":1}"#),
            ("{\"a\":1} \n", "{\"a\":1,\"temperature\":1.0,\"n\":1} \n"),
            (
                r#"{ "model" : "x" ,  "messages" : [ ] }"#,
                r#"{ "model" : "x" ,  "messages" : [ ] ,"temperature":1.0,"n":1}"#,
            ),
            (
                r#"{ "model" : "x" }"#,
                r#"{ "model" : "x" ,"temperature":1.0,"n":1}"#,
            ),
            (
                r#"{"model":"x","temperature":null}"#,
                r#"{"model":"x","temperature":null,"temperature":1.0,"n":1}"#,
            ),
        ] {
            let body = Bytes::copy_from_slice(raw.as_bytes());
            let inject = resolve_sampling_defaults(&config, &fields_of(raw), &metrics()).unwrap();
            assert_eq!(inject.len(), 2);
            for value in [None, Some(serde_json::from_slice(&body).unwrap())] {
                let out = build_outgoing_body(&body, value, None, None, &inject, None).unwrap();
                assert_eq!(std::str::from_utf8(&out).unwrap(), expected, "{raw}");
                let parsed: Value = serde_json::from_slice(&out).unwrap();
                assert_eq!(parsed["temperature"], json!(1.0));
                assert_eq!(parsed["n"], json!(1));
            }
        }
    }

    #[test]
    fn values_the_engine_reads_as_numbers_are_judged_not_waved_through() {
        for (body_value, engine_sees) in [("false", 0.0), ("true", 1.0), (r#""0.5_0""#, 0.5)] {
            let fields = fields_of(&format!(r#"{{"model":"x","temperature":{body_value}}}"#));
            assert_eq!(
                fields.sampling_field(SamplingField::Temperature),
                SamplingValue::Number(engine_sees),
                "{body_value} must be read as the number the engine will use"
            );

            let pinned_elsewhere = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 2}"#);
            assert!(
                resolve_sampling_defaults(&pinned_elsewhere, &fields, &metrics()).is_err(),
                "{body_value} differs from the pin and must be rejected"
            );

            let pinned_here = overrides_of(
                ConflictPolicy::Reject,
                &format!(r#"{{"temperature": {engine_sees}}}"#),
            );
            assert!(
                resolve_sampling_defaults(&pinned_here, &fields, &metrics()).is_ok(),
                "{body_value} IS the pinned value to the engine, so it must pass"
            );
        }
    }

    #[test]
    // Expected values were checked against pydantic 2.13.5 with Optional[float].
    fn numeric_strings_are_read_the_way_the_engine_reads_them() {
        #[rustfmt::skip]
        let cases: &[(&str, Option<f64>)] = &[
            ("1_0", Some(10.0_f64)),
            ("1_000.5", Some(1000.5_f64)),
            ("0.5_0", Some(0.5_f64)),
            ("1_0.5_0", Some(10.5_f64)),
            ("0.5e1_0", Some(5000000000.0_f64)),
            ("1_2_3", Some(123.0_f64)),
            ("1_000_000", Some(1000000.0_f64)),
            ("0_1", Some(1.0_f64)),
            ("1_0.0_1", Some(10.01_f64)),
            ("-1_0", Some(-10.0_f64)),
            ("+1_0", Some(10.0_f64)),
            ("1_0e1_0", Some(100000000000.0_f64)),
            ("1._5", Some(1.5_f64)),
            ("1_.5", Some(1.5_f64)),
            ("1e_5", Some(100000.0_f64)),
            ("1_e5", Some(100000.0_f64)),
            ("-_1", Some(-1.0_f64)),
            ("+_1", Some(1.0_f64)),
            ("._5", Some(0.5_f64)),
            ("-_.5", Some(-0.5_f64)),
            ("1_._5", Some(1.5_f64)),
            ("+_.5", Some(0.5_f64)),
            ("1_.", Some(1.0_f64)),
            ("_1", None),
            ("1_", None),
            ("1__0", None),
            ("_", None),
            ("__", None),
            ("._", None),
            ("-_", None),
            ("_.5", None),
            ("1e5_", None),
            ("_1.5", None),
            ("1.5_", None),
            ("0_x10", None),
            ("1_0e_1_0", Some(100000000000.0_f64)),
            (" 1_0 ", None),
            ("_ 1", None),
            ("1 _0", None),
            (" __1 ", None),
            ("\t1_0\n", None),
            ("0.5", Some(0.5_f64)),
            ("  1.5  ", Some(1.5_f64)),
            ("1e-1", Some(0.1_f64)),
            ("+1.5", Some(1.5_f64)),
            (".5", Some(0.5_f64)),
            ("1.", Some(1.0_f64)),
            ("-0", Some(-0.0_f64)),
            ("1E5", Some(100000.0_f64)),
            ("inf", Some(f64::INFINITY)),
            ("-inf", Some(f64::NEG_INFINITY)),
            ("Infinity", Some(f64::INFINITY)),
            ("nan", Some(f64::NAN)),
            ("NaN", Some(f64::NAN)),
            ("0x10", None),
            ("0b101", None),
            ("0o17", None),
            ("1,5", None),
            ("1.5f", None),
            ("", None),
            (" ", None),
            ("abc", None),
            ("1e400", Some(f64::INFINITY)),
        ];
        for &(input, want) in cases {
            let got = parse_engine_numeric_string(input);
            match (got, want) {
                (Some(g), Some(w)) if g.is_nan() && w.is_nan() => {}
                _ => assert_eq!(got, want, "parse_engine_numeric_string({input:?})"),
            }
        }
    }

    #[test]
    fn overlong_numeric_string_is_not_normalized_and_is_refused() {
        let long = format!("1{}", "_0".repeat(MAX_SAMPLING_NUMERIC_LEN));
        assert!(long.len() > MAX_SAMPLING_NUMERIC_LEN);
        assert_eq!(parse_engine_numeric_string(&long), None);

        let fields = fields_of(&format!(r#"{{"model":"x","temperature":"{long}"}}"#));
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
        assert!(resolve_sampling_defaults(&overrides, &fields, &metrics()).is_err());
    }

    #[test]
    fn unreadable_value_rejection_is_counted_and_named() {
        for (config, body, expected_detail) in [
            (
                r#"{"temperature": 1}"#,
                r#"{"model":"x","temperature":"abc"}"#,
                "expected 1 (or omit the field), got a non-numeric value",
            ),
            (
                r#"{"temperature": {"min": 0.5, "max": 1.5}}"#,
                r#"{"model":"x","temperature":{"a":1}}"#,
                "must be a number between 0.5 and 1.5, got a non-numeric value",
            ),
        ] {
            let overrides = overrides_of(ConflictPolicy::Reject, config);
            let metrics = metrics();
            let fields = fields_of(body);

            let err = resolve_sampling_defaults(&overrides, &fields, &metrics).unwrap_err();
            match &err {
                ApiError::SamplingContract { param, detail } => {
                    assert_eq!(*param, "temperature");
                    assert_eq!(detail, expected_detail);
                }
                other => panic!("expected SamplingContract, got {other:?}"),
            }
            assert_rejections(&metrics, "temperature", 1);
        }
    }

    #[test]
    fn out_of_range_number_literal_does_not_fail_parsing() {
        for body in [
            r#"{"model":"x","temperature":1e400}"#,
            r#"{"model":"x","temperature":-1e309}"#,
            r#"{"model":"x","top_k":1E1000,"stream":true}"#,
        ] {
            let fields = parse_routing_fields(&Bytes::copy_from_slice(body.as_bytes()))
                .unwrap_or_else(|e| panic!("{body} must still parse: {e:?}"));
            assert_eq!(fields.model.as_deref(), Some("x"), "{body}");
        }
        let fields = fields_of(r#"{"model":"x","temperature":1e400,"stream":true}"#);
        assert_eq!(fields.stream, Some(true));

        let fields = fields_of(r#"{"model":"x","temperature":1e400}"#);
        assert_eq!(
            fields.sampling_field(SamplingField::Temperature),
            SamplingValue::Unusable
        );
        let allow = overrides_of(ConflictPolicy::Allow, r#"{"temperature": 1}"#);
        assert!(resolve_sampling_defaults(&allow, &fields, &metrics())
            .unwrap()
            .is_empty());
        let reject = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
        assert!(resolve_sampling_defaults(&reject, &fields, &metrics()).is_err());

        for bad in [
            r#"{"model":"x","stream":true,"stream":false}"#,
            "[]",
            "null",
            "{oops",
        ] {
            assert!(
                parse_routing_fields(&Bytes::copy_from_slice(bad.as_bytes())).is_err(),
                "{bad} must still be rejected"
            );
        }
    }

    #[test]
    fn every_violated_parameter_is_counted_not_just_the_first() {
        let overrides = overrides_of(
            ConflictPolicy::Reject,
            r#"{"temperature": 1, "top_p": 0.95, "n": 1}"#,
        );
        let metrics = metrics();
        let fields = fields_of(r#"{"model":"x","temperature":0.7,"top_p":0.8,"n":2}"#);

        let err = resolve_sampling_defaults(&overrides, &fields, &metrics).unwrap_err();
        let ApiError::SamplingContract { param, .. } = &err else {
            panic!("expected SamplingContract, got {err:?}");
        };
        assert_eq!(*param, "temperature");

        for name in ["temperature", "top_p", "n"] {
            assert_rejections(&metrics, name, 1);
        }
    }

    #[test]
    fn overlong_plain_numeric_string_is_not_parsed() {
        let long = "1".repeat(MAX_SAMPLING_NUMERIC_LEN + 1);
        assert_eq!(parse_engine_numeric_string(&long), None);
        assert_eq!(
            parse_engine_numeric_string(&"1".repeat(MAX_SAMPLING_NUMERIC_LEN)),
            "1".repeat(MAX_SAMPLING_NUMERIC_LEN).parse::<f64>().ok(),
            "a value at the cap is still read"
        );
    }

    #[test]
    fn abort_opt_outs_follow_caller_rid_and_effective_sample_count() {
        for raw in [r#"{"rid":"abc"}"#, r#"{"rid":["a","b"]}"#] {
            assert!(fields_of(raw).caller_set_rid);
        }
        assert!(!fields_of(r#"{"rid":null}"#).caller_set_rid);
        for (raw, fan_out) in [
            (r#"{}"#, false),
            (r#"{"n":1}"#, false),
            (r#"{"n":2}"#, true),
            (r#"{"n":"3"}"#, true),
            (r#"{"n":[2]}"#, true),
        ] {
            assert_eq!(
                requests_multiple_samples(&fields_of(raw), &[]),
                fan_out,
                "{raw}"
            );
        }
        for (config, fan_out) in [(r#"{"n":1}"#, false), (r#"{"n":4}"#, true)] {
            let fields = fields_of("{}");
            let defaults = resolve_sampling_defaults(
                &overrides_of(ConflictPolicy::Reject, config),
                &fields,
                &metrics(),
            )
            .unwrap();
            assert_eq!(requests_multiple_samples(&fields, &defaults), fan_out);
        }
    }
}
