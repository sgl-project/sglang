// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat request validation, optional tokenization, and outgoing body preparation.

use crate::config::{ConflictPolicy, ParamSpec, SamplingField, SamplingOverrides};
use crate::discovery::ModelId;
use crate::policies::{request_tokens_for, RequestTokens};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::MetricsRegistry;
use bytes::Bytes;
use serde::de::IgnoredAny;
use serde::Deserialize;

const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

pub(crate) struct ChatRequest {
    pub(crate) model: ModelId,
    pub(crate) streaming: bool,
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) body: Bytes,
    pub(crate) tokens: Option<RequestTokens>,
    pub(crate) prefill_load: usize,
    value: Option<serde_json::Value>,
    sampling: Vec<(SamplingField, serde_json::Number)>,
}

impl ChatRequest {
    pub(crate) fn prepare(
        ctx: &AppContext,
        model: ModelId,
        probe: RequestProbe,
        body: Bytes,
        policy_needs_tokens: bool,
    ) -> Result<Self, ApiError> {
        let sampling =
            apply_sampling_overrides(&ctx.config.model.sampling_overrides, &probe, &ctx.metrics)?;
        let want_tokens = should_tokenize_request(
            ctx.tokenizers.has_chat_formatter(&model.0),
            policy_needs_tokens,
            ctx.bucket_selector.is_enabled(),
        );
        let value = want_tokens
            .then(|| serde_json::from_slice(&body))
            .transpose()
            .map_err(|_| {
                ApiError::BadRequest("invalid request: body must be a JSON object".into())
            })?;
        let tokens = value
            .as_ref()
            .and_then(|value| request_tokens_for(&ctx.tokenizers, &model, value));
        let prefill_load = tokens
            .as_ref()
            .map(|tokens| tokens.ids.len().max(1))
            .unwrap_or_else(|| estimate_prefill_tokens(&body));
        Ok(Self {
            model,
            streaming: probe.stream.unwrap_or(false),
            max_output_tokens: probe.requested_max_output_tokens(),
            body,
            tokens,
            prefill_load,
            value,
            sampling,
        })
    }

    pub(crate) fn into_outgoing_body(
        self,
        ctx: &AppContext,
        bootstrap: Option<&BootstrapFields>,
    ) -> Result<Bytes, ApiError> {
        let input_ids = match (self.tokens.as_ref(), self.value.as_ref()) {
            (Some(tokens), Some(value))
                if tokens.rendered_from_chat && input_ids_safe_to_forward(value) =>
            {
                Some(tokens.ids.as_slice())
            }
            _ => None,
        };
        if ingress_tokenize_offload_failed(
            ctx.tokenizers.has_chat_formatter(&self.model.0),
            self.value.as_ref(),
            self.tokens.as_ref(),
        ) {
            ctx.metrics.record_ingress_tokenize_error(&self.model.0);
        }
        build_outgoing_body(&self.body, self.value, input_ids, bootstrap, &self.sampling)
    }
}

/// Reads routing and sampling fields without retaining unrelated client data.
#[derive(Debug, Default)]
pub(crate) struct RequestProbe {
    stream: Option<bool>,
    pub(crate) model: Option<String>,
    max_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    sampling: [ProbedValue; SamplingField::ALL.len()],
}

/// Null is absent; unrepresentable values are rejected only under a sampling contract.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum ProbedValue {
    #[default]
    Absent,
    Number(f64),
    Unusable,
}

const MAX_SAMPLING_NUMERIC_LEN: usize = 64;

/// Match engine coercion: trim ordinary numbers, but preserve whitespace for underscored ones.
fn parse_as_engine_number(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    if trimmed.len() > MAX_SAMPLING_NUMERIC_LEN {
        return None;
    }
    if let Ok(v) = trimmed.parse::<f64>() {
        return Some(v);
    }
    if s.len() > MAX_SAMPLING_NUMERIC_LEN
        || !s.contains('_')
        || s.starts_with('_')
        || s.ends_with('_')
        || s.contains("__")
    {
        return None;
    }
    // Bound numeric parsing and keep underscore removal on the stack.
    let mut buf = [0u8; MAX_SAMPLING_NUMERIC_LEN];
    let mut len = 0;
    for &b in s.as_bytes() {
        if b != b'_' {
            buf[len] = b;
            len += 1;
        }
    }
    std::str::from_utf8(&buf[..len]).ok()?.parse().ok()
}

impl<'de> Deserialize<'de> for ProbedValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> serde::de::Visitor<'de> for ValueVisitor {
            type Value = ProbedValue;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sampling parameter value")
            }

            fn visit_i64<E>(self, v: i64) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Number(v as f64))
            }

            fn visit_u64<E>(self, v: u64) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Number(v as f64))
            }

            fn visit_f64<E>(self, v: f64) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Number(v))
            }

            fn visit_str<E>(self, v: &str) -> Result<ProbedValue, E> {
                Ok(parse_as_engine_number(v).map_or(ProbedValue::Unusable, ProbedValue::Number))
            }

            fn visit_unit<E>(self) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Absent)
            }

            fn visit_bool<E>(self, v: bool) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Number(if v { 1.0 } else { 0.0 }))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<ProbedValue, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(ProbedValue::Unusable)
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<ProbedValue, M::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(ProbedValue::Unusable)
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

enum ProbeKey {
    Routing(RoutingKey),
    Sampling(SamplingField),
    Other,
}

impl<'de> Deserialize<'de> for ProbeKey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl serde::de::Visitor<'_> for KeyVisitor {
            type Value = ProbeKey;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a request field name")
            }

            fn visit_str<E>(self, v: &str) -> Result<ProbeKey, E> {
                Ok(match v {
                    "stream" => ProbeKey::Routing(RoutingKey::Stream),
                    "model" => ProbeKey::Routing(RoutingKey::Model),
                    "max_tokens" => ProbeKey::Routing(RoutingKey::MaxTokens),
                    "max_completion_tokens" => ProbeKey::Routing(RoutingKey::MaxCompletionTokens),
                    other => match SamplingField::from_wire_name(other) {
                        Some(field) => ProbeKey::Sampling(field),
                        None => ProbeKey::Other,
                    },
                })
            }
        }
        d.deserialize_str(KeyVisitor)
    }
}

struct ProbeVisitor {
    read_sampling: bool,
}

impl<'de> serde::de::Visitor<'de> for ProbeVisitor {
    type Value = RequestProbe;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON object")
    }

    fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<RequestProbe, M::Error> {
        let mut probe = RequestProbe::default();
        let mut seen = 0u8;
        while let Some(key) = map.next_key::<ProbeKey>()? {
            match key {
                ProbeKey::Routing(r) => {
                    if seen & r.bit() != 0 {
                        return Err(serde::de::Error::custom(format_args!(
                            "duplicate field `{}`",
                            r.wire_name()
                        )));
                    }
                    // Routing duplicates are ambiguous; sampling duplicates below use the last value.
                    seen |= r.bit();
                    match r {
                        RoutingKey::Stream => probe.stream = map.next_value()?,
                        RoutingKey::Model => probe.model = map.next_value()?,
                        RoutingKey::MaxTokens => probe.max_tokens = map.next_value()?,
                        RoutingKey::MaxCompletionTokens => {
                            probe.max_completion_tokens = map.next_value()?
                        }
                    }
                }
                ProbeKey::Sampling(field) => {
                    probe.sampling[field.index()] = if self.read_sampling {
                        map.next_value()?
                    } else {
                        map.next_value::<IgnoredAny>()?;
                        ProbedValue::Unusable
                    };
                }
                ProbeKey::Other => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(probe)
    }
}

impl<'de> Deserialize<'de> for RequestProbe {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(ProbeVisitor {
            read_sampling: true,
        })
    }
}

/// Preserve valid JSON with sampling numbers outside f64 range (for example, 1e400).
fn probe_without_sampling_values(body: &[u8]) -> Result<RequestProbe, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_slice(body);
    let probe = serde::Deserializer::deserialize_map(
        &mut de,
        ProbeVisitor {
            read_sampling: false,
        },
    )?;
    de.end()?;
    Ok(probe)
}

impl RequestProbe {
    fn requested_max_output_tokens(&self) -> Option<u64> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    fn sampling_field(&self, field: SamplingField) -> ProbedValue {
        self.sampling[field.index()]
    }
}

fn should_tokenize_request(
    has_chat_formatter: bool,
    policy_needs_request_tokens: bool,
    bucket_enabled: bool,
) -> bool {
    has_chat_formatter || policy_needs_request_tokens || bucket_enabled
}

fn estimate_prefill_tokens(body: &Bytes) -> usize {
    (body.len() / CHARS_PER_TOKEN_ESTIMATE).max(1)
}

/// The engine stores bootstrap rooms as signed int64 values.
pub(crate) fn generate_room_id() -> u64 {
    rand::random::<u64>() & (i64::MAX as u64)
}

pub(crate) struct BootstrapFields {
    pub(crate) host: String,
    pub(crate) port: Option<u16>,
    pub(crate) room: u64,
}

/// Append before the closing brace so injected values win over explicit nulls.
fn splice_top_level(
    body: &Bytes,
    members: &[(SamplingField, serde_json::Number)],
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
    let mut out = Vec::with_capacity(body.len() + 24 * members.len() + 1);
    out.extend_from_slice(&body[..close]);
    for (i, (field, value)) in members.iter().enumerate() {
        if has_members || i > 0 {
            out.push(b',');
        }
        write!(out, "\"{}\":{}", field.wire_name(), value).ok()?;
    }
    out.extend_from_slice(&body[close..]);
    Some(Bytes::from(out))
}

/// Reuse the parsed body when injecting tokens or bootstrap fields; splice sampling alone.
fn build_outgoing_body(
    body: &Bytes,
    value: Option<serde_json::Value>,
    input_ids: Option<&[u32]>,
    bootstrap: Option<&BootstrapFields>,
    sampling: &[(SamplingField, serde_json::Number)],
) -> Result<Bytes, ApiError> {
    let only_sampling = input_ids.is_none() && bootstrap.is_none();
    if only_sampling && sampling.is_empty() {
        return Ok(body.clone());
    }
    if only_sampling {
        if let Some(spliced) = splice_top_level(body, sampling) {
            return Ok(spliced);
        }
    }
    let parsed = match value {
        Some(v) => v,
        None => serde_json::from_slice(body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?,
    };
    let mut obj = match parsed {
        serde_json::Value::Object(map) => map,
        _ => {
            return Err(ApiError::BadRequest(
                "invalid request: body must be a JSON object".to_string(),
            ));
        }
    };
    for (field, value) in sampling {
        obj.insert(
            field.wire_name().to_string(),
            serde_json::Value::Number(value.clone()),
        );
    }
    if let Some(ids) = input_ids {
        obj.insert(
            "input_ids".to_string(),
            serde_json::Value::Array(
                ids.iter()
                    .map(|&i| serde_json::Value::Number(i.into()))
                    .collect(),
            ),
        );
    }
    if let Some(b) = bootstrap {
        obj.insert(
            "bootstrap_host".to_string(),
            serde_json::Value::String(b.host.clone()),
        );
        obj.insert(
            "bootstrap_port".to_string(),
            match b.port {
                Some(p) => serde_json::Value::Number(p.into()),
                None => serde_json::Value::Null,
            },
        );
        obj.insert(
            "bootstrap_room".to_string(),
            serde_json::Value::Number(b.room.into()),
        );
    }
    let bytes = serde_json::to_vec(&obj).map_err(|e| {
        ApiError::Internal(anyhow::Error::new(e).context("re-serialize injected request body"))
    })?;
    Ok(Bytes::from(bytes))
}

/// Only forward tokens when the router replicated all template inputs.
fn input_ids_safe_to_forward(value: &serde_json::Value) -> bool {
    if request_has_tools(value) || request_has_non_text_content(value) {
        return false;
    }
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

fn ingress_tokenize_offload_failed(
    has_chat_formatter: bool,
    request_value: Option<&serde_json::Value>,
    request_tokens: Option<&RequestTokens>,
) -> bool {
    if !has_chat_formatter {
        return false;
    }
    let chat_request = request_value.is_some_and(|v| {
        v.get("messages").is_some_and(|m| m.is_array()) && input_ids_safe_to_forward(v)
    });
    if !chat_request {
        return false;
    }
    !request_tokens.is_some_and(|t| t.rendered_from_chat)
}

fn last_message_is_assistant(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        == Some("assistant")
}

fn request_has_tools(value: &serde_json::Value) -> bool {
    let nonempty = |key: &str| {
        value.get(key).is_some_and(|v| match v {
            serde_json::Value::Array(a) => !a.is_empty(),
            serde_json::Value::Null => false,
            _ => true,
        })
    };
    nonempty("tools") || nonempty("functions")
}

fn request_has_non_text_content(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|msgs| {
            msgs.iter()
                .any(|m| !matches!(m.get("content"), Some(serde_json::Value::String(_))))
        })
}

fn apply_sampling_overrides(
    overrides: &SamplingOverrides,
    probe: &RequestProbe,
    metrics: &MetricsRegistry,
) -> Result<Vec<(SamplingField, serde_json::Number)>, ApiError> {
    let mut inject = Vec::with_capacity(overrides.params.len());
    // Count every violated parameter, but report only the first to the client.
    let mut first_violation: Option<ApiError> = None;
    for (&field, spec) in &overrides.params {
        let got = probe.sampling_field(field);
        if got == ProbedValue::Absent {
            if let ParamSpec::Exact(value) = spec {
                inject.push((field, value.clone()));
            }
            continue;
        }
        if overrides.conflict == ConflictPolicy::Allow {
            continue;
        }
        if let Some(detail) = sampling_violation(spec, got) {
            let param = field.wire_name();
            metrics.record_sampling_contract_rejection(param);
            first_violation.get_or_insert(ApiError::SamplingContract { param, detail });
        }
    }
    match first_violation {
        Some(err) => Err(err),
        None => Ok(inject),
    }
}

fn sampling_violation(spec: &ParamSpec, got: ProbedValue) -> Option<String> {
    match (spec, got) {
        (ParamSpec::Exact(want), ProbedValue::Unusable) => Some(format!(
            "expected {want} (or omit the field), got a non-numeric value"
        )),
        (ParamSpec::Range { lo, hi }, ProbedValue::Unusable) => Some(format!(
            "must be a number between {lo} and {hi}, got a non-numeric value"
        )),
        (ParamSpec::Exact(want), ProbedValue::Number(got)) if Some(got) != want.as_f64() => {
            Some(format!("got {got}, expected {want} (or omit the field)"))
        }
        (&ParamSpec::Range { lo, hi }, ProbedValue::Number(got)) if !(lo..=hi).contains(&got) => {
            Some(format!("must be between {lo} and {hi}, got {got}"))
        }
        _ => None,
    }
}

pub(crate) fn parse_probe(body: &Bytes) -> Result<RequestProbe, ApiError> {
    let err = match serde_json::from_slice::<RequestProbe>(body) {
        Ok(probe) => return Ok(probe),
        Err(e) => e,
    };
    if let Ok(probe) = probe_without_sampling_values(body) {
        return Ok(probe);
    }
    // Keep deserialization details out of the client-visible error.
    tracing::debug!(error = %err, "chat-completions request-probe deserialize failed");
    Err(ApiError::BadRequest(
        "invalid request: body must be a JSON object".to_string(),
    ))
}

#[cfg(test)]
mod tests;
