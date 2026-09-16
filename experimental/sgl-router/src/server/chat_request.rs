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
    fn build_outgoing_body_emits_null_for_missing_port() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let bootstrap = BootstrapFields {
            host: "host".into(),
            port: None,
            room: 42,
        };
        let injected =
            build_outgoing_body(&body, Some(value), None, Some(&bootstrap), &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&injected).unwrap();
        assert_eq!(parsed.get("bootstrap_port"), Some(&serde_json::Value::Null));
        assert_eq!(
            parsed.get("bootstrap_host"),
            Some(&serde_json::Value::String("host".into()))
        );
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(42.into()))
        );
    }

    #[test]
    fn build_outgoing_body_injects_input_ids_and_keeps_messages() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [1u32, 2, 3];
        let out = build_outgoing_body(&body, Some(value), Some(&ids), None, &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([1, 2, 3])));
        assert!(
            parsed.get("messages").is_some(),
            "messages must be retained alongside input_ids"
        );
    }

    #[test]
    fn build_outgoing_body_no_injection_returns_original_bytes() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let out = build_outgoing_body(&body, Some(value), None, None, &[]).unwrap();
        assert_eq!(
            out, body,
            "no injection must forward the original bytes unchanged"
        );
    }

    #[test]
    fn build_outgoing_body_injects_both_input_ids_and_bootstrap() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [7u32, 8];
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(9),
            room: 5,
        };
        let out =
            build_outgoing_body(&body, Some(value), Some(&ids), Some(&bootstrap), &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([7, 8])));
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(5.into()))
        );
        assert_eq!(
            parsed.get("bootstrap_port"),
            Some(&serde_json::Value::Number(9.into()))
        );
    }

    #[test]
    fn request_has_tools_detects_tools_and_functions() {
        assert!(request_has_tools(
            &serde_json::json!({"tools":[{"type":"function"}]})
        ));
        assert!(request_has_tools(
            &serde_json::json!({"functions":[{"name":"f"}]})
        ));
        assert!(!request_has_tools(&serde_json::json!({"tools":[]})));
        assert!(!request_has_tools(&serde_json::json!({"messages":[]})));
    }

    #[test]
    fn request_has_non_text_content_detects_non_string_content() {
        for content in [
            serde_json::json!([{"type":"image_url","image_url":"x"}]),
            serde_json::json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]),
            serde_json::Value::Null,
        ] {
            assert!(
                request_has_non_text_content(&serde_json::json!({
                    "messages":[{"role":"user","content":"hi"},{"role":"assistant","content":content}]
                })),
                "content {content} must block"
            );
        }
        assert!(request_has_non_text_content(&serde_json::json!({
            "messages":[{"role":"assistant","tool_calls":[]}]
        })));
        assert!(!request_has_non_text_content(&serde_json::json!({
            "messages":[{"role":"user","content":"hello"}]
        })));
    }

    #[test]
    fn input_ids_safe_to_forward_allows_plain_text_chat() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        })));
    }

    #[test]
    fn input_ids_safe_to_forward_blocks_unreplicated_signals() {
        let blockers = [
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}),
            serde_json::json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template":"{{ custom }}"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning":{"enabled":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"task":"generate"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"continue_final_message":true}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"partial"}]}),
        ];
        for b in blockers {
            assert!(
                !input_ids_safe_to_forward(&b),
                "must NOT forward input_ids for: {b}"
            );
        }
    }

    #[test]
    fn input_ids_safe_to_forward_ignores_null_and_false_fields() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template": null,
            "reasoning_effort": null,
            "chat_template_kwargs": null,
            "continue_final_message": false
        })));
    }

    #[test]
    fn build_outgoing_body_reparses_when_value_absent() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(1),
            room: 2,
        };
        let out = build_outgoing_body(&body, None, None, Some(&bootstrap), &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(2.into()))
        );
        assert!(parsed.get("input_ids").is_none());
    }

    #[test]
    fn offload_failed_false_when_tokens_rendered_from_chat() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            rendered_from_chat: true,
        };
        assert!(!ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    #[test]
    fn offload_failed_false_for_unforwardable_request() {
        let value = serde_json::json!({
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f"}}]
        });
        assert!(!ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    #[test]
    fn offload_failed_true_when_chat_formatter_request_has_no_tokens() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    #[test]
    fn offload_failed_true_when_tokens_not_rendered_from_chat() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            rendered_from_chat: false,
        };
        assert!(ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    #[test]
    fn offload_failed_false_without_chat_formatter() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(!ingress_tokenize_offload_failed(false, Some(&value), None));
    }

    #[test]
    fn offload_failed_false_for_non_messages_request() {
        let value = serde_json::json!({"prompt":"hi"});
        assert!(!ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    #[test]
    fn parse_probe_reads_stream_bool_from_object() {
        let b = Bytes::from_static(br#"{"stream": true, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
        let b = Bytes::from_static(br#"{"stream": false, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_defaults_when_stream_absent() {
        let b = Bytes::from_static(br#"{"model": "tiny", "messages": []}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, None);
        assert_eq!(p.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn parse_probe_accepts_modern_openai_completion_budget() {
        let body =
            Bytes::from_static(br#"{"model":"tiny","messages":[],"max_completion_tokens":256}"#);
        assert_eq!(
            parse_probe(&body).unwrap().requested_max_output_tokens(),
            Some(256)
        );
    }

    #[test]
    fn modern_completion_budget_takes_precedence_when_both_fields_are_present() {
        let body =
            Bytes::from_static(br#"{"model":"tiny","max_tokens":128,"max_completion_tokens":256}"#);
        assert_eq!(
            parse_probe(&body).unwrap().requested_max_output_tokens(),
            Some(256)
        );
    }

    #[test]
    fn parse_probe_rejects_non_object_shapes() {
        for bad in [&b"null"[..], &b"[]"[..], &b"\"hi\""[..], &b"42"[..]] {
            let b = Bytes::copy_from_slice(bad);
            let err = parse_probe(&b).unwrap_err();
            match err {
                ApiError::BadRequest(_) => {}
                other => panic!("expected BadRequest for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_probe_rejects_malformed_json() {
        let b = Bytes::from_static(b"{not json}");
        let err = parse_probe(&b).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_true() {
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": true
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_false() {
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": false
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_handles_duplicate_stream_keys() {
        let b = Bytes::from_static(br#"{"stream": true, "stream": false}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(_) => {}
            other => panic!("expected BadRequest on duplicate `stream` key, got {other:?}"),
        }
    }

    #[test]
    fn parse_probe_bad_request_message_does_not_leak_serde_detail() {
        let b = Bytes::from_static(br#"{"stream": "not-a-bool"}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => assert_eq!(
                msg, "invalid request: body must be a JSON object",
                "client-visible message must be fixed; got: {msg}"
            ),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    fn probe_of(body: &str) -> RequestProbe {
        parse_probe(&Bytes::copy_from_slice(body.as_bytes())).unwrap()
    }

    fn overrides_of(conflict: ConflictPolicy, json: &str) -> SamplingOverrides {
        crate::config::parse_sampling_overrides(json, conflict).expect("test config must parse")
    }

    fn metrics() -> Arc<MetricsRegistry> {
        MetricsRegistry::new()
    }

    #[test]
    fn unconfigured_sampling_overrides_inject_nothing() {
        let overrides = SamplingOverrides::default();
        let p = probe_of(r#"{"model":"x","temperature":0.7,"n":4}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics()).unwrap(),
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
        let inject = apply_sampling_overrides(
            &reject_overrides(),
            &probe_of(r#"{"model":"x","messages":[]}"#),
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
            let result = apply_sampling_overrides(&overrides, &probe_of(&body), &metrics());
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
        let p = probe_of(r#"{"model":"x","temperature":0.6}"#);
        assert!(apply_sampling_overrides(&exact, &p, &metrics()).is_err());
        assert_eq!(
            apply_sampling_overrides(&exact, &probe_of(r#"{"model":"x"}"#), &metrics()).unwrap(),
            vec![(
                SamplingField::Temperature,
                serde_json::Number::from_f64(1.0).unwrap()
            )]
        );
    }

    #[test]
    fn allow_mode_never_rejects_and_never_masks_a_client_value() {
        let overrides = overrides_of(
            ConflictPolicy::Allow,
            r#"{"temperature": 1, "top_p": 0.95, "n": 1}"#,
        );

        let p = probe_of(r#"{"model":"x"}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics())
                .unwrap()
                .iter()
                .map(|(f, _)| f.wire_name())
                .collect::<Vec<_>>(),
            vec!["temperature", "top_p", "n"]
        );

        let p = probe_of(r#"{"model":"x","temperature":0.6,"top_p":0.8,"n":4}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics()).unwrap(),
            vec![]
        );

        let p = probe_of(r#"{"model":"x","temperature":0.6}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics())
                .unwrap()
                .iter()
                .map(|(f, _)| f.wire_name())
                .collect::<Vec<_>>(),
            vec!["top_p", "n"]
        );
    }

    #[test]
    fn explicit_null_sampling_value_counts_as_omitted() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
        let p = probe_of(r#"{"model":"x","temperature":null}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics())
                .unwrap()
                .iter()
                .map(|(f, _)| f.wire_name())
                .collect::<Vec<_>>(),
            vec!["temperature"]
        );
    }

    #[test]
    fn build_outgoing_body_injects_sampling_overrides_alongside_input_ids() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [1u32, 2, 3];
        let overrides = overrides_of(
            ConflictPolicy::Reject,
            r#"{"top_p": 0.95, "top_k": 1000, "frequency_penalty": 0.0,
                "presence_penalty": 0.0, "n": 1}"#,
        );
        let inject = apply_sampling_overrides(
            &overrides,
            &probe_of(r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#),
            &metrics(),
        )
        .unwrap();
        let out = build_outgoing_body(&body, Some(value), Some(&ids), None, &inject).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("top_p"), Some(&serde_json::json!(0.95)));
        assert_eq!(parsed.get("top_k"), Some(&serde_json::json!(1000)));
        assert_eq!(parsed.get("n"), Some(&serde_json::json!(1)));
        assert_eq!(
            parsed.get("frequency_penalty"),
            Some(&serde_json::json!(0.0))
        );
        assert_eq!(
            parsed.get("presence_penalty"),
            Some(&serde_json::json!(0.0))
        );
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([1, 2, 3])));
        assert!(parsed.get("messages").is_some());
    }
    #[test]
    fn duplicate_sampling_key_takes_the_last_value_like_the_engine() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);

        let p = probe_of(r#"{"model":"x","temperature":0.5,"temperature":1}"#);
        assert!(apply_sampling_overrides(&overrides, &p, &metrics()).is_ok());

        let p = probe_of(r#"{"model":"x","temperature":1,"temperature":0.5}"#);
        let err = apply_sampling_overrides(&overrides, &p, &metrics()).unwrap_err();
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
                parse_probe(&b).is_err(),
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
            let probe = probe_of(&body);
            assert_eq!(
                probe.sampling_field(SamplingField::Temperature),
                ProbedValue::Unusable,
                "a non-numeric value must collapse to Unusable"
            );
            let allow = overrides_of(ConflictPolicy::Allow, r#"{"temperature": 1}"#);
            let inject = apply_sampling_overrides(&allow, &probe, &metrics()).unwrap();
            assert!(inject.is_empty(), "must not inject over a client value");
            let reject = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
            assert!(
                apply_sampling_overrides(&reject, &probe, &metrics()).is_err(),
                "reject must refuse a value it cannot read as a number"
            );
        }
    }

    #[test]
    fn probed_sampling_values_normalize_to_numbers() {
        let p = probe_of(r#"{"model":"x","temperature":" 0.7 ","top_k":40,"min_p":0.05}"#);
        assert_eq!(
            p.sampling_field(SamplingField::Temperature),
            ProbedValue::Number(0.7)
        );
        assert_eq!(
            p.sampling_field(SamplingField::TopK),
            ProbedValue::Number(40.0)
        );
        assert_eq!(
            p.sampling_field(SamplingField::MinP),
            ProbedValue::Number(0.05)
        );
        assert_eq!(p.sampling_field(SamplingField::N), ProbedValue::Absent);
    }

    #[test]
    fn contract_rejection_has_its_own_error_code_and_counter() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"top_p": 0.95}"#);
        let metrics = metrics();
        let p = probe_of(r#"{"model":"x","top_p":0.5}"#);

        let err = apply_sampling_overrides(&overrides, &p, &metrics).unwrap_err();
        let ApiError::SamplingContract { param, .. } = &err else {
            panic!("expected SamplingContract, got {err:?}");
        };
        assert_eq!(*param, "top_p");
        let msg = format!("{err}");
        assert!(msg.contains("top_p") && msg.contains("0.5"), "got {msg}");

        apply_sampling_overrides(&overrides, &p, &metrics).unwrap_err();
        assert!(
            metrics
                .render()
                .contains(r#"sgl_router_sampling_contract_rejections_total{param="top_p"} 2"#),
            "rejections must be counted per parameter:\n{}",
            metrics.render()
        );
    }

    #[test]
    fn build_outgoing_body_splices_sampling_without_reparsing() {
        let body = Bytes::from_static(br#"{ "model" : "x" ,  "messages" : [ ] }"#);
        let inject = apply_sampling_overrides(
            &overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0, "n": 1}"#),
            &probe_of(r#"{"model":"x"}"#),
            &metrics(),
        )
        .unwrap();

        let out = build_outgoing_body(&body, None, None, None, &inject).unwrap();
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            r#"{ "model" : "x" ,  "messages" : [ ] ,"temperature":1.0,"n":1}"#
        );
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("temperature"), Some(&serde_json::json!(1.0)));
        assert_eq!(parsed.get("n"), Some(&serde_json::json!(1)));
        assert_eq!(parsed.get("model"), Some(&serde_json::json!("x")));
    }

    #[test]
    fn splice_top_level_handles_empty_objects_and_leading_whitespace() {
        let inject = apply_sampling_overrides(
            &overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#),
            &probe_of(r#"{"model":"x"}"#),
            &metrics(),
        )
        .unwrap();

        for (raw, want) in [
            (r#"{}"#, r#"{"temperature":1.0}"#),
            (r#"{ }"#, r#"{ "temperature":1.0}"#),
            ("\n\t {\"a\":1}", "\n\t {\"a\":1,\"temperature\":1.0}"),
            (r#"{"a":"}"}"#, r#"{"a":"}","temperature":1.0}"#),
            ("{\"a\":1} \n", "{\"a\":1,\"temperature\":1.0} \n"),
        ] {
            let out = splice_top_level(&Bytes::copy_from_slice(raw.as_bytes()), &inject).unwrap();
            assert_eq!(std::str::from_utf8(&out).unwrap(), want, "input {raw:?}");
            serde_json::from_slice::<serde_json::Value>(&out)
                .unwrap_or_else(|e| panic!("{raw:?} spliced to invalid JSON: {e}"));
        }
    }

    #[test]
    fn build_outgoing_body_without_injection_forwards_the_same_allocation() {
        let body = Bytes::from_static(br#"{"model":"x"}"#);
        let out = build_outgoing_body(&body, None, None, None, &[]).unwrap();
        assert_eq!(
            out.as_ptr(),
            body.as_ptr(),
            "must be an Arc clone, not a copy"
        );
    }

    #[test]
    fn values_the_engine_reads_as_numbers_are_judged_not_waved_through() {
        for (body_value, engine_sees) in [("false", 0.0), ("true", 1.0), (r#""0.5_0""#, 0.5)] {
            let probe = probe_of(&format!(r#"{{"model":"x","temperature":{body_value}}}"#));
            assert_eq!(
                probe.sampling_field(SamplingField::Temperature),
                ProbedValue::Number(engine_sees),
                "{body_value} must be read as the number the engine will use"
            );

            let pinned_elsewhere = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 2}"#);
            assert!(
                apply_sampling_overrides(&pinned_elsewhere, &probe, &metrics()).is_err(),
                "{body_value} differs from the pin and must be rejected"
            );

            let pinned_here = overrides_of(
                ConflictPolicy::Reject,
                &format!(r#"{{"temperature": {engine_sees}}}"#),
            );
            assert!(
                apply_sampling_overrides(&pinned_here, &probe, &metrics()).is_ok(),
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
            let got = parse_as_engine_number(input);
            match (got, want) {
                (Some(g), Some(w)) if g.is_nan() && w.is_nan() => {}
                _ => assert_eq!(got, want, "parse_as_engine_number({input:?})"),
            }
        }
    }

    #[test]
    fn overlong_numeric_string_is_not_normalized_and_is_refused() {
        let long = format!("1{}", "_0".repeat(MAX_SAMPLING_NUMERIC_LEN));
        assert!(long.len() > MAX_SAMPLING_NUMERIC_LEN);
        assert_eq!(parse_as_engine_number(&long), None);

        let probe = probe_of(&format!(r#"{{"model":"x","temperature":"{long}"}}"#));
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
        assert!(apply_sampling_overrides(&overrides, &probe, &metrics()).is_err());
    }

    #[test]
    fn allow_never_rejects_an_unreadable_value() {
        let probe = probe_of(r#"{"model":"x","temperature":"abc","top_p":[1]}"#);
        let overrides = overrides_of(ConflictPolicy::Allow, r#"{"temperature": 1, "top_p": 0.9}"#);
        let inject = apply_sampling_overrides(&overrides, &probe, &metrics()).unwrap();
        assert!(inject.is_empty(), "a client value is never overwritten");
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
            let probe = probe_of(body);

            let err = apply_sampling_overrides(&overrides, &probe, &metrics).unwrap_err();
            match &err {
                ApiError::SamplingContract { param, detail } => {
                    assert_eq!(*param, "temperature");
                    assert_eq!(detail, expected_detail);
                }
                other => panic!("expected SamplingContract, got {other:?}"),
            }
            assert!(
                metrics.render().contains(
                    r#"sgl_router_sampling_contract_rejections_total{param="temperature"} 1"#
                ),
                "a refusal must be visible to an operator rolling the flag out:\n{}",
                metrics.render()
            );
        }
    }

    #[test]
    fn out_of_range_number_literal_does_not_fail_the_probe() {
        for body in [
            r#"{"model":"x","temperature":1e400}"#,
            r#"{"model":"x","temperature":-1e309}"#,
            r#"{"model":"x","top_k":1E1000,"stream":true}"#,
        ] {
            let probe = parse_probe(&Bytes::copy_from_slice(body.as_bytes()))
                .unwrap_or_else(|e| panic!("{body} must still parse: {e:?}"));
            assert_eq!(probe.model.as_deref(), Some("x"), "{body}");
        }
        let probe = probe_of(r#"{"model":"x","temperature":1e400,"stream":true}"#);
        assert_eq!(probe.stream, Some(true));

        let probe = probe_of(r#"{"model":"x","temperature":1e400}"#);
        assert_eq!(
            probe.sampling_field(SamplingField::Temperature),
            ProbedValue::Unusable
        );
        let allow = overrides_of(ConflictPolicy::Allow, r#"{"temperature": 1}"#);
        assert!(apply_sampling_overrides(&allow, &probe, &metrics())
            .unwrap()
            .is_empty());
        let reject = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
        assert!(apply_sampling_overrides(&reject, &probe, &metrics()).is_err());

        for bad in [
            r#"{"model":"x","stream":true,"stream":false}"#,
            "[]",
            "null",
            "{oops",
        ] {
            assert!(
                parse_probe(&Bytes::copy_from_slice(bad.as_bytes())).is_err(),
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
        let probe = probe_of(r#"{"model":"x","temperature":0.7,"top_p":0.8,"n":2}"#);

        let err = apply_sampling_overrides(&overrides, &probe, &metrics).unwrap_err();
        let ApiError::SamplingContract { param, .. } = &err else {
            panic!("expected SamplingContract, got {err:?}");
        };
        assert_eq!(*param, "temperature");

        let rendered = metrics.render();
        for name in ["temperature", "top_p", "n"] {
            assert!(
                rendered.contains(&format!(
                    r#"sgl_router_sampling_contract_rejections_total{{param="{name}"}} 1"#
                )),
                "{name} must be counted:\n{rendered}"
            );
        }
    }

    #[test]
    fn overlong_plain_numeric_string_is_not_parsed() {
        let long = "1".repeat(MAX_SAMPLING_NUMERIC_LEN + 1);
        assert_eq!(parse_as_engine_number(&long), None);
        assert_eq!(
            parse_as_engine_number(&"1".repeat(MAX_SAMPLING_NUMERIC_LEN)),
            "1".repeat(MAX_SAMPLING_NUMERIC_LEN).parse::<f64>().ok(),
            "a value at the cap is still read"
        );
    }

    #[test]
    fn spliced_value_outranks_an_explicit_null_the_client_sent() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#);
        let raw = r#"{"model":"x","temperature":null}"#;
        let body = Bytes::copy_from_slice(raw.as_bytes());
        let inject = apply_sampling_overrides(&overrides, &probe_of(raw), &metrics()).unwrap();
        assert_eq!(inject.len(), 1, "null must be treated as omitted");

        let out = build_outgoing_body(&body, None, None, None, &inject).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            parsed.get("temperature"),
            Some(&serde_json::json!(1.0)),
            "the engine must read the configured value, not the client's null: {}",
            std::str::from_utf8(&out).unwrap()
        );
    }

    #[test]
    fn splice_fires_even_when_a_parse_is_already_on_hand() {
        let body = Bytes::from_static(br#"{ "model" : "x" }"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let inject = apply_sampling_overrides(
            &overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#),
            &probe_of(r#"{"model":"x"}"#),
            &metrics(),
        )
        .unwrap();

        let out = build_outgoing_body(&body, Some(value), None, None, &inject).unwrap();
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            r#"{ "model" : "x" ,"temperature":1.0}"#
        );
    }
}
