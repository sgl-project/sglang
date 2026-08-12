// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod active_load;
pub mod cache_aware_zmq;
pub mod factory;
pub mod kv_events;
pub mod load_based;
pub mod power_of_two;
pub mod random;
pub mod registry;
pub mod round_robin;
pub mod sticky;

use crate::discovery::ModelId;
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::{adapter, TokenizerRegistry};
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::Arc;

/// Tokens produced once at ingress for a request. Consumed by the
/// cache-aware selection decision and, when `engine_equivalent`, forwarded
/// to the engine as `input_ids` so the engine skips its own prompt
/// tokenization (the router and engine would otherwise tokenize the same
/// prompt twice in the same cluster).
pub struct RequestTokens {
    /// The prompt token ids.
    pub ids: Vec<u32>,
    /// True only when the ids were produced via the model's chat encoder —
    /// i.e. they match what the engine would tokenize from the chat
    /// template. False for the raw-prompt fallback, where the engine must
    /// tokenize the text itself, so the ids are NOT safe to forward.
    pub engine_equivalent: bool,
}

/// Produce the routing tokens — and whether they are engine-equivalent —
/// from an already-parsed request body, using the shared tokenizer registry.
///
/// Tokenization is a property of the MODEL (does it have a chat encoder?),
/// not of the routing policy, so this lives here as a free function the
/// ingress calls directly with `ctx.tokenizers` — every policy (sticky,
/// round-robin, cache-aware) shares one tokenize. The cache-aware policy also
/// calls it as a body-tokenize fallback for callers that didn't pre-tokenize.
///
/// Chat requests (`messages`) on a model that has a chat encoder are rendered
/// through that encoder and tokenized the way the engine does, so the query
/// hashes match the engine's cached blocks (chat-templated tokens) AND the ids
/// are safe to hand the engine as `input_ids` (`engine_equivalent = true`).
/// Everything else — `/v1/completions` (`prompt`), `/generate` (`text`), or a
/// chat model without an encoder — tokenizes the raw extracted prompt text;
/// those ids only match the engine after it applies its own template, so they
/// are NOT engine-equivalent. A failed encoder render/encode falls through to
/// the raw path rather than failing the request.
pub fn request_tokens_for(
    tokenizers: &TokenizerRegistry,
    model_id: &ModelId,
    value: &serde_json::Value,
) -> Option<RequestTokens> {
    if tokenizers.has_chat_encoder(&model_id.0) {
        if let Some(messages) = value.get("messages").filter(|m| m.is_array()) {
            if let Some(ids) = tokenizers.encode_chat(&model_id.0, messages) {
                return Some(RequestTokens {
                    ids,
                    engine_equivalent: true,
                });
            }
        }
    }
    let text = extract_prompt_text_from_value(value)?;
    let ids = tokenize_text(tokenizers, model_id, &text)?;
    Some(RequestTokens {
        ids,
        engine_equivalent: false,
    })
}

/// Tokenize `text` for `model_id` via the shared registry. Returns `None` if
/// no tokenizer is loaded (the model_id may be misconfigured) or if encoding
/// fails / yields no tokens. An encode error logs at WARN (a loaded-but-erroring
/// tokenizer silently disables the offload); the no-text / empty-output paths
/// are expected and stay quiet.
fn tokenize_text(
    tokenizers: &TokenizerRegistry,
    model_id: &ModelId,
    text: &str,
) -> Option<Vec<u32>> {
    let tokenizer = tokenizers.get(&model_id.0)?;
    match adapter::encode(&tokenizer, text) {
        Ok(ids) if !ids.is_empty() => Some(ids),
        Ok(_) => None,
        Err(e) => {
            // WARN, not DEBUG: a tokenizer that is loaded but consistently
            // erroring silently turns the whole tokenization offload into a
            // no-op, so the failure must be visible above DEBUG. Sustained
            // failure logs once per request; the volume signal is the
            // `sgl_router_ingress_tokenize_errors_total` counter (which the
            // chat handler bumps on the chat-encode failure), so no
            // rate-limiter here.
            tracing::warn!(
                model = %model_id,
                error = %e,
                "ingress tokenize failed; routing/forwarding skips this prompt",
            );
            None
        }
    }
}

/// Extract a raw prompt-text candidate from an already-parsed JSON request
/// body. Returns `None` when there's no routable text field; the caller then
/// skips tokenization. This is the raw path — chat requests on a model with a
/// chat encoder are rendered via the encoder instead (see [`request_tokens_for`]).
///
/// Supported shapes (in priority order):
///   1. `"prompt": "..."` — `/v1/completions`-style.
///   2. `"prompt": ["...", "..."]` — `/v1/completions` array form;
///      concatenated with `"\n"`.
///   3. `"messages": [{"content": "..."}]` — `/v1/chat/completions`
///      with string content; concatenated with `"\n"`.
///   4. `"messages": [{"content": [{"text": "..."}]}]` — chat with
///      multimodal content blocks; text-only blocks concatenated.
///   5. `"text": "..."` — SGLang `/generate` native form.
///
/// Anything else yields `None`.
pub(crate) fn extract_prompt_text_from_value(v: &serde_json::Value) -> Option<String> {
    if let Some(s) = v.get("prompt").and_then(|p| p.as_str()) {
        return Some(s.to_string());
    }
    if let Some(arr) = v.get("prompt").and_then(|p| p.as_array()) {
        let parts: Vec<&str> = arr.iter().filter_map(|x| x.as_str()).collect();
        if !parts.is_empty() {
            return Some(parts.join("\n"));
        }
    }
    if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        let mut buf = String::new();
        for m in msgs {
            match m.get("content") {
                Some(serde_json::Value::String(s)) => {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(s);
                }
                Some(serde_json::Value::Array(parts)) => {
                    for part in parts {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            if !buf.is_empty() {
                                buf.push('\n');
                            }
                            buf.push_str(t);
                        }
                    }
                }
                _ => {}
            }
        }
        if !buf.is_empty() {
            return Some(buf);
        }
    }
    if let Some(s) = v.get("text").and_then(|t| t.as_str()) {
        return Some(s.to_string());
    }
    None
}

/// Selection input — carries the request body and the routing tokens
/// (computed once at ingress) so cache-aware policies can hash prefix
/// tokens without reshaping the [`Policy`] trait or re-tokenizing.  Today's
/// load-only policies (round-robin, random, power-of-two, load-based) read
/// only `workers`; sticky reads `routing_key`.
///
/// Constructed via [`Self::new`] / [`Self::with_routing_key`]; the
/// ingress-computed tokens are attached with [`Self::with_request_tokens`].
/// Accessors expose immutable references so callers cannot mutate the model
/// id or swap in a different body without going through the constructor.
pub struct SelectionContext<'a> {
    model: &'a ModelId,
    request_body: Option<&'a [u8]>,
    routing_key: Option<&'a str>,
    request_tokens: Option<&'a [u32]>,
}

impl<'a> SelectionContext<'a> {
    pub fn new(model: &'a ModelId, request_body: Option<&'a [u8]>) -> Self {
        Self {
            model,
            request_body,
            routing_key: None,
            request_tokens: None,
        }
    }

    pub fn with_routing_key(
        model: &'a ModelId,
        request_body: Option<&'a [u8]>,
        routing_key: Option<&'a str>,
    ) -> Self {
        Self {
            model,
            request_body,
            routing_key,
            request_tokens: None,
        }
    }

    /// Attach the ingress-computed routing tokens. When present, the
    /// cache-aware policy consumes these instead of re-parsing and
    /// re-tokenizing the body.
    pub fn with_request_tokens(mut self, request_tokens: Option<&'a [u32]>) -> Self {
        self.request_tokens = request_tokens;
        self
    }

    pub fn model(&self) -> &ModelId {
        self.model
    }

    pub fn request_body(&self) -> Option<&[u8]> {
        self.request_body
    }

    pub fn routing_key(&self) -> Option<&str> {
        self.routing_key
    }

    /// Ingress-precomputed routing tokens, if any. `None` means the policy
    /// must derive tokens itself (e.g. a caller that didn't pre-tokenize).
    pub fn request_tokens(&self) -> Option<&[u32]> {
        self.request_tokens
    }
}

pub trait Policy: Send + Sync + std::fmt::Debug {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>>;

    /// Whether this policy's ROUTING decision needs the request tokens (i.e.
    /// it routes by prompt prefix). Ingress tokenization itself is no longer
    /// gated on this — that is a model property (`has_chat_encoder`) decided at
    /// ingress via [`request_tokens_for`]. This flag is the EXTRA gate that
    /// keeps the cache-aware policy's RAW-prompt routing path alive: a
    /// cache-aware model with no chat encoder still wants its `/v1/completions`
    /// /`text` prompt tokenized for tree matching, which `has_chat_encoder`
    /// alone would not trigger. Default `false` (load-only + sticky route
    /// without prefix tokens); only the cache-aware policy overrides it.
    fn needs_request_tokens(&self) -> bool {
        false
    }

    /// Attach the process metrics registry after construction. Default is a
    /// no-op — only policies that emit metrics (cache-aware-zmq's
    /// `sgl_router_overlap_blocks`) override it. Mirrors
    /// `ActiveLoadRegistry::attach_metrics`: the registry is built after the
    /// policies, so it is injected here rather than passed to the constructor.
    fn attach_metrics(&self, _metrics: Arc<MetricsRegistry>) {}
}

#[derive(Debug, Default)]
pub struct PolicyRegistry {
    by_model: DashMap<ModelId, Arc<dyn Policy>>,
}

impl PolicyRegistry {
    pub fn insert(&self, model: ModelId, policy: Arc<dyn Policy>) {
        self.by_model.insert(model, policy);
    }

    pub fn get(&self, model: &ModelId) -> Option<Arc<dyn Policy>> {
        self.by_model.get(model).map(|p| p.clone())
    }

    /// Inject the metrics registry into every registered policy. Called once
    /// at startup (after the registry is built) so metrics-emitting policies
    /// can record into the shared registry.
    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        for entry in self.by_model.iter() {
            entry.value().attach_metrics(Arc::clone(&metrics));
        }
    }
}
