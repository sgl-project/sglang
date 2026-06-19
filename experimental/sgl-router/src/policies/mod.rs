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

    /// Produce the routing token sequence from the already-parsed request
    /// body, computed once at ingress and reused by the selection decision
    /// (and, when engine-equivalent, forwarded to the engine). Default:
    /// `None` — load-only policies (round-robin, random, power-of-two,
    /// load-based) and sticky don't tokenize. Only the cache-aware policy
    /// overrides this.
    fn request_tokens(&self, _model: &ModelId, _body: &serde_json::Value) -> Option<RequestTokens> {
        None
    }

    /// Whether this policy consumes [`Self::request_tokens`]. Lets the ingress
    /// skip the full-body JSON parse + tokenization for load-only policies
    /// (round-robin, random, power-of-two, load-based, sticky), which read only
    /// `workers` / `routing_key`. Default `false`; only the cache-aware policy
    /// overrides it.
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
