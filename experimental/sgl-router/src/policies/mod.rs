// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod active_load;
pub mod admission;
pub mod cache_aware_zmq;
pub mod engine_load;
pub mod factory;
pub mod kv_events;
pub mod load_based;
pub mod power_of_two;
pub mod random;
pub mod registry;
pub mod round_robin;
pub mod scoring;
pub mod session_aware;
pub mod sticky;

use crate::discovery::ModelId;
use crate::policies::engine_load::EngineLoadSnapshot;
use crate::policies::scoring::{EligibilityFilter, ScoringPolicy};
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::{adapter, TokenizerRegistry};
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::Arc;

/// Tokens produced at ingress for routing and optional engine forwarding.
pub struct RequestTokens {
    /// The prompt token ids.
    pub ids: Vec<u32>,
    /// Whether the token ids are safe to forward as engine `input_ids`.
    pub engine_equivalent: bool,
}

/// External indexer answer prepared by the async ingress path for the
/// synchronous cache-aware policy.
pub struct ExternalPrefixSignal {
    pub outcome: sgl_kv_indexer::PrefixOutcome,
    pub query_blocks: usize,
}

/// Tokenizes a request for routing. Chat-encoder tokens are engine-equivalent;
/// raw prompt tokens are used only for routing.
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

/// Tokenizes text with the model tokenizer.
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
            tracing::warn!(
                model = %model_id,
                error = %e,
                "ingress tokenize failed; routing/forwarding skips this prompt",
            );
            None
        }
    }
}

/// Extracts raw prompt text from a parsed request body.
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

/// Immutable request data consumed by a routing policy.
#[derive(Clone)]
pub struct SelectionContext<'a> {
    model: &'a ModelId,
    request_body: Option<&'a [u8]>,
    routing_key: Option<&'a str>,
    session_id: Option<&'a str>,
    candidate_range_id: &'a str,
    input_tokens: Option<u64>,
    request_tokens: Option<&'a [u32]>,
    external_prefix: Option<&'a ExternalPrefixSignal>,
    load_snapshot: Option<&'a EngineLoadSnapshot>,
    affinity_lookup_enabled: bool,
    affinity_assignment_enabled: bool,
}

impl<'a> SelectionContext<'a> {
    pub fn new(model: &'a ModelId, request_body: Option<&'a [u8]>) -> Self {
        Self {
            model,
            request_body,
            routing_key: None,
            session_id: None,
            candidate_range_id: "global",
            input_tokens: None,
            request_tokens: None,
            external_prefix: None,
            load_snapshot: None,
            affinity_lookup_enabled: true,
            affinity_assignment_enabled: true,
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
            session_id: None,
            candidate_range_id: "global",
            input_tokens: None,
            request_tokens: None,
            external_prefix: None,
            load_snapshot: None,
            affinity_lookup_enabled: true,
            affinity_assignment_enabled: true,
        }
    }

    /// Attaches ingress-computed routing tokens.
    pub fn with_request_tokens(mut self, request_tokens: Option<&'a [u32]>) -> Self {
        self.request_tokens = request_tokens;
        self
    }

    /// 附加 Session-Aware session id。
    pub fn with_session_id(mut self, session_id: Option<&'a str>) -> Self {
        self.session_id = session_id;
        self
    }

    /// 标识本次 policy 的候选域。
    pub fn with_candidate_range_id(mut self, candidate_range_id: &'a str) -> Self {
        self.candidate_range_id = candidate_range_id;
        self
    }

    /// 附加请求 input token 数。
    pub fn with_input_tokens(mut self, input_tokens: u64) -> Self {
        self.input_tokens = Some(input_tokens);
        self
    }

    pub fn with_external_prefix(
        mut self,
        external_prefix: Option<&'a ExternalPrefixSignal>,
    ) -> Self {
        self.external_prefix = external_prefix;
        self
    }

    /// 附加请求开始时捕获的 Engine Load snapshot。
    pub fn with_load_snapshot(mut self, load_snapshot: &'a EngineLoadSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    /// 禁用 affinity lookup 和 assignment。
    pub fn without_affinity_lookup(mut self) -> Self {
        self.affinity_lookup_enabled = false;
        self.affinity_assignment_enabled = false;
        self
    }

    /// 保留 affinity lookup，但禁用 assignment 写入。
    pub fn without_affinity_assignment(mut self) -> Self {
        self.affinity_assignment_enabled = false;
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

    pub fn session_id(&self) -> Option<&str> {
        self.session_id
    }

    pub fn candidate_range_id(&self) -> &str {
        self.candidate_range_id
    }
    pub fn input_tokens(&self) -> Option<u64> {
        self.input_tokens
    }

    /// Returns ingress-computed routing tokens.
    pub fn request_tokens(&self) -> Option<&[u32]> {
        self.request_tokens
    }

    pub fn external_prefix(&self) -> Option<&ExternalPrefixSignal> {
        self.external_prefix
    }

    pub fn load_snapshot(&self) -> Option<&EngineLoadSnapshot> {
        self.load_snapshot
    }

    pub fn affinity_lookup_enabled(&self) -> bool {
        self.affinity_lookup_enabled
    }

    pub fn affinity_assignment_enabled(&self) -> bool {
        self.affinity_assignment_enabled
    }
}

/// Policy 产生的 primary/backup 提案。
#[derive(Clone)]
pub struct SelectionProposal {
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub kind: ProposalKind,
    /// EligibilityFilter 之后可用于 fallback 的 worker。
    pub eligible_workers: Option<Vec<Arc<Worker>>>,
}

/// 一个 Cache-Aware Prefill 候选，`E = L - H`。
#[derive(Clone)]
pub struct CacheCandidate {
    pub worker: Arc<Worker>,
    pub matched_prefix_tokens: u64,
    pub uncached_tokens: u64,
    /// 候选所属 domain。
    pub candidate_range_id: String,
    /// 使用 `E` 检查的可选 pending Prefill 上限。
    pub max_pending_prefill_tokens: Option<u64>,
}

/// 有界 Cache-Aware 候选集。
#[derive(Clone)]
pub struct CacheCandidateProposal {
    pub candidates: Vec<CacheCandidate>,
    pub cache_switch_margin_tokens: u64,
}

/// Prefill policy 返回 pair 或 Cache-Aware 候选集。
#[derive(Clone)]
pub enum PrefillProposal {
    Pair(SelectionProposal),
    CacheCandidates(CacheCandidateProposal),
}

impl PrefillProposal {
    /// 将 EligibilityFilter 结果应用到两种 proposal。
    pub fn with_eligible_workers(self, workers: Vec<Arc<Worker>>) -> Self {
        match self {
            Self::Pair(proposal) => Self::Pair(proposal.with_eligible_workers(workers)),
            Self::CacheCandidates(mut proposal) => {
                proposal.candidates.retain(|candidate| {
                    workers
                        .iter()
                        .any(|worker| worker.id == candidate.worker.id)
                });
                Self::CacheCandidates(proposal)
            }
        }
    }
}

impl SelectionProposal {
    /// 创建无 backup 的提案。
    pub fn primary(primary: Arc<Worker>) -> Self {
        Self {
            primary,
            backup: None,
            kind: ProposalKind::Generic,
            eligible_workers: None,
        }
    }

    /// 创建 primary/backup 提案。
    pub fn with_backup(primary: Arc<Worker>, backup: Arc<Worker>) -> Self {
        Self {
            primary,
            backup: Some(backup),
            kind: ProposalKind::PowerOfTwo,
            eligible_workers: None,
        }
    }

    pub fn with_kind(mut self, kind: ProposalKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_eligible_workers(mut self, workers: Vec<Arc<Worker>>) -> Self {
        self.eligible_workers = Some(workers);
        self
    }
}

/// primary/backup 的来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalKind {
    Generic,
    PowerOfTwo,
    SessionAffinity,
    CacheAffinity,
    Score,
}

pub trait Policy: Send + Sync + std::fmt::Debug {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>>;

    /// Produces a primary worker and an optional backup.
    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        self.select(workers, ctx).map(SelectionProposal::primary)
    }

    /// Produces a prefill proposal, including cache-aware candidate sets.
    fn propose_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        self.propose(workers, ctx).map(PrefillProposal::Pair)
    }

    /// Commits policy-owned affinity state after choosing the final prefill worker.
    fn commit_prefill_selection(
        &self,
        _ctx: &SelectionContext<'_>,
        _proposal_kind: ProposalKind,
        _selected: &Arc<Worker>,
    ) {
    }

    /// Indicates whether this policy uses shared prefill admission and guards.
    fn uses_shared_prefill_admission(&self) -> bool {
        false
    }

    /// Whether this policy resolves an affinity primary within the candidate range.
    fn is_bucket_affinity_policy(&self) -> bool {
        false
    }

    /// Whether this policy's routing decision needs request tokens (i.e.
    /// it routes by prompt prefix). Ingress tokenization itself is no longer
    /// gated on this — that is a model property (`has_chat_encoder`) decided at
    /// ingress via [`request_tokens_for`]. This flag is the EXTRA gate that
    /// keeps the cache-aware policy's RAW-prompt routing path alive: a
    /// cache-aware model with no chat encoder still wants its `/v1/completions`
    /// /`text` prompt tokenized for tree matching, which `has_chat_encoder`
    /// alone would not trigger. Default `false` for load-only and sticky
    /// routes; only the cache-aware policy overrides it.
    fn needs_request_tokens(&self) -> bool {
        false
    }

    /// Attaches the process metrics registry after construction.
    fn attach_metrics(&self, _metrics: Arc<MetricsRegistry>) {}

    /// Returns the optional per-worker scoring view.
    fn as_scoring(&self) -> Option<&dyn ScoringPolicy> {
        None
    }

    /// Returns the optional per-worker eligibility view.
    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        None
    }

    /// Whether this policy exposes scores for `--fuse`.
    fn can_fuse(&self) -> bool {
        self.as_scoring().is_some()
    }

    /// Whether this policy exposes an eligibility filter.
    fn can_filter(&self) -> bool {
        self.as_filter().is_some()
    }
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

    /// Attaches metrics to each registered policy.
    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        for entry in self.by_model.iter() {
            entry.value().attach_metrics(Arc::clone(&metrics));
        }
    }
}
