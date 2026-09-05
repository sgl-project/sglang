// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod active_load;
pub mod admission;
pub mod cache_aware;
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

    /// Attaches the Session-Aware session ID.
    pub fn with_session_id(mut self, session_id: Option<&'a str>) -> Self {
        self.session_id = session_id;
        self
    }

    /// Attaches this policy evaluation's candidate range ID.
    pub fn with_candidate_range_id(mut self, candidate_range_id: &'a str) -> Self {
        self.candidate_range_id = candidate_range_id;
        self
    }

    /// Attaches the request input-token count.
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

    /// Attaches the Engine Load snapshot captured at request start.
    pub fn with_load_snapshot(mut self, load_snapshot: &'a EngineLoadSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    /// Disables affinity lookup and assignment.
    pub fn without_affinity_lookup(mut self) -> Self {
        self.affinity_lookup_enabled = false;
        self.affinity_assignment_enabled = false;
        self
    }

    /// Keeps affinity lookup but disables assignment writes.
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

/// A policy's primary/backup proposal.
#[derive(Clone)]
pub struct SelectionProposal {
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub kind: ProposalKind,
    /// Workers still eligible for fallback after filtering.
    pub eligible_workers: Option<Vec<Arc<Worker>>>,
}

/// A Cache-Aware Prefill candidate with `E = L - H`.
#[derive(Clone)]
pub struct CacheCandidate {
    pub worker: Arc<Worker>,
    pub matched_prefix_tokens: u64,
    pub uncached_tokens: u64,
    /// Candidate domain.
    pub candidate_range_id: String,
    /// Optional pending-Prefill limit checked with `E`.
    pub max_pending_prefill_tokens: Option<u64>,
}

/// A bounded Cache-Aware candidate set.
#[derive(Clone)]
pub struct CacheCandidateProposal {
    pub candidates: Vec<CacheCandidate>,
    pub cache_switch_margin_tokens: u64,
}

/// A Prefill policy result: a pair or Cache-Aware candidates.
#[derive(Clone)]
pub enum PrefillProposal {
    Pair(SelectionProposal),
    CacheCandidates(CacheCandidateProposal),
}

impl PrefillProposal {
    /// Applies EligibilityFilter results to either proposal form.
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
    /// Creates a proposal without a backup.
    pub fn primary(primary: Arc<Worker>) -> Self {
        Self {
            primary,
            backup: None,
            kind: ProposalKind::Generic,
            eligible_workers: None,
        }
    }

    /// Creates a primary/backup proposal.
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

/// The source of a primary/backup proposal.
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

    /// Whether routing needs one request-scoped Engine Load snapshot.
    fn needs_load_snapshot(&self) -> bool {
        self.uses_shared_prefill_admission()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AffinityConfig, SessionAffinityMode};
    use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::admission::{
        resolve_cache_candidates, resolve_prefill, CandidateRange, DecisionReason, FreshLoadLookup,
    };
    use crate::policies::cache_aware::CacheAwarePolicy;
    use crate::policies::engine_load::{EngineLoadSnapshot, EngineWorkerLoad};
    use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
    use crate::policies::round_robin::RoundRobinPolicy;
    use crate::policies::session_aware::SessionAwarePolicy;
    use std::collections::HashMap;
    use std::time::Instant;

    /// Aggregated `LoadStat` values used only by policy tests.
    #[derive(Clone, Default)]
    struct TestEngineLoad {
        num_running_reqs: u64,
        num_waiting_reqs: u64,
        num_tokens: u64,
        max_total_num_tokens: u64,
    }

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: None,
        }))
    }

    #[test]
    fn default_proposal_preserves_legacy_single_worker_selection() {
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None);
        let only = worker("only");
        let policy = PowerOfTwoChoicesPolicy::new();

        let proposal = policy
            .propose(&[Arc::clone(&only)], &ctx)
            .expect("one candidate must produce a proposal");
        assert_eq!(proposal.primary.id, only.id);
        assert!(proposal.backup.is_none());
    }

    #[test]
    fn only_step_one_policies_opt_into_shared_prefill_admission() {
        assert!(PowerOfTwoChoicesPolicy::new().uses_shared_prefill_admission());
        assert!(SessionAwarePolicy::new(AffinityConfig::default()).uses_shared_prefill_admission());
        assert!(CacheAwarePolicy::new(AffinityConfig::default()).uses_shared_prefill_admission());
        assert!(!RoundRobinPolicy::new().uses_shared_prefill_admission());
    }

    #[test]
    fn shared_prefill_admission_policies_need_a_load_snapshot() {
        assert!(PowerOfTwoChoicesPolicy::new().needs_load_snapshot());
        assert!(SessionAwarePolicy::new(AffinityConfig::default()).needs_load_snapshot());
        assert!(CacheAwarePolicy::new(AffinityConfig::default()).needs_load_snapshot());
        assert!(!RoundRobinPolicy::new().needs_load_snapshot());
    }

    #[test]
    fn power_of_two_proposal_keeps_the_other_sample_as_backup() {
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None);
        let workers = vec![worker("first"), worker("second")];
        let policy = PowerOfTwoChoicesPolicy::new();

        let proposal = policy
            .propose(&workers, &ctx)
            .expect("two candidates must produce a proposal");
        let backup = proposal.backup.expect("P2 must retain its second sample");

        assert_ne!(proposal.primary.id, backup.id);
        assert_eq!(proposal.kind, ProposalKind::PowerOfTwo);
    }

    #[test]
    fn prefill_proposal_adapter_keeps_existing_pair_semantics() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = PowerOfTwoChoicesPolicy::new();
        let ctx = SelectionContext::new(&model, None);

        let proposal = policy
            .propose_prefill(&workers, &ctx)
            .expect("P2 must produce a prefill proposal");

        let PrefillProposal::Pair(pair) = proposal else {
            panic!("existing policies must use the pair adapter");
        };
        assert_eq!(pair.kind, ProposalKind::PowerOfTwo);
        assert!(pair.backup.is_some());
    }

    #[test]
    fn cache_candidate_proposal_carries_target_specific_work() {
        let hot = worker("hot");
        let proposal = CacheCandidateProposal {
            candidates: vec![CacheCandidate {
                worker: Arc::clone(&hot),
                matched_prefix_tokens: 75,
                uncached_tokens: 25,
                candidate_range_id: "global".into(),
                max_pending_prefill_tokens: None,
            }],
            cache_switch_margin_tokens: 8,
        };

        assert_eq!(proposal.candidates[0].worker.id, hot.id);
        assert_eq!(proposal.candidates[0].matched_prefix_tokens, 75);
        assert_eq!(proposal.candidates[0].uncached_tokens, 25);
    }

    #[test]
    fn power_of_two_orders_its_sample_with_fresh_engine_load_snapshot() {
        let model = ModelId("model".into());
        let busy = worker("busy");
        let idle = worker("idle");
        let workers = vec![Arc::clone(&busy), Arc::clone(&idle)];
        let load_snapshot = snapshot(&[
            (
                &busy,
                TestEngineLoad {
                    num_waiting_reqs: 512,
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
            (
                &idle,
                TestEngineLoad {
                    num_waiting_reqs: 16,
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
        ]);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&load_snapshot);

        let proposal = PowerOfTwoChoicesPolicy::new()
            .propose(&workers, &ctx)
            .expect("two candidates must produce a proposal");

        assert_eq!(proposal.primary.id, idle.id);
        assert_eq!(
            proposal.backup.expect("P2 keeps its other sample").id,
            busy.id
        );
    }

    #[test]
    fn session_affinity_reuses_primary_and_stable_backup_without_remapping() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second"), worker("third")];
        let policy = SessionAwarePolicy::new(AffinityConfig {
            stable_pair: true,
            ..Default::default()
        });
        let ctx = SelectionContext::new(&model, None).with_session_id(Some("session-a"));

        let first = policy
            .propose(&workers, &ctx)
            .expect("a new session must get an initial P2 proposal");
        policy.commit_prefill_selection(&ctx, first.kind, &first.primary);
        let second = policy
            .propose(&workers, &ctx)
            .expect("a mapped session must produce an affinity proposal");
        let third = policy
            .propose(&workers, &ctx)
            .expect("the session assignment must remain stable");

        assert_eq!(second.kind, ProposalKind::SessionAffinity);
        assert_eq!(second.primary.id, first.primary.id);
        assert_eq!(third.primary.id, second.primary.id);
        assert_eq!(
            third.backup.expect("stable pair has backup").id,
            second.backup.expect("stable pair has backup").id,
        );
    }

    #[test]
    fn new_session_commits_the_final_capacity_admitted_worker() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = SessionAwarePolicy::new(AffinityConfig::default());
        let ctx = SelectionContext::new(&model, None).with_session_id(Some("session-a"));
        let proposal = policy
            .propose(&workers, &ctx)
            .expect("a new session produces a P2 proposal");
        let backup = proposal
            .backup
            .clone()
            .expect("two workers retain a backup");
        let loads = snapshot(&[
            (
                &proposal.primary,
                TestEngineLoad {
                    num_running_reqs: 1,
                    num_tokens: 4_090,
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
            (
                &backup,
                TestEngineLoad {
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
        ]);
        let decision = resolve_prefill(&CandidateRange::global(&workers), &proposal, 32, &loads)
            .expect("the admitted backup must become Final P");
        assert_eq!(decision.selected.id, backup.id);
        policy.commit_prefill_selection(&ctx, proposal.kind, &decision.selected);

        let mapped = policy
            .propose(&workers, &ctx)
            .expect("the next turn must reuse the actual first-turn worker");
        assert_eq!(mapped.kind, ProposalKind::SessionAffinity);
        assert_eq!(mapped.primary.id, backup.id);
    }

    #[test]
    fn read_only_affinity_probe_does_not_create_a_session_assignment() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = SessionAwarePolicy::new(AffinityConfig::default());
        let probe = SelectionContext::new(&model, None)
            .with_session_id(Some("session-a"))
            .without_affinity_assignment();

        let first = policy
            .propose(&workers, &probe)
            .expect("read-only probe still gets a P2 candidate");
        assert_eq!(first.kind, ProposalKind::PowerOfTwo);

        let normal = SelectionContext::new(&model, None).with_session_id(Some("session-a"));
        let second = policy
            .propose(&workers, &normal)
            .expect("first admitted route creates the session assignment");
        assert_eq!(second.kind, ProposalKind::PowerOfTwo);
        policy.commit_prefill_selection(&normal, second.kind, &second.primary);

        let mapped = policy
            .propose(&workers, &normal)
            .expect("subsequent route resolves the admitted assignment");
        assert_eq!(mapped.kind, ProposalKind::SessionAffinity);
    }

    #[test]
    fn bucket_scoped_session_affinity_remembers_each_bucket_independently() {
        let model = ModelId("model".into());
        let short = worker("short");
        let long = worker("long");
        let policy = SessionAwarePolicy::new(AffinityConfig {
            session_affinity_mode: SessionAffinityMode::Bucket,
            ..Default::default()
        });
        let short_ctx = SelectionContext::new(&model, None)
            .with_session_id(Some("session-a"))
            .with_candidate_range_id("p-short");
        let long_ctx = SelectionContext::new(&model, None)
            .with_session_id(Some("session-a"))
            .with_candidate_range_id("p-long");

        let short_proposal = policy
            .propose(&[Arc::clone(&short)], &short_ctx)
            .expect("short bucket creates its assignment");
        policy.commit_prefill_selection(&short_ctx, short_proposal.kind, &short_proposal.primary);
        let long_proposal = policy
            .propose(&[Arc::clone(&long)], &long_ctx)
            .expect("long bucket creates an independent assignment");
        policy.commit_prefill_selection(&long_ctx, long_proposal.kind, &long_proposal.primary);
        let returned = policy
            .propose(&[short], &short_ctx)
            .expect("returning to short bucket reuses its assignment");

        assert_eq!(returned.kind, ProposalKind::SessionAffinity);
    }

    #[test]
    fn decode_pressure_tie_is_not_broken_by_worker_id() {
        let a = worker("a");
        let z = worker("z");
        assert_eq!(
            admission::compare_decode_pressure(&a, &z, None),
            std::cmp::Ordering::Equal,
            "P2 must preserve random sampling when observable pressure is equal"
        );
    }

    #[test]
    fn cache_affinity_uses_longest_routable_prefix_holder() {
        let model = ModelId("model".into());
        let hot = worker("hot");
        let other = worker("other");
        let workers = vec![Arc::clone(&hot), Arc::clone(&other)];
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 8,
                        worker_id: "gone".into(),
                        address: "http://gone:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 6,
                        worker_id: "hot".into(),
                        address: "http://hot:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 4,
                        worker_id: "other".into(),
                        address: "http://other:30000".into(),
                    },
                ],
                best_prefix_blocks: 8,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_request_tokens(Some(&[1, 2, 3, 4, 5, 6, 7, 8]))
            .with_input_tokens(8_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig::default());

        let proposal = policy
            .propose(&workers, &ctx)
            .expect("a routable indexer hit must propose a worker");

        assert_eq!(proposal.kind, ProposalKind::CacheAffinity);
        assert_eq!(proposal.primary.id, hot.id);
    }

    #[test]
    fn cache_candidates_keep_bounded_target_specific_uncached_work() {
        let model = ModelId("model".into());
        let hot = worker("hot");
        let warm = worker("warm");
        let workers = vec![Arc::clone(&hot), Arc::clone(&warm)];
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 8,
                        worker_id: "gone".into(),
                        address: "http://gone:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 6,
                        worker_id: "hot".into(),
                        address: "http://hot:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 4,
                        worker_id: "warm".into(),
                        address: "http://warm:30000".into(),
                    },
                ],
                best_prefix_blocks: 8,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(8_000)
            .with_external_prefix(Some(&signal));
        let config = AffinityConfig {
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 2,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::new(config);

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("routable matches must produce cache candidates")
        else {
            panic!("cache hits must not be collapsed to a primary/backup pair");
        };

        assert_eq!(proposal.candidates.len(), 2);
        assert_eq!(proposal.candidates[0].worker.id, hot.id);
        assert_eq!(proposal.candidates[0].matched_prefix_tokens, 6_000);
        assert_eq!(proposal.candidates[0].uncached_tokens, 2_000);
        assert_eq!(proposal.candidates[1].worker.id, warm.id);
        assert_eq!(proposal.candidates[1].matched_prefix_tokens, 4_000);
        assert_eq!(proposal.candidates[1].uncached_tokens, 4_000);
    }

    #[test]
    fn cache_candidate_bound_keeps_the_best_k_from_a_large_match_set() {
        let model = ModelId("model".into());
        let workers: Vec<Arc<Worker>> = (0..64)
            .map(|index| worker(&format!("w{index:02}")))
            .collect();
        let matches = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| sgl_kv_indexer::PrefixMatch {
                matched_prefix_blocks: (index + 1) as u32,
                worker_id: worker.id.0.clone(),
                address: worker.url.clone(),
            })
            .collect();
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches,
                best_prefix_blocks: workers.len() as u32,
            },
            query_blocks: 64,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(64_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 4,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 4,
            ..Default::default()
        });

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("the bounded best candidates must survive")
        else {
            panic!("cache hits must retain candidate-set semantics");
        };

        assert_eq!(proposal.candidates.len(), 4);
        assert_eq!(
            proposal
                .candidates
                .iter()
                .map(|candidate| candidate.matched_prefix_tokens)
                .collect::<Vec<_>>(),
            vec![64_000, 63_000, 62_000, 61_000]
        );
    }

    #[test]
    fn equal_cache_hits_bound_by_the_captured_local_load_before_worker_id() {
        let model = ModelId("model".into());
        let workers: Vec<Arc<Worker>> = (0..8)
            .map(|index| {
                let worker = worker(&format!("w{index}"));
                worker
                    .active_requests
                    .store(8 - index, std::sync::atomic::Ordering::Relaxed);
                worker
            })
            .collect();
        let matches = workers
            .iter()
            .map(|worker| sgl_kv_indexer::PrefixMatch {
                matched_prefix_blocks: 4,
                worker_id: worker.id.0.clone(),
                address: worker.url.clone(),
            })
            .collect();
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches,
                best_prefix_blocks: 4,
            },
            query_blocks: 4,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(4_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig {
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 2,
            ..Default::default()
        });

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("equal hits must retain the least-loaded replicas")
        else {
            panic!("cache hits must retain candidate-set semantics");
        };

        assert_eq!(
            proposal
                .candidates
                .iter()
                .map(|candidate| candidate.worker.id.0.as_str())
                .collect::<Vec<_>>(),
            vec!["w7", "w6"]
        );
    }

    #[test]
    fn cache_candidate_gates_are_configurable_lower_bounds_with_and_semantics() {
        let model = ModelId("model".into());
        let half = worker("half");
        let below_ratio = worker("below-ratio");
        let workers = vec![Arc::clone(&half), Arc::clone(&below_ratio)];
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 4,
                        worker_id: "half".into(),
                        address: "http://half:30000".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        matched_prefix_blocks: 3,
                        worker_id: "below-ratio".into(),
                        address: "http://below-ratio:30000".into(),
                    },
                ],
                best_prefix_blocks: 4,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(80)
            .with_external_prefix(Some(&signal));
        let config = AffinityConfig {
            cache_affinity_min_matched_tokens: Some(30),
            cache_affinity_min_match_ratio: Some(0.5),
            cache_candidate_min_workers: 8,
            cache_candidate_max_workers: 8,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::new(config);

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("one candidate satisfies both lower bounds")
        else {
            panic!("the admitted cache candidate must retain H/E");
        };

        assert_eq!(proposal.candidates.len(), 1);
        assert_eq!(proposal.candidates[0].worker.id, half.id);
    }

    #[test]
    fn default_cache_gate_rejects_a_prefix_below_the_absolute_floor() {
        let model = ModelId("model".into());
        let weak = worker("weak");
        let workers = vec![Arc::clone(&weak)];
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![sgl_kv_indexer::PrefixMatch {
                    matched_prefix_blocks: 3,
                    worker_id: "weak".into(),
                    address: "http://weak:30000".into(),
                }],
                best_prefix_blocks: 3,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(80)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig::default());

        let proposal = policy
            .propose_prefill(&workers, &ctx)
            .expect("a weak hit must degrade to no-hit P2, not fail selection");

        assert!(
            matches!(proposal, PrefillProposal::Pair(_)),
            "the default gate must keep a tiny hit from forcing cache affinity"
        );
    }

    #[test]
    fn default_cache_gate_accepts_the_indexer_scan_cap_for_a_long_prompt() {
        let model = ModelId("model".into());
        let holder = worker("holder");
        let workers = vec![Arc::clone(&holder)];
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![sgl_kv_indexer::PrefixMatch {
                    matched_prefix_blocks: 2_048,
                    worker_id: "holder".into(),
                    address: "http://holder:30000".into(),
                }],
                best_prefix_blocks: 2,
            },
            query_blocks: 4_125,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(4_125)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig::default());

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("the default absolute gate must accept a 2048-token lower bound")
        else {
            panic!("a server-truncated long-prefix hit must not degrade to P2");
        };

        assert_eq!(proposal.candidates[0].worker.id, holder.id);
        assert_eq!(proposal.candidates[0].matched_prefix_tokens, 2_048);
        assert_eq!(proposal.candidates[0].uncached_tokens, 2_077);
    }

    #[test]
    fn cache_affinity_without_signal_degrades_to_a_plain_p2_proposal() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = CacheAwarePolicy::new(AffinityConfig::default());
        let ctx = SelectionContext::new(&model, None);

        let proposal = policy
            .propose(&workers, &ctx)
            .expect("cache miss must still route through P2");

        assert_eq!(proposal.kind, ProposalKind::PowerOfTwo);
        assert!(proposal.backup.is_some());
    }

    fn snapshot(entries: &[(&Arc<Worker>, TestEngineLoad)]) -> EngineLoadSnapshot {
        EngineLoadSnapshot::from_workers(
            1,
            entries
                .iter()
                .map(|(worker, aggregate)| {
                    (
                        worker.url.clone(),
                        EngineWorkerLoad {
                            num_running_reqs: aggregate.num_running_reqs,
                            num_waiting_reqs: aggregate.num_waiting_reqs,
                            num_tokens: aggregate.num_tokens,
                            max_total_num_tokens: aggregate.max_total_num_tokens,
                            captured_at: Instant::now(),
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        )
    }

    #[test]
    fn mixed_freshness_uses_one_captured_local_level_for_the_candidate_set() {
        let aggregate_idle = worker("aggregate-idle");
        let aggregate_busy = worker("aggregate-busy");
        let stale = worker("stale");
        aggregate_idle
            .active_requests
            .store(5, std::sync::atomic::Ordering::Relaxed);
        aggregate_busy
            .active_requests
            .store(1, std::sync::atomic::Ordering::Relaxed);
        let snapshot = snapshot(&[
            (
                &aggregate_idle,
                TestEngineLoad {
                    num_waiting_reqs: 0,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &aggregate_busy,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let lookup =
            FreshLoadLookup::new(Some(&snapshot), [&aggregate_idle, &aggregate_busy, &stale]);
        assert!(lookup.get(&aggregate_idle.id).is_some());
        assert!(lookup.get(&stale.id).is_none());
        assert_eq!(
            lookup.compare_prefill_pressure(&aggregate_idle, &aggregate_busy),
            std::cmp::Ordering::Greater,
            "one stale member makes the complete candidate set compare by the captured local level"
        );
    }

    fn cache_candidate(
        worker: &Arc<Worker>,
        matched_prefix_tokens: u64,
        uncached_tokens: u64,
        max_pending_prefill_tokens: Option<u64>,
    ) -> CacheCandidate {
        CacheCandidate {
            worker: Arc::clone(worker),
            matched_prefix_tokens,
            uncached_tokens,
            candidate_range_id: "global".into(),
            max_pending_prefill_tokens,
        }
    }

    #[test]
    fn cache_tournament_skips_capacity_exhausted_matches_and_returns_no_backup() {
        let full = worker("full");
        let winner = worker("winner");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&full, 90, 10, None),
                cache_candidate(&winner, 70, 30, None),
            ],
            cache_switch_margin_tokens: 16,
        };
        let loads = snapshot(&[
            (
                &full,
                TestEngineLoad {
                    num_running_reqs: 8,
                    num_tokens: 9_950,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &winner,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads)
            .expect("a later admitted cache match must survive");

        assert_eq!(decision.selected.id, winner.id);
        assert_eq!(decision.primary.id, winner.id);
        assert!(decision.backup.is_none());
        assert_eq!(decision.reason, DecisionReason::CacheCandidate);
    }

    #[test]
    fn cache_tournament_compares_every_admitted_challenger_before_finalizing() {
        let first = worker("first");
        let second = worker("second");
        let final_winner = worker("final-winner");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&first, 40, 60, None),
                cache_candidate(&second, 60, 40, None),
                cache_candidate(&final_winner, 80, 20, None),
            ],
            cache_switch_margin_tokens: 0,
        };
        let loads = snapshot(&[
            (
                &first,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &second,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &final_winner,
                TestEngineLoad {
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads)
            .expect("all admitted candidates must participate in the tournament");

        assert_eq!(decision.selected.id, final_winner.id);
        assert_eq!(decision.primary.id, final_winner.id);
        assert!(decision.backup.is_none());
    }

    #[test]
    fn cache_tournament_uses_uncached_work_for_pending_but_full_input_for_kv() {
        let candidate = worker("candidate");
        let proposal = CacheCandidateProposal {
            candidates: vec![cache_candidate(&candidate, 80, 20, Some(30))],
            cache_switch_margin_tokens: 16,
        };
        let pending_allows = snapshot(&[(
            &candidate,
            TestEngineLoad {
                num_waiting_reqs: 5,
                max_total_num_tokens: 1_000,
                ..TestEngineLoad::default()
            },
        )]);
        assert!(
            resolve_cache_candidates(&proposal, 100, &pending_allows).is_some(),
            "pending admission must project E=20, not L=100"
        );

        let kv_rejects = snapshot(&[(
            &candidate,
            TestEngineLoad {
                num_tokens: 30,
                num_waiting_reqs: 5,
                max_total_num_tokens: 100,
                ..TestEngineLoad::default()
            },
        )]);
        assert!(
            resolve_cache_candidates(&proposal, 100, &kv_rejects).is_none(),
            "KV safety must conservatively project the complete input L=100"
        );
    }

    #[test]
    fn cache_tournament_keeps_cache_gain_when_legacy_token_guard_is_unavailable() {
        let congested = worker("congested");
        let idle = worker("idle");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&congested, 90, 10, None),
                cache_candidate(&idle, 80, 20, None),
            ],
            cache_switch_margin_tokens: 32,
        };
        let loads = snapshot(&[
            (
                &congested,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &idle,
                TestEngineLoad {
                    num_waiting_reqs: 10,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads).unwrap();
        assert_eq!(decision.selected.id, congested.id);
    }

    #[test]
    fn cache_tournament_keeps_a_material_cache_gain_despite_pressure() {
        let hot = worker("hot");
        let idle = worker("idle");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&hot, 90, 10, None),
                cache_candidate(&idle, 20, 80, None),
            ],
            cache_switch_margin_tokens: 32,
        };
        let loads = snapshot(&[
            (
                &hot,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &idle,
                TestEngineLoad {
                    num_waiting_reqs: 10,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads).unwrap();
        assert_eq!(
            decision.selected.id, hot.id,
            "pressure may break a near tie, but must not erase a material cache-work gain"
        );
    }

    #[test]
    fn cache_tournament_uses_work_order_when_legacy_token_guard_is_unavailable() {
        let best_work = worker("best-work");
        let near_tie = worker("near-tie");
        let beyond_margin = worker("beyond-margin");
        let proposal = CacheCandidateProposal {
            // The policy supplies candidates in increasing E order. Each
            // adjacent pair is a near tie, but the last candidate is more
            // than one configured margin away from the global work minimum.
            candidates: vec![
                cache_candidate(&best_work, 100, 0, None),
                cache_candidate(&near_tie, 80, 20, None),
                cache_candidate(&beyond_margin, 60, 40, None),
            ],
            cache_switch_margin_tokens: 32,
        };
        let loads = snapshot(&[
            (
                &best_work,
                TestEngineLoad {
                    num_waiting_reqs: 10_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &near_tie,
                TestEngineLoad {
                    num_waiting_reqs: 1_000,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
            (
                &beyond_margin,
                TestEngineLoad {
                    num_waiting_reqs: 0,
                    max_total_num_tokens: 10_000,
                    ..TestEngineLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads).unwrap();
        assert_eq!(
            decision.selected.id, best_work.id,
            "without a unit-compatible token-pressure signal, cache work remains authoritative"
        );
    }

    #[test]
    fn admission_uses_admitted_backup_before_scanning_candidate_range() {
        let primary = worker("primary");
        let backup = worker("backup");
        let fallback = worker("fallback");
        let workers = vec![
            Arc::clone(&primary),
            Arc::clone(&backup),
            Arc::clone(&fallback),
        ];
        let snapshot = snapshot(&[
            (
                &primary,
                TestEngineLoad {
                    num_running_reqs: 4,
                    num_tokens: 990,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                TestEngineLoad {
                    num_tokens: 10,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &fallback,
                TestEngineLoad {
                    num_tokens: 10,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let range = CandidateRange::global(&workers);
        let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

        let decision = resolve_prefill(&range, &proposal, 32, &snapshot)
            .expect("an admitted backup must be selected");

        assert_eq!(decision.selected.id, backup.id);
        assert_eq!(decision.reason, DecisionReason::BackupPrimaryAdmission);
    }

    #[test]
    fn missing_engine_snapshot_does_not_hard_reject_a_registry_healthy_primary() {
        let primary = worker("primary");
        let workers = vec![Arc::clone(&primary)];
        let snapshot = EngineLoadSnapshot::default();

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &SelectionProposal::primary(Arc::clone(&primary)),
            1_000_000,
            &snapshot,
        )
        .expect("disabled reporting must preserve the healthy registry candidate");

        assert_eq!(decision.selected.id, primary.id);
        assert_eq!(decision.reason, DecisionReason::Primary);
    }

    #[test]
    fn prefill_pair_keeps_primary_when_both_workers_fit_capacity() {
        let primary = worker("primary");
        let backup = worker("backup");
        let workers = vec![Arc::clone(&primary), Arc::clone(&backup)];
        let snapshot = snapshot(&[
            (
                &primary,
                TestEngineLoad {
                    num_waiting_reqs: 200,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                TestEngineLoad {
                    num_waiting_reqs: 20,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let proposal = SelectionProposal::with_backup(primary, backup);

        let decision = resolve_prefill(&CandidateRange::global(&workers), &proposal, 80, &snapshot)
            .expect("both candidates fit capacity");

        assert_eq!(decision.reason, DecisionReason::Primary);
    }

    #[test]
    fn admission_scans_range_only_after_primary_and_backup_both_fail() {
        let primary = worker("primary");
        let backup = worker("backup");
        let fallback = worker("fallback");
        let workers = vec![
            Arc::clone(&primary),
            Arc::clone(&backup),
            Arc::clone(&fallback),
        ];
        let snapshot = snapshot(&[
            (
                &primary,
                TestEngineLoad {
                    num_running_reqs: 4,
                    num_tokens: 990,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                TestEngineLoad {
                    num_tokens: 990,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &fallback,
                TestEngineLoad {
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let proposal = SelectionProposal::with_backup(primary, backup);

        let decision = resolve_prefill(&CandidateRange::global(&workers), &proposal, 32, &snapshot)
            .expect("an admitted range fallback must be selected");

        assert_eq!(decision.selected.id, fallback.id);
        assert_eq!(decision.reason, DecisionReason::RangeFallback);
    }
}
