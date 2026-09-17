// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Routing strategies and their shared selection workflow.
//!
//! Request handling enters `selection`; policies supply pair preferences or an
//! evaluated cache decision. `admission` owns shared constraints, while worker
//! observations and KV indexing live outside the policy layer.

pub mod admission;
pub mod affinity;
pub mod balancing;
pub mod buckets;
pub mod cache_aware;
pub mod factory;
pub mod scoring;
pub mod selection;

use crate::discovery::ModelId;
use crate::kv_events::PrefixSignal;
use crate::policies::admission::EligibilityFilter;
use crate::policies::buckets::{BucketRequest, BucketSelector};
use crate::policies::cache_aware::CacheSelection;
use crate::policies::scoring::ScoringPolicy;
use crate::server::metrics::MetricsRegistry;
use crate::workers::engine_reports::EngineSnapshot;
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::Arc;

/// Whether a routing attempt may consult or create affinity assignments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffinityAccess {
    Disabled,
    LookupOnly,
    LookupAndAssign,
}

/// Immutable request data consumed by a routing policy.
#[derive(Clone)]
pub struct SelectionContext<'a> {
    model: &'a ModelId,
    routable_fleet: Option<&'a [Arc<Worker>]>,
    request_body: Option<&'a [u8]>,
    routing_key: Option<&'a str>,
    session_id: Option<&'a str>,
    candidate_range_id: &'a str,
    input_tokens: Option<u64>,
    request_tokens: Option<&'a [u32]>,
    external_prefix: Option<&'a PrefixSignal>,
    load_snapshot: Option<&'a EngineSnapshot>,
    prefill_cache_bucket: Option<(&'a BucketSelector, BucketRequest)>,
    affinity_access: AffinityAccess,
}

impl<'a> SelectionContext<'a> {
    pub fn new(model: &'a ModelId, request_body: Option<&'a [u8]>) -> Self {
        Self {
            model,
            routable_fleet: None,
            request_body,
            routing_key: None,
            session_id: None,
            candidate_range_id: "global",
            input_tokens: None,
            request_tokens: None,
            external_prefix: None,
            load_snapshot: None,
            prefill_cache_bucket: None,
            affinity_access: AffinityAccess::LookupAndAssign,
        }
    }

    pub fn with_routing_key(
        model: &'a ModelId,
        request_body: Option<&'a [u8]>,
        routing_key: Option<&'a str>,
    ) -> Self {
        Self {
            model,
            routable_fleet: None,
            request_body,
            routing_key,
            session_id: None,
            candidate_range_id: "global",
            input_tokens: None,
            request_tokens: None,
            external_prefix: None,
            load_snapshot: None,
            prefill_cache_bucket: None,
            affinity_access: AffinityAccess::LookupAndAssign,
        }
    }

    /// Attaches ingress-computed routing tokens.
    pub fn with_request_tokens(mut self, request_tokens: Option<&'a [u32]>) -> Self {
        self.request_tokens = request_tokens;
        self
    }

    /// Attaches a Session-Aware session ID.
    pub fn with_session_id(mut self, session_id: Option<&'a str>) -> Self {
        self.session_id = session_id;
        self
    }

    /// Identifies the candidate domain for this policy call.
    pub fn with_candidate_range_id(mut self, candidate_range_id: &'a str) -> Self {
        self.candidate_range_id = candidate_range_id;
        self
    }

    /// Attaches the request input token count.
    pub fn with_input_tokens(mut self, input_tokens: u64) -> Self {
        self.input_tokens = Some(input_tokens);
        self
    }

    pub fn with_external_prefix(mut self, external_prefix: Option<&'a PrefixSignal>) -> Self {
        self.external_prefix = external_prefix;
        self
    }

    /// Attaches the engine load snapshot captured at request ingress.
    pub fn with_load_snapshot(mut self, load_snapshot: &'a EngineSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    /// Cache-Aware applies Bucket constraints before Top-K truncation so an
    /// incompatible cache holder cannot displace a lower-ranked usable one.
    pub fn with_prefill_cache_bucket(
        mut self,
        selector: &'a BucketSelector,
        request: BucketRequest,
    ) -> Self {
        self.prefill_cache_bucket = Some((selector, request));
        self
    }

    pub fn with_affinity_access(mut self, access: AffinityAccess) -> Self {
        self.affinity_access = access;
        self
    }

    /// Disables affinity lookup and assignment.
    pub fn without_affinity_lookup(mut self) -> Self {
        self.affinity_access = AffinityAccess::Disabled;
        self
    }

    /// Disables assignment without re-enabling a disabled lookup.
    pub fn without_affinity_assignment(mut self) -> Self {
        if self.affinity_access != AffinityAccess::Disabled {
            self.affinity_access = AffinityAccess::LookupOnly;
        }
        self
    }

    pub fn with_routable_fleet(mut self, workers: &'a [Arc<Worker>]) -> Self {
        self.routable_fleet = Some(workers);
        self
    }

    pub fn routable_fleet(&self) -> Option<&[Arc<Worker>]> {
        self.routable_fleet
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

    pub fn external_prefix(&self) -> Option<&PrefixSignal> {
        self.external_prefix
    }

    pub fn load_snapshot(&self) -> Option<&EngineSnapshot> {
        self.load_snapshot
    }

    pub fn prefill_cache_bucket(&self) -> Option<(&BucketSelector, BucketRequest)> {
        self.prefill_cache_bucket
    }

    pub fn affinity_lookup_enabled(&self) -> bool {
        self.affinity_access != AffinityAccess::Disabled
    }

    pub fn affinity_assignment_enabled(&self) -> bool {
        self.affinity_access == AffinityAccess::LookupAndAssign
    }
}

/// Primary and backup workers proposed by a policy.
#[derive(Clone)]
pub struct SelectionProposal {
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub kind: ProposalKind,
    /// Optional pressure guard settings for this pair.
    /// Applied only when both workers have complete, fresh native monitor data.
    pub guard_hints: GuardHints,
    /// Workers available for fallback after eligibility filtering.
    pub eligible_workers: Option<Vec<Arc<Worker>>>,
}

/// A pair for shared admission, or a completed cache evaluation with fallback audit.
#[derive(Clone)]
pub enum PrefillEvaluation {
    Pair(SelectionProposal),
    Cache(CacheSelection),
}

impl PrefillEvaluation {
    /// Carries the eligible fallback pool; cache evaluations already used this pool.
    pub fn with_eligible_workers(self, workers: Vec<Arc<Worker>>) -> Self {
        match self {
            Self::Pair(proposal) => Self::Pair(proposal.with_eligible_workers(workers)),
            Self::Cache(selection) => {
                debug_assert!(selection.candidates.iter().all(|candidate| workers
                    .iter()
                    .any(|worker| worker.id == candidate.worker.id)));
                Self::Cache(selection)
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
            guard_hints: GuardHints::default(),
            eligible_workers: None,
        }
    }

    /// Creates a primary/backup proposal.
    pub fn with_backup(primary: Arc<Worker>, backup: Arc<Worker>) -> Self {
        Self {
            primary,
            backup: Some(backup),
            kind: ProposalKind::PowerOfTwo,
            guard_hints: GuardHints::default(),
            eligible_workers: None,
        }
    }

    pub fn with_kind(mut self, kind: ProposalKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_guard_hints(mut self, guard_hints: GuardHints) -> Self {
        self.guard_hints = guard_hints;
        self
    }

    pub fn with_eligible_workers(mut self, workers: Vec<Arc<Worker>>) -> Self {
        self.eligible_workers = Some(workers);
        self
    }
}

/// Source of a primary/backup proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalKind {
    Generic,
    PowerOfTwo,
    SessionAffinity,
    CacheAffinity,
    Score,
}

/// Optional guard settings for a pair proposal.
#[derive(Debug, Clone)]
pub struct GuardHints {
    pub enable_pressure_guard: bool,
    pub pressure_abs_threshold_tokens: u64,
    pub pressure_abs_threshold_ms: Option<f64>,
    pub pressure_rel_threshold: f64,
}

impl Default for GuardHints {
    fn default() -> Self {
        Self {
            enable_pressure_guard: false,
            pressure_abs_threshold_tokens: 0,
            pressure_abs_threshold_ms: None,
            pressure_rel_threshold: 1.0,
        }
    }
}

pub trait Policy: Send + Sync + std::fmt::Debug {
    /// Returns an unconstrained preference. Request routing uses the workflow
    /// to apply admission, domain fallback, and final affinity commitment.
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>>;

    /// Produces a provisional pair, including for affinity probes.
    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        self.select(workers, ctx).map(SelectionProposal::primary)
    }

    /// Evaluates prefill routing. Cache policies resolve their own candidates;
    /// pair proposals enter shared admission in the routing workflow.
    fn evaluate_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillEvaluation> {
        self.propose(workers, ctx).map(PrefillEvaluation::Pair)
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

    /// Whether in-flight requests must be timestamped for load correction.
    fn needs_dispatch_timestamps(&self) -> bool {
        false
    }

    /// Whether this policy resolves an affinity primary within its candidate range.
    fn resolves_affinity_in_range(&self) -> bool {
        false
    }

    /// Whether this policy's routing decision needs request tokens (i.e.
    /// it routes by prompt prefix). Ingress tokenization itself is no longer
    /// gated on this — that is a model property (`has_chat_formatter`) decided at
    /// ingress via [`crate::tokenizer::request_tokens_for`]. This flag is the EXTRA gate that
    /// keeps the cache-aware policy's RAW-prompt routing path alive: a
    /// cache-aware model with no chat formatter still wants its `/v1/completions`
    /// /`text` prompt tokenized for tree matching, which `has_chat_formatter`
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
pub(crate) mod test_support {
    use super::*;
    use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::cache_aware::CacheCandidate;
    use crate::workers::engine_reports::NativeCacheWorkerLoad;
    use std::collections::HashMap;
    use std::time::Instant;
    #[derive(Clone, Default)]
    pub(crate) struct TestEngineLoad {
        pub(crate) num_running_reqs: u64,
        pub(crate) num_waiting_reqs: u64,
        pub(crate) num_tokens: u64,
        pub(crate) max_total_num_tokens: u64,
        pub(crate) num_waiting_uncached_tokens: Option<u64>,
        pub(crate) num_total_tokens: Option<u64>,
        pub(crate) max_running_requests: Option<u64>,
    }
    pub(crate) fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: None,
        }))
    }
    pub(crate) fn snapshot(entries: &[(&Arc<Worker>, TestEngineLoad)]) -> EngineSnapshot {
        EngineSnapshot::from_native_cache_workers(
            1,
            entries
                .iter()
                .map(|(worker, aggregate)| {
                    (
                        worker.url.clone(),
                        NativeCacheWorkerLoad {
                            num_running_reqs: aggregate.num_running_reqs,
                            num_waiting_reqs: aggregate.num_waiting_reqs,
                            num_waiting_uncached_tokens: aggregate
                                .num_waiting_uncached_tokens
                                .unwrap_or(aggregate.num_waiting_reqs),
                            num_used_tokens: aggregate.num_tokens,
                            num_total_tokens: aggregate
                                .num_total_tokens
                                .unwrap_or(aggregate.num_tokens),
                            max_total_num_tokens: aggregate.max_total_num_tokens,
                            max_running_requests: aggregate.max_running_requests.unwrap_or(64),
                            prefill_throughput_tokens_per_s: None,
                            estimated_prefill_queue_ms: None,
                            captured_at: Instant::now(),
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        )
    }
    pub(crate) fn cache_candidate(
        worker: &Arc<Worker>,
        matched_prefix_tokens: u64,
        uncached_tokens: u64,
        max_pending_prefill_tokens: Option<u64>,
    ) -> CacheCandidate {
        CacheCandidate {
            worker: Arc::clone(worker),
            matched_prefix_tokens,
            uncached_tokens,
            matched_prefix_blocks: 0,
            candidate_range_id: "global".into(),
            max_pending_prefill_tokens,
        }
    }
}

#[cfg(test)]
mod proposal_tests {
    use crate::config::AffinityConfig;
    use crate::policies::affinity::SessionAwarePolicy;
    use crate::policies::balancing::PowerOfTwoChoicesPolicy;
    use crate::policies::balancing::RoundRobinPolicy;
    use crate::policies::cache_aware::CacheAwarePolicy;
    use crate::policies::test_support::worker;
    use crate::policies::*;
    #[test]
    fn disabling_assignment_does_not_reenable_affinity_lookup() {
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None)
            .without_affinity_lookup()
            .without_affinity_assignment();
        assert!(!ctx.affinity_lookup_enabled());
        assert!(!ctx.affinity_assignment_enabled());
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
    fn prefill_proposal_adapter_keeps_existing_pair_semantics() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = PowerOfTwoChoicesPolicy::new();
        let ctx = SelectionContext::new(&model, None);

        let proposal = policy
            .evaluate_prefill(&workers, &ctx)
            .expect("P2 must produce a prefill proposal");

        let PrefillEvaluation::Pair(pair) = proposal else {
            panic!("existing policies must use the pair adapter");
        };
        assert_eq!(pair.kind, ProposalKind::PowerOfTwo);
        assert!(pair.backup.is_some());
    }
}
