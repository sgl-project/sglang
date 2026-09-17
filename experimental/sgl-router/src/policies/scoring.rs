// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Scoring, selection, and composition over eligible candidates.

use crate::kv_events::{BlockSizeOracle, HashTree};
use crate::policies::admission::{apply_filters, EligibilityFilter};
use crate::policies::{Policy, PrefillEvaluation, SelectionContext, SelectionProposal};
use crate::workers::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A soft preference for eligible candidates.
pub trait ScoringPolicy: Send + Sync + std::fmt::Debug {
    /// What each candidate is worth, parallel to `workers`; higher is better.
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32>;

    /// Default multiplier as a fused term; `--fuse name=weight` overrides it.
    fn weight(&self) -> f32 {
        1.0
    }

    /// Whether scoring needs request tokens.
    fn needs_tokens(&self) -> bool {
        false
    }

    /// Whether scoring reads the request-scoped Engine Load snapshot.
    fn needs_load_snapshot(&self) -> bool {
        false
    }

    /// Whether scoring corrects Engine Load with recent dispatch timestamps.
    fn needs_dispatch_timestamps(&self) -> bool {
        false
    }

    /// Optional eligibility view for policies that provide both signals.
    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        None
    }

    /// Selects a winner from the score vector.
    fn selector(&self) -> &dyn Selector {
        &ARGMAX
    }
}

/// Selects the best-scoring worker.
impl<T: ScoringPolicy> Policy for T {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let scores = self.scores(workers, ctx);
        let i = self.selector().pick(workers, &scores)?;
        workers.get(i).map(Arc::clone)
    }

    fn needs_request_tokens(&self) -> bool {
        ScoringPolicy::needs_tokens(self) || self.as_filter().is_some_and(|f| f.needs_tokens())
    }

    fn needs_load_snapshot(&self) -> bool {
        ScoringPolicy::needs_load_snapshot(self)
    }

    fn needs_dispatch_timestamps(&self) -> bool {
        ScoringPolicy::needs_dispatch_timestamps(self)
    }

    fn as_scoring(&self) -> Option<&dyn ScoringPolicy> {
        Some(self)
    }

    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        ScoringPolicy::as_filter(self)
    }
}

/// A weighted sum of scoring terms.
#[derive(Debug)]
pub struct FusedScorePolicy {
    /// Terms and optional `--fuse name=weight` overrides.
    terms: Vec<(Arc<dyn Policy>, Option<f32>)>,
}

fn view(t: &(Arc<dyn Policy>, Option<f32>)) -> (&dyn ScoringPolicy, f32) {
    let s = t.0.as_scoring().expect("checked by FusedScorePolicy::new");
    (s, t.1.unwrap_or_else(|| s.weight()))
}

impl FusedScorePolicy {
    /// Reject non-scoring terms during construction.
    pub fn new(terms: Vec<(Arc<dyn Policy>, Option<f32>)>) -> anyhow::Result<Self> {
        for (p, _) in &terms {
            anyhow::ensure!(p.can_fuse(), "policy {p:?} does not support fusion");
        }
        Ok(Self { terms })
    }
}

/// Applies eligibility filters before an inner policy.
#[derive(Debug)]
pub struct Pipeline {
    filters: Vec<Arc<dyn Policy>>,
    inner: Arc<dyn Policy>,
}

impl Pipeline {
    /// Reject policies that do not expose an eligibility view.
    pub fn new(filters: Vec<Arc<dyn Policy>>, inner: Arc<dyn Policy>) -> anyhow::Result<Self> {
        for f in &filters {
            anyhow::ensure!(f.can_filter(), "policy {f:?} imposes no eligibility rule");
        }
        Ok(Self { filters, inner })
    }

    fn views(&self) -> impl Iterator<Item = &dyn EligibilityFilter> {
        (self.filters.iter()).map(|p| p.as_filter().expect("checked by Pipeline::new"))
    }

    /// Apply eligibility without rewriting an existing Session assignment.
    fn evaluate_prefill_filtered(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillEvaluation> {
        let fleet_ctx = ctx
            .clone()
            .with_routable_fleet(ctx.routable_fleet().unwrap_or(workers));
        let ctx = &fleet_ctx;
        let eligible = apply_filters(self.views(), workers, ctx)?;
        if self.inner.resolves_affinity_in_range() && ctx.affinity_lookup_enabled() {
            let probe_ctx = (*ctx).clone().without_affinity_assignment();
            if let Some(
                proposal @ PrefillEvaluation::Pair(SelectionProposal {
                    kind: crate::policies::ProposalKind::SessionAffinity,
                    ..
                }),
            ) = self.inner.evaluate_prefill(workers, &probe_ctx)
            {
                return Some(proposal.with_eligible_workers(eligible));
            }
        }
        self.inner
            .evaluate_prefill(&eligible, ctx)
            .map(|proposal| proposal.with_eligible_workers(eligible))
    }
}

impl Policy for Pipeline {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let (proposal_kind, selected) = match self.evaluate_prefill_filtered(workers, ctx)? {
            PrefillEvaluation::Pair(proposal) => {
                let eligible = proposal.eligible_workers.as_deref().unwrap_or(workers);
                if eligible
                    .iter()
                    .any(|worker| worker.id == proposal.primary.id)
                {
                    (proposal.kind, proposal.primary)
                } else {
                    let selected = proposal
                        .backup
                        .filter(|backup| eligible.iter().any(|worker| worker.id == backup.id))
                        .or_else(|| eligible.first().cloned())?;
                    (proposal.kind, selected)
                }
            }
            PrefillEvaluation::Cache(proposal) => {
                let selected = proposal.candidates.into_iter().next()?.worker;
                (crate::policies::ProposalKind::CacheAffinity, selected)
            }
        };
        self.inner
            .commit_prefill_selection(ctx, proposal_kind, &selected);
        Some(selected)
    }

    /// Preserves the inner policy's complete proposal.
    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match self.evaluate_prefill_filtered(workers, ctx)? {
            PrefillEvaluation::Pair(proposal) => Some(proposal),
            PrefillEvaluation::Cache(proposal) => {
                let candidate = proposal.candidates.into_iter().next()?;
                Some(
                    SelectionProposal::primary(candidate.worker)
                        .with_kind(crate::policies::ProposalKind::CacheAffinity),
                )
            }
        }
    }

    fn evaluate_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillEvaluation> {
        self.evaluate_prefill_filtered(workers, ctx)
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        self.inner.uses_shared_prefill_admission()
    }

    fn needs_load_snapshot(&self) -> bool {
        self.inner.needs_load_snapshot() || self.filters.iter().any(|p| p.needs_load_snapshot())
    }

    fn needs_dispatch_timestamps(&self) -> bool {
        self.inner.needs_dispatch_timestamps()
            || self
                .filters
                .iter()
                .any(|policy| policy.needs_dispatch_timestamps())
    }

    fn commit_prefill_selection(
        &self,
        ctx: &SelectionContext<'_>,
        proposal_kind: crate::policies::ProposalKind,
        selected: &Arc<Worker>,
    ) {
        self.inner
            .commit_prefill_selection(ctx, proposal_kind, selected);
    }

    /// Preserves the inner policy's Bucket-affinity semantics.
    fn resolves_affinity_in_range(&self) -> bool {
        self.inner.resolves_affinity_in_range()
    }

    fn needs_request_tokens(&self) -> bool {
        self.inner.needs_request_tokens() || self.views().any(|f| f.needs_tokens())
    }

    fn attach_metrics(&self, metrics: Arc<crate::server::metrics::MetricsRegistry>) {
        self.inner.attach_metrics(metrics);
    }
}

/// Top-level score policy that enters shared Prefill admission.
#[derive(Debug)]
pub struct ScorePolicy {
    inner: Arc<dyn Policy>,
}

impl ScorePolicy {
    pub fn new(inner: Arc<dyn Policy>) -> Self {
        Self { inner }
    }
}

impl Policy for ScorePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        self.inner.select(workers, ctx)
    }

    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        self.inner
            .propose(workers, ctx)
            .map(|proposal| proposal.with_kind(crate::policies::ProposalKind::Score))
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }

    fn needs_request_tokens(&self) -> bool {
        self.inner.needs_request_tokens()
    }

    fn needs_dispatch_timestamps(&self) -> bool {
        self.inner.needs_dispatch_timestamps()
    }

    fn attach_metrics(&self, metrics: Arc<crate::server::metrics::MetricsRegistry>) {
        self.inner.attach_metrics(metrics);
    }

    fn as_scoring(&self) -> Option<&dyn ScoringPolicy> {
        self.inner.as_scoring()
    }

    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        self.inner.as_filter()
    }
}

impl ScoringPolicy for FusedScorePolicy {
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let mut total = vec![0.0f32; workers.len()];
        for (term, w) in self.terms.iter().map(view) {
            for (acc, s) in total.iter_mut().zip(term.scores(workers, ctx)) {
                *acc += w * s;
            }
        }
        total
    }

    fn needs_tokens(&self) -> bool {
        self.terms.iter().map(view).any(|(t, _)| t.needs_tokens())
    }

    fn needs_load_snapshot(&self) -> bool {
        self.terms
            .iter()
            .any(|(policy, _)| policy.needs_load_snapshot())
    }

    fn needs_dispatch_timestamps(&self) -> bool {
        self.terms
            .iter()
            .any(|(policy, _)| policy.needs_dispatch_timestamps())
    }
}

/// Owned boxes as the borrowed views [`admit`] consumes. Shared by the tests
/// in this module and its siblings.
#[cfg(test)]
pub(crate) fn refs(
    fs: &[Box<dyn EligibilityFilter>],
) -> impl Iterator<Item = &dyn EligibilityFilter> {
    fs.iter().map(|f| &**f)
}

/// Scores within this distance of the best are tied.
pub const TIE_EPSILON: f32 = 1e-6;

pub trait Selector: Send + Sync + std::fmt::Debug {
    /// Index into `workers` of the chosen candidate, or `None` when there is
    /// nothing to choose from. `scores[i]` belongs to `workers[i]`.
    fn pick(&self, workers: &[Arc<Worker>], scores: &[f32]) -> Option<usize>;
}

/// Highest score wins; ties choose the least-loaded candidate and rotate.
#[derive(Debug, Default)]
pub struct Argmax {
    rotor: AtomicUsize,
}

/// The default selector, shared by every scoring policy that does not override
/// [`ScoringPolicy::selector`].
pub static ARGMAX: Argmax = Argmax {
    rotor: AtomicUsize::new(0),
};

impl Selector for Argmax {
    fn pick(&self, workers: &[Arc<Worker>], scores: &[f32]) -> Option<usize> {
        if workers.is_empty() {
            return None;
        }
        let n = workers.len().min(scores.len());
        let best = (0..n)
            .map(|i| scores[i])
            .filter(|s| !s.is_nan())
            .fold(None::<f32>, |acc, s| Some(acc.map_or(s, |b| b.max(s))));
        let mut band: Vec<usize> = match best {
            Some(b) => (0..n)
                .filter(|&i| !scores[i].is_nan() && scores[i] >= b - TIE_EPSILON)
                .collect(),
            None => Vec::new(),
        };
        if band.is_empty() {
            tracing::debug!(
                n_workers = workers.len(),
                n_scores = scores.len(),
                "no usable score; falling back to load + rotation",
            );
            band = (0..workers.len()).collect();
        }
        let min_load = band.iter().map(|&i| workers[i].active_load()).min()?;
        let tied: Vec<usize> = band
            .into_iter()
            .filter(|&i| workers[i].active_load() == min_load)
            .collect();
        let k = self.rotor.fetch_add(1, Ordering::Relaxed) % tied.len();
        Some(tied[k])
    }
}

/// Score for a miss or unavailable prefix signal.
const NO_HOLDING: f32 = 0.0;

/// Default fused-term weight.
pub const DEFAULT_WEIGHT: f32 = 1.0;

pub struct PrefixCachePolicy {
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
    weight: f32,
    /// Minimum cached share for eligibility; zero disables filtering.
    min_share: f32,
}

impl std::fmt::Debug for PrefixCachePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixCachePolicy")
            .field("weight", &self.weight)
            .field("min_share", &self.min_share)
            .field("tree_nodes", &self.tree.node_count())
            .finish()
    }
}

impl PrefixCachePolicy {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>, weight: f32) -> Self {
        Self {
            tree,
            block_size_oracle,
            weight,
            min_share: 0.0,
        }
    }

    /// Require a cached share for eligibility.
    pub fn with_min_share(mut self, share: f32) -> Self {
        self.min_share = share;
        self
    }
}

impl PrefixCachePolicy {
    /// Returns each worker's cached prompt share.
    fn shares(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let flat = || vec![NO_HOLDING; workers.len()];

        let Some(tokens) = ctx.request_tokens().filter(|t| !t.is_empty()) else {
            return flat();
        };
        let Some((query_blocks, depths)) =
            crate::kv_events::prefix_depths_by_url(&self.tree, &self.block_size_oracle, tokens)
        else {
            return flat();
        };
        workers
            .iter()
            .map(|worker| {
                depths
                    .get(&worker.url)
                    .map_or(NO_HOLDING, |depth| *depth as f32 / query_blocks as f32)
            })
            .collect()
    }
}

impl ScoringPolicy for PrefixCachePolicy {
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        self.shares(workers, ctx)
    }

    fn weight(&self) -> f32 {
        self.weight
    }

    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        (self.min_share > 0.0).then_some(self as &dyn EligibilityFilter)
    }

    fn needs_tokens(&self) -> bool {
        true
    }
}

impl EligibilityFilter for PrefixCachePolicy {
    fn keep(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<bool> {
        (self.shares(workers, ctx).into_iter())
            .map(|share| share >= self.min_share)
            .collect()
    }

    fn needs_tokens(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod scoring_tests {
    use super::*;
    use crate::config::AffinityConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::admission::OnEmpty;
    use crate::policies::admission::{resolve_prefill, CandidateRange};
    use crate::policies::affinity::SessionAwarePolicy;
    use crate::policies::balancing::LoadBasedPolicy;
    use crate::policies::balancing::PowerOfTwoChoicesPolicy;
    use crate::policies::balancing::RoundRobinPolicy;
    use crate::workers::engine_reports::{EngineSnapshot, NativeCacheWorkerLoad};
    use std::collections::HashMap;
    use std::time::Instant;

    #[test]
    fn pipeline_preserves_fleet_scope_for_cache_saturation() {
        use crate::kv_events::PrefixSignal;
        use crate::policies::cache_aware::CacheAwarePolicy;
        let owner = worker("owner");
        let idle = worker("idle");
        let workers = vec![Arc::clone(&owner), Arc::clone(&idle)];
        let loads = snapshot(&[(&owner, 1, 4, 0, 10_000), (&idle, 0, 0, 0, 10_000)]);
        let model = ModelId("tiny".into());
        let signal = PrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![sgl_kv_indexer::PrefixMatch {
                    worker_id: owner.id.0.clone(),
                    address: owner.url.clone(),
                    matched_prefix_blocks: 1,
                }],
                best_prefix_blocks: 1,
            },
            query_blocks: 1,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(100)
            .with_external_prefix(Some(&signal))
            .with_load_snapshot(&loads);
        let cache = Arc::new(CacheAwarePolicy::new(AffinityConfig {
            worker_queue_limit: Some(4),
            saturation_queue_floor: Some(1),
            cache_affinity_min_matched_tokens: None,
            cache_affinity_min_match_ratio: None,
            ..Default::default()
        }));
        let pipeline =
            Pipeline::new(vec![Arc::new(Keep(vec!["owner"], OnEmpty::Hold))], cache).unwrap();
        let PrefillEvaluation::Cache(selection) =
            pipeline.evaluate_prefill(&workers, &ctx).unwrap()
        else {
            panic!("the owner must produce a cache evaluation");
        };
        assert_eq!(selection.candidates.len(), 1);
        assert_eq!(selection.resolution.queue_gate_rejected_candidates, 1);
        assert!(!selection.resolution.fleet_all_queued);
        assert!(selection.resolution.decision.is_none(), "the idle fleet worker prevents a saturation pin even when a filter excluded it from the cache candidate set");
    }

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    fn fleet() -> Vec<Arc<Worker>> {
        vec![worker("a"), worker("b"), worker("c")]
    }

    fn snapshot(entries: &[(&Arc<Worker>, u64, u64, u64, u64)]) -> EngineSnapshot {
        EngineSnapshot::from_native_cache_workers(
            1,
            entries
                .iter()
                .map(|(worker, running, waiting, used, capacity)| {
                    (
                        worker.url.clone(),
                        NativeCacheWorkerLoad {
                            num_running_reqs: *running,
                            num_waiting_reqs: *waiting,
                            num_waiting_uncached_tokens: *waiting,
                            num_used_tokens: *used,
                            num_total_tokens: *used,
                            max_total_num_tokens: *capacity,
                            max_running_requests: 64,
                            prefill_throughput_tokens_per_s: None,
                            estimated_prefill_queue_ms: None,
                            captured_at: Instant::now(),
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        )
    }

    fn urls(ws: &[Arc<Worker>]) -> Vec<String> {
        ws.iter().map(|w| w.url.clone()).collect()
    }

    fn term(p: impl Policy + 'static, w: Option<f32>) -> (Arc<dyn Policy>, Option<f32>) {
        (Arc::new(p), w)
    }

    #[derive(Debug)]
    struct ByIndex(f32, bool, bool);

    fn by(w: f32) -> ByIndex {
        ByIndex(w, false, false)
    }

    impl ScoringPolicy for ByIndex {
        fn scores(&self, workers: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<f32> {
            (0..workers.len()).map(|i| i as f32).collect()
        }
        fn weight(&self) -> f32 {
            self.0
        }
        fn needs_tokens(&self) -> bool {
            self.2
        }
        fn selector(&self) -> &dyn Selector {
            if self.1 {
                &PICK_FIRST
            } else {
                &ARGMAX
            }
        }
    }

    #[derive(Debug)]
    struct Keep(Vec<&'static str>, OnEmpty);

    impl EligibilityFilter for Keep {
        fn keep(&self, workers: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<bool> {
            (workers.iter())
                .map(|w| self.0.iter().any(|n| w.url.contains(n)))
                .collect()
        }
        fn on_empty(&self) -> OnEmpty {
            self.1
        }
    }

    impl Policy for Keep {
        fn select(&self, ws: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
            (ws.iter().zip(self.keep(ws, ctx)))
                .find(|(_, ok)| *ok)
                .map(|(w, _)| Arc::clone(w))
        }
        fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
            Some(self)
        }
    }

    fn keep(names: &[&'static str], on_empty: OnEmpty) -> Box<dyn EligibilityFilter> {
        Box::new(Keep(names.to_vec(), on_empty))
    }

    fn boxed(f: impl EligibilityFilter + 'static) -> Box<dyn EligibilityFilter> {
        Box::new(f)
    }

    #[derive(Debug)]
    struct PickFirst;
    static PICK_FIRST: PickFirst = PickFirst;
    impl Selector for PickFirst {
        fn pick(&self, workers: &[Arc<Worker>], _: &[f32]) -> Option<usize> {
            (!workers.is_empty()).then_some(0)
        }
    }

    #[test]
    fn selector_dispatch_uses_the_policys_own_selector() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let scores = by(1.0).scores(&ws, &ctx);
        assert_eq!(ARGMAX.pick(&ws, &scores), Some(2));
        assert_eq!(PICK_FIRST.pick(&ws, &scores), Some(0));

        assert_eq!(by(1.0).select(&ws, &ctx).unwrap().id, ws[2].id);
        let first = ByIndex(1.0, true, false);
        assert_eq!(first.select(&ws, &ctx).unwrap().id, ws[0].id);
    }

    #[test]
    fn can_fuse_is_derived_and_gates_construction() {
        let fused = FusedScorePolicy::new(vec![term(by(1.0), None)]).unwrap();
        let fusable: Vec<Arc<dyn Policy>> = vec![Arc::new(by(1.0)), Arc::new(fused)];
        for p in &fusable {
            assert!(p.can_fuse());
            assert!(p.as_scoring().is_some(), "the flag agrees with the view");
        }
        let rr: Arc<dyn Policy> = Arc::new(RoundRobinPolicy::new());
        assert!(!rr.can_fuse());
        assert!(rr.as_scoring().is_none());
        let err = FusedScorePolicy::new(vec![term(RoundRobinPolicy::new(), None)])
            .expect_err("round_robin has no per-worker preference to contribute");
        assert!(err.to_string().contains("does not support fusion"), "{err}");
    }

    #[test]
    fn fusion_nests_and_the_override_replaces_the_terms_own_weight() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let inner = FusedScorePolicy::new(vec![term(by(2.0), None)]).unwrap();
        assert_eq!(inner.scores(&ws, &ctx), vec![0.0, 2.0, 4.0], "its own 2i");

        let outer =
            FusedScorePolicy::new(vec![term(inner, None), term(by(3.0), Some(10.0))]).unwrap();
        assert_eq!(outer.scores(&ws, &ctx), vec![0.0, 12.0, 24.0], "2i + 10i");
        assert_eq!(outer.select(&ws, &ctx).unwrap().id, ws[2].id);
    }

    #[test]
    fn composer_propagates_needs_tokens_from_any_term() {
        let plain = FusedScorePolicy::new(vec![term(by(1.0), None)]).unwrap();
        assert!(!plain.needs_request_tokens());
        let hungry = FusedScorePolicy::new(vec![
            term(by(1.0), None),
            term(ByIndex(1.0, false, true), None),
        ])
        .unwrap();
        assert!(hungry.needs_request_tokens(), "any one term is enough");

        #[derive(Debug)]
        struct Hungry;
        impl EligibilityFilter for Hungry {
            fn keep(&self, ws: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<bool> {
                vec![true; ws.len()]
            }
            fn needs_tokens(&self) -> bool {
                true
            }
        }
        impl Policy for Hungry {
            fn select(&self, ws: &[Arc<Worker>], _: &SelectionContext<'_>) -> Option<Arc<Worker>> {
                ws.first().map(Arc::clone)
            }
            fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
                Some(self)
            }
        }
        let filtered = Pipeline::new(
            vec![Arc::new(Hungry)],
            Arc::new(FusedScorePolicy::new(vec![term(by(1.0), None)]).unwrap()),
        )
        .unwrap();
        assert!(filtered.needs_request_tokens(), "the filter is hungry");
    }

    #[test]
    fn composer_propagates_load_snapshot_capability() {
        #[derive(Debug)]
        struct LoadHungry;
        impl ScoringPolicy for LoadHungry {
            fn scores(&self, workers: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<f32> {
                vec![0.0; workers.len()]
            }
            fn needs_load_snapshot(&self) -> bool {
                true
            }
        }

        let plain = FusedScorePolicy::new(vec![term(by(1.0), None)]).unwrap();
        assert!(!Policy::needs_load_snapshot(&plain));
        assert!(!Policy::needs_dispatch_timestamps(&plain));
        let fused =
            FusedScorePolicy::new(vec![term(by(1.0), None), term(LoadHungry, None)]).unwrap();
        assert!(Policy::needs_load_snapshot(&fused));
        assert!(!Policy::needs_dispatch_timestamps(&fused));

        let load_fused = FusedScorePolicy::new(vec![
            term(by(1.0), None),
            term(LoadBasedPolicy::new(), None),
        ])
        .unwrap();
        assert!(Policy::needs_dispatch_timestamps(&load_fused));

        let pipeline = Pipeline::new(
            vec![Arc::new(Keep(vec!["a"], OnEmpty::Abstain))],
            Arc::new(load_fused),
        )
        .unwrap();
        assert!(pipeline.needs_load_snapshot());
        assert!(pipeline.needs_dispatch_timestamps());

        let score = ScorePolicy::new(Arc::new(by(1.0)));
        assert!(
            score.needs_load_snapshot(),
            "shared admission requires a snapshot"
        );
        assert!(!score.needs_dispatch_timestamps());

        let load_score = ScorePolicy::new(Arc::new(LoadBasedPolicy::new()));
        assert!(load_score.needs_dispatch_timestamps());
    }

    #[test]
    fn a_rejected_worker_cannot_be_out_weighed() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let fused = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b"], OnEmpty::Abstain))],
            Arc::new(FusedScorePolicy::new(vec![term(by(1.0), Some(1e9))]).unwrap()),
        )
        .unwrap();
        assert_eq!(
            fused.select(&ws, &ctx).unwrap().url,
            ws[1].url,
            "the best ELIGIBLE, not the best"
        );

        let open = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b", "c"], OnEmpty::Abstain))],
            Arc::new(FusedScorePolicy::new(vec![term(by(1.0), Some(1e9))]).unwrap()),
        )
        .unwrap();
        assert_eq!(open.select(&ws, &ctx).unwrap().url, ws[2].url);
    }

    #[test]
    fn pipeline_preserves_the_inner_step_one_proposal_and_admission_opt_in() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let pipeline = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b", "c"], OnEmpty::Abstain))],
            Arc::new(PowerOfTwoChoicesPolicy::new()),
        )
        .expect("valid filter and inner policy");

        let proposal = pipeline
            .propose(&ws, &ctx)
            .expect("eligible P2 must retain a pair");

        assert!(
            proposal.backup.is_some(),
            "Pipeline must not collapse P2 to one primary"
        );
        assert!(pipeline.uses_shared_prefill_admission());

        let session_pipeline = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b", "c"], OnEmpty::Abstain))],
            Arc::new(SessionAwarePolicy::new(AffinityConfig::default())),
        )
        .expect("valid filter and inner session policy");
        assert!(
            session_pipeline.resolves_affinity_in_range(),
            "Pipeline must forward the inner Session affinity range capability"
        );
    }

    #[test]
    fn shared_admission_fallback_cannot_reintroduce_a_filtered_worker() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let pipeline = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b"], OnEmpty::Abstain))],
            Arc::new(PowerOfTwoChoicesPolicy::new()),
        )
        .expect("valid filter and inner policy");
        let proposal = pipeline
            .propose(&ws, &ctx)
            .expect("the two eligible workers produce a P2 proposal");
        let snapshot = snapshot(&[
            (&ws[0], 0, 0, 4_090, 4_096),
            (&ws[1], 0, 0, 4_090, 4_096),
            (&ws[2], 0, 0, 0, 4_096),
        ]);

        let decision = resolve_prefill(
            &CandidateRange::global(&ws),
            &proposal,
            32,
            &snapshot,
            None,
            crate::policies::admission::CapacityFallback::Allowed,
        )
        .expect("capacity exhaustion must degrade inside the filtered domain");
        assert!(matches!(decision.selected.id.0.as_str(), "a" | "b"));
    }

    #[test]
    fn eligibility_escape_does_not_rewrite_an_existing_session_assignment() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_session_id(Some("session-a"));
        let session = Arc::new(SessionAwarePolicy::new(AffinityConfig::default()));

        let initial = session
            .propose(&ws[2..], &ctx)
            .expect("one-worker domain establishes c");
        assert_eq!(initial.primary.id, ws[2].id);
        session.commit_prefill_selection(&ctx, initial.kind, &initial.primary);

        let pipeline = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b"], OnEmpty::Abstain))],
            session.clone(),
        )
        .expect("valid filter and session policy");
        let PrefillEvaluation::Pair(proposal) = pipeline
            .evaluate_prefill(&ws, &ctx)
            .expect("filtered session proposal")
        else {
            panic!("Session-Aware must retain pair semantics");
        };
        assert_eq!(
            proposal.kind,
            crate::policies::ProposalKind::SessionAffinity
        );
        assert_eq!(proposal.primary.id, ws[2].id);

        let snapshot = EngineSnapshot::default();
        let decision = resolve_prefill(
            &CandidateRange::global(&ws),
            &proposal,
            32,
            &snapshot,
            None,
            crate::policies::admission::CapacityFallback::Allowed,
        )
        .expect("an eligible escape worker exists");
        assert_ne!(decision.selected.id, ws[2].id);
        assert!(matches!(decision.selected.id.0.as_str(), "a" | "b"));

        let after = session
            .propose(&ws, &ctx)
            .expect("the original assignment remains readable");
        assert_eq!(after.kind, crate::policies::ProposalKind::SessionAffinity);
        assert_eq!(after.primary.id, ws[2].id);
    }

    #[test]
    fn new_session_assignment_is_created_inside_the_eligible_set() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_session_id(Some("session-new"));
        let session = Arc::new(SessionAwarePolicy::new(AffinityConfig::default()));
        let pipeline = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b"], OnEmpty::Abstain))],
            session.clone(),
        )
        .expect("valid filter and session policy");

        let PrefillEvaluation::Pair(proposal) = pipeline
            .evaluate_prefill(&ws, &ctx)
            .expect("eligible workers establish the session")
        else {
            panic!("Session-Aware must retain pair semantics");
        };
        assert!(matches!(proposal.primary.id.0.as_str(), "a" | "b"));
        pipeline.commit_prefill_selection(&ctx, proposal.kind, &proposal.primary);

        let mapped = session
            .propose(&ws, &ctx)
            .expect("the assignment is stored by the inner policy");
        assert_eq!(mapped.kind, crate::policies::ProposalKind::SessionAffinity);
        assert_eq!(mapped.primary.id, proposal.primary.id);
    }

    /// Order is priority: the LOWER-priority filter yields, and what the
    /// higher-priority one narrowed to is kept. Asserted on the surviving set
    /// rather than on the winner, because with three workers a wrong rule can
    /// still land on the right one by luck.
    #[test]
    fn a_conflict_yields_the_later_filter_and_keeps_the_earlier_narrowing() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let chain = vec![
            keep(&["a", "b"], OnEmpty::Abstain),
            keep(&["c"], OnEmpty::Abstain),
        ];
        let out = apply_filters(refs(&chain), &ws, &ctx).expect("Abstain never holds");
        assert_eq!(
            urls(&out),
            urls(&ws[..2]),
            "the second filter yields; falling back to the raw fleet would read [a, b, c]",
        );

        let rev = vec![
            keep(&["c"], OnEmpty::Abstain),
            keep(&["a", "b"], OnEmpty::Abstain),
        ];
        assert_eq!(
            urls(&apply_filters(refs(&rev), &ws, &ctx).unwrap()),
            urls(&ws[2..])
        );
    }

    #[test]
    fn a_filter_after_a_conflict_still_applies() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let chain = vec![
            keep(&["a", "b"], OnEmpty::Abstain),
            keep(&["c"], OnEmpty::Abstain),
            keep(&["b", "c"], OnEmpty::Abstain),
        ];
        assert_eq!(
            urls(&apply_filters(refs(&chain), &ws, &ctx).unwrap()),
            vec![ws[1].url.clone()]
        );
    }

    #[test]
    fn a_holding_filter_refuses_instead_of_yielding() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let held = vec![
            keep(&["a", "b"], OnEmpty::Abstain),
            keep(&["c"], OnEmpty::Hold),
        ];
        assert!(
            apply_filters(refs(&held), &ws, &ctx).is_none(),
            "no eligible worker, and the filter said Hold",
        );

        let ok = vec![
            keep(&["a", "b"], OnEmpty::Abstain),
            keep(&["b"], OnEmpty::Hold),
        ];
        assert_eq!(
            urls(&apply_filters(refs(&ok), &ws, &ctx).unwrap()),
            vec![ws[1].url.clone()]
        );

        let fused = Pipeline::new(
            vec![
                Arc::new(Keep(vec!["a", "b"], OnEmpty::Abstain)),
                Arc::new(Keep(vec!["c"], OnEmpty::Hold)),
            ],
            Arc::new(FusedScorePolicy::new(vec![term(by(1.0), None)]).unwrap()),
        )
        .unwrap();
        assert!(fused.select(&ws, &ctx).is_none());
    }

    #[test]
    fn a_short_flag_vector_degrades_instead_of_panicking() {
        #[derive(Debug)]
        struct Short;
        impl EligibilityFilter for Short {
            fn keep(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<bool> {
                vec![false]
            }
        }
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let out = apply_filters(refs(&[boxed(Short)]), &ws, &ctx).expect("the tail was admitted");
        assert_eq!(urls(&out), urls(&ws[1..]), "only index 0 rejected");
    }

    #[test]
    fn a_short_hold_filter_fails_closed() {
        #[derive(Debug)]
        struct ShortHold;
        impl EligibilityFilter for ShortHold {
            fn keep(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<bool> {
                vec![true]
            }

            fn on_empty(&self) -> OnEmpty {
                OnEmpty::Hold
            }
        }

        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert!(apply_filters(refs(&[boxed(ShortHold)]), &ws, &ctx).is_none());
    }

    #[test]
    fn a_dual_role_term_exposes_its_filter_half_through_policy() {
        #[derive(Debug)]
        struct Dual;
        impl EligibilityFilter for Dual {
            fn keep(&self, ws: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<bool> {
                ws.iter().map(|w| !w.url.contains('c')).collect()
            }
            fn needs_tokens(&self) -> bool {
                true
            }
        }
        impl ScoringPolicy for Dual {
            fn scores(&self, ws: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<f32> {
                vec![0.0; ws.len()]
            }
            fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
                Some(self)
            }
        }

        let p: Arc<dyn Policy> = Arc::new(Dual);
        assert!(p.can_fuse(), "it still scores");
        let f = p.as_filter().expect("and it filters");
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert_eq!(f.keep(&ws, &ctx), vec![true, true, false]);
        assert!(
            p.needs_request_tokens(),
            "hunger comes from the filter half"
        );
    }
}

#[cfg(test)]
mod argmax_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use std::collections::HashSet;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    #[test]
    fn score_wins_unless_the_gap_is_inside_the_tie_band() {
        let ws = vec![worker("a"), worker("b")];
        let sel = Argmax::default();
        let _loaded = ws[1].load_guard();

        assert_eq!(sel.pick(&ws, &[1.0, 1.0 - 1e-3]), Some(0), "clear winner");
        let tie = [1.0 - 5e-7, 1.0];
        assert_eq!(sel.pick(&ws, &tie), Some(0), "tie -> less load");
        assert_eq!(sel.pick(&[], &[]), None, "nothing to choose from");
    }

    #[test]
    fn nan_never_wins_from_either_position() {
        let ws = vec![worker("a"), worker("b")];
        let sel = Argmax::default();
        assert_eq!(sel.pick(&ws, &[f32::NAN, 0.0]), Some(1));
        assert_eq!(sel.pick(&ws, &[0.0, f32::NAN]), Some(0));
        assert!(sel.pick(&ws, &[f32::NAN, f32::NAN]).is_some());
    }

    #[test]
    fn a_total_tie_rotates_over_every_candidate() {
        let ws = vec![worker("a"), worker("b"), worker("c")];
        let sel = Argmax::default();
        let picks: HashSet<usize> = (0..3).filter_map(|_| sel.pick(&ws, &[1.0; 3])).collect();
        assert_eq!(
            picks.len(),
            3,
            "three tied picks must cover all three: {picks:?}"
        );
    }
}

#[cfg(test)]
mod prefix_cache_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::kv_events::KvWorkerId;
    use crate::kv_events::{compute_block_hashes, compute_block_hashes_bigram};

    const BLOCK: usize = 4;

    fn worker(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    fn tokens() -> Vec<u32> {
        (0..(BLOCK as u32 * 4)).collect()
    }

    fn insert(tree: &HashTree, url: &str, rank: u32, from: usize, blocks: usize) {
        let all = compute_block_hashes(&tokens(), BLOCK);
        let parent = if from == 0 { None } else { Some(all[from - 1]) };
        tree.insert(
            &KvWorkerId::new(url.into(), rank),
            parent,
            &all[from..from + blocks],
        );
    }

    fn policy(tree: Arc<HashTree>) -> PrefixCachePolicy {
        let oracle = BlockSizeOracle::new();
        oracle
            .try_set(BLOCK as u32)
            .expect("a fresh oracle accepts the first block size");
        PrefixCachePolicy::new(tree, oracle, 1.0)
    }

    fn shares(p: &PrefixCachePolicy, ws: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        assert!(
            ScoringPolicy::as_filter(p).is_none(),
            "no floor configured, so this term must not be a filter at all",
        );
        p.scores(ws, ctx)
    }

    #[test]
    fn depth_is_a_fraction_and_a_tail_without_block_zero_misses() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        insert(&tree, "tail", 0, 2, 2);
        let ws = vec![worker("deep"), worker("tail"), worker("cold")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let scores = shares(&policy(tree), &ws, &ctx);
        assert_eq!(scores[0], 0.75, "3 of 4 blocks held, not a neutral 1.0");
        assert_eq!(scores[1], 0.0, "tail without block 0 holds nothing");
        assert_eq!(scores[2], 0.0, "never seen");
    }

    #[test]
    fn several_dp_ranks_of_one_worker_collapse_to_the_deepest() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "dp", 0, 0, 1);
        insert(&tree, "dp", 1, 0, 3);
        let ws = vec![worker("dp")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        assert_eq!(
            shares(&policy(tree), &ws, &ctx),
            vec![0.75],
            "3 of 4, not 1"
        );
    }

    #[test]
    fn without_tokens_every_worker_scores_the_same() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        let ws = vec![worker("deep"), worker("cold")];
        let model = ModelId("tiny".into());
        let policy = policy(tree);

        let ids = tokens();
        let with = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        assert_eq!(
            shares(&policy, &ws, &with),
            vec![0.75, 0.0],
            "signal is live"
        );

        let without = SelectionContext::new(&model, None);
        assert_eq!(
            shares(&policy, &ws, &without),
            vec![0.0, 0.0],
            "and inert here"
        );
    }

    #[test]
    fn without_a_block_size_no_worker_looks_like_a_hit() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        let ws = vec![worker("deep"), worker("cold")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let cold = PrefixCachePolicy::new(tree, BlockSizeOracle::new(), 1.0);
        assert_eq!(shares(&cold, &ws, &ctx), vec![0.0, 0.0]);
    }

    #[test]
    fn the_bigram_branch_queries_a_different_chain() {
        let ids = tokens();
        let unigram = compute_block_hashes(&ids, BLOCK);
        let bigram = compute_block_hashes_bigram(&ids, BLOCK);
        assert_ne!(unigram, bigram, "the two hashers must disagree, else this");

        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 4);
        let ws = vec![worker("deep")];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let oracle = BlockSizeOracle::new();
        oracle.try_set(BLOCK as u32).unwrap();
        oracle.set_bigram(true);
        let p = PrefixCachePolicy::new(Arc::clone(&tree), oracle, 1.0);
        assert_eq!(
            shares(&p, &ws, &ctx),
            vec![0.0],
            "bigram query, unigram tree"
        );
        assert_eq!(shares(&policy(tree), &ws, &ctx), vec![1.0]);
    }
}
