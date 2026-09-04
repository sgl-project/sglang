// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Candidate eligibility and scoring policies.

pub mod admission;
pub mod argmax;
pub mod prefix_cache;

use crate::policies::{Policy, PrefillProposal, SelectionContext, SelectionProposal};
use crate::workers::Worker;
use argmax::{Selector, ARGMAX};
use std::sync::Arc;

/// What a filter means when it has rejected every worker it was shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnEmpty {
    /// Ignore this filter when it rejects every candidate.
    Abstain,
    /// Keep the rejection: no worker is admissible.
    Hold,
}

/// A hard constraint applied before scoring.
pub trait EligibilityFilter: Send + Sync + std::fmt::Debug {
    /// Returns one admission flag per worker; `true` keeps the candidate.
    fn keep(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<bool>;

    /// Whether this constraint reads `ctx.request_tokens()`.
    fn needs_tokens(&self) -> bool {
        false
    }

    /// Controls the result when this filter rejects every candidate.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Abstain
    }
}

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

    /// Optional eligibility view for policies that provide both signals.
    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        None
    }

    /// Selects a winner from the score vector.
    fn selector(&self) -> &dyn Selector {
        &ARGMAX
    }
}

/// Applies ordered filters. `None` means a holding filter rejected all candidates.
pub fn admit<'f>(
    filters: impl IntoIterator<Item = &'f dyn EligibilityFilter>,
    workers: &[Arc<Worker>],
    ctx: &SelectionContext<'_>,
) -> Option<Vec<Arc<Worker>>> {
    let mut alive: Vec<Arc<Worker>> = workers.to_vec();

    for filter in filters {
        if alive.is_empty() {
            break;
        }
        let flags = filter.keep(&alive, ctx);
        let on_empty = filter.on_empty();
        if flags.len() != alive.len() {
            tracing::debug!(
                filter = ?filter,
                n_workers = alive.len(),
                n_flags = flags.len(),
                "eligibility filter returned the wrong arity",
            );
            if on_empty == OnEmpty::Hold {
                return None;
            }
        }
        let untouched = alive.len() == workers.len();

        let next: Vec<Arc<Worker>> = (alive.iter().enumerate())
            .filter(|(i, _)| flags.get(*i).copied().unwrap_or(true))
            .map(|(_, w)| Arc::clone(w))
            .collect();

        if next.is_empty() {
            match on_empty {
                OnEmpty::Hold => return None,
                OnEmpty::Abstain if untouched => {
                    tracing::debug!(
                        filter = ?filter,
                        n_workers = workers.len(),
                        "eligibility filter has no eligible workers; falling back to the full candidate set",
                    );
                    continue;
                }
                OnEmpty::Abstain => {
                    tracing::debug!(
                        filter = ?filter,
                        n_alive = alive.len(),
                        "eligibility filter conflicts with a higher-priority one; yielding",
                    );
                    continue;
                }
            }
        }
        alive = next;
    }

    Some(alive)
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
    fn propose_prefill_filtered(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        let eligible = admit(self.views(), workers, ctx)?;
        if self.inner.is_bucket_affinity_policy() && ctx.affinity_lookup_enabled() {
            let probe_ctx = (*ctx).clone().without_affinity_assignment();
            if let Some(
                proposal @ PrefillProposal::Pair(SelectionProposal {
                    kind: crate::policies::ProposalKind::SessionAffinity,
                    ..
                }),
            ) = self.inner.propose_prefill(workers, &probe_ctx)
            {
                return Some(proposal.with_eligible_workers(eligible));
            }
        }
        self.inner
            .propose_prefill(&eligible, ctx)
            .map(|proposal| proposal.with_eligible_workers(eligible))
    }
}

impl Policy for Pipeline {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let (proposal_kind, selected) = match self.propose_prefill_filtered(workers, ctx)? {
            PrefillProposal::Pair(proposal) => {
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
            PrefillProposal::CacheCandidates(proposal) => {
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
        match self.propose_prefill_filtered(workers, ctx)? {
            PrefillProposal::Pair(proposal) => Some(proposal),
            PrefillProposal::CacheCandidates(proposal) => {
                let candidate = proposal.candidates.into_iter().next()?;
                Some(
                    SelectionProposal::primary(candidate.worker)
                        .with_kind(crate::policies::ProposalKind::CacheAffinity),
                )
            }
        }
    }

    fn propose_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        self.propose_prefill_filtered(workers, ctx)
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        self.inner.uses_shared_prefill_admission()
    }

    fn needs_load_snapshot(&self) -> bool {
        self.inner.needs_load_snapshot() || self.filters.iter().any(|p| p.needs_load_snapshot())
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
    fn is_bucket_affinity_policy(&self) -> bool {
        self.inner.is_bucket_affinity_policy()
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
}

/// Owned boxes as the borrowed views [`admit`] consumes. Shared by the tests
/// in this module and its siblings.
#[cfg(test)]
pub(crate) fn refs(
    fs: &[Box<dyn EligibilityFilter>],
) -> impl Iterator<Item = &dyn EligibilityFilter> {
    fs.iter().map(|f| &**f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AffinityConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::admission::{resolve_prefill, CandidateRange};
    use crate::policies::engine_load::{EngineLoadSnapshot, EngineWorkerLoad};
    use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
    use crate::policies::round_robin::RoundRobinPolicy;
    use crate::policies::session_aware::SessionAwarePolicy;
    use std::collections::HashMap;
    use std::time::Instant;

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

    fn snapshot(entries: &[(&Arc<Worker>, u64, u64, u64, u64)]) -> EngineLoadSnapshot {
        EngineLoadSnapshot::from_workers(
            1,
            entries
                .iter()
                .map(|(worker, running, waiting, used, capacity)| {
                    (
                        worker.url.clone(),
                        EngineWorkerLoad {
                            num_running_reqs: *running,
                            num_waiting_reqs: *waiting,
                            num_tokens: *used,
                            max_total_num_tokens: *capacity,
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
        let fused =
            FusedScorePolicy::new(vec![term(by(1.0), None), term(LoadHungry, None)]).unwrap();
        assert!(Policy::needs_load_snapshot(&fused));

        let pipeline = Pipeline::new(
            vec![Arc::new(Keep(vec!["a"], OnEmpty::Abstain))],
            Arc::new(fused),
        )
        .unwrap();
        assert!(pipeline.needs_load_snapshot());

        let score = ScorePolicy::new(Arc::new(by(1.0)));
        assert!(
            score.needs_load_snapshot(),
            "shared admission requires a snapshot"
        );
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
            session_pipeline.is_bucket_affinity_policy(),
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

        let decision = resolve_prefill(&CandidateRange::global(&ws), &proposal, 32, &snapshot)
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
        let PrefillProposal::Pair(proposal) = pipeline
            .propose_prefill(&ws, &ctx)
            .expect("filtered session proposal")
        else {
            panic!("Session-Aware must retain pair semantics");
        };
        assert_eq!(
            proposal.kind,
            crate::policies::ProposalKind::SessionAffinity
        );
        assert_eq!(proposal.primary.id, ws[2].id);

        let snapshot = EngineLoadSnapshot::default();
        let decision = resolve_prefill(&CandidateRange::global(&ws), &proposal, 32, &snapshot)
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

        let PrefillProposal::Pair(proposal) = pipeline
            .propose_prefill(&ws, &ctx)
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
        let out = admit(refs(&chain), &ws, &ctx).expect("Abstain never holds");
        assert_eq!(
            urls(&out),
            urls(&ws[..2]),
            "the second filter yields; falling back to the raw fleet would read [a, b, c]",
        );

        let rev = vec![
            keep(&["c"], OnEmpty::Abstain),
            keep(&["a", "b"], OnEmpty::Abstain),
        ];
        assert_eq!(urls(&admit(refs(&rev), &ws, &ctx).unwrap()), urls(&ws[2..]));
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
            urls(&admit(refs(&chain), &ws, &ctx).unwrap()),
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
            admit(refs(&held), &ws, &ctx).is_none(),
            "no eligible worker, and the filter said Hold",
        );

        let ok = vec![
            keep(&["a", "b"], OnEmpty::Abstain),
            keep(&["b"], OnEmpty::Hold),
        ];
        assert_eq!(
            urls(&admit(refs(&ok), &ws, &ctx).unwrap()),
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
        let out = admit(refs(&[boxed(Short)]), &ws, &ctx).expect("the tail was admitted");
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
        assert!(admit(refs(&[boxed(ShortHold)]), &ws, &ctx).is_none());
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
