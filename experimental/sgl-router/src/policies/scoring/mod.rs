// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Routing in two layers: who is ELIGIBLE, then who among them is PREFERRED.
//!
//! [`EligibilityFilter`] answers a yes/no per worker and runs first;
//! [`ScoringPolicy`] answers a number per worker and runs only over the
//! survivors. Two traits rather than one with two methods, so a term declares
//! which layers it takes part in by what it implements — a type is never handed
//! a defaulted "no opinion" it forgot to override, and the filter pass costs
//! nothing for terms that only score.
//!
//! A hard constraint is deliberately NOT expressible as a very negative score:
//! a number can be out-weighed, so admission or bucket rules written that way
//! are rules a sufficiently confident other term can overrule.
//!
//! One type may implement BOTH — see [`ScoringPolicy::as_filter`]. Reserve that
//! for a signal whose hard and soft halves come from the SAME expensive
//! computation (a prefix-depth walk); a signal that splits into two cheap
//! lookups should be two terms.

pub mod admission;
pub mod argmax;
pub mod prefix_cache;

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use argmax::{Selector, ARGMAX};
use std::sync::Arc;

/// What a filter means when it has rejected every worker it was shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnEmpty {
    /// Treat the filter as having said nothing, and route without it. For
    /// affinity: "nobody holds this prompt" must not become a 503.
    Abstain,
    /// The rejection stands and no worker is admissible. For admission: "every
    /// worker is over its cap" must not silently pick one anyway. Also a
    /// declaration that this filter never yields to a lower-priority one.
    Hold,
}

/// A hard constraint: which candidates are even allowed to be considered.
///
/// Runs before any scoring, and a rejected worker cannot be scored back in.
pub trait EligibilityFilter: Send + Sync + std::fmt::Debug {
    /// Admissible flags parallel to `workers`; `true` keeps the candidate.
    ///
    /// The whole surviving set at once, so a filter can compare candidates
    /// against each other, take a top-k, or hash the prompt once per decision
    /// rather than once per worker.
    ///
    /// With no signal this request, ABSTAIN — all `true` — rather than
    /// rejecting everyone. A cache miss is "no opinion", not "nobody is fit to
    /// serve"; see [`FusedScorePolicy`] for why the difference is load-bearing.
    fn keep(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<bool>;

    /// Whether this constraint reads `ctx.request_tokens()`.
    fn needs_tokens(&self) -> bool {
        false
    }

    /// See [`OnEmpty`]. Defaulted to the safe direction: a filter that rejects
    /// everyone is ignored rather than dropping the request.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Abstain
    }
}

/// A soft preference: how much each ELIGIBLE candidate is worth.
pub trait ScoringPolicy: Send + Sync + std::fmt::Debug {
    /// What each candidate is worth, parallel to `workers`; higher is better.
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32>;

    /// Default multiplier as a fused term; `--fuse name=weight` overrides it.
    fn weight(&self) -> f32 {
        1.0
    }

    /// Whether these scores need `ctx.request_tokens()`; surfaces as
    /// [`Policy::needs_request_tokens`], which is what makes ingress tokenize.
    fn needs_tokens(&self) -> bool {
        false
    }

    /// The hard half of this same signal, when it has one. Overriding it is how
    /// one type serves both layers off a single computation; the blanket impl
    /// below is what makes that reachable as [`Policy::as_filter`].
    ///
    /// Rust cannot derive this from `Self: EligibilityFilter` — a second
    /// blanket `impl Policy` would collide with the one below — so it is one
    /// explicit line in the implementer rather than magic.
    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        None
    }

    /// How the scores collapse to one winner. The default is why a bare
    /// scoring policy needs nothing but `scores()`.
    fn selector(&self) -> &dyn Selector {
        &ARGMAX
    }
}

/// Apply `filters` in order and return the candidates still standing.
///
/// `None` means nothing is admissible AND the filter that said so declared
/// [`OnEmpty::Hold`] — the caller must not route.
///
/// Each filter sees only the survivors of the ones before it, so it is never
/// asked about a worker a higher-priority constraint already excluded. That
/// also makes the walk lazy in the useful direction: a filter that would have
/// been overruled never runs a second pass to find out.
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
        if flags.len() != alive.len() {
            // Arity is the filter's contract, but breaking it must degrade
            // rather than mis-index: the tail is admitted and the decision
            // still routes. Said out loud, because a filter that always lands
            // here has silently stopped constraining anything.
            tracing::debug!(
                filter = ?filter,
                n_workers = alive.len(),
                n_flags = flags.len(),
                "eligibility filter returned the wrong arity; missing flags admit",
            );
        }
        // Nothing has narrowed the set yet, so a total rejection here is this
        // filter's own doing rather than a clash with a higher-priority one.
        // Worth telling apart: the first is a filter that should have
        // abstained, the second is ordinary contention between constraints.
        let untouched = alive.len() == workers.len();

        let next: Vec<Arc<Worker>> = (alive.iter().enumerate())
            .filter(|(i, _)| flags.get(*i).copied().unwrap_or(true))
            .map(|(_, w)| Arc::clone(w))
            .collect();

        if next.is_empty() {
            match filter.on_empty() {
                OnEmpty::Hold => return None,
                OnEmpty::Abstain if untouched => {
                    tracing::warn!(
                        filter = ?filter,
                        n_workers = workers.len(),
                        "eligibility filter rejected every worker on its own; a filter \
                         with no signal should abstain (all true), not veto the fleet",
                    );
                    continue;
                }
                OnEmpty::Abstain => {
                    // Conflict, not a bug: this filter and a higher-priority
                    // one cannot both be satisfied. Order is priority, so THIS
                    // filter is the one that yields -- it is skipped whole, the
                    // earlier narrowing stands, and the rest of the list still
                    // runs.
                    //
                    // Skipping only this filter, rather than it and everything
                    // after it, is what keeps a later constraint alive: drop
                    // the tail instead and `--filter prefix_cache,overloaded`
                    // would route to an over-capacity worker on a cache miss.
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

/// A scoring policy is usable on its own: argmax its own preference over the
/// whole (already health-filtered) fleet.
impl<T: ScoringPolicy> Policy for T {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let scores = self.scores(workers, ctx);
        let i = self.selector().pick(workers, &scores)?;
        workers.get(i).map(Arc::clone)
    }

    fn needs_request_tokens(&self) -> bool {
        // The filter half, when there is one, may be hungrier than the scores.
        ScoringPolicy::needs_tokens(self) || self.as_filter().is_some_and(|f| f.needs_tokens())
    }

    fn as_scoring(&self) -> Option<&dyn ScoringPolicy> {
        Some(self)
    }

    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        ScoringPolicy::as_filter(self)
    }
}

/// `Filter -> Score -> Select`: the eligible set, then the weighted sum over
/// it, then one winner.
///
/// **Filter order is priority.** When two filters cannot both be satisfied —
/// the later one would reject everyone the earlier admitted — the LATER one
/// yields and the rest of the list still runs. So
/// `--filter overloaded,prefix_cache` keeps the capacity rule and gives up only
/// the cache constraint, rather than throwing the tail away, or throwing all of
/// it away and routing over the raw fleet. [`OnEmpty::Hold`] opts out of
/// yielding, which is what makes an admission rule safe to put anywhere.
///
/// Scoring order does NOT matter: it is a sum. `--fuse` is a set of terms,
/// `--filter` is a priority list.
///
/// Itself a [`ScoringPolicy`], so fusion nests. A term is an `Arc<dyn Policy>`
/// and NOT a `Box<dyn ScoringPolicy>`, because `as_scoring()` hands out a
/// BORROW that a factory holding an `Arc<dyn Policy>` cannot turn into an owned
/// box.
#[derive(Debug)]
pub struct FusedScorePolicy {
    /// Each term with its `--fuse name=weight` override; `None` keeps the
    /// term's own [`ScoringPolicy::weight`].
    terms: Vec<(Arc<dyn Policy>, Option<f32>)>,
}

/// A term's scoring view and the weight to apply. The `expect` cannot fire:
/// `new` rejected every term whose view was `None`, and `terms` never grows.
fn view(t: &(Arc<dyn Policy>, Option<f32>)) -> (&dyn ScoringPolicy, f32) {
    let s = t.0.as_scoring().expect("checked by FusedScorePolicy::new");
    (s, t.1.unwrap_or_else(|| s.weight()))
}

impl FusedScorePolicy {
    /// Rejects a non-scoring term at CONFIG time, so `--fuse round_robin` fails
    /// at startup instead of contributing nothing per request. The only place
    /// [`Policy::can_fuse`] is consulted; without this call it is decoration.
    pub fn new(terms: Vec<(Arc<dyn Policy>, Option<f32>)>) -> anyhow::Result<Self> {
        for (p, _) in &terms {
            anyhow::ensure!(p.can_fuse(), "policy {p:?} does not support fusion");
        }
        Ok(Self { terms })
    }
}

/// The eligibility layer in front of any [`Policy`]: admit, then let `inner`
/// choose among the survivors.
///
/// Separate from [`FusedScorePolicy`] on purpose. Constraints are not part of
/// "scoring", and keeping them out here means the layer composes with policies
/// that do not score at all -- a hand-written `impl Policy` gets capacity and
/// bucket rules without being rewritten as terms.
#[derive(Debug)]
pub struct Pipeline {
    /// Owned as `Arc<dyn Policy>` rather than `Box<dyn EligibilityFilter>` so
    /// that a dual-role term listed in BOTH `--filter` and `--fuse` is one
    /// instance with one configuration, not two that can drift apart.
    /// Construction has already checked that each yields a filter view.
    filters: Vec<Arc<dyn Policy>>,
    inner: Arc<dyn Policy>,
}

impl Pipeline {
    /// Rejects a policy with no eligibility view at CONFIG time, so
    /// `--filter round_robin` fails at startup rather than silently
    /// constraining nothing.
    pub fn new(filters: Vec<Arc<dyn Policy>>, inner: Arc<dyn Policy>) -> anyhow::Result<Self> {
        for f in &filters {
            anyhow::ensure!(f.can_filter(), "policy {f:?} imposes no eligibility rule");
        }
        Ok(Self { filters, inner })
    }

    /// The filter views, in priority order. The `expect` cannot fire: `new`
    /// checked every entry and `filters` never grows.
    fn views(&self) -> impl Iterator<Item = &dyn EligibilityFilter> {
        (self.filters.iter()).map(|p| p.as_filter().expect("checked by Pipeline::new"))
    }
}

impl Policy for Pipeline {
    /// `None` when an [`OnEmpty::Hold`] filter found nobody eligible -- the
    /// caller turns that into a refusal rather than routing anyway.
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let eligible = admit(self.views(), workers, ctx)?;
        self.inner.select(&eligible, ctx)
    }

    /// An upper bound over BOTH layers: over-reporting costs one tokenization,
    /// under-reporting silently blinds a prompt-routed filter or term.
    fn needs_request_tokens(&self) -> bool {
        self.inner.needs_request_tokens() || self.views().any(|f| f.needs_tokens())
    }

    fn attach_metrics(&self, metrics: Arc<crate::server::metrics::MetricsRegistry>) {
        self.inner.attach_metrics(metrics);
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
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::round_robin::RoundRobinPolicy;

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

    fn urls(ws: &[Arc<Worker>]) -> Vec<String> {
        ws.iter().map(|w| w.url.clone()).collect()
    }

    /// A `--fuse` term: the policy, and the optional `name=weight` override.
    fn term(p: impl Policy + 'static, w: Option<f32>) -> (Arc<dyn Policy>, Option<f32>) {
        (Arc::new(p), w)
    }

    /// Scores a worker by its position, so the ranking is known exactly.
    /// Positional fields: weight, take the DISAGREEING selector, need tokens.
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

    /// Admits exactly the named workers, so the surviving set is known by name
    /// rather than inferred. `Keep(vec![], _)` is the admits-nobody filter.
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

    /// Filter-only, so its `Policy` is hand-written: the blanket impl belongs
    /// to `ScoringPolicy` and a second one would collide.
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

    /// DISAGREES with argmax on those scores -- argmax takes the last index,
    /// this the first. Two that agreed would let the dispatch test pass vacuously.
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

        // The two selectors really do disagree on this score vector.
        let scores = by(1.0).scores(&ws, &ctx);
        assert_eq!(ARGMAX.pick(&ws, &scores), Some(2));
        assert_eq!(PICK_FIRST.pick(&ws, &scores), Some(0));

        // So routing the SAME scores through each is a real discrimination.
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
        // A hand-written `impl Policy` opts out, and cannot claim otherwise...
        let rr: Arc<dyn Policy> = Arc::new(RoundRobinPolicy::new());
        assert!(!rr.can_fuse());
        assert!(rr.as_scoring().is_none());
        // ...and `new` REFUSES it, so the derived flag is load-bearing.
        let err = FusedScorePolicy::new(vec![term(RoundRobinPolicy::new(), None)])
            .expect_err("round_robin has no per-worker preference to contribute");
        assert!(err.to_string().contains("does not support fusion"), "{err}");
    }

    /// Nesting, plus both weight paths: `None` keeps the term's own, `Some` wins.
    #[test]
    fn fusion_nests_and_the_override_replaces_the_terms_own_weight() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let inner = FusedScorePolicy::new(vec![term(by(2.0), None)]).unwrap();
        assert_eq!(inner.scores(&ws, &ctx), vec![0.0, 2.0, 4.0], "its own 2i");

        // Own weight 3.0, override 10.0: an ignored override reads [0, 5, 10].
        let outer =
            FusedScorePolicy::new(vec![term(inner, None), term(by(3.0), Some(10.0))]).unwrap();
        assert_eq!(outer.scores(&ws, &ctx), vec![0.0, 12.0, 24.0], "2i + 10i");
        assert_eq!(outer.select(&ws, &ctx).unwrap().id, ws[2].id);
    }

    /// Drop this and ingress never tokenizes: hungry terms score nothing, silently.
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

        // ...and from a token-hungry FILTER, which no term would reveal.
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

    /// The point of the eligibility layer: a rejected worker cannot be scored
    /// back in, however loud the preference for it.
    #[test]
    fn a_rejected_worker_cannot_be_out_weighed() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        // `by` ranks c > b > a, and a 1e9 override makes that preference as
        // loud as an f32 can be -- but `c` is not eligible.
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

        // The control: same term, same weight, `c` eligible -> it wins. So the
        // assert above is about eligibility, not about the ranking.
        let open = Pipeline::new(
            vec![Arc::new(Keep(vec!["a", "b", "c"], OnEmpty::Abstain))],
            Arc::new(FusedScorePolicy::new(vec![term(by(1.0), Some(1e9))]).unwrap()),
        )
        .unwrap();
        assert_eq!(open.select(&ws, &ctx).unwrap().url, ws[2].url);
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

        // Reversed, the OTHER one yields -- so this is priority, not a fixed
        // preference for one of the two.
        let rev = vec![
            keep(&["c"], OnEmpty::Abstain),
            keep(&["a", "b"], OnEmpty::Abstain),
        ];
        assert_eq!(urls(&admit(refs(&rev), &ws, &ctx).unwrap()), urls(&ws[2..]));
    }

    /// A yielding filter must not take the REST of the list down with it, or a
    /// capacity rule stops applying on exactly the requests that miss the cache.
    #[test]
    fn a_filter_after_a_conflict_still_applies() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);

        let chain = vec![
            keep(&["a", "b"], OnEmpty::Abstain),
            keep(&["c"], OnEmpty::Abstain), // conflicts, yields
            keep(&["b", "c"], OnEmpty::Abstain),
        ];
        // Only `b` satisfies the first and the third. Skipping the tail after
        // the conflict would read [a, b].
        assert_eq!(
            urls(&admit(refs(&chain), &ws, &ctx).unwrap()),
            vec![ws[1].url.clone()]
        );
    }

    /// `Hold` is the other half of the contract, and what makes an admission
    /// rule safe to put anywhere in the list: it refuses instead of yielding.
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

        // And a Hold that CAN be satisfied does not refuse -- otherwise the
        // assert above would pass for a filter that always refuses.
        let ok = vec![
            keep(&["a", "b"], OnEmpty::Abstain),
            keep(&["b"], OnEmpty::Hold),
        ];
        assert_eq!(
            urls(&admit(refs(&ok), &ws, &ctx).unwrap()),
            vec![ws[1].url.clone()]
        );

        // And it surfaces through the policy as "did not route".
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

    /// Missing flags admit rather than mis-index, and the decision still routes
    /// over the workers the filter did answer for.
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

    /// One type serving both layers off one computation, and the plumbing that
    /// makes its filter half reachable from an `Arc<dyn Policy>`.
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
        // The filter half is the hungry one; reading only the scores would
        // report false and ingress would never tokenize for it.
        assert!(
            p.needs_request_tokens(),
            "hunger comes from the filter half"
        );
    }
}
