// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Routing terms that judge each worker: admit it with a preference, or reject
//! it outright.
//!
//! [`Criterion`] is the whole of it. A term answers [`Verdict`] per worker, and
//! the blanket `impl<T: Criterion> Policy for T` makes one usable standalone.
//! [`Verdict::Reject`] is a VARIANT and not a very small number on purpose: a
//! number can be out-weighed, so an admission or bucket rule expressed as
//! `-1e9` is a rule any sufficiently confident other term can overrule. A
//! rejected worker cannot win whatever the weights say.
//!
//! Terms fuse in LIST ORDER, and that order is a priority: see
//! [`FusedScorePolicy`] for what happens when two of them cannot both be
//! satisfied.

pub mod argmax;
pub mod prefix_cache;

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use argmax::{Selector, ARGMAX};
use std::sync::Arc;

/// One term's answer about one worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Not eligible. Never enters the sum, so no weight can overrule it.
    Reject,
    /// Eligible, and worth this much; higher is better. `Score(0.0)` is the
    /// abstention: eligible, no opinion.
    Score(f32),
}

/// What a term means when it has rejected every worker it was shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnEmpty {
    /// Treat the term as having said nothing, and route without it. For
    /// affinity: "nobody holds this prompt" must not become a 503.
    Abstain,
    /// The rejection stands and the request is not routable here. For
    /// admission: "every worker is over its cap" must not silently pick one
    /// anyway. Also a declaration that this term never yields to a later one.
    Hold,
}

pub trait Criterion: Send + Sync + std::fmt::Debug {
    /// Judge the candidates still standing, parallel to `workers`.
    ///
    /// The whole surviving set at once, so a term can compare candidates
    /// against each other, take a top-k, or hash the prompt once per decision
    /// rather than once per worker.
    ///
    /// A term with no signal this request must abstain — `Score(0.0)` for
    /// everyone — and NOT reject everyone. A cache miss is "no opinion", not
    /// "nobody is fit to serve"; see [`FusedScorePolicy`] for why the
    /// difference is load-bearing rather than stylistic.
    fn judge(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<Verdict>;

    /// Default multiplier as a fused term; `--fuse name=weight` overrides it.
    /// Applies to [`Verdict::Score`] only — a rejection has no magnitude.
    fn weight(&self) -> f32 {
        1.0
    }

    /// Whether this judgement needs `ctx.request_tokens()`; surfaces as
    /// [`Policy::needs_request_tokens`], which is what makes ingress tokenize.
    fn needs_tokens(&self) -> bool {
        false
    }

    /// See [`OnEmpty`]. Defaulted to the safe direction: a term that rejects
    /// everyone gets ignored rather than dropping the request.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Abstain
    }

    /// How the admitted scores collapse to one winner. The default is why a
    /// bare criterion needs nothing but `judge()`.
    fn selector(&self) -> &dyn Selector {
        &ARGMAX
    }
}

/// The candidates a chain of terms admitted, with their summed scores.
/// `workers[i]` scored `scores[i]`; both are shorter than the fleet whenever
/// some term rejected somebody.
#[derive(Debug)]
pub struct Admitted {
    pub workers: Vec<Arc<Worker>>,
    pub scores: Vec<f32>,
}

/// Apply `terms` in order and return who survived.
///
/// `None` means no worker is admissible AND the term that said so declared
/// [`OnEmpty::Hold`] — the caller must not route.
///
/// Terms are applied one at a time and each sees only the survivors, so a term
/// is never asked about a worker an earlier, higher-priority term already
/// excluded. That also makes the walk lazy: once a term is dropped, the ones
/// after it are never even evaluated.
fn admit(
    terms: &mut dyn Iterator<Item = (&dyn Criterion, f32)>,
    workers: &[Arc<Worker>],
    ctx: &SelectionContext<'_>,
) -> Option<Admitted> {
    let mut alive: Vec<Arc<Worker>> = workers.to_vec();
    let mut scores: Vec<f32> = vec![0.0; workers.len()];

    for (term, weight) in terms {
        if alive.is_empty() {
            break;
        }
        let verdicts = term.judge(&alive, ctx);
        if verdicts.len() != alive.len() {
            // Arity is the term's contract, but breaking it must degrade
            // rather than mis-index: the tail abstains and the decision still
            // routes. Said out loud, because a term that always lands here has
            // silently stopped contributing.
            tracing::debug!(
                term = ?term,
                n_workers = alive.len(),
                n_verdicts = verdicts.len(),
                "criterion returned the wrong arity; missing verdicts abstain",
            );
        }
        // `alive.len() == workers.len()` means nothing has narrowed yet, so a
        // total rejection here is this term's own doing rather than a clash
        // with an earlier one. Worth telling apart: the first is a term that
        // should have abstained, the second is ordinary contention.
        let untouched = alive.len() == workers.len();

        let mut next = Vec::with_capacity(alive.len());
        let mut next_scores = Vec::with_capacity(alive.len());
        for (i, w) in alive.iter().enumerate() {
            match verdicts.get(i).copied().unwrap_or(Verdict::Score(0.0)) {
                Verdict::Reject => {}
                Verdict::Score(s) => {
                    next.push(Arc::clone(w));
                    next_scores.push(scores[i] + weight * s);
                }
            }
        }

        if next.is_empty() {
            match term.on_empty() {
                OnEmpty::Hold => return None,
                OnEmpty::Abstain if untouched => {
                    tracing::warn!(
                        term = ?term,
                        n_workers = workers.len(),
                        "criterion rejected every worker on its own; a term with no \
                         signal should abstain (Score(0.0)), not veto the fleet",
                    );
                    continue;
                }
                OnEmpty::Abstain => {
                    // Conflict, not a bug: this term and an earlier one cannot
                    // both be satisfied. Order is priority, so THIS term is the
                    // one that yields -- its whole verdict is discarded, the
                    // earlier narrowing stands, and the chain carries on.
                    //
                    // Discarding only this term, rather than it and everything
                    // after it, is what keeps a later pure scorer ranking: drop
                    // the tail instead and `--fuse cache,load` would stop load
                    // balancing on exactly the requests that miss the cache.
                    tracing::debug!(
                        term = ?term,
                        n_alive = alive.len(),
                        "criterion conflicts with a higher-priority one; abstaining",
                    );
                    continue;
                }
            }
        }
        alive = next;
        scores = next_scores;
    }

    Some(Admitted {
        workers: alive,
        scores,
    })
}

/// A criterion is usable on its own: admit, then argmax the admitted.
impl<T: Criterion> Policy for T {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let one = (self as &dyn Criterion, self.weight());
        let admitted = admit(&mut std::iter::once(one), workers, ctx)?;
        let i = self.selector().pick(&admitted.workers, &admitted.scores)?;
        admitted.workers.get(i).map(Arc::clone)
    }

    fn needs_request_tokens(&self) -> bool {
        self.needs_tokens()
    }

    fn as_criterion(&self) -> Option<&dyn Criterion> {
        Some(self)
    }
}

/// Several terms applied in list order: reject wins over any score, and the
/// scores of everyone still standing are summed.
///
/// **Order is priority.** When two terms cannot both be satisfied — the later
/// one would reject everyone the earlier admitted — the LATER one abstains: its
/// verdict is discarded whole, the earlier narrowing stands, and the rest of
/// the list still runs. So `--fuse admission,session,prefix_cache,load_based`
/// keeps admission and the session pin, gives up only the cache preference, and
/// still ranks by load — rather than throwing the tail away, or throwing all of
/// it away and routing over the raw fleet.
///
/// A term that only ever answers [`Verdict::Score`] can never be the one that
/// yields, so putting `load_based` last costs it nothing: ranking always runs.
///
/// Itself a [`Criterion`], so fusion nests. A term is an `Arc<dyn Policy>` and
/// NOT a `Box<dyn Criterion>`, because `as_criterion()` hands out a BORROW that
/// a factory holding an `Arc<dyn Policy>` cannot turn into an owned box.
#[derive(Debug)]
pub struct FusedScorePolicy {
    /// Each term with its `--fuse name=weight` override; `None` keeps the
    /// term's own [`Criterion::weight`]. Order is the priority order.
    terms: Vec<(Arc<dyn Policy>, Option<f32>)>,
}

/// A term's criterion view and the weight to apply. The `expect` cannot fire:
/// `new` rejected every term whose view was `None`, and `terms` never grows.
fn view(t: &(Arc<dyn Policy>, Option<f32>)) -> (&dyn Criterion, f32) {
    let c =
        t.0.as_criterion()
            .expect("checked by FusedScorePolicy::new");
    (c, t.1.unwrap_or_else(|| c.weight()))
}

impl FusedScorePolicy {
    /// Rejects a non-judging term at CONFIG time, so `--fuse round_robin` fails
    /// at startup instead of contributing nothing per request. The only place
    /// [`Policy::can_fuse`] is consulted; without this call it is decoration.
    pub fn new(terms: Vec<(Arc<dyn Policy>, Option<f32>)>) -> anyhow::Result<Self> {
        for (p, _) in &terms {
            anyhow::ensure!(p.can_fuse(), "policy {p:?} does not support fusion");
        }
        Ok(Self { terms })
    }
}

impl Criterion for FusedScorePolicy {
    fn judge(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<Verdict> {
        let Some(admitted) = admit(&mut self.terms.iter().map(view), workers, ctx) else {
            // An inner Hold is a hold: propagate it as a rejection of everyone
            // rather than letting the outer chain route around it.
            return vec![Verdict::Reject; workers.len()];
        };
        // Back onto the caller's indexing: whoever did not survive the inner
        // chain is rejected, so an outer sum cannot re-admit them.
        workers
            .iter()
            .map(|w| {
                match admitted
                    .workers
                    .iter()
                    .position(|a| Arc::ptr_eq(a, w))
                    .map(|i| admitted.scores[i])
                {
                    Some(s) => Verdict::Score(s),
                    None => Verdict::Reject,
                }
            })
            .collect()
    }

    fn needs_tokens(&self) -> bool {
        self.terms.iter().map(view).any(|(t, _)| t.needs_tokens())
    }

    /// A fusion holds exactly when the inner chain did; `judge` has already
    /// turned that into an all-reject vector, and re-abstaining here would
    /// undo it.
    fn on_empty(&self) -> OnEmpty {
        OnEmpty::Hold
    }
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

    impl Criterion for ByIndex {
        fn judge(&self, workers: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<Verdict> {
            (0..workers.len())
                .map(|i| Verdict::Score(i as f32))
                .collect()
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
    /// rather than inferred. `Keep(vec![], _)` is the admits-nobody term.
    #[derive(Debug)]
    struct Keep(Vec<&'static str>, OnEmpty);

    impl Criterion for Keep {
        fn judge(&self, workers: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<Verdict> {
            workers
                .iter()
                .map(|w| match self.0.iter().any(|n| w.url.contains(n)) {
                    true => Verdict::Score(0.0),
                    false => Verdict::Reject,
                })
                .collect()
        }
        fn on_empty(&self) -> OnEmpty {
            self.1
        }
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

    fn ctx_for(model: &ModelId) -> SelectionContext<'_> {
        SelectionContext::new(model, None)
    }

    /// [`admit`] over borrowed terms at weight 1.0, which is what the ordering
    /// cases below are about; weights are covered by the fusion tests.
    fn admit_all(
        terms: &[&dyn Criterion],
        ws: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<Admitted> {
        admit(&mut terms.iter().map(|c| (*c, 1.0)), ws, ctx)
    }

    #[test]
    fn selector_dispatch_uses_the_policys_own_selector() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = ctx_for(&model);

        // The two selectors really do disagree on this score vector.
        let scores = [0.0, 1.0, 2.0];
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
            assert!(p.as_criterion().is_some(), "the flag agrees with the view");
        }
        // A hand-written `impl Policy` opts out, and cannot claim otherwise...
        let rr: Arc<dyn Policy> = Arc::new(RoundRobinPolicy::new());
        assert!(!rr.can_fuse());
        assert!(rr.as_criterion().is_none());
        // ...and `new` REFUSES it, so the derived flag is load-bearing.
        let err = FusedScorePolicy::new(vec![term(RoundRobinPolicy::new(), None)])
            .expect_err("round_robin has no per-worker judgement to contribute");
        assert!(err.to_string().contains("does not support fusion"), "{err}");
    }

    /// Nesting, plus both weight paths: `None` keeps the term's own, `Some` wins.
    #[test]
    fn fusion_nests_and_the_override_replaces_the_terms_own_weight() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = ctx_for(&model);

        let inner = FusedScorePolicy::new(vec![term(by(2.0), None)]).unwrap();
        assert_eq!(
            inner.judge(&ws, &ctx),
            vec![
                Verdict::Score(0.0),
                Verdict::Score(2.0),
                Verdict::Score(4.0)
            ],
            "its own 2i"
        );

        // Own weight 3.0, override 10.0: an ignored override reads [0, 5, 10].
        let outer =
            FusedScorePolicy::new(vec![term(inner, None), term(by(3.0), Some(10.0))]).unwrap();
        assert_eq!(
            outer.judge(&ws, &ctx),
            vec![
                Verdict::Score(0.0),
                Verdict::Score(12.0),
                Verdict::Score(24.0)
            ],
            "2i + 10i"
        );
        assert_eq!(outer.select(&ws, &ctx).unwrap().id, ws[2].id);
    }

    /// Drop this and ingress never tokenizes: hungry terms judge nothing, silently.
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
    }

    /// The point of `Reject` being a variant and not a very small number: no
    /// weight, however large, can buy a rejected worker back into the running.
    #[test]
    fn a_rejection_cannot_be_out_weighed() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = ctx_for(&model);

        // `by` ranks c > b > a, and a 1e9 override makes that preference as
        // loud as an f32 can be — but `c` is not admissible.
        let fused = FusedScorePolicy::new(vec![
            term(Keep(vec!["a", "b"], OnEmpty::Abstain), None),
            term(by(1.0), Some(1e9)),
        ])
        .unwrap();
        let won = fused.select(&ws, &ctx).expect("two workers remain");
        assert_eq!(won.url, ws[1].url, "the best ADMITTED, not the best");

        // The control: same terms, same weights, `c` admitted -> it wins. So
        // the assert above is about admissibility, not about the ranking.
        let open = FusedScorePolicy::new(vec![
            term(Keep(vec!["a", "b", "c"], OnEmpty::Abstain), None),
            term(by(1.0), Some(1e9)),
        ])
        .unwrap();
        assert_eq!(open.select(&ws, &ctx).unwrap().url, ws[2].url);
    }

    /// Order is priority: the LOWER-priority term yields, and what the
    /// higher-priority one narrowed to is kept. Asserted on the surviving set
    /// rather than on the winner, because with three workers a wrong rule can
    /// still land on the right one by luck.
    #[test]
    fn a_conflict_drops_the_later_term_and_keeps_the_earlier_narrowing() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = ctx_for(&model);

        let ab = Keep(vec!["a", "b"], OnEmpty::Abstain);
        let c = Keep(vec!["c"], OnEmpty::Abstain);

        let out = admit_all(&[&ab, &c], &ws, &ctx).expect("Abstain never holds");
        assert_eq!(
            urls(&out.workers),
            urls(&ws[..2]),
            "the second term yields; falling back to the raw fleet would read [a, b, c]",
        );

        // Reversed, the OTHER one yields — so this is priority, not a fixed
        // preference for one of the two terms.
        let out = admit_all(&[&c, &ab], &ws, &ctx).expect("Abstain never holds");
        assert_eq!(urls(&out.workers), urls(&ws[2..]));
    }

    /// `Hold` is the other half of the contract, and it is what makes an
    /// admission rule safe to put anywhere in the list: it refuses instead of
    /// yielding, so a mis-ordered chain cannot route around it.
    #[test]
    fn a_holding_term_refuses_instead_of_yielding() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = ctx_for(&model);

        let ab = Keep(vec!["a", "b"], OnEmpty::Abstain);
        let only_c = Keep(vec!["c"], OnEmpty::Hold);
        assert!(
            admit_all(&[&ab, &only_c], &ws, &ctx).is_none(),
            "no admissible worker, and the term said Hold",
        );

        // And a Hold that CAN be satisfied does not refuse — otherwise the
        // assert above would pass for a term that always refuses.
        let only_b = Keep(vec!["b"], OnEmpty::Hold);
        let out = admit_all(&[&ab, &only_b], &ws, &ctx).expect("b satisfies both");
        assert_eq!(urls(&out.workers), vec![ws[1].url.clone()]);
    }

    /// A term that only ever scores can never be the one that yields, so
    /// ranking survives a conflict earlier in the list. Without this the
    /// natural `--fuse cache,load` would silently stop load-balancing on every
    /// cache miss.
    #[test]
    fn a_pure_scorer_still_ranks_after_an_earlier_conflict() {
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = ctx_for(&model);

        let fused = FusedScorePolicy::new(vec![
            term(Keep(vec!["a", "b"], OnEmpty::Abstain), None),
            term(Keep(vec!["c"], OnEmpty::Abstain), None), // conflicts, yields
            term(by(1.0), None),
        ])
        .unwrap();
        // `by` ranks b(1.0) over a(0.0) WITHIN the survivors. A chain that
        // dropped the tail wholesale would score both 0.0 and the load tie
        // would answer `a`.
        assert_eq!(fused.select(&ws, &ctx).unwrap().url, ws[1].url);
    }

    /// Missing verdicts abstain rather than mis-index, and the decision still
    /// routes over the workers the term did answer for.
    #[test]
    fn a_short_verdict_vector_degrades_instead_of_panicking() {
        #[derive(Debug)]
        struct Short;
        impl Criterion for Short {
            fn judge(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Vec<Verdict> {
                vec![Verdict::Reject]
            }
        }
        let ws = fleet();
        let model = ModelId("tiny".into());
        let ctx = ctx_for(&model);
        let out = admit_all(&[&Short], &ws, &ctx).expect("the tail abstained");
        assert_eq!(urls(&out.workers), urls(&ws[1..]), "only index 0 rejected");
    }
}
