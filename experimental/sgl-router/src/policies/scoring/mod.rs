// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Policies whose decision reduces to a per-worker preference.
//!
//! [`ScoringPolicy`] is the whole of it: say what each worker is worth, and the
//! blanket `impl<T: ScoringPolicy> Policy for T` makes it usable standalone by
//! argmaxing its own scores. No wrapper, and no flag to keep in sync:
//! [`super::Policy::can_fuse`] derives from `as_scoring`. That blanket impl
//! coexists with the hand-written `impl Policy`s: all of it is crate-local.

pub mod argmax;
pub mod prefix_cache;

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use argmax::{Selector, ARGMAX};
use std::sync::Arc;

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

    /// How the scores collapse to one winner. The default is why a bare
    /// scoring policy needs nothing but `scores()`.
    fn selector(&self) -> &dyn Selector {
        &ARGMAX
    }
}

/// A scoring policy is usable on its own: argmax its own preference.
impl<T: ScoringPolicy> Policy for T {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let scores = self.scores(workers, ctx);
        let i = self.selector().pick(workers, &scores)?;
        workers.get(i).map(Arc::clone)
    }

    fn needs_request_tokens(&self) -> bool {
        self.needs_tokens()
    }

    fn as_scoring(&self) -> Option<&dyn ScoringPolicy> {
        Some(self)
    }
}

/// The weighted sum of several scoring policies. Itself a [`ScoringPolicy`], so
/// fusion nests: the blanket impl makes a composer a `Policy`, hence a legal
/// term of another composer. A term is an `Arc<dyn Policy>` and NOT a
/// `Box<dyn ScoringPolicy>`, because `as_scoring()` hands out a BORROW that a
/// factory holding an `Arc<dyn Policy>` cannot turn into an owned box.
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
        let ws = vec![worker("a"), worker("b"), worker("c")];
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
        let ws = vec![worker("a"), worker("b"), worker("c")];
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
    }
}
