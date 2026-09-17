// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Basic balancing strategies for prefill and decode.

use crate::config::DecodePolicyKind;
use crate::policies::admission::{
    compare_decode_pressure, compare_prefill_pressure, resolve_decode, CandidateDomain,
    DecisionReason, FinalDecision, FreshLoadLookup, RoutingStage,
};
use crate::policies::scoring::ScoringPolicy;
use crate::policies::{Policy, ProposalKind, SelectionContext, SelectionProposal};
use crate::workers::engine_reports::EngineSnapshot;
use crate::workers::pools::select_decode_with_affinity;
use crate::workers::Worker;
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Policy for RoundRobinPolicy {
    fn select(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }
        let i = self.counter.fetch_add(1, Ordering::Relaxed) % workers.len();
        Some(workers[i].clone())
    }
}

#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl ScoringPolicy for RandomPolicy {
    /// Argmax of n iid uniforms IS a uniform choice: exactly the old `choose`.
    /// Never constrains: a coin toss is not an eligibility rule.
    fn scores(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..workers.len()).map(|_| rng.gen()).collect()
    }
}

#[derive(Debug, Default)]
pub struct PowerOfTwoChoicesPolicy;

impl PowerOfTwoChoicesPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Policy for PowerOfTwoChoicesPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        select_with_snapshot(workers, ctx.load_snapshot())
    }

    /// Returns the primary and backup from one sample.
    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match workers.len() {
            0 => None,
            1 => Some(
                SelectionProposal::primary(workers[0].clone()).with_kind(ProposalKind::PowerOfTwo),
            ),
            len => {
                let (i, j) = sample_two_distinct(len);
                let (primary, backup) = ordered_pair(&workers[i], &workers[j], ctx);
                Some(SelectionProposal::with_backup(primary, backup))
            }
        }
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }
}

pub(crate) fn select_with_snapshot(
    workers: &[Arc<Worker>],
    snapshot: Option<&EngineSnapshot>,
) -> Option<Arc<Worker>> {
    match workers.len() {
        0 => None,
        1 => Some(workers[0].clone()),
        len => {
            let (i, j) = sample_two_distinct(len);
            Some(select_lower_pressure(&workers[i], &workers[j], snapshot))
        }
    }
}

fn select_lower_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&EngineSnapshot>,
) -> Arc<Worker> {
    ordered_pair_with_snapshot(left, right, snapshot).0
}

fn ordered_pair(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    ctx: &SelectionContext<'_>,
) -> (Arc<Worker>, Arc<Worker>) {
    ordered_pair_with_snapshot(left, right, ctx.load_snapshot())
}

fn ordered_pair_with_snapshot(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&EngineSnapshot>,
) -> (Arc<Worker>, Arc<Worker>) {
    if compare_prefill_pressure(left, right, snapshot).is_gt() {
        (Arc::clone(right), Arc::clone(left))
    } else {
        (Arc::clone(left), Arc::clone(right))
    }
}

/// Prefers the least-loaded candidate; `select()` is the blanket impl's.
#[derive(Debug, Default)]
pub struct LoadBasedPolicy;

impl LoadBasedPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl ScoringPolicy for LoadBasedPolicy {
    fn needs_load_snapshot(&self) -> bool {
        true
    }

    fn needs_dispatch_timestamps(&self) -> bool {
        true
    }

    /// `1.0` for the least loaded down to `0.0` for the most, min-max scaled to
    /// the CURRENT fleet -- relative, not absolute, so it cannot saturate:
    /// `1 - load/256` reads a busy fleet as all-`0.0`, tied inside
    /// `TIE_EPSILON`, so the term dies exactly when load matters most.
    ///
    /// Purely a preference: "everybody is busy" is not a reason to refuse to
    /// route, so this term never constrains. Capacity is `--filter`'s job.
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let lookup = FreshLoadLookup::new(ctx.load_snapshot(), workers.iter());
        let loads: Vec<usize> = workers.iter().map(|w| lookup.score_load(w)).collect();
        let lo = loads.iter().min().copied().unwrap_or(0);
        let span = (loads.iter().max().copied().unwrap_or(0) - lo) as f32;
        // `max(1.0)` is exact: a zero span means every `l - lo` is zero too.
        let score = |l: usize| 1.0 - (l - lo) as f32 / span.max(1.0);
        loads.into_iter().map(score).collect()
    }
}

#[derive(Debug, Default)]
pub struct DecodeSelectionContext<'a> {
    load_snapshot: Option<&'a EngineSnapshot>,
    prefill_url: Option<&'a str>,
}

impl<'a> DecodeSelectionContext<'a> {
    pub fn new() -> Self {
        Self {
            load_snapshot: None,
            prefill_url: None,
        }
    }

    /// Engine load snapshot captured at request ingress.
    pub fn with_load_snapshot(mut self, load_snapshot: &'a EngineSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    pub fn load_snapshot(&self) -> Option<&EngineSnapshot> {
        self.load_snapshot
    }

    /// Prefill URL used by `legacy_host_affinity`.
    pub fn with_prefill_url(mut self, prefill_url: &'a str) -> Self {
        self.prefill_url = Some(prefill_url);
        self
    }

    pub fn prefill_url(&self) -> Option<&str> {
        self.prefill_url
    }
}

pub trait DecodePolicy: Send + Sync + std::fmt::Debug {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal>;
}

/// Resolves decode admission and degrades to Power-of-Two when capacity is exhausted.
pub fn resolve_decode_with_capacity_fallback(
    domain: &CandidateDomain,
    proposal: &SelectionProposal,
    request_kv_tokens: u64,
    snapshot: &EngineSnapshot,
) -> Option<FinalDecision> {
    if let Some(decision) = resolve_decode(domain, proposal, request_kv_tokens, snapshot) {
        return Some(decision);
    }
    if domain.stage != RoutingStage::Decode
        || !domain
            .workers
            .iter()
            .any(|worker| worker.id == proposal.primary.id)
    {
        return None;
    }

    let fallback = DecodePowerOfTwoPolicy::new().propose(
        domain,
        &DecodeSelectionContext::new().with_load_snapshot(snapshot),
    )?;
    Some(FinalDecision {
        selected: fallback.primary,
        primary: Arc::clone(&proposal.primary),
        backup: proposal
            .backup
            .as_ref()
            .filter(|backup| domain.workers.iter().any(|worker| worker.id == backup.id))
            .cloned(),
        reason: DecisionReason::CapacityFallbackPowerOfTwo,
        candidate_range_id: domain.id.clone(),
        load_snapshot_version: snapshot.version,
    })
}

/// Samples two workers from a decode domain and orders them by decode pressure.
#[derive(Debug, Default)]
pub struct DecodePowerOfTwoPolicy;

impl DecodePowerOfTwoPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl DecodePolicy for DecodePowerOfTwoPolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match domain.workers.len() {
            0 => None,
            1 => Some(
                SelectionProposal::primary(Arc::clone(&domain.workers[0]))
                    .with_kind(ProposalKind::PowerOfTwo),
            ),
            len => {
                let (i, j) = sample_two_distinct(len);
                let left = &domain.workers[i];
                let right = &domain.workers[j];
                let (primary, backup) =
                    if compare_decode_pressure(left, right, ctx.load_snapshot()).is_gt() {
                        (Arc::clone(right), Arc::clone(left))
                    } else {
                        (Arc::clone(left), Arc::clone(right))
                    };
                Some(
                    SelectionProposal::with_backup(primary, backup)
                        .with_kind(ProposalKind::PowerOfTwo),
                )
            }
        }
    }
}

/// Compatibility policy for legacy same-host PD decode selection.
#[derive(Debug, Default)]
pub struct LegacyHostAffinityDecodePolicy;

impl DecodePolicy for LegacyHostAffinityDecodePolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        let prefill_url = ctx.prefill_url()?;
        select_decode_with_affinity(prefill_url, &domain.workers).map(SelectionProposal::primary)
    }
}

/// Builds a decode policy scoped to one role.
pub fn build_decode_policy(kind: DecodePolicyKind) -> Box<dyn DecodePolicy> {
    match kind {
        DecodePolicyKind::PowerOfTwo => Box::new(DecodePowerOfTwoPolicy::new()),
        DecodePolicyKind::LegacyHostAffinity => Box::new(LegacyHostAffinityDecodePolicy),
    }
}

/// Uniform ordered pair without replacement. Callers handle pools smaller than two.
fn sample_two_distinct(len: usize) -> (usize, usize) {
    debug_assert!(len >= 2);
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..len);
    let j = rng.gen_range(0..len - 1);
    (i, j + usize::from(j >= i))
}

#[cfg(test)]
mod random_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::Policy;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    /// Distributional: `select()` is not pure. Marginals alone are satisfied by
    /// a ROTATION -- what a constant `scores()` becomes under ARGMAX's rotating
    /// tiebreak -- so REPEATS share the band: P(pick==prev) is 1/N iid, 0 rotating.
    #[test]
    fn picks_uniformly_over_20k_draws_and_repeats_at_the_iid_rate() {
        const MEAN: f64 = 5_000.0; // 20_000 draws over 4 workers; repeats too
        const BAND: f64 = 5.0 * 61.237_244; // 5 sigma, sqrt(20_000 / 4 * 3 / 4)
        let (policy, model) = (RandomPolicy::new(), ModelId("tiny".into()));
        let ctx = SelectionContext::new(&model, None);
        let ws: Vec<Arc<Worker>> = (0..4).map(|i| worker(&format!("w{i}"))).collect();
        assert!(policy.select(&[], &ctx).is_none(), "empty fleet");
        let (mut counts, mut repeats, mut prev) = ([0usize; 4], 0usize, None);
        for _ in 0..20_000 {
            let got = policy.select(&ws, &ctx).expect("non-empty fleet");
            let i = ws.iter().position(|w| w.id == got.id).expect("a candidate");
            counts[i] += 1;
            repeats += usize::from(prev.replace(i) == Some(i));
        }
        for n in counts.iter().chain([&repeats]) {
            let dev = (*n as f64 - MEAN).abs(); // `> 0` is no-starvation
            assert!(*n > 0 && dev < BAND, "{counts:?} {repeats}");
        }
    }
}

#[cfg(test)]
mod load_based_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::scoring::TIE_EPSILON;
    use crate::policies::Policy;
    use crate::workers::engine_reports::{EngineSnapshot, EngineWorkerLoad};
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

    #[test] // upstream's, retargeted from `pick_min_load` onto blanket `select`
    fn empty_returns_none() {
        let m = ModelId("tiny".into());
        let ctx = SelectionContext::new(&m, None);
        assert!(LoadBasedPolicy::new().select(&[], &ctx).is_none());
    }

    /// `select()` alone CANNOT detect a broken score: ARGMAX breaks a tie on
    /// load, so a constant `scores()` still lands on the minimum and that arm
    /// passes for the wrong reason. Ranking is therefore asserted on the vector
    /// itself, strictly outside `TIE_EPSILON` so a saturating curve cannot hide
    /// in the tie band -- what `300,900` is for. NaN needs its own arm because
    /// no ORDERING sees it: it makes every comparison false, which on `0,0` is
    /// the expected answer. Upstream's `picks_lowest_active_load` goes under
    /// rule 4 -- the unique-minimum 2-worker case, which `0,1` subsumes.
    #[test]
    fn scores_rank_strictly_by_load_and_the_choice_lands_on_the_minimum() {
        let model = ModelId("tiny".into());
        let (ctx, p) = (SelectionContext::new(&model, None), LoadBasedPolicy::new());
        for spec in ["0,1", "2,1,0", "1,0,1", "0,0", "5,2,9,2", "300,900"] {
            let loads: Vec<usize> = spec.split(',').map(|s| s.parse().unwrap()).collect();
            let ws: Vec<Arc<Worker>> = (0..loads.len()).map(|i| worker(&format!("w{i}"))).collect();
            let _held: Vec<_> = (ws.iter().zip(&loads))
                .flat_map(|(w, n)| (0..*n).map(move |_| w.load_guard()))
                .collect();
            let scores = p.scores(&ws, &ctx);
            for (i, j) in (0..loads.len()).flat_map(|i| (0..loads.len()).map(move |j| (i, j))) {
                let ok = (scores[i] > scores[j] + TIE_EPSILON, scores[i].is_nan());
                assert_eq!(ok, (loads[i] < loads[j], false), "{spec} scored {scores:?}");
            }
            let got = p.select(&ws, &ctx).expect("non-empty").active_load();
            assert_eq!(got, *loads.iter().min().expect("non-empty"), "{spec}");
        }
    }

    #[test]
    fn request_snapshot_overrides_later_router_active_load() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        // After the request snapshot, local counters say w0 is lighter.
        // The policy must still preserve the frozen Engine Load ordering.
        let _after_snapshot: Vec<_> = (0..10).map(|_| w1.load_guard()).collect();
        let snapshot = EngineSnapshot::from_workers(
            23,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 50,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at: Instant::now(),
                    },
                ),
                (
                    w1.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at: Instant::now(),
                    },
                ),
            ]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "load-based scoring must use the request snapshot before local active-load"
        );
    }

    #[test]
    fn recent_dispatches_after_snapshot_change_load_based_choice() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let captured_at = Instant::now();
        let snapshot = EngineSnapshot::from_workers(
            37,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 0,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
                (
                    w1.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
            ]),
        );
        let _after_snapshot = [w0.timestamped_load_guard(), w0.timestamped_load_guard()];
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "dispatches newer than Engine Load must correct its queue depth"
        );
    }

    #[test]
    fn dispatches_before_snapshot_are_not_double_counted() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _before_snapshot = [w0.timestamped_load_guard(), w0.timestamped_load_guard()];
        std::thread::sleep(std::time::Duration::from_millis(5));
        let captured_at = Instant::now();
        let snapshot = EngineSnapshot::from_workers(
            41,
            HashMap::from([
                (
                    w0.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 0,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
                (
                    w1.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: 1,
                        num_waiting_reqs: 0,
                        num_tokens: 0,
                        max_total_num_tokens: 0,
                        captured_at,
                    },
                ),
            ]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w0.id,
            "slots already covered by the snapshot must not be added again"
        );
    }

    #[test]
    fn incomplete_snapshot_uses_frozen_local_active_fallback() {
        let model = ModelId("tiny".into());
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _local_load = [w0.load_guard(), w0.load_guard()];
        let snapshot = EngineSnapshot::from_workers(
            43,
            HashMap::from([(
                w0.url.clone(),
                EngineWorkerLoad {
                    num_running_reqs: 0,
                    num_waiting_reqs: 0,
                    num_tokens: 0,
                    max_total_num_tokens: 0,
                    captured_at: Instant::now(),
                },
            )]),
        );
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&snapshot);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert_eq!(
            LoadBasedPolicy::new().select(&workers, &ctx).unwrap().id,
            w1.id,
            "a partial Engine Load set must not mix engine and local gauges"
        );
    }
}

#[cfg(test)]
mod proposal_tests {
    use crate::policies::balancing::PowerOfTwoChoicesPolicy;
    use crate::policies::test_support::snapshot;
    use crate::policies::test_support::worker;
    use crate::policies::test_support::TestEngineLoad;
    use crate::policies::*;
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
}
