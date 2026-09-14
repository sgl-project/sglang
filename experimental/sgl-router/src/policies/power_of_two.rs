// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Power-of-k-choices load balancing. Each selection samples `choices`
//! random distinct candidates and dispatches to the least-pressured one,
//! so N router replicas reading the same load snapshot do not converge on
//! one shared fleet minimum. When a queue limit is configured the sample
//! is drawn from the tier of workers the queue gate admits, and only from
//! the whole pool when every worker is queueing (the second tier keeps an
//! all-queueing fleet routable).
//!
//! `select` and `propose` share one scan so they cannot disagree on the
//! same pool: see [`best_two_of_sample`] for why it is a linear scan and
//! never a sort.

use crate::config::DEFAULT_MIN_LOAD_CHOICES;
use crate::policies::admission::{compare_prefill_pressure, queue_gate_admits};
use crate::policies::engine_load::EngineLoadSnapshot;
use crate::policies::{Policy, ProposalKind, SelectionContext, SelectionProposal};
use crate::workers::Worker;
use rand::seq::index::sample;
use rand::Rng;
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Debug)]
pub struct PowerOfTwoChoicesPolicy {
    choices: usize,
    queue_limit: Option<u64>,
}

impl Default for PowerOfTwoChoicesPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerOfTwoChoicesPolicy {
    pub fn new() -> Self {
        Self {
            choices: DEFAULT_MIN_LOAD_CHOICES,
            queue_limit: None,
        }
    }

    /// Sets the sample size and the queue gate for the fallback path.
    /// `choices` is clamped to at least 1.
    pub fn with_load_control(mut self, choices: usize, queue_limit: Option<u64>) -> Self {
        self.choices = choices.max(1);
        self.queue_limit = queue_limit;
        self
    }
}

impl Policy for PowerOfTwoChoicesPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        select_k_with_snapshot(workers, ctx.load_snapshot(), self.choices, self.queue_limit)
    }

    /// Returns the primary and backup from one sample. A sample of one
    /// (`--min-load-choices 1`) has no second member, so the proposal
    /// carries no backup and admission loses its backup paths.
    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        let snapshot = ctx.load_snapshot();
        let pool = sample_pool(workers, snapshot, self.queue_limit);
        let (primary, backup) = best_two_of_sample(&pool, snapshot, self.choices)?;
        match backup {
            Some(backup) => Some(SelectionProposal::with_backup(primary, backup)),
            None => Some(SelectionProposal::primary(primary).with_kind(ProposalKind::PowerOfTwo)),
        }
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }
}

pub(crate) fn select_k_with_snapshot(
    workers: &[Arc<Worker>],
    snapshot: Option<&EngineLoadSnapshot>,
    choices: usize,
    queue_limit: Option<u64>,
) -> Option<Arc<Worker>> {
    let pool = sample_pool(workers, snapshot, queue_limit);
    best_two_of_sample(&pool, snapshot, choices).map(|(primary, _)| primary)
}

/// The tier the sample is drawn from: workers the queue gate admits,
/// or the whole pool when every worker is queueing (the second tier
/// keeps an all-queueing fleet routable). A sample must never land on
/// a queueing worker while an unqueued one exists.
fn sample_pool<'w>(
    workers: &'w [Arc<Worker>],
    snapshot: Option<&EngineLoadSnapshot>,
    queue_limit: Option<u64>,
) -> Cow<'w, [Arc<Worker>]> {
    // Without a limit there is nothing to gate on, and without a snapshot
    // the gate cannot be evaluated per-worker: both mean the whole pool.
    let (Some(snapshot), Some(limit)) = (snapshot, queue_limit) else {
        return Cow::Borrowed(workers);
    };
    let unqueued: Vec<Arc<Worker>> = workers
        .iter()
        .filter(|worker| queue_gate_admits(snapshot, worker.as_ref(), Some(limit)))
        .cloned()
        .collect();
    if unqueued.is_empty() {
        Cow::Borrowed(workers)
    } else {
        Cow::Owned(unqueued)
    }
}

/// The sample's two least-pressured members, best first, as pool indices
/// resolved to workers. `choices >= pool` skips the shuffle and scans in
/// pool order, so the winner is the exact minimum and ties resolve in pool
/// order; a smaller `choices` draws that many distinct indices, which
/// `rand`'s `sample` returns fully shuffled, so ties inside a sample
/// resolve randomly.
///
/// Both tiers scan linearly and never sort. `compare_prefill_pressure`
/// is only a pairwise comparison: two workers that both publish
/// `estimated_prefill_queue_ms` are ordered on that estimate, and any
/// other pair on the waiting-token tuple, so it is not a total order
/// across a mixed set (an idle worker publishes no estimate). Handing it
/// to `sort_by` lets the standard library abort the process with
/// "comparison function does not correctly implement a total order".
fn best_two_of_sample(
    pool: &[Arc<Worker>],
    snapshot: Option<&EngineLoadSnapshot>,
    choices: usize,
) -> Option<(Arc<Worker>, Option<Arc<Worker>>)> {
    let len = pool.len();
    if len == 0 {
        return None;
    }
    let choices = choices.max(1);
    let drawn: Vec<usize> = if choices >= len {
        (0..len).collect()
    } else if choices == 1 {
        vec![rand::thread_rng().gen_range(0..len)]
    } else {
        sample(&mut rand::thread_rng(), len, choices).into_vec()
    };
    let mut best: Option<usize> = None;
    let mut runner_up: Option<usize> = None;
    for index in drawn {
        // Only a strict improvement displaces the incumbent, so a tie keeps
        // whichever member the draw presented first.
        if best.is_none_or(|current| {
            compare_prefill_pressure(&pool[index], &pool[current], snapshot).is_lt()
        }) {
            runner_up = best;
            best = Some(index);
        } else if runner_up.is_none_or(|current| {
            compare_prefill_pressure(&pool[index], &pool[current], snapshot).is_lt()
        }) {
            runner_up = Some(index);
        }
    }
    best.map(|index| {
        (
            Arc::clone(&pool[index]),
            runner_up.map(|index| Arc::clone(&pool[index])),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::engine_load::NativeCacheWorkerLoad;
    use std::time::Instant;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: None,
        }))
    }

    /// Snapshot keyed on waiting depth; `waiting` sets both the queue-gate
    /// reading (`num_waiting_reqs`) and the pressure ordering
    /// (`num_waiting_uncached_tokens`), so one knob drives both.
    fn snapshot(entries: &[(&Arc<Worker>, u64)]) -> EngineLoadSnapshot {
        EngineLoadSnapshot::from_native_cache_workers(
            7,
            entries
                .iter()
                .map(|(worker, waiting)| {
                    (
                        worker.url.clone(),
                        NativeCacheWorkerLoad {
                            num_running_reqs: 0,
                            num_waiting_reqs: *waiting,
                            num_waiting_uncached_tokens: *waiting,
                            num_used_tokens: 10,
                            num_total_tokens: 10,
                            max_total_num_tokens: 10_000,
                            max_running_requests: 64,
                            prefill_throughput_tokens_per_s: None,
                            estimated_prefill_queue_ms: None,
                            captured_at: Instant::now(),
                        },
                    )
                })
                .collect(),
        )
    }

    /// A fleet where only some workers publish `estimated_prefill_queue_ms`.
    /// An idle worker has no throughput delta to derive one from, so this is
    /// the steady state, not an edge case - and it makes
    /// `compare_prefill_pressure` intransitive: a slow worker with a shallow
    /// queue loses to a fast worker with a deep one on the estimate, while
    /// both are ordered against an estimate-less worker on waiting tokens.
    fn mixed_estimate_snapshot(workers: &[Arc<Worker>]) -> EngineLoadSnapshot {
        EngineLoadSnapshot::from_native_cache_workers(
            11,
            workers
                .iter()
                .enumerate()
                .map(|(index, worker)| {
                    let waiting = (index as u64 * 7) % 13;
                    (
                        worker.url.clone(),
                        NativeCacheWorkerLoad {
                            num_running_reqs: 0,
                            num_waiting_reqs: 0,
                            num_waiting_uncached_tokens: waiting,
                            num_used_tokens: 10,
                            num_total_tokens: 10,
                            max_total_num_tokens: 10_000,
                            max_running_requests: 64,
                            prefill_throughput_tokens_per_s: None,
                            // Every third worker is idle and publishes no
                            // estimate; the rest rank inversely to `waiting`.
                            estimated_prefill_queue_ms: (index % 3 != 0)
                                .then(|| (13 - waiting) as f64),
                            captured_at: Instant::now(),
                        },
                    )
                })
                .collect(),
        )
    }

    /// `compare_prefill_pressure` is a pairwise comparison, not a total
    /// order, so the k-way minimum must be a linear scan. Sorting a sample
    /// this size aborts the process with "comparison function does not
    /// correctly implement a total order" - a live outage on any fleet with
    /// a large `--min-load-choices` and a mix of idle and busy workers.
    #[test]
    fn a_large_sample_over_mixed_estimates_never_aborts() {
        let model = ModelId("model".into());
        let workers: Vec<Arc<Worker>> = (0..64).map(|i| worker(&format!("w{i}"))).collect();
        let loads = mixed_estimate_snapshot(&workers);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&loads);

        for choices in [2, 3, 21, 32, 64, 128] {
            let policy = PowerOfTwoChoicesPolicy::new().with_load_control(choices, None);
            for _ in 0..16 {
                let proposal = policy
                    .propose(&workers, &ctx)
                    .expect("a non-empty fleet must produce a proposal");
                assert!(workers.iter().any(|w| w.id == proposal.primary.id));
                let backup = proposal.backup.expect("k >= 2 must carry a runner-up");
                assert_ne!(
                    backup.id, proposal.primary.id,
                    "the sample draws distinct indices"
                );
                select_k_with_snapshot(&workers, Some(&loads), choices, None)
                    .expect("select must agree that the fleet is routable");
            }
        }
    }

    /// `propose` and `select` must rank the same pool the same way.
    #[test]
    fn propose_and_select_agree_on_the_sample_minimum() {
        let model = ModelId("model".into());
        let deep = worker("deep");
        let middle = worker("middle");
        let shallow = worker("shallow");
        let workers = vec![Arc::clone(&deep), Arc::clone(&middle), Arc::clone(&shallow)];
        let loads = snapshot(&[(&deep, 100), (&middle, 50), (&shallow, 1)]);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&loads);

        // `choices >= pool` on both paths: the exact minimum, and the
        // runner-up is the second-lowest rather than a shuffle artifact.
        let proposal = PowerOfTwoChoicesPolicy::new()
            .with_load_control(3, None)
            .propose(&workers, &ctx)
            .expect("three candidates must produce a proposal");
        assert_eq!(proposal.primary.id, shallow.id);
        assert_eq!(
            proposal.backup.expect("a three-member sample has one").id,
            middle.id
        );
        assert_eq!(
            select_k_with_snapshot(&workers, Some(&loads), 3, None)
                .expect("the pool is non-empty")
                .id,
            proposal.primary.id
        );
    }

    /// A one-member sample has no runner-up, so admission loses its backup
    /// paths entirely. Documented on `--min-load-choices`; pinned here.
    #[test]
    fn one_choice_proposes_no_backup() {
        let model = ModelId("model".into());
        let left = worker("left");
        let right = worker("right");
        let workers = vec![Arc::clone(&left), Arc::clone(&right)];
        let loads = snapshot(&[(&left, 1), (&right, 2)]);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&loads);

        let proposal = PowerOfTwoChoicesPolicy::new()
            .with_load_control(1, None)
            .propose(&workers, &ctx)
            .expect("a non-empty pool must produce a proposal");
        assert!(proposal.backup.is_none());
        assert_eq!(proposal.kind, ProposalKind::PowerOfTwo);
    }

    #[test]
    fn sample_never_lands_on_a_queueing_worker_while_an_unqueued_one_exists() {
        let queued_a = worker("queued_a");
        let queued_b = worker("queued_b");
        let queued_c = worker("queued_c");
        let unqueued = worker("unqueued");
        let workers = vec![
            Arc::clone(&queued_a),
            Arc::clone(&queued_b),
            Arc::clone(&queued_c),
            Arc::clone(&unqueued),
        ];
        let loads = snapshot(&[
            (&queued_a, 9),
            (&queued_b, 5),
            (&queued_c, 12),
            (&unqueued, 3),
        ]);

        for _ in 0..64 {
            let selected = select_k_with_snapshot(&workers, Some(&loads), 2, Some(4))
                .expect("an unqueued worker exists");
            assert_eq!(selected.id, unqueued.id);
        }
    }

    #[test]
    fn all_queueing_fleet_still_selects_from_the_whole_pool() {
        let left = worker("left");
        let right = worker("right");
        let third = worker("third");
        let workers = vec![Arc::clone(&left), Arc::clone(&right), Arc::clone(&third)];
        let loads = snapshot(&[(&left, 9), (&right, 5), (&third, 12)]);

        // `choices >= pool` on the second tier, so the winner is the exact
        // pressure minimum of the whole fleet rather than any pool member.
        for _ in 0..64 {
            let selected = select_k_with_snapshot(&workers, Some(&loads), 3, Some(4))
                .expect("an all-queueing fleet must still route");
            assert_eq!(
                selected.id, right.id,
                "the whole-pool tier must rank the all-queueing fleet by pressure"
            );
        }
        // With a two-member sample the winner is still never the worker the
        // other two both beat on pressure.
        for _ in 0..64 {
            let selected = select_k_with_snapshot(&workers, Some(&loads), 2, Some(4))
                .expect("an all-queueing fleet must still route");
            assert!(
                workers.iter().any(|worker| worker.id == selected.id),
                "the whole-pool tier must return a pool member"
            );
        }
    }

    #[test]
    fn choices_at_or_above_the_pool_size_returns_the_exact_minimum() {
        let deep = worker("deep");
        let shallow = worker("shallow");
        let middle = worker("middle");
        // Pool order deliberately disagrees with the pressure ordering.
        let workers = vec![Arc::clone(&deep), Arc::clone(&shallow), Arc::clone(&middle)];
        let loads = snapshot(&[(&deep, 100), (&shallow, 1), (&middle, 50)]);

        for choices in [3, 8] {
            let selected = select_k_with_snapshot(&workers, Some(&loads), choices, None)
                .expect("the pool is non-empty");
            assert_eq!(
                selected.id, shallow.id,
                "choices >= pool must be the deterministic exact minimum"
            );
        }
    }

    #[test]
    fn one_choice_draws_within_the_tier_and_ignores_pressure() {
        let deep = worker("deep");
        let shallow = worker("shallow");
        let workers = vec![Arc::clone(&deep), Arc::clone(&shallow)];
        // Both sit under the gate, so the tier is the whole pool and a
        // single draw must reach the deeper worker too - that is exactly
        // what stops N replicas converging on one shared minimum.
        let loads = snapshot(&[(&deep, 3), (&shallow, 0)]);

        let mut saw_deep = false;
        for _ in 0..256 {
            let selected = select_k_with_snapshot(&workers, Some(&loads), 1, Some(4))
                .expect("both workers are unqueued");
            saw_deep |= selected.id == deep.id;
        }
        assert!(
            saw_deep,
            "a one-member sample must be a draw, not the pressure minimum"
        );
    }

    #[test]
    fn one_choice_stays_inside_the_queue_gate_tier() {
        let queued = worker("queued");
        let also_queued = worker("also_queued");
        let unqueued = worker("unqueued");
        let workers = vec![
            Arc::clone(&queued),
            Arc::clone(&also_queued),
            Arc::clone(&unqueued),
        ];
        let loads = snapshot(&[(&queued, 9), (&also_queued, 20), (&unqueued, 0)]);

        for _ in 0..64 {
            let selected = select_k_with_snapshot(&workers, Some(&loads), 1, Some(4))
                .expect("an unqueued worker exists");
            assert_eq!(
                selected.id, unqueued.id,
                "a single draw must still respect the queue-gate tier"
            );
        }
    }

    #[test]
    fn two_workers_with_defaults_propose_both_ordered_by_pressure() {
        let model = ModelId("model".into());
        let busy = worker("busy");
        let idle = worker("idle");
        let workers = vec![Arc::clone(&busy), Arc::clone(&idle)];
        let loads = snapshot(&[(&busy, 512), (&idle, 16)]);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&loads);

        let proposal = PowerOfTwoChoicesPolicy::new()
            .with_load_control(2, None)
            .propose(&workers, &ctx)
            .expect("two candidates must produce a proposal");

        assert_eq!(proposal.primary.id, idle.id);
        assert_eq!(
            proposal.backup.expect("P2 keeps its other sample").id,
            busy.id
        );
    }

    #[test]
    fn proposal_primary_respects_the_queue_gate_tier() {
        let model = ModelId("model".into());
        let queued_a = worker("queued_a");
        let queued_b = worker("queued_b");
        let unqueued = worker("unqueued");
        let workers = vec![
            Arc::clone(&queued_a),
            Arc::clone(&queued_b),
            Arc::clone(&unqueued),
        ];
        let loads = snapshot(&[(&queued_a, 9), (&queued_b, 5), (&unqueued, 3)]);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&loads);
        let policy = PowerOfTwoChoicesPolicy::new().with_load_control(2, Some(4));

        for _ in 0..64 {
            let proposal = policy
                .propose(&workers, &ctx)
                .expect("an unqueued worker exists");
            assert_eq!(proposal.primary.id, unqueued.id);
        }
    }
}
