// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-local affinity assignments. Sticky routing pins URLs; session-aware
//! routing proposes worker IDs and commits new assignments after admission.

use crate::config::{AffinityConfig, SessionAffinityMode};
use crate::discovery::WorkerId;
use crate::policies::admission::compare_prefill_pressure;
use crate::policies::balancing::PowerOfTwoChoicesPolicy;
use crate::policies::{GuardHints, Policy, ProposalKind, SelectionContext, SelectionProposal};
use crate::server::metrics::{MetricsRegistry, StickyOutcome};
use crate::workers::request_tracker::{spawn_sweeper, Clock, JanitorHandle, SystemTimeClock};
use crate::workers::Worker;
use dashmap::DashMap;
use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Assignment<T> {
    target: T,
    last_seen: Instant,
}

#[derive(Debug)]
struct AffinityAssignments<T> {
    assignments: DashMap<String, Assignment<T>>,
    clock: Arc<dyn Clock>,
    idle: Duration,
}

impl<T: Send + Sync + 'static> AffinityAssignments<T> {
    fn new(idle: Duration, clock: Arc<dyn Clock>) -> Self {
        Self {
            assignments: DashMap::new(),
            clock,
            idle,
        }
    }

    fn sweep_expired(&self) -> usize {
        let now = self.clock.now();
        let mut removed = 0;
        self.assignments.retain(|_, assignment| {
            let keep = now.saturating_duration_since(assignment.last_seen) <= self.idle;
            removed += usize::from(!keep);
            keep
        });
        removed
    }

    fn spawn_eviction(
        self: &Arc<Self>,
        interval: Duration,
        label: &'static str,
    ) -> Option<JanitorHandle> {
        if tokio::runtime::Handle::try_current().is_err() {
            tracing::debug!(label, "affinity eviction disabled outside a Tokio runtime");
            return None;
        }
        let state = Arc::clone(self);
        Some(spawn_sweeper(
            move || state.sweep_expired(),
            interval,
            label,
        ))
    }
}

/// Sticky-session policy. See the module docs for behavior and limitations.
pub struct StickyPolicy {
    state: Arc<AffinityAssignments<String>>,
    metrics: OnceLock<Arc<MetricsRegistry>>,
    /// Selector for keyless requests and for the initial pin of a new key.
    fallback: Arc<dyn Policy>,
    /// Background idle-eviction sweeper. `None` when constructed outside a
    /// Tokio runtime (unit tests). Dropping it cancels the task, so the
    /// sweeper lives exactly as long as the policy.
    _janitor: Option<JanitorHandle>,
}

impl StickyPolicy {
    /// Production constructor: monotonic `SystemTimeClock`, with a
    /// background eviction sweeper spawned on `eviction_interval` cadence
    /// (only if called inside a Tokio runtime — the factory runs inside
    /// `main`'s runtime).
    pub fn new(idle: Duration, eviction_interval: Duration, fallback: Arc<dyn Policy>) -> Self {
        let state = Arc::new(AffinityAssignments::new(idle, Arc::new(SystemTimeClock)));
        let janitor = state.spawn_eviction(eviction_interval, "sticky-eviction");
        Self {
            state,
            metrics: OnceLock::new(),
            fallback,
            _janitor: janitor,
        }
    }

    #[cfg(test)]
    fn with_clock(idle: Duration, fallback: Arc<dyn Policy>, clock: Arc<dyn Clock>) -> Self {
        Self {
            state: Arc::new(AffinityAssignments::new(idle, clock)),
            metrics: OnceLock::new(),
            fallback,
            _janitor: None,
        }
    }

    fn record(&self, outcome: StickyOutcome) {
        if let Some(metrics) = self.metrics.get() {
            metrics.record_sticky(outcome);
        }
    }

    #[cfg(test)]
    fn sweep_expired(&self) -> usize {
        self.state.sweep_expired()
    }

    #[cfg(test)]
    fn assignment_count(&self) -> usize {
        self.state.assignments.len()
    }
}

impl Policy for StickyPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let Some(key) = ctx.routing_key().filter(|k| !k.is_empty()) else {
            self.record(StickyOutcome::NoRoutingKey);
            return self.fallback.select(workers, ctx);
        };

        // Fast path: an existing pin whose worker is still in the healthy set.
        let mut existing = false;
        if let Some(mut entry) = self.state.assignments.get_mut(key) {
            existing = true;
            if let Some(worker) = workers.iter().find(|w| w.url == entry.target).cloned() {
                entry.last_seen = self.state.clock.now();
                drop(entry); // release the shard lock before recording
                self.record(StickyOutcome::Hit);
                return Some(worker);
            }
            // Pinned worker is no longer healthy — fall through to reassign.
            drop(entry);
        }

        // Vacant key, or the pinned worker dropped out: (re)assign via the
        // fallback. The read-miss above and this insert are intentionally NOT
        // atomic — the shard lock is released before `fallback.select` (which
        // may do real work, e.g. `load_based`) so it is never held across an
        // unrelated computation. Two requests racing the *same* fresh key may
        // therefore both assign (last-writer-wins in the map; both may record
        // `Assigned`). The scatter is transient and self-heals: the next
        // request for that key hits the surviving pin.
        let chosen = self.fallback.select(workers, ctx)?;
        self.state.assignments.insert(
            key.to_string(),
            Assignment {
                target: chosen.url.clone(),
                last_seen: self.state.clock.now(),
            },
        );
        self.record(if existing {
            StickyOutcome::Remap
        } else {
            StickyOutcome::Assigned
        });
        Some(chosen)
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }

    fn needs_load_snapshot(&self) -> bool {
        self.fallback.needs_load_snapshot()
    }

    fn needs_dispatch_timestamps(&self) -> bool {
        self.fallback.needs_dispatch_timestamps()
    }
}

impl std::fmt::Debug for StickyPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StickyPolicy")
            .field("fallback", &self.fallback)
            .field("idle", &self.state.idle)
            .field("assignments", &self.state.assignments.len())
            .finish_non_exhaustive()
    }
}

pub struct SessionAwarePolicy {
    state: Arc<AffinityAssignments<WorkerId>>,
    config: AffinityConfig,
    _janitor: Option<JanitorHandle>,
}

impl SessionAwarePolicy {
    pub fn new(config: AffinityConfig) -> Self {
        let state = Arc::new(AffinityAssignments::new(
            Duration::from_secs(config.session_idle_secs),
            Arc::new(SystemTimeClock),
        ));
        let janitor = state.spawn_eviction(
            Duration::from_secs(config.session_eviction_interval_secs),
            "session-affinity-eviction",
        );
        Self {
            state,
            config,
            _janitor: janitor,
        }
    }

    #[cfg(test)]
    fn with_clock(config: AffinityConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            state: Arc::new(AffinityAssignments::new(
                Duration::from_secs(config.session_idle_secs),
                clock,
            )),
            config,
            _janitor: None,
        }
    }

    #[cfg(test)]
    fn sweep_expired(&self) -> usize {
        self.state.sweep_expired()
    }

    #[cfg(test)]
    fn assignment_count(&self) -> usize {
        self.state.assignments.len()
    }

    fn assignment_key(&self, session_id: &str, ctx: &SelectionContext<'_>) -> String {
        match self.config.session_affinity_mode {
            SessionAffinityMode::Bucket => {
                format!("{}\0{}", ctx.candidate_range_id(), session_id)
            }
            SessionAffinityMode::GlobalRebind | SessionAffinityMode::GlobalPreserve => {
                session_id.to_string()
            }
        }
    }

    fn initial_proposal(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        PowerOfTwoChoicesPolicy::new().propose(workers, ctx)
    }

    fn affinity_proposal(
        &self,
        primary: Arc<Worker>,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
        session_id: &str,
    ) -> SelectionProposal {
        let backup = affinity_backup(
            workers,
            &primary,
            session_id,
            ctx.candidate_range_id(),
            self.config.stable_pair,
            ctx,
        );
        let proposal = match backup {
            Some(backup) => SelectionProposal::with_backup(primary, backup),
            None => SelectionProposal::primary(primary),
        };
        proposal
            .with_kind(ProposalKind::SessionAffinity)
            .with_guard_hints(GuardHints {
                enable_pressure_guard: self.config.pressure_guard
                    && self.config.mode == crate::config::AffinityMode::Soft,
                pressure_abs_threshold_tokens: self.config.pressure_abs_threshold_tokens,
                pressure_abs_threshold_ms: self.config.pressure_abs_threshold_ms,
                pressure_rel_threshold: self.config.pressure_rel_threshold,
            })
    }
}

impl Policy for SessionAwarePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let proposal = self.propose(workers, ctx)?;
        let proposal_kind = proposal.kind;
        let selected = proposal.primary;
        self.commit_prefill_selection(ctx, proposal_kind, &selected);
        Some(selected)
    }

    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        if !ctx.affinity_lookup_enabled() {
            return PowerOfTwoChoicesPolicy::new().propose(workers, ctx);
        }
        let Some(session_id) = ctx.session_id().filter(|id| !id.is_empty()) else {
            return self.initial_proposal(workers, ctx);
        };

        let assignment_key = self.assignment_key(session_id, ctx);
        let assigned = self
            .state
            .assignments
            .get_mut(&assignment_key)
            .map(|mut assignment| {
                assignment.last_seen = self.state.clock.now();
                assignment.target.clone()
            });
        if let Some(assigned) = assigned {
            if let Some(primary) = workers.iter().find(|worker| worker.id == assigned).cloned() {
                return Some(self.affinity_proposal(primary, workers, ctx, session_id));
            }
        }

        // Persist new assignments only after selecting the final prefill worker.
        self.initial_proposal(workers, ctx)
    }

    fn commit_prefill_selection(
        &self,
        ctx: &SelectionContext<'_>,
        proposal_kind: ProposalKind,
        selected: &Arc<Worker>,
    ) {
        if proposal_kind != ProposalKind::PowerOfTwo || !ctx.affinity_assignment_enabled() {
            return;
        }
        let Some(session_id) = ctx.session_id().filter(|id| !id.is_empty()) else {
            return;
        };
        self.state.assignments.insert(
            self.assignment_key(session_id, ctx),
            Assignment {
                target: selected.id.clone(),
                last_seen: self.state.clock.now(),
            },
        );
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }

    fn is_bucket_affinity_policy(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for SessionAwarePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionAwarePolicy")
            .field("config", &self.config)
            .field("assignments", &self.state.assignments.len())
            .finish_non_exhaustive()
    }
}

pub(crate) fn affinity_backup(
    workers: &[Arc<Worker>],
    primary: &Arc<Worker>,
    affinity_key: &str,
    candidate_range_id: &str,
    stable_pair: bool,
    ctx: &SelectionContext<'_>,
) -> Option<Arc<Worker>> {
    if stable_pair {
        return stable_backup(workers, primary, affinity_key, candidate_range_id);
    }
    sampled_backup_excluding(workers, primary, ctx)
}

fn sampled_backup_excluding(
    workers: &[Arc<Worker>],
    primary: &Arc<Worker>,
    ctx: &SelectionContext<'_>,
) -> Option<Arc<Worker>> {
    let mut rng = rand::thread_rng();
    let first = sample_index_excluding(workers, &primary.id, None, &mut rng)?;
    let Some(second) = sample_index_excluding(workers, &primary.id, Some(first), &mut rng) else {
        return Some(Arc::clone(&workers[first]));
    };
    let left = &workers[first];
    let right = &workers[second];
    if compare_prefill_pressure(left, right, ctx.load_snapshot()).is_gt() {
        Some(Arc::clone(right))
    } else {
        Some(Arc::clone(left))
    }
}

fn sample_index_excluding(
    workers: &[Arc<Worker>],
    primary_id: &WorkerId,
    other_index: Option<usize>,
    rng: &mut impl Rng,
) -> Option<usize> {
    if workers.is_empty() {
        return None;
    }
    for _ in 0..32 {
        let index = rng.gen_range(0..workers.len());
        if Some(index) != other_index && workers[index].id != *primary_id {
            return Some(index);
        }
    }
    workers.iter().enumerate().find_map(|(index, worker)| {
        (Some(index) != other_index && worker.id != *primary_id).then_some(index)
    })
}

fn stable_backup(
    workers: &[Arc<Worker>],
    primary: &Arc<Worker>,
    session_id: &str,
    candidate_range_id: &str,
) -> Option<Arc<Worker>> {
    let mut others: Vec<Arc<Worker>> = workers
        .iter()
        .filter(|worker| worker.id != primary.id)
        .cloned()
        .collect();
    others.sort_by(|left, right| left.id.0.cmp(&right.id.0));
    if others.is_empty() {
        return None;
    }
    let mut hasher = DefaultHasher::new();
    session_id.hash(&mut hasher);
    candidate_range_id.hash(&mut hasher);
    Some(others[(hasher.finish() as usize) % others.len()].clone())
}

#[cfg(test)]
mod sticky_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

    #[test]
    fn sticky_propagates_fallback_load_snapshot_capability() {
        let policy = StickyPolicy::new(
            Duration::from_secs(60),
            Duration::from_secs(10),
            Arc::new(PowerOfTwoChoicesPolicy::new()),
        );
        assert!(policy.needs_load_snapshot());
        assert!(!policy.needs_dispatch_timestamps());

        let load_policy = StickyPolicy::new(
            Duration::from_secs(60),
            Duration::from_secs(10),
            Arc::new(crate::policies::balancing::LoadBasedPolicy::new()),
        );
        assert!(load_policy.needs_dispatch_timestamps());
    }
    use crate::policies::balancing::RoundRobinPolicy;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    fn fallback() -> Arc<dyn Policy> {
        Arc::new(RoundRobinPolicy::new())
    }

    fn policy(idle_secs: u64) -> StickyPolicy {
        let clock = Arc::new(crate::workers::request_tracker::MockClock::new(
            Instant::now(),
        ));
        StickyPolicy::with_clock(Duration::from_secs(idle_secs), fallback(), clock)
    }

    #[test]
    fn empty_workers_returns_none() {
        let model = ModelId("tiny".into());
        let p = policy(600);
        let ctx = SelectionContext::with_routing_key(&model, None, Some("u1"));
        assert!(p.select(&[], &ctx).is_none());
    }

    #[test]
    fn keyless_request_delegates_to_fallback_without_pinning() {
        let model = ModelId("tiny".into());
        let p = policy(600);
        let workers = vec![worker("w0"), worker("w1")];
        // No routing key on the context.
        let ctx = SelectionContext::new(&model, None);
        assert!(p.select(&workers, &ctx).is_some());
        assert_eq!(p.assignment_count(), 0, "keyless request must not pin");
    }

    #[test]
    fn same_key_sticks_to_same_worker() {
        let model = ModelId("tiny".into());
        let p = policy(600);
        let workers = vec![worker("w0"), worker("w1")];
        let ctx = SelectionContext::with_routing_key(&model, None, Some("u1"));

        let first = p.select(&workers, &ctx).unwrap();
        // Many repeats must all return the same worker (the hit path never
        // consults the fallback, so this is independent of round-robin).
        for _ in 0..10 {
            let again = p.select(&workers, &ctx).unwrap();
            assert_eq!(again.id, first.id);
        }
        assert_eq!(p.assignment_count(), 1);
    }

    #[test]
    fn distinct_keys_get_independent_pins() {
        let model = ModelId("tiny".into());
        let p = policy(600);
        let workers = vec![worker("w0"), worker("w1")];

        let ctx_a = SelectionContext::with_routing_key(&model, None, Some("a"));
        let ctx_b = SelectionContext::with_routing_key(&model, None, Some("b"));
        let a = p.select(&workers, &ctx_a).unwrap();
        let b = p.select(&workers, &ctx_b).unwrap();
        // Two keys are tracked independently (two map entries), and the
        // round-robin fallback hands the two fresh keys distinct workers.
        assert_ne!(a.id, b.id);
        assert_eq!(p.assignment_count(), 2);
        // The core property: each key independently stays on its own pin.
        for _ in 0..5 {
            assert_eq!(p.select(&workers, &ctx_a).unwrap().id, a.id);
            assert_eq!(p.select(&workers, &ctx_b).unwrap().id, b.id);
        }
    }

    #[test]
    fn adding_a_worker_does_not_redistribute_existing_key() {
        let model = ModelId("tiny".into());
        let p = policy(600);
        let w0 = worker("w0");
        let w1 = worker("w1");
        let ctx = SelectionContext::with_routing_key(&model, None, Some("u1"));

        let pinned = p.select(&[Arc::clone(&w0), Arc::clone(&w1)], &ctx).unwrap();
        // Scale up: a third worker joins. The existing key must stay pinned.
        let w2 = worker("w2");
        let after = p
            .select(&[Arc::clone(&w0), Arc::clone(&w1), w2], &ctx)
            .unwrap();
        assert_eq!(after.id, pinned.id, "true-sticky: no redistribution on add");
    }

    #[test]
    fn remaps_when_pinned_worker_becomes_unhealthy() {
        let model = ModelId("tiny".into());
        let p = policy(600);
        let w0 = worker("w0");
        let w1 = worker("w1");
        let ctx = SelectionContext::with_routing_key(&model, None, Some("u1"));

        let pinned = p.select(&[Arc::clone(&w0), Arc::clone(&w1)], &ctx).unwrap();
        // Drop the pinned worker from the healthy set; only the other remains.
        let survivor = if pinned.id == w0.id {
            Arc::clone(&w1)
        } else {
            Arc::clone(&w0)
        };
        let remapped = p.select(&[Arc::clone(&survivor)], &ctx).unwrap();
        assert_eq!(remapped.id, survivor.id);
        // The new pin sticks across subsequent calls.
        let again = p.select(&[Arc::clone(&survivor)], &ctx).unwrap();
        assert_eq!(again.id, survivor.id);
    }

    #[test]
    fn sweep_evicts_idle_entries_keeps_fresh() {
        let model = ModelId("tiny".into());
        let clock = Arc::new(crate::workers::request_tracker::MockClock::new(
            Instant::now(),
        ));
        let p = StickyPolicy::with_clock(Duration::from_secs(10), fallback(), clock.clone());
        let workers = vec![worker("w0"), worker("w1")];

        // Pin key "old" at t0.
        let ctx_old = SelectionContext::with_routing_key(&model, None, Some("old"));
        p.select(&workers, &ctx_old).unwrap();

        // Advance 6s, pin key "new" at t6.
        clock.advance(Duration::from_secs(6));
        let ctx_new = SelectionContext::with_routing_key(&model, None, Some("new"));
        p.select(&workers, &ctx_new).unwrap();
        assert_eq!(p.assignment_count(), 2);

        // Advance to t11: "old" has been idle 11s (> 10), "new" idle 5s.
        clock.advance(Duration::from_secs(5));
        assert_eq!(p.sweep_expired(), 1);
        assert_eq!(p.assignment_count(), 1);

        // "new" survived and is still pinned.
        assert!(p.select(&workers, &ctx_new).is_some());
        assert_eq!(p.assignment_count(), 1);
    }

    #[test]
    fn hit_refreshes_last_seen_so_active_key_is_not_evicted() {
        let model = ModelId("tiny".into());
        let clock = Arc::new(crate::workers::request_tracker::MockClock::new(
            Instant::now(),
        ));
        let p = StickyPolicy::with_clock(Duration::from_secs(10), fallback(), clock.clone());
        let workers = vec![worker("w0")];
        let ctx = SelectionContext::with_routing_key(&model, None, Some("u1"));

        p.select(&workers, &ctx).unwrap();
        // Keep referencing the key just under the idle window each step.
        for _ in 0..5 {
            clock.advance(Duration::from_secs(8));
            p.select(&workers, &ctx).unwrap(); // hit → refreshes last_seen
            assert_eq!(
                p.sweep_expired(),
                0,
                "an actively-used key must not be evicted"
            );
        }
        assert_eq!(p.assignment_count(), 1);
    }

    /// Exercises the production path: `new` (not `with_clock`) spawns the
    /// real background sweeper because we are inside a Tokio runtime. Uses
    /// sub-second idle + interval so the sweep fires within the test's
    /// wall-time, proving `StickyPolicy::new` correctly wires `sweep_expired`
    /// into the runtime sweeper.
    #[tokio::test]
    async fn background_sweeper_evicts_idle_entry_in_runtime() {
        let model = ModelId("tiny".into());
        let p = StickyPolicy::new(
            Duration::from_millis(20),
            Duration::from_millis(10),
            fallback(),
        );
        let workers = vec![worker("w0")];
        let ctx = SelectionContext::with_routing_key(&model, None, Some("u1"));
        p.select(&workers, &ctx).unwrap();
        assert_eq!(p.assignment_count(), 1);

        // Idle window is 20ms; wait well past it plus several sweep ticks.
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(
            p.assignment_count(),
            0,
            "background sweeper should have evicted the idle assignment"
        );
    }

    /// Many concurrent first-touch requests for the SAME fresh key converge:
    /// the map ends with exactly one pin and every subsequent select agrees
    /// on it (the documented self-heal after the benign assign race).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_first_touch_converges_to_one_pin() {
        let p = Arc::new(StickyPolicy::new(
            Duration::from_secs(3600),
            Duration::from_secs(3600),
            fallback(),
        ));
        let workers = Arc::new(vec![worker("w0"), worker("w1"), worker("w2")]);
        let model = ModelId("tiny".into());

        let mut handles = Vec::new();
        for _ in 0..32 {
            let p = Arc::clone(&p);
            let workers = Arc::clone(&workers);
            let model = model.clone();
            handles.push(tokio::spawn(async move {
                let ctx = SelectionContext::with_routing_key(&model, None, Some("race"));
                p.select(&workers[..], &ctx).map(|w| w.id.clone())
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }

        assert_eq!(
            p.assignment_count(),
            1,
            "concurrent first-touch must converge to a single pin"
        );
        let ctx = SelectionContext::with_routing_key(&model, None, Some("race"));
        let pinned = p.select(&workers[..], &ctx).unwrap().id.clone();
        for _ in 0..10 {
            assert_eq!(p.select(&workers[..], &ctx).unwrap().id, pinned);
        }
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerMode, WorkerSpec};
    use crate::workers::request_tracker::MockClock;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

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
    fn idle_session_assignments_are_evicted() {
        let clock = Arc::new(MockClock::new(Instant::now()));
        let policy = SessionAwarePolicy::with_clock(
            AffinityConfig {
                session_idle_secs: 10,
                ..Default::default()
            },
            clock.clone(),
        );
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None).with_session_id(Some("session-a"));
        let proposal = policy.propose(&[worker("w")], &ctx).unwrap();
        policy.commit_prefill_selection(&ctx, proposal.kind, &proposal.primary);
        assert_eq!(policy.assignment_count(), 1);

        clock.advance(Duration::from_secs(11));
        assert_eq!(policy.sweep_expired(), 1);
        assert_eq!(policy.assignment_count(), 0);
    }

    #[test]
    fn sampled_backup_excludes_primary_without_materializing_the_remaining_fleet() {
        let primary = worker("primary");
        let busy = worker("busy");
        let idle = worker("idle");
        busy.active_requests.store(8, Ordering::Relaxed);
        idle.active_requests.store(1, Ordering::Relaxed);
        let workers = vec![Arc::clone(&primary), busy, Arc::clone(&idle)];
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None);

        let backup = sampled_backup_excluding(&workers, &primary, &ctx)
            .expect("two non-primary workers are available");
        assert_eq!(backup.id, idle.id);
    }
}

#[cfg(test)]
mod proposal_tests {
    use crate::config::AffinityConfig;
    use crate::config::SessionAffinityMode;
    use crate::policies::admission::resolve_prefill;
    use crate::policies::admission::CandidateRange;
    use crate::policies::affinity::SessionAwarePolicy;
    use crate::policies::test_support::snapshot;
    use crate::policies::test_support::worker;
    use crate::policies::test_support::TestEngineLoad;
    use crate::policies::*;
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
        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &proposal,
            32,
            &loads,
            None,
        )
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
}
