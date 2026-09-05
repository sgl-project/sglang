// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Session-aware prefill policy. Admission does not rewrite session assignments.

use crate::config::{AffinityConfig, SessionAffinityMode};
use crate::discovery::WorkerId;
use crate::policies::active_load::{spawn_sweeper, Clock, JanitorHandle, SystemTimeClock};
use crate::policies::admission::compare_prefill_pressure;
use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
use crate::policies::{Policy, ProposalKind, SelectionContext, SelectionProposal};
use crate::workers::Worker;
use dashmap::DashMap;
use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Assignment {
    worker_id: WorkerId,
    last_seen: Instant,
}

#[derive(Debug)]
struct SessionState {
    assignments: DashMap<String, Assignment>,
    clock: Arc<dyn Clock>,
    idle: Duration,
}

impl SessionState {
    fn sweep_expired(&self) -> usize {
        let now = self.clock.now();
        let mut removed = 0;
        self.assignments.retain(|_, assignment| {
            let keep = now.saturating_duration_since(assignment.last_seen) <= self.idle;
            if !keep {
                removed += 1;
            }
            keep
        });
        removed
    }
}

pub struct SessionAwarePolicy {
    state: Arc<SessionState>,
    config: AffinityConfig,
    _janitor: Option<JanitorHandle>,
}

impl SessionAwarePolicy {
    pub fn new(config: AffinityConfig) -> Self {
        let state = Arc::new(SessionState {
            assignments: DashMap::new(),
            clock: Arc::new(SystemTimeClock),
            idle: Duration::from_secs(config.session_idle_secs),
        });
        let _janitor = if tokio::runtime::Handle::try_current().is_ok() {
            let swept = Arc::clone(&state);
            Some(spawn_sweeper(
                move || swept.sweep_expired(),
                Duration::from_secs(config.session_eviction_interval_secs),
                "session-affinity-eviction",
            ))
        } else {
            tracing::debug!(
                "SessionAwarePolicy constructed outside a Tokio runtime; idle eviction is disabled"
            );
            None
        };
        Self {
            state,
            config,
            _janitor,
        }
    }

    #[cfg(test)]
    fn with_clock(config: AffinityConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            state: Arc::new(SessionState {
                assignments: DashMap::new(),
                clock,
                idle: Duration::from_secs(config.session_idle_secs),
            }),
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
        proposal.with_kind(ProposalKind::SessionAffinity)
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
                assignment.worker_id.clone()
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
                worker_id: selected.id.clone(),
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
mod lifecycle_tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerMode, WorkerSpec};
    use crate::policies::active_load::MockClock;
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
