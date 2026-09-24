// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Session and routing-key assignments with idle expiry. The store keeps
//! bindings; deciding whether to reuse, create or replace one is the policy's.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use super::load_monitor::router_inflight_load::{
    spawn_sweeper, Clock, JanitorHandle, SystemTimeClock,
};
use crate::discovery::WorkerId;
use crate::workers::Worker;

#[derive(Debug)]
struct Assignment {
    engine: WorkerId,
    last_seen: Instant,
}

#[derive(Debug)]
pub struct AffinityStore {
    assignments: DashMap<String, Assignment>,
    clock: Arc<dyn Clock>,
    idle: Duration,
}

impl AffinityStore {
    pub fn new(idle: Duration) -> Arc<Self> {
        Self::with_clock(idle, Arc::new(SystemTimeClock))
    }

    pub fn with_clock(idle: Duration, clock: Arc<dyn Clock>) -> Arc<Self> {
        Arc::new(Self {
            assignments: DashMap::new(),
            clock,
            idle,
        })
    }

    /// The bound engine when it is still among `engines`; refreshes the binding.
    pub fn bound<'e>(&self, key: &str, engines: &'e [Arc<Worker>]) -> Option<&'e Arc<Worker>> {
        let mut assignment = self.assignments.get_mut(key)?;
        let engine = engines
            .iter()
            .find(|engine| engine.id == assignment.engine)?;
        assignment.last_seen = self.clock.now();
        Some(engine)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.assignments.contains_key(key)
    }

    /// The bound engine id, whether or not it is still a candidate.
    pub fn binding(&self, key: &str) -> Option<WorkerId> {
        self.assignments.get(key).map(|a| a.engine.clone())
    }

    /// Binds `engine`, unless a concurrent pick already bound another engine
    /// that is still in `engines`; that binding wins so racing first touches
    /// converge on one engine.
    pub fn bind<'e>(
        &self,
        key: String,
        engine: &'e Arc<Worker>,
        engines: &'e [Arc<Worker>],
    ) -> &'e Arc<Worker> {
        let now = self.clock.now();
        let mut assignment = self.assignments.entry(key).or_insert_with(|| Assignment {
            engine: engine.id.clone(),
            last_seen: now,
        });
        match engines.iter().find(|bound| bound.id == assignment.engine) {
            Some(bound) => {
                assignment.last_seen = now;
                bound
            }
            None => {
                assignment.engine = engine.id.clone();
                assignment.last_seen = now;
                engine
            }
        }
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    pub fn sweep_expired(&self) -> usize {
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

    /// Periodic eviction; `None` outside a Tokio runtime.
    pub fn spawn_sweeper(self: &Arc<Self>, interval: Duration) -> Option<JanitorHandle> {
        let store = Arc::clone(self);
        tokio::runtime::Handle::try_current()
            .ok()
            .map(|_| spawn_sweeper(move || store.sweep_expired(), interval, "affinity-eviction"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerMode, WorkerSpec};
    use crate::state::load_monitor::router_inflight_load::MockClock;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    #[test]
    fn a_live_binding_wins_over_a_later_bind_and_a_dead_one_is_replaced() {
        let store = AffinityStore::new(Duration::from_secs(60));
        let (a, b) = (worker("a"), worker("b"));
        let fleet = [a.clone(), b.clone()];
        assert_eq!(store.bind("k".into(), &a, &fleet).id.0, "a");
        assert_eq!(store.bind("k".into(), &b, &fleet).id.0, "a");
        assert_eq!(store.bind("k".into(), &b, &fleet[1..]).id.0, "b");
        assert_eq!(store.bound("k", &fleet).unwrap().id.0, "b");
        assert!(store.bound("k", &fleet[..1]).is_none());
    }

    #[test]
    fn idle_bindings_are_swept_and_hits_refresh() {
        let clock = Arc::new(MockClock::new(Instant::now()));
        let store = AffinityStore::with_clock(Duration::from_secs(10), clock.clone());
        let fleet = [worker("a")];
        store.bind("hot".into(), &fleet[0], &fleet);
        store.bind("cold".into(), &fleet[0], &fleet);
        clock.advance(Duration::from_secs(8));
        store.bound("hot", &fleet);
        clock.advance(Duration::from_secs(8));
        assert_eq!(store.sweep_expired(), 1);
        assert!(store.contains("hot") && !store.contains("cold"));
    }

    #[test]
    fn concurrent_bindings_do_not_count_as_evictions() {
        use std::sync::Barrier;

        let clock = Arc::new(MockClock::new(Instant::now()));
        let store = AffinityStore::with_clock(Duration::from_secs(60), clock);
        let engine = worker("a");
        let start = Arc::new(Barrier::new(5));
        std::thread::scope(|scope| {
            for writer in 0..4 {
                let store = Arc::clone(&store);
                let engine = Arc::clone(&engine);
                let start = Arc::clone(&start);
                scope.spawn(move || {
                    start.wait();
                    for key in 0..5000 {
                        store.bind(
                            format!("{writer}-{key}"),
                            &engine,
                            std::slice::from_ref(&engine),
                        );
                    }
                });
            }
            start.wait();
            for _ in 0..1000 {
                // The clock never advances: every binding must survive,
                // even when requests insert new keys during a sweep.
                assert_eq!(store.sweep_expired(), 0);
            }
        });
        assert_eq!(store.len(), 20_000);
    }
}
