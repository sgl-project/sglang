// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Sticky-session routing policy.
//!
//! Pins a request's routing key — read from a configurable header into
//! [`SelectionContext::routing_key`] by the chat handler — to a consistent
//! worker via an in-memory map, so stateful sessions land on the same
//! backend. Unlike consistent hashing, this policy never redistributes
//! existing keys when a worker is *added*; a key is only remapped when its
//! assigned worker leaves the healthy candidate set.
//!
//! # Behavior
//! - **No routing key** → delegate to the configured `fallback` policy (no
//!   pinning). This lets clients that don't send the header still be served.
//! - **Known key, worker healthy** → return the pinned worker.
//! - **New key, or pinned worker unhealthy** → pick a worker via `fallback`
//!   and record the assignment.
//!
//! Worker identity is the worker ID supplied by discovery.
//!
//! # Eviction
//! A background sweeper (shared engine with the active-load janitor, see
//! [`crate::state::load_monitor::router_inflight_load::spawn_sweeper`]) removes assignments idle longer
//! than `idle`, bounding the map against unbounded routing-key cardinality.
//! The sweeper is spawned only when constructed inside a Tokio runtime;
//! unit tests use [`StickyPolicy::with_clock`] and drive eviction
//! deterministically via a `MockClock` + direct `sweep_expired`.
//!
//! # HA
//! This map is per-router-instance state, so it is NOT consistent across
//! multiple router replicas or across a failover. HA sticky routing needs
//! a stateless deterministic scheme (rendezvous / consistent hashing) and
//! is intentionally out of scope here.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::policies::{Policy, SelectionContext};
use crate::server::metrics::{MetricsRegistry, StickyOutcome};
use crate::state::load_monitor::router_inflight_load::JanitorHandle;
use crate::state::AffinityStore;
use crate::workers::Worker;

/// Sticky-session policy. See the module docs for behavior and limitations.
pub struct StickyPolicy {
    store: Arc<AffinityStore>,
    /// Selector for keyless requests and for the initial pin of a new key.
    fallback: Arc<dyn Policy>,
    /// Set once via `Policy::attach_metrics`; recording is a no-op until then.
    metrics: OnceLock<Arc<MetricsRegistry>>,
    /// Background idle-eviction sweeper; `None` outside a Tokio runtime.
    _janitor: Option<JanitorHandle>,
}

impl StickyPolicy {
    pub fn new(idle: Duration, eviction_interval: Duration, fallback: Arc<dyn Policy>) -> Self {
        let store = AffinityStore::new(idle);
        let _janitor = store.spawn_sweeper(eviction_interval);
        Self {
            store,
            fallback,
            metrics: OnceLock::new(),
            _janitor,
        }
    }

    /// Test constructor: injectable clock, no background sweeper.
    #[cfg(test)]
    fn with_clock(
        idle: Duration,
        fallback: Arc<dyn Policy>,
        clock: Arc<dyn crate::state::load_monitor::router_inflight_load::Clock>,
    ) -> Self {
        Self {
            store: AffinityStore::with_clock(idle, clock),
            fallback,
            metrics: OnceLock::new(),
            _janitor: None,
        }
    }

    #[cfg(test)]
    fn sweep_expired(&self) -> usize {
        self.store.sweep_expired()
    }

    #[cfg(test)]
    fn assignment_count(&self) -> usize {
        self.store.len()
    }

    fn record(&self, outcome: StickyOutcome) {
        if let Some(m) = self.metrics.get() {
            m.record_sticky(outcome);
        }
    }
}

impl Policy for StickyPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let Some(key) = ctx.routing_key().filter(|k| !k.is_empty()) else {
            self.record(StickyOutcome::NoRoutingKey);
            return self.fallback.select(workers, ctx);
        };
        if let Some(worker) = self.store.bound(key, workers) {
            self.record(StickyOutcome::Hit);
            return Some(Arc::clone(worker));
        }
        // Vacant key, or the pinned worker dropped out: (re)pin the fallback's choice.
        let remap = self.store.contains(key);
        let chosen = self.fallback.select(workers, ctx)?;
        let chosen = Arc::clone(self.store.bind(key.to_string(), &chosen, workers));
        self.record(if remap {
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
            .field("assignments", &self.store.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use std::time::Instant;

    #[test]
    fn sticky_propagates_fallback_load_snapshot_capability() {
        let policy = StickyPolicy::new(
            Duration::from_secs(60),
            Duration::from_secs(10),
            Arc::new(crate::policies::power_of_two::PowerOfTwoChoicesPolicy::new()),
        );
        assert!(policy.needs_load_snapshot());
        assert!(!policy.needs_dispatch_timestamps());

        let load_policy = StickyPolicy::new(
            Duration::from_secs(60),
            Duration::from_secs(10),
            Arc::new(crate::policies::load_based::LoadBasedPolicy::new()),
        );
        assert!(load_policy.needs_dispatch_timestamps());
    }
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

    fn fallback() -> Arc<dyn Policy> {
        Arc::new(RoundRobinPolicy::new())
    }

    fn policy(idle_secs: u64) -> StickyPolicy {
        let clock = Arc::new(
            crate::state::load_monitor::router_inflight_load::MockClock::new(Instant::now()),
        );
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
        let clock = Arc::new(
            crate::state::load_monitor::router_inflight_load::MockClock::new(Instant::now()),
        );
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
        let clock = Arc::new(
            crate::state::load_monitor::router_inflight_load::MockClock::new(Instant::now()),
        );
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
