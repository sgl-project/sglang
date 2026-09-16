// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;

/// Always returns 200 — liveness probe.
pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe — 200 only when the pod can actually serve traffic.
///
/// Requires ALL of:
/// 1. `AppContext::mark_ready()` was called by main (process bootstrap
///    finished — config loaded, tokenizers built, server bound), AND
/// 2. At least one worker is registered. Without this second check,
///    `/readyz` flips green before the first `DiscoveryEvent::Added`
///    has been processed — the Service starts sending traffic to a
///    pod whose registry is empty, and every request returns 503
///    `no_healthy_workers`.
/// 3. Cache-aware peer bootstrap has settled. A replica with an empty KV tree
///    routes cache-blind AND scatters prefixes the warm replicas were keeping
///    consolidated, so it is held out of the Service until it has pulled a
///    snapshot from a sibling — or until every sibling proves it has no state
///    to give — or until `--kv-bootstrap-timeout-ms` gives up. Always true
///    unless BOTH cache-aware routing and `--kv-peer-selector` are configured;
///    that pair is what enables the gate at all.
/// 4. The seed-required gate is open (`--kv-bootstrap-seed-required`, off by
///    default). Condition 3 settles either way once the bootstrap deadline
///    expires — that escape is deliberate and load-bearing — so on its own it
///    cannot distinguish "seeded" from "gave up and is about to route
///    cache-blind". This condition does: it refuses readiness when a sweep
///    proved siblings were present and their tree could not be pulled, which
///    stalls a rolling update with the previous generation still serving
///    instead of replacing it with cache-blind replicas.
///
/// Conditions 3 and 4 latch once satisfied (see `BootstrapTracker::settled`
/// and `mark_seed_gate_passed`): a later scale-up must never drag an
/// already-serving replica back to 503.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    if ctx.is_ready()
        && !ctx.registry.is_empty()
        && ctx.kv_bootstrap_settled()
        && ctx.kv_seed_gate_open()
    {
        // Latch BEFORE returning 200: from here on a sweep started by a
        // late-discovered worker must not be able to un-ready a replica the
        // Service is already routing to.
        ctx.mark_kv_seed_gate_passed();
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn healthz_always_200() {
        let app = crate::server::app::build_router(test_ctx(false, false));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readyz_503_when_not_ready() {
        let app = crate::server::app::build_router(test_ctx(false, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readyz_503_when_ready_but_registry_empty() {
        // Regression: `/readyz` previously returned 200 the moment
        // `mark_ready()` was called, even with an empty worker
        // registry. The Service would route traffic to a pod that
        // could only return 503 no_healthy_workers.
        let app = crate::server::app::build_router(test_ctx(true, false));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "ready=true + empty registry must still be 503"
        );
    }

    #[tokio::test]
    async fn readyz_200_when_ready_and_worker_registered() {
        let app = crate::server::app::build_router(test_ctx(true, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    async fn readyz_status(ctx: Arc<AppContext>) -> StatusCode {
        crate::server::app::build_router(ctx)
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    /// A context whose KV index has peer bootstrap configured, driven to the
    /// state the test needs.
    ///
    /// The rank is driven to a TERMINAL state on purpose, so `settled()` (the
    /// pre-existing condition 3) is satisfied and the seed gate is the ONLY
    /// thing that can still hold `/readyz` down. Without that the 503 these
    /// tests assert would arrive from condition 3 and pass whatever the gate
    /// did — the failure mode these tests exist to rule out.
    fn ctx_with_seed_gate(ready: bool, seed_required: bool, failed: bool) -> Arc<AppContext> {
        use crate::policies::kv_events::bootstrap::SweepOutcome;
        use crate::policies::kv_events::{
            BlockSizeOracle, BootstrapState, BootstrapTracker, KvEventIndex, KvWorkerId,
        };

        let mut ctx = AppContext::stub();
        if ready {
            ctx.mark_ready();
        }
        add_test_worker(&ctx);

        let tracker = Arc::new(BootstrapTracker::new_with_opts(
            std::time::Duration::from_secs(300),
            std::time::Duration::from_secs(120),
            seed_required,
        ));
        let rank = KvWorkerId::new("http://w1:30000".into(), 0);
        tracker.register(std::slice::from_ref(&rank));
        tracker.set(&rank, BootstrapState::Failed);
        assert!(tracker.settled(), "condition 3 must not be what gates here");
        if failed {
            // Nine siblings were there and none gave up its tree — the exact
            // shape a first-wave surge pod hits on a rolling update.
            tracker.record_sweep_result(SweepOutcome::TimedOut, 9);
        }
        ctx.kv_index = KvEventIndex::new_with_bootstrap(
            reqwest::Client::new(),
            BlockSizeOracle::new(),
            tracker,
        )
        .snapshot_source();
        Arc::new(ctx)
    }

    fn add_test_worker(ctx: &AppContext) {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("test-w".into()),
                url: "http://test:30000".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("test".into())],
                bootstrap_port: None,
            })
            .expect("test worker accepted");
    }

    /// The gate's reason for existing: a replica that could not pull the
    /// fleet's tree must stay OUT of the Service, so a rolling update stalls
    /// with the previous generation serving instead of completing with
    /// cache-blind replicas.
    #[tokio::test]
    async fn readyz_503_when_a_required_seed_failed() {
        assert_eq!(
            readyz_status(ctx_with_seed_gate(true, true, true)).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );
    }

    /// Same failed sweep, gate not opted in: today's behaviour is preserved
    /// exactly, so enabling the flag is the only thing that can change a
    /// fleet's rollout semantics.
    #[tokio::test]
    async fn readyz_200_on_the_same_failure_when_the_gate_is_off() {
        assert_eq!(
            readyz_status(ctx_with_seed_gate(true, false, true)).await,
            StatusCode::OK,
        );
    }

    /// Answering 200 once latches the gate: a sweep started by a
    /// late-discovered worker must not pull a serving replica back out of the
    /// Service, which would be fleet-amplifying — an unready replica leaves its
    /// own EndpointSlice, so its siblings lose a bootstrap source too.
    #[tokio::test]
    async fn readyz_stays_200_after_a_later_sweep_fails() {
        use crate::policies::kv_events::bootstrap::SweepOutcome;
        let ctx = ctx_with_seed_gate(true, true, false);
        assert_eq!(readyz_status(Arc::clone(&ctx)).await, StatusCode::OK);

        ctx.kv_index
            .as_ref()
            .expect("index attached")
            .bootstrap()
            .record_sweep_result(SweepOutcome::TimedOut, 9);
        assert_eq!(
            readyz_status(ctx).await,
            StatusCode::OK,
            "an already-serving replica may not be un-readied",
        );
    }

    /// Condition 3 on its own: a rank still mid-bootstrap holds readiness down
    /// whether or not the seed gate is configured.
    #[tokio::test]
    async fn readyz_503_while_a_rank_is_still_bootstrapping() {
        use crate::policies::kv_events::{
            BlockSizeOracle, BootstrapTracker, KvEventIndex, KvWorkerId,
        };

        let mut ctx = AppContext::stub();
        ctx.mark_ready();
        add_test_worker(&ctx);
        let tracker = Arc::new(BootstrapTracker::new(std::time::Duration::from_secs(300)));
        tracker.register(&[KvWorkerId::new("http://w1:30000".into(), 0)]);
        assert!(!tracker.settled());
        ctx.kv_index = KvEventIndex::new_with_bootstrap(
            reqwest::Client::new(),
            BlockSizeOracle::new(),
            tracker,
        )
        .snapshot_source();

        assert_eq!(
            readyz_status(Arc::new(ctx)).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );
    }

    fn test_ctx(ready: bool, with_worker: bool) -> Arc<AppContext> {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = AppContext::stub();
        if ready {
            ctx.mark_ready();
        }
        if with_worker {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId("test-w".into()),
                    url: "http://test:30000".into(),
                    mode: WorkerMode::Plain,
                    model_ids: vec![ModelId("test".into())],
                    bootstrap_port: None,
                })
                .expect("test worker accepted");
        }
        Arc::new(ctx)
    }
}
