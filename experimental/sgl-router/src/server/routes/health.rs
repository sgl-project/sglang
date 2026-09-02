// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::WorkerMode;
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
///    to give — or until `--kv-bootstrap-timeout-ms` gives up.
///    Always true unless BOTH cache-aware-zmq and `--kv-peer-selector` are
///    configured; that pair is what enables the gate at all.
///
/// Condition 3 latches once satisfied (see `BootstrapTracker::settled`): a
/// later scale-up must never drag an already-serving replica back to 503.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    if !ctx.is_ready() || ctx.registry.is_empty() || !ctx.kv_bootstrap_settled() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    // PD-aware readiness: a PD deployment can only serve when it has at
    // least one ROUTABLE prefill worker (mode Prefill with a bootstrap
    // port) AND at least one decode worker. Reporting Ready with an empty
    // pool would let the Service send traffic to a pod that 503s every
    // request (`no_prefill_workers_available` / `no_decode_workers_available`).
    let all = ctx.registry.all();
    let pd = all
        .iter()
        .any(|w| matches!(w.mode(), WorkerMode::Prefill | WorkerMode::Decode));
    if pd {
        // Pairing is per transfer group, so readiness needs at least one
        // group that is COMPLETE: a routable prefill and a decode worker
        // with the same group (ungrouped counts as its own group).
        let complete_group = all
            .iter()
            .filter(|p| p.mode() == WorkerMode::Prefill && p.bootstrap_port().is_some())
            .any(|p| {
                all.iter().any(|d| {
                    d.mode() == WorkerMode::Decode && d.transfer_group() == p.transfer_group()
                })
            });
        if !complete_group {
            return StatusCode::SERVICE_UNAVAILABLE;
        }
    }
    StatusCode::OK
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

    /// A KV index whose bootstrap is deliberately unsettled: one rank
    /// registered, deadline far out.
    fn unsettled_kv_index() -> Arc<crate::policies::kv_events::KvEventIndex> {
        use crate::policies::kv_events::bootstrap::BootstrapTracker;
        use crate::policies::kv_events::{BlockSizeOracle, KvEventIndex, KvWorkerId};
        let tracker = Arc::new(BootstrapTracker::new(std::time::Duration::from_secs(3600)));
        tracker.register(&[KvWorkerId::new("http://w1".into(), 0)]);
        KvEventIndex::new_with_bootstrap(reqwest::Client::new(), BlockSizeOracle::new(), tracker)
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

    /// A replica that is otherwise serviceable is still held out
    /// while its cache-aware tree is cold, so it cannot scatter prefixes the
    /// warm replicas were consolidating.
    #[tokio::test]
    async fn readyz_503_while_kv_bootstrap_pending() {
        let ctx = test_ctx(true, true);
        ctx.attach_kv_index(unsettled_kv_index());
        assert_eq!(
            readyz_status(ctx).await,
            StatusCode::SERVICE_UNAVAILABLE,
            "ready + workers + cold KV tree must still be 503",
        );
    }

    /// Once every rank reaches a terminal state the gate opens — including when
    /// the outcome was Failed, because cold-but-known is a valid outcome.
    #[tokio::test]
    async fn readyz_200_once_kv_bootstrap_settles_even_if_failed() {
        use crate::policies::kv_events::bootstrap::BootstrapState;
        use crate::policies::kv_events::KvWorkerId;

        let ctx = test_ctx(true, true);
        let index = unsettled_kv_index();
        ctx.attach_kv_index(index.clone());
        assert_eq!(
            readyz_status(ctx.clone()).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );

        index.bootstrap().set(
            &KvWorkerId::new("http://w1".into(), 0),
            BootstrapState::Failed,
        );
        assert_eq!(readyz_status(ctx).await, StatusCode::OK);
    }

    /// The latch: a worker discovered after settlement must not drag a serving
    /// replica back to 503. Without this, a routine scale-up would look like an
    /// availability incident.
    #[tokio::test]
    async fn readyz_stays_200_when_a_worker_appears_after_settling() {
        use crate::policies::kv_events::bootstrap::BootstrapState;
        use crate::policies::kv_events::KvWorkerId;

        let ctx = test_ctx(true, true);
        let index = unsettled_kv_index();
        ctx.attach_kv_index(index.clone());
        index.bootstrap().set(
            &KvWorkerId::new("http://w1".into(), 0),
            BootstrapState::Recovered,
        );
        assert_eq!(readyz_status(ctx.clone()).await, StatusCode::OK);

        // Scale-up: a brand-new pending rank shows up.
        index
            .bootstrap()
            .register(&[KvWorkerId::new("http://w2".into(), 0)]);
        assert_eq!(
            readyz_status(ctx).await,
            StatusCode::OK,
            "a later pending rank must not un-ready a serving replica",
        );
    }

    /// A router without cache-aware-zmq has no tree to warm, so the gate must
    /// be completely inert for it.
    #[tokio::test]
    async fn readyz_200_when_no_kv_index_attached() {
        let ctx = test_ctx(true, true);
        assert!(ctx.kv_bootstrap_settled(), "absent index means settled");
        assert_eq!(readyz_status(ctx).await, StatusCode::OK);
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
                    transfer_group: None,
                })
                .expect("test worker accepted");
        }
        Arc::new(ctx)
    }
}
