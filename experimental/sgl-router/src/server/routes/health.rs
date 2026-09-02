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
/// Requires BOTH:
/// 1. `AppContext::mark_ready()` was called by main (process bootstrap
///    finished — config loaded, tokenizers built, server bound), AND
/// 2. At least one worker is registered. Without this second check,
///    `/readyz` flips green before the first `DiscoveryEvent::Added`
///    has been processed — the Service starts sending traffic to a
///    pod whose registry is empty, and every request returns 503
///    `no_healthy_workers`.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    if !ctx.is_ready() || ctx.registry.is_empty() {
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

    #[tokio::test]
    async fn readyz_503_for_pd_deployment_missing_decode_pool() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = AppContext::stub();
        ctx.mark_ready();
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("p1".into()),
                url: "http://p1:30000".into(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: Some(8997),
                transfer_group: None,
            })
            .unwrap();
        let app = crate::server::app::build_router(Arc::new(ctx));
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
    async fn readyz_503_for_pd_deployment_whose_prefill_has_no_bootstrap_port() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = AppContext::stub();
        ctx.mark_ready();
        for (id, mode, port) in [
            ("p1", WorkerMode::Prefill, None),
            ("d1", WorkerMode::Decode, None),
        ] {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:30000"),
                    mode,
                    model_ids: vec![ModelId("m".into())],
                    bootstrap_port: port,
                    transfer_group: None,
                })
                .unwrap();
        }
        let app = crate::server::app::build_router(Arc::new(ctx));
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
    async fn readyz_503_when_no_transfer_group_is_complete() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = AppContext::stub();
        ctx.mark_ready();
        // prefill in group a, decode in group b → no pairable group.
        for (id, mode, port, group) in [
            ("p1", WorkerMode::Prefill, Some(8997), "a"),
            ("d1", WorkerMode::Decode, None, "b"),
        ] {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:30000"),
                    mode,
                    model_ids: vec![ModelId("m".into())],
                    bootstrap_port: port,
                    transfer_group: Some(group.into()),
                })
                .unwrap();
        }
        let app = crate::server::app::build_router(Arc::new(ctx));
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
    async fn readyz_200_for_complete_pd_deployment() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = AppContext::stub();
        ctx.mark_ready();
        for (id, mode, port) in [
            ("p1", WorkerMode::Prefill, Some(8997)),
            ("d1", WorkerMode::Decode, None),
        ] {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:30000"),
                    mode,
                    model_ids: vec![ModelId("m".into())],
                    bootstrap_port: port,
                    transfer_group: None,
                })
                .unwrap();
        }
        let app = crate::server::app::build_router(Arc::new(ctx));
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
