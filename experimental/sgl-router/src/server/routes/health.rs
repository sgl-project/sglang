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
/// Requires BOTH:
/// 1. `AppContext::mark_ready()` was called by main (process bootstrap
///    finished — config loaded, tokenizers built, server bound), AND
/// 2. At least one worker is registered AND reported able to serve by
///    discovery (`Worker::serving`). Without this second check,
///    `/readyz` flips green before the first `DiscoveryEvent::Added`
///    has been processed — the Service starts sending traffic to a
///    pod whose registry is empty, and every request returns 503
///    `no_healthy_workers`.
///
/// Deliberately does NOT consult the circuit breaker: that is the router's own
/// observation of failures, and coupling k8s readiness to it would withdraw the
/// router from its Service on a transient engine blip — destroying every radix
/// tree the `ReadyChanged` path exists to preserve. A fleet that is registered
/// and serving but breaker-open still reports Ready and returns a structured
/// 503 per request, which is the diagnosable outcome.
///
/// Note the shipped manifest points `livenessProbe` at `/healthz`, not here; if
/// that ever changes, a fleet-wide engine restart would restart the router too.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    // Registered is not the same as selectable. `healthy_workers_for` drops
    // any worker discovery reports not-ready, so a registry made entirely of
    // not-ready workers 503s on every request (`no_healthy_workers`, or the
    // per-pool code for a PD model). Condition
    // 2 used to catch that through the emptiness test alone, because the k8s
    // backend expressed not-ready as `Removed`; now that it is `ReadyChanged`
    // the entry survives, so the check has to read `serving` to keep meaning
    // what it says.
    if ctx.is_ready() && ctx.registry.all().iter().any(|w| w.serving()) {
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

    /// The same regression, one layer in: a registry that is NOT empty but
    /// whose every worker is discovery-not-ready.
    ///
    /// Before `ReadyChanged` existed, the k8s backend expressed not-ready as
    /// `Removed`, so this case collapsed into the empty-registry check above.
    /// It no longer does — the entry survives on purpose, to keep the KV tree
    /// — so `/readyz` has to test `serving` explicitly. Otherwise a router
    /// whose whole engine fleet is restarting reports Ready, joins the
    /// Service, and 503s `no_healthy_workers` on every request.
    #[tokio::test]
    async fn readyz_503_when_every_worker_is_not_serving() {
        let ctx = test_ctx(true, true);
        for w in ctx.registry.all() {
            w.set_serving(false);
        }
        let app = crate::server::app::build_router(ctx);
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
            "a registry of only not-ready workers must be 503, not Ready"
        );
    }

    /// One restarting engine out of many must NOT pull the router out of its
    /// Service.
    ///
    /// The predicate is `any(serving)`, and with a single-worker fixture `any`
    /// and `all` are indistinguishable — so this is the case that pins which
    /// one it is. An `all`-flavoured regression would turn one rolling pod into
    /// a fleet-wide router withdrawal.
    #[tokio::test]
    async fn readyz_200_when_only_some_workers_are_not_serving() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = test_ctx(true, true);
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("test-w2".into()),
                url: "http://test2:30000".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("test".into())],
                bootstrap_port: None,
            })
            .expect("second test worker accepted");
        // Exactly one of the two is out of service.
        ctx.registry
            .get(&WorkerId("test-w".into()))
            .unwrap()
            .set_serving(false);
        assert_eq!(ctx.registry.all().len(), 2, "fixture: two workers");

        let app = crate::server::app::build_router(ctx);
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
            StatusCode::OK,
            "one not-ready engine among several must not withdraw the router",
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
