// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::ModelId;
use crate::policies::registry::{PdPoolResolver, PdPools};
use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;

/// Always returns 200 — liveness probe.
pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

/// Ready after startup only when the configured model has a usable plain or PD pool.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    if !ctx.is_ready() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let ready = match resolver.resolve(&ModelId(ctx.config.model.id.clone())) {
        Ok(PdPools::Plain { workers }) => !workers.is_empty(),
        Ok(PdPools::Pd { prefill, decode }) => !prefill.is_empty() && !decode.is_empty(),
        Err(_) => false,
    };
    if ready {
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

    #[tokio::test]
    async fn readiness_requires_resolved_model_and_both_pd_roles() {
        use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};

        use WorkerMode::{Decode, Plain, Prefill};

        for (model, modes, ready) in [
            (None, vec![Plain], false),
            (Some("other"), vec![Plain], false),
            (Some("stub-model"), vec![Prefill], false),
            (Some("stub-model"), vec![Decode], false),
            (Some("stub-model"), vec![Prefill, Decode], true),
        ] {
            let ctx = test_ctx(true, false);
            for (i, mode) in modes.into_iter().enumerate() {
                ctx.registry
                    .add(WorkerSpec {
                        id: WorkerId(i.to_string()),
                        url: format!("http://worker-{i}:30000"),
                        mode,
                        model_ids: model.map(|m| ModelId(m.into())).into_iter().collect(),
                        bootstrap_port: None,
                    })
                    .unwrap();
            }
            assert_eq!(readyz(State(ctx)).await == StatusCode::OK, ready);
        }
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
                    model_ids: vec![ModelId(ctx.config.model.id.clone())],
                    bootstrap_port: None,
                })
                .expect("test worker accepted");
        }
        Arc::new(ctx)
    }
}
