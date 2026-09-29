// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::ModelId;
use crate::policies::registry::PdPoolResolver;
use crate::server::app_context::{AppContext, ChatRouting};
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
    let model = ModelId(ctx.config.model.id.clone());
    let ready = match &ctx.chat_routing {
        ChatRouting::Legacy => PdPoolResolver::new(Arc::clone(&ctx.registry))
            .prefill_candidates(&model)
            .is_ok_and(|workers| !workers.is_empty()),
        ChatRouting::Reorg(resolvers) => resolvers.get(&model).is_some_and(|resolver| {
            resolver
                .buckets
                .iter()
                .any(|bucket| bucket.has_ready_workers(&ctx.registry, &model))
        }),
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

        for (model, workers, ready) in [
            (None, vec![(Plain, None)], false),
            (Some("other"), vec![(Plain, None)], false),
            (Some("stub-model"), vec![(Prefill, None)], false),
            (Some("stub-model"), vec![(Decode, None)], false),
            (
                Some("stub-model"),
                vec![(Prefill, None), (Decode, None)],
                true,
            ),
            // Both roles present, but in different version groups: nothing can pair.
            (
                Some("stub-model"),
                vec![(Prefill, Some("v1")), (Decode, Some("v2"))],
                false,
            ),
            (
                Some("stub-model"),
                vec![(Prefill, Some("v1")), (Decode, Some("v1"))],
                true,
            ),
        ] {
            let ctx = test_ctx(true, false);
            for (i, (mode, group)) in workers.into_iter().enumerate() {
                ctx.registry
                    .add(WorkerSpec {
                        id: WorkerId(i.to_string()),
                        url: format!("http://worker-{i}:30000"),
                        mode,
                        model_ids: model.map(|m| ModelId(m.into())).into_iter().collect(),
                        bootstrap_port: (mode == Prefill).then_some(8997),
                        version_group: group.map(str::to_owned),
                    })
                    .unwrap();
            }
            assert_eq!(readyz(State(ctx)).await == StatusCode::OK, ready);
        }
    }

    #[tokio::test]
    async fn readiness_rejects_portless_prefill() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = test_ctx(true, false);
        for mode in [WorkerMode::Prefill, WorkerMode::Decode] {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(format!("{mode:?}")),
                    url: format!("http://{mode:?}:30000"),
                    mode,
                    model_ids: vec![ModelId(ctx.config.model.id.clone())],
                    ..Default::default()
                })
                .unwrap();
        }
        assert_eq!(readyz(State(ctx)).await, StatusCode::SERVICE_UNAVAILABLE);
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
                    ..Default::default()
                })
                .expect("test worker accepted");
        }
        Arc::new(ctx)
    }
}
