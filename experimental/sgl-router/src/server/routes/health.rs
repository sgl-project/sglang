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

/// Ready after startup only when ALL hold:
/// 1. `mark_ready()` was called.
/// 2. The configured model has a usable plain or PD pool.
/// 3. Peer bootstrap has settled (`BootstrapTracker::settled`): every rank
///    resolved, or the `--kv-bootstrap-timeout-ms` deadline passed. Always
///    true without cache-aware and `--kv-peer-selector`.
/// 4. The seed gate is open (`BootstrapTracker::seed_gate_open`). Condition 3
///    settles on the deadline either way, so it cannot tell "seeded" from
///    "gave up"; with `--kv-bootstrap-seed-required` this holds a failed seed
///    out of the Service for up to max(3x --kv-bootstrap-timeout-ms, 60s).
///
/// Conditions 3 and 4 latch on the first 200 (`BootstrapTracker::admit_ready`),
/// so a later scale-up cannot drag a serving replica back to 503.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    if !ctx.is_ready() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let pool_ready = match resolver.resolve(&ModelId(ctx.config.model.id.clone())) {
        Ok(PdPools::Plain { workers }) => !workers.is_empty(),
        Ok(PdPools::Pd { prefill, decode }) => !prefill.is_empty() && !decode.is_empty(),
        Err(_) => false,
    };
    if pool_ready && ctx.kv_bootstrap_admit_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::kv_events::BootstrapTracker;
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

    /// A ready context with one worker whose KV index uses `tracker`.
    fn ctx_with_tracker(tracker: Arc<BootstrapTracker>) -> Arc<AppContext> {
        use crate::state::kv_events::{BlockSizeOracle, KvEventIndex};
        let mut ctx = AppContext::stub();
        ctx.mark_ready();
        add_test_worker(&ctx);
        ctx.kv_index = KvEventIndex::new_with_bootstrap(
            reqwest::Client::new(),
            BlockSizeOracle::new(),
            tracker,
        )
        .snapshot_source();
        Arc::new(ctx)
    }

    /// A context whose seed gate is the ONLY thing that can hold `/readyz`
    /// down: the rank is terminal, so condition 3 is satisfied and a 503 cannot
    /// come from it.
    fn ctx_with_seed_gate(seed_required: bool, failed: bool) -> Arc<AppContext> {
        use crate::state::kv_events::bootstrap::SweepOutcome;
        use crate::state::kv_events::{BootstrapState, KvWorkerId};

        let tracker = Arc::new(BootstrapTracker::new_with_opts(
            std::time::Duration::from_secs(300),
            std::time::Duration::from_secs(120),
            seed_required,
        ));
        let rank = KvWorkerId::new("http://w1:30000".into(), 0);
        tracker.register(std::slice::from_ref(&rank));
        tracker.set(&rank, BootstrapState::Failed);
        assert!(tracker.settled(), "condition 3 must not be what gates here");
        // The verdict that terminated the rank. Failed: nine siblings were
        // there and none gave up its tree, the shape a first-wave surge pod
        // hits on a rolling update. Otherwise: they proved they hold nothing.
        let verdict = if failed {
            SweepOutcome::TimedOut
        } else {
            SweepOutcome::FleetCold
        };
        tracker.record_sweep_result(verdict, 9);
        ctx_with_tracker(tracker)
    }

    fn add_test_worker(ctx: &AppContext) {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
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

    /// The gate's reason for existing: a replica that could not pull the
    /// fleet's tree must stay OUT of the Service, so a rolling update stalls
    /// with the previous generation serving instead of completing with
    /// cache-blind replicas.
    #[tokio::test]
    async fn readyz_503_when_a_required_seed_failed() {
        assert_eq!(
            readyz_status(ctx_with_seed_gate(true, true)).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );
    }

    /// Same failed sweep, gate not opted in: today's behaviour is preserved
    /// exactly, so enabling the flag is the only thing that can change a
    /// fleet's rollout semantics.
    #[tokio::test]
    async fn readyz_200_on_the_same_failure_when_the_gate_is_off() {
        assert_eq!(
            readyz_status(ctx_with_seed_gate(false, true)).await,
            StatusCode::OK,
        );
    }

    /// Answering 200 once latches the gate: a sweep started by a
    /// late-discovered worker must not pull a serving replica back out of the
    /// Service; an unready replica also leaves its own EndpointSlice, so its
    /// siblings lose a bootstrap source too.
    #[tokio::test]
    async fn readyz_stays_200_after_a_later_sweep_fails() {
        use crate::state::kv_events::bootstrap::SweepOutcome;
        let ctx = ctx_with_seed_gate(true, false);
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
        use crate::state::kv_events::KvWorkerId;

        let tracker = Arc::new(BootstrapTracker::new(std::time::Duration::from_secs(300)));
        tracker.register(&[KvWorkerId::new("http://w1:30000".into(), 0)]);
        assert!(!tracker.settled());

        assert_eq!(
            readyz_status(ctx_with_tracker(tracker)).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );
    }

    fn test_ctx(ready: bool, with_worker: bool) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        if ready {
            ctx.mark_ready();
        }
        if with_worker {
            add_test_worker(&ctx);
        }
        Arc::new(ctx)
    }
}
