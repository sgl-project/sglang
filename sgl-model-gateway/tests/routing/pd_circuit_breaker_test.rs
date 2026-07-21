//! PD circuit-breaker dispatch tests (issue #31206)
//!
//! Regression coverage for the "open prefill breaker still dispatches decode"
//! bug. In PD disaggregation a request is only meaningful as a prefill -> decode
//! pair. When the prefill worker's circuit breaker is open, the router must fail
//! the whole request fast (503) rather than skipping the prefill leg and still
//! dispatching decode. Decode-only dispatch leaves the decode worker waiting for
//! a KV/bootstrap transfer that never arrives, so the request hangs and the
//! prefill side looks permanently "dead" even after it has recovered.
//!
//! The report's key detail is that the prefill worker's heartbeat stayed
//! *healthy* while its breaker was stuck open and no POSTs reached it. So the
//! decisive state to test is "healthy worker, open breaker": the request must
//! still fail as a whole, purely because of the breaker. These tests separate
//! the two claims:
//!   * `repeated_prefill_failures_open_the_breaker` -- real failing prefill
//!     requests trip the breaker open (the "how it got tripped" half).
//!   * `open_prefill_breaker_fails_whole_request_not_decode_only` -- with a
//!     *healthy* prefill whose breaker is open, the request fails fast (503),
//!     not decode-only. Health is asserted true so the 503 is attributable to
//!     the breaker alone; a decode-only regression would instead return 200.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::{
    config::{CircuitBreakerConfig, RetryConfig, RouterConfig},
    core::Worker,
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod pd_circuit_breaker_tests {
    use super::*;

    /// Build a 1-prefill / 1-decode PD config with a low breaker `failure_threshold`
    /// (so a few real failures open it) and bounded, fast retries.
    fn pd_config(
        port: u16,
        prefill_url: &str,
        decode_url: &str,
        failure_threshold: u32,
    ) -> RouterConfig {
        RouterConfig::builder()
            .prefill_decode_mode(
                vec![(prefill_url.to_string(), None)],
                vec![decode_url.to_string()],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(port)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold,
                ..Default::default()
            })
            .build_unchecked()
    }

    /// Send one non-streaming /generate request and return its status.
    async fn generate_status(app: &axum::Router) -> StatusCode {
        let payload = json!({ "text": "hello", "stream": false });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        app.clone().oneshot(req).await.unwrap().status()
    }

    /// Grab the single prefill worker registered for this context.
    fn prefill_worker(ctx: &AppTestContext) -> std::sync::Arc<dyn Worker> {
        ctx.app_context
            .worker_registry
            .get_prefill_workers()
            .into_iter()
            .next()
            .expect("exactly one prefill worker should be registered")
    }

    /// Positive control: a healthy 1P1D pair serves requests. This makes the 503
    /// in the regression test below attributable to the open breaker rather than
    /// to a mis-wired harness.
    #[tokio::test]
    async fn healthy_pd_pair_serves_requests() {
        let config = pd_config(3810, "http://127.0.0.1:19840", "http://127.0.0.1:19841", 3);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19840),
                TestWorkerConfig::decode(19841),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        assert_eq!(
            generate_status(&app).await,
            StatusCode::OK,
            "a healthy PD pair should return 200"
        );

        ctx.shutdown().await;
    }

    /// (a) Real repeated prefill failures trip the breaker open. Mirrors the
    /// report's "we rebuilt one worker; the unhealthy count opened the breaker":
    /// each failing request records a failure until the breaker transitions open.
    #[tokio::test]
    async fn repeated_prefill_failures_open_the_breaker() {
        const FAILURE_THRESHOLD: u32 = 3;

        let config = pd_config(
            3811,
            "http://127.0.0.1:19842",
            "http://127.0.0.1:19843",
            FAILURE_THRESHOLD,
        );
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill returns 500 on every /generate (fail_rate 1.0).
                MockWorkerConfig {
                    port: 19842,
                    worker_type: WorkerType::Prefill,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0,
                },
                TestWorkerConfig::decode(19843),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        let prefill = prefill_worker(&ctx);
        assert!(
            prefill.circuit_breaker().can_execute(),
            "breaker should start closed"
        );

        // Drive real failing requests until the breaker opens; bounded so a
        // regression that never opens the breaker fails instead of spinning.
        let mut opened = false;
        for _ in 0..(FAILURE_THRESHOLD as usize * 4) {
            let status = generate_status(&app).await;
            assert_ne!(
                status,
                StatusCode::OK,
                "a request to a failing prefill worker must not return 200"
            );
            if !prefill.circuit_breaker().can_execute() {
                opened = true;
                break;
            }
        }
        assert!(
            opened,
            "prefill breaker should open after repeated prefill failures"
        );

        ctx.shutdown().await;
    }

    /// (b) The decisive regression: a *healthy* prefill worker whose breaker is
    /// open must cause the whole PD request to fail fast (503) instead of being
    /// dispatched decode-only. This is exactly the report's state -- healthy
    /// heartbeat, breaker stuck open, no POSTs to prefill.
    ///
    /// Health is asserted true so the 503 is attributable solely to the breaker.
    /// If prefill selection ignored the breaker (the pre-fix / regression path),
    /// this healthy worker would be selected and the request would return 200
    /// decode-only rather than 503 -- which this assertion catches.
    #[tokio::test]
    async fn open_prefill_breaker_fails_whole_request_not_decode_only() {
        let config = pd_config(3812, "http://127.0.0.1:19844", "http://127.0.0.1:19845", 3);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Both workers are fully healthy (fail_rate 0.0).
                TestWorkerConfig::prefill(19844),
                TestWorkerConfig::decode(19845),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        // Sanity: the healthy pair serves before we touch the breaker.
        assert_eq!(
            generate_status(&app).await,
            StatusCode::OK,
            "healthy pair should serve before the breaker is opened"
        );

        // Open the prefill breaker while the worker stays healthy -- the report's
        // "heartbeat stayed healthy" state.
        let prefill = prefill_worker(&ctx);
        prefill.circuit_breaker().force_open();

        assert!(
            prefill.is_healthy(),
            "prefill must remain healthy so the 503 is attributable to the breaker"
        );
        assert!(
            !prefill.circuit_breaker().can_execute(),
            "prefill breaker must be open"
        );

        // The whole request must fail fast at selection (503), not go decode-only.
        let status = generate_status(&app).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "open prefill breaker must fail the whole PD request, got {status}"
        );

        ctx.shutdown().await;
    }
}
