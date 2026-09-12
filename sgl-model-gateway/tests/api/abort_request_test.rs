use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use serial_test::serial;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{clear_aborts_received, get_aborts_received, MockWorkerConfig},
    AppTestContext,
};

#[cfg(test)]
mod abort_request_tests {
    use super::*;

    // All three tests share a process-global abort tracker, so they must run
    // serially to avoid clearing/reading each other's records.
    #[tokio::test]
    #[serial]
    async fn test_abort_request_fans_out_to_all_workers_with_body() {
        clear_aborts_received();
        let ctx = AppTestContext::new(vec![
            MockWorkerConfig {
                port: 18501,
                ..MockWorkerConfig::default()
            },
            MockWorkerConfig {
                port: 18502,
                ..MockWorkerConfig::default()
            },
        ])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({"rid": "req-abc-123", "abort_all": false});
        let req = Request::builder()
            .method("POST")
            .uri("/abort_request")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Both workers must have received the abort — the rid may live on
        // either worker, so the gateway must fan out to all of them rather
        // than returning on the first success.
        let aborts = get_aborts_received();
        let ports: Vec<u16> = aborts.iter().map(|(p, _)| *p).collect();
        assert!(ports.contains(&18501), "worker 18501 missing: {ports:?}");
        assert!(ports.contains(&18502), "worker 18502 missing: {ports:?}");

        // Body must be forwarded intact to every worker.
        for (_port, body) in &aborts {
            assert_eq!(body, &json!({"rid": "req-abc-123", "abort_all": false}));
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    #[serial]
    async fn test_abort_request_no_workers_returns_unavailable() {
        clear_aborts_received();
        let ctx = AppTestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let payload = json!({"rid": "req-xyz", "abort_all": false});
        let req = Request::builder()
            .method("POST")
            .uri("/abort_request")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(get_aborts_received().is_empty());

        ctx.shutdown().await;
    }

    // Fan-out must reach every worker — including the failing one — and a
    // success on any worker must win over a failure on another.
    #[tokio::test]
    #[serial]
    async fn test_abort_request_success_wins_over_partial_failure() {
        clear_aborts_received();
        let ctx = AppTestContext::new(vec![
            MockWorkerConfig {
                port: 18511,
                fail_rate: 1.0, // always 500 — simulates a worker that errors
                ..MockWorkerConfig::default()
            },
            MockWorkerConfig {
                port: 18512,
                ..MockWorkerConfig::default()
            },
        ])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({"rid": "req-partial", "abort_all": false});
        let req = Request::builder()
            .method("POST")
            .uri("/abort_request")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a success on any worker must win over a failure on another"
        );

        // Fan-out reaches every worker regardless of individual outcomes —
        // the failing worker is probed too (the mock records before failing).
        let ports: Vec<u16> = get_aborts_received().iter().map(|(p, _)| *p).collect();
        assert!(
            ports.contains(&18511),
            "failing worker 18511 missing: {ports:?}"
        );
        assert!(
            ports.contains(&18512),
            "healthy worker 18512 missing: {ports:?}"
        );

        ctx.shutdown().await;
    }
}
