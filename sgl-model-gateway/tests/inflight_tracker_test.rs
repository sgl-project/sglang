mod common;

use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};
use serde_json::json;
use tower::ServiceExt;

#[tokio::test]
async fn test_multiple_concurrent_requests_tracking() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19002,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 50,
        fail_rate: 0.0,
    }])
    .await;

    let tracker = ctx.app_context.inflight_tracker.clone();

    let mut handles = vec![];
    for i in 0..5 {
        let app = ctx.create_app().await;
        handles.push(tokio::spawn(async move {
            let payload = json!({
                "text": format!("Request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            app.oneshot(req).await.unwrap()
        }));
    }

    for handle in handles {
        let resp = handle.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    assert!(tracker.is_empty());

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_inflight_request_appears_in_bucket() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19004,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 2000,
        fail_rate: 0.0,
    }])
    .await;

    let tracker = &ctx.app_context.inflight_tracker;
    assert!(tracker.is_empty(), "Tracker should start empty");

    let app = ctx.create_app().await;
    let payload = json!({
        "text": "Long running request",
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let tracker_clone = ctx.app_context.inflight_tracker.clone();
    let response_future = tokio::spawn(async move { app.oneshot(req).await });

    tokio::time::sleep(Duration::from_millis(500)).await;

    let inflight_count = tracker_clone.len();
    assert!(
        inflight_count > 0,
        "Should have at least one in-flight request, got {}",
        inflight_count
    );

    let buckets = tracker_clone.compute_bucket_counts();
    assert!(buckets[0] > 0, "first bucket (<=30s) should have requests");

    let resp = response_future.await.unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        tracker_clone.is_empty(),
        "Request should be deregistered after completion"
    );

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_failed_request_still_deregisters() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19003,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 1.0,
    }])
    .await;

    let tracker = &ctx.app_context.inflight_tracker;
    assert!(tracker.is_empty());

    let app = ctx.create_app().await;

    let payload = json!({
        "text": "This should fail",
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    assert!(tracker.is_empty());

    ctx.shutdown().await;
}
