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
use smg::observability::inflight_tracker::{get_tracker, init_inflight_tracker};
use tower::ServiceExt;

#[tokio::test]
async fn test_inflight_tracking_with_delayed_worker() {
    init_inflight_tracker(Duration::from_secs(1));

    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19001,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 100,
        fail_rate: 0.0,
    }])
    .await;

    let app = ctx.create_app().await;
    let tracker = get_tracker();
    let initial_count = tracker.map(|t| t.len()).unwrap_or(0);

    let payload = json!({
        "text": "Test tracking",
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let final_count = tracker.map(|t| t.len()).unwrap_or(0);
    assert_eq!(final_count, initial_count);

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_multiple_concurrent_requests_tracking() {
    init_inflight_tracker(Duration::from_secs(1));

    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19002,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 50,
        fail_rate: 0.0,
    }])
    .await;

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

    let tracker = get_tracker();
    assert!(tracker.is_some());

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_inflight_request_appears_in_bucket() {
    init_inflight_tracker(Duration::from_secs(1));

    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19004,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 2000,
        fail_rate: 0.0,
    }])
    .await;

    let tracker = get_tracker().expect("Tracker should be initialized");
    let initial_count = tracker.len();
    let initial_buckets = tracker.compute_bucket_counts();

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

    let response_future = tokio::spawn(async move { app.oneshot(req).await });

    tokio::time::sleep(Duration::from_millis(500)).await;

    let inflight_count = tracker.len();
    assert!(
        inflight_count > initial_count,
        "Should have at least one in-flight request, got {} (initial: {})",
        inflight_count,
        initial_count
    );

    let buckets = tracker.compute_bucket_counts();
    assert!(
        buckets[0] > initial_buckets[0],
        "le=30 bucket should have more requests (age < 30s): got {}, initial {}",
        buckets[0],
        initial_buckets[0]
    );
    assert!(
        buckets[5] > initial_buckets[5],
        "+Inf bucket should have more requests: got {}, initial {}",
        buckets[5],
        initial_buckets[5]
    );

    let resp = response_future.await.unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    tokio::time::sleep(Duration::from_millis(50)).await;
    let final_count = tracker.len();
    assert_eq!(final_count, initial_count, "Request should be deregistered after completion");

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_failed_request_still_deregisters() {
    init_inflight_tracker(Duration::from_secs(1));

    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19003,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 1.0,
    }])
    .await;

    let tracker = get_tracker();
    let initial_count = tracker.map(|t| t.len()).unwrap_or(0);

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

    let final_count = tracker.map(|t| t.len()).unwrap_or(0);
    assert_eq!(final_count, initial_count);

    ctx.shutdown().await;
}
