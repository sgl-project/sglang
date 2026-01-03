mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    TestContext,
};
use serde_json::json;
use smg::observability::inflight_tracker::{get_tracker, init_inflight_tracker, InFlightRequestTracker};
use tower::ServiceExt;

#[cfg(test)]
mod inflight_tracker_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_tracker_direct_register_deregister() {
        let tracker = InFlightRequestTracker::new_for_test();

        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);

        tracker.register("req-1");
        assert_eq!(tracker.len(), 1);

        tracker.register("req-2");
        assert_eq!(tracker.len(), 2);

        tracker.deregister("req-1");
        assert_eq!(tracker.len(), 1);

        tracker.deregister("req-2");
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_tracker_bucket_counts_empty() {
        let tracker = InFlightRequestTracker::new_for_test();
        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts, [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_tracker_bucket_counts_fresh_requests() {
        let tracker = InFlightRequestTracker::new_for_test();

        tracker.register("req-1");
        tracker.register("req-2");
        tracker.register("req-3");

        let counts = tracker.compute_bucket_counts();

        // All fresh requests (age ~0s) should be in all buckets
        assert_eq!(counts[0], 3, "le=30 should have 3");
        assert_eq!(counts[1], 3, "le=60 should have 3");
        assert_eq!(counts[2], 3, "le=180 should have 3");
        assert_eq!(counts[3], 3, "le=300 should have 3");
        assert_eq!(counts[4], 3, "le=600 should have 3");
        assert_eq!(counts[5], 3, "+Inf should have 3");
    }

    #[test]
    fn test_tracker_bucket_counts_cumulative_semantics() {
        let tracker = InFlightRequestTracker::new_for_test();
        let now = Instant::now();

        // Insert requests with different ages
        tracker.insert_with_time("req-0s", now);
        tracker.insert_with_time("req-45s", now - Duration::from_secs(45));
        tracker.insert_with_time("req-100s", now - Duration::from_secs(100));
        tracker.insert_with_time("req-250s", now - Duration::from_secs(250));
        tracker.insert_with_time("req-500s", now - Duration::from_secs(500));
        tracker.insert_with_time("req-700s", now - Duration::from_secs(700));

        let counts = tracker.compute_bucket_counts();

        // Verify cumulative semantics:
        // le="30":  1 (only req-0s)
        // le="60":  2 (req-0s, req-45s)
        // le="180": 3 (req-0s, req-45s, req-100s)
        // le="300": 4 (req-0s, req-45s, req-100s, req-250s)
        // le="600": 5 (req-0s, req-45s, req-100s, req-250s, req-500s)
        // +Inf:     6 (all)
        assert_eq!(counts[0], 1, "le=30");
        assert_eq!(counts[1], 2, "le=60");
        assert_eq!(counts[2], 3, "le=180");
        assert_eq!(counts[3], 4, "le=300");
        assert_eq!(counts[4], 5, "le=600");
        assert_eq!(counts[5], 6, "+Inf");
    }

    #[test]
    fn test_tracker_bucket_boundary_exact() {
        let tracker = InFlightRequestTracker::new_for_test();
        let now = Instant::now();

        // Exact boundary: 30s should be included in le=30
        tracker.insert_with_time("req-30s", now - Duration::from_secs(30));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 1, "le=30 should include exact boundary");
    }

    #[test]
    fn test_tracker_bucket_boundary_just_over() {
        let tracker = InFlightRequestTracker::new_for_test();
        let now = Instant::now();

        // Just over boundary: 31s should NOT be in le=30
        tracker.insert_with_time("req-31s", now - Duration::from_secs(31));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 0, "le=30 should NOT include 31s");
        assert_eq!(counts[1], 1, "le=60 should include 31s");
    }

    #[test]
    fn test_tracker_concurrent_operations() {
        use std::thread;

        let tracker: Arc<InFlightRequestTracker> =
            Arc::new(InFlightRequestTracker::new_for_test());
        let mut handles = vec![];

        // Spawn register threads
        for i in 0..5 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for j in 0..50 {
                    t.register(&format!("t{}-r{}", i, j));
                }
            }));
        }

        // Spawn deregister threads (partial overlap)
        for i in 0..2 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for j in 0..50 {
                    t.deregister(&format!("t{}-r{}", i, j));
                }
            }));
        }

        for h in handles {
            h.join().expect("Thread should not panic");
        }

        // Should be able to compute bucket counts without panic
        let _counts = tracker.compute_bucket_counts();
    }

    #[tokio::test]
    async fn test_inflight_tracking_with_delayed_worker() {
        // Initialize the global tracker if not already initialized
        init_inflight_tracker(Duration::from_secs(1));

        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 100, // 100ms delay
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // Get tracker reference
        let tracker = get_tracker();

        // Record initial count
        let initial_count = tracker.map(|t| t.len()).unwrap_or(0);

        // Send a request
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

        // After request completes, count should return to initial
        let final_count = tracker.map(|t| t.len()).unwrap_or(0);
        assert_eq!(
            final_count, initial_count,
            "In-flight count should return to initial after request completes"
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_multiple_concurrent_requests_tracking() {
        init_inflight_tracker(Duration::from_secs(1));

        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19002,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 50,
            fail_rate: 0.0,
        }])
        .await;

        // Send multiple requests concurrently
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

        // Wait for all requests to complete
        for handle in handles {
            let resp = handle.await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        // Verify tracker is functioning (global tracker may have entries from other tests)
        let tracker = get_tracker();
        assert!(tracker.is_some(), "Tracker should be initialized");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_failed_request_still_deregisters() {
        init_inflight_tracker(Duration::from_secs(1));

        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19003,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0, // Always fail
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
        // Request fails but should still deregister
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let final_count = tracker.map(|t| t.len()).unwrap_or(0);
        assert_eq!(
            final_count, initial_count,
            "Failed requests should also be deregistered"
        );

        ctx.shutdown().await;
    }
}

