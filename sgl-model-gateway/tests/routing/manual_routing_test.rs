//! Manual routing policy integration tests
//!
//! Tests for the manual routing policy with sticky sessions using X-SMG-Routing-Key header.

use std::collections::{HashMap, HashSet};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

const ROUTING_KEY_HEADER: &str = "X-SMG-Routing-Key";

#[cfg(test)]
mod manual_routing_tests {
    use super::*;

    /// Test sticky routing with X-SMG-Routing-Key header
    #[tokio::test]
    async fn test_manual_routing_with_header() {
        let config = TestRouterConfig::manual(3700);

        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19700, 2))
                .await;

        let app = ctx.create_app().await;

        // Send requests with different routing keys
        let mut key_workers: HashMap<String, HashSet<String>> = HashMap::new();

        for key_id in 0..5 {
            let routing_key = format!("user-{}", key_id);

            for _ in 0..4 {
                let payload = json!({
                    "text": format!("Request for {}", routing_key),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .header(ROUTING_KEY_HEADER, &routing_key)
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app.clone().oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);

                // Extract worker ID from response header
                if let Some(worker_id) = resp.headers().get("x-worker-id") {
                    let worker = worker_id.to_str().unwrap().to_string();
                    key_workers
                        .entry(routing_key.clone())
                        .or_default()
                        .insert(worker);
                }
            }
        }

        // Verify sticky: each routing key should route to exactly one worker
        for (key, workers) in &key_workers {
            assert_eq!(
                workers.len(),
                1,
                "Routing key {} should route to exactly one worker, got {:?}",
                key,
                workers
            );
        }

        ctx.shutdown().await;
    }

    /// Test random fallback when no routing key header is provided
    #[tokio::test]
    async fn test_manual_routing_without_header() {
        let config = TestRouterConfig::manual(3701);

        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19702, 2))
                .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        // Send requests without routing key - should fall back to random selection
        for i in 0..20 {
            let payload = json!({
                "text": format!("Request without key {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                // No ROUTING_KEY_HEADER
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // All requests should succeed via random fallback
        assert_eq!(
            success_count, 20,
            "All requests should succeed with random fallback when no routing key provided"
        );

        ctx.shutdown().await;
    }

    /// Test that same routing key consistently routes to same worker
    #[tokio::test]
    async fn test_manual_routing_consistency() {
        let config = TestRouterConfig::manual(3702);

        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19704, 3))
                .await;

        let app = ctx.create_app().await;

        let routing_key = "consistent-user-123";
        let mut seen_workers: Vec<String> = Vec::new();

        // Send multiple requests with same routing key
        for i in 0..10 {
            let payload = json!({
                "text": format!("Consistent request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .header(ROUTING_KEY_HEADER, routing_key)
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            if let Some(worker_id) = resp.headers().get("x-worker-id") {
                seen_workers.push(worker_id.to_str().unwrap().to_string());
            }
        }

        // All requests should go to the same worker
        if !seen_workers.is_empty() {
            let first_worker = &seen_workers[0];
            for worker in &seen_workers {
                assert_eq!(
                    worker, first_worker,
                    "All requests with same routing key should go to same worker"
                );
            }
        }

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod manual_min_group_tests {
    use super::*;

    #[tokio::test]
    async fn test_min_group_handles_multiple_routing_keys() {
        let config = TestRouterConfig::manual_min_group(3910);

        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(29910, 3))
                .await;

        let app = ctx.create_app().await;
        let mut key_to_worker: HashMap<String, String> = HashMap::new();

        for i in 0..9 {
            let routing_key = format!("key-{}", i);

            let payload = json!({
                "text": format!("Request for {}", routing_key),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .header(ROUTING_KEY_HEADER, &routing_key)
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            let worker_id = resp
                .headers()
                .get("x-worker-id")
                .expect("Response should have x-worker-id header")
                .to_str()
                .unwrap()
                .to_string();

            key_to_worker.insert(routing_key, worker_id);
        }

        let worker_counts: HashMap<String, usize> =
            key_to_worker.values().fold(HashMap::new(), |mut acc, w| {
                *acc.entry(w.clone()).or_default() += 1;
                acc
            });

        assert_eq!(
            worker_counts.len(),
            3,
            "min_group should distribute keys across all 3 workers"
        );
        for (worker, count) in &worker_counts {
            assert_eq!(
                *count, 3,
                "Worker {} should have exactly 3 keys, got {}",
                worker, count
            );
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_min_group_sticky_routing() {
        let config = TestRouterConfig::manual_min_group(3911);

        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(29920, 3))
                .await;

        let app = ctx.create_app().await;

        let routing_key = "sticky-key-123";
        let mut seen_workers: Vec<String> = Vec::new();

        for i in 0..5 {
            let payload = json!({
                "text": format!("Request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .header(ROUTING_KEY_HEADER, routing_key)
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            let worker_id = resp
                .headers()
                .get("x-worker-id")
                .expect("Response should have x-worker-id header")
                .to_str()
                .unwrap()
                .to_string();

            seen_workers.push(worker_id);
        }

        let first_worker = &seen_workers[0];
        for (i, worker) in seen_workers.iter().enumerate() {
            assert_eq!(
                worker, first_worker,
                "Request {} should go to same worker as first request (expected {}, got {})",
                i, first_worker, worker
            );
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_min_group_mixed_routing_keys() {
        let config = TestRouterConfig::manual_min_group(3912);

        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(29930, 2))
                .await;

        let app = ctx.create_app().await;
        let mut key_to_workers: HashMap<String, HashSet<String>> = HashMap::new();

        for i in 0..4 {
            let routing_key = format!("key-{}", i);

            for j in 0..3 {
                let payload = json!({
                    "text": format!("Request {} for {}", j, routing_key),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .header(ROUTING_KEY_HEADER, &routing_key)
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app.clone().oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);

                let worker_id = resp
                    .headers()
                    .get("x-worker-id")
                    .expect("Response should have x-worker-id header")
                    .to_str()
                    .unwrap()
                    .to_string();

                key_to_workers
                    .entry(routing_key.clone())
                    .or_default()
                    .insert(worker_id);
            }
        }

        for (key, workers) in &key_to_workers {
            assert_eq!(
                workers.len(),
                1,
                "Key {} should route to exactly one worker (sticky), but got {:?}",
                key,
                workers
            );
        }

        let all_workers: HashSet<String> =
            key_to_workers.values().flatten().cloned().collect();
        assert_eq!(
            all_workers.len(),
            2,
            "Keys should be distributed across both workers"
        );

        ctx.shutdown().await;
    }
}
