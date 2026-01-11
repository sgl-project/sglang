use std::collections::{HashMap, HashSet};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{
    manual_routing_test_helpers::{
        manual_routing_all_backend_test, send_request, TestManualConfig, ROUTING_KEY_HEADER,
    },
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

// ============================================================================
// Basic Manual Routing Tests
// ============================================================================

async fn test_routing_with_header_impl(cfg: TestManualConfig, base_port: u16) {
    let config = TestRouterConfig::manual_with_full_options(
        base_port,
        smg::config::ManualAssignmentMode::Random,
        &cfg,
    );
    let ctx = AppTestContext::new_with_config(
        config,
        TestWorkerConfig::healthy_workers(base_port + 16000, 2),
    )
    .await;

    let app = ctx.create_app().await;
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

            if let Some(worker_id) = resp.headers().get("x-worker-id") {
                let worker = worker_id.to_str().unwrap().to_string();
                key_workers
                    .entry(routing_key.clone())
                    .or_default()
                    .insert(worker);
            }
        }
    }

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

manual_routing_all_backend_test!(test_routing_with_header, 3700);

async fn test_routing_without_header_impl(cfg: TestManualConfig, base_port: u16) {
    let config = TestRouterConfig::manual_with_full_options(
        base_port,
        smg::config::ManualAssignmentMode::Random,
        &cfg,
    );
    let ctx = AppTestContext::new_with_config(
        config,
        TestWorkerConfig::healthy_workers(base_port + 16000, 2),
    )
    .await;

    let app = ctx.create_app().await;
    let mut success_count = 0;

    for i in 0..20 {
        let payload = json!({
            "text": format!("Request without key {}", i),
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        if resp.status() == StatusCode::OK {
            success_count += 1;
        }
    }

    assert_eq!(
        success_count, 20,
        "All requests should succeed with random fallback when no routing key provided"
    );

    ctx.shutdown().await;
}

manual_routing_all_backend_test!(test_routing_without_header, 3710);

async fn test_routing_consistency_impl(cfg: TestManualConfig, base_port: u16) {
    let config = TestRouterConfig::manual_with_full_options(
        base_port,
        smg::config::ManualAssignmentMode::Random,
        &cfg,
    );
    let ctx = AppTestContext::new_with_config(
        config,
        TestWorkerConfig::healthy_workers(base_port + 16000, 3),
    )
    .await;

    let app = ctx.create_app().await;
    let routing_key = "consistent-user-123";
    let mut seen_workers: Vec<String> = Vec::new();

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

manual_routing_all_backend_test!(test_routing_consistency, 3720);

// ============================================================================
// Min Group Mode Tests
// ============================================================================

async fn test_min_group_concurrent_distribution_impl(cfg: TestManualConfig, base_port: u16) {
    let config = TestRouterConfig::manual_with_full_options(
        base_port,
        smg::config::ManualAssignmentMode::MinGroup,
        &cfg,
    );
    let ctx = AppTestContext::new_with_config(
        config,
        TestWorkerConfig::slow_workers(base_port + 26000, 3, 500),
    )
    .await;

    let app = ctx.create_app().await;

    let mut handles = Vec::new();
    for i in 0..9 {
        let routing_key = format!("key-{}", i);
        let app_clone = app.clone();
        let handle = tokio::spawn(async move { send_request(app_clone, &routing_key).await });
        handles.push(handle);
    }

    let results: Vec<(String, String)> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let key_to_worker: HashMap<String, String> = results.into_iter().collect();

    let worker_counts: HashMap<String, usize> =
        key_to_worker.values().fold(HashMap::new(), |mut acc, w| {
            *acc.entry(w.clone()).or_default() += 1;
            acc
        });

    assert_eq!(
        worker_counts.len(),
        3,
        "min_group should distribute keys across all 3 workers, got {:?}",
        worker_counts
    );
    for (worker, count) in &worker_counts {
        assert_eq!(
            *count, 3,
            "Worker {} should have exactly 3 keys, got {}. Distribution: {:?}",
            worker, count, key_to_worker
        );
    }

    ctx.shutdown().await;
}

manual_routing_all_backend_test!(test_min_group_concurrent_distribution, 3910);

async fn test_min_group_sticky_routing_impl(cfg: TestManualConfig, base_port: u16) {
    let config = TestRouterConfig::manual_with_full_options(
        base_port,
        smg::config::ManualAssignmentMode::MinGroup,
        &cfg,
    );
    let ctx = AppTestContext::new_with_config(
        config,
        TestWorkerConfig::slow_workers(base_port + 26000, 3, 200),
    )
    .await;

    let app = ctx.create_app().await;
    let routing_key = "sticky-key-123";

    let mut handles = Vec::new();
    for _ in 0..5 {
        let app_clone = app.clone();
        let key = routing_key.to_string();
        let handle = tokio::spawn(async move { send_request(app_clone, &key).await });
        handles.push(handle);
    }

    let results: Vec<(String, String)> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let workers: Vec<String> = results.into_iter().map(|(_, w)| w).collect();
    let unique_workers: HashSet<&String> = workers.iter().collect();
    assert_eq!(
        unique_workers.len(),
        1,
        "All requests with same routing key should route to same worker, got {:?}",
        unique_workers
    );

    ctx.shutdown().await;
}

manual_routing_all_backend_test!(test_min_group_sticky_routing, 3920);

async fn test_min_group_mixed_concurrent_routing_impl(cfg: TestManualConfig, base_port: u16) {
    let config = TestRouterConfig::manual_with_full_options(
        base_port,
        smg::config::ManualAssignmentMode::MinGroup,
        &cfg,
    );
    let ctx = AppTestContext::new_with_config(
        config,
        TestWorkerConfig::slow_workers(base_port + 26000, 2, 300),
    )
    .await;

    let app = ctx.create_app().await;

    let mut handles = Vec::new();
    for i in 0..4 {
        let routing_key = format!("key-{}", i);
        for _ in 0..3 {
            let app_clone = app.clone();
            let key = routing_key.clone();
            let handle = tokio::spawn(async move { send_request(app_clone, &key).await });
            handles.push(handle);
        }
    }

    let results: Vec<(String, String)> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let mut key_to_workers: HashMap<String, HashSet<String>> = HashMap::new();
    for (key, worker) in results {
        key_to_workers.entry(key).or_default().insert(worker);
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

    let all_workers: HashSet<String> = key_to_workers.values().flatten().cloned().collect();
    assert_eq!(
        all_workers.len(),
        2,
        "Keys should be distributed across both workers"
    );

    ctx.shutdown().await;
}

manual_routing_all_backend_test!(test_min_group_mixed_concurrent_routing, 3930);
