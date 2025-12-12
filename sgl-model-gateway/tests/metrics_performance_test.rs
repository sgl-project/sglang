use std::sync::Arc;
use std::time::Instant;
use http::StatusCode;
use reqwest::Client;
use sgl_model_gateway::core::{
    WorkerManager, WorkerRegistry, BasicWorkerBuilder, WorkerType as CoreWorkerType
};

// Import common test utilities
mod common;
use common::mock_worker::{
    MockWorker, MockWorkerConfig, HealthStatus, WorkerType as MockWorkerType
};

// Helper to extract body text from Axum response
async fn response_to_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// === Requested Reproduction Test ===
#[tokio::test]
async fn test_metrics_performance_waterfall() {
    // Configuration: 2 seconds delay per worker
    let delay_ms = 2000;
    let worker_count = 3;

    let worker_config = MockWorkerConfig {
        port: 0, // Find available port
        // Use the Mock type here
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: delay_ms,
        fail_rate: 0.0,
    };

    println!("-> Starting {} mock workers with {}ms delay each...", worker_count, delay_ms);

    let mut workers = Vec::new();
    let registry = Arc::new(WorkerRegistry::default());
    let client = Client::new();

    // 1. Start Workers and Register them
    for i in 0..worker_count {
        let mut worker = MockWorker::new(worker_config.clone());
        let url = worker.start().await.expect("Failed to start worker");

        // Build the worker instance using the Core type
        let worker_instance = BasicWorkerBuilder::new(&url)
            .worker_type(CoreWorkerType::Regular)
            .build();

        // Use register() instead of add_worker(), and wrap in Arc
        registry.register(Arc::new(worker_instance));

        workers.push(worker);
        println!("   Worker {} started at {}", i+1, url);
    }

    println!("-> Requesting engine metrics (calling Gateway)...");
    let start = Instant::now();

    // 2. Call the function in question
    let _response = WorkerManager::get_engine_metrics(&registry, &client).await;

    let duration = start.elapsed();
    let seconds = duration.as_secs_f64();

    println!("-> Metrics request completed in: {:.2} seconds", seconds);

    // 3. Analysis
    let expected_serial = (delay_ms as f64 / 1000.0) * worker_count as f64;
    let expected_parallel = delay_ms as f64 / 1000.0;

    println!("---------------------------------------------------");
    println!("Expected Time (Serial)   : ~{:.2} seconds (Sum of delays)", expected_serial);
    println!("Expected Time (Parallel) : ~{:.2} seconds (Max of delays)", expected_parallel);
    println!("Actual Time              : {:.2} seconds", seconds);
    println!("---------------------------------------------------");

    // Cleanup
    for mut w in workers {
        w.stop().await;
    }

    // 4. Assertion
    if seconds >= expected_serial - 0.5 {
        println!("RESULT: Confirmed SERIAL execution (Issue Reproduced)");
    } else if seconds <= expected_parallel + 0.5 {
        println!("RESULT: Confirmed PARALLEL execution (Issue Fixed)");
    } else {
        println!("RESULT: Inconclusive timing");
    }
}

// === Existing Verification Tests ===

#[tokio::test]
async fn test_metrics_parallel_latency() {
    println!("\n=== TEST: Parallel Execution Latency (Fast) ===");
    let delay_ms = 1000;
    let worker_count = 3;

    let worker_config = MockWorkerConfig {
        port: 0,
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: delay_ms,
        fail_rate: 0.0,
    };

    let mut workers = Vec::new();
    let registry = Arc::new(WorkerRegistry::default());
    let client = Client::new();

    for _ in 0..worker_count {
        let mut worker = MockWorker::new(worker_config.clone());
        let url = worker.start().await.expect("Failed to start worker");
        let worker_instance = BasicWorkerBuilder::new(&url)
            .worker_type(CoreWorkerType::Regular)
            .build();
        registry.register(Arc::new(worker_instance));
        workers.push(worker);
    }

    let start = Instant::now();
    let response = WorkerManager::get_engine_metrics(&registry, &client).await;
    let duration = start.elapsed().as_secs_f64();

    assert_eq!(response.status(), StatusCode::OK);

    println!("Execution Time: {:.2}s", duration);

    for mut w in workers { w.stop().await; }

    assert!(duration < (delay_ms as f64 / 1000.0) + 0.8, "Requests ran sequentially! Time: {}s", duration);
}

#[tokio::test]
async fn test_metrics_data_aggregation() {
    println!("\n=== TEST: Data Aggregation Correctness ===");
    let worker_config = MockWorkerConfig {
        port: 0,
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    };

    let mut workers = Vec::new();
    let mut ports = Vec::new();
    let registry = Arc::new(WorkerRegistry::default());
    let client = Client::new();

    for _ in 0..2 {
        let mut worker = MockWorker::new(worker_config.clone());
        let url = worker.start().await.expect("Failed to start worker");
        let port = url.split(':').last().unwrap();
        ports.push(port.to_string());

        let worker_instance = BasicWorkerBuilder::new(&url)
            .worker_type(CoreWorkerType::Regular)
            .build();
        registry.register(Arc::new(worker_instance));
        workers.push(worker);
    }

    let response = WorkerManager::get_engine_metrics(&registry, &client).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body_text = response_to_text(response).await;
    assert!(!body_text.is_empty(), "Metrics body is empty");

    for port in ports {
        assert!(body_text.contains(&format!("worker_port=\"{}\"", port)),
                "Missing metrics from worker on port {}. Full output:\n{}", port, body_text);
    }

    for mut w in workers { w.stop().await; }
}

#[tokio::test]
async fn test_metrics_partial_failure_resilience() {
    println!("\n=== TEST: Partial Failure Resilience ===");
    let registry = Arc::new(WorkerRegistry::default());
    let client = Client::new();
    let mut workers = Vec::new();

    // 1. Healthy Worker
    let mut w1 = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let url1 = w1.start().await.unwrap();
    let port1 = url1.split(':').last().unwrap().to_string();
    registry.register(Arc::new(BasicWorkerBuilder::new(&url1).build()));
    workers.push(w1);

    // 2. Failing Worker (Returns 500)
    let mut w2 = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 1.0,
    });
    let url2 = w2.start().await.unwrap();
    registry.register(Arc::new(BasicWorkerBuilder::new(&url2).build()));
    workers.push(w2);

    // 3. Request Metrics
    let response = WorkerManager::get_engine_metrics(&registry, &client).await;

    assert_eq!(response.status(), StatusCode::OK);

    let body_text = response_to_text(response).await;
    assert!(body_text.contains(&format!("worker_port=\"{}\"", port1)), "Healthy worker data missing. Full output:\n{}", body_text);

    println!("Metrics returned successfully despite one worker failing.");

    for mut w in workers { w.stop().await; }
}
