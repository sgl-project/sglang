use std::sync::Arc;
use std::time::Instant;
use http::StatusCode;
use reqwest::Client;
use sgl_model_gateway::core::{
    WorkerManager, WorkerRegistry, BasicWorkerBuilder, WorkerType as CoreWorkerType
};

// Import common test utilities
#[path = "./common/mod.rs"]
mod common;
use common::mock_worker::{
    MockWorker, MockWorkerConfig, HealthStatus, WorkerType as MockWorkerType
};

// Helper to extract body text from Axum response
async fn response_to_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn test_metrics_parallel_latency() {
    println!("\n=== TEST: Parallel Execution Latency ===");
    // Setup: 3 workers, 1s delay each.
    // Serial would take 3s, Parallel should take ~1s.
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

    // Assert Parallelism: Time should be much closer to 1s than 3s
    // We allow a small buffer for overhead, but it must be significantly less than serial sum
    assert!(duration < (delay_ms as f64 / 1000.0) + 0.8, "Requests ran sequentially! Time: {}s", duration);
}

#[tokio::test]
async fn test_metrics_data_aggregation() {
    println!("\n=== TEST: Data Aggregation Correctness ===");
    //  2 workers with different ports. Ensure both show up in the output.
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

        // Extract port to verify in metrics later
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
    println!("Aggregated Metrics Output:\n{}", body_text);

    assert!(!body_text.is_empty(), "Metrics body is empty");

    // Verify both workers are present in the Prometheus output
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

    //  Healthy Worker
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

    //  Failing Worker (Returns 500)
    let mut w2 = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 1.0, // 100% failure rate
    });
    let url2 = w2.start().await.unwrap();
    registry.register(Arc::new(BasicWorkerBuilder::new(&url2).build()));
    workers.push(w2);


    let response = WorkerManager::get_engine_metrics(&registry, &client).await;


    // The gateway should return 200 OK containing partial results, NOT fail completely.
    assert_eq!(response.status(), StatusCode::OK);

    let body_text = response_to_text(response).await;
    println!("Partial Failure Metrics Output:\n{}", body_text);

    // Should contain data from healthy worker
    assert!(body_text.contains(&format!("worker_port=\"{}\"", port1)), "Healthy worker data missing. Full output:\n{}", body_text);

    println!("Metrics returned successfully despite one worker failing.");

    for mut w in workers { w.stop().await; }
}
