use std::sync::Arc;
use std::time::Instant;
use sgl_model_gateway::core::{
    WorkerManager, WorkerRegistry, BasicWorkerBuilder, WorkerType as CoreWorkerType
};
use reqwest::Client;


#[path = "./common/mod.rs"]
mod common;
use common::mock_worker::{
    MockWorker, MockWorkerConfig, HealthStatus, WorkerType as MockWorkerType
};

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
}use std::sync::Arc;
use std::time::Instant;
// Rename core WorkerType to avoid conflict with MockWorkerType
use sgl_model_gateway::core::{
    WorkerManager, WorkerRegistry, BasicWorkerBuilder, WorkerType as CoreWorkerType
};
use reqwest::Client;

// Import common test utilities
#[path = "./common/mod.rs"]
mod common;
// Rename mock WorkerType
use common::mock_worker::{
    MockWorker, MockWorkerConfig, HealthStatus, WorkerType as MockWorkerType
};

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
