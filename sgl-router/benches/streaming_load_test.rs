use actix_web::{test as actix_test, web, App};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::future::join_all;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// Import common benchmark utilities
mod common;

// Import from the test module
#[path = "../tests/common/mod.rs"]
mod test_common;

use sglang_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use sglang_router_rs::server::{add_worker, generate, AppState};
use test_common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};

/// Configuration for load testing
#[derive(Clone)]
struct LoadTestConfig {
    /// Number of concurrent requests
    concurrent_requests: usize,
    /// Number of mock workers
    num_workers: usize,
    /// Batch size for request submission
    batch_size: usize,
    /// Mock worker response delay in ms
    worker_delay_ms: u64,
    /// Whether to parse response bodies
    parse_responses: bool,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrent_requests: 100,
            num_workers: 4,
            batch_size: 50,
            worker_delay_ms: 0,
            parse_responses: false,
        }
    }
}

async fn setup_test_environment(config: &LoadTestConfig) -> (Vec<MockWorker>, web::Data<AppState>) {
    let mut workers = Vec::new();
    let mut worker_urls = Vec::new();

    // Start mock workers
    for i in 0..config.num_workers {
        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 20000 + i as u16,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: config.worker_delay_ms,
            fail_rate: 0.0,
        });
        let url = worker.start().await.unwrap();
        worker_urls.push(url);
        workers.push(worker);
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create router config
    let router_config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![],
        },
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 3010,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 1,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: None,
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(router_config.request_timeout_secs))
        .build()
        .unwrap();

    let app_state = AppState::new(router_config, client).unwrap();
    let app_state = web::Data::new(app_state);

    // Add workers
    let app =
        actix_test::init_service(App::new().app_data(app_state.clone()).service(add_worker)).await;

    for url in &worker_urls {
        let req = actix_test::TestRequest::post()
            .uri(&format!("/add_worker?url={}", url))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    (workers, app_state)
}

fn run_streaming_load_test(config: LoadTestConfig) -> (usize, Duration) {
    use actix_web::rt::System;

    System::new().block_on(async move { run_streaming_load_test_async(config).await })
}

async fn run_streaming_load_test_async(config: LoadTestConfig) -> (usize, Duration) {
    let (mut workers, app_state) = setup_test_environment(&config).await;

    let app =
        actix_test::init_service(App::new().app_data(app_state.clone()).service(generate)).await;

    let successful_requests = Arc::new(AtomicUsize::new(0));
    let failed_requests = Arc::new(AtomicUsize::new(0));

    let start = std::time::Instant::now();

    // Process requests in batches
    for batch in 0..(config.concurrent_requests / config.batch_size) {
        let mut futures = Vec::new();

        for i in 0..config.batch_size {
            let req_num = batch * config.batch_size + i;
            let app_ref = &app;
            let successful_ref = successful_requests.clone();
            let failed_ref = failed_requests.clone();
            let parse_responses = config.parse_responses;

            let future = async move {
                let payload = json!({
                    "text": format!("Request {}", req_num),
                    "stream": true,
                    "max_new_tokens": 5
                });

                let req = actix_test::TestRequest::post()
                    .uri("/generate")
                    .set_json(&payload)
                    .to_request();

                let resp = actix_test::call_service(app_ref, req).await;

                if resp.status().is_success() {
                    if parse_responses {
                        // Optionally parse the response body
                        let _body = actix_test::read_body(resp).await;
                    }
                    successful_ref.fetch_add(1, Ordering::Relaxed);
                } else {
                    failed_ref.fetch_add(1, Ordering::Relaxed);
                }
            };

            futures.push(future);
        }

        join_all(futures).await;
    }

    let elapsed = start.elapsed();
    let total_successful = successful_requests.load(Ordering::Relaxed);

    // Cleanup
    for worker in &mut workers {
        worker.stop().await;
    }

    (total_successful, elapsed)
}

fn benchmark_streaming_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Test different configurations
    let configs = vec![
        (100, 2, "100_requests_2_workers"),
        (500, 4, "500_requests_4_workers"),
        (1000, 8, "1000_requests_8_workers"),
    ];

    for (requests, workers, name) in configs {
        let config = LoadTestConfig {
            concurrent_requests: requests,
            num_workers: workers,
            batch_size: 50,
            worker_delay_ms: 0,
            parse_responses: false,
        };

        group.throughput(Throughput::Elements(requests as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            b.iter(|| run_streaming_load_test(config.clone()));
        });
    }

    group.finish();
}

fn benchmark_response_parsing_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_parsing_overhead");
    group.sample_size(10);

    let base_config = LoadTestConfig {
        concurrent_requests: 100,
        num_workers: 4,
        batch_size: 20,
        worker_delay_ms: 0,
        parse_responses: false,
    };

    group.bench_function("without_parsing", |b| {
        let config = base_config.clone();
        b.iter(|| run_streaming_load_test(config.clone()));
    });

    let mut parse_config = base_config.clone();
    parse_config.parse_responses = true;

    group.bench_function("with_parsing", |b| {
        let config = parse_config.clone();
        b.iter(|| run_streaming_load_test(config.clone()));
    });

    group.finish();
}

fn benchmark_worker_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_scaling");
    group.sample_size(10);

    for num_workers in [1, 2, 4, 8].iter() {
        let config = LoadTestConfig {
            concurrent_requests: 200,
            num_workers: *num_workers,
            batch_size: 25,
            worker_delay_ms: 0,
            parse_responses: false,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_workers", num_workers)),
            &config,
            |b, config| {
                b.iter(|| run_streaming_load_test(config.clone()));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_streaming_throughput,
    benchmark_response_parsing_overhead,
    benchmark_worker_scaling
);
criterion_main!(benches);
