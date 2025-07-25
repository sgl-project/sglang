use actix_web::{test as actix_test, web, App};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::future::join_all;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

mod common;

#[path = "../tests/common/mod.rs"]
mod test_common;

use sglang_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use sglang_router_rs::server::{add_worker, generate, AppState};
use test_common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};

struct PolicyTestConfig {
    num_workers: usize,
    num_requests: usize,
    policy: PolicyConfig,
    worker_delay_ms: u64,
}

async fn setup_router_with_policy(
    config: &PolicyTestConfig,
) -> (Vec<MockWorker>, web::Data<AppState>) {
    let mut workers = Vec::new();
    let mut worker_urls = Vec::new();

    for i in 0..config.num_workers {
        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 21000 + i as u16,
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

    let router_config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![],
        },
        policy: config.policy.clone(),
        host: "127.0.0.1".to_string(),
        port: 3020,
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

async fn run_policy_benchmark(config: PolicyTestConfig) -> (usize, Duration, Vec<usize>) {
    let (mut workers, app_state) = setup_router_with_policy(&config).await;

    let app =
        actix_test::init_service(App::new().app_data(app_state.clone()).service(generate)).await;

    let successful_requests = Arc::new(AtomicUsize::new(0));
    let worker_counters: Arc<Vec<AtomicUsize>> = Arc::new(
        (0..config.num_workers)
            .map(|_| AtomicUsize::new(0))
            .collect(),
    );

    let start = std::time::Instant::now();

    let mut futures = Vec::new();
    for i in 0..config.num_requests {
        let app_ref = &app;
        let successful_ref = successful_requests.clone();
        let worker_counters_ref = worker_counters.clone();

        let future = async move {
            let payload = json!({
                "text": format!("Request {}", i),
                "stream": false,
                "max_new_tokens": 10
            });

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(app_ref, req).await;

            if resp.status().is_success() {
                successful_ref.fetch_add(1, Ordering::Relaxed);
                // Simple worker tracking based on request number
                let worker_idx = i % worker_counters_ref.len();
                worker_counters_ref[worker_idx].fetch_add(1, Ordering::Relaxed);
            }
        };

        futures.push(future);
    }

    join_all(futures).await;

    let elapsed = start.elapsed();
    let total_successful = successful_requests.load(Ordering::Relaxed);

    // Get worker distribution
    let distribution: Vec<usize> = worker_counters
        .iter()
        .map(|c| c.load(Ordering::Relaxed))
        .collect();

    // Cleanup
    for worker in &mut workers {
        worker.stop().await;
    }

    (total_successful, elapsed, distribution)
}

fn benchmark_policies(c: &mut Criterion) {
    use actix_web::rt::System;

    let mut group = c.benchmark_group("routing_policies");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let policies = vec![
        ("random", PolicyConfig::Random),
        ("round_robin", PolicyConfig::RoundRobin),
        (
            "power_of_two",
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            },
        ),
    ];

    for (name, policy) in policies {
        let config = PolicyTestConfig {
            num_workers: 4,
            num_requests: 1000,
            policy,
            worker_delay_ms: 0,
        };

        group.bench_with_input(BenchmarkId::from_parameter(name), &config, |b, config| {
            b.iter(|| {
                System::new()
                    .block_on(async { black_box(run_policy_benchmark(config.clone()).await) })
            });
        });
    }

    group.finish();
}

fn benchmark_worker_scaling(c: &mut Criterion) {
    use actix_web::rt::System;

    let mut group = c.benchmark_group("worker_scaling");
    group.sample_size(10);

    for num_workers in [2, 4, 8].iter() {
        let config = PolicyTestConfig {
            num_workers: *num_workers,
            num_requests: 1000,
            policy: PolicyConfig::RoundRobin,
            worker_delay_ms: 0,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_workers", num_workers)),
            &config,
            |b, config| {
                b.iter(|| {
                    System::new()
                        .block_on(async { black_box(run_policy_benchmark(config.clone()).await) })
                });
            },
        );
    }

    group.finish();
}

// Add Clone implementation for PolicyTestConfig
impl Clone for PolicyTestConfig {
    fn clone(&self) -> Self {
        Self {
            num_workers: self.num_workers,
            num_requests: self.num_requests,
            policy: self.policy.clone(),
            worker_delay_ms: self.worker_delay_ms,
        }
    }
}

criterion_group!(benches, benchmark_policies, benchmark_worker_scaling);
criterion_main!(benches);
