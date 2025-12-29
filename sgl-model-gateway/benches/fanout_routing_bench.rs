use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
// Adjust paths based on your actual library structure
use sgl_model_gateway::{
    app_context::AppContext,
    core::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    routers::{http::router::Router, RouterTrait},
};
use sysinfo::{CpuRefreshKind, RefreshKind, System};
use tokio::runtime::Runtime;
use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};

// --- SCENARIO SETUP ---
// Creates a cluster where:
// - Worker 0 is SLOW (200ms delay) -> Simulates a "Bad Apple"
// - Worker 1 is TARGET (Instant 200 OK) -> Has the data
// - Workers 2..N are FILLERS (404 Not Found)
async fn setup_cluster(size: usize) -> (Arc<Router>, Vec<MockServer>) {
    let registry = Arc::new(WorkerRegistry::new());
    let mut servers = Vec::new();

    // 1. Bad Apple (Timeouts/Hangs)
    let slow_server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(404).set_delay(Duration::from_millis(200)))
        .mount(&slow_server)
        .await;
    registry.register(Arc::new(
        BasicWorkerBuilder::new(&slow_server.uri()).build(),
    ));
    servers.push(slow_server);

    // 2. Target (Success)
    let target_server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status":"ok"})))
        .mount(&target_server)
        .await;
    registry.register(Arc::new(
        BasicWorkerBuilder::new(&target_server.uri()).build(),
    ));
    servers.push(target_server);

    // 3. Fillers (Efficiently mocked using one server)
    if size > 2 {
        let filler_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&filler_server)
            .await;

        // Register multiple workers pointing to the same mock server to save OS resources
        for _ in 2..size {
            let w = BasicWorkerBuilder::new(&filler_server.uri())
                .worker_type(WorkerType::Regular)
                .build();
            registry.register(Arc::new(w));
        }
        servers.push(filler_server);
    }

    let ctx = Arc::new(AppContext::mock(registry, PolicyRegistry::new_default()));
    let router = Router::new(&ctx).await.unwrap();

    (Arc::new(router), servers)
}

// --- CUSTOM RESOURCE MONITOR (CPU & RPS) ---
// Runs a quick load test to print CPU usage to stdout
fn run_resource_monitor(_c: &mut Criterion) {
    println!("\n\n=== RESOURCE USAGE MONITOR ===");
    println!(
        "{0: <15} | {1: <10} | {2: <10} | {3: <10}",
        "Cluster Size", "RPS", "P99 (ms)", "CPU (%)"
    );
    println!("{}", "-".repeat(55));

    let rt = Runtime::new().unwrap();
    let sizes = [10, 100, 1000];

    for &size in &sizes {
        rt.block_on(async {
            let (router, _guards) = setup_cluster(size).await;

            // Warmup
            let _ = router
                .get_response(None, "warmup", &Default::default())
                .await;

            // Setup CPU Monitor
            let mut system = System::new_with_specifics(
                RefreshKind::new().with_cpu(CpuRefreshKind::everything()),
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
            system.refresh_cpu();
            let start_cpu = system.global_cpu_info().cpu_usage();

            // Run Burst (send 50 requests concurrently)
            let start = Instant::now();
            let mut handles = Vec::new();
            for _ in 0..50 {
                let r = router.clone();
                handles.push(tokio::spawn(async move {
                    let t0 = Instant::now();
                    let _ = r.get_response(None, "bench", &Default::default()).await;
                    t0.elapsed().as_millis() as u64
                }));
            }

            // Collect results
            let mut latencies = Vec::new();
            for h in handles {
                latencies.push(h.await.unwrap());
            }
            let duration = start.elapsed();

            // Measure CPU after work
            system.refresh_cpu();
            let end_cpu = system.global_cpu_info().cpu_usage();
            let avg_cpu = (start_cpu + end_cpu) / 2.0;

            // Stats
            let rps = 50.0 / duration.as_secs_f64();
            latencies.sort();
            let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

            println!(
                "{0: <15} | {1: <10.0} | {2: <10} | {3: <10.2}",
                format!("{} Workers", size),
                rps,
                p99,
                avg_cpu
            );
        });
    }
    println!("================================\n");
}

// --- STANDARD CRITERION LATENCY BENCHMARK ---
fn bench_latency(c: &mut Criterion) {
    // Run our custom monitor first to output the table
    run_resource_monitor(c);

    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("fanout_routing_latency");

    // We test these cluster sizes
    let sizes = [10, 100, 500, 1000];

    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.to_async(&rt).iter_with_setup(
                || {
                    // Setup happens OUTSIDE the timing loop
                    rt.block_on(async { setup_cluster(s).await })
                },
                |(router, _guards)| async move {
                    // MEASURED HOT PATH
                    let _ = router
                        .get_response(None, "test-id", &Default::default())
                        .await;
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_latency);
criterion_main!(benches);
