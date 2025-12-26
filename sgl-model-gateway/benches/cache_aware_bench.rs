use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sgl_model_gateway::policies::{CacheAwarePolicy, CacheAwareConfig, LoadBalancingPolicy, SelectWorkerInfo};
use sgl_model_gateway::core::{Worker, BasicWorkerBuilder, WorkerType, ModelCard};
use std::sync::Arc;

fn bench_cache_aware_selection(c: &mut Criterion) {
    // 1. Setup Configuration
    // Set balance_abs_threshold to 0 to ensure the path with logging overhead
    // is exercised during the benchmark.
    let config = CacheAwareConfig {
        balance_abs_threshold: 0,
        ..Default::default()
    };
    let policy = CacheAwarePolicy::with_config(config);

    // 2. Setup 50 Mock Workers
    // Using ModelCard to assign models as required by the latest builder API
    let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
    for i in 0..50 {
        let model_card = ModelCard::new("test-model");

        workers.push(Arc::new(
            BasicWorkerBuilder::new(&format!("http://worker-{}:8000", i))
                .worker_type(WorkerType::Regular)
                .model(model_card) // Correct method
                .build(),
        ));
    }

    // Initialize policy state
    policy.init_workers(&workers);

    // 3. Prepare Selection Info
    // Use 'request_text' as defined in SelectWorkerInfo
    let prompt = "This is a standard prompt used to test the overhead of string traversal and heap allocations.";
    let info = SelectWorkerInfo {
        request_text: Some(prompt),
        headers: None,
    };

    let mut group = c.benchmark_group("LoadBalancer");

    group.bench_function("cache_aware_selection_50_workers", |b| {
        b.iter(|| {
            // Measure the performance of the selection decision
            let _Result = policy.select_worker(
                black_box(&workers),
                black_box(&info)
            );
        })
    });

    group.finish();
}

criterion_group!(benches, bench_cache_aware_selection);
criterion_main!(benches);
