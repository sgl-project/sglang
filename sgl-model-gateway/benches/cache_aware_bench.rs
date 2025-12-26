use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sgl_model_gateway::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

fn bench_cache_aware_selection(c: &mut Criterion) {
    let config = CacheAwareConfig {
        balance_abs_threshold: 0, // Force the imbalanced path to show log overhead
        ..Default::default()
    };
    let policy = CacheAwarePolicy::with_config(config);

    // Setup 50 workers
    let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
    for i in 0..50 {
        workers.push(Arc::new(
            BasicWorkerBuilder::new(&format!("http://worker-{}:8000", i))
                .worker_type(WorkerType::Regular)
                .build(),
        ));
    }

    policy.init_workers(&workers);
    let prompt = "This is a standard prompt used to test the overhead of string traversal in the selection logic.";
    let info = SelectWorkerInfo { text: Some(prompt) };

    c.bench_function("cache_aware_selection_50_workers", |b| {
        b.iter(|| {
            let _ = policy.select_worker(black_box(&workers), black_box(&info));
        })
    });
}

criterion_group!(benches, bench_cache_aware_selection);
criterion_main!(benches);
