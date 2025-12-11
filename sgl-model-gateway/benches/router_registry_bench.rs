use std::{collections::HashMap, sync::Arc};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sgl_model_gateway::core::{
    BasicWorkerBuilder, CircuitBreakerConfig, WorkerRegistry, WorkerType,
};

// Helper to populate registry
fn setup_registry(count: usize) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());

    for i in 0..count {
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "benchmark-model".to_string());

        let worker_type = if i % 2 == 0 {
            WorkerType::Regular
        } else {
            WorkerType::Decode
        };

        let worker = BasicWorkerBuilder::new(format!("http://worker-{}:8000", i))
            .worker_type(worker_type)
            .labels(labels)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build();

        registry.register(Arc::from(worker));
    }
    registry
}

fn bench_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Registry Optimizations");

    // We test with 5000 workers to simulate high load
    let size = 5000;
    let registry = setup_registry(size);

    //  The OLD method (Slow: Allocates vector + Clones ARCs)
    group.bench_function(BenchmarkId::new("Old: get_all()", size), |b| {
        b.iter(|| {
            black_box(registry.get_all());
        });
    });

    //  The NEW method (Fast: O(1) Lookup, Zero Allocation)
    group.bench_function(
        BenchmarkId::new("New: get_worker_distribution()", size),
        |b| {
            b.iter(|| {
                black_box(registry.get_worker_distribution());
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_optimizations);
criterion_main!(benches);
