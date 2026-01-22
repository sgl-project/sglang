use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sgl_model_gateway::core::{worker_registry::HashRing, BasicWorkerBuilder, Worker, WorkerType};

fn bench_hash_ring_new(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_ring");

    // Benchmark for 10, 100, 1000 workers to see scaling behavior
    for num_workers in [10, 100, 1000] {
        let workers: Vec<Arc<dyn Worker>> = (0..num_workers)
            .map(|i| {
                let url = format!("http://worker-{}:8000", i);
                // Create a basic worker with minimal configuration
                let worker = BasicWorkerBuilder::new(&url)
                    .worker_type(WorkerType::Regular)
                    .build();
                Arc::from(worker)
            })
            .collect();

        group.bench_function(format!("new_{}_workers", num_workers), |b| {
            b.iter(|| {
                // Measure the time to create a new HashRing from the worker list.
                // This triggers the 150 virtual nodes creation per worker.
                HashRing::new(black_box(&workers))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hash_ring_new);
criterion_main!(benches);
