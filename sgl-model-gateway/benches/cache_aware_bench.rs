use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sgl_model_gateway::{
    core::{BasicWorkerBuilder, ModelCard, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

fn bench_cache_aware_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("CacheAware_Scaling");

    // Test multiple cluster sizes
    for n_workers in [50, 100, 500, 1000, 2000].iter() {
        let config = CacheAwareConfig {
            balance_abs_threshold: 0,
            balance_rel_threshold: 0.0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        // Setup N workers
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
        for i in 0..*n_workers {
            let model_card = ModelCard::new("test-model");
            workers.push(Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{}:8000", i))
                    .worker_type(WorkerType::Regular)
                    .model(model_card)
                    .build(),
            ));
        }

        policy.init_workers(&workers);

        let info = SelectWorkerInfo {
            request_text: Some("This is a scaling test prompt."),
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(n_workers), n_workers, |b, _| {
            b.iter(|| {
                let _result = policy.select_worker(black_box(&workers), black_box(&info));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cache_aware_scaling);
criterion_main!(benches);
