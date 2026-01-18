use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smg::{
    core::{BasicWorkerBuilder, ModelCard, Worker, WorkerType},
    mesh::{stores::StateStores, sync::MeshSyncManager},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

fn bench_mesh_sync_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("MeshSync_Bottleneck_Investigation");

    let model_id = "test-model";
    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
        BasicWorkerBuilder::new("http://worker-1:8000")
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new(model_id))
            .build(),
    )];

    let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
    let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

    let config = CacheAwareConfig::default();
    let mut policy = CacheAwarePolicy::with_config(config);
    policy.init_workers(&workers);
    policy.set_mesh_sync(Some(mesh_sync));

    group.bench_function("select_worker_WITH_mesh_enabled", |b| {
        let mut iteration = 0;
        b.iter(|| {
            iteration += 1;
            // Use unique strings to force new 'Insert' operations and trigger sync
            let text = format!("unique_request_prefix_{}", iteration);
            let info = SelectWorkerInfo {
                request_text: Some(black_box(&text)),
                ..Default::default()
            };

            // This synchronously calls sync_tree_operation in the current code
            policy.select_worker(&workers, &info);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_mesh_sync_overhead);
criterion_main!(benches);
