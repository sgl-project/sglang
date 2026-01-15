use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smg::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    mesh::{stores::StateStores, sync::MeshSyncManager},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

fn bench_mesh_sync_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("MeshSync_Bottleneck_Investigation");

    // Setup a single worker to isolate the synchronization overhead
    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
        BasicWorkerBuilder::new("http://worker-1:8000")
            .worker_type(WorkerType::Regular)
            .model("test-model")
            .build(),
    )];

    // We use a real StateStore and MeshSyncManager to simulate the current production path
    let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
    let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

    let config = CacheAwareConfig::default();
    let mut policy = CacheAwarePolicy::with_config(config);
    policy.init_workers(&workers);
    policy.set_mesh_sync(Some(mesh_sync));

    // Each iteration uses a unique string to ensure a new 'Insert' operation is triggered
    group.bench_function("select_worker_WITH_mesh_enabled_baseline", |b| {
        let mut iteration = 0;
        b.iter(|| {
            iteration += 1;
            let text = format!("unique_request_prefix_string_{}", iteration);
            let info = SelectWorkerInfo {
                request_text: Some(&text),
                ..Default::default()
            };
            // This calls sync_tree_operation internally in the current code
            policy.select_worker(&workers, &info);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_mesh_sync_overhead);
criterion_main!(benches);
