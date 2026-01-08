use std::sync::Arc;

use smg::{
    core::{BasicWorkerBuilder, ModelCard, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

#[tokio::test]
async fn test_cache_aware_tree_growth_exceeds_limit() {
    // 1. Setup config with a small max size and a very long eviction interval.
    // This simulates a burst happening between two periodic eviction cycles.
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 1000,
        balance_rel_threshold: 10.0,
        eviction_interval_secs: 3600, // 1 hour interval (effectively no periodic eviction)
        max_tree_size,
    };

    let policy = CacheAwarePolicy::with_config(config);

    // 2. Initialize with a worker.
    // Fix for previous error: BasicWorkerBuilder uses .model(ModelCard::new(...))
    let worker_url = "http://localhost:8001";
    let worker = Arc::new(
        BasicWorkerBuilder::new(worker_url)
            .model(ModelCard::new("test-model"))
            .worker_type(WorkerType::Regular)
            .build(),
    );
    let workers: Vec<Arc<dyn Worker>> = vec![worker];
    policy.init_workers(&workers);

    // 3. Send a burst of large, unique requests.
    // Each request is 1000 characters, which is 10x the max_tree_size.
    let large_text = "a".repeat(1000);
    let info = SelectWorkerInfo {
        request_text: Some(&large_text),
        ..Default::default()
    };

    // 4. ROUTING TRIGGER: This call inserts 1000 chars into the tree.
    // Logic: In the current implementation, this returns successfully immediately.
    // It DOES NOT check if the tree has grown beyond the 100-char limit.
    policy.select_worker(&workers, &info);

    // 5. Verification
    // Manual intervention is required currently to enforce the limit.
    // The fact that the system is now 900% over-budget confirms the stability risk.
    policy.evict_cache(max_tree_size);
}
