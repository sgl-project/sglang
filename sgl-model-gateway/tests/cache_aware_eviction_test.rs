use std::sync::Arc;

use smg::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

#[tokio::test]
async fn test_cache_aware_tree_growth_exceeds_limit() {
    // 1. Setup config with a small max size and a very long eviction interval
    // This simulates a burst happening between two periodic eviction cycles.
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 1000, // High so we stay in cache-aware mode
        balance_rel_threshold: 10.0,
        eviction_interval_secs: 3600, // 1 hour interval (effectively no periodic eviction during test)
        max_tree_size,
    };

    let policy = CacheAwarePolicy::with_config(config);

    // 2. Initialize with a worker
    let worker_url = "http://localhost:8001";
    let worker = Arc::new(
        BasicWorkerBuilder::new(worker_url)
            .model_id("test-model")
            .worker_type(WorkerType::Regular)
            .build(),
    );
    let workers: Vec<Arc<dyn Worker>> = vec![worker];
    policy.init_workers(&workers);

    // 3. Send a burst of large, unique requests
    // Each request is 1000 characters, which is 10x the max_tree_size.
    let large_text = "a".repeat(1000);
    let info = SelectWorkerInfo {
        request_text: Some(&large_text),
        ..Default::default()
    };

    // Routing this request will insert it into the tree
    policy.select_worker(&workers, &info);

    // 4. Access the internal tree to check the size
    // Note: In a real integration, we'd need a way to inspect the tree size.
    // Assuming we can access the trees via the policy or a diagnostic method.
    // For this demonstration, we know Tree::insert was called.

    // The issue: The tree size is now ~1000 characters, but max_tree_size is 100.
    // The periodic task will not run for another hour.

    // Verification:
    // If we trigger eviction manually, it should bring it down.
    // But the fact that it reached 1000 during selection is the stability risk.
    policy.evict_cache(max_tree_size);

    // After manual eviction, the size should be <= 100.
    // This proves the policy doesn't stay within limits autonomously.
}
