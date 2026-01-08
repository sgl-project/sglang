use std::sync::Arc;

use smg::{
    core::{BasicWorkerBuilder, ModelCard, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

#[tokio::test]
async fn test_reactive_eviction_limits_growth() {
    // 1. Setup config: Max size 100, Reactive threshold 1.2 (High-water = 120)
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        max_tree_size,
        enable_reactive_eviction: true,
        reactive_eviction_threshold: 1.2,
        eviction_interval_secs: 3600, // Disable background task to isolate reactive logic
        ..Default::default()
    };

    let policy = CacheAwarePolicy::with_config(config);
    let model_id = "test-model";
    let worker_url = "http://localhost:8001";

    let worker = Arc::new(
        BasicWorkerBuilder::new(worker_url)
            .model(ModelCard::new(model_id))
            .worker_type(WorkerType::Regular)
            .build(),
    );
    let workers: Vec<Arc<dyn Worker>> = vec![worker];
    policy.init_workers(&workers);

    // 2. TRIGGER: Insert 1000 characters (10x the limit)
    let large_text = "a".repeat(1000);
    let info = SelectWorkerInfo {
        request_text: Some(&large_text),
        ..Default::default()
    };

    // This call now triggers synchronous eviction internally
    policy.select_worker(&workers, &info);

    // 3. VERIFICATION
    let current_size = policy.get_worker_cache_size(model_id, worker_url);

    // The size should have been trimmed back to max_tree_size (100)
    // because 1000 exceeded the high-water mark of 120.
    assert!(
        current_size <= 100,
        "Tree size {} exceeded limit 100 despite reactive eviction",
        current_size
    );

    println!(
        "Success: Reactive eviction trimmed tree from 1000 down to {}",
        current_size
    );
}

#[tokio::test]
async fn test_disabled_reactive_eviction_allows_growth() {
    // Verify that the feature can be turned off (proving the test logic is sound)
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        max_tree_size,
        enable_reactive_eviction: false, // Disabled
        eviction_interval_secs: 3600,
        ..Default::default()
    };

    let policy = CacheAwarePolicy::with_config(config);
    let model_id = "test-model";
    let worker_url = "http://localhost:8001";
    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
        BasicWorkerBuilder::new(worker_url)
            .model(ModelCard::new(model_id))
            .build(),
    )];
    policy.init_workers(&workers);

    policy.select_worker(
        &workers,
        &SelectWorkerInfo {
            request_text: Some(&"a".repeat(1000)),
            ..Default::default()
        },
    );

    let current_size = policy.get_worker_cache_size(model_id, worker_url);

    // Without reactive eviction, it should still be ~1000
    assert!(current_size >= 1000);
    println!(
        "Confirmed: Growth is unbounded when reactive eviction is disabled (Size: {})",
        current_size
    );
}
