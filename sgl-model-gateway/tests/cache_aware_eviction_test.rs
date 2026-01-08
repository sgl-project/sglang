use std::sync::Arc;

use smg::{
    core::{BasicWorkerBuilder, ModelCard, Worker},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

#[tokio::test]
async fn test_reactive_eviction_limits_growth() {
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        max_tree_size,
        enable_reactive_eviction: true,
        reactive_eviction_threshold: 1.2,
        eviction_interval_secs: 3600, // Isolated from periodic task
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

    // Trigger: Insert 1000 characters (10x limit)
    policy.select_worker(
        &workers,
        &SelectWorkerInfo {
            request_text: Some(&"a".repeat(1000)),
            ..Default::default()
        },
    );

    let current_size = policy.get_worker_cache_size(model_id, worker_url);

    // Assert: Reactive logic should have trimmed it back to 100 immediately
    assert!(
        current_size <= 100,
        "Reactive eviction failed: size is {}",
        current_size
    );
}
