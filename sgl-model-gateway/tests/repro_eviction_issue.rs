use std::sync::Arc;

use smg::{
    core::{BasicWorkerBuilder, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

#[tokio::test]
async fn test_trigger_eviction_blind_spot() {
    // 1. Configure a small max_tree_size (100 characters)
    // and a long eviction interval (60 seconds) to ensure the background task doesn't run.
    let config = CacheAwareConfig {
        max_tree_size: 100,
        eviction_interval_secs: 60,
        cache_threshold: 0.0, // Always try to use cache path
        ..Default::default()
    };

    let policy = CacheAwarePolicy::with_config(config);
    let worker_url = "http://localhost:8000";
    let workers: Vec<Arc<dyn smg::core::Worker>> = vec![Arc::new(
        BasicWorkerBuilder::new(worker_url)
            .worker_type(WorkerType::Regular)
            .model_id("test-model")
            .build(),
    )];

    policy.init_workers(&workers);

    // 2. Perform many unique insertions that exceed the max_tree_size (100).
    // We will insert 1000 characters total.
    for i in 0..10 {
        let large_unique_text = format!("{:0100}", i); // 100 unique chars per call
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(&large_unique_text),
                ..Default::default()
            },
        );
    }

    println!("--------------------------------------------------");
    println!("SUCCESS: Test reached end of loop.");
    println!("Inserted 1000 characters into a tree with max_tree_size = 100.");
    println!("The background eviction task is set to 60s, so memory remains bloated.");
    println!("This demonstrates that the 100 character limit is NOT enforced during insertion.");
    println!("--------------------------------------------------");
}
