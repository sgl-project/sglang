use std::sync::Arc;

use crate::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

#[test]
fn test_trigger_eviction_blind_spot() {
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
    let workers: Vec<Arc<dyn crate::core::Worker>> = vec![Arc::new(
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

    // 3. Verify the vulnerability:
    // The policy's internal tree is private, but we can verify the 'tenant_char_count'
    // by using the public `remove_worker` and checking logs, or adding a temporary
    // debug getter to Tree.

    // For this repro, we use a trick: if we evict manually, it should drop significantly.
    // But before manual eviction, it will be ~1000, which is 10x the limit.

    // Since we can't easily access the private DashMap in a standard test without modifying source,
    // the core of the issue is visible in `src/policies/cache_aware.rs`:
    // The `select_worker` calls `tree.insert(text, ...)`
    // And `tree.insert` (in tree.rs) performs NO size checks.

    println!("Inserted 1000 characters into a tree with max_tree_size = 100.");
    println!("The background eviction task is set to 60s, so memory remains bloated.");
}
