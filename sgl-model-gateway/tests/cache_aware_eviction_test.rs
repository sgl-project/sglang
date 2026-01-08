use std::sync::Arc;

use smg::{
    core::{BasicWorkerBuilder, ModelCard, Worker},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

/// CASE 1: Demonstrates the issue (Before Fix behavior)
/// Reactive eviction is DISABLED. The tree should grow far beyond max_tree_size.
#[tokio::test]
async fn test_case_1_disabled_reactive_eviction_allows_unbounded_growth() {
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        max_tree_size,
        enable_reactive_eviction: false,
        eviction_interval_secs: 3600, // Disable periodic background task
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

    // Burst: Insert 2000 characters (20x the limit)
    let burst_text = "a".repeat(2000);
    policy.select_worker(
        &workers,
        &SelectWorkerInfo {
            request_text: Some(&burst_text),
            ..Default::default()
        },
    );

    let current_size = policy.get_worker_cache_size(model_id, worker_url);

    println!("\n[METRICS - CASE 1 (DISABLED)]");
    println!("Configured Max Size: {}", max_tree_size);
    println!("Actual Tree Size:   {}", current_size);
    println!(
        "Status:             Vulnerable (Over-budget by {}%)",
        (current_size as f32 / max_tree_size as f32) * 100.0
    );

    // Growth is unbounded
    assert!(current_size >= 2000);
}

/// CASE 2: Demonstrates the fix (After Fix behavior)
/// Reactive eviction is ENABLED. The tree should be trimmed immediately during the request.
#[tokio::test]
async fn test_case_2_enabled_reactive_eviction_constrains_growth() {
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        max_tree_size,
        enable_reactive_eviction: true,   // Feature ENABLED
        reactive_eviction_threshold: 1.2, // High-water mark = 120
        eviction_interval_secs: 3600,     // Disable periodic background task
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

    // Burst: Insert 2000 characters (20x the limit)
    let burst_text = "a".repeat(2000);
    policy.select_worker(
        &workers,
        &SelectWorkerInfo {
            request_text: Some(&burst_text),
            ..Default::default()
        },
    );

    let current_size = policy.get_worker_cache_size(model_id, worker_url);

    println!("\n[METRICS - CASE 2 (ENABLED)]");
    println!("Configured Max Size: {}", max_tree_size);
    println!("Actual Tree Size:   {}", current_size);
    println!("Status:             Protected (Synchronous eviction triggered)");

    // Tree size is strictly constrained
    assert!(current_size <= max_tree_size);
}

/// CASE 3: Verifies aggregate growth is capped across multiple workers.
#[tokio::test]
async fn test_case_3_multi_worker_aggregate_growth_is_capped() {
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        max_tree_size,
        enable_reactive_eviction: true,
        reactive_eviction_threshold: 1.2,
        eviction_interval_secs: 3600,
        ..Default::default()
    };

    let policy = CacheAwarePolicy::with_config(config);
    let model_id = "test-model";

    // Setup 4 workers
    let mut workers = Vec::new();
    let urls = vec!["http://w1", "http://w2", "http://w3", "http://w4"];
    for url in &urls {
        workers.push(Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model_id))
                .build(),
        ));
    }
    policy.init_workers(&workers);

    // Each worker tries to take 80 chars (Total = 320, well over the 100 limit)
    // Individually they are under 120, but the AGGREGATE is over.
    for i in 0..4 {
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(&"a".repeat(80)),
                ..Default::default()
            },
        );
    }

    // Sum the total size
    let total_size: usize = urls
        .iter()
        .map(|url| policy.get_worker_cache_size(model_id, url))
        .sum();

    println!("\n[METRICS - CASE 3 (AGGREGATE PROTECTION)]");
    println!("Configured Global Max: {}", max_tree_size);
    println!("Actual Total Size:     {}", total_size);

    // The total must be constrained to the global budget
    assert!(
        total_size <= max_tree_size,
        "Aggregate size {} exceeded global limit",
        total_size
    );
}
