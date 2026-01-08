use std::sync::Arc;

use smg::{
    core::{BasicWorkerBuilder, ModelCard, Worker},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};

/// CASE 1: Demonstrates the issue (Reactive eviction DISABLED)
#[tokio::test]
async fn test_case_1_disabled_reactive_eviction_allows_unbounded_growth() {
    let max_tree_size = 100;
    let config = CacheAwareConfig {
        max_tree_size,
        enable_reactive_eviction: false,
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
            request_text: Some(&"a".repeat(2000)),
            ..Default::default()
        },
    );

    let current_size = policy.get_worker_cache_size(model_id, worker_url);
    println!("\n[METRICS - CASE 1 (DISABLED)]");
    println!("Actual Tree Size: {}", current_size);

    assert!(current_size >= 2000);
}

/// CASE 2: Demonstrates the fix (Reactive eviction ENABLED)
#[tokio::test]
async fn test_case_2_enabled_reactive_eviction_constrains_growth() {
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
            request_text: Some(&"a".repeat(2000)),
            ..Default::default()
        },
    );

    let current_size = policy.get_worker_cache_size(model_id, worker_url);
    println!("\n[METRICS - CASE 2 (ENABLED)]");
    println!("Actual Tree Size: {}", current_size);

    assert!(current_size <= max_tree_size);
}

/// CASE 3: Multi-worker scenario verifying aggregate growth is capped
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

    let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
    let urls = vec!["http://w1", "http://w2", "http://w3", "http://w4"];
    for url in &urls {
        workers.push(Arc::new(
            BasicWorkerBuilder::new(*url)
                .model(ModelCard::new(model_id))
                .build(),
        ));
    }
    policy.init_workers(&workers);

    for _ in 0..4 {
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(&"a".repeat(80)),
                ..Default::default()
            },
        );
    }

    let total_size: usize = urls
        .iter()
        .map(|url| policy.get_worker_cache_size(model_id, url))
        .sum();

    println!("\n[METRICS - CASE 3 (AGGREGATE PROTECTION)]");
    println!("Total Aggregate Size: {}", total_size);

    assert!(
        total_size <= max_tree_size,
        "Aggregate size {} exceeded global limit",
        total_size
    );
}
