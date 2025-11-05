use std::{collections::HashMap, sync::Arc};

use sglang_router_rs::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy},
};

#[test]
fn test_backward_compatibility_with_empty_model_id() {
    let config = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 2,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 0, // Disable background eviction for testing
        max_tree_size: 100,
    };

    let policy = CacheAwarePolicy::with_config(config);

    // Create workers with empty model_id (simulating existing routers)
    let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
        .worker_type(WorkerType::Regular)
        .api_key("test_api_key")
        .build();
    // No model_id label - should default to "unknown"

    let mut labels2 = HashMap::new();
    labels2.insert("model_id".to_string(), "unknown".to_string());
    let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
        .worker_type(WorkerType::Regular)
        .api_key("test_api_key")
        .labels(labels2)
        .build();

    // Add workers - should both go to "default" tree
    policy.add_worker(&worker1);
    policy.add_worker(&worker2);

    // Create worker list
    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1.clone()), Arc::new(worker2.clone())];

    // Select worker - should work without errors
    let selected = policy.select_worker(&workers, Some("test request"));
    assert!(selected.is_some(), "Should select a worker");

    // Remove workers - should work without errors
    policy.remove_worker(&worker1);
    policy.remove_worker(&worker2);
}

#[test]
fn test_mixed_model_ids() {
    let config = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 2,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 0,
        max_tree_size: 100,
    };

    let policy = CacheAwarePolicy::with_config(config);

    // Create workers with different model_id scenarios
    let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
        .worker_type(WorkerType::Regular)
        .api_key("test_api_key")
        .build();
    // No model_id label - defaults to "unknown" which goes to "default" tree

    let mut labels2 = HashMap::new();
    labels2.insert("model_id".to_string(), "llama-3".to_string());
    let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
        .worker_type(WorkerType::Regular)
        .labels(labels2)
        .api_key("test_api_key")
        .build();

    let mut labels3 = HashMap::new();
    labels3.insert("model_id".to_string(), "unknown".to_string());
    let worker3 = BasicWorkerBuilder::new("http://worker3:8080")
        .worker_type(WorkerType::Regular)
        .labels(labels3)
        .build();

    let mut labels4 = HashMap::new();
    labels4.insert("model_id".to_string(), "llama-3".to_string());
    let worker4 = BasicWorkerBuilder::new("http://worker4:8080")
        .worker_type(WorkerType::Regular)
        .labels(labels4)
        .build();

    // Add all workers
    policy.add_worker(&worker1);
    policy.add_worker(&worker2);
    policy.add_worker(&worker3);
    policy.add_worker(&worker4);

    let default_workers: Vec<Arc<dyn Worker>> =
        vec![Arc::new(worker1.clone()), Arc::new(worker3.clone())];
    let selected = policy.select_worker(&default_workers, Some("test request"));
    assert!(selected.is_some(), "Should select from default workers");

    let llama_workers: Vec<Arc<dyn Worker>> =
        vec![Arc::new(worker2.clone()), Arc::new(worker4.clone())];
    let selected = policy.select_worker(&llama_workers, Some("test request"));
    assert!(selected.is_some(), "Should select from llama-3 workers");

    let all_workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(worker1.clone()),
        Arc::new(worker2.clone()),
        Arc::new(worker3.clone()),
        Arc::new(worker4.clone()),
    ];
    let selected = policy.select_worker(&all_workers, Some("test request"));
    assert!(selected.is_some(), "Should select from all workers");
}

#[test]
fn test_remove_worker_by_url_backward_compat() {
    let config = CacheAwareConfig::default();
    let policy = CacheAwarePolicy::with_config(config);

    // Create workers with different model_ids
    let mut labels1 = HashMap::new();
    labels1.insert("model_id".to_string(), "llama-3".to_string());
    let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
        .worker_type(WorkerType::Regular)
        .labels(labels1)
        .api_key("test_api_key")
        .build();

    let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
        .worker_type(WorkerType::Regular)
        .api_key("test_api_key")
        .build();
    // No model_id label - defaults to "unknown"

    // Add workers
    policy.add_worker(&worker1);
    policy.add_worker(&worker2);

    // Remove by URL (backward compatibility method)
    // Should remove from all trees since we don't know the model
    policy.remove_worker_by_url("http://worker1:8080");

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker2.clone())];
    let selected = policy.select_worker(&workers, Some("test"));
    assert_eq!(selected, Some(0), "Should only have worker2 left");
}
