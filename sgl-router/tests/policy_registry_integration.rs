//! Integration tests for PolicyRegistry with RouterManager

use std::{collections::HashMap, sync::Arc};

use sglang_router_rs::{
    config::PolicyConfig, core::WorkerRegistry, policies::PolicyRegistry,
    protocols::worker_spec::WorkerConfigRequest, routers::router_manager::RouterManager,
};

#[tokio::test]
async fn test_policy_registry_with_router_manager() {
    // Create HTTP client
    let _client = reqwest::Client::new();

    // Create shared registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));

    // Create RouterManager with shared registries
    let _router_manager = RouterManager::new(worker_registry.clone());

    // Add first worker for llama-3 with cache_aware policy hint
    let mut labels1 = HashMap::new();
    labels1.insert("policy".to_string(), "cache_aware".to_string());

    let _worker1_config = WorkerConfigRequest {
        url: "http://worker1:8000".to_string(),
        model_id: Some("llama-3".to_string()),
        api_key: Some("test_api_key".to_string()),
        worker_type: None,
        priority: None,
        cost: None,
        labels: labels1,
        bootstrap_port: None,
        tokenizer_path: None,
        reasoning_parser: None,
        tool_parser: None,
        chat_template: None,
        health_check_timeout_secs: 30,
        health_check_interval_secs: 60,
        health_success_threshold: 2,
        health_failure_threshold: 3,
        max_connection_attempts: 20,
        dp_aware: false,
    };

    // This would normally connect to a real worker, but for testing we'll just verify the structure
    // In a real test, we'd need to mock the worker or use a test server

    let _llama_policy = policy_registry.get_policy("llama-3");
    // After first worker is added, llama-3 should have a policy

    // Add second worker for llama-3 with different policy hint (should be ignored)
    let mut labels2 = HashMap::new();
    labels2.insert("policy".to_string(), "random".to_string());

    let _worker2_config = WorkerConfigRequest {
        url: "http://worker2:8000".to_string(),
        model_id: Some("llama-3".to_string()),
        api_key: Some("test_api_key".to_string()),
        worker_type: None,
        priority: None,
        cost: None,
        labels: labels2,
        bootstrap_port: None,
        tokenizer_path: None,
        reasoning_parser: None,
        tool_parser: None,
        chat_template: None,
        health_check_timeout_secs: 30,
        health_check_interval_secs: 60,
        health_success_threshold: 2,
        health_failure_threshold: 3,
        max_connection_attempts: 20,
        dp_aware: false,
    };

    // The second worker should use the same policy as the first (cache_aware)

    // Add worker for different model (gpt-4) with random policy
    let mut labels3 = HashMap::new();
    labels3.insert("policy".to_string(), "random".to_string());

    let _worker3_config = WorkerConfigRequest {
        url: "http://worker3:8000".to_string(),
        model_id: Some("gpt-4".to_string()),
        api_key: Some("test_api_key".to_string()),
        worker_type: None,
        priority: None,
        cost: None,
        labels: labels3,
        bootstrap_port: None,
        tokenizer_path: None,
        reasoning_parser: None,
        tool_parser: None,
        chat_template: None,
        health_check_timeout_secs: 30,
        health_check_interval_secs: 60,
        health_success_threshold: 2,
        health_failure_threshold: 3,
        max_connection_attempts: 20,
        dp_aware: false,
    };

    let _gpt_policy = policy_registry.get_policy("gpt-4");

    // When we remove both llama-3 workers, the policy should be cleaned up

    println!("PolicyRegistry integration test structure created");
    println!("Note: This test requires mocking or test servers to fully execute");
}

#[test]
fn test_policy_registry_cleanup() {
    use sglang_router_rs::{config::PolicyConfig, policies::PolicyRegistry};

    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

    // Add workers for a model
    let policy1 = registry.on_worker_added("model-1", Some("cache_aware"));
    assert_eq!(policy1.name(), "cache_aware");

    // Second worker uses existing policy
    let policy2 = registry.on_worker_added("model-1", Some("random"));
    assert_eq!(policy2.name(), "cache_aware"); // Should still be cache_aware

    assert!(registry.get_policy("model-1").is_some());

    // Remove first worker - policy should remain
    registry.on_worker_removed("model-1");
    assert!(registry.get_policy("model-1").is_some());

    // Remove second worker - policy should be cleaned up
    registry.on_worker_removed("model-1");
    assert!(registry.get_policy("model-1").is_none());

    println!("✓ PolicyRegistry cleanup test passed");
}

#[test]
fn test_policy_registry_multiple_models() {
    use sglang_router_rs::{config::PolicyConfig, policies::PolicyRegistry};

    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

    // Add workers for different models with different policies
    let llama_policy = registry.on_worker_added("llama-3", Some("cache_aware"));
    let gpt_policy = registry.on_worker_added("gpt-4", Some("random"));
    let mistral_policy = registry.on_worker_added("mistral", None); // Uses default

    assert_eq!(llama_policy.name(), "cache_aware");
    assert_eq!(gpt_policy.name(), "random");
    assert_eq!(mistral_policy.name(), "round_robin"); // Default

    assert!(registry.get_policy("llama-3").is_some());
    assert!(registry.get_policy("gpt-4").is_some());
    assert!(registry.get_policy("mistral").is_some());

    // Get all mappings
    let mappings = registry.get_all_mappings();
    assert_eq!(mappings.len(), 3);
    assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
    assert_eq!(mappings.get("gpt-4").unwrap(), "random");
    assert_eq!(mappings.get("mistral").unwrap(), "round_robin");

    println!("✓ PolicyRegistry multiple models test passed");
}
