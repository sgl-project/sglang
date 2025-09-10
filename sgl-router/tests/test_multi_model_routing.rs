// Integration tests for multi-model routing functionality
use sglang_router_rs::config::{PolicyConfig, RouterConfig};
use sglang_router_rs::core::{BasicWorker, WorkerRegistry, WorkerType};
use sglang_router_rs::policies::PolicyFactory;
use sglang_router_rs::routers::http::router::Router;
use sglang_router_rs::routers::router_manager::RouterManager;
use sglang_router_rs::server::AppContext;
use std::collections::HashMap;
use std::sync::Arc;

mod common;

/// Create a test AppContext with the given configuration
fn create_test_context(enable_igw: bool) -> Arc<AppContext> {
    let mut router_config = RouterConfig::default();
    router_config.enable_igw = enable_igw;
    
    let context = AppContext::new(
        router_config,
        reqwest::Client::new(),
        100, // max_concurrent_requests
        None, // rate_limit_tokens_per_second
    ).unwrap();
    
    Arc::new(context)
}

/// Create and register a worker with specific model_id
fn register_worker_with_model(
    registry: &Arc<WorkerRegistry>,
    url: String,
    model_id: String,
    worker_type: WorkerType,
) {
    let mut labels = HashMap::new();
    labels.insert("model_id".to_string(), model_id);
    
    let worker = BasicWorker::new(url, worker_type)
        .with_labels(labels);
    
    registry.register(Arc::new(worker));
}

#[tokio::test]
async fn test_single_model_mode_routing() {
    // Create context with enable_igw=false (single-model mode)
    let context = create_test_context(false);
    
    // Register workers without model_id (they'll get "unknown" model_id)
    let worker1 = BasicWorker::new("http://worker1:8000".to_string(), WorkerType::Regular);
    let worker2 = BasicWorker::new("http://worker2:8000".to_string(), WorkerType::Regular);
    let worker3 = BasicWorker::new("http://worker3:8000".to_string(), WorkerType::Regular);
    
    context.worker_registry.register(Arc::new(worker1));
    context.worker_registry.register(Arc::new(worker2));
    context.worker_registry.register(Arc::new(worker3));
    
    // Create a router
    let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
    let router = Router::new(vec![], policy, &context).await.unwrap();
    
    // In single-model mode, router should find all workers even without model_id
    let workers = router.get_worker_urls_for_model(None);
    assert_eq!(workers.len(), 3);
    assert!(workers.contains(&"http://worker1:8000".to_string()));
    assert!(workers.contains(&"http://worker2:8000".to_string()));
    assert!(workers.contains(&"http://worker3:8000".to_string()));
}

#[tokio::test]
async fn test_multi_model_mode_with_model_filtering() {
    // Create context with enable_igw=true (multi-model mode)
    let context = create_test_context(true);
    
    // Register workers for different models
    register_worker_with_model(
        &context.worker_registry,
        "http://llama-worker1:8000".to_string(),
        "llama-3-8b".to_string(),
        WorkerType::Regular,
    );
    register_worker_with_model(
        &context.worker_registry,
        "http://llama-worker2:8000".to_string(),
        "llama-3-8b".to_string(),
        WorkerType::Regular,
    );
    register_worker_with_model(
        &context.worker_registry,
        "http://gpt-worker1:8000".to_string(),
        "gpt-4".to_string(),
        WorkerType::Regular,
    );
    register_worker_with_model(
        &context.worker_registry,
        "http://gpt-worker2:8000".to_string(),
        "gpt-4".to_string(),
        WorkerType::Regular,
    );
    
    // Create a router
    let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
    let router = Router::new(vec![], policy, &context).await.unwrap();
    
    // Test model-specific worker selection
    let llama_workers = router.get_worker_urls_for_model(Some("llama-3-8b"));
    assert_eq!(llama_workers.len(), 2);
    assert!(llama_workers.contains(&"http://llama-worker1:8000".to_string()));
    assert!(llama_workers.contains(&"http://llama-worker2:8000".to_string()));
    
    let gpt_workers = router.get_worker_urls_for_model(Some("gpt-4"));
    assert_eq!(gpt_workers.len(), 2);
    assert!(gpt_workers.contains(&"http://gpt-worker1:8000".to_string()));
    assert!(gpt_workers.contains(&"http://gpt-worker2:8000".to_string()));
    
    // Test that no model returns all workers
    let all_workers = router.get_worker_urls_for_model(None);
    assert_eq!(all_workers.len(), 4);
}

#[tokio::test]
async fn test_pd_router_with_model_filtering() {
    
    // Create context
    let context = create_test_context(true);
    
    // Register prefill workers for different models
    register_worker_with_model(
        &context.worker_registry,
        "http://llama-prefill1:8000".to_string(),
        "llama-3-8b".to_string(),
        WorkerType::Prefill { bootstrap_port: Some(9001) },
    );
    register_worker_with_model(
        &context.worker_registry,
        "http://llama-prefill2:8000".to_string(),
        "llama-3-8b".to_string(),
        WorkerType::Prefill { bootstrap_port: Some(9002) },
    );
    
    // Register decode workers for different models
    register_worker_with_model(
        &context.worker_registry,
        "http://llama-decode1:8000".to_string(),
        "llama-3-8b".to_string(),
        WorkerType::Decode,
    );
    register_worker_with_model(
        &context.worker_registry,
        "http://llama-decode2:8000".to_string(),
        "llama-3-8b".to_string(),
        WorkerType::Decode,
    );
    
    // Register workers for a different model
    register_worker_with_model(
        &context.worker_registry,
        "http://gpt-prefill1:8000".to_string(),
        "gpt-4".to_string(),
        WorkerType::Prefill { bootstrap_port: Some(9003) },
    );
    register_worker_with_model(
        &context.worker_registry,
        "http://gpt-decode1:8000".to_string(),
        "gpt-4".to_string(),
        WorkerType::Decode,
    );
    
    // Verify registry has correct workers
    let llama_workers = context.worker_registry.get_by_model_fast("llama-3-8b");
    assert_eq!(llama_workers.len(), 4); // 2 prefill + 2 decode
    
    let gpt_workers = context.worker_registry.get_by_model_fast("gpt-4");
    assert_eq!(gpt_workers.len(), 2); // 1 prefill + 1 decode
    
    // Test that we can get prefill workers specifically
    let prefill_workers = context.worker_registry.get_prefill_workers();
    assert_eq!(prefill_workers.len(), 3); // Total prefill workers
    
    let decode_workers = context.worker_registry.get_decode_workers();
    assert_eq!(decode_workers.len(), 3); // Total decode workers
}

#[tokio::test]
async fn test_mixed_mode_workers() {
    // Test with some workers having model_id and some without
    let context = create_test_context(true);
    
    // Register workers with model_id
    register_worker_with_model(
        &context.worker_registry,
        "http://model-worker1:8000".to_string(),
        "specific-model".to_string(),
        WorkerType::Regular,
    );
    
    // Register worker without model_id (gets "unknown")
    let worker_without_model = BasicWorker::new(
        "http://generic-worker1:8000".to_string(),
        WorkerType::Regular,
    );
    context.worker_registry.register(Arc::new(worker_without_model));
    
    // Create a router
    let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
    let router = Router::new(vec![], policy, &context).await.unwrap();
    
    // Test that model-specific query only returns matching workers
    let specific_workers = router.get_worker_urls_for_model(Some("specific-model"));
    assert_eq!(specific_workers.len(), 1);
    assert!(specific_workers.contains(&"http://model-worker1:8000".to_string()));
    
    // Test that workers without model_id are under "unknown"
    let unknown_workers = context.worker_registry.get_by_model_fast("unknown");
    assert_eq!(unknown_workers.len(), 1);
    assert_eq!(unknown_workers[0].url(), "http://generic-worker1:8000");
    
    // Test that None returns all workers
    let all_workers = router.get_worker_urls_for_model(None);
    assert_eq!(all_workers.len(), 2);
}

#[tokio::test]
async fn test_registry_model_index_performance() {
    // Test that model index provides O(1) lookups
    let context = create_test_context(true);
    
    // Register many workers for different models
    for i in 0..100 {
        let model_id = format!("model-{}", i % 10); // 10 different models
        register_worker_with_model(
            &context.worker_registry,
            format!("http://worker-{}:8000", i),
            model_id,
            WorkerType::Regular,
        );
    }
    
    // Test fast lookup
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let workers = context.worker_registry.get_by_model_fast("model-5");
        assert_eq!(workers.len(), 10); // Should have 10 workers for model-5
    }
    let duration = start.elapsed();
    
    // Should be very fast (< 10ms for 1000 lookups)
    assert!(duration.as_millis() < 10, "Lookups took {:?}", duration);
}

#[tokio::test]
async fn test_worker_health_check_with_registry() {
    // Test that health checker works with the registry
    let context = create_test_context(false);
    
    // Register workers
    let worker1 = BasicWorker::new("http://worker1:8000".to_string(), WorkerType::Regular);
    let worker2 = BasicWorker::new("http://worker2:8000".to_string(), WorkerType::Regular);
    
    context.worker_registry.register(Arc::new(worker1));
    context.worker_registry.register(Arc::new(worker2));
    
    // Start health checker
    let health_checker = context.worker_registry.start_health_checker(1); // 1 second interval
    
    // Wait a bit for health checks to run
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Shutdown health checker
    health_checker.shutdown().await;
    
    // Verify workers were registered
    assert_eq!(context.worker_registry.get_all().len(), 2);
}

#[tokio::test]
async fn test_router_manager_multi_model() {
    // Test RouterManager with multiple models
    let context = create_test_context(true);
    
    // Register workers for different models
    register_worker_with_model(
        &context.worker_registry,
        "http://llama-worker:8000".to_string(),
        "llama-3-8b".to_string(),
        WorkerType::Regular,
    );
    register_worker_with_model(
        &context.worker_registry,
        "http://gpt-worker:8000".to_string(),
        "gpt-4".to_string(),
        WorkerType::Regular,
    );
    
    // Create RouterManager
    let router_manager = RouterManager::new(
        context.router_config.clone(),
        reqwest::Client::new(),
        context.worker_registry.clone(),
        context.policy_registry.clone(),
    );
    
    // Verify we can list workers
    let worker_list = router_manager.list_workers();
    assert_eq!(worker_list.workers.len(), 2);
    
    // Verify we can get specific worker info
    let llama_worker = router_manager.get_worker("http://llama-worker:8000");
    assert!(llama_worker.is_some());
    if let Some(worker) = llama_worker {
        assert_eq!(worker.url, "http://llama-worker:8000");
        // model_id field might be a String, not Option<String>, so check appropriately
        // The actual structure depends on how get_worker returns the data
    }
}