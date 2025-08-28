#[cfg(test)]
mod dynamic_capacity_tests {
    use sglang_router_rs::config::{PolicyConfig, RetryConfig, RouterConfig, RoutingMode};
    use sglang_router_rs::core::{
        BasicWorker, CapacityManager, SGLangWorker, TokenBucket, Worker, WorkerType,
    };
    use sglang_router_rs::server::AppContext;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};

    #[tokio::test]
    async fn test_capacity_manager_with_basic_workers() {
        // Create token bucket with initial capacity
        let token_bucket = Arc::new(TokenBucket::new(50, 50));

        // Create capacity manager
        let manager = Arc::new(CapacityManager::new(
            token_bucket.clone(),
            Duration::from_secs(10),
        ));

        // Add some workers
        let worker1 = Arc::new(BasicWorker::new(
            "http://worker1:8080".to_string(),
            WorkerType::Regular,
        ));
        worker1.set_healthy(true);

        let worker2 = Arc::new(BasicWorker::new(
            "http://worker2:8080".to_string(),
            WorkerType::Regular,
        ));
        worker2.set_healthy(true);

        let worker3 = Arc::new(BasicWorker::new(
            "http://worker3:8080".to_string(),
            WorkerType::Regular,
        ));
        worker3.set_healthy(false); // Unhealthy worker

        // Update workers in capacity manager
        let workers: Vec<Arc<dyn Worker>> = vec![
            worker1 as Arc<dyn Worker>,
            worker2 as Arc<dyn Worker>,
            worker3 as Arc<dyn Worker>,
        ];
        manager.update_workers(workers).await;

        // Check that capacity was updated (2 healthy workers * 100 default capacity)
        let (capacity, rate) = token_bucket.get_parameters().await;
        assert_eq!(capacity, 200.0);
        assert_eq!(rate, 200.0);
    }

    #[tokio::test]
    async fn test_capacity_manager_worker_removal() {
        let token_bucket = Arc::new(TokenBucket::new(50, 50));
        let manager = Arc::new(CapacityManager::new(
            token_bucket.clone(),
            Duration::from_secs(10),
        ));

        // Start with two workers
        let worker1 = Arc::new(BasicWorker::new(
            "http://worker1:8080".to_string(),
            WorkerType::Regular,
        ));
        worker1.set_healthy(true);

        let worker2 = Arc::new(BasicWorker::new(
            "http://worker2:8080".to_string(),
            WorkerType::Regular,
        ));
        worker2.set_healthy(true);

        manager
            .update_workers(vec![
                worker1.clone() as Arc<dyn Worker>,
                worker2 as Arc<dyn Worker>,
            ])
            .await;

        // Verify initial capacity
        let (capacity, _) = token_bucket.get_parameters().await;
        assert_eq!(capacity, 200.0);

        // Remove one worker
        manager.remove_worker("http://worker2:8080").await;

        // Verify reduced capacity
        let (capacity, _) = token_bucket.get_parameters().await;
        assert_eq!(capacity, 100.0);
    }

    #[tokio::test]
    async fn test_capacity_manager_health_changes() {
        let token_bucket = Arc::new(TokenBucket::new(50, 50));
        let manager = Arc::new(CapacityManager::new(
            token_bucket.clone(),
            Duration::from_secs(10),
        ));

        let worker1 = Arc::new(BasicWorker::new(
            "http://worker1:8080".to_string(),
            WorkerType::Regular,
        ));
        worker1.set_healthy(true);

        let worker2 = Arc::new(BasicWorker::new(
            "http://worker2:8080".to_string(),
            WorkerType::Regular,
        ));
        worker2.set_healthy(true);

        manager
            .update_workers(vec![
                worker1.clone() as Arc<dyn Worker>,
                worker2.clone() as Arc<dyn Worker>,
            ])
            .await;

        // Initial capacity with both healthy
        let (capacity, _) = token_bucket.get_parameters().await;
        assert_eq!(capacity, 200.0);

        // Make one worker unhealthy
        worker2.set_healthy(false);

        // Trigger recalculation by updating workers
        manager
            .update_workers(vec![worker1 as Arc<dyn Worker>, worker2 as Arc<dyn Worker>])
            .await;

        // Capacity should decrease
        let (capacity, _) = token_bucket.get_parameters().await;
        assert_eq!(capacity, 100.0);
    }

    #[tokio::test]
    async fn test_no_healthy_workers() {
        let token_bucket = Arc::new(TokenBucket::new(50, 50));
        let manager = Arc::new(CapacityManager::new(
            token_bucket.clone(),
            Duration::from_secs(10),
        ));

        let worker = Arc::new(BasicWorker::new(
            "http://worker1:8080".to_string(),
            WorkerType::Regular,
        ));
        worker.set_healthy(false);

        manager
            .update_workers(vec![worker as Arc<dyn Worker>])
            .await;

        // With no healthy workers, capacity should be minimal (1)
        let (capacity, _) = token_bucket.get_parameters().await;
        assert_eq!(capacity, 1.0);
    }

    // Test the core functionality without mocking workers
    #[tokio::test]
    async fn test_token_bucket_dynamic_update() {
        let bucket = TokenBucket::new(100, 100);

        // Initial state
        let (capacity, rate) = bucket.get_parameters().await;
        assert_eq!(capacity, 100.0);
        assert_eq!(rate, 100.0);

        // Update parameters
        bucket.update_parameters(200, 150).await;

        // Verify update
        let (capacity, rate) = bucket.get_parameters().await;
        assert_eq!(capacity, 200.0);
        assert_eq!(rate, 150.0);
    }

    #[tokio::test]
    async fn test_token_bucket_capacity_increase() {
        let bucket = TokenBucket::new(50, 50);

        // Use some tokens
        assert!(bucket.try_acquire(30.0).await.is_ok());

        // Should have ~20 tokens left
        let available = bucket.available_tokens().await;
        assert!((available - 20.0).abs() < 0.1);

        // Increase capacity
        bucket.update_parameters(100, 100).await;

        // Should now have ~70 tokens (20 + 50 from capacity increase)
        let available_after = bucket.available_tokens().await;
        assert!((available_after - 70.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_token_bucket_capacity_decrease() {
        let bucket = TokenBucket::new(100, 100);

        // Start with full bucket
        assert_eq!(bucket.available_tokens().await, 100.0);

        // Decrease capacity
        bucket.update_parameters(50, 50).await;

        // Tokens should be capped at new capacity
        assert_eq!(bucket.available_tokens().await, 50.0);
    }

    #[tokio::test]
    async fn test_token_bucket_zero_rate() {
        let bucket = TokenBucket::new(100, 0);

        // Should default to 1.0 rate
        let (_, rate) = bucket.get_parameters().await;
        assert_eq!(rate, 1.0);

        // Update with zero rate
        bucket.update_parameters(100, 0).await;

        // Should still default to 1.0
        let (_, rate) = bucket.get_parameters().await;
        assert_eq!(rate, 1.0);
    }

    #[tokio::test]
    async fn test_app_context_with_dynamic_capacity() {
        // Test with enabled dynamic capacity
        let config = RouterConfig {
            enable_dynamic_capacity: Some(true),
            capacity_update_interval_secs: Some(30),
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::RoundRobin,
            host: "127.0.0.1".to_string(),
            port: 3001,
            max_payload_size: 1024 * 1024,
            request_timeout_secs: 60,
            worker_startup_timeout_secs: 10,
            worker_startup_check_interval_secs: 1,
            dp_aware: false,
            api_key: None,
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 100,
            queue_size: 0,
            queue_timeout_secs: 60,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: Default::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: Default::default(),
            enable_igw: false,
            rate_limit_tokens_per_second: Some(100),
        };

        let context = AppContext::new(config, reqwest::Client::new(), 100, Some(100));
        assert!(context.capacity_manager.is_some());
    }

    #[tokio::test]
    async fn test_app_context_without_dynamic_capacity() {
        // Test with disabled dynamic capacity
        let config = RouterConfig {
            enable_dynamic_capacity: Some(false),
            capacity_update_interval_secs: Some(30),
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::RoundRobin,
            host: "127.0.0.1".to_string(),
            port: 3002,
            max_payload_size: 1024 * 1024,
            request_timeout_secs: 60,
            worker_startup_timeout_secs: 10,
            worker_startup_check_interval_secs: 1,
            dp_aware: false,
            api_key: None,
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 100,
            queue_size: 0,
            queue_timeout_secs: 60,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: Default::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: Default::default(),
            enable_igw: false,
            rate_limit_tokens_per_second: Some(100),
        };

        let context = AppContext::new(config, reqwest::Client::new(), 100, Some(100));
        assert!(context.capacity_manager.is_none());
    }

    #[tokio::test]
    async fn test_capacity_manager_creation() {
        let bucket = Arc::new(TokenBucket::new(100, 100));
        let _manager = Arc::new(CapacityManager::new(
            bucket.clone(),
            Duration::from_secs(30),
        ));

        // Manager created successfully - basic smoke test
        // Test passes if we reach this point without panicking
    }

    #[tokio::test]
    async fn test_multiple_parameter_updates() {
        let bucket = TokenBucket::new(100, 100);

        // Rapid updates
        for i in 1..=10 {
            bucket.update_parameters(100 * i, 100 * i).await;
        }

        // Final state should be last update
        let (capacity, rate) = bucket.get_parameters().await;
        assert_eq!(capacity, 1000.0);
        assert_eq!(rate, 1000.0);
    }

    #[tokio::test]
    async fn test_token_acquisition_after_update() {
        let bucket = TokenBucket::new(10, 10);

        // Try to acquire more than capacity
        assert!(bucket.try_acquire(20.0).await.is_err());

        // Increase capacity
        bucket.update_parameters(30, 30).await;

        // Now should succeed
        assert!(bucket.try_acquire(20.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_edge_case_zero_initial_capacity() {
        // Edge case: Start with zero capacity
        let bucket = TokenBucket::new(0, 0);

        // Should have minimal functional capacity (1.0 rate)
        let (capacity, rate) = bucket.get_parameters().await;
        assert_eq!(capacity, 0.0);
        assert_eq!(rate, 1.0); // Zero rate defaults to 1.0

        // Cannot acquire any tokens
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Update to functional capacity
        bucket.update_parameters(100, 100).await;

        // Now should work
        assert!(bucket.try_acquire(50.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_edge_case_very_large_capacity() {
        // Edge case: Very large capacity values
        let bucket = TokenBucket::new(1_000_000, 1_000_000);

        // Should handle large values correctly
        let (capacity, rate) = bucket.get_parameters().await;
        assert_eq!(capacity, 1_000_000.0);
        assert_eq!(rate, 1_000_000.0);

        // Can acquire large amounts
        assert!(bucket.try_acquire(500_000.0).await.is_ok());

        // Update to even larger values
        bucket.update_parameters(10_000_000, 10_000_000).await;

        let (capacity, rate) = bucket.get_parameters().await;
        assert_eq!(capacity, 10_000_000.0);
        assert_eq!(rate, 10_000_000.0);
    }

    #[tokio::test]
    async fn test_edge_case_rapid_capacity_changes() {
        // Edge case: Rapid capacity changes
        let bucket = TokenBucket::new(100, 100);

        // Rapidly change capacity
        for i in 0..100 {
            let new_capacity = if i % 2 == 0 { 50 } else { 150 };
            bucket.update_parameters(new_capacity, new_capacity).await;
        }

        // Final state should be stable
        let (capacity, _) = bucket.get_parameters().await;
        assert_eq!(capacity, 150.0);
    }

    #[tokio::test]
    async fn test_edge_case_capacity_with_concurrent_acquisition() {
        // Edge case: Update capacity while tokens are being acquired
        let bucket = Arc::new(TokenBucket::new(100, 100));

        // Start multiple acquisition tasks
        let mut handles = vec![];
        for _ in 0..10 {
            let bucket_clone = bucket.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..10 {
                    let _ = bucket_clone.try_acquire(5.0).await;
                    sleep(Duration::from_millis(1)).await;
                }
            }));
        }

        // Concurrently update capacity
        let bucket_clone = bucket.clone();
        let update_handle = tokio::spawn(async move {
            for i in 1..=5 {
                bucket_clone.update_parameters(100 * i, 100 * i).await;
                sleep(Duration::from_millis(10)).await;
            }
        });

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }
        update_handle.await.unwrap();

        // Should end up with final update values
        let (capacity, rate) = bucket.get_parameters().await;
        assert_eq!(capacity, 500.0);
        assert_eq!(rate, 500.0);
    }

    #[tokio::test]
    async fn test_edge_case_fractional_token_amounts() {
        // Edge case: Fractional token amounts
        let bucket = TokenBucket::new(100, 100);

        // Acquire fractional amounts
        assert!(bucket.try_acquire(33.33).await.is_ok());
        assert!(bucket.try_acquire(33.33).await.is_ok());
        assert!(bucket.try_acquire(33.33).await.is_ok());

        // Should have ~0.01 tokens left
        let available = bucket.available_tokens().await;
        assert!(available < 1.0);
        assert!(available >= 0.0);
    }

    #[tokio::test]
    async fn test_edge_case_return_tokens() {
        // Edge case: Return tokens functionality
        let bucket = TokenBucket::new(100, 100);

        // Use all tokens
        assert!(bucket.try_acquire(100.0).await.is_ok());
        let tokens_after_use = bucket.available_tokens().await;
        assert!(tokens_after_use < 0.1); // Allow small timing variance

        // Return some tokens
        bucket.return_tokens(50.0).await;
        let tokens_after_return = bucket.available_tokens().await;
        assert!((tokens_after_return - 50.0).abs() < 0.1);

        // Return more than capacity
        bucket.return_tokens(100.0).await;
        let tokens_final = bucket.available_tokens().await;
        assert!((tokens_final - 100.0).abs() < 0.1); // Capped at capacity
    }

    #[tokio::test]
    async fn test_edge_case_acquire_with_timeout() {
        // Edge case: Acquire with timeout
        let bucket = TokenBucket::new(10, 1); // 1 token per second refill

        // Use all tokens
        assert!(bucket.try_acquire(10.0).await.is_ok());

        // Try to acquire with short timeout - should fail
        let result = bucket
            .acquire_timeout(5.0, Duration::from_millis(100))
            .await;
        assert!(result.is_err());

        // Wait a bit for refill
        sleep(Duration::from_millis(1500)).await;

        // Try with reasonable timeout - should succeed
        let result = bucket.acquire_timeout(1.0, Duration::from_secs(1)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_edge_case_negative_protection() {
        // Edge case: Ensure no negative tokens
        let bucket = TokenBucket::new(50, 50);

        // Try to acquire more than available
        assert!(bucket.try_acquire(60.0).await.is_err());

        // Available tokens should still be 50
        assert_eq!(bucket.available_tokens().await, 50.0);

        // Decrease capacity below current tokens
        bucket.update_parameters(30, 30).await;

        // Should cap at new capacity
        assert_eq!(bucket.available_tokens().await, 30.0);
    }

    #[tokio::test]
    async fn test_edge_case_refill_rate_calculation() {
        // Edge case: Verify refill rate works correctly
        let bucket = TokenBucket::new(100, 10); // 10 tokens per second

        // Use half the tokens
        assert!(bucket.try_acquire(50.0).await.is_ok());

        // Wait 2 seconds
        sleep(Duration::from_secs(2)).await;

        // Should have refilled ~20 tokens
        let available = bucket.available_tokens().await;
        assert!((available - 70.0).abs() < 1.0); // Allow small timing variance
    }

    // ==========================================================================
    // Integration Tests with Router Configuration
    // ==========================================================================

    #[tokio::test]
    async fn test_router_config_dynamic_capacity_scenarios() {
        // Test various router configurations
        let configs = vec![
            // Config 1: Dynamic capacity enabled with default interval
            (true, None, 100),
            // Config 2: Dynamic capacity enabled with custom interval
            (true, Some(60), 200),
            // Config 3: Dynamic capacity disabled
            (false, Some(30), 150),
            // Config 4: Dynamic capacity with zero initial capacity
            (true, Some(10), 0),
        ];

        for (enabled, interval, initial_capacity) in configs {
            let config = RouterConfig {
                enable_dynamic_capacity: Some(enabled),
                capacity_update_interval_secs: interval,
                mode: RoutingMode::Regular {
                    worker_urls: vec![],
                },
                policy: PolicyConfig::RoundRobin,
                host: "127.0.0.1".to_string(),
                port: 3000,
                max_payload_size: 1024 * 1024,
                request_timeout_secs: 60,
                worker_startup_timeout_secs: 10,
                worker_startup_check_interval_secs: 1,
                dp_aware: false,
                api_key: None,
                discovery: None,
                metrics: None,
                log_dir: None,
                log_level: None,
                request_id_headers: None,
                max_concurrent_requests: initial_capacity,
                queue_size: 0,
                queue_timeout_secs: 60,
                cors_allowed_origins: vec![],
                retry: RetryConfig::default(),
                circuit_breaker: Default::default(),
                disable_retries: false,
                disable_circuit_breaker: false,
                health_check: Default::default(),
                enable_igw: false,
                rate_limit_tokens_per_second: Some(initial_capacity),
            };

            let context = AppContext::new(
                config.clone(),
                reqwest::Client::new(),
                initial_capacity,
                Some(initial_capacity),
            );

            if enabled {
                assert!(context.capacity_manager.is_some());
            } else {
                assert!(context.capacity_manager.is_none());
            }
        }
    }

    #[tokio::test]
    async fn test_sglang_worker_capacity_behavior() {
        // Test SGLangWorker capacity reporting
        let worker = SGLangWorker::new("http://test-worker:8080".to_string(), WorkerType::Regular);

        // Initially no capacity (not fetched yet)
        assert_eq!(worker.capacity(), None);

        // Worker should implement all required traits
        assert_eq!(worker.url(), "http://test-worker:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 0);
    }

    #[tokio::test]
    async fn test_capacity_boundary_conditions() {
        // Test boundary conditions
        let test_cases = vec![
            (usize::MAX, usize::MAX), // Maximum values
            (0, 0),                   // Zero values
            (1, 1),                   // Minimum functional values
            (100, 0),                 // Zero rate with non-zero capacity
            (0, 100),                 // Zero capacity with non-zero rate
        ];

        for (capacity, rate) in test_cases {
            let bucket = TokenBucket::new(capacity, rate);
            let (actual_capacity, actual_rate) = bucket.get_parameters().await;

            // Verify handling of edge values
            assert_eq!(actual_capacity, capacity as f64);
            if rate == 0 {
                assert_eq!(actual_rate, 1.0); // Zero rate defaults to 1.0
            } else {
                assert_eq!(actual_rate, rate as f64);
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_parameter_updates() {
        // Test concurrent parameter updates don't cause issues
        let bucket = Arc::new(TokenBucket::new(100, 100));
        let mut handles = vec![];

        // Spawn multiple tasks updating parameters
        for i in 0..10 {
            let bucket_clone = bucket.clone();
            handles.push(tokio::spawn(async move {
                for j in 0..10 {
                    let capacity = 100 + i * 10 + j;
                    bucket_clone.update_parameters(capacity, capacity).await;
                    sleep(Duration::from_millis(1)).await;
                }
            }));
        }

        // Wait for all updates
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify bucket is still functional
        let (capacity, rate) = bucket.get_parameters().await;
        assert!(capacity > 0.0);
        assert!(rate > 0.0);

        // Should still be able to acquire tokens
        let acquire_amount = capacity / 2.0;
        assert!(bucket.try_acquire(acquire_amount).await.is_ok());
    }

    #[tokio::test]
    async fn test_capacity_update_notification() {
        // Test that capacity updates notify waiters
        let bucket = Arc::new(TokenBucket::new(10, 1));

        // Use all tokens
        assert!(bucket.try_acquire(10.0).await.is_ok());

        // Start a task that waits for tokens
        let bucket_clone = bucket.clone();
        let waiter = tokio::spawn(async move {
            // This would normally wait ~50 seconds with 1 token/sec rate
            bucket_clone.acquire(50.0).await
        });

        // Give the waiter time to start waiting
        sleep(Duration::from_millis(100)).await;

        // Update capacity to make tokens immediately available
        bucket.update_parameters(100, 100).await;

        // The waiter should complete quickly now
        let result = timeout(Duration::from_secs(1), waiter).await;
        assert!(result.is_ok());
        assert!(result.unwrap().unwrap().is_ok());
    }
}
