//! Comprehensive integration tests for health checking functionality
//!
//! This test suite covers realistic health check scenarios including:
//! - Network failure recovery
//! - Health check retry mechanisms
//! - Timeout and error handling
//! - Real HTTP server interactions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sglang_router_rs::core::worker::{Worker, WorkerFactory};
use sglang_router_rs::test_utils::mock_servers;

#[cfg(test)]
mod health_check_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_network_failure_recovery() {
        // Test scenario: Multiple workers showing different health states
        let unhealthy_server = mock_servers::create_enhanced_mock_health_server(
            vec![(503, r#"{"error": "Service Unavailable"}"#.to_string())],
            vec![Duration::from_millis(10)],
            None,
        )
        .await;

        let healthy_server = mock_servers::create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(10)],
            None,
        )
        .await;

        let unhealthy_worker = WorkerFactory::create_regular(unhealthy_server.0);
        let healthy_worker = WorkerFactory::create_regular(healthy_server.0);

        // Test unhealthy worker
        let result1 = unhealthy_worker.check_health().await;
        assert!(result1.is_err());
        assert!(!unhealthy_worker.is_healthy());

        // Test healthy worker
        let result2 = healthy_worker.check_health().await;
        assert!(result2.is_ok());
        assert!(healthy_worker.is_healthy());

        // Subsequent checks on healthy worker will use cache
        let result3 = healthy_worker.check_health().await;
        assert!(result3.is_ok());
        assert!(healthy_worker.is_healthy());
    }

    #[tokio::test]
    async fn test_health_check_different_response_codes() {
        // Test different HTTP status codes result in appropriate health states
        let test_cases = vec![
            (200, true),  // Healthy
            (503, false), // Service Unavailable
            (500, false), // Internal Server Error
            (404, false), // Not Found
        ];

        for (status_code, expected_healthy) in test_cases {
            let (server_url, _) = mock_servers::create_enhanced_mock_health_server(
                vec![(status_code, r#"{"status": "test"}"#.to_string())],
                vec![Duration::from_millis(5)],
                Some(1),
            )
            .await;

            let worker = WorkerFactory::create_regular(server_url);
            let result = worker.check_health().await;

            if expected_healthy {
                assert!(
                    result.is_ok(),
                    "Status code {} should be healthy",
                    status_code
                );
                assert!(worker.is_healthy());
            } else {
                assert!(
                    result.is_err(),
                    "Status code {} should be unhealthy",
                    status_code
                );
                assert!(!worker.is_healthy());
            }
        }
    }

    #[tokio::test]
    async fn test_health_check_with_delays() {
        // Test health checks work with server response delays
        // Note: Due to caching, only the first request will have the delay
        let (server_url, call_count) = mock_servers::create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(100)], // 100ms delay
            None,
        )
        .await;

        let worker = WorkerFactory::create_regular(server_url);

        let start = std::time::Instant::now();
        let result = worker.check_health().await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Health check should succeed despite delay");
        assert!(worker.is_healthy());
        assert!(
            elapsed >= Duration::from_millis(80),
            "Should take at least 80ms with delay"
        );

        // Second call should be much faster due to caching
        let start2 = std::time::Instant::now();
        let result2 = worker.check_health().await;
        let elapsed2 = start2.elapsed();

        assert!(result2.is_ok(), "Cached health check should succeed");
        assert!(
            elapsed2 < Duration::from_millis(50),
            "Cached call should be fast"
        );
        assert_eq!(
            call_count.load(Ordering::Relaxed),
            1,
            "Should only make one actual HTTP call"
        );
    }

    #[tokio::test]
    async fn test_manual_wait_for_healthy_workers() {
        // Test manual waiting for multiple workers to become healthy (without using private utils)
        let healthy_responses = vec![(200, r#"{"status": "healthy"}"#.to_string())];
        let unhealthy_responses = vec![(503, r#"{"error": "Not ready"}"#.to_string())];

        // Create one healthy server and one persistently unhealthy server
        let (healthy_server_url, _) = mock_servers::create_enhanced_mock_health_server(
            healthy_responses,
            vec![Duration::from_millis(10)],
            None,
        )
        .await;

        let (unhealthy_server_url, _) = mock_servers::create_enhanced_mock_health_server(
            unhealthy_responses,
            vec![Duration::from_millis(10)],
            None, // No limit - will always be unhealthy
        )
        .await;

        let workers = vec![
            WorkerFactory::create_regular(healthy_server_url),
            WorkerFactory::create_regular(unhealthy_server_url),
        ];

        // Check workers individually - should get mixed results
        let results = vec![
            workers[0].check_health().await,
            workers[1].check_health().await,
        ];

        assert!(results[0].is_ok(), "First worker should be healthy");
        assert!(results[1].is_err(), "Second worker should be unhealthy");
    }

    #[tokio::test]
    async fn test_health_check_with_different_worker_types() {
        // Test health checking works consistently across different worker types
        let health_response = vec![(200, r#"{"status": "healthy"}"#.to_string())];

        let (regular_server_url, _) = mock_servers::create_enhanced_mock_health_server(
            health_response.clone(),
            vec![Duration::from_millis(10)],
            None,
        )
        .await;

        let (decode_server_url, _) = mock_servers::create_enhanced_mock_health_server(
            health_response.clone(),
            vec![Duration::from_millis(10)],
            None,
        )
        .await;

        let (prefill_server_url, _) = mock_servers::create_enhanced_mock_health_server(
            health_response,
            vec![Duration::from_millis(10)],
            None,
        )
        .await;

        let workers: Vec<Arc<dyn Worker>> = vec![
            WorkerFactory::create_regular(regular_server_url),
            WorkerFactory::create_decode(decode_server_url),
            WorkerFactory::create_prefill(prefill_server_url, Some(9000)),
        ];

        // All workers should pass health checks
        for (i, worker) in workers.iter().enumerate() {
            let result = worker.check_health().await;
            assert!(result.is_ok(), "Worker {} health check should succeed", i);
            assert!(worker.is_healthy(), "Worker {} should be healthy", i);
        }
    }

    #[tokio::test]
    async fn test_health_check_concurrent_access() {
        // Test concurrent health checks on the same worker
        let (server_url, call_count) = mock_servers::create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(50)], // Introduce some delay
            None,
        )
        .await;

        let worker = WorkerFactory::create_regular(server_url);
        let worker_arc = Arc::new(worker);

        // Launch multiple concurrent health checks
        let handles = (0..5)
            .map(|_| {
                let w = Arc::clone(&worker_arc);
                tokio::spawn(async move { w.check_health().await })
            })
            .collect::<Vec<_>>();

        // Wait for all to complete
        let results = futures::future::join_all(handles).await;

        // All should succeed
        for (i, result) in results.into_iter().enumerate() {
            let health_result = result.unwrap();
            assert!(
                health_result.is_ok(),
                "Concurrent health check {} should succeed",
                i
            );
        }

        // Due to caching, we might not get exactly 5 calls
        let total_calls = call_count.load(Ordering::Relaxed);
        assert!(
            total_calls >= 1,
            "Should have made at least 1 health check call"
        );
        assert!(
            total_calls <= 5,
            "Should not have made more than 5 health check calls"
        );
    }

    #[tokio::test]
    async fn test_health_check_cache_behavior() {
        // Test health check caching behavior - workers use default 30s cache TTL
        let (server_url, call_count) = mock_servers::create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(10)],
            None,
        )
        .await;

        let worker = WorkerFactory::create_regular(server_url);

        // First health check should make a call
        let result1 = worker.check_health().await;
        assert!(result1.is_ok());
        assert_eq!(call_count.load(Ordering::Relaxed), 1);

        // Immediate second check should use cache (within 30s default TTL)
        let result2 = worker.check_health().await;
        assert!(result2.is_ok());
        assert_eq!(call_count.load(Ordering::Relaxed), 1); // No new call due to caching

        // The cache TTL is 30s by default, so we can't easily test expiration
        // in integration tests without waiting too long. This test verifies
        // the caching mechanism works for immediate consecutive calls.
        assert!(worker.is_healthy());
    }

    #[tokio::test]
    async fn test_health_check_load_balancer_simulation() {
        // Simulate a load balancer scenario with multiple workers
        let worker_configs = vec![
            (200, r#"{"status": "healthy", "load": 10}"#),
            (200, r#"{"status": "healthy", "load": 25}"#),
            (503, r#"{"error": "overloaded"}"#),
            (200, r#"{"status": "healthy", "load": 5}"#),
        ];

        let mut workers = Vec::new();
        for (status, response) in worker_configs {
            let (server_url, _) = mock_servers::create_enhanced_mock_health_server(
                vec![(status, response.to_string())],
                vec![Duration::from_millis(20)],
                None,
            )
            .await;
            workers.push(WorkerFactory::create_regular(server_url));
        }

        // Check health of all workers
        let mut healthy_workers = Vec::new();
        for (i, worker) in workers.iter().enumerate() {
            let result = worker.check_health().await;
            if result.is_ok() {
                healthy_workers.push(i);
            }
        }

        // Should have 3 healthy workers (indices 0, 1, 3)
        assert_eq!(healthy_workers, vec![0, 1, 3]);
    }

    #[tokio::test]
    async fn test_health_check_retry_after_connection_refused() {
        // Test what happens when connection is refused (server not running)
        let worker = WorkerFactory::create_regular("http://localhost:0".to_string()); // Invalid port

        let result = worker.check_health().await;
        assert!(result.is_err());
        assert!(!worker.is_healthy());

        // Multiple retries should continue to fail
        for _ in 0..3 {
            let retry_result = worker.check_health().await;
            assert!(retry_result.is_err());
            assert!(!worker.is_healthy());
        }
    }
}
