//! Comprehensive unit tests for Worker trait implementation
//!
//! This test suite covers areas that are not fully tested in the core worker module:
//! - Trait object cloning behavior
//! - Concurrent access to worker state
//! - Error handling edge cases
//! - Worker endpoint configuration

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use sglang_router_rs::core::worker::{WorkerFactory, WorkerType};

#[cfg(test)]
mod worker_trait_tests {
    use super::*;
    #[test]
    fn test_worker_trait_object_cloning() {
        // Test that trait objects can be cloned properly
        let worker = WorkerFactory::create_regular("http://test:8080".to_string());
        let cloned_worker = worker.clone();

        assert_eq!(worker.url(), cloned_worker.url());
        assert_eq!(worker.worker_type(), cloned_worker.worker_type());

        // Verify they share the same load counter (Arc cloning)
        worker.load().store(10, Ordering::Relaxed);
        assert_eq!(cloned_worker.load().load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_worker_concurrent_load_access() {
        let worker = WorkerFactory::create_regular("http://test:8080".to_string());
        let worker_clone = worker.clone();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let w = worker_clone.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        w.load().fetch_add(1, Ordering::Relaxed);
                    }
                    i
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have incremented 10 * 100 = 1000 times
        assert_eq!(worker.load().load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_worker_concurrent_health_updates() {
        let worker = WorkerFactory::create_regular("http://test:8080".to_string());
        let worker_clone = worker.clone();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let w = worker_clone.clone();
                thread::spawn(move || {
                    for j in 0..10 {
                        w.update_health(j % 2 == 0);
                        thread::sleep(Duration::from_millis(1));
                    }
                    i
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should not panic and worker should have some health state
        let _is_healthy = worker.is_healthy();
    }

    #[test]
    fn test_worker_type_endpoint_configuration() {
        let regular_worker = WorkerFactory::create_regular("http://regular:8080".to_string());
        let decode_worker = WorkerFactory::create_decode("http://decode:8080".to_string());
        let prefill_worker =
            WorkerFactory::create_prefill("http://prefill:8080".to_string(), Some(9000));

        // Test that different worker types have the same endpoint configuration
        let regular_endpoints = regular_worker.worker_type().get_endpoints();
        let decode_endpoints = decode_worker.worker_type().get_endpoints();
        let prefill_endpoints = prefill_worker.worker_type().get_endpoints();

        assert_eq!(regular_endpoints.health, "/health");
        assert_eq!(decode_endpoints.health, "/health");
        assert_eq!(prefill_endpoints.health, "/health");

        assert_eq!(regular_endpoints.load, "/get_load");
        assert_eq!(decode_endpoints.load, "/get_load");
        assert_eq!(prefill_endpoints.load, "/get_load");
    }

    #[test]
    fn test_worker_display_and_debug_consistency() {
        let workers = vec![
            WorkerFactory::create_regular("http://regular:8080".to_string()),
            WorkerFactory::create_decode("http://decode:8080".to_string()),
            WorkerFactory::create_prefill("http://prefill:8080".to_string(), Some(9000)),
            WorkerFactory::create_prefill("http://prefill-no-port:8080".to_string(), None),
        ];

        for worker in workers {
            let display_str = format!("{}", worker);
            let debug_str = format!("{:?}", worker);

            // Both should contain the URL
            assert!(display_str.contains(worker.url()));
            assert!(debug_str.contains(worker.url()));

            // Display should be more concise than debug
            assert!(debug_str.len() >= display_str.len());
        }
    }

    #[test]
    fn test_worker_health_check_basic_functionality() {
        // Test basic health check functionality without network calls
        let worker = WorkerFactory::create_regular("http://test:8080".to_string());

        // Initially workers should be unhealthy (no network call has succeeded)
        assert!(!worker.is_healthy());

        // We can manually update health status
        worker.update_health(true);
        assert!(worker.is_healthy());

        worker.update_health(false);
        assert!(!worker.is_healthy());
    }

    #[test]
    fn test_worker_type_hash_and_equality() {
        use std::collections::HashMap;

        let worker_types = vec![
            WorkerType::Regular,
            WorkerType::Decode,
            WorkerType::Prefill(None),
            WorkerType::Prefill(Some(9000)),
            WorkerType::Prefill(Some(9001)),
        ];

        // Test that worker types can be used as HashMap keys
        let mut type_counts = HashMap::new();
        for worker_type in worker_types {
            *type_counts.entry(worker_type.clone()).or_insert(0) += 1;
        }

        assert_eq!(type_counts.len(), 5);
        assert_eq!(type_counts[&WorkerType::Regular], 1);
        assert_eq!(type_counts[&WorkerType::Decode], 1);
        assert_eq!(type_counts[&WorkerType::Prefill(None)], 1);
        assert_eq!(type_counts[&WorkerType::Prefill(Some(9000))], 1);
        assert_eq!(type_counts[&WorkerType::Prefill(Some(9001))], 1);
    }

    #[test]
    fn test_worker_arc_sharing_across_threads() {
        let worker = WorkerFactory::create_regular("http://test:8080".to_string());
        let worker_arc = Arc::new(worker);

        let handles: Vec<_> = (0..5)
            .map(|i| {
                let worker_clone = Arc::clone(&worker_arc);
                thread::spawn(move || {
                    // Each thread increments load and checks health
                    worker_clone.load().fetch_add(i, Ordering::Relaxed);
                    worker_clone.update_health(i % 2 == 0);
                    worker_clone.url().len() // Just to use the worker
                })
            })
            .collect();

        let mut total_url_len = 0;
        for handle in handles {
            total_url_len += handle.join().unwrap();
        }

        // Verify the Arc was shared properly
        assert_eq!(total_url_len, worker_arc.url().len() * 5);

        // Verify load was accumulated: 0 + 1 + 2 + 3 + 4 = 10
        assert_eq!(worker_arc.load().load(Ordering::Relaxed), 10);
    }
}
