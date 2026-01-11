//! Integration tests for ManualPolicy with Redis backend
//!
//! These tests require a running Redis server. Use RedisTestServer to start one.

use std::{collections::HashMap, sync::Arc};

use smg::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{LoadBalancingPolicy, ManualConfig, ManualPolicy, SelectWorkerInfo},
};

use crate::common::redis_test_server::RedisTestServer;

fn create_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
    urls.iter()
        .map(|url| {
            Arc::new(
                BasicWorkerBuilder::new(*url)
                    .worker_type(WorkerType::Regular)
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

fn headers_with_routing_key(key: &str) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-routing-key", key.parse().unwrap());
    headers
}

fn create_redis_policy(redis_url: &str) -> ManualPolicy {
    let config = ManualConfig {
        redis_url: Some(redis_url.to_string()),
        ..Default::default()
    };
    ManualPolicy::with_config(config)
}

// ============================================================================
// Basic Flow Tests
// ============================================================================

#[tokio::test]
async fn test_redis_vacant_to_occupied_hit() {
    let server = RedisTestServer::start().await.unwrap();
    let policy = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("user-123");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_idx = policy.select_worker(&workers, &info).await.unwrap();

    for _ in 0..5 {
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(
            idx, first_idx,
            "Same routing key should route to same worker"
        );
    }
}

#[tokio::test]
async fn test_redis_different_routing_ids_distribute() {
    let server = RedisTestServer::start().await.unwrap();
    let policy = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    let mut distribution: HashMap<usize, usize> = HashMap::new();
    for i in 0..30 {
        let headers = headers_with_routing_key(&format!("user-{}", i));
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        *distribution.entry(idx).or_insert(0) += 1;
    }

    assert!(
        distribution.len() > 1,
        "Should distribute across multiple workers"
    );
}

// ============================================================================
// Failover Tests
// ============================================================================

#[tokio::test]
async fn test_redis_failover_to_second_candidate() {
    let server = RedisTestServer::start().await.unwrap();
    let policy = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("failover-test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_idx = policy.select_worker(&workers, &info).await.unwrap();
    workers[first_idx].set_healthy(false);

    let second_idx = policy.select_worker(&workers, &info).await.unwrap();
    assert_ne!(second_idx, first_idx, "Should failover to different worker");

    for _ in 0..3 {
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, second_idx, "Should stick to new worker");
    }
}

#[tokio::test]
async fn test_redis_worker_recovery() {
    let server = RedisTestServer::start().await.unwrap();
    let policy = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("recovery-test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_idx = policy.select_worker(&workers, &info).await.unwrap();
    workers[first_idx].set_healthy(false);

    let second_idx = policy.select_worker(&workers, &info).await.unwrap();
    assert_ne!(second_idx, first_idx);

    workers[first_idx].set_healthy(true);

    let after_recovery = policy.select_worker(&workers, &info).await.unwrap();
    assert_eq!(
        after_recovery, first_idx,
        "Should return to original worker (first in candidate list)"
    );
}

#[tokio::test]
async fn test_redis_both_candidates_unhealthy() {
    let server = RedisTestServer::start().await.unwrap();
    let policy = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    let headers = headers_with_routing_key("both-unhealthy");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_idx = policy.select_worker(&workers, &info).await.unwrap();
    workers[first_idx].set_healthy(false);

    let second_idx = policy.select_worker(&workers, &info).await.unwrap();
    workers[second_idx].set_healthy(false);

    let third_idx = policy.select_worker(&workers, &info).await.unwrap();
    assert!(
        third_idx != first_idx && third_idx != second_idx,
        "Should select a new healthy worker"
    );
}

// ============================================================================
// Multi-Instance Consistency Tests
// ============================================================================

#[tokio::test]
async fn test_redis_multi_instance_consistency() {
    let server = RedisTestServer::start().await.unwrap();
    let policy1 = create_redis_policy(server.url());
    let policy2 = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("shared-key");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let idx_from_policy1 = policy1.select_worker(&workers, &info).await.unwrap();
    let idx_from_policy2 = policy2.select_worker(&workers, &info).await.unwrap();

    assert_eq!(
        idx_from_policy1, idx_from_policy2,
        "Both policy instances should route to same worker via shared Redis state"
    );
}

#[tokio::test]
async fn test_redis_cross_instance_failover() {
    let server = RedisTestServer::start().await.unwrap();
    let policy1 = create_redis_policy(server.url());
    let policy2 = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("cross-failover");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_idx = policy1.select_worker(&workers, &info).await.unwrap();
    workers[first_idx].set_healthy(false);

    let second_idx = policy2.select_worker(&workers, &info).await.unwrap();
    assert_ne!(second_idx, first_idx, "Policy2 should failover");

    let idx_from_policy1 = policy1.select_worker(&workers, &info).await.unwrap();
    assert_eq!(
        idx_from_policy1, second_idx,
        "Policy1 should see the failover done by policy2"
    );
}

// ============================================================================
// CAS Race Tests
// ============================================================================

#[tokio::test]
async fn test_redis_concurrent_same_key() {
    let server = RedisTestServer::start().await.unwrap();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
    let workers = Arc::new(workers);

    let mut handles = Vec::new();
    for _ in 0..10 {
        let redis_url = server.url().to_string();
        let workers_clone = Arc::clone(&workers);
        let handle = tokio::spawn(async move {
            let policy = create_redis_policy(&redis_url);

            let headers = headers_with_routing_key("concurrent-key");
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            policy.select_worker(&workers_clone, &info).await
        });
        handles.push(handle);
    }

    let results: Vec<Option<usize>> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let first = results[0].unwrap();
    for result in &results {
        assert_eq!(
            result.unwrap(),
            first,
            "All concurrent requests should converge to same worker"
        );
    }
}

#[tokio::test]
async fn test_redis_concurrent_different_keys() {
    let server = RedisTestServer::start().await.unwrap();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let workers = Arc::new(workers);

    let mut handles = Vec::new();
    for i in 0..30 {
        let redis_url = server.url().to_string();
        let workers_clone = Arc::clone(&workers);
        let key = format!("key-{}", i);
        let handle = tokio::spawn(async move {
            let policy = create_redis_policy(&redis_url);

            let headers = headers_with_routing_key(&key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            (
                key,
                policy.select_worker(&workers_clone, &info).await.unwrap(),
            )
        });
        handles.push(handle);
    }

    let results: Vec<(String, usize)> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let mut distribution: HashMap<usize, usize> = HashMap::new();
    for (_, idx) in results {
        *distribution.entry(idx).or_insert(0) += 1;
    }

    assert!(
        distribution.len() > 1,
        "Different keys should distribute across workers"
    );
}

// ============================================================================
// TTL Tests
// ============================================================================

#[tokio::test]
async fn test_redis_ttl_expiry() {
    let server = RedisTestServer::start().await.unwrap();

    let config = ManualConfig {
        redis_url: Some(server.url().to_string()),
        max_idle_secs: 2,
        ..Default::default()
    };
    let policy = ManualPolicy::with_config(config);

    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("ttl-test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let _first_idx = policy.select_worker(&workers, &info).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    let _after_expiry = policy.select_worker(&workers, &info).await.unwrap();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_redis_fallback_on_no_healthy_workers() {
    let server = RedisTestServer::start().await.unwrap();
    let policy = create_redis_policy(server.url());
    let workers = create_workers(&["http://w1:8000"]);
    workers[0].set_healthy(false);

    let headers = headers_with_routing_key("no-healthy");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let result = policy.select_worker(&workers, &info).await;
    assert!(
        result.is_none(),
        "Should return None when no healthy workers"
    );
}
