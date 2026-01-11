use std::{collections::HashMap, sync::Arc};

use smg::policies::{LoadBalancingPolicy, ManualConfig, ManualPolicy, SelectWorkerInfo};

use crate::common::{
    manual_routing_test_helpers::{
        create_workers, headers_with_routing_key, random_prefix, TestManualConfig,
    },
    redis_test_server::get_shared_server,
};

// ============================================================================
// Multi-Instance Consistency Tests
// ============================================================================

#[tokio::test]
async fn test_redis_multi_instance_consistency() {
    let prefix = random_prefix("test_multi_instance");
    let cfg = TestManualConfig::redis_with_prefix(&prefix);
    let policy1 = cfg.build_policy();
    let policy2 = cfg.build_policy();
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
    let prefix = random_prefix("test_cross_failover");
    let cfg = TestManualConfig::redis_with_prefix(&prefix);
    let policy1 = cfg.build_policy();
    let policy2 = cfg.build_policy();
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
    let server = get_shared_server();
    let prefix = random_prefix("test_concurrent_same");
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
    let workers = Arc::new(workers);

    let mut handles = Vec::new();
    for _ in 0..10 {
        let redis_url = server.url().to_string();
        let prefix_clone = prefix.clone();
        let workers_clone = Arc::clone(&workers);
        let handle = tokio::spawn(async move {
            let config = ManualConfig {
                redis_url: Some(redis_url),
                redis_key_prefix: Some(prefix_clone),
                ..Default::default()
            };
            let policy = ManualPolicy::with_config(config);

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
    let server = get_shared_server();
    let prefix = random_prefix("test_concurrent_diff");
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let workers = Arc::new(workers);

    let mut handles = Vec::new();
    for i in 0..30 {
        let redis_url = server.url().to_string();
        let prefix_clone = prefix.clone();
        let workers_clone = Arc::clone(&workers);
        let key = format!("key-{}", i);
        let handle = tokio::spawn(async move {
            let config = ManualConfig {
                redis_url: Some(redis_url),
                redis_key_prefix: Some(prefix_clone),
                ..Default::default()
            };
            let policy = ManualPolicy::with_config(config);

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
    let policy = TestManualConfig::redis("test_ttl").with_ttl(2).build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("ttl-test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    policy.select_worker(&workers, &info).await.unwrap();
    assert_eq!(policy.iter_urls().await.unwrap().len(), 1);

    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    assert_eq!(
        policy.iter_urls().await.unwrap().len(),
        0,
        "Key should be expired and removed from Redis"
    );

    policy.select_worker(&workers, &info).await.unwrap();
    assert_eq!(
        policy.iter_urls().await.unwrap().len(),
        1,
        "New key should be created after expiry"
    );
}
