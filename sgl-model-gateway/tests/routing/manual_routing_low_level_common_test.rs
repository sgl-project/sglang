use std::{collections::HashMap, sync::Arc};

use smg::{
    config::ManualAssignmentMode,
    core::Worker,
    policies::{LoadBalancingPolicy, ManualConfig, ManualPolicy, SelectWorkerInfo},
};

use crate::common::manual_routing_test_helpers::{
    create_workers, headers_with_routing_key, manual_routing_all_backend_test, TestManualConfig,
};

// ============================================================================
// Consistent Routing Tests
// ============================================================================

manual_routing_all_backend_test!(test_consistent_routing);
async fn test_consistent_routing_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    let headers = headers_with_routing_key("user-123");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_result = policy.select_worker(&workers, &info).await;
    let first_idx = first_result.unwrap();

    for _ in 0..10 {
        let result = policy.select_worker(&workers, &info).await;
        assert_eq!(
            result,
            Some(first_idx),
            "Same routing_id should route to same worker"
        );
    }
}

manual_routing_all_backend_test!(test_different_routing_ids);
async fn test_different_routing_ids_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    let mut distribution = HashMap::new();
    for i in 0..100 {
        let headers = headers_with_routing_key(&format!("user-{}", i));
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let result = policy.select_worker(&workers, &info).await;
        *distribution.entry(result.unwrap()).or_insert(0) += 1;
    }

    assert!(
        distribution.len() > 1,
        "Should distribute across multiple workers"
    );
}

// ============================================================================
// Fallback and Edge Case Tests
// ============================================================================

manual_routing_all_backend_test!(test_fallback_random);
async fn test_fallback_random_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let mut counts = HashMap::new();
    for _ in 0..100 {
        let info = SelectWorkerInfo::default();
        let result = policy.select_worker(&workers, &info).await;
        if let Some(idx) = result {
            *counts.entry(idx).or_insert(0) += 1;
        }
    }

    assert_eq!(counts.len(), 2, "Random fallback should use all workers");
}

manual_routing_all_backend_test!(test_empty_routing_id);
async fn test_empty_routing_id_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let mut counts = HashMap::new();
    for _ in 0..100 {
        let headers = headers_with_routing_key("");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let result = policy.select_worker(&workers, &info).await;
        if let Some(idx) = result {
            *counts.entry(idx).or_insert(0) += 1;
        }
    }

    assert_eq!(
        counts.len(),
        2,
        "Empty routing_id should use random fallback"
    );
}

manual_routing_all_backend_test!(test_empty_workers);
async fn test_empty_workers_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers: Vec<Arc<dyn Worker>> = vec![];
    let headers = headers_with_routing_key("test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    let result = policy.select_worker(&workers, &info).await;
    assert_eq!(result, None);
}

manual_routing_all_backend_test!(test_single_worker);
async fn test_single_worker_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000"]);

    let headers = headers_with_routing_key("single-test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let result = policy.select_worker(&workers, &info).await;
    assert_eq!(result, Some(0));

    for _ in 0..10 {
        let result = policy.select_worker(&workers, &info).await;
        assert_eq!(result, Some(0));
    }
}

// ============================================================================
// Unhealthy Worker Tests
// ============================================================================

manual_routing_all_backend_test!(test_with_unhealthy_workers);
async fn test_with_unhealthy_workers_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    workers[0].set_healthy(false);

    let headers = headers_with_routing_key("test-routing-id");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let result = policy.select_worker(&workers, &info).await;
    assert_eq!(result, Some(1), "Should only select healthy worker");

    for _ in 0..10 {
        let result = policy.select_worker(&workers, &info).await;
        assert_eq!(result, Some(1), "Should only select healthy worker");
    }
}

manual_routing_all_backend_test!(test_no_healthy_workers);
async fn test_no_healthy_workers_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000"]);

    workers[0].set_healthy(false);
    let headers = headers_with_routing_key("test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    let result = policy.select_worker(&workers, &info).await;
    assert_eq!(result, None);
}

manual_routing_all_backend_test!(test_remaps_when_worker_becomes_unhealthy);
async fn test_remaps_when_worker_becomes_unhealthy_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("sticky-user");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_result = policy.select_worker(&workers, &info).await;
    let first_idx = first_result.unwrap();

    workers[first_idx].set_healthy(false);

    let new_result = policy.select_worker(&workers, &info).await;
    let new_idx = new_result.unwrap();
    assert_ne!(new_idx, first_idx, "Should remap to healthy worker");

    for _ in 0..10 {
        let result = policy.select_worker(&workers, &info).await;
        assert_eq!(
            result,
            Some(new_idx),
            "Should consistently route to new worker"
        );
    }
}

// ============================================================================
// Worker Recovery and Candidate Eviction Tests
// ============================================================================

manual_routing_all_backend_test!(test_worker_recovery);
async fn test_worker_recovery_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    let headers = headers_with_routing_key("recovery-test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_result = policy.select_worker(&workers, &info).await;
    let first_idx = first_result.unwrap();

    workers[first_idx].set_healthy(false);

    let second_result = policy.select_worker(&workers, &info).await;
    let second_idx = second_result.unwrap();
    assert_ne!(second_idx, first_idx);

    workers[first_idx].set_healthy(true);

    let after_recovery = policy.select_worker(&workers, &info).await;
    assert_eq!(
        after_recovery,
        Some(first_idx),
        "Should return to original worker after recovery since it's first in candidate list"
    );
}

manual_routing_all_backend_test!(test_max_candidate_workers_eviction);
async fn test_max_candidate_workers_eviction_impl(cfg: TestManualConfig) {
    let policy = cfg.build_policy();
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    let headers = headers_with_routing_key("eviction-test");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_result = policy.select_worker(&workers, &info).await;
    let first_idx = first_result.unwrap();

    workers[first_idx].set_healthy(false);

    let second_result = policy.select_worker(&workers, &info).await;
    let second_idx = second_result.unwrap();
    assert_ne!(second_idx, first_idx);

    workers[second_idx].set_healthy(false);

    let remaining_idx = (0..3).find(|&i| i != first_idx && i != second_idx).unwrap();
    let third_result = policy.select_worker(&workers, &info).await;
    assert_eq!(
        third_result,
        Some(remaining_idx),
        "Should select the only remaining healthy worker"
    );

    workers[first_idx].set_healthy(true);

    let idx_after_restore = policy.select_worker(&workers, &info).await;
    assert_ne!(
        idx_after_restore,
        Some(first_idx),
        "First worker should be evicted from candidates due to MAX_CANDIDATE_WORKERS=2"
    );
}

// ============================================================================
// Assignment Mode Tests
// ============================================================================

manual_routing_all_backend_test!(test_min_group_distributes_evenly);
async fn test_min_group_distributes_evenly_impl(cfg: TestManualConfig) {
    let policy = ManualPolicy::with_config(ManualConfig {
        assignment_mode: ManualAssignmentMode::MinGroup,
        redis_url: cfg.redis_url,
        redis_key_prefix: cfg.redis_key_prefix,
        ..Default::default()
    });
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    for i in 0..9 {
        let routing_key = format!("key-{}", i);
        let headers = headers_with_routing_key(&routing_key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let result = policy.select_worker(&workers, &info).await;
        assert!(result.is_some());

        let selected_idx = result.unwrap();
        workers[selected_idx]
            .worker_routing_key_load()
            .increment(&routing_key);
    }

    for worker in &workers {
        let load = worker.worker_routing_key_load().value();
        assert_eq!(load, 3, "Each worker should have exactly 3 routing keys");
    }

    let all_urls: Vec<Vec<String>> = policy.iter_urls().await.unwrap();
    assert_eq!(all_urls.len(), 9, "Should have 9 routing keys stored");

    let distribution: HashMap<String, usize> = all_urls
        .iter()
        .filter_map(|urls: &Vec<String>| urls.first().cloned())
        .fold(HashMap::new(), |mut acc, url: String| {
            *acc.entry(url).or_default() += 1;
            acc
        });
    assert_eq!(distribution.len(), 3, "Should use all 3 workers");
    for count in distribution.values() {
        assert_eq!(*count, 3, "Each worker should have exactly 3 routing keys in storage");
    }
}

manual_routing_all_backend_test!(test_min_group_prefers_fewer_keys);
async fn test_min_group_prefers_fewer_keys_impl(cfg: TestManualConfig) {
    let policy = ManualPolicy::with_config(ManualConfig {
        assignment_mode: ManualAssignmentMode::MinGroup,
        redis_url: cfg.redis_url,
        redis_key_prefix: cfg.redis_key_prefix,
        ..Default::default()
    });
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    workers[0].worker_routing_key_load().increment("existing-1");
    workers[0].worker_routing_key_load().increment("existing-2");
    workers[1].worker_routing_key_load().increment("existing-3");

    assert_eq!(workers[0].worker_routing_key_load().value(), 2);
    assert_eq!(workers[1].worker_routing_key_load().value(), 1);
    assert_eq!(workers[2].worker_routing_key_load().value(), 0);

    let headers = headers_with_routing_key("new-key");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    let result = policy.select_worker(&workers, &info).await;
    let selected_idx = result.unwrap();

    assert_eq!(selected_idx, 2, "Should select worker with 0 routing keys");
}

manual_routing_all_backend_test!(test_min_load_prefers_fewer_requests);
async fn test_min_load_prefers_fewer_requests_impl(cfg: TestManualConfig) {
    let policy = ManualPolicy::with_config(ManualConfig {
        assignment_mode: ManualAssignmentMode::MinLoad,
        redis_url: cfg.redis_url,
        redis_key_prefix: cfg.redis_key_prefix,
        ..Default::default()
    });
    let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    workers[0].increment_load();
    workers[0].increment_load();
    workers[1].increment_load();

    assert_eq!(workers[0].load(), 2);
    assert_eq!(workers[1].load(), 1);
    assert_eq!(workers[2].load(), 0);

    let headers = headers_with_routing_key("new-key");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    let result = policy.select_worker(&workers, &info).await;
    let selected_idx = result.unwrap();

    assert_eq!(selected_idx, 2, "Should select worker with 0 load");
}

manual_routing_all_backend_test!(test_min_group_sticky_after_assignment);
async fn test_min_group_sticky_after_assignment_impl(cfg: TestManualConfig) {
    let policy = ManualPolicy::with_config(ManualConfig {
        assignment_mode: ManualAssignmentMode::MinGroup,
        redis_url: cfg.redis_url,
        redis_key_prefix: cfg.redis_key_prefix,
        ..Default::default()
    });
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    workers[0].worker_routing_key_load().increment("key-0");
    workers[1].worker_routing_key_load().increment("key-1");
    workers[1].worker_routing_key_load().increment("key-2");

    let headers = headers_with_routing_key("new-sticky-key");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first_result = policy.select_worker(&workers, &info).await;
    let first_idx = first_result.unwrap();
    assert_eq!(first_idx, 0, "Should initially select worker 0 (fewer keys)");

    workers[0].worker_routing_key_load().increment("key-3");
    workers[0].worker_routing_key_load().increment("key-4");
    workers[0].worker_routing_key_load().increment("key-5");

    for _ in 0..5 {
        let result = policy.select_worker(&workers, &info).await;
        assert_eq!(
            result,
            Some(0),
            "Should stay sticky even when worker has more keys"
        );
    }
}

manual_routing_all_backend_test!(test_random_mode_does_not_consider_load);
async fn test_random_mode_does_not_consider_load_impl(cfg: TestManualConfig) {
    let policy = ManualPolicy::with_config(ManualConfig {
        assignment_mode: ManualAssignmentMode::Random,
        redis_url: cfg.redis_url,
        redis_key_prefix: cfg.redis_key_prefix,
        ..Default::default()
    });
    let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

    workers[0].worker_routing_key_load().increment("key-1");
    workers[0].worker_routing_key_load().increment("key-2");
    workers[0].worker_routing_key_load().increment("key-3");

    let mut selected_worker_0 = false;
    for i in 0..50 {
        let headers = headers_with_routing_key(&format!("test-{}", i));
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let result = policy.select_worker(&workers, &info).await;
        if result == Some(0) {
            selected_worker_0 = true;
            break;
        }
    }
    assert!(
        selected_worker_0,
        "Random mode should sometimes select worker 0 despite higher load"
    );
}
