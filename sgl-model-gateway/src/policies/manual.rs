//! Manual routing policy with header-based routing support
//!
//! Supports two routing mechanisms via HTTP headers:
//! - `X-SMG-Target-Worker`: Direct routing by worker index (0-based), returns None if unavailable
//! - `X-SMG-Routing-Key`: Consistent hash routing for session affinity
//!
//! Complexity: O(n) for get_healthy_worker_indices (unavoidable), O(1) for routing decisions.

use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use http::header::HeaderName;
use rand::Rng as _;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, observability::metrics::Metrics};

/// Header for direct worker targeting by index (0-based)
static HEADER_TARGET_WORKER: HeaderName = HeaderName::from_static("x-smg-target-worker");
/// Header for consistent hash routing
static HEADER_ROUTING_KEY: HeaderName = HeaderName::from_static("x-smg-routing-key");

/// Execution branch for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    NoHealthyWorkers,
    TargetWorkerHit,
    TargetWorkerMiss,
    RoutingKeyHit,
    RandomFallback,
}

impl Branch {
    #[inline]
    const fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::TargetWorkerHit => "target_worker_hit",
            Self::TargetWorkerMiss => "target_worker_miss",
            Self::RoutingKeyHit => "routing_key_hit",
            Self::RandomFallback => "random_fallback",
        }
    }
}

#[derive(Debug, Default)]
pub struct ManualPolicy;

impl ManualPolicy {
    pub fn new() -> Self {
        Self
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        // O(n) - unavoidable, need to know which workers are healthy
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Extract routing headers - to_str() is O(1), just validates ASCII, no allocation
        let target_worker = info
            .headers
            .and_then(|h| h.get(&HEADER_TARGET_WORKER))
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty());

        let routing_key = info
            .headers
            .and_then(|h| h.get(&HEADER_ROUTING_KEY))
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty());

        // Priority 1: X-SMG-Target-Worker - direct routing by worker index
        // O(1) parse + O(1) bounds check + O(1) health check
        if let Some(idx_str) = target_worker {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx < workers.len() && workers[idx].is_healthy() {
                    return (Some(idx), Branch::TargetWorkerHit);
                }
            }
            return (None, Branch::TargetWorkerMiss);
        }

        // Priority 2: X-SMG-Routing-Key - consistent hash routing
        // O(key_len) hash + O(1) modulo + O(1) index
        if let Some(key) = routing_key {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hasher);
            let idx = (hasher.finish() as usize) % healthy_indices.len();
            return (Some(healthy_indices[idx]), Branch::RoutingKeyHit);
        }

        // Fallback: random selection using thread-local RNG (fast, no allocation)
        let idx = rand::rng().random_range(0..healthy_indices.len());
        (Some(healthy_indices[idx]), Branch::RandomFallback)
    }
}

impl LoadBalancingPolicy for ManualPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_manual_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "manual"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn headers_with_routing_key(key: &str) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", key.parse().unwrap());
        headers
    }

    fn headers_with_target_worker(idx: usize) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-target-worker", idx.to_string().parse().unwrap());
        headers
    }

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

    #[test]
    fn test_consistent_routing() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let headers = headers_with_routing_key("user-123");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (first_result, _) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();

        // Same key should always route to same worker
        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(first_idx));
            assert_eq!(branch, Branch::RoutingKeyHit);
        }
    }

    #[test]
    fn test_different_keys_distribute() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let mut distribution = HashMap::new();
        for i in 0..100 {
            let headers = headers_with_routing_key(&format!("user-{}", i));
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            let (result, _) = policy.select_worker_impl(&workers, &info);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(distribution.len() > 1, "Should distribute across workers");
    }

    #[test]
    fn test_target_worker_hit() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_target_worker(1);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1));
        assert_eq!(branch, Branch::TargetWorkerHit);
    }

    #[test]
    fn test_target_worker_miss_out_of_bounds() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_target_worker(5); // Out of bounds
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::TargetWorkerMiss);
    }

    #[test]
    fn test_target_worker_miss_unhealthy() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        workers[1].set_healthy(false);

        let headers = headers_with_target_worker(1);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::TargetWorkerMiss);
    }

    #[test]
    fn test_target_worker_priority_over_routing_key() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-target-worker", "1".parse().unwrap());
        headers.insert("x-smg-routing-key", "some-key".parse().unwrap());

        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1));
        assert_eq!(branch, Branch::TargetWorkerHit);
    }

    #[test]
    fn test_fallback_random_distribution() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        // Without routing headers, should distribute randomly across workers
        let mut distribution = HashMap::new();
        for _ in 0..100 {
            let info = SelectWorkerInfo::default();
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert!(result.is_some());
            assert_eq!(branch, Branch::RandomFallback);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        // Should distribute across multiple workers (not always same one)
        assert!(
            distribution.len() > 1,
            "Random fallback should distribute across workers"
        );
    }

    #[test]
    fn test_no_healthy_workers() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000"]);
        workers[0].set_healthy(false);

        let headers = headers_with_routing_key("test");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }

    #[test]
    fn test_empty_workers() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];

        let info = SelectWorkerInfo::default();
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }

    #[test]
    fn test_routing_key_remaps_when_worker_unhealthy() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("sticky-user");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (first_result, _) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();

        // Mark that worker unhealthy
        workers[first_idx].set_healthy(false);

        // Should now route to the other worker
        let (new_result, _) = policy.select_worker_impl(&workers, &info);
        let new_idx = new_result.unwrap();
        assert_ne!(new_idx, first_idx);
    }

    #[test]
    fn test_empty_routing_key_uses_fallback() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert!(result.is_some());
        assert_eq!(branch, Branch::RandomFallback);
    }

    #[test]
    fn test_policy_name() {
        let policy = ManualPolicy::new();
        assert_eq!(policy.name(), "manual");
    }
}
