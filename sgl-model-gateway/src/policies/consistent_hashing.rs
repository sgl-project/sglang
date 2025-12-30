//! Consistent hashing routing policy with header-based routing support
//!
//! Supports two routing mechanisms via HTTP headers:
//! - `X-SMG-Target-Worker`: Direct routing by worker index (0-based), returns None if unavailable
//! - `X-SMG-Routing-Key`: Consistent hash routing for session affinity
//!
//! ## Consistent Hashing
//!
//! Uses a pre-computed hash ring from WorkerRegistry where:
//! 1. Each worker is placed at a fixed position based on hash(worker_url)
//! 2. Keys are hashed to the ring, then walk clockwise to find first healthy worker
//! 3. When workers scale up/down, only keys in the affected range redistribute (~1/N keys move)
//!
//! The ring is built once when workers are added/removed, not per-request.
//! This ensures O(log n) lookup performance.
//!
//! Complexity: O(log n) binary search + O(k) walk where k = consecutive unhealthy workers.

use std::sync::Arc;

use http::header::HeaderName;
use rand::Rng as _;

use super::{LoadBalancingPolicy, SelectWorkerInfo};
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
pub struct ConsistentHashingPolicy;

impl ConsistentHashingPolicy {
    pub fn new() -> Self {
        Self
    }

    /// Use consistent hashing to find a worker for the given key.
    /// Uses pre-computed ring from SelectWorkerInfo if available.
    ///
    /// The ring returns a worker URL, which we then map to an index in the workers array.
    /// This correctly handles filtered worker arrays since we match by URL, not by index.
    ///
    /// Complexity: O(n) to build healthy URL map + O(log n) ring lookup + O(k) walk
    fn find_by_consistent_hash(
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        key: &str,
    ) -> Option<usize> {
        // Build URL→index map for healthy workers: O(n) once, O(1) lookups
        let healthy_url_to_idx: std::collections::HashMap<&str, usize> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .map(|(i, w)| (w.url(), i))
            .collect();

        if healthy_url_to_idx.is_empty() {
            return None;
        }

        // Use pre-computed ring if available
        if let Some(ref ring) = info.hash_ring {
            // O(1) lookup per URL checked instead of O(n)
            let url = ring.find_healthy_url(key, |url| healthy_url_to_idx.contains_key(url))?;
            return healthy_url_to_idx.get(url).copied();
        }

        // Fallback: no ring provided, use simple modulo (less optimal but functional)
        // This shouldn't happen in normal operation as WorkerSelectionStage provides the ring
        let mut healthy_indices: Vec<usize> = healthy_url_to_idx.values().copied().collect();
        healthy_indices.sort_unstable(); // Ensure deterministic order

        // Use blake3 for consistent hashing in fallback too
        let hash = blake3::hash(key.as_bytes());
        let hash_val = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        let idx = (hash_val as usize) % healthy_indices.len();
        Some(healthy_indices[idx])
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        if workers.is_empty() {
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

        // Priority 2: X-SMG-Routing-Key - consistent hash routing (O(log n))
        if let Some(key) = routing_key {
            return match Self::find_by_consistent_hash(workers, info, key) {
                Some(idx) => (Some(idx), Branch::RoutingKeyHit),
                None => (None, Branch::NoHealthyWorkers),
            };
        }

        // Priority 3: Implicit routing key from stable headers (session affinity)
        let implicit_key = info.headers.and_then(|h| {
            h.get("authorization")
                .or_else(|| h.get("x-forwarded-for"))
                .or_else(|| h.get("cookie"))
                .and_then(|v| v.to_str().ok())
                .filter(|s| !s.is_empty())
        });

        if let Some(key) = implicit_key {
            return match Self::find_by_consistent_hash(workers, info, key) {
                Some(idx) => (Some(idx), Branch::RoutingKeyHit),
                None => (None, Branch::NoHealthyWorkers),
            };
        }

        // Fallback: random selection (truly anonymous client)
        let healthy_count = workers.iter().filter(|w| w.is_healthy()).count();
        if healthy_count == 0 {
            return (None, Branch::NoHealthyWorkers);
        }

        let random_healthy_idx = rand::rng().random_range(0..healthy_count);
        let idx = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .nth(random_healthy_idx)
            .map(|(i, _)| i)
            .unwrap();

        (Some(idx), Branch::RandomFallback)
    }
}

impl LoadBalancingPolicy for ConsistentHashingPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_consistent_hashing_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "consistent_hashing"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, HashRing, WorkerType};

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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];

        let info = SelectWorkerInfo::default();
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }

    #[test]
    fn test_consistent_hash_minimal_redistribution() {
        // Test that consistent hashing moves fewer keys than random redistribution
        let policy = ConsistentHashingPolicy::new();
        let workers = create_workers(&[
            "http://w0:8000",
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
        ]);
        let ring = Arc::new(HashRing::new(&workers));

        // Record which worker each key routes to with all workers healthy
        let mut key_to_worker_before: HashMap<String, usize> = HashMap::new();
        for i in 0..100 {
            let key = format!("user-{}", i);
            let headers = headers_with_routing_key(&key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            let (result, _) = policy.select_worker_impl(&workers, &info);
            key_to_worker_before.insert(key, result.unwrap());
        }

        // Mark worker 1 as unhealthy
        workers[1].set_healthy(false);

        // Record new routing and count how many keys moved
        let mut moved_count = 0;
        for i in 0..100 {
            let key = format!("user-{}", i);
            let headers = headers_with_routing_key(&key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            let (result, _) = policy.select_worker_impl(&workers, &info);
            let new_worker = result.unwrap();
            let old_worker = key_to_worker_before[&key];

            if new_worker != old_worker {
                moved_count += 1;
            }
        }

        // With consistent hashing, approximately 1/N keys should move (N = worker count)
        // Random redistribution would move approximately (N-1)/N = 75% of keys
        // Verify we're significantly better than random (< 50% moved)
        let keys_on_failed_worker = key_to_worker_before.values().filter(|&&w| w == 1).count();
        assert!(
            moved_count <= keys_on_failed_worker + 5,
            "Consistent hashing should only move keys from failed worker (+small variance). \
             Expected ~{}, got {}",
            keys_on_failed_worker,
            moved_count
        );
        assert!(
            moved_count < 50,
            "Consistent hashing should move fewer than 50% of keys (random would move ~75%), got {}%",
            moved_count
        );
    }

    #[test]
    fn test_routing_key_failover_and_recovery() {
        // Test that when a worker fails, keys move to another worker,
        // and when it recovers, keys return to the original worker
        let policy = ConsistentHashingPolicy::new();
        let workers = create_workers(&["http://w0:8000", "http://w1:8000", "http://w2:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Find which worker a key routes to when all are healthy
        let test_key = "session-abc-123";
        let headers = headers_with_routing_key(test_key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (result, _) = policy.select_worker_impl(&workers, &info);
        let original_idx = result.unwrap();

        // Mark that worker unhealthy
        workers[original_idx].set_healthy(false);

        // Key should now route to a different healthy worker
        let (failover_result, _) = policy.select_worker_impl(&workers, &info);
        let failover_idx = failover_result.unwrap();
        assert_ne!(
            failover_idx, original_idx,
            "Should failover to different worker"
        );
        assert!(
            workers[failover_idx].is_healthy(),
            "Failover target should be healthy"
        );

        // Failover should be consistent
        for _ in 0..5 {
            let (result, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(failover_idx), "Failover should be consistent");
        }

        // Recover the original worker
        workers[original_idx].set_healthy(true);

        // Key should route back to original worker
        let (recovered_result, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            recovered_result,
            Some(original_idx),
            "Should return to original worker after recovery"
        );
    }

    #[test]
    fn test_empty_routing_key_uses_fallback() {
        let policy = ConsistentHashingPolicy::new();
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
        let policy = ConsistentHashingPolicy::new();
        assert_eq!(policy.name(), "consistent_hashing");
    }
}
