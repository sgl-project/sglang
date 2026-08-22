//! Opt-in consistent hashing with a bounded load skew.
//!
//! This policy keeps the existing `consistent_hashing` behavior for target
//! worker headers, implicit session keys, and anonymous requests. Only an
//! explicit `X-SMG-Routing-Key` may spill over to the next eligible worker on
//! the ring when its preferred worker is above the configured load bound.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use rand::Rng as _;

use super::{ConsistentHashingPolicy, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{
    core::Worker,
    observability::metrics::Metrics,
    routers::header_utils::{extract_routing_key, extract_target_worker},
};

/// Configuration for the bounded consistent hashing policy.
#[derive(Debug, Clone, Copy)]
pub struct BoundedConsistentHashingConfig {
    /// Maximum preferred-worker load relative to the healthy-worker mean.
    pub max_load_skew: f64,

    /// Minimum preferred-to-least-loaded gap, in active requests, before spillover.
    pub min_load_gap: usize,
}

impl Default for BoundedConsistentHashingConfig {
    fn default() -> Self {
        Self {
            max_load_skew: 1.5,
            min_load_gap: 2,
        }
    }
}

/// Execution branch for metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    NoHealthyWorkers,
    TargetWorkerHit,
    TargetWorkerMiss,
    ExplicitRoutingKeyHit,
    ExplicitRoutingKeySpillover,
    ExplicitRoutingKeyNoEligibleCandidate,
    ImplicitRoutingKeyHit,
    RandomFallback,
}

impl Branch {
    #[inline]
    const fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::TargetWorkerHit => "target_worker_hit",
            Self::TargetWorkerMiss => "target_worker_miss",
            Self::ExplicitRoutingKeyHit => "explicit_routing_key_hit",
            Self::ExplicitRoutingKeySpillover => "explicit_routing_key_spillover",
            Self::ExplicitRoutingKeyNoEligibleCandidate => {
                "explicit_routing_key_no_eligible_candidate"
            }
            Self::ImplicitRoutingKeyHit => "implicit_routing_key_hit",
            Self::RandomFallback => "random_fallback",
        }
    }
}

/// Consistent hashing with opt-in bounded load spillover for explicit keys.
#[derive(Debug)]
pub struct BoundedConsistentHashingPolicy {
    config: BoundedConsistentHashingConfig,
}

impl BoundedConsistentHashingPolicy {
    pub fn new(config: BoundedConsistentHashingConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(BoundedConsistentHashingConfig::default())
    }

    #[inline]
    fn within_relative_bound(&self, worker_load: usize, mean_healthy_load: f64) -> bool {
        worker_load as f64 <= mean_healthy_load * self.config.max_load_skew
    }

    #[inline]
    fn should_spill(
        &self,
        preferred_load: usize,
        min_healthy_load: usize,
        mean_healthy_load: f64,
    ) -> bool {
        preferred_load.saturating_sub(min_healthy_load) > self.config.min_load_gap
            && preferred_load as f64 > mean_healthy_load * self.config.max_load_skew
    }

    fn find_bounded_by_consistent_hash(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        key: &str,
    ) -> (Option<usize>, Branch) {
        let healthy_url_to_idx: HashMap<&str, usize> = workers
            .iter()
            .enumerate()
            .filter(|(_, worker)| worker.is_healthy())
            .map(|(idx, worker)| (worker.url(), idx))
            .collect();

        if healthy_url_to_idx.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        let (total_load, min_healthy_load) = workers
            .iter()
            .filter(|worker| worker.is_healthy())
            .fold((0usize, usize::MAX), |(total, min_load), worker| {
                let load = worker.load();
                (total.saturating_add(load), min_load.min(load))
            });
        let worker_count = healthy_url_to_idx.len();
        let mean_healthy_load = total_load as f64 / worker_count as f64;

        // Use the exact same preferred-worker selection as consistent_hashing.
        // This preserves ring behavior when a worker is unhealthy or the ring
        // is not available in a direct policy call.
        let preferred_idx =
            match ConsistentHashingPolicy::find_by_consistent_hash(workers, info, key) {
                Some(idx) => idx,
                None => return (None, Branch::NoHealthyWorkers),
            };

        let preferred_load = workers[preferred_idx].load();
        if !self.should_spill(preferred_load, min_healthy_load, mean_healthy_load) {
            return (Some(preferred_idx), Branch::ExplicitRoutingKeyHit);
        }

        // The ring predicate is evaluated in its existing clockwise order, so
        // the first match is deterministic and preserves the affinity search
        // order. If every worker is above the bound, retain the preferred
        // worker instead of failing an otherwise routable request.
        let candidate_idx = info.hash_ring.as_ref().and_then(|ring| {
            ring.find_healthy_url(key, |url| {
                healthy_url_to_idx.get(url).is_some_and(|idx| {
                    self.within_relative_bound(workers[*idx].load(), mean_healthy_load)
                })
            })
            .and_then(|url| healthy_url_to_idx.get(url).copied())
        });

        match candidate_idx {
            Some(idx) if idx != preferred_idx => (Some(idx), Branch::ExplicitRoutingKeySpillover),
            Some(_) => (Some(preferred_idx), Branch::ExplicitRoutingKeyHit),
            None => (
                Some(preferred_idx),
                Branch::ExplicitRoutingKeyNoEligibleCandidate,
            ),
        }
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        if workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        let target_worker = extract_target_worker(info.headers);
        let routing_key = extract_routing_key(info.headers);

        // X-SMG-Target-Worker is always strict and has priority over all key
        // based policies, including bounded spillover.
        if let Some(idx_str) = target_worker {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx < workers.len() && workers[idx].is_healthy() {
                    return (Some(idx), Branch::TargetWorkerHit);
                }
            }
            return (None, Branch::TargetWorkerMiss);
        }

        // Bounded load spillover is deliberately limited to the explicit key
        // header. This keeps inferred affinity behavior backward compatible.
        if let Some(key) = routing_key {
            return self.find_bounded_by_consistent_hash(workers, info, key);
        }

        let implicit_key = info.headers.and_then(|headers| {
            headers
                .get("authorization")
                .or_else(|| headers.get("x-forwarded-for"))
                .or_else(|| headers.get("cookie"))
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
        });

        if let Some(key) = implicit_key {
            return match ConsistentHashingPolicy::find_by_consistent_hash(workers, info, key) {
                Some(idx) => (Some(idx), Branch::ImplicitRoutingKeyHit),
                None => (None, Branch::NoHealthyWorkers),
            };
        }

        let healthy_count = workers.iter().filter(|worker| worker.is_healthy()).count();
        if healthy_count == 0 {
            return (None, Branch::NoHealthyWorkers);
        }

        let random_healthy_idx = rand::rng().random_range(0..healthy_count);
        let idx = workers
            .iter()
            .enumerate()
            .filter(|(_, worker)| worker.is_healthy())
            .nth(random_healthy_idx)
            .map(|(idx, _)| idx)
            .unwrap();

        (Some(idx), Branch::RandomFallback)
    }
}

#[async_trait]
impl LoadBalancingPolicy for BoundedConsistentHashingPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_bounded_consistent_hashing_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "bounded_consistent_hashing"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, HashRing, WorkerLoadGuard, WorkerType};

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

    fn headers_with_implicit_key(name: &'static str, value: &str) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert(name, value.parse().unwrap());
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

    fn key_for_worker(workers: &[Arc<dyn Worker>], ring: &Arc<HashRing>, target: usize) -> String {
        for index in 0..10_000 {
            let key = format!("bounded-routing-key-{index}");
            let info = SelectWorkerInfo {
                hash_ring: Some(Arc::clone(ring)),
                ..Default::default()
            };
            if ConsistentHashingPolicy::find_by_consistent_hash(workers, &info, &key)
                == Some(target)
            {
                return key;
            }
        }
        panic!("could not find a key for worker {target}");
    }

    fn set_loads(workers: &[Arc<dyn Worker>], loads: &[usize]) {
        assert_eq!(workers.len(), loads.len());
        for (worker, load) in workers.iter().zip(loads) {
            for _ in 0..*load {
                worker.increment_load();
            }
        }
    }

    fn select_explicit(
        policy: &BoundedConsistentHashingPolicy,
        workers: &[Arc<dyn Worker>],
        ring: &Arc<HashRing>,
        key: &str,
    ) -> (Option<usize>, Branch) {
        let headers = headers_with_routing_key(key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(Arc::clone(ring)),
            ..Default::default()
        };
        policy.select_worker_impl(workers, &info)
    }
    #[tokio::test]
    async fn preserves_affinity_when_preferred_worker_is_within_bound() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.5,
            min_load_gap: 2,
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);

        for worker in &workers {
            worker.increment_load();
        }

        let headers = headers_with_routing_key(&key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(ring),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);

        assert_eq!(result, Some(0));
        assert_eq!(branch, Branch::ExplicitRoutingKeyHit);
    }

    #[tokio::test]
    async fn absolute_gap_preserves_affinity_at_low_load() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.0,
            min_load_gap: 2,
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        workers[0].increment_load();

        let headers = headers_with_routing_key(&key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(ring),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);

        assert_eq!(result, Some(0));
        assert_eq!(branch, Branch::ExplicitRoutingKeyHit);
    }

    #[tokio::test]
    async fn spills_over_clockwise_when_preferred_worker_is_over_bound() {
        let policy = BoundedConsistentHashingPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);

        for _ in 0..5 {
            workers[0].increment_load();
        }

        let headers = headers_with_routing_key(&key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(Arc::clone(&ring)),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        let (second_result, second_branch) = policy.select_worker_impl(&workers, &info);

        assert_ne!(result, Some(0));
        assert_eq!(result, second_result);
        assert_eq!(branch, Branch::ExplicitRoutingKeySpillover);
        assert_eq!(second_branch, Branch::ExplicitRoutingKeySpillover);
    }

    #[tokio::test]
    async fn unhealthy_preferred_worker_fails_over_and_recovers() {
        let policy = BoundedConsistentHashingPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        let headers = headers_with_routing_key(&key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(Arc::clone(&ring)),
            ..Default::default()
        };

        workers[0].set_healthy(false);
        let (failed_over, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(failed_over, Some(1));

        workers[0].set_healthy(true);
        let (recovered, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(recovered, Some(0));
    }

    #[tokio::test]
    async fn strict_target_worker_ignores_load_skew() {
        let policy = BoundedConsistentHashingPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        for _ in 0..10 {
            workers[0].increment_load();
        }

        let headers = headers_with_target_worker(0);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(0));
        assert_eq!(branch, Branch::TargetWorkerHit);

        workers[0].set_healthy(false);
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::TargetWorkerMiss);
    }

    #[tokio::test]
    async fn implicit_keys_keep_strict_consistent_hashing() {
        let policy = BoundedConsistentHashingPolicy::with_defaults();
        for (header_name, implicit_key) in [
            ("authorization", "Bearer implicit-session"),
            ("x-forwarded-for", "192.0.2.10"),
            ("cookie", "session=implicit-session"),
        ] {
            let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
            let ring = Arc::new(HashRing::new(&workers));
            let headers = headers_with_implicit_key(header_name, implicit_key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                hash_ring: Some(Arc::clone(&ring)),
                ..Default::default()
            };
            let preferred =
                ConsistentHashingPolicy::find_by_consistent_hash(&workers, &info, implicit_key)
                    .unwrap();
            for _ in 0..5 {
                workers[preferred].increment_load();
            }

            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(preferred));
            assert_eq!(branch, Branch::ImplicitRoutingKeyHit);
        }
    }

    #[tokio::test]
    async fn one_worker_is_always_selected() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.0,
            min_load_gap: 2,
        });
        let workers = create_workers(&["http://w1:8000"]);
        for _ in 0..100 {
            workers[0].increment_load();
        }
        let headers = headers_with_routing_key("single-worker");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(Arc::new(HashRing::new(&workers))),
            ..Default::default()
        };

        let (result, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(0));
    }
    #[test]
    fn near_idle_four_worker_vectors_require_both_thresholds() {
        for (loads, should_spill) in [
            ([0usize, 0, 0, 0], false),
            ([1usize, 0, 0, 0], false),
            ([2usize, 0, 0, 0], false),
            ([3usize, 0, 0, 0], true),
        ] {
            let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
                max_load_skew: 1.5,
                min_load_gap: 2,
            });
            let workers = create_workers(&[
                "http://w1:8000",
                "http://w2:8000",
                "http://w3:8000",
                "http://w4:8000",
            ]);
            let ring = Arc::new(HashRing::new(&workers));
            let key = key_for_worker(&workers, &ring, 0);
            set_loads(&workers, &loads);

            let (result, branch) = select_explicit(&policy, &workers, &ring, &key);
            if should_spill {
                assert_ne!(result, Some(0), "loads={loads:?}");
                assert_eq!(branch, Branch::ExplicitRoutingKeySpillover);
            } else {
                assert_eq!(result, Some(0), "loads={loads:?}");
                assert_eq!(branch, Branch::ExplicitRoutingKeyHit);
            }
        }
    }

    #[test]
    fn relative_threshold_alone_does_not_spill() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.5,
            min_load_gap: 2,
        });
        let workers = create_workers(&[
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
            "http://w4:8000",
        ]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        set_loads(&workers, &[2, 0, 0, 0]);

        let (result, branch) = select_explicit(&policy, &workers, &ring, &key);
        assert_eq!(result, Some(0));
        assert_eq!(branch, Branch::ExplicitRoutingKeyHit);
    }

    #[test]
    fn absolute_threshold_alone_does_not_spill() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.5,
            min_load_gap: 0,
        });
        let workers = create_workers(&[
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
            "http://w4:8000",
        ]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        set_loads(&workers, &[4, 3, 3, 3]);

        let (result, branch) = select_explicit(&policy, &workers, &ring, &key);
        assert_eq!(result, Some(0));
        assert_eq!(branch, Branch::ExplicitRoutingKeyHit);
    }

    #[test]
    fn both_thresholds_spill_to_first_clockwise_eligible_worker() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.5,
            min_load_gap: 2,
        });
        let workers = create_workers(&[
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
            "http://w4:8000",
        ]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        set_loads(&workers, &[3, 0, 0, 0]);

        let mean_healthy_load = 3.0 / 4.0;
        let expected_url = ring
            .find_healthy_url(&key, |url| {
                workers
                    .iter()
                    .find(|worker| worker.url() == url)
                    .is_some_and(|worker| worker.load() as f64 <= mean_healthy_load * 1.5)
            })
            .unwrap();
        let expected_idx = workers
            .iter()
            .position(|worker| worker.url() == expected_url)
            .unwrap();

        let first = select_explicit(&policy, &workers, &ring, &key);
        let second = select_explicit(&policy, &workers, &ring, &key);
        assert_eq!(
            first,
            (Some(expected_idx), Branch::ExplicitRoutingKeySpillover)
        );
        assert_eq!(second, first);
    }

    #[test]
    fn candidate_gaps_compare_placement_and_same_key_concurrency() {
        for min_load_gap in [1usize, 2, 4] {
            let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
                max_load_skew: 1.5,
                min_load_gap,
            });
            let workers = create_workers(&[
                "http://w1:8000",
                "http://w2:8000",
                "http://w3:8000",
                "http://w4:8000",
            ]);
            let ring = Arc::new(HashRing::new(&workers));
            let key = key_for_worker(&workers, &ring, 0);
            let headers = headers_with_routing_key(&key);
            let mut guards = Vec::new();
            let mut spill_idx = None;

            for request_index in 0..=(min_load_gap + 1) {
                let (result, branch) = select_explicit(&policy, &workers, &ring, &key);
                let selected = result.unwrap();
                if request_index <= min_load_gap {
                    assert_eq!(selected, 0);
                    assert_eq!(branch, Branch::ExplicitRoutingKeyHit);
                } else {
                    assert_ne!(selected, 0);
                    assert_eq!(branch, Branch::ExplicitRoutingKeySpillover);
                    spill_idx = Some(selected);
                }
                guards.push(WorkerLoadGuard::new(
                    Arc::clone(&workers[selected]),
                    Some(&headers),
                ));
            }

            let spill_idx = spill_idx.unwrap();
            assert_eq!(workers[0].load(), min_load_gap + 1);
            assert_eq!(workers[0].worker_routing_key_load().value(), 1);
            assert_eq!(workers[spill_idx].load(), 1);
            assert_eq!(workers[spill_idx].worker_routing_key_load().value(), 1);

            let mut placement: Vec<_> = workers.iter().map(|worker| worker.load()).collect();
            placement.sort_unstable_by(|left, right| right.cmp(left));
            assert_eq!(placement, vec![min_load_gap + 1, 1, 0, 0]);

            drop(guards);
            for worker in &workers {
                assert_eq!(worker.load(), 0);
                assert_eq!(worker.worker_routing_key_load().value(), 0);
            }
        }
    }

    #[test]
    fn non_igw_bounded_selection_is_invariant_to_available_slice_order() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.5,
            min_load_gap: 1,
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        set_loads(&workers, &[4, 0, 0]);
        let headers = headers_with_routing_key(&key);
        let expected_url = ring
            .find_healthy_url(&key, |url| url != workers[0].url())
            .unwrap()
            .to_string();

        let forward_info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(Arc::clone(&ring)),
            ..Default::default()
        };
        let (forward_idx, forward_branch) = policy.select_worker_impl(&workers, &forward_info);
        let forward_url = workers[forward_idx.unwrap()].url().to_string();

        let reversed: Vec<Arc<dyn Worker>> = workers.iter().rev().cloned().collect();
        let reversed_info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(ring),
            ..Default::default()
        };
        let (reversed_idx, reversed_branch) = policy.select_worker_impl(&reversed, &reversed_info);
        let reversed_url = reversed[reversed_idx.unwrap()].url().to_string();

        assert_eq!(forward_url, expected_url);
        assert_eq!(reversed_url, expected_url);
        assert_eq!(forward_branch, Branch::ExplicitRoutingKeySpillover);
        assert_eq!(reversed_branch, Branch::ExplicitRoutingKeySpillover);

        for _ in 0..4 {
            workers[0].decrement_load();
        }
        assert!(workers.iter().all(|worker| worker.load() == 0));
    }

    #[tokio::test]
    async fn missing_ring_retains_preferred_instead_of_slice_first_spill() {
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 1.5,
            min_load_gap: 1,
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let key = (0..10_000)
            .map(|index| format!("missing-ring-key-{index}"))
            .find(|key| {
                ConsistentHashingPolicy::find_by_consistent_hash(
                    &workers,
                    &SelectWorkerInfo::default(),
                    key,
                ) == Some(2)
            })
            .expect("could not find a no-ring modulo key for worker 2");
        set_loads(&workers, &[0, 0, 4]);
        let headers = headers_with_routing_key(&key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: None,
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(2));
        assert_eq!(branch, Branch::ExplicitRoutingKeyNoEligibleCandidate);
        assert_eq!(
            branch.as_str(),
            "explicit_routing_key_no_eligible_candidate"
        );
        assert_eq!(policy.select_worker(&workers, &info).await, Some(2));

        for _ in 0..4 {
            workers[2].decrement_load();
        }
        assert!(workers.iter().all(|worker| worker.load() == 0));
    }

    #[tokio::test]
    async fn no_eligible_candidate_retains_preferred_worker() {
        // Public validation rejects max_load_skew < 1.0. Constructing the
        // policy directly exercises the defensive no-candidate fallback.
        let policy = BoundedConsistentHashingPolicy::new(BoundedConsistentHashingConfig {
            max_load_skew: 0.1,
            min_load_gap: 0,
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        set_loads(&workers, &[4, 1]);
        let headers = headers_with_routing_key(&key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(0));
        assert_eq!(branch, Branch::ExplicitRoutingKeyNoEligibleCandidate);
        assert_eq!(
            branch.as_str(),
            "explicit_routing_key_no_eligible_candidate"
        );
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));

        for _ in 0..4 {
            workers[0].decrement_load();
        }
        workers[1].decrement_load();
        assert!(workers.iter().all(|worker| worker.load() == 0));
    }

    #[test]
    fn no_healthy_worker_returns_none() {
        let policy = BoundedConsistentHashingPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let key = key_for_worker(&workers, &ring, 0);
        for worker in &workers {
            worker.set_healthy(false);
        }

        let (result, branch) = select_explicit(&policy, &workers, &ring, &key);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }
}
