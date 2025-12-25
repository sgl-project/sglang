//! Manual routing policy based on routing_id

use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use rand::Rng;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, observability::metrics::Metrics};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionBranch {
    NoHealthyWorkers,
    FastPathHit,
    SlowPathOccupiedHit,
    SlowPathOccupiedMiss,
    SlowPathVacant,
    NoRoutingId,
}

impl ExecutionBranch {
    // TODO auto generate
    fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::FastPathHit => "fast_path_hit",
            Self::SlowPathOccupiedHit => "slow_path_occupied_hit",
            Self::SlowPathOccupiedMiss => "slow_path_occupied_miss",
            Self::SlowPathVacant => "slow_path_vacant",
            Self::NoRoutingId => "no_routing_id",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RoutingId(String);

impl RoutingId {
    fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

const MAX_CANDIDATE_WORKERS: usize = 2;

#[derive(Debug, Clone)]
struct RoutingInfo {
    candi_worker_urls: Vec<String>,
}

impl RoutingInfo {
    fn push_bounded(&mut self, url: String) {
        while self.candi_worker_urls.len() >= MAX_CANDIDATE_WORKERS {
            self.candi_worker_urls.remove(0);
        }
        self.candi_worker_urls.push(url);
    }
}

// TODO may optimize performance
// TODO evict old data periodically
#[derive(Debug, Default)]
pub struct ManualPolicy {
    routing_map: DashMap<RoutingId, RoutingInfo>,
}

impl ManualPolicy {
    pub fn new() -> Self {
        Self {
            routing_map: DashMap::new(),
        }
    }

    fn select_by_routing_id(
        &self,
        workers: &[Arc<dyn Worker>],
        routing_id: &str,
        healthy_indices: &[usize],
    ) -> (usize, ExecutionBranch) {
        let routing_id = RoutingId::new(routing_id);

        // Fast path
        if let Some(info) = self.routing_map.get(&routing_id) {
            if let Some(idx) =
                find_healthy_worker(&info.candi_worker_urls, workers, healthy_indices)
            {
                return (idx, ExecutionBranch::FastPathHit);
            }
        }

        // Slow path
        match self.routing_map.entry(routing_id) {
            Entry::Occupied(mut entry) => {
                if let Some(idx) =
                    find_healthy_worker(&entry.get().candi_worker_urls, workers, healthy_indices)
                {
                    return (idx, ExecutionBranch::SlowPathOccupiedHit);
                }
                let selected_idx = random_select(healthy_indices);
                entry
                    .get_mut()
                    .push_bounded(workers[selected_idx].url().to_string());
                (selected_idx, ExecutionBranch::SlowPathOccupiedMiss)
            }
            Entry::Vacant(entry) => {
                let selected_idx = random_select(healthy_indices);
                entry.insert(RoutingInfo {
                    candi_worker_urls: vec![workers[selected_idx].url().to_string()],
                });
                (selected_idx, ExecutionBranch::SlowPathVacant)
            }
        }
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, ExecutionBranch) {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return (None, ExecutionBranch::NoHealthyWorkers);
        }

        if let Some(routing_id) = info.routing_id {
            if !routing_id.is_empty() {
                let (idx, branch) =
                    self.select_by_routing_id(workers, routing_id, &healthy_indices);
                return (Some(idx), branch);
            }
        }

        (
            Some(random_select(&healthy_indices)),
            ExecutionBranch::NoRoutingId,
        )
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

    fn needs_routing_id(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn find_healthy_worker(
    urls: &[String],
    workers: &[Arc<dyn Worker>],
    healthy_indices: &[usize],
) -> Option<usize> {
    for url in urls {
        if let Some(idx) = find_worker_index_by_url(workers, url) {
            if healthy_indices.contains(&idx) {
                return Some(idx);
            }
        }
    }
    None
}

fn find_worker_index_by_url(workers: &[Arc<dyn Worker>], url: &str) -> Option<usize> {
    workers.iter().position(|w| w.url() == url)
}

// TODO: use load-aware selection later
fn random_select(healthy_indices: &[usize]) -> usize {
    let mut rng = rand::rng();
    let random_idx = rng.random_range(0..healthy_indices.len());
    healthy_indices[random_idx]
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_manual_consistent_routing() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("user-123"),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(first_idx),
                "Same routing_id should route to same worker"
            );
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_different_routing_ids() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let mut distribution = HashMap::new();
        for i in 0..100 {
            let routing_id = format!("user-{}", i);
            let info = SelectWorkerInfo {
                routing_id: Some(&routing_id),
                ..Default::default()
            };
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::SlowPathVacant);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(
            distribution.len() > 1,
            "Should distribute across multiple workers"
        );
    }

    #[test]
    fn test_manual_fallback_random() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let mut counts = HashMap::new();
        for _ in 0..100 {
            let info = SelectWorkerInfo::default();
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::NoRoutingId);
            if let Some(idx) = result {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        assert_eq!(counts.len(), 2, "Random fallback should use all workers");
    }

    #[test]
    fn test_manual_with_unhealthy_workers() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        workers[0].set_healthy(false);

        let info = SelectWorkerInfo {
            routing_id: Some("test-routing-id"),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1), "Should only select healthy worker");
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(1), "Should only select healthy worker");
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_no_healthy_workers() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        workers[0].set_healthy(false);
        let info = SelectWorkerInfo {
            routing_id: Some("test"),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, ExecutionBranch::NoHealthyWorkers);
    }

    #[test]
    fn test_manual_empty_routing_id() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let mut counts = HashMap::new();
        for _ in 0..100 {
            let info = SelectWorkerInfo {
                routing_id: Some(""),
                ..Default::default()
            };
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::NoRoutingId);
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

    #[test]
    fn test_manual_remaps_when_worker_becomes_unhealthy() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("sticky-user"),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[first_idx].set_healthy(false);

        let (new_result, branch) = policy.select_worker_impl(&workers, &info);
        let new_idx = new_result.unwrap();
        assert_ne!(new_idx, first_idx, "Should remap to healthy worker");
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(new_idx),
                "Should consistently route to new worker"
            );
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_empty_workers() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let info = SelectWorkerInfo {
            routing_id: Some("test"),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, ExecutionBranch::NoHealthyWorkers);
    }

    #[test]
    fn test_manual_single_worker() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        let info = SelectWorkerInfo {
            routing_id: Some("single-test"),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(0));
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(0));
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_worker_recovery() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("recovery-test"),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (after_recovery, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            after_recovery,
            Some(first_idx),
            "Should return to original worker after recovery since it's first in candidate list"
        );
        assert_eq!(branch, ExecutionBranch::FastPathHit);
    }

    #[test]
    fn test_manual_max_candidate_workers_eviction() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("eviction-test"),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        workers[second_idx].set_healthy(false);

        let remaining_idx = (0..3).find(|&i| i != first_idx && i != second_idx).unwrap();
        let (third_result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            third_result,
            Some(remaining_idx),
            "Should select the only remaining healthy worker"
        );
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (idx_after_restore, branch) = policy.select_worker_impl(&workers, &info);
        assert_ne!(
            idx_after_restore,
            Some(first_idx),
            "First worker should be evicted from candidates due to MAX_CANDIDATE_WORKERS=2"
        );
        assert_eq!(branch, ExecutionBranch::FastPathHit);
    }

    #[test]
    fn test_manual_execution_branch_fast_path_hit() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("fast-path-test"),
            ..Default::default()
        };

        let _ = policy.select_worker_impl(&workers, &info);

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert!(result.is_some());
        assert_eq!(branch, ExecutionBranch::FastPathHit);
    }

    #[test]
    fn test_manual_execution_branch_no_routing_id() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        let info = SelectWorkerInfo::default();
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert!(result.is_some());
        assert_eq!(branch, ExecutionBranch::NoRoutingId);
    }

    #[test]
    fn test_manual_execution_branch_slow_path_occupied_miss() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("occupied-miss-test"),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[first_idx].set_healthy(false);

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert!(result.is_some());
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);
    }

    #[test]
    fn test_manual_execution_branch_slow_path_occupied_hit() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("occupied-hit-test"),
            ..Default::default()
        };

        let _ = policy.select_worker_impl(&workers, &info);

        policy.routing_map.clear();

        policy.routing_map.insert(
            RoutingId::new("occupied-hit-test"),
            RoutingInfo {
                candi_worker_urls: vec!["http://w1:8000".to_string()],
            },
        );

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert!(result.is_some());
        assert_eq!(branch, ExecutionBranch::FastPathHit);
    }

    #[test]
    fn test_manual_routing_info_push_bounded() {
        let mut info = RoutingInfo {
            candi_worker_urls: vec!["http://w1:8000".to_string()],
        };

        info.push_bounded("http://w2:8000".to_string());
        assert_eq!(info.candi_worker_urls.len(), 2);
        assert_eq!(info.candi_worker_urls[0], "http://w1:8000");
        assert_eq!(info.candi_worker_urls[1], "http://w2:8000");

        info.push_bounded("http://w3:8000".to_string());
        assert_eq!(info.candi_worker_urls.len(), 2);
        assert_eq!(
            info.candi_worker_urls[0], "http://w2:8000",
            "Oldest entry should be removed"
        );
        assert_eq!(info.candi_worker_urls[1], "http://w3:8000");
    }

    #[test]
    fn test_manual_find_healthy_worker_priority() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let urls = vec![
            "http://w1:8000".to_string(),
            "http://w2:8000".to_string(),
            "http://w3:8000".to_string(),
        ];
        let healthy_indices = vec![0, 1, 2];

        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(
            result,
            Some(0),
            "Should return first healthy worker in urls"
        );

        workers[0].set_healthy(false);
        let healthy_indices = vec![1, 2];
        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(result, Some(1), "Should skip unhealthy and return next");

        workers[1].set_healthy(false);
        let healthy_indices = vec![2];
        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(result, Some(2), "Should return last healthy worker");

        workers[2].set_healthy(false);
        let healthy_indices: Vec<usize> = vec![];
        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(result, None, "Should return None when no healthy workers");
    }

    #[test]
    fn test_manual_find_worker_index_by_url() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        assert_eq!(
            find_worker_index_by_url(&workers, "http://w1:8000"),
            Some(0)
        );
        assert_eq!(
            find_worker_index_by_url(&workers, "http://w2:8000"),
            Some(1)
        );
        assert_eq!(
            find_worker_index_by_url(&workers, "http://w3:8000"),
            None,
            "Should return None for unknown URL"
        );
    }

    #[test]
    fn test_manual_policy_name() {
        let policy = ManualPolicy::new();
        assert_eq!(policy.name(), "manual");
    }

    #[test]
    fn test_manual_policy_needs_routing_id() {
        let policy = ManualPolicy::new();
        assert!(policy.needs_routing_id());
    }

    #[test]
    fn test_manual_all_workers_become_unhealthy_then_recover() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo {
            routing_id: Some("all-unhealthy-test"),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[0].set_healthy(false);
        workers[1].set_healthy(false);

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            result, None,
            "Should return None when all workers are unhealthy"
        );
        assert_eq!(branch, ExecutionBranch::NoHealthyWorkers);

        workers[first_idx].set_healthy(true);

        let (after_recovery, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            after_recovery,
            Some(first_idx),
            "Should route to recovered worker in candidate list"
        );
        assert_eq!(branch, ExecutionBranch::FastPathHit);
    }
}
