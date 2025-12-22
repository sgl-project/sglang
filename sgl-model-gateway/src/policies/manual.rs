//! Manual routing policy based on routing_id

use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use rand::Rng;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;
use crate::observability::metrics::Metrics;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionBranch {
    NoHealthyWorkers,
    RoutingIdFastPathHit,
    RoutingIdSlowPathOccupiedHit,
    RoutingIdSlowPathOccupiedMiss,
    RoutingIdSlowPathVacant,
    RandomFallback,
}

impl ExecutionBranch {
    // TODO auto generate
    fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::RoutingIdFastPathHit => "routing_id_fast_path_hit",
            Self::RoutingIdSlowPathOccupiedHit => "routing_id_slow_path_occupied_hit",
            Self::RoutingIdSlowPathOccupiedMiss => "routing_id_slow_path_occupied_miss",
            Self::RoutingIdSlowPathVacant => "routing_id_slow_path_vacant",
            Self::RandomFallback => "random_fallback",
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
                return (idx, ExecutionBranch::RoutingIdFastPathHit);
            }
        }

        // Slow path
        match self.routing_map.entry(routing_id) {
            Entry::Occupied(mut entry) => {
                if let Some(idx) =
                    find_healthy_worker(&entry.get().candi_worker_urls, workers, healthy_indices)
                {
                    return (idx, ExecutionBranch::RoutingIdSlowPathOccupiedHit);
                }
                let selected_idx = random_select(healthy_indices);
                entry
                    .get_mut()
                    .push_bounded(workers[selected_idx].url().to_string());
                (selected_idx, ExecutionBranch::RoutingIdSlowPathOccupiedMiss)
            }
            Entry::Vacant(entry) => {
                let selected_idx = random_select(healthy_indices);
                entry.insert(RoutingInfo {
                    candi_worker_urls: vec![workers[selected_idx].url().to_string()],
                });
                (selected_idx, ExecutionBranch::RoutingIdSlowPathVacant)
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
            ExecutionBranch::RandomFallback,
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

        let routing_id = "user-123";
        let info = SelectWorkerInfo {
            routing_id: Some(routing_id),
            ..Default::default()
        };
        let first_idx = policy.select_worker(&workers, &info).unwrap();

        for _ in 0..10 {
            let idx = policy.select_worker(&workers, &info).unwrap();
            assert_eq!(
                idx, first_idx,
                "Same routing_id should route to same worker"
            );
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
            let idx = policy.select_worker(&workers, &info).unwrap();
            *distribution.entry(idx).or_insert(0) += 1;
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
            if let Some(idx) = policy.select_worker(&workers, &info) {
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

        for _ in 0..10 {
            let info = SelectWorkerInfo {
                routing_id: Some("test-routing-id"),
                ..Default::default()
            };
            let idx = policy.select_worker(&workers, &info).unwrap();
            assert_eq!(idx, 1, "Should only select healthy worker");
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
        assert_eq!(policy.select_worker(&workers, &info), None);
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
            if let Some(idx) = policy.select_worker(&workers, &info) {
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
        let first_idx = policy.select_worker(&workers, &info).unwrap();

        workers[first_idx].set_healthy(false);

        let new_idx = policy.select_worker(&workers, &info).unwrap();
        assert_ne!(new_idx, first_idx, "Should remap to healthy worker");

        for _ in 0..10 {
            let idx = policy.select_worker(&workers, &info).unwrap();
            assert_eq!(idx, new_idx, "Should consistently route to new worker");
        }
    }
}
