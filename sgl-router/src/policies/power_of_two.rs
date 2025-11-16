//! Power-of-two choices load balancing policy

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use rand::Rng;
use tracing::info;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, DPLoadManager};
use crate::{core::Worker, metrics::RouterMetrics};

/// Power-of-two choices policy
///
/// Randomly selects two workers and routes to the one with lower load.
/// This provides good load distribution with minimal coordination overhead.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
    dp_load_manager: DPLoadManager,
}

impl PowerOfTwoPolicy {
    pub fn new() -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
            dp_load_manager: DPLoadManager::new(),
        }
    }

    fn get_worker_load(&self, worker: &dyn Worker) -> isize {
        // First check cached loads (from external monitoring)
        if let Ok(loads) = self.cached_loads.read() {
            if let Some(&load) = loads.get(worker.url()) {
                return load;
            }
        }

        // Fall back to local load counter
        worker.load() as isize
    }
}

impl LoadBalancingPolicy for PowerOfTwoPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        // Select two random workers
        let mut rng = rand::rng();
        let idx1 = rng.random_range(0..healthy_indices.len());
        let mut idx2 = rng.random_range(0..healthy_indices.len());

        // Ensure we pick two different workers
        while idx2 == idx1 {
            idx2 = rng.random_range(0..healthy_indices.len());
        }

        let worker_idx1 = healthy_indices[idx1];
        let worker_idx2 = healthy_indices[idx2];

        // Compare loads and select the less loaded one
        let load1 = self.get_worker_load(workers[worker_idx1].as_ref());
        let load2 = self.get_worker_load(workers[worker_idx2].as_ref());

        // Log selection for debugging
        let selected_idx = if load1 <= load2 {
            worker_idx1
        } else {
            worker_idx2
        };

        info!(
            "Power-of-two selection: {}={} vs {}={} -> selected {}",
            workers[worker_idx1].url(),
            load1,
            workers[worker_idx2].url(),
            load2,
            workers[selected_idx].url()
        );

        // Increment processed counter
        workers[selected_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[selected_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "power_of_two"
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        return self.dp_load_manager.update_dp_loads(loads);
    }

    fn get_lowest_dp_load(&self, worker: &dyn Worker) -> Option<isize> {
        return self.dp_load_manager.get_lowest_dp_load(worker);
    }

    fn load_increment(&self, worker: &dyn Worker, dp_rank: isize, tokens: isize) {
        return self.dp_load_manager.load_increment(worker, dp_rank, tokens);
    }
}

impl Default for PowerOfTwoPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_power_of_two_selection() {
        let policy = PowerOfTwoPolicy::new();
        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker3 = BasicWorkerBuilder::new("http://w3:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Set different loads
        for _ in 0..10 {
            worker1.increment_load();
        }
        for _ in 0..5 {
            worker2.increment_load();
        }
        // worker3 has load 0

        let workers: Vec<Arc<dyn Worker>> =
            vec![Arc::new(worker1), Arc::new(worker2), Arc::new(worker3)];

        // Run multiple selections
        let mut selected_counts = [0; 3];
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                selected_counts[idx] += 1;
            }
        }

        // Worker with lowest load (worker3) should be selected most often
        assert!(selected_counts[2] > selected_counts[1]);
        assert!(selected_counts[1] > selected_counts[0]);
    }

    #[test]
    fn test_power_of_two_with_cached_loads() {
        let policy = PowerOfTwoPolicy::new();
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

        // Update cached loads
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 100);
        loads.insert("http://w2:8000".to_string(), 10);
        policy.update_loads(&loads);

        // Should prefer worker2 with lower cached load
        let mut w2_selected = 0;
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        // Worker2 should be selected significantly more often
        assert!(w2_selected > 35); // Should win most of the time
    }

    #[test]
    fn test_power_of_two_single_worker() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        // With single worker, should always select it
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }
}
