//! Power-of-two choices load balancing policy

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use rand::Rng;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy};
use crate::{core::Worker, observability::metrics::RouterMetrics};

/// Power-of-two choices policy
///
/// Randomly selects two workers and routes to the one with lower load.
/// This provides good load distribution with minimal coordination overhead.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
}

impl PowerOfTwoPolicy {
    pub fn new() -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
        }
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

        // Select two random workers - use offset to guarantee different selection in O(1)
        let mut rng = rand::rng();
        let idx1 = rng.random_range(0..healthy_indices.len());
        // Pick idx2 from remaining indices: offset by 1 + random from (len-1) to guarantee different
        let idx2 =
            (idx1 + 1 + rng.random_range(0..healthy_indices.len() - 1)) % healthy_indices.len();

        let worker_idx1 = healthy_indices[idx1];
        let worker_idx2 = healthy_indices[idx2];
        let worker1 = &workers[worker_idx1];
        let worker2 = &workers[worker_idx2];


        // Access cached loads safely
        let loads_guard = self.cached_loads.read().ok();

        // Try to get high-fidelity token loads for BOTH workers
        let load1_tokens = loads_guard.as_ref().and_then(|m| m.get(worker1.url()).copied());
        let load2_tokens = loads_guard.as_ref().and_then(|m| m.get(worker2.url()).copied());

        // If either worker is missing token data (e.g. monitor failure),
        // we must degrade BOTH to request counts to ensure fairness.
        let (load1, load2) = match (load1_tokens, load2_tokens) {
            (Some(t1), Some(t2)) => {
                // Both have token data. Compare Tokens.
                (t1, t2)
            },
            _ => {
                // If One or both are missing token data.
                // Fallback to local request counts for BOTH.
                (worker1.load() as isize, worker2.load() as isize)
            }
        };

        // Select worker with lower load
        let selected_idx = if load1 <= load2 {
            worker_idx1
        } else {
            worker_idx2
        };

        debug!(
            "Power-of-two selection: {}={} vs {}={} -> selected {}",
            worker1.url(),
            load1,
            worker2.url(),
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

    #[test]
    fn test_reproduce_incompatible_metric_bug() {
        use std::collections::HashMap;
        use std::sync::Arc;
        use crate::core::{BasicWorkerBuilder, WorkerType};

        // 1. Setup the policy
        let policy = PowerOfTwoPolicy::new();

        // 2. Create Worker A: Idle (0 reqs), but has high token usage in cache
        let worker_a = BasicWorkerBuilder::new("http://worker_a:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // 3. Create Worker B: Busy (5 reqs), but missing from cache
        let worker_b = BasicWorkerBuilder::new("http://worker_b:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Manually increment load on Worker B to simulate active requests
        for _ in 0..5 {
            worker_b.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(worker_a),
            Arc::new(worker_b)
        ];

        // 4. Simulate LoadMonitor update:
        // Only Worker A gets a token report. Worker B is missing (e.g. monitor failure).
        let mut loads = HashMap::new();
        loads.insert("http://worker_a:8000".to_string(), 50_000); // 50k tokens load
        policy.update_loads(&loads);

        // 5. Run selection
        let selected_idx = policy.select_worker(&workers, None).expect("Should select a worker");

        // 6. Verify the Fix
        // Logic:
        // - Worker A has token load (50k) but Worker B has NO token load.
        // - Policy should fallback to request counts for BOTH.
        // - A has 0 requests, B has 5 requests.
        // - 0 <= 5, so A should be selected.

        if selected_idx == 0 {
            println!("Bug Fixed: System correctly fell back to request counts and selected idle Worker A.");
        } else {
            println!("Bug PERSISTS: Selected Worker B (Load: 5 reqs) over Worker A (Load: 50k tokens)");
        }

        // Assert that the CORRECT worker (A, index 0) is selected
        assert_eq!(selected_idx, 0, "The policy failed to handle incompatible metrics. Should select idle Worker A.");
    }
}
