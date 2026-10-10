//! Power-of-two choices load balancing policy

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use rand::Rng;
use tracing::{debug, info};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

/// Load compared between candidates, set by `SGLANG_ROUTER_P2C_LOAD_METRIC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadMetric {
    /// Polled token load, then in-flight requests.
    Tokens,
    /// In-flight requests, then polled token load.
    Requests,
    /// As `Requests`, over every worker with a polled load.
    LeastRequests,
}

/// Power-of-two choices policy
///
/// Randomly selects two workers and routes to the one with lower load.
/// This provides good load distribution with minimal coordination overhead.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
    metric: LoadMetric,
}

impl PowerOfTwoPolicy {
    pub fn new() -> Self {
        let metric = match std::env::var("SGLANG_ROUTER_P2C_LOAD_METRIC").as_deref() {
            Ok("requests") => LoadMetric::Requests,
            Ok("least_requests") => LoadMetric::LeastRequests,
            _ => LoadMetric::Tokens,
        };
        Self::with_metric(metric)
    }

    fn with_metric(metric: LoadMetric) -> Self {
        info!("Power-of-two load metric: {:?}", metric);
        Self {
            cached_loads: RwLock::new(HashMap::new()),
            metric,
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for PowerOfTwoPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        let mut rng = rand::rng();
        let loads_guard = self.cached_loads.read().ok();
        let tokens = |idx: usize| {
            loads_guard
                .as_ref()
                .and_then(|m| m.get(workers[idx].url()).copied())
        };

        if self.metric == LoadMetric::LeastRequests {
            // Workers without a polled load are skipped unless none has one.
            let reported: Vec<usize> = healthy_indices
                .iter()
                .copied()
                .filter(|&idx| tokens(idx).is_some())
                .collect();
            let candidates = if reported.is_empty() {
                &healthy_indices
            } else {
                &reported
            };
            // A random start spreads ties instead of favouring low indices.
            let start = rng.random_range(0..candidates.len());
            let selected_idx = candidates
                .iter()
                .cycle()
                .skip(start)
                .take(candidates.len())
                .copied()
                .min_by_key(|&idx| (workers[idx].load(), tokens(idx).unwrap_or(0)))?;
            workers[selected_idx].increment_processed();
            return Some(selected_idx);
        }

        // Select two random workers - use offset to guarantee different selection in O(1)
        let idx1 = rng.random_range(0..healthy_indices.len());
        // Pick idx2 from remaining indices: offset by 1 + random from (len-1) to guarantee different
        let idx2 =
            (idx1 + 1 + rng.random_range(0..healthy_indices.len() - 1)) % healthy_indices.len();

        let worker_idx1 = healthy_indices[idx1];
        let worker_idx2 = healthy_indices[idx2];
        let worker1 = &workers[worker_idx1];
        let worker2 = &workers[worker_idx2];

        // Compare the primary load for BOTH workers and fall back to the other
        // one when it is tied or, for tokens, missing on either side.
        let (r1, r2) = (worker1.load() as isize, worker2.load() as isize);
        let (load1, load2) = match (self.metric, tokens(worker_idx1), tokens(worker_idx2)) {
            (LoadMetric::Tokens, Some(t1), Some(t2)) if t1 != t2 => (t1, t2),
            (LoadMetric::Requests, Some(t1), Some(t2)) if r1 == r2 => (t1, t2),
            _ => (r1, r2),
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

    #[tokio::test]
    async fn test_power_of_two_selection() {
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
        let info = SelectWorkerInfo::default();
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                selected_counts[idx] += 1;
            }
        }

        // Worker with lowest load (worker3) should be selected most often
        assert!(selected_counts[2] > selected_counts[1]);
        assert!(selected_counts[1] > selected_counts[0]);
    }

    #[tokio::test]
    async fn test_power_of_two_with_cached_loads() {
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
        let info = SelectWorkerInfo::default();
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        // Worker2 should be selected significantly more often
        assert!(w2_selected > 35); // Should win most of the time
    }

    #[tokio::test]
    async fn test_power_of_two_single_worker() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        // With single worker, should always select it
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            Some(0)
        );
    }

    #[tokio::test]
    async fn test_reproduce_incompatible_metric_bug() {
        use std::{collections::HashMap, sync::Arc};

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

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker_a), Arc::new(worker_b)];

        // 4. Simulate LoadMonitor update:
        // Only Worker A gets a token report. Worker B is missing (e.g. monitor failure).
        let mut loads = HashMap::new();
        loads.insert("http://worker_a:8000".to_string(), 50_000); // 50k tokens load
        policy.update_loads(&loads);

        // 5. Run selection
        let selected_idx = policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .await
            .expect("Should select a worker");

        // 6. Verify the Fix
        // Logic:
        // - Worker A has token load (50k) but Worker B has NO token load.
        // - Policy should fallback to request counts for BOTH.
        // - A has 0 requests, B has 5 requests.
        // - 0 <= 5, so A should be selected.

        if selected_idx == 0 {
            println!("Bug Fixed: System correctly fell back to request counts and selected idle Worker A.");
        } else {
            println!(
                "Bug PERSISTS: Selected Worker B (Load: 5 reqs) over Worker A (Load: 50k tokens)"
            );
        }

        // Assert that the CORRECT worker (A, index 0) is selected
        assert_eq!(
            selected_idx, 0,
            "The policy failed to handle incompatible metrics. Should select idle Worker A."
        );
    }
    #[tokio::test]
    async fn test_power_of_two_edge_cases() {
        use std::{collections::HashMap, sync::Arc};

        use crate::core::{BasicWorkerBuilder, WorkerType};

        let policy = PowerOfTwoPolicy::new();

        // Helper to create a worker with specific request load
        let create_worker = |url: &str, reqs: usize| {
            let w = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build();
            for _ in 0..reqs {
                w.increment_load();
            }
            Arc::new(w)
        };

        //  Scenario 1: Happy Path (Both have Token Data)
        // Worker A: 10 requests, but only 1,000 tokens (Light usage) -> Should be CHOSEN
        // Worker B:  2 requests, but 100,000 tokens (Heavy usage) -> Should be AVOIDED
        // This proves we use high-fidelity metrics when available, ignoring request counts.
        let w_a = create_worker("http://a:8000", 10);
        let w_b = create_worker("http://b:8000", 2);
        let workers_1: Vec<Arc<dyn Worker>> = vec![w_a.clone(), w_b.clone()];

        let mut loads_1 = HashMap::new();
        loads_1.insert("http://a:8000".to_string(), 1_000);
        loads_1.insert("http://b:8000".to_string(), 100_000);
        policy.update_loads(&loads_1);

        let idx_1 = policy
            .select_worker(&workers_1, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(
            idx_1, 0,
            "Happy Path Failed: Should select Worker A (fewer tokens) despite higher request count"
        );

        // Scenario 2: Partial Failure (Worker A has tokens, Worker B is missing)
        // Worker A: 10 requests, 1,000 tokens (Cached)
        // Worker B:  2 requests, MISSING cache
        // Logic: Fallback to requests -> Compare 10 (A) vs 2 (B) -> Select B
        let w_c = create_worker("http://c:8000", 10);
        let w_d = create_worker("http://d:8000", 2);
        let workers_2: Vec<Arc<dyn Worker>> = vec![w_c.clone(), w_d.clone()];

        let mut loads_2 = HashMap::new();
        loads_2.insert("http://c:8000".to_string(), 1_000);
        // http://d:8000 is MISSING
        policy.update_loads(&loads_2);

        let idx_2 = policy
            .select_worker(&workers_2, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(idx_2, 1, "Partial Fail 1 Failed: Should fallback to requests and select Worker B (fewer requests)");

        // Scenario 3: Partial Failure (Worker A is missing, Worker B has tokens)
        // Worker A:  2 requests, MISSING cache
        // Worker B: 10 requests, 1,000 tokens (Cached)
        // Logic: Fallback to requests -> Compare 2 (A) vs 10 (B) -> Select A
        let w_e = create_worker("http://e:8000", 2);
        let w_f = create_worker("http://f:8000", 10);
        let workers_3: Vec<Arc<dyn Worker>> = vec![w_e.clone(), w_f.clone()];

        let mut loads_3 = HashMap::new();
        // http://e:8000 is MISSING
        loads_3.insert("http://f:8000".to_string(), 1_000);
        policy.update_loads(&loads_3);

        let idx_3 = policy
            .select_worker(&workers_3, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(idx_3, 0, "Partial Fail 2 Failed: Should fallback to requests and select Worker A (fewer requests)");

        // Scenario 4: Total Failure (Both missing)
        // Worker A: 5 requests
        // Worker B: 3 requests
        // Logic: Requests vs Requests -> Select B
        let w_g = create_worker("http://g:8000", 5);
        let w_h = create_worker("http://h:8000", 3);
        let workers_4: Vec<Arc<dyn Worker>> = vec![w_g.clone(), w_h.clone()];

        let loads_4 = HashMap::new();
        policy.update_loads(&loads_4);

        let idx_4 = policy
            .select_worker(&workers_4, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(
            idx_4, 1,
            "Total Fail Failed: Should select Worker B based on request count"
        );

        println!("All edge case tests passed successfully.");
    }

    /// Workers with the given in-flight requests and polled token loads.
    fn workers_with_loads(
        policy: &PowerOfTwoPolicy,
        loads: &[(usize, Option<isize>)],
    ) -> Vec<Arc<dyn Worker>> {
        let mut tokens = HashMap::new();
        let workers = loads
            .iter()
            .enumerate()
            .map(|(rank, &(requests, token_load))| {
                let worker = BasicWorkerBuilder::new(format!("http://d:8000@{rank}"))
                    .worker_type(WorkerType::Decode)
                    .build();
                (0..requests).for_each(|_| worker.increment_load());
                if let Some(t) = token_load {
                    tokens.insert(worker.url().to_string(), t);
                }
                Arc::new(worker) as Arc<dyn Worker>
            })
            .collect();
        policy.update_loads(&tokens);
        workers
    }

    #[tokio::test]
    async fn test_requests_metric_prefers_fewer_in_flight_requests() {
        let policy = PowerOfTwoPolicy::with_metric(LoadMetric::Requests);
        let workers = workers_with_loads(&policy, &[(5, Some(10)), (2, Some(1000))]);
        for _ in 0..20 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await;
            assert_eq!(idx, Some(1));
        }
    }

    #[tokio::test]
    async fn test_least_requests_skips_workers_without_load() {
        let policy = PowerOfTwoPolicy::with_metric(LoadMetric::LeastRequests);
        let workers = workers_with_loads(
            &policy,
            &[(0, None), (5, Some(1)), (1, Some(1000)), (3, Some(1))],
        );
        for _ in 0..20 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await;
            assert_eq!(idx, Some(2));
        }
    }
}
