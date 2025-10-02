//! Rule-based worker selection policy
//!
//! This module provides a deterministic load balancing policy using lexicographic ordering
//! with support for request header filtering.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RoutingContext};
use crate::core::Worker;
use std::cmp::Ordering;
use std::sync::Arc;
use tracing::{debug, warn};

/// Rule-based policy: ordering by priority, cost then load
///
/// This policy enables fine-grained control over worker selection through request headers:
/// - `x-worker-priority`: Minimum priority threshold (filters out lower priority workers)
/// - `x-max-cost`: Maximum cost threshold (filters out expensive workers)
///
/// **Selection criteria (lexicographic ordering):**
/// 1. Primary: Highest priority
/// 2. Tiebreaker: Lowest cost
/// 3. Second tiebreaker: Lowest load
#[derive(Debug, Clone)]
pub struct RuleBasedPolicy {
    name: &'static str,
}

impl RuleBasedPolicy {
    pub fn new() -> Self {
        Self { name: "rule_based" }
    }

    /// Parse priority threshold from headers
    fn get_priority_threshold(context: &RoutingContext) -> Option<u32> {
        context
            .headers?
            .get("x-worker-priority")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok())
    }

    /// Parse max cost from headers
    fn get_max_cost(context: &RoutingContext) -> Option<f32> {
        context
            .headers?
            .get("x-max-cost")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f32>().ok())
    }

    /// Compare two workers using lexicographic ordering
    /// Returns Ordering indicating which worker should be preferred
    fn compare_workers(a: &Arc<dyn Worker>, b: &Arc<dyn Worker>) -> Ordering {
        // Primary: Higher priority is better
        match b.priority().cmp(&a.priority()) {
            Ordering::Equal => {
                // Tiebreaker: Lower cost is better
                match a.cost().partial_cmp(&b.cost()).unwrap_or(Ordering::Equal) {
                    Ordering::Equal => {
                        // Second tiebreaker: Lower load is better
                        a.load().cmp(&b.load())
                    }
                    other => other,
                }
            }
            other => other,
        }
    }

    /// Select worker using lexicographic ordering
    fn select_worker_by_rules(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        context: &RoutingContext,
    ) -> Option<usize> {
        if healthy_indices.is_empty() {
            return None;
        }

        // Parse header filters
        let priority_threshold = Self::get_priority_threshold(context);
        let max_cost = Self::get_max_cost(context);

        // Apply filters
        let filtered_indices: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| {
                let worker = &workers[idx];

                // Priority filter
                if let Some(min_priority) = priority_threshold {
                    if worker.priority() < min_priority {
                        return false;
                    }
                }

                // Cost filter
                if let Some(max_cost_val) = max_cost {
                    if worker.cost() > max_cost_val {
                        return false;
                    }
                }

                true
            })
            .collect();

        if filtered_indices.is_empty() {
            warn!(
                "RuleBasedPolicy: All workers filtered out by priority/cost thresholds. \
                 Priority threshold: {:?}, Max cost: {:?}",
                priority_threshold, max_cost
            );
            return None;
        }

        // Select best worker using lexicographic ordering
        let best_idx = filtered_indices
            .iter()
            .copied()
            .min_by(|&a, &b| Self::compare_workers(&workers[a], &workers[b]))
            .unwrap();

        debug!(
            "RuleBasedPolicy selected worker {} (priority={}, load={}, cost={})",
            workers[best_idx].url(),
            workers[best_idx].priority(),
            workers[best_idx].load(),
            workers[best_idx].cost()
        );

        Some(best_idx)
    }
}

impl Default for RuleBasedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for RuleBasedPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        context: &RoutingContext,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        self.select_worker_by_rules(workers, &healthy_indices, context)
    }

    fn select_worker_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        context: &RoutingContext,
    ) -> Option<(usize, usize)> {
        let prefill_healthy = get_healthy_worker_indices(prefill_workers);
        let decode_healthy = get_healthy_worker_indices(decode_workers);

        let prefill_idx =
            self.select_worker_by_rules(prefill_workers, &prefill_healthy, context)?;
        let decode_idx = self.select_worker_by_rules(decode_workers, &decode_healthy, context)?;

        Some((prefill_idx, decode_idx))
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use axum::http::HeaderMap;

    fn create_test_worker(url: &str, priority: u32, cost: f32, load: usize) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url.to_string())
            .worker_type(WorkerType::Regular)
            .label("priority", priority.to_string())
            .label("cost", cost.to_string())
            .build();

        // Set load
        for _ in 0..load {
            worker.increment_load();
        }

        Arc::new(worker) as Arc<dyn Worker>
    }

    #[test]
    fn test_rule_based_policy_with_priority_filter() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 30, 1.0, 2), // Low priority, filtered out
            create_test_worker("http://w2", 60, 1.0, 5), // High priority
            create_test_worker("http://w3", 80, 1.0, 3), // Highest priority, should win
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "50".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        assert_eq!(idx, 2); // Should select w3 with highest priority
    }

    #[test]
    fn test_rule_based_policy_with_cost_filter() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 2), // Same priority, lower load, should win
            create_test_worker("http://w2", 50, 5.0, 1), // Too expensive, filtered out
            create_test_worker("http://w3", 50, 2.0, 3), // Same priority, higher load
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-max-cost", "3.0".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // w2 filtered out by cost. Between w1 and w3, same priority, so w1 wins (lower load)
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_rule_based_policy_all_filtered_by_priority() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 30, 1.0, 2),
            create_test_worker("http://w2", 40, 1.0, 3),
            create_test_worker("http://w3", 45, 1.0, 1),
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "50".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context);
        assert_eq!(idx, None); // All workers filtered out
    }

    #[test]
    fn test_rule_based_policy_all_filtered_by_cost() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 50, 3.0, 2),
            create_test_worker("http://w2", 50, 4.0, 3),
            create_test_worker("http://w3", 50, 5.0, 1),
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-max-cost", "2.0".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context);
        assert_eq!(idx, None); // All workers too expensive
    }

    #[test]
    fn test_rule_based_policy_combined_filters() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 30, 1.0, 2), // Low priority, filtered
            create_test_worker("http://w2", 60, 5.0, 3), // High cost, filtered
            create_test_worker("http://w3", 70, 2.0, 4), // Pass both filters
            create_test_worker("http://w4", 80, 1.5, 1), // Pass both, should win
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "50".parse().unwrap());
        headers.insert("x-max-cost", "3.0".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        assert_eq!(idx, 3); // w4 has highest priority among filtered workers
    }

    #[test]
    fn test_rule_based_policy_no_headers() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 80, 1.0, 5), // Highest priority, should win
            create_test_worker("http://w2", 50, 1.0, 2),
            create_test_worker("http://w3", 30, 1.0, 8),
        ];

        let context = RoutingContext::new(None, None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Without headers, lexicographic ordering: highest priority wins
        assert_eq!(idx, 0); // w1 with highest priority
    }

    #[test]
    fn test_rule_based_policy_invalid_priority_header() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 80, 1.0, 2), // Higher priority, should win
            create_test_worker("http://w2", 50, 1.0, 3),
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "invalid".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Invalid header should be ignored, w1 should win (higher priority)
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_rule_based_policy_invalid_cost_header() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 2), // Same priority, lower cost, should win
            create_test_worker("http://w2", 50, 5.0, 1), // Same priority, higher cost
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-max-cost", "not-a-number".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Invalid header should be ignored, lexicographic: same priority, so lower cost wins
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_rule_based_policy_negative_priority() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 2),
            create_test_worker("http://w2", 80, 1.0, 3), // Higher priority, should win
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "-10".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Negative priority should fail to parse, lexicographic ordering applies
        assert_eq!(idx, 1); // w2 has higher priority
    }

    #[test]
    fn test_rule_based_policy_zero_priority_threshold() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 0, 1.0, 5),
            create_test_worker("http://w2", 10, 1.0, 3),
            create_test_worker("http://w3", 50, 1.0, 2),
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "0".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // All workers should pass (priority >= 0)
        assert_eq!(idx, 2); // w3 has highest priority
    }

    #[test]
    fn test_rule_based_policy_zero_max_cost() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 50, 0.0, 2), // Zero cost
            create_test_worker("http://w2", 50, 1.0, 1),
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-max-cost", "0.0".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Only w1 should pass (cost <= 0.0)
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_rule_based_policy_priority_dominates() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 100, 5.0, 0), // Max priority, high cost, zero load, should win
            create_test_worker("http://w2", 0, 0.0, 10),  // Min priority, zero cost, high load
        ];

        let context = RoutingContext::new(None, None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Lexicographic: priority is primary criterion, w1 wins
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_rule_based_policy_load_tiebreaker() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 100, 5.0, 3), // High priority, high cost
            create_test_worker("http://w2", 100, 1.0, 1), // Same priority, medium cost
            create_test_worker("http://w3", 100, 0.0, 5), // Same priority, lowest cost, should win
        ];

        let context = RoutingContext::new(None, None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Same priority, so cost is tiebreaker: w3 has lowest cost
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_rule_based_policy_cost_tiebreaker() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 50, 3.0, 2), // Same priority, same load, higher cost
            create_test_worker("http://w2", 50, 1.0, 2), // Same priority, same load, lower cost, should win
        ];

        let context = RoutingContext::new(None, None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Same priority and load, so cost is final tiebreaker: w2 has lower cost
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_rule_based_policy_pd_mode_with_priority() {
        let policy = RuleBasedPolicy::default();
        let prefill = vec![
            create_test_worker("http://p1", 30, 1.0, 1),
            create_test_worker("http://p2", 80, 1.0, 2), // High priority
        ];
        let decode = vec![
            create_test_worker("http://d1", 40, 1.0, 1),
            create_test_worker("http://d2", 90, 1.0, 3), // Highest priority
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "50".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let (p_idx, d_idx) = policy
            .select_worker_pair(&prefill, &decode, &context)
            .unwrap();

        assert_eq!(p_idx, 1); // p2 passes filter
        assert_eq!(d_idx, 1); // d2 has highest priority
    }

    #[test]
    fn test_rule_based_policy_pd_mode_one_pool_filtered() {
        let policy = RuleBasedPolicy::default();
        let prefill = vec![
            create_test_worker("http://p1", 30, 1.0, 1), // Below threshold
            create_test_worker("http://p2", 40, 1.0, 2), // Below threshold
        ];
        let decode = vec![
            create_test_worker("http://d1", 60, 1.0, 1),
            create_test_worker("http://d2", 70, 1.0, 2),
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "50".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let result = policy.select_worker_pair(&prefill, &decode, &context);

        assert_eq!(result, None); // All prefill workers filtered out
    }

    #[test]
    fn test_rule_based_policy_respects_priority_with_varying_loads() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 90, 1.0, 5), // High priority, high load, should win
            create_test_worker("http://w2", 50, 1.0, 0), // Medium priority, zero load
            create_test_worker("http://w3", 10, 1.0, 0), // Low priority, zero load, filtered out
        ];

        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", "40".parse().unwrap());

        let context = RoutingContext::new(Some(&headers), None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // w3 filtered out by priority threshold
        // Between w1 and w2, w1 has higher priority, so w1 wins
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_multiple_workers_same_priority_different_loads() {
        let policy = RuleBasedPolicy::default();
        let workers = vec![
            create_test_worker("http://w1", 75, 1.0, 8),
            create_test_worker("http://w2", 75, 1.0, 2), // Same priority, lowest load, should win
            create_test_worker("http://w3", 75, 1.0, 5),
        ];

        let context = RoutingContext::new(None, None, None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Same priority, so load is tiebreaker: w2 has lowest load
        assert_eq!(idx, 1);
    }
}
