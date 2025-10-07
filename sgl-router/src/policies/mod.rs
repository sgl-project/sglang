//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use crate::core::Worker;
use axum::http::HeaderMap;
use std::fmt::Debug;
use std::sync::Arc;

mod cache_aware;
mod factory;
mod load_aware;
mod power_of_two;
mod random;
mod registry;
mod round_robin;
mod rule_based;

pub use cache_aware::CacheAwarePolicy;
pub use factory::PolicyFactory;
pub use load_aware::LoadAwarePolicy;
pub use power_of_two::PowerOfTwoPolicy;
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;
pub use rule_based::RuleBasedPolicy;

/// Routing context that bundles request information for header-aware policies
#[derive(Debug, Clone)]
pub struct RoutingContext<'a> {
    /// Request headers (optional, may not be present for all routing calls)
    pub headers: Option<&'a HeaderMap>,
    /// Request text for cache-aware routing (optional)
    pub request_text: Option<&'a str>,
    /// Model ID for model-specific routing (optional)
    pub model_id: Option<&'a str>,
}

impl<'a> RoutingContext<'a> {
    /// Create a new routing context
    pub fn new(
        headers: Option<&'a HeaderMap>,
        request_text: Option<&'a str>,
        model_id: Option<&'a str>,
    ) -> Self {
        Self {
            headers,
            request_text,
            model_id,
        }
    }

    /// Create a minimal context with just request text (for backward compatibility)
    pub fn from_text(request_text: Option<&'a str>) -> Self {
        Self {
            headers: None,
            request_text,
            model_id: None,
        }
    }
}

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers using routing context
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Uses RoutingContext to provide request headers, text, and model information
    /// for routing decisions.
    fn select_worker(&self, workers: &[Arc<dyn Worker>], context: &RoutingContext)
        -> Option<usize>;

    /// Select a pair of workers (prefill and decode) for PD routing using routing context
    ///
    /// Returns indices of (prefill_worker, decode_worker) from their respective arrays.
    /// Default implementation uses select_worker for each array independently.
    fn select_worker_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        context: &RoutingContext,
    ) -> Option<(usize, usize)> {
        // Default implementation: independently select from each pool
        let prefill_idx = self.select_worker(prefill_workers, context)?;
        let decode_idx = self.select_worker(decode_workers, context)?;
        Some((prefill_idx, decode_idx))
    }

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // Default: no-op for policies that don't use load information
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
