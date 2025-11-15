use std::{
    collections::HashMap,
    sync::{
        Arc,
        RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

use tracing::{debug, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy,
    PowerOfTwoPolicy, RandomPolicy, RoundRobinPolicy,
};
use crate::{config::types::PolicyConfig, core::Worker};

/// Registry for managing model-to-policy mappings
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Model ID -> Policy instance mapping
    model_policies: Arc<RwLock<HashMap<String, Arc<dyn LoadBalancingPolicy>>>>,

    /// Model ID -> Worker count for cleanup tracking
    model_worker_counts: Arc<RwLock<HashMap<String, usize>>>,

    /// Default policy instance (cached)
    default_policy: Arc<dyn LoadBalancingPolicy>,

    /// Prefill policy for PD mode
    prefill_policy: Arc<RwLock<Option<Arc<dyn LoadBalancingPolicy>>>>,

    /// Decode policy for PD mode
    decode_policy: Arc<RwLock<Option<Arc<dyn LoadBalancingPolicy>>>>,

    /// Enable minimum tokens scheduler for dp group
    dp_minimum_tokens_scheduler: Arc<AtomicBool>,
}

impl PolicyRegistry {
    /// Create a new PolicyRegistry with a default policy
    pub fn new(default_policy_config: PolicyConfig) -> Self {
        let default_policy = Self::create_policy_from_config(&default_policy_config);

        Self {
            model_policies: Arc::new(RwLock::new(HashMap::new())),
            model_worker_counts: Arc::new(RwLock::new(HashMap::new())),
            default_policy,
            prefill_policy: Arc::new(RwLock::new(None)),
            decode_policy: Arc::new(RwLock::new(None)),
            dp_minimum_tokens_scheduler: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn enable_dp_minimum_tokens_scheduler(&self) {
        self.dp_minimum_tokens_scheduler.store(true, Ordering::Relaxed);
    }

    pub fn is_dp_minimum_tokens_scheduler_enabled(&self) -> bool {
        self.dp_minimum_tokens_scheduler.load(Ordering::Relaxed)
    }

    /// Called when a worker is added
    /// Returns the policy that should be used for this worker's model
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // Increment worker count
        {
            let mut counts = self.model_worker_counts.write().unwrap();
            *counts.entry(model_id.to_string()).or_insert(0) += 1;
            debug!(
                "Worker added for model {}, count: {}",
                model_id,
                counts.get(model_id).unwrap()
            );
        }

        // Check if model already has a policy
        {
            let policies = self.model_policies.read().unwrap();
            if let Some(existing_policy) = policies.get(model_id) {
                debug!(
                    "Model {} already has policy: {}",
                    model_id,
                    existing_policy.name()
                );
                return Arc::clone(existing_policy);
            }
        }

        // New model - determine policy
        let policy = self.determine_policy_for_model(model_id, policy_hint);

        info!(
            "Assigning policy {} to new model {}",
            policy.name(),
            model_id
        );

        // Store policy for this model
        {
            let mut policies = self.model_policies.write().unwrap();
            policies.insert(model_id.to_string(), Arc::clone(&policy));
        }

        policy
    }

    /// Called when a worker is removed
    pub fn on_worker_removed(&self, model_id: &str) {
        let should_cleanup = {
            let mut counts = self.model_worker_counts.write().unwrap();
            if let Some(count) = counts.get_mut(model_id) {
                *count = count.saturating_sub(1);
                debug!("Worker removed for model {}, count: {}", model_id, *count);
                if *count == 0 {
                    counts.remove(model_id);
                    true
                } else {
                    false
                }
            } else {
                warn!(
                    "Attempted to remove worker for model {} with no registered workers",
                    model_id
                );
                false
            }
        };

        // Clean up policy if this was the last worker
        if should_cleanup {
            let mut policies = self.model_policies.write().unwrap();
            if let Some(policy) = policies.remove(model_id) {
                info!(
                    "Removed policy {} for model {} (last worker removed)",
                    policy.name(),
                    model_id
                );
                // Policy will be dropped here, cleaning up any resources
                drop(policy);
            }
        }
    }

    /// Get the policy for a model
    pub fn get_policy(&self, model_id: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.model_policies.read().unwrap().get(model_id).cloned()
    }

    /// Get the default policy
    pub fn get_default_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        Arc::clone(&self.default_policy)
    }

    /// Get policy for a model, or default if not found
    pub fn get_policy_or_default(&self, model_id: &str) -> Arc<dyn LoadBalancingPolicy> {
        self.get_policy(model_id)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Determine policy for a new model
    fn determine_policy_for_model(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // 1. Check policy hint from worker
        if let Some(policy_type) = policy_hint {
            debug!("Using policy hint '{}' for model {}", policy_type, model_id);
            return self.create_policy_from_type(policy_type);
        }

        // 2. Use default policy
        debug!("Using default policy for model {}", model_id);
        Arc::clone(&self.default_policy)
    }

    /// Create a policy from a type string
    fn create_policy_from_type(&self, policy_type: &str) -> Arc<dyn LoadBalancingPolicy> {
        match policy_type {
            "round_robin" => Arc::new(RoundRobinPolicy::new()),
            "random" => Arc::new(RandomPolicy::new()),
            "cache_aware" => Arc::new(CacheAwarePolicy::new()),
            "power_of_two" => Arc::new(PowerOfTwoPolicy::new()),
            "bucket" => Arc::new(BucketPolicy::new()),
            _ => {
                warn!("Unknown policy type '{}', using default", policy_type);
                Arc::clone(&self.default_policy)
            }
        }
    }

    /// Create a policy from a PolicyConfig
    fn create_policy_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let cache_config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                };
                Arc::new(CacheAwarePolicy::with_config(cache_config))
            }
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                let config = BucketConfig {
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    bucket_adjust_interval_secs: *bucket_adjust_interval_secs,
                };
                Arc::new(BucketPolicy::with_config(config))
            }
        }
    }

    /// Get current model->policy mappings (for debugging/monitoring)
    pub fn get_all_mappings(&self) -> HashMap<String, String> {
        let policies = self.model_policies.read().unwrap();
        policies
            .iter()
            .map(|(model, policy)| (model.clone(), policy.name().to_string()))
            .collect()
    }

    /// Get worker counts per model
    pub fn get_worker_counts(&self) -> HashMap<String, usize> {
        self.model_worker_counts.read().unwrap().clone()
    }

    /// Clear all policies (useful for testing)
    pub fn clear(&self) {
        let mut policies = self.model_policies.write().unwrap();
        policies.clear();
        let mut counts = self.model_worker_counts.write().unwrap();
        counts.clear();
    }

    /// Set the prefill policy for PD mode
    pub fn set_prefill_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        let mut prefill_policy = self.prefill_policy.write().unwrap();
        *prefill_policy = Some(policy);
    }

    /// Set the decode policy for PD mode
    pub fn set_decode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        let mut decode_policy = self.decode_policy.write().unwrap();
        *decode_policy = Some(policy);
    }

    /// Get the prefill policy for PD mode, or default if not set
    pub fn get_prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        let prefill_policy = self.prefill_policy.read().unwrap();
        prefill_policy
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the decode policy for PD mode, or default if not set
    pub fn get_decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        let decode_policy = self.decode_policy.read().unwrap();
        decode_policy
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get all PowerOfTwo policies that need load updates
    pub fn get_all_power_of_two_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        let mut power_of_two_policies = Vec::new();

        if self.default_policy.name() == "power_of_two" {
            power_of_two_policies.push(Arc::clone(&self.default_policy));
        }

        if let Some(ref policy) = *self.prefill_policy.read().unwrap() {
            if policy.name() == "power_of_two" && !Arc::ptr_eq(policy, &self.default_policy) {
                power_of_two_policies.push(Arc::clone(policy));
            }
        }

        if let Some(ref policy) = *self.decode_policy.read().unwrap() {
            if policy.name() == "power_of_two"
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !self
                    .prefill_policy
                    .read()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                power_of_two_policies.push(Arc::clone(policy));
            }
        }

        let model_policies = self.model_policies.read().unwrap();
        for policy in model_policies.values() {
            if policy.name() == "power_of_two" {
                let already_added = power_of_two_policies.iter().any(|p| Arc::ptr_eq(p, policy));
                if !already_added {
                    power_of_two_policies.push(Arc::clone(policy));
                }
            }
        }

        power_of_two_policies
    }

    pub fn get_all_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        let mut all_policies = Vec::new();

        all_policies.push(Arc::clone(&self.default_policy));

        if let Some(ref policy) = *self.prefill_policy.read().unwrap() {
            if !Arc::ptr_eq(policy, &self.default_policy) {
                all_policies.push(Arc::clone(policy));
            }
        }

        if let Some(ref policy) = *self.decode_policy.read().unwrap() {
            if !Arc::ptr_eq(policy, &self.default_policy)
                && !self
                    .prefill_policy
                    .read()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                all_policies.push(Arc::clone(policy));
            }
        }

        let model_policies = self.model_policies.read().unwrap();
        for policy in model_policies.values() {
            let already_added = all_policies.iter().any(|p| Arc::ptr_eq(p, policy));
            if !already_added {
                all_policies.push(Arc::clone(policy));
            }
        }

        all_policies
    }

    /// Initialize cache-aware policy with workers if applicable
    /// This should be called after workers are registered for a model
    pub fn init_cache_aware_policy(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    debug!(
                        "Initializing cache-aware policy with {} workers for model {}",
                        workers.len(),
                        model_id
                    );
                    cache_aware.init_workers(workers);
                }
            }
        }
    }

    /// Remove a worker from cache-aware policy if applicable
    /// This should be called when a worker is being removed
    pub fn remove_worker_from_cache_aware(&self, model_id: &str, worker_url: &str) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.remove_worker_by_url(worker_url);
                    debug!(
                        "Removed worker {} from cache-aware policy for model {}",
                        worker_url, model_id
                    );
                }
            }
        }
    }

    /// Initialize cache-aware policies for PD mode (prefill and decode)
    pub fn init_pd_cache_aware_policies(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) {
        // Initialize prefill policy if it's cache-aware
        if let Some(prefill_policy) = self.prefill_policy.read().unwrap().as_ref() {
            if prefill_policy.name() == "cache_aware" {
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill cache-aware policy with {} workers",
                            prefill_workers.len()
                        );
                        cache_aware.init_workers(prefill_workers);
                    }
                }
            }
        }

        // Initialize decode policy if it's cache-aware
        if let Some(decode_policy) = self.decode_policy.read().unwrap().as_ref() {
            if decode_policy.name() == "cache_aware" {
                if let Some(cache_aware) = decode_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !decode_workers.is_empty() {
                        debug!(
                            "Initializing decode cache-aware policy with {} workers",
                            decode_workers.len()
                        );
                        cache_aware.init_workers(decode_workers);
                    }
                }
            }
        }
    }

    pub fn init_pd_bucket_policies(&self, prefill_workers: &[Arc<dyn Worker>]) {
        // Initialize prefill policy if it's bucket
        if let Some(prefill_policy) = self.prefill_policy.read().unwrap().as_ref() {
            if prefill_policy.name() == "bucket" {
                if let Some(bucket) = prefill_policy.as_any().downcast_ref::<BucketPolicy>() {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill bucket policy with {} workers",
                            prefill_workers.len()
                        );
                        bucket.init_prefill_worker_urls(prefill_workers);
                    }
                }
            }
        }
    }
}

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRegistry")
            .field("model_policies", &self.model_policies)
            .field("model_worker_counts", &self.model_worker_counts)
            .field("default_policy", &self.default_policy.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_registry_basic() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // First worker of a model sets the policy
        let policy1 = registry.on_worker_added("llama-3", Some("cache_aware"));
        assert_eq!(policy1.name(), "cache_aware");

        // Second worker of same model uses existing policy
        let policy2 = registry.on_worker_added("llama-3", Some("round_robin"));
        assert_eq!(policy2.name(), "cache_aware"); // Ignores hint, uses existing

        // Different model can have different policy
        let policy3 = registry.on_worker_added("gpt-4", Some("random"));
        assert_eq!(policy3.name(), "random");

        // Check mappings
        let mappings = registry.get_all_mappings();
        assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
        assert_eq!(mappings.get("gpt-4").unwrap(), "random");

        // Check worker counts
        let counts = registry.get_worker_counts();
        assert_eq!(*counts.get("llama-3").unwrap(), 2);
        assert_eq!(*counts.get("gpt-4").unwrap(), 1);
    }

    #[test]
    fn test_policy_registry_cleanup() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Add workers
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.on_worker_added("llama-3", None);
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&2));

        // Remove one worker - policy should remain
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_some());
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&1));

        // Remove last worker - policy should be cleaned up
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_none());
        assert_eq!(registry.get_worker_counts().get("llama-3"), None);
    }

    #[test]
    fn test_default_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // No hint, no template - uses default
        let policy = registry.on_worker_added("unknown-model", None);
        assert_eq!(policy.name(), "round_robin");

        // Get default directly
        let default = registry.get_default_policy();
        assert_eq!(default.name(), "round_robin");
    }
}
