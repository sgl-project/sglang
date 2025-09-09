# Policy Registry Design

## Overview

A centralized PolicyRegistry that manages model→policy mappings, handles dynamic policy assignment, and cleans up when models are removed.

## Core Rules

1. **New Model + No Policy in Payload** → Use router's default policy
2. **New Model + Policy in Payload** → Use specified policy
3. **Existing Model + Any Policy in Payload** → Use existing policy (ignore payload)
4. **Last Worker Removed** → Remove model's policy mapping

## Architecture

```rust
// src/policies/registry.rs

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Manages model-to-policy mappings dynamically
pub struct PolicyRegistry {
    /// Model ID -> Policy instance
    model_policies: Arc<RwLock<HashMap<String, Arc<dyn LoadBalancingPolicy>>>>,
    
    /// Model ID -> Worker count (for cleanup tracking)
    model_worker_counts: Arc<RwLock<HashMap<String, usize>>>,
    
    /// Default policy factory
    default_policy_factory: Arc<dyn Fn() -> Arc<dyn LoadBalancingPolicy> + Send + Sync>,
}

impl PolicyRegistry {
    pub fn new(default_policy: PolicyConfig) -> Self {
        let factory = Arc::new(move || -> Arc<dyn LoadBalancingPolicy> {
            create_policy(&default_policy)
        });
        
        Self {
            model_policies: Arc::new(RwLock::new(HashMap::new())),
            model_worker_counts: Arc::new(RwLock::new(HashMap::new())),
            default_policy_factory: factory,
        }
    }
    
    /// Called when a worker is added
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // Increment worker count
        {
            let mut counts = self.model_worker_counts.write().unwrap();
            *counts.entry(model_id.to_string()).or_insert(0) += 1;
        }
        
        // Check if model already has a policy
        {
            let policies = self.model_policies.read().unwrap();
            if let Some(existing_policy) = policies.get(model_id) {
                // Model exists - use existing policy, ignore hint
                return Arc::clone(existing_policy);
            }
        }
        
        // New model - determine policy
        let policy = if let Some(policy_type) = policy_hint {
            // Use provided policy hint
            self.create_policy_from_type(policy_type)
        } else {
            // Use default policy
            (self.default_policy_factory)()
        };
        
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
                if *count == 0 {
                    counts.remove(model_id);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        
        // Clean up policy if this was the last worker
        if should_cleanup {
            let mut policies = self.model_policies.write().unwrap();
            if let Some(policy) = policies.remove(model_id) {
                // If policy implements Drop for cleanup (like CacheAwarePolicy's eviction thread)
                drop(policy);
                info!("Removed policy for model {} (last worker removed)", model_id);
            }
        }
    }
    
    /// Get policy for a model
    pub fn get_policy(&self, model_id: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.model_policies.read().unwrap().get(model_id).cloned()
    }
    
    /// Create policy from type string
    fn create_policy_from_type(&self, policy_type: &str) -> Arc<dyn LoadBalancingPolicy> {
        match policy_type {
            "round_robin" => Arc::new(RoundRobinPolicy::new()),
            "random" => Arc::new(RandomPolicy::new()),
            "shortest_queue" => Arc::new(ShortestQueuePolicy::new()),
            "cache_aware" => Arc::new(CacheAwarePolicy::new()),
            _ => {
                warn!("Unknown policy type: {}, using default", policy_type);
                (self.default_policy_factory)()
            }
        }
    }
    
    /// Get current model->policy mappings (for debugging/monitoring)
    pub fn get_all_mappings(&self) -> HashMap<String, String> {
        let policies = self.model_policies.read().unwrap();
        policies.iter()
            .map(|(model, policy)| (model.clone(), policy.name().to_string()))
            .collect()
    }
    
    /// Get worker counts per model
    pub fn get_worker_counts(&self) -> HashMap<String, usize> {
        self.model_worker_counts.read().unwrap().clone()
    }
}
```

## Router Integration

```rust
// src/routers/http/router.rs

pub struct HttpRouter {
    policy_registry: Arc<PolicyRegistry>,
    worker_registry: Arc<WorkerRegistry>,
}

impl HttpRouter {
    pub fn new(
        default_policy: PolicyConfig,
        worker_registry: Arc<WorkerRegistry>,
    ) -> Self {
        Self {
            policy_registry: Arc::new(PolicyRegistry::new(default_policy)),
            worker_registry,
        }
    }
    
    pub fn add_worker(&self, worker: Box<dyn Worker>, policy_hint: Option<String>) -> Result<()> {
        let model_id = worker.model_id();
        
        // Register policy for this model (or get existing)
        let policy = self.policy_registry.on_worker_added(
            model_id,
            policy_hint.as_deref(),
        );
        
        // Add worker to registry
        self.worker_registry.add_worker(worker)?;
        
        info!(
            "Added worker for model {} using policy {}",
            model_id,
            policy.name()
        );
        
        Ok(())
    }
    
    pub fn remove_worker(&self, worker_url: &str) -> Result<()> {
        // Get worker to find its model
        if let Some(worker) = self.worker_registry.find_worker(worker_url) {
            let model_id = worker.model_id();
            
            // Remove from registry first
            self.worker_registry.remove_worker(worker_url)?;
            
            // Update policy registry (may clean up if last worker)
            self.policy_registry.on_worker_removed(model_id);
        }
        
        Ok(())
    }
    
    pub fn select_worker(&self, request: &Request) -> Option<String> {
        let model_id = extract_model_from_request(request)?;
        
        // Get workers for this model
        let workers = self.worker_registry.get_workers_for_model(&model_id)?;
        if workers.is_empty() {
            return None;
        }
        
        // Get policy for this model
        let policy = self.policy_registry.get_policy(&model_id)?;
        
        // Use policy to select worker
        let selected_idx = policy.select_worker(&workers, request.text())?;
        Some(workers[selected_idx].url().to_string())
    }
}
```

## API Examples

### Example 1: New Model Without Policy Hint
```http
POST /add_worker
{
  "url": "http://gpu1:8080",
  "model_id": "llama-3"
  // No policy specified
}

Response: Worker added for model llama-3 using policy round_robin (default)
```

### Example 2: New Model With Policy Hint
```http
POST /add_worker
{
  "url": "http://gpu2:8080",
  "model_id": "gpt-4",
  "policy": "cache_aware"  // Policy hint
}

Response: Worker added for model gpt-4 using policy cache_aware
```

### Example 3: Existing Model (Policy Ignored)
```http
POST /add_worker
{
  "url": "http://gpu3:8080",
  "model_id": "gpt-4",
  "policy": "round_robin"  // Ignored! gpt-4 already uses cache_aware
}

Response: Worker added for model gpt-4 using policy cache_aware
```

### Example 4: Remove Last Worker
```http
DELETE /remove_worker
{
  "url": "http://gpu2:8080"  // Last gpt-4 worker
}

Response: Worker removed, policy for model gpt-4 cleaned up
```

## State Management Example

```
Initial State:
  models: {}
  counts: {}

Add llama-3 worker (no policy):
  models: {"llama-3": RoundRobinPolicy}  // default
  counts: {"llama-3": 1}

Add llama-3 worker #2:
  models: {"llama-3": RoundRobinPolicy}  // same
  counts: {"llama-3": 2}

Add gpt-4 worker (cache_aware hint):
  models: {"llama-3": RoundRobinPolicy, "gpt-4": CacheAwarePolicy}
  counts: {"llama-3": 2, "gpt-4": 1}

Remove llama-3 worker:
  models: {"llama-3": RoundRobinPolicy, "gpt-4": CacheAwarePolicy}
  counts: {"llama-3": 1, "gpt-4": 1}  // Decreased

Remove last llama-3 worker:
  models: {"gpt-4": CacheAwarePolicy}  // llama-3 cleaned up!
  counts: {"gpt-4": 1}
```

## Benefits

1. **Centralized Management**: Single place for all policy logic
2. **Automatic Cleanup**: Removes policies when models are gone
3. **Consistent Behavior**: All routers use same registry
4. **Dynamic Discovery**: No pre-configuration needed
5. **Resource Efficient**: Cleans up unused policies
6. **Thread-Safe**: Proper locking for concurrent access

## Implementation Notes

1. **Policy Lifecycle**: Policies are created on first worker, destroyed on last worker removal
2. **Thread Safety**: Use RwLock for concurrent access
3. **Memory Management**: Arc for shared ownership, automatic cleanup
4. **Cache-Aware Special Case**: Already handles per-model trees internally
5. **Metrics**: Track policy creation/destruction for monitoring

## Configuration

```yaml
# Minimal - just default policy
routers:
  http:
    default_policy: round_robin
  grpc:
    default_policy: shortest_queue

# No model pre-configuration needed!
# Models and their policies are discovered dynamically
```

## Summary

The PolicyRegistry provides:
- ✅ Dynamic model→policy mapping
- ✅ First worker sets policy (with optional hint)
- ✅ Subsequent workers use existing policy
- ✅ Automatic cleanup on last worker removal
- ✅ Thread-safe concurrent access
- ✅ Clean separation of concerns