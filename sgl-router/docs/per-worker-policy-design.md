# Per-Worker/Per-Model Policy Design

## Problem Statement

Currently, all workers in a router use the same static policy. However, different workers or models may benefit from different routing strategies:
- GPU workers might prefer cache-aware routing
- CPU workers might prefer round-robin for predictable distribution
- High-memory workers might prefer cache-aware with larger trees
- Different models might have different optimal routing patterns

## Design Options

### Option 1: Policy Per Model (Recommended)
Each model gets its own policy configuration, workers inherit from their model.

```rust
// In RouterConfig
pub struct RouterConfig {
    pub policy: PolicyConfig,  // Default policy
    pub model_policies: HashMap<String, PolicyConfig>,  // Per-model overrides
}

// In Router
pub struct Router {
    default_policy: Arc<dyn LoadBalancingPolicy>,
    model_policies: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    workers: Vec<Box<dyn Worker>>,
}

impl Router {
    pub fn select_worker(&self, request: &Request) -> Option<usize> {
        let model_id = extract_model_from_request(request);
        
        // Get policy for this model, or use default
        let policy = self.model_policies
            .get(&model_id)
            .unwrap_or(&self.default_policy);
        
        // Filter workers for this model
        let model_workers: Vec<&Box<dyn Worker>> = self.workers
            .iter()
            .filter(|w| w.model_id() == model_id)
            .collect();
        
        policy.select_worker(&model_workers, request.text())
    }
}
```

**Pros:**
- Clean separation by model
- Easy to configure (one policy per model)
- Works well with multi-model scenarios
- Maintains backward compatibility

**Cons:**
- All workers of same model must use same policy
- Doesn't support per-worker granularity

### Option 2: Policy Per Worker
Each worker can have its own policy configuration.

```rust
// In Worker trait
pub trait Worker {
    fn preferred_policy(&self) -> Option<&str> {
        self.metadata().labels.get("policy").map(|s| s.as_str())
    }
}

// In Router
pub struct Router {
    policies: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    workers: Vec<Box<dyn Worker>>,
}

impl Router {
    pub fn select_worker(&self, request: &Request) -> Option<usize> {
        // Group workers by their preferred policy
        let mut policy_groups: HashMap<String, Vec<usize>> = HashMap::new();
        
        for (idx, worker) in self.workers.iter().enumerate() {
            if worker.is_healthy() {
                let policy_name = worker.preferred_policy()
                    .unwrap_or("default");
                policy_groups.entry(policy_name.to_string())
                    .or_default()
                    .push(idx);
            }
        }
        
        // Select best worker across all policies
        let mut candidates = Vec::new();
        for (policy_name, worker_indices) in policy_groups {
            if let Some(policy) = self.policies.get(&policy_name) {
                let workers_subset: Vec<&Box<dyn Worker>> = worker_indices
                    .iter()
                    .map(|&idx| &self.workers[idx])
                    .collect();
                
                if let Some(selected) = policy.select_worker(&workers_subset, request.text()) {
                    candidates.push((worker_indices[selected], policy.priority()));
                }
            }
        }
        
        // Return highest priority candidate
        candidates.sort_by_key(|(_, priority)| *priority);
        candidates.first().map(|(idx, _)| *idx)
    }
}
```

**Pros:**
- Maximum flexibility
- Per-worker granularity
- Supports heterogeneous deployments

**Cons:**
- More complex routing logic
- Harder to reason about behavior
- Configuration complexity

### Option 3: Hierarchical Policies (Hybrid)
Combine model-level and worker-level policies with precedence.

```rust
pub struct Router {
    default_policy: Arc<dyn LoadBalancingPolicy>,
    model_policies: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    worker_policies: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    workers: Vec<Box<dyn Worker>>,
}

impl Router {
    pub fn get_worker_policy(&self, worker: &Box<dyn Worker>) -> Arc<dyn LoadBalancingPolicy> {
        // Priority: worker-specific > model-specific > default
        if let Some(worker_policy_name) = worker.metadata().labels.get("policy") {
            if let Some(policy) = self.worker_policies.get(worker_policy_name) {
                return Arc::clone(policy);
            }
        }
        
        let model_id = worker.model_id();
        if let Some(policy) = self.model_policies.get(model_id) {
            return Arc::clone(policy);
        }
        
        Arc::clone(&self.default_policy)
    }
}
```

**Pros:**
- Flexible with sensible defaults
- Supports both model and worker level configuration
- Gradual adoption path

**Cons:**
- Most complex implementation
- Potential for configuration conflicts

## Recommended Implementation Plan

### Phase 1: Model-Level Policies (Option 1)
Start with per-model policies as it covers most use cases:

1. **Configuration**:
```yaml
policy:
  default: round_robin
  models:
    llama-3:
      type: cache_aware
      cache_threshold: 0.7
    gpt-4:
      type: shortest_queue
    mistral:
      type: random
```

2. **Implementation**:
```rust
// Extend RouterFactory
impl RouterFactory {
    pub async fn create_with_model_policies(
        workers: &[Box<dyn Worker>],
        default_policy: &PolicyConfig,
        model_policies: &HashMap<String, PolicyConfig>,
        context: &AppContext,
    ) -> Result<Box<dyn RouterTrait>> {
        // Create default policy
        let default = Self::create_policy(default_policy, context)?;
        
        // Create per-model policies
        let mut policies = HashMap::new();
        for (model_id, config) in model_policies {
            policies.insert(
                model_id.clone(),
                Self::create_policy(config, context)?
            );
        }
        
        Ok(Box::new(ModelAwareRouter {
            default_policy: default,
            model_policies: policies,
            workers: workers.to_vec(),
        }))
    }
}
```

3. **Router Selection Logic**:
```rust
pub struct ModelAwareRouter {
    default_policy: Arc<dyn LoadBalancingPolicy>,
    model_policies: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    workers: Vec<Box<dyn Worker>>,
}

impl ModelAwareRouter {
    fn select_worker_for_model(&self, model_id: &str) -> Option<usize> {
        // Get workers for this model
        let model_workers: Vec<(usize, &Box<dyn Worker>)> = self.workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.model_id() == model_id && w.is_healthy())
            .collect();
        
        if model_workers.is_empty() {
            return None;
        }
        
        // Get policy for this model
        let policy = self.model_policies
            .get(model_id)
            .unwrap_or(&self.default_policy);
        
        // Create temporary worker list for policy
        let workers_for_policy: Vec<Box<dyn Worker>> = model_workers
            .iter()
            .map(|(_, w)| (*w).clone_box())
            .collect();
        
        // Select worker using model's policy
        policy.select_worker(&workers_for_policy, Some("request"))
            .map(|selected| model_workers[selected].0)
    }
}
```

### Phase 2: Worker-Level Override (Optional)
Add worker-level policy override if needed:

1. **Worker Configuration**:
```yaml
workers:
  - url: http://gpu1:8080
    model: llama-3
    labels:
      policy_override: cache_aware_aggressive  # Override model policy
  - url: http://cpu1:8080
    model: llama-3
    labels:
      policy_override: round_robin  # Different policy for CPU worker
```

2. **Selection with Override**:
```rust
impl ModelAwareRouter {
    fn get_effective_policy(&self, worker: &Box<dyn Worker>) -> Arc<dyn LoadBalancingPolicy> {
        // Check for worker-level override
        if let Some(override_name) = worker.metadata().labels.get("policy_override") {
            if let Some(policy) = self.worker_override_policies.get(override_name) {
                return Arc::clone(policy);
            }
        }
        
        // Fall back to model policy
        let model_id = worker.model_id();
        self.model_policies
            .get(model_id)
            .unwrap_or(&self.default_policy)
            .clone()
    }
}
```

## Migration Path

1. **Backward Compatible**: Start with current single-policy approach
2. **Add Model Policies**: Gradually add per-model policies
3. **Worker Overrides**: Add worker-level overrides only if needed
4. **Policy Composition**: Eventually support policy chaining/composition

## Configuration Examples

### Simple Configuration (Current)
```yaml
policy: round_robin
```

### Model-Aware Configuration
```yaml
policy:
  default: round_robin
  models:
    llama-3:
      type: cache_aware
      cache_threshold: 0.7
      balance_abs_threshold: 5
    gpt-4:
      type: shortest_queue
```

### Full Hierarchical Configuration
```yaml
policy:
  default: round_robin
  models:
    llama-3:
      type: cache_aware
      cache_threshold: 0.7
  worker_overrides:
    cpu_policy:
      type: round_robin
    gpu_aggressive:
      type: cache_aware
      cache_threshold: 0.9
      max_tree_size: 100000
```

## Benefits

1. **Flexibility**: Different models can use optimal routing strategies
2. **Performance**: CPU/GPU workers can use appropriate policies
3. **Backward Compatible**: Existing configs continue to work
4. **Gradual Adoption**: Can start simple and add complexity as needed
5. **Multi-Model Support**: Natural fit for multi-model deployments

## Considerations

1. **Cache-Aware Policy**: Already supports per-model trees internally
2. **Metrics**: Need to track metrics per policy per model
3. **Debugging**: More complex routing requires better observability
4. **Configuration Validation**: Need to validate policy configurations at startup
5. **Hot Reload**: Consider supporting dynamic policy updates

## Next Steps

1. Implement Phase 1 (Model-Level Policies)
2. Add configuration parsing for model policies
3. Update RouterFactory to create model-aware routers
4. Add tests for multi-policy scenarios
5. Update documentation with examples
6. Consider Phase 2 based on user feedback