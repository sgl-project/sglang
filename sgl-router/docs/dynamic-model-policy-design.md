# Dynamic Model Policy Design

## Problem with Pre-Configuration

Pre-configuring all models defeats the purpose of multi-model routing:
- Cannot know all models in advance
- New models should work without router restart
- Dynamic model discovery is essential

## Revised Design: Dynamic Policy Assignment

### Core Concept: Policy Hints

Workers can provide "policy hints" when they register, but the router makes the final decision based on the first worker of each model.

### Option 1: First Worker Sets Policy (Simple)

```rust
pub struct HttpRouter {
    default_policy: Arc<dyn LoadBalancingPolicy>,
    model_policies: RwLock<HashMap<String, Arc<dyn LoadBalancingPolicy>>>,
    worker_registry: Arc<WorkerRegistry>,
}

impl HttpRouter {
    pub fn on_worker_added(&self, worker: &Box<dyn Worker>) {
        let model_id = worker.model_id();
        
        // Check if we already have a policy for this model
        let policies = self.model_policies.read();
        if policies.contains_key(model_id) {
            // Model already has a policy, new worker uses it
            return;
        }
        drop(policies);
        
        // First worker of this model - determine policy
        let policy = self.determine_policy_for_worker(worker);
        
        // Store policy for this model
        let mut policies = self.model_policies.write();
        policies.insert(model_id.to_string(), Arc::new(policy));
    }
    
    fn determine_policy_for_worker(&self, worker: &Box<dyn Worker>) -> Box<dyn LoadBalancingPolicy> {
        // Check for policy hint in worker metadata
        if let Some(policy_hint) = worker.metadata().labels.get("preferred_policy") {
            match policy_hint.as_str() {
                "cache_aware" => Box::new(CacheAwarePolicy::new()),
                "round_robin" => Box::new(RoundRobinPolicy::new()),
                "shortest_queue" => Box::new(ShortestQueuePolicy::new()),
                _ => self.create_default_policy()
            }
        } else {
            self.create_default_policy()
        }
    }
}
```

**Example Flow:**
```yaml
# First llama-3 worker
POST /add_worker
{
  "url": "http://gpu1:8080",
  "model_id": "llama-3",
  "labels": {
    "preferred_policy": "cache_aware"  # Hint, not command
  }
}
# Router: "First llama-3 worker suggests cache_aware, using it"

# Second llama-3 worker  
POST /add_worker
{
  "url": "http://gpu2:8080",
  "model_id": "llama-3",
  "labels": {
    "preferred_policy": "round_robin"  # Ignored!
  }
}
# Router: "llama-3 already uses cache_aware, ignoring hint"
```

### Option 2: Policy Templates (Flexible)

Router has policy templates that match patterns:

```rust
pub struct PolicyTemplate {
    pub pattern: PolicyPattern,
    pub policy_config: PolicyConfig,
}

pub enum PolicyPattern {
    ModelPrefix(String),      // "llama-*"
    ModelSuffix(String),      // "*-instruct"
    WorkerLabel(String, String), // label_key, label_value
    Default,
}

pub struct HttpRouter {
    policy_templates: Vec<PolicyTemplate>,
    model_policies: RwLock<HashMap<String, Arc<dyn LoadBalancingPolicy>>>,
}

impl HttpRouter {
    fn determine_policy_for_model(&self, model_id: &str, first_worker: &Box<dyn Worker>) -> Box<dyn LoadBalancingPolicy> {
        // Check templates in order
        for template in &self.policy_templates {
            if template.matches(model_id, first_worker) {
                return create_policy(template.policy_config.clone());
            }
        }
        
        // Default fallback
        Box::new(RoundRobinPolicy::new())
    }
}
```

**Configuration:**
```yaml
policy_templates:
  - pattern: 
      model_prefix: "llama-"
    policy:
      type: cache_aware
      cache_threshold: 0.7
  - pattern:
      model_suffix: "-chat"
    policy:
      type: shortest_queue
  - pattern:
      worker_label: 
        key: "hardware"
        value: "gpu"
    policy:
      type: cache_aware
  - pattern: default
    policy:
      type: round_robin
```

### Option 3: Smart Auto-Detection (Intelligent)

Router analyzes worker characteristics to choose optimal policy:

```rust
impl HttpRouter {
    fn auto_detect_policy(&self, model_id: &str, first_worker: &Box<dyn Worker>) -> Box<dyn LoadBalancingPolicy> {
        let labels = &first_worker.metadata().labels;
        
        // Check various heuristics
        let use_cache_aware = 
            // GPU workers benefit from cache locality
            labels.get("hardware") == Some(&"gpu".to_string()) ||
            // Large models benefit from cache reuse
            labels.get("model_size").and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) > 10_000_000_000 ||
            // Explicitly requested
            labels.get("preferred_policy") == Some(&"cache_aware".to_string());
        
        let use_shortest_queue =
            // CPU workers need even distribution
            labels.get("hardware") == Some(&"cpu".to_string()) ||
            // High throughput models
            labels.get("max_batch_size").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0) > 64;
        
        if use_cache_aware {
            Box::new(CacheAwarePolicy::new())
        } else if use_shortest_queue {
            Box::new(ShortestQueuePolicy::new())
        } else {
            Box::new(RoundRobinPolicy::new())
        }
    }
}
```

## Recommended Solution: Hybrid Approach

Combine the best aspects:

```rust
pub struct HttpRouter {
    // Optional pre-configured policies for known models
    configured_policies: HashMap<String, PolicyConfig>,
    
    // Policy templates for pattern matching
    policy_templates: Vec<PolicyTemplate>,
    
    // Dynamic policies determined at runtime
    runtime_policies: RwLock<HashMap<String, Arc<dyn LoadBalancingPolicy>>>,
    
    // Default policy fallback
    default_policy: Arc<dyn LoadBalancingPolicy>,
}

impl HttpRouter {
    pub fn on_worker_added(&self, worker: &Box<dyn Worker>) {
        let model_id = worker.model_id();
        
        // Check if already have a policy for this model
        if self.runtime_policies.read().contains_key(model_id) {
            return; // Use existing policy
        }
        
        // Determine policy for this new model
        let policy = self.determine_policy(model_id, worker);
        
        // Store for future workers of same model
        self.runtime_policies.write().insert(
            model_id.to_string(),
            Arc::new(policy)
        );
    }
    
    fn determine_policy(&self, model_id: &str, first_worker: &Box<dyn Worker>) -> Box<dyn LoadBalancingPolicy> {
        // 1. Check pre-configured (if any)
        if let Some(config) = self.configured_policies.get(model_id) {
            return create_policy(config.clone());
        }
        
        // 2. Check templates
        for template in &self.policy_templates {
            if template.matches(model_id, first_worker) {
                return create_policy(template.policy_config.clone());
            }
        }
        
        // 3. Check worker hint
        if let Some(hint) = first_worker.metadata().labels.get("preferred_policy") {
            if let Some(policy) = self.create_policy_from_hint(hint) {
                return policy;
            }
        }
        
        // 4. Auto-detect based on characteristics
        if self.should_use_cache_aware(first_worker) {
            return Box::new(CacheAwarePolicy::new());
        }
        
        // 5. Default fallback
        self.create_default_policy()
    }
}
```

## Configuration Example

```yaml
# Minimal config - everything dynamic
routers:
  http:
    default_policy: round_robin

---

# With templates for common patterns
routers:
  http:
    default_policy: round_robin
    policy_templates:
      - pattern: "llama-*"
        policy: cache_aware
      - pattern: "*-instruct"
        policy: shortest_queue

---

# With some known models pre-configured
routers:
  http:
    default_policy: round_robin
    known_models:
      llama-3: cache_aware  # Pre-configured
      gpt-4: shortest_queue # Pre-configured
    # Everything else is dynamic
```

## Worker Registration Examples

```python
# Worker provides hint
add_worker({
    "url": "http://gpu1:8080",
    "model_id": "new-model-xyz",  # Unknown model
    "labels": {
        "preferred_policy": "cache_aware",
        "hardware": "gpu",
        "model_size": "70B"
    }
})
# Router: "First worker of new-model-xyz, using cache_aware based on hint"

# Second worker of same model
add_worker({
    "url": "http://gpu2:8080", 
    "model_id": "new-model-xyz",
    "labels": {
        "preferred_policy": "round_robin"  # Ignored
    }
})
# Router: "new-model-xyz already using cache_aware"

# Worker without hint
add_worker({
    "url": "http://cpu1:8080",
    "model_id": "another-model",
    "labels": {
        "hardware": "cpu"
    }
})
# Router: "First worker of another-model, auto-detected shortest_queue for CPU"
```

## Benefits

1. **Fully Dynamic**: No need to know models in advance
2. **Flexible**: Supports hints, patterns, and auto-detection
3. **Consistent**: All workers of same model use same policy
4. **Discoverable**: New models work immediately
5. **Configurable**: Can still pre-configure known models

## Implementation Priority

1. **Phase 1**: First worker sets policy (Option 1) - Simplest
2. **Phase 2**: Add policy hints in worker metadata
3. **Phase 3**: Add pattern templates for common cases
4. **Phase 4**: Add smart auto-detection

## Key Principles

- **First worker decides** the policy for its model
- **Subsequent workers follow** the established policy
- **Policy is immutable** once set (within router lifetime)
- **Hints are suggestions**, not commands
- **Router has final say** on policy selection