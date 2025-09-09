# Per-Model Policy Management Design

## Key Questions & Decisions

### 1. Where to Maintain Model Policies?

#### Option A: Router Level (Recommended)
**Each router maintains its own model → policy mapping**

```rust
pub struct HttpRouter {
    // Each router has its own policies per model
    model_policies: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    default_policy: Arc<dyn LoadBalancingPolicy>,
    worker_registry: Arc<WorkerRegistry>,
}
```

**Pros:**
- Different routers can use different policies for same model
- Example: HTTP router uses cache-aware, gRPC router uses round-robin
- Policies are routing-specific, naturally belongs to router
- Clean separation of concerns

**Cons:**
- Need to configure policies for each router type
- Potential inconsistency between routers

#### Option B: WorkerRegistry Level
**Centralized policy management in WorkerRegistry**

```rust
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, Vec<Box<dyn Worker>>>>>,
    model_policies: HashMap<String, PolicyConfig>,  // Config, not instances
}
```

**Pros:**
- Single source of truth for model policies
- Consistent across all routers

**Cons:**
- WorkerRegistry shouldn't know about routing policies
- Violates separation of concerns
- Policies are routing concepts, not worker management

### 2. Policy Initialization & Worker Addition Rules

#### Scenario 1: First Worker Defines Policy
```yaml
# Initial state: empty
# Add first llama-3 worker
POST /add_worker
{
  "url": "http://gpu1:8080",
  "model_id": "llama-3",
  "policy": "cache_aware"  # Optional, uses default if not specified
}

# Router creates cache_aware policy for llama-3
# Add second llama-3 worker
POST /add_worker
{
  "url": "http://gpu2:8080",
  "model_id": "llama-3",
  "policy": "round_robin"  # REJECTED or IGNORED?
}
```

#### Scenario 2: Pre-configured Model Policies
```yaml
# Config file
models:
  llama-3:
    policy: cache_aware
    config:
      cache_threshold: 0.7
  gpt-4:
    policy: shortest_queue
  
# All workers of a model MUST use the configured policy
# No per-worker policy override allowed
```

## Recommended Design

### Architecture

```rust
// 1. Configuration
pub struct RouterConfig {
    pub default_policy: PolicyConfig,
    pub model_policies: HashMap<String, PolicyConfig>,  // From config file
}

// 2. Router maintains policy instances
pub struct HttpRouter {
    default_policy: Arc<dyn LoadBalancingPolicy>,
    model_policies: HashMap<String, Arc<dyn LoadBalancingPolicy>>,
    worker_registry: Arc<WorkerRegistry>,
}

impl HttpRouter {
    pub fn new(config: RouterConfig, registry: Arc<WorkerRegistry>) -> Self {
        // Create policy instances from config
        let mut model_policies = HashMap::new();
        for (model_id, policy_config) in config.model_policies {
            let policy = create_policy(policy_config);
            model_policies.insert(model_id, Arc::new(policy));
        }
        
        Self {
            default_policy: Arc::new(create_policy(config.default_policy)),
            model_policies,
            worker_registry: registry,
        }
    }
    
    pub fn select_worker(&self, request: &Request) -> Option<WorkerUrl> {
        let model_id = extract_model_from_request(request)?;
        
        // Get all workers for this model from registry
        let workers = self.worker_registry.get_workers_for_model(&model_id)?;
        
        // Get policy for this model (or default)
        let policy = self.model_policies
            .get(&model_id)
            .unwrap_or(&self.default_policy);
        
        // Use policy to select from these workers
        let selected_idx = policy.select_worker(&workers, request.text())?;
        Some(workers[selected_idx].url().to_string())
    }
}

// 3. WorkerRegistry only manages workers, not policies
impl WorkerRegistry {
    pub fn add_worker(&self, worker: Box<dyn Worker>) -> Result<()> {
        let model_id = worker.model_id();
        
        // Registry just stores workers by model
        // Doesn't care about policies
        self.workers.write()
            .entry(model_id.to_string())
            .or_default()
            .push(worker);
        
        Ok(())
    }
    
    pub fn get_workers_for_model(&self, model_id: &str) -> Option<Vec<Box<dyn Worker>>> {
        self.workers.read()
            .get(model_id)
            .cloned()
    }
}
```

### Rules for Worker Addition

1. **Model Policy is Immutable Once Set**
   - First worker of a model uses configured policy (or default)
   - All subsequent workers of same model use same policy
   - Cannot change policy without removing all workers first

2. **Validation on Worker Addition**
   ```rust
   impl RouterManager {
       pub fn add_worker(&self, worker: WorkerInfo) -> Result<()> {
           // Validate model consistency
           if let Some(existing_model_workers) = self.registry.get_workers_for_model(&worker.model_id) {
               // Model already exists, ensure consistency
               // But we don't validate policy here - that's router's concern
           }
           
           // Add to registry
           self.registry.add_worker(Box::new(worker))?;
           
           // Each router will use its own configured policy for this model
           Ok(())
       }
   }
   ```

3. **Policy Configuration Priority**
   ```
   1. Explicit model config (from config file)
   2. Default policy for router type
   3. System default (round_robin)
   ```

### Example Workflow

```yaml
# 1. Config file
routers:
  http:
    default_policy: round_robin
    model_policies:
      llama-3:
        type: cache_aware
        cache_threshold: 0.7
      gpt-4:
        type: shortest_queue
  grpc:
    default_policy: shortest_queue
    # No model-specific policies, all use shortest_queue

# 2. Add workers
POST /add_worker {"url": "http://gpu1:8080", "model_id": "llama-3"}
POST /add_worker {"url": "http://gpu2:8080", "model_id": "llama-3"}
# Both workers added to registry under "llama-3"
# HTTP router will use cache_aware for both
# gRPC router will use shortest_queue for both

POST /add_worker {"url": "http://cpu1:8080", "model_id": "gpt-4"}
# HTTP router will use shortest_queue
# gRPC router will use shortest_queue (default)

POST /add_worker {"url": "http://gpu3:8080", "model_id": "mistral"}
# No explicit config for mistral
# HTTP router will use round_robin (default)
# gRPC router will use shortest_queue (default)
```

### Benefits of This Design

1. **Clear Separation**: WorkerRegistry manages workers, Routers manage policies
2. **Flexibility**: Different router types can have different policies for same model
3. **Consistency**: All workers of a model within a router use same policy
4. **Simple Mental Model**: Model → Policy is 1:1 within each router
5. **Easy Configuration**: Pre-configure policies in config file
6. **Backward Compatible**: Works with existing single-policy setup

### Implementation Considerations

1. **Dynamic Policy Updates**: Not supported initially (requires worker drain/re-add)
2. **Policy Validation**: Validate at router startup, not at worker addition
3. **Metrics**: Track per-model-per-policy metrics for analysis
4. **Cache-Aware Special Case**: Already handles per-model trees internally

### Migration Path

```rust
// Phase 1: Current state - single policy per router
router.policy = cache_aware

// Phase 2: Add model policies with backward compat
router.default_policy = cache_aware
router.model_policies = {}  // Empty, all use default

// Phase 3: Configure specific models
router.default_policy = round_robin
router.model_policies = {
    "llama-3": cache_aware,
    "gpt-4": shortest_queue
}
```

## Summary

- **Maintain policies at Router level** (each router has its own model→policy mapping)
- **WorkerRegistry is policy-agnostic** (just stores workers by model)
- **All workers of same model use same policy** within a router
- **Configure policies upfront** in config file, not per-worker
- **Different routers can use different policies** for the same model

This provides a clean, maintainable design that supports multi-model scenarios while keeping concerns properly separated.