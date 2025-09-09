# Router-WorkerRegistry Integration Fix

## Problem

1. RouterManager adds workers to WorkerRegistry
2. Individual routers (HTTP, gRPC) have their own `workers` field initialized as empty
3. Routers never get informed about workers added to WorkerRegistry
4. When routing requests, routers use their empty `workers` list

## Current Flow (Broken)
```
RouterManager.add_worker()
  → Creates worker
  → Adds to WorkerRegistry ✓
  → Individual routers still have empty workers ✗
  
Router.route_request()
  → Uses self.workers (empty!) ✗
  → No workers available
```

## Solution Options

### Option 1: Routers Use WorkerRegistry Directly (Recommended)
Remove the `workers` field from individual routers and have them query WorkerRegistry.

```rust
pub struct HttpRouter {
    // Remove: workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    worker_registry: Arc<WorkerRegistry>,  // Add this
    policy: Arc<dyn LoadBalancingPolicy>,
}

impl HttpRouter {
    fn select_worker_with_circuit_breaker(&self, text: Option<&str>) -> Option<Box<dyn Worker>> {
        // Get workers from registry based on connection type
        let workers = self.worker_registry.get_workers_by_type(
            ConnectionMode::Http,
            WorkerType::Regular
        );
        
        let available: Vec<Box<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .map(|w| w.clone_worker())
            .collect();
            
        if available.is_empty() {
            return None;
        }
        
        let idx = self.policy.select_worker(&available, text)?;
        Some(available[idx].clone_worker())
    }
}
```

### Option 2: Sync Workers to Routers
RouterManager notifies routers when workers are added/removed.

```rust
impl RouterManager {
    pub fn add_worker(&self, config: WorkerConfig) -> Result<WorkerApiResponse> {
        // Create and register worker
        let worker = create_worker(config);
        self.worker_registry.register(worker.clone());
        
        // Notify appropriate routers
        match worker.connection_mode() {
            ConnectionMode::Http => {
                if let Some(router) = &self.http_router {
                    router.add_worker(worker.clone());
                }
                if worker.worker_type() == WorkerType::Regular {
                    if let Some(pd_router) = &self.http_pd_router {
                        pd_router.add_prefill_worker(worker.clone());
                    }
                }
            }
            ConnectionMode::Grpc => {
                // Similar for gRPC routers
            }
        }
    }
}
```

### Option 3: WorkerRegistry with Filters
Enhance WorkerRegistry to support filtered queries.

```rust
impl WorkerRegistry {
    /// Get workers matching specific criteria
    pub fn get_workers_filtered(
        &self,
        connection_mode: Option<ConnectionMode>,
        worker_type: Option<WorkerType>,
        model_id: Option<&str>,
    ) -> Vec<Arc<dyn Worker>> {
        self.workers.read()
            .values()
            .filter(|w| {
                connection_mode.map_or(true, |cm| w.connection_mode() == cm) &&
                worker_type.map_or(true, |wt| w.worker_type() == wt) &&
                model_id.map_or(true, |m| w.model_id() == m)
            })
            .cloned()
            .collect()
    }
}
```

## Recommended Implementation (Option 1 + 3)

### Step 1: Enhance WorkerRegistry

```rust
// src/routers/worker_registry.rs
impl WorkerRegistry {
    /// Get all workers for a specific connection mode and type
    pub fn get_workers_for_router(
        &self,
        connection_mode: ConnectionMode,
        worker_type: WorkerType,
    ) -> Vec<Arc<dyn Worker>> {
        self.workers
            .read()
            .unwrap()
            .values()
            .filter(|w| {
                w.connection_mode() == connection_mode &&
                match worker_type {
                    WorkerType::Regular => w.worker_type() == WorkerType::Regular,
                    WorkerType::Prefill { .. } => matches!(w.worker_type(), WorkerType::Prefill { .. }),
                    WorkerType::Decode => w.worker_type() == WorkerType::Decode,
                }
            })
            .cloned()
            .collect()
    }
    
    /// Get workers for a specific model
    pub fn get_workers_for_model(
        &self,
        model_id: &str,
        connection_mode: Option<ConnectionMode>,
    ) -> Vec<Arc<dyn Worker>> {
        self.workers
            .read()
            .unwrap()
            .values()
            .filter(|w| {
                w.model_id() == model_id &&
                connection_mode.map_or(true, |cm| w.connection_mode() == cm)
            })
            .cloned()
            .collect()
    }
}
```

### Step 2: Update Routers

```rust
// src/routers/http/router.rs
pub struct HttpRouter {
    worker_registry: Arc<WorkerRegistry>,
    policy: Arc<dyn LoadBalancingPolicy>,
    connection_mode: ConnectionMode,
    worker_type: WorkerType,
    // Remove: workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
}

impl HttpRouter {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy: Arc<dyn LoadBalancingPolicy>,
    ) -> Self {
        Self {
            worker_registry,
            policy,
            connection_mode: ConnectionMode::Http,
            worker_type: WorkerType::Regular,
        }
    }
    
    fn get_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.worker_registry.get_workers_for_router(
            self.connection_mode,
            self.worker_type,
        )
    }
    
    fn select_worker_with_circuit_breaker(&self, text: Option<&str>) -> Option<Arc<dyn Worker>> {
        let workers = self.get_workers();
        
        // Convert Arc<dyn Worker> to Box<dyn Worker> for policy
        let boxed_workers: Vec<Box<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .map(|w| w.clone_worker())
            .collect();
            
        if boxed_workers.is_empty() {
            return None;
        }
        
        let idx = self.policy.select_worker(&boxed_workers, text)?;
        Some(workers[idx].clone())
    }
}
```

### Step 3: Update RouterFactory

```rust
// src/routers/factory.rs
impl RouterFactory {
    pub async fn create_regular_router(
        worker_registry: Arc<WorkerRegistry>,
        policy_config: &PolicyConfig,
        context: &AppContext,
    ) -> Result<Box<dyn RouterTrait>> {
        let policy = Self::create_policy(policy_config, context)?;
        
        Ok(Box::new(HttpRouter::new(
            worker_registry,
            Arc::new(policy),
        )))
    }
    
    pub async fn create_pd_router(
        worker_registry: Arc<WorkerRegistry>,
        prefill_policy_config: &PolicyConfig,
        decode_policy_config: &PolicyConfig,
        context: &AppContext,
    ) -> Result<Box<dyn RouterTrait>> {
        let prefill_policy = Self::create_policy(prefill_policy_config, context)?;
        let decode_policy = Self::create_policy(decode_policy_config, context)?;
        
        Ok(Box::new(PdRouter::new(
            worker_registry,
            Arc::new(prefill_policy),
            Arc::new(decode_policy),
        )))
    }
}
```

### Step 4: Update RouterManager

```rust
// src/routers/router_manager.rs
impl RouterManager {
    pub fn new(config: Config) -> Self {
        let worker_registry = Arc::new(WorkerRegistry::new());
        
        // Create routers with shared worker registry
        let http_router = if config.enable_igw {
            Some(RouterFactory::create_regular_router(
                Arc::clone(&worker_registry),
                &config.policy,
                &app_context,
            ).await?)
        } else {
            None
        };
        
        Self {
            worker_registry,
            http_router,
            // ... other routers
        }
    }
}
```

## Benefits

1. **Single Source of Truth**: WorkerRegistry is the only place workers are stored
2. **No Synchronization Issues**: No need to keep router workers in sync
3. **Dynamic Updates**: Workers added to registry are immediately available to routers
4. **Clean Separation**: Routers focus on routing, registry focuses on worker management
5. **Model Support**: Easy to filter workers by model for per-model routing

## Migration Path

1. Add `get_workers_for_router` method to WorkerRegistry
2. Update routers to use WorkerRegistry instead of local workers
3. Remove workers field from routers
4. Update RouterFactory to pass WorkerRegistry to routers
5. Test with dynamic worker addition

## Example Flow

```
RouterManager.add_worker()
  → Creates worker
  → Adds to WorkerRegistry ✓
  
Router.route_request()
  → Queries WorkerRegistry.get_workers_for_router() ✓
  → Gets current workers ✓
  → Routes successfully ✓
```

## Required Changes in Routers

### 1. HttpRouter (src/routers/http/router.rs)

#### Current Implementation (Remove)
```rust
pub struct HttpRouter {
    workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    policy: Arc<dyn LoadBalancingPolicy>,
    // ...
}

impl HttpRouter {
    // REMOVE these methods
    pub fn add_worker(&self, worker: Box<dyn Worker>) {
        self.workers.write().unwrap().push(worker);
    }
    
    pub fn remove_worker(&self, url: &str) {
        self.workers.write().unwrap().retain(|w| w.url() != url);
    }
    
    fn get_worker_urls(&self) -> Vec<String> {
        self.workers.read().unwrap()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }
}
```

#### New Implementation
```rust
pub struct HttpRouter {
    worker_registry: Arc<WorkerRegistry>,
    policy: Arc<dyn LoadBalancingPolicy>,
    connection_mode: ConnectionMode,
    worker_type: WorkerType,
    // Remove: workers field
}

impl HttpRouter {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy: Arc<dyn LoadBalancingPolicy>,
    ) -> Self {
        Self {
            worker_registry,
            policy,
            connection_mode: ConnectionMode::Http,
            worker_type: WorkerType::Regular,
        }
    }
    
    // No more add_worker/remove_worker - registry handles it
    
    fn get_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.worker_registry.get_workers_for_router(
            self.connection_mode,
            self.worker_type,
        )
    }
    
    fn get_worker_urls(&self) -> Vec<String> {
        self.get_workers()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }
    
    fn select_worker_with_circuit_breaker(&self, text: Option<&str>) -> Option<Arc<dyn Worker>> {
        let workers = self.get_workers();
        
        // Convert Arc to Box for policy compatibility
        let boxed_workers: Vec<Box<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .map(|w| w.clone_worker())
            .collect();
            
        if boxed_workers.is_empty() {
            return None;
        }
        
        let idx = self.policy.select_worker(&boxed_workers, text)?;
        Some(workers[idx].clone())
    }
    
    fn readiness(&self) -> Response {
        let workers = self.get_workers();
        let healthy_count = workers.iter().filter(|w| w.is_healthy()).count();
        
        if healthy_count > 0 {
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_count,
                "total_workers": workers.len()
            })).into_response()
        } else {
            // ...
        }
    }
}
```

### 2. PdRouter (src/routers/http/pd_router.rs)

#### Current Implementation (Remove)
```rust
pub struct PdRouter {
    prefill_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    decode_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    prefill_policy: Arc<dyn LoadBalancingPolicy>,
    decode_policy: Arc<dyn LoadBalancingPolicy>,
    // ...
}

impl PdRouter {
    // REMOVE these methods
    pub fn add_prefill_worker(&self, worker: Box<dyn Worker>) {
        // Update cache policy if needed
        if let Some(cache_policy) = self.prefill_policy.as_any()
            .downcast_ref::<CacheAwarePolicy>() {
            cache_policy.add_worker(worker.as_ref());
        }
        self.prefill_workers.write().unwrap().push(worker);
    }
    
    pub fn add_decode_worker(&self, worker: Box<dyn Worker>) {
        self.decode_workers.write().unwrap().push(worker);
    }
    
    pub fn remove_prefill_worker(&self, url: &str) {
        if let Some(cache_policy) = self.prefill_policy.as_any()
            .downcast_ref::<CacheAwarePolicy>() {
            cache_policy.remove_worker_by_url(url);
        }
        self.prefill_workers.write().unwrap().retain(|w| w.url() != url);
    }
}
```

#### New Implementation
```rust
pub struct PdRouter {
    worker_registry: Arc<WorkerRegistry>,
    prefill_policy: Arc<dyn LoadBalancingPolicy>,
    decode_policy: Arc<dyn LoadBalancingPolicy>,
    // Remove: prefill_workers and decode_workers fields
}

impl PdRouter {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        prefill_policy: Arc<dyn LoadBalancingPolicy>,
        decode_policy: Arc<dyn LoadBalancingPolicy>,
    ) -> Self {
        // Initialize cache policy with current workers if needed
        if let Some(cache_policy) = prefill_policy.as_any()
            .downcast_ref::<CacheAwarePolicy>() {
            let prefill_workers = worker_registry.get_workers_for_router(
                ConnectionMode::Http,
                WorkerType::Prefill { bootstrap_port: None },
            );
            // Convert Arc to Box for init
            let boxed: Vec<Box<dyn Worker>> = prefill_workers
                .iter()
                .map(|w| w.clone_worker())
                .collect();
            cache_policy.init_workers(&boxed);
        }
        
        Self {
            worker_registry,
            prefill_policy,
            decode_policy,
        }
    }
    
    fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.worker_registry.get_workers_for_router(
            ConnectionMode::Http,
            WorkerType::Prefill { bootstrap_port: None },
        )
    }
    
    fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.worker_registry.get_workers_for_router(
            ConnectionMode::Http,
            WorkerType::Decode,
        )
    }
    
    fn select_workers(&self, text: Option<&str>) -> Option<(Arc<dyn Worker>, Arc<dyn Worker>)> {
        let prefill_workers = self.get_prefill_workers();
        let decode_workers = self.get_decode_workers();
        
        if prefill_workers.is_empty() || decode_workers.is_empty() {
            return None;
        }
        
        // Convert to Box for policy
        let prefill_boxed: Vec<Box<dyn Worker>> = prefill_workers
            .iter()
            .map(|w| w.clone_worker())
            .collect();
        let decode_boxed: Vec<Box<dyn Worker>> = decode_workers
            .iter()
            .map(|w| w.clone_worker())
            .collect();
        
        // Select using policies
        let prefill_idx = self.prefill_policy.select_worker(&prefill_boxed, text)?;
        let decode_idx = self.decode_policy.select_worker(&decode_boxed, None)?;
        
        Some((
            prefill_workers[prefill_idx].clone(),
            decode_workers[decode_idx].clone()
        ))
    }
    
    fn readiness(&self) -> Response {
        let prefill_workers = self.get_prefill_workers();
        let decode_workers = self.get_decode_workers();
        
        let prefill_healthy = prefill_workers.iter().filter(|w| w.is_healthy()).count();
        let decode_healthy = decode_workers.iter().filter(|w| w.is_healthy()).count();
        
        if prefill_healthy > 0 && decode_healthy > 0 {
            Json(json!({
                "status": "ready",
                "prefill_healthy": prefill_healthy,
                "decode_healthy": decode_healthy,
                "prefill_total": prefill_workers.len(),
                "decode_total": decode_workers.len()
            })).into_response()
        } else {
            // ...
        }
    }
}
```

### 3. CacheAwarePolicy Integration

Since CacheAwarePolicy maintains its own tree structure, we need to sync it with WorkerRegistry:

```rust
impl WorkerRegistry {
    /// Notify policies when workers are added/removed
    pub fn register_with_notification(
        &self,
        worker: Arc<dyn Worker>,
        policies: &[Arc<dyn LoadBalancingPolicy>],
    ) -> WorkerId {
        let worker_id = self.register(worker.clone());
        
        // Notify cache-aware policies
        for policy in policies {
            if let Some(cache_policy) = policy.as_any()
                .downcast_ref::<CacheAwarePolicy>() {
                cache_policy.add_worker(worker.as_ref());
            }
        }
        
        worker_id
    }
    
    pub fn remove_with_notification(
        &self,
        worker_id: &WorkerId,
        policies: &[Arc<dyn LoadBalancingPolicy>],
    ) -> Option<Arc<dyn Worker>> {
        let worker = self.remove(worker_id)?;
        
        // Notify cache-aware policies
        for policy in policies {
            if let Some(cache_policy) = policy.as_any()
                .downcast_ref::<CacheAwarePolicy>() {
                cache_policy.remove_worker(worker.as_ref());
            }
        }
        
        Some(worker)
    }
}
```

### 4. RouterManager Updates

```rust
impl RouterManager {
    pub fn add_worker(&self, config: WorkerConfig) -> Result<WorkerApiResponse> {
        let worker = create_worker(config);
        
        // Collect policies that need notification
        let mut policies_to_notify = Vec::new();
        
        if let Some(http_router) = &self.http_router {
            policies_to_notify.push(http_router.policy.clone());
        }
        if let Some(pd_router) = &self.http_pd_router {
            policies_to_notify.push(pd_router.prefill_policy.clone());
            policies_to_notify.push(pd_router.decode_policy.clone());
        }
        
        // Register with notification
        let worker_id = self.worker_registry.register_with_notification(
            Arc::from(worker),
            &policies_to_notify,
        );
        
        Ok(WorkerApiResponse {
            success: true,
            message: format!("Worker {} added", worker_id.as_str()),
            worker: Some(worker_info),
        })
    }
    
    pub fn remove_worker(&self, url: &str) -> Result<WorkerApiResponse> {
        // Find worker
        let worker = self.worker_registry.get_by_url(url)?;
        let worker_id = worker.id();
        
        // Collect policies for notification
        let mut policies_to_notify = Vec::new();
        // ... same as add_worker
        
        // Remove with notification
        self.worker_registry.remove_with_notification(
            &worker_id,
            &policies_to_notify,
        );
        
        Ok(WorkerApiResponse {
            success: true,
            message: format!("Worker removed"),
            worker: None,
        })
    }
}
```

## Summary of Changes

1. **Remove** `workers` field from all routers
2. **Add** `worker_registry: Arc<WorkerRegistry>` to all routers
3. **Remove** `add_worker`/`remove_worker` methods from routers
4. **Add** `get_workers()` method that queries WorkerRegistry
5. **Update** `select_worker` to use registry instead of local list
6. **Update** RouterFactory to pass WorkerRegistry to routers
7. **Add** notification mechanism for CacheAwarePolicy sync

## Benefits

- ✅ Single source of truth (WorkerRegistry)
- ✅ No synchronization issues
- ✅ Dynamic worker updates work immediately
- ✅ Cleaner code (routers don't manage workers)
- ✅ CacheAwarePolicy stays in sync