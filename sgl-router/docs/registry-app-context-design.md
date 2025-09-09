# Registry App Context Design

## Overview

Initialize PolicyRegistry and WorkerRegistry at server startup and inject them as dependencies through AppContext. This provides a clean, centralized architecture where registries are shared across all components.

## Current vs Proposed Architecture

### Current (Problematic)
```
Server starts
  → Creates RouterManager
    → RouterManager creates WorkerRegistry
    → RouterManager creates Routers
      → Each Router manages its own workers/policies
  → Workers added later via API
    → Complex synchronization needed
```

### Proposed (Clean)
```
Server starts
  → Creates WorkerRegistry (with initial workers from CLI/config)
  → Creates PolicyRegistry (with default policies)
  → Creates AppContext with registries
  → Creates RouterManager with AppContext
    → RouterManager uses shared registries
  → Creates Routers with AppContext
    → Routers use shared registries
```

## Implementation Design

### 1. Enhanced AppContext

```rust
// src/context.rs

use std::sync::Arc;
use crate::policies::{PolicyRegistry, PolicyConfig};
use crate::routers::WorkerRegistry;

pub struct AppContext {
    // Existing fields
    pub config: Arc<Config>,
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub metrics: Arc<MetricsRegistry>,
    
    // New registry fields
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
}

impl AppContext {
    pub fn new(
        config: Config,
        tokenizer: Option<Arc<Tokenizer>>,
        initial_workers: Vec<String>,  // From CLI args
    ) -> Result<Self> {
        // Initialize WorkerRegistry with initial workers
        let worker_registry = Arc::new(WorkerRegistry::new());
        
        // Add initial workers from CLI/config
        for worker_url in initial_workers {
            let worker = WorkerFactory::create_from_url(worker_url)?;
            worker_registry.register(Arc::from(worker));
        }
        
        // Initialize PolicyRegistry with default policies
        let policy_registry = Arc::new(PolicyRegistry::new(
            config.router_config.policy.clone(),
        ));
        
        // Pre-register common policy configurations if provided
        if let Some(policy_configs) = &config.router_config.model_policies {
            for (model_pattern, policy_config) in policy_configs {
                policy_registry.register_pattern(model_pattern, policy_config.clone());
            }
        }
        
        Ok(Self {
            config: Arc::new(config),
            tokenizer,
            metrics: Arc::new(MetricsRegistry::new()),
            worker_registry,
            policy_registry,
        })
    }
}
```

### 2. Server Initialization

```rust
// src/server.rs

pub async fn run_server(args: Args) -> Result<()> {
    // Parse config
    let config = Config::from_args(&args)?;
    
    // Initialize tokenizer if needed
    let tokenizer = if args.enable_grpc {
        Some(Arc::new(load_tokenizer(&args.tokenizer_path)?))
    } else {
        None
    };
    
    // Create app context with registries
    let app_context = AppContext::new(
        config.clone(),
        tokenizer,
        args.worker_urls.clone(),  // Initial workers from CLI
    )?;
    
    // Single router mode (backward compatible)
    if !config.enable_igw {
        // Create single router with context
        let router = create_single_router(&app_context).await?;
        
        // Add remaining workers to registry
        for url in args.worker_urls.iter().skip(app_context.worker_registry.count()) {
            let worker = WorkerFactory::create_from_url(url.clone())?;
            app_context.worker_registry.register(Arc::from(worker));
        }
        
        // Start server with router
        start_server(router, app_context).await?;
    } 
    // Multi-router mode (IGW)
    else {
        // Create RouterManager with context
        let router_manager = Arc::new(RouterManager::new(app_context.clone()).await?);
        
        // Start server with RouterManager
        start_igw_server(router_manager, app_context).await?;
    }
    
    Ok(())
}

async fn create_single_router(context: &AppContext) -> Result<Box<dyn RouterTrait>> {
    // Use registries from context
    RouterFactory::create_router(
        context.worker_registry.clone(),
        context.policy_registry.clone(),
        &context.config.router_config,
    ).await
}
```

### 3. RouterManager with Context

```rust
// src/routers/router_manager.rs

pub struct RouterManager {
    // Use registries from context
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    
    // Routers
    http_router: Option<Arc<dyn RouterTrait>>,
    http_pd_router: Option<Arc<dyn RouterTrait>>,
    grpc_router: Option<Arc<dyn RouterTrait>>,
    grpc_pd_router: Option<Arc<dyn RouterTrait>>,
}

impl RouterManager {
    pub async fn new(context: AppContext) -> Result<Self> {
        let worker_registry = context.worker_registry.clone();
        let policy_registry = context.policy_registry.clone();
        
        // Create routers with shared registries
        let http_router = if should_create_http_router(&context.config) {
            Some(Arc::new(
                RouterFactory::create_http_router(
                    worker_registry.clone(),
                    policy_registry.clone(),
                    &context.config.router_config,
                ).await?
            ))
        } else {
            None
        };
        
        let http_pd_router = if should_create_pd_router(&context.config) {
            Some(Arc::new(
                RouterFactory::create_pd_router(
                    worker_registry.clone(),
                    policy_registry.clone(),
                    &context.config.router_config,
                ).await?
            ))
        } else {
            None
        };
        
        // TODO: gRPC routers when tokenizer loading is dynamic
        
        Ok(Self {
            worker_registry,
            policy_registry,
            http_router,
            http_pd_router,
            grpc_router: None,
            grpc_pd_router: None,
        })
    }
    
    pub fn add_worker(&self, config: WorkerConfig) -> Result<WorkerApiResponse> {
        let worker = create_worker(config);
        let model_id = worker.model_id();
        
        // Get or create policy for this model
        let policy = self.policy_registry.on_worker_added(
            model_id,
            config.policy_hint.as_deref(),
        );
        
        // Register worker
        let worker_id = self.worker_registry.register(Arc::from(worker));
        
        info!(
            "Added worker {} for model {} using policy {}",
            worker_id.as_str(),
            model_id,
            policy.name()
        );
        
        Ok(WorkerApiResponse {
            success: true,
            message: format!("Worker {} added", worker_id.as_str()),
            worker: Some(worker_info),
        })
    }
    
    pub fn remove_worker(&self, url: &str) -> Result<WorkerApiResponse> {
        if let Some(worker) = self.worker_registry.get_by_url(url) {
            let model_id = worker.model_id();
            
            // Remove from registry
            self.worker_registry.remove_by_url(url)?;
            
            // Update policy registry (cleanup if last worker)
            self.policy_registry.on_worker_removed(model_id);
            
            Ok(WorkerApiResponse {
                success: true,
                message: format!("Worker removed"),
                worker: None,
            })
        } else {
            Err(WorkerError::NotFound)
        }
    }
}
```

### 4. Router with Context

```rust
// src/routers/http/router.rs

pub struct HttpRouter {
    // Registries from context
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    
    // Router specific
    connection_mode: ConnectionMode,
    worker_type: WorkerType,
}

impl HttpRouter {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            connection_mode: ConnectionMode::Http,
            worker_type: WorkerType::Regular,
        }
    }
    
    fn select_worker(&self, request: &Request) -> Option<Arc<dyn Worker>> {
        let model_id = extract_model_from_request(request)?;
        
        // Get workers for this model
        let workers = self.worker_registry.get_workers_for_model(
            &model_id,
            Some(self.connection_mode),
        );
        
        if workers.is_empty() {
            return None;
        }
        
        // Get policy for this model
        let policy = self.policy_registry.get_policy(&model_id)
            .unwrap_or_else(|| self.policy_registry.get_default_policy());
        
        // Convert Arc to Box for policy
        let boxed: Vec<Box<dyn Worker>> = workers
            .iter()
            .map(|w| w.clone_worker())
            .collect();
        
        // Select using policy
        let idx = policy.select_worker(&boxed, request.text())?;
        Some(workers[idx].clone())
    }
}
```

### 5. Configuration

```yaml
# config.yaml

# Default policy for all routers
default_policy: round_robin

# Optional: Pattern-based policy templates
model_policies:
  patterns:
    - match: "llama-*"
      policy: cache_aware
      config:
        cache_threshold: 0.7
    - match: "*-instruct"
      policy: shortest_queue
    - match: "gpt-*"
      policy: cache_aware
      config:
        cache_threshold: 0.8

# Initial workers (can also come from CLI)
initial_workers:
  - url: "http://gpu1:8080"
    model_id: "llama-3"
  - url: "http://gpu2:8080"
    model_id: "llama-3"
```

## Benefits

### 1. **Clean Initialization**
- Registries created once at startup
- Initial workers populated immediately
- No complex lazy initialization

### 2. **Dependency Injection**
- Clean dependency flow through AppContext
- Easy to test with mock registries
- Clear ownership and lifecycle

### 3. **Shared State**
- Single WorkerRegistry for all routers
- Single PolicyRegistry for all policies
- No synchronization issues

### 4. **Backward Compatible**
- Single router mode still works
- CLI worker URLs still supported
- Existing configs continue to work

### 5. **Extensible**
- Easy to add new registries
- Clean place for cross-cutting concerns
- Centralized configuration

## Migration Steps

1. **Add registries to AppContext**
   - Add worker_registry field
   - Add policy_registry field
   - Initialize in AppContext::new()

2. **Update server initialization**
   - Create AppContext early
   - Pass to RouterManager/Router creation
   - Use registries from context

3. **Update RouterManager**
   - Accept AppContext in new()
   - Use registries from context
   - Remove local registry creation

4. **Update Routers**
   - Accept registries in new()
   - Remove local worker management
   - Query registries for workers

5. **Update RouterFactory**
   - Accept registries as parameters
   - Pass to router constructors
   - Remove worker list parameters

## Example Flow

```
Server Start:
1. Parse CLI args: --worker http://gpu1:8080 --worker http://gpu2:8080
2. Create WorkerRegistry with initial workers
3. Create PolicyRegistry with default policy
4. Create AppContext with registries
5. Create RouterManager with AppContext
   - RouterManager uses shared registries
6. Create HTTP/gRPC routers
   - Each router gets registries from context
7. Server ready to handle requests

Request Flow:
1. Request arrives at RouterManager
2. RouterManager selects router
3. Router queries WorkerRegistry for workers
4. Router queries PolicyRegistry for policy
5. Policy selects worker
6. Request forwarded to worker

Add Worker:
1. API call to add worker
2. RouterManager adds to WorkerRegistry
3. PolicyRegistry updated with model→policy
4. All routers immediately see new worker

Remove Worker:
1. API call to remove worker
2. RouterManager removes from WorkerRegistry
3. PolicyRegistry cleaned up if last of model
4. All routers immediately see removal
```

## Summary

Moving PolicyRegistry and WorkerRegistry to AppContext provides:
- ✅ Clean initialization at server startup
- ✅ Proper dependency injection
- ✅ Single source of truth
- ✅ No synchronization issues
- ✅ Better testability
- ✅ Cleaner architecture

This is definitely the right architectural decision!