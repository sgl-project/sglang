# Phase 2 Architecture Clarification

## Router Count: Exactly 4 Routers Maximum

In Phase 2, RouterManager will manage **exactly 4 routers total**, NOT one router per model:

### The 4 Routers

1. **HTTP Regular Router**
   - Handles standard HTTP requests
   - Can serve multiple models
   - Round-robin or priority-based worker selection

2. **HTTP PD Router**
   - Handles HTTP requests with Prefill/Decode separation
   - Can serve multiple models that have both prefill and decode workers
   - Coordinates prefill → decode pipeline

3. **gRPC Regular Router**
   - Handles standard gRPC requests
   - Can serve multiple models
   - Uses gRPC-specific features (tokenizer, parsers)

4. **gRPC PD Router**
   - Handles gRPC requests with Prefill/Decode separation
   - Can serve multiple models with PD workers
   - gRPC-optimized prefill → decode pipeline

## How Models Map to Routers

### Single Model, Multiple Routers
A single model can be served by multiple routers if it has appropriate workers:
```
Model: "llama-3.1-8b"
├── HTTP Regular Router (if has regular HTTP workers)
├── HTTP PD Router (if has prefill + decode HTTP workers)
├── gRPC Regular Router (if has regular gRPC workers)
└── gRPC PD Router (if has prefill + decode gRPC workers)
```

### Multiple Models, Single Router
A single router can serve multiple models:
```
HTTP Regular Router
├── Model: "llama-3.1-8b"
├── Model: "mistral-7b"
└── Model: "qwen-2.5"
```

## Router Selection Logic (Phase 2)

```rust
impl RouterManager {
    fn select_router(&self, request: &Request) -> Arc<dyn RouterTrait> {
        let model_id = extract_model_id(request);
        let connection_type = detect_connection_type(request);
        
        // 1. Check which routers can serve this model
        let available_routers = self.get_routers_for_model(&model_id);
        
        // 2. Filter by connection type preference
        let compatible_routers = available_routers
            .filter(|r| r.connection_mode() == connection_type);
        
        // 3. Prefer PD routers if available (better performance)
        if let Some(pd_router) = compatible_routers.find(|r| r.is_pd()) {
            return pd_router;
        }
        
        // 4. Fall back to regular router
        compatible_routers.first().unwrap_or(&self.default_router)
    }
}
```

## Worker to Router Assignment

### Phase 1 (Current)
- Single router handles all workers
- RouterManager manages workers but doesn't create routers

### Phase 2 (Future)
- RouterManager creates up to 4 routers based on worker availability
- Workers assigned to appropriate routers based on:
  - Connection mode (HTTP/gRPC)
  - Worker type (Regular/Prefill/Decode)
  - Model compatibility

## Example Deployment

```yaml
# Large deployment with multiple models
workers:
  # Llama model workers
  - url: "http://worker1:8080"
    model: "llama-3.1-8b"
    type: "regular"
    connection: "http"
  
  - url: "http://worker2:8080"
    model: "llama-3.1-8b"
    type: "prefill"
    connection: "http"
    
  - url: "http://worker3:8080"
    model: "llama-3.1-8b"
    type: "decode"
    connection: "http"
    
  # Mistral model workers
  - url: "grpc://worker4:50051"
    model: "mistral-7b"
    type: "regular"
    connection: "grpc"
    
  # Qwen model workers  
  - url: "http://worker5:8080"
    model: "qwen-2.5"
    type: "regular"
    connection: "http"

# Results in:
routers:
  - http_regular: [llama-3.1-8b, qwen-2.5]
  - http_pd: [llama-3.1-8b]
  - grpc_regular: [mistral-7b]
  # grpc_pd not created (no PD workers for gRPC)
```

## Benefits of 4-Router Design

1. **Simplicity**: Fixed number of routers, easy to understand and manage
2. **Efficiency**: Models share routers, reducing overhead
3. **Flexibility**: Each router optimized for its specific use case
4. **Scalability**: Can handle many models without router explosion
5. **Performance**: PD routers provide better performance when available