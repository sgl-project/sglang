# [Router] Enhanced Worker Management API with Multi-Router Support

## Summary
This PR implements Phase 1 of the enhanced worker management system, introducing a comprehensive API for worker configuration and laying the groundwork for multi-router coordination. The changes enable fine-grained control over worker attributes including priority, cost, and gRPC-specific configurations while maintaining full backward compatibility.

## Motivation
The current worker management system only accepts worker URLs, limiting routing decisions to model ID alone. This enhancement enables:
- Priority-based worker selection for load balancing
- Cost-aware routing for resource optimization  
- Model-aware routing with additional metadata
- gRPC-specific configurations (tokenizer, parsers, chat templates)
- Multi-router coordination for complex deployments

## Key Changes

### 1. Enhanced Worker Trait (`src/core/worker.rs`)
- Added model awareness methods: `model_id()`, `priority()`, `cost()`
- Added gRPC-specific methods: `tokenizer_path()`, `reasoning_parser()`, `tool_parser()`, `chat_template()`
- Default implementations use metadata labels for backward compatibility

### 2. Worker Registry (`src/core/worker_registry.rs`)
- Centralized registry with multiple indices (by ID, model, type, connection mode)
- Model-based worker lookup for intelligent routing
- URL-based worker management for RESTful API
- Thread-safe implementation using DashMap

### 3. Worker Management API (`src/protocols/worker_spec.rs`)
- Comprehensive `WorkerConfigRequest` structure with all configuration options
- `WorkerInfo` response with complete worker state
- Statistics tracking with `WorkerStats` and `WorkerTypeStats`
- Error handling with structured `WorkerErrorResponse`

### 4. Router Manager (`src/routers/router_manager.rs`)
- Central coordinator for multi-router deployments (only active when `enable_igw=true`)
- Worker lifecycle management (add, remove, list, get)
- Model-to-router mapping for model-aware routing
- Automatic server info querying for model discovery
- Support for priority and cost-based routing decisions

### 5. Server Integration (`src/server.rs`)
- RESTful endpoints: `POST /workers`, `GET /workers`, `DELETE /workers`, `GET /workers/{url}`
- Conditional RouterManager initialization based on `enable_igw` flag
- Dependency injection through `AppContext`
- Full backward compatibility with existing single-router deployments

## Architecture

### Dual-Mode Operation
The system supports two modes based on the `enable_igw` flag:

#### Single Router Mode (`enable_igw=false`) - Default
- Router owns and manages workers directly
- No RouterManager initialization
- Preserves current behavior for existing deployments
- Minimal overhead

#### Multi-Router Mode (`enable_igw=true`)
- RouterManager coordinates all routers and workers
- Centralized worker registry
- Model-aware routing across multiple routers
- Enhanced worker configuration capabilities

## API Examples

### Add Worker with Full Configuration
```bash
curl -X POST http://localhost:8080/workers \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://worker1:8080",
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "priority": 100,
    "cost": 0.5,
    "tokenizer_path": "/models/tokenizer",
    "reasoning_parser": "llama3",
    "tool_parser": "json",
    "chat_template": "llama3",
    "labels": {
      "region": "us-west",
      "gpu": "A100"
    }
  }'
```

### List Workers with Statistics
```bash
curl http://localhost:8080/workers
```

Response:
```json
{
  "workers": [...],
  "total": 10,
  "stats": {
    "total_workers": 10,
    "healthy_workers": 8,
    "total_models": 3,
    "total_load": 150,
    "by_type": {
      "regular": 6,
      "prefill": 2,
      "decode": 2
    }
  }
}
```

## Testing
- All existing tests pass
- Worker registry unit tests added
- Backward compatibility verified
- Manual testing with both single and multi-router modes

## Breaking Changes
None. The changes are fully backward compatible:
- Default behavior unchanged (`enable_igw=false`)
- Legacy endpoints preserved
- Worker trait changes use default implementations

## Future Work (Phase 2)
- Service discovery enhancement with automatic `/get_server_info` querying
- Router modifications for stateless worker handling
- PD Router validation for complete worker sets
- Health monitoring and circuit breaker integration

## Configuration
Enable multi-router mode in configuration:
```yaml
router_config:
  enable_igw: true  # Enable multi-router coordination
```

## Dependencies
No new dependencies added. Uses existing:
- `dashmap` for thread-safe collections
- `serde` for serialization
- `axum` for HTTP endpoints
- `reqwest` for HTTP client

## Review Checklist
- [x] Code follows project style guidelines
- [x] All tests pass
- [x] No clippy warnings
- [x] API documentation included
- [x] Backward compatibility maintained
- [x] Error handling implemented
- [x] Thread safety ensured