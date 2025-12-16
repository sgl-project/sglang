# WebAssembly (WASM) Extensibility for sgl-model-gateway

This module provides WebAssembly-based extensibility for sgl-model-gateway, enabling dynamic, safe, and portable middleware execution without requiring router restarts or recompilation.

## Overview

The WASM module allows you to extend sgl-model-gateway functionality by deploying WebAssembly components that can:

- **Intercept requests/responses** at various lifecycle points (OnRequest, OnResponse)
- **Modify HTTP headers and bodies** before/after processing
- **Reject requests** with custom status codes
- **Execute custom logic** in a sandboxed, isolated environment

## Architecture

### Components

The WASM module consists of several key components:

```
src/wasm/
├── module.rs           # Data structures (metadata, types, attach points)
├── module_manager.rs   # Module lifecycle management (add/remove/list)
├── runtime.rs          # WASM execution engine and thread pool
├── route.rs            # HTTP API endpoints for module management
├── spec.rs             # WASM interface types bindings and type conversions
├── types.rs            # Generic input/output types
├── errors.rs           # Error definitions
├── config.rs           # Runtime configuration
└── interface/          # WebAssembly Interface Types definitions
```

### Execution Flow

```
1. HTTP Request arrives at router
   ↓
2. Middleware chain checks for WASM modules attached to OnRequest
   ↓
3. For each module:
   a. Module manager retrieves pre-loaded WASM bytes
   b. Runtime executes component in isolated worker thread
   c. Component processes request via WASM type interface
   d. Returns Action (Continue/Reject/Modify)
   ↓
4. If Continue: proceed to next middleware/upstream
   If Reject: return error response immediately
   If Modify: apply changes (headers, body, status)
   ↓
5. After upstream response:
   - Modules attached to OnResponse process response
   - Apply modifications
   ↓
6. Return final response to client
```

### WebAssembly Interface Types

The module uses the WebAssembly Component Model with WASM interface type for type-safe communication between host and WASM components:

- **Request Processing**: `middleware-on-request::on-request(req: Request) -> Action`
- **Response Processing**: `middleware-on-response::on-response(resp: Response) -> Action`
- **Actions**: `Continue`, `Reject(status)`, or `Modify(modify-action)`

See [`interface/`](./interface/) for the complete interface definition.

## Usage

### Prerequisites

- sgl-model-gateway compiled with WASM support
- Rust toolchain (for building WASM components)
- `wasm32-wasip2` target: `rustup target add wasm32-wasip2`
- `wasm-tools`: `cargo install wasm-tools`

### Starting the Router

Enable WASM support when starting the router:

```bash
./sgl-model-gateway --enable-wasm --worker-urls=http://0.0.0.0:30000 --port=3000
```

### Deploying a WASM Module

Use the `/wasm` POST endpoint to deploy modules:

```bash
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "my-middleware",
      "file_path": "/path/to/my-component.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'
```

### Managing Modules

**List all modules:**
```bash
curl http://localhost:3000/wasm
```

**Remove a module:**
```bash
curl -X DELETE http://localhost:3000/wasm/{module-uuid}
```

### Module Configuration

Each module requires:

- **name**: Unique identifier for the module
- **file_path**: Absolute path to the WASM component file
- **module_type**: Currently supports `"Middleware"`
- **attach_points**: List of attachment points, e.g., `[{"Middleware": "OnRequest"}]`

Supported attachment points:
- `{"Middleware": "OnRequest"}` - Execute before forwarding to upstream
- `{"Middleware": "OnResponse"}` - Execute after receiving upstream response
- `{"Middleware": "OnError"}` - Not yet implemented

## Examples

See [`examples/wasm/`](../../examples/wasm/) for complete examples:

1. **[wasm-guest-auth](../../examples/wasm/wasm-guest-auth/)** - API key authentication middleware
2. **[wasm-guest-logging](../../examples/wasm/wasm-guest-logging/)** - Request tracking and status code conversion
3. **[wasm-guest-ratelimit](../../examples/wasm/wasm-guest-ratelimit/)** - Rate limiting middleware

Each example includes:
- Complete source code
- Build instructions
- Deployment examples
- Testing guidelines

## Security and Resource Management

### Sandboxing

WASM modules run in isolated environments provided by wasmtime, preventing:
- Direct system access
- Memory corruption of the host process
- Unauthorized network access
- File system access (unless explicitly granted via WASI)

### Resource Limits

Runtime configuration allows setting limits:

```rust
WasmRuntimeConfig {
    max_memory_pages: 1024,           // 64MB limit
    max_execution_time_ms: 1000,       // 1 second timeout
    max_stack_size: 1024 * 1024,      // 1MB stack
    thread_pool_size: 4,               // Worker threads
    module_cache_size: 10,             // Cached modules per worker
}
```

### Error Handling

- Failed module executions are logged and don't crash the router
- Invalid WASM components are rejected during load time
- Metrics track execution success/failure rates

## Metrics

The module exposes execution metrics via the `/wasm` GET endpoint:

```json
{
  "modules": [...],
  "metrics": {
    "total_executions": 1000,
    "successful_executions": 995,
    "failed_executions": 5,
    "total_execution_time_ms": 50000,
    "max_execution_time_ms": 150,
    "average_execution_time_ms": 50.0
  }
}
```

## Development

### Building WASM Components

WASM components must be built using the Component Model. For Rust:

```bash
# 1. Build as WASM module
cargo build --target wasm32-wasip2 --release

# 2. Wrap into component format
wasm-tools component new target/wasm32-wasip2/release/my_module.wasm \
  -o my_module.component.wasm
```

### WASM Interface Type

Define your component using the WASM interface from `interface/spec.*`:

```rust
wit_bindgen::generate!({
    path: "../../../src/wasm/interface",
    world: "sgl-model-gateway",
});

use exports::sgl::model_gateway::middleware_on_request::Guest as OnRequestGuest;
use sgl::model_gateway::middleware_types::{Request, Action};

struct Middleware;

impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        // Your logic here
        Action::Continue
    }
}

export!(Middleware);
```
