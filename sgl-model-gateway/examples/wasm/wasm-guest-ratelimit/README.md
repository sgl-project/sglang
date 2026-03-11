# WASM Rate Limit Example for sgl-model-gateway

This example demonstrates rate limiting middleware for sgl-model-gateway using the WebAssembly Component Model.

## Overview

This middleware provides rate limiting:

- **Default**: 60 requests per minute per identifier
- **Identifier Priority**: API Key > IP Address > Request ID
- **Response**: Returns `429 Too Many Requests` when limit exceeded

**Important**: This is a simplified demonstration. Since WASM components are stateless, each worker thread maintains its own counter. For production, implement rate limiting at the router/host level with shared state.

## Quick Start

### Build and Deploy

```bash
# Build
cd examples/wasm-guest-ratelimit
./build.sh

# Deploy (replace file_path with actual path)
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "ratelimit-middleware",
      "file_path": "/absolute/path/to/wasm_guest_ratelimit.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'
```

### Customization

Modify constants in `src/lib.rs`:

```rust
const RATE_LIMIT_REQUESTS: u64 = 100; // requests per window
const RATE_LIMIT_WINDOW_MS: u64 = 60_000; // time window in ms
```

## Testing

```bash
# Send multiple requests (first 60 succeed, then 429)
for i in {1..65}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    http://localhost:3000/v1/models \
    -H "Authorization: Bearer secret-api-key-12345"
done
```

## Limitations

- Per-instance state (not shared across workers)
- No cross-process state sharing
- Memory growth with unique identifiers
- State lost on instance restart

## Troubleshooting

- Verify module attached to `OnRequest` phase
- Check identifier extraction logic matches request format
- Note: Each WASM worker has separate counter
