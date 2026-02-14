# WASM Auth Example for sgl-model-gateway

This example demonstrates API key authentication middleware for sgl-model-gateway using the WebAssembly Component Model.

## Overview

This middleware validates API keys for requests to `/api` and `/v1` paths:

- Supports `Authorization: Bearer <key>` header
- Supports `Authorization: ApiKey <key>` header
- Supports `x-api-key` header
- Returns `401 Unauthorized` for missing or invalid keys

**Default API Key**: `secret-api-key-12345`

## Quick Start

### Build and Deploy

```bash
# Build
cd examples/wasm-guest-auth
./build.sh

# Deploy (replace file_path with actual path)
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "auth-middleware",
      "file_path": "/absolute/path/to/wasm_guest_auth.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'
```

### Customization

Modify `EXPECTED_API_KEY` in `src/lib.rs`:

```rust
const EXPECTED_API_KEY: &str = "your-secret-key";
```

## Testing

```bash
# Test unauthorized (returns 401)
curl -v http://localhost:3000/api/test

# Test authorized (passes)
curl -v http://localhost:3000/api/test \
  -H "Authorization: Bearer secret-api-key-12345"
```

## Troubleshooting

- Verify API key matches `EXPECTED_API_KEY` in code
- Check request header format and path (`/api` or `/v1`)
- Verify module is attached to `OnRequest` phase
- Check router logs for errors
