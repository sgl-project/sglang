# WASM Logging Example for sgl-model-gateway

This example demonstrates logging and tracing middleware for sgl-model-gateway using the WebAssembly Component Model.

## Overview

This middleware provides:

- **Request Tracking** - Adds tracking headers (`x-request-id`, `x-wasm-processed`, `x-processed-at`, `x-api-route`)
- **Status Code Conversion** - Converts `500` errors to `503`

## Quick Start

### Build and Deploy

```bash
# Build
cd examples/wasm-guest-logging
./build.sh

# Deploy (replace file_path with actual path)
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "logging-middleware",
      "file_path": "/absolute/path/to/wasm_guest_logging.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}, {"Middleware": "OnResponse"}]
    }]
  }'
```

### Customization

Modify `on_request` or `on_response` functions in `src/lib.rs` to add custom tracking headers or status code conversions.

## Testing

```bash
# Check tracking headers
curl -v http://localhost:3000/v1/models 2>&1 | \
  grep -E "(x-request-id|x-wasm-processed|x-processed-at)"

# Test status code conversion (requires endpoint returning 500)
curl -v http://localhost:3000/some-endpoint 2>&1 | grep -E "(< HTTP|500|503)"
```

## Troubleshooting

- Verify module attached to both `OnRequest` and `OnResponse` phases
- Check router logs for execution errors
- Ensure module built successfully
