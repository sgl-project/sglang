# WASM Guest Examples for sgl-model-gateway

This directory contains example WASM middleware components demonstrating how to implement custom middleware for sgl-model-gateway using the WebAssembly Component Model.

## Examples Overview

### [wasm-guest-auth](./wasm-guest-auth/)

API key authentication middleware that validates API keys for requests to `/api` and `/v1` paths.

**Features:**
- Validates API keys from `Authorization` header or `x-api-key` header
- Returns `401 Unauthorized` for missing or invalid keys
- Attach point: `OnRequest` only

**Use case:** Protect API endpoints with API key authentication.

### [wasm-guest-logging](./wasm-guest-logging/)

Request tracking and status code conversion middleware.

**Features:**
- Adds tracking headers (`x-request-id`, `x-wasm-processed`, `x-processed-at`, `x-api-route`)
- Converts `500` errors to `503` for better client handling
- Attach points: `OnRequest` and `OnResponse`

**Use case:** Request tracing and error status code conversion.

### [wasm-guest-ratelimit](./wasm-guest-ratelimit/)

Rate limiting middleware with configurable limits.

**Features:**
- Rate limiting per identifier (API Key, IP, or Request ID)
- Default: 60 requests per minute
- Returns `429 Too Many Requests` when limit exceeded
- Attach point: `OnRequest` only

**Note:** This is a simplified demonstration with per-instance state. For production, use router-level rate limiting with shared state.

**Use case:** Protect against request flooding and abuse.

## Quick Start

Each example includes its own README with detailed build and deployment instructions. See individual example directories for:

- Build instructions
- Deployment configuration
- Customization options
- Testing examples

## Common Prerequisites

All examples require:

- Rust toolchain (latest stable)
- `wasm32-wasip2` target: `rustup target add wasm32-wasip2`
- `wasm-tools`: `cargo install wasm-tools`
- sgl-model-gateway running with WASM enabled, and told which directory modules
  may be loaded from:

  ```bash
  smg launch --enable-wasm --wasm-module-root /srv/wasm-modules
  ```

  Registration accepts a caller-supplied path, so `--wasm-module-root` is what
  confines it: a module is only accepted if it resolves to a location inside one
  of the roots. Pass the flag more than once, or give it several directories, to
  allow more than one. Enabling WASM without a root is refused at startup rather
  than defaulting to "anywhere on the filesystem", so place the built
  `.component.wasm` files under a root before registering them.

## Building All Examples

```bash
cd examples/wasm
for example in wasm-guest-auth wasm-guest-logging wasm-guest-ratelimit; do
  echo "Building $example..."
  cd $example && ./build.sh && cd ..
done
```

## Deploying Multiple Modules

You can deploy all three modules together:

```bash
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [
      {
        "name": "auth-middleware",
        "file_path": "/srv/wasm-modules/wasm_guest_auth.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}]
      },
      {
        "name": "logging-middleware",
        "file_path": "/srv/wasm-modules/wasm_guest_logging.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}, {"Middleware": "OnResponse"}]
      },
      {
        "name": "ratelimit-middleware",
        "file_path": "/srv/wasm-modules/wasm_guest_ratelimit.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}]
      }
    ]
  }'
```

Modules execute in the order they are deployed. If a module returns `Reject`, subsequent modules won't execute.
