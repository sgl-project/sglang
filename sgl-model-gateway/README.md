# SGLang Model Gateway

High-performance model routing control and data plane for large-scale LLM deployments. The gateway orchestrates fleets of workers, balances traffic across HTTP and gRPC backends, and exposes OpenAI-compatible APIs with pluggable history storage and tool integrations—while remaining deeply optimized for the SGLang serving runtime.

## Overview
- Unified control plane for registering, monitoring, and orchestrating prefill, decode, and regular workers across heterogeneous model fleets.
- Data plane that routes requests across HTTP, PD (prefill/decode), gRPC, and OpenAI-compatible backends with shared reliability features.
- Industry-first gRPC pipeline with native Rust tokenization, reasoning, and tool-call execution for high-throughput OpenAI-compatible serving.
- Multi-model inference gateway mode (`--enable-igw`) that runs several routers at once and applies per-model policies.
- Conversation, response, and chat-history connectors that centralize state at the router, enabling compliant sharing across models/MCP loops with in-memory, no-op, or Oracle ATP storage options.
- Built-in reliability primitives: retries with exponential backoff, circuit breakers, token-bucket rate limiting, and queuing.
- First-class observability with structured logging, OpenTelemetry trace and Prometheus metrics.

### Architecture at a Glance
**Control Plane**
- Worker Manager validates workers, discovers capabilities, and keeps the registry in sync.
- Job Queue serializes background operations (add/remove) and exposes status via `/workers/{worker_id}`.
- Background health checker and load monitor keep circuit breakers and policies informed.
- Optional Kubernetes service discovery keeps the registry aligned with pods.

**Data Plane**
- SGLang HTTP routers for regular and PD (prefill/decode) traffic with policy-aware selection.
- SGLang gRPC router and pipeline that stream tokenized requests through SRT gRPC workers with fully Rust tokenizer, reasoning parser, and tool parser implementations for maximal OpenAI API performance, supporting both single-stage and PD serving topologies.
- OpenAI router that proxies OpenAI-style requests, responses, and conversations to remote vendors (OpenAI, xAI, Gemini, and other OpenAI-compatible providers) while preserving streaming/SSE semantics.
- Router Manager coordinates multiple router implementations when IGW is enabled.
- Resilience layer delivers token-bucket rate limiting, request queuing, retry executor, and per-worker circuit breakers to keep traffic flowing through failures.
- Advanced load balancing with cache-aware request reuse, load-aware (power-of-two) selection, and per-model policy overrides.

## Feature Highlights
- Multiple load balancing strategies (`random`, `round_robin`, `cache_aware`, `power_of_two`, `bucket`) with DP-aware scheduling.
- Multi-model HTTP serving and inference gateway routing with model-specific policies.
- Prefill/decode disaggregation, including bootstrap port handling and cache-aware merging.
- gRPC routing with fully Rust tokenizer loading, reasoning parser selection, and tool parser integration for OpenAI-compatible endpoints—supporting streaming and non-streaming modes across DeepSeek, Llama, Kimi K2, Qwen, GPT-OSS, Mistral, Step-3, GLM4, GLM4.7 and other reasoning-capable models.
- OpenAI-compatible `/v1/chat/completions`, `/v1/responses`, `/v1/conversations`, `/v1/embeddings`, `/v1/rerank`, `/v1/classify` endpoints.
- **Tokenization APIs**: HTTP endpoints for tokenize (`/v1/tokenize`) and detokenize (`/v1/detokenize`) with batch support; tokenizer management APIs for dynamic registration.
- **Parser endpoints**: Reasoning parser (`/parse/reasoning`) and function call parser (`/parse/function_call`) for separating reasoning content and extracting tool calls.
- Native MCP client integration supporting all MCP transport protocols (STDIO, HTTP, SSE, and Streamable) for tool execution loops.
- Pluggable history connectors: in-memory, disabled, Oracle ATP, or PostgreSQL (with pooling and credential support).
- Reliability controls: retry with jitter, worker-scoped circuit breakers, token bucket limiter with optional queue, and cache flush APIs.
- Service discovery for regular and PD workloads with independent selectors.
- **Comprehensive observability**: 40+ Prometheus metrics across HTTP, router, worker, circuit breaker, retry, discovery, MCP, and database layers; OpenTelemetry tracing with OTLP export; structured logging with request ID propagation.

## Documentation
- **User Guide**: [docs.sglang.io/advanced_features/sgl_model_gateway.html](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
- Additional guides, API references, and deployment patterns are continuously updated alongside SGLang releases.

## Installation

### Docker
Pre-built Docker images are available on Docker Hub with multi-architecture support (x86_64 and ARM64):
```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### Prerequisites
- **Rust and Cargo**
  ```bash
  # Install rustup (Rust installer and version manager)
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  # Reload shell environment
  source "$HOME/.cargo/env"

  # Verify installation
  rustc --version
  cargo --version
  ```
- **Python** with `pip` and virtualenv tooling available.

### Rust Binary
```bash
# Build release binary
cargo build --release
```

### Python Package
```bash
pip install maturin

# Fast development mode (debug build, no wheel, instant)
# Uses system OpenSSL (requires libssl-dev/openssl-devel)
cd bindings/python
maturin develop

# Production build (optimized, creates wheel)
# Uses vendored OpenSSL (cross-platform compatibility)
cd bindings/python
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl

# Development build with system OpenSSL (faster)
# Requires: apt install libssl-dev pkg-config (Ubuntu/Debian)
#       or: yum install openssl-devel (RHEL/CentOS)
cd bindings/python
maturin build --release --out dist
pip install --force-reinstall dist/*.whl
```
> **Note:** Python bindings are located in `bindings/python/` with their own Cargo.toml. Use `maturin develop` for fast iteration during development (builds in debug mode and installs directly). Use `maturin build --release --features vendored-openssl` for production wheels with full optimizations (opt-level="z", lto="fat") and cross-platform compatibility. The package uses abi3 support for Python 3.8+ compatibility.

## Checking Version

After installation, verify the installation and check version information:

```bash
# Simple version (Rust binary)
./target/release/sgl-model-gateway --version
# or use aliases
./target/release/smg --version
./target/release/amg --version

# Full version info with build details
./target/release/sgl-model-gateway --version-verbose

# Python CLI
amg --version
amg --version-verbose
python3 -m sglang_router --version
```

The `--version` (or `-V`) flag displays the version string. Use `--version-verbose` for comprehensive build information including Git commit, build time, compiler versions, and platform details.

## Quick Start
### Regular HTTP Routing
- **Rust binary**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```
  `cargo run --release -- …` provides the same behavior during development.
- **Python launcher**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```

### Prefill/Decode Disaggregation (PD)
- **Rust binary**
  ```bash
  ./target/release/sgl-model-gateway \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
  ```
- **Python launcher**
  ```bash
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware
  ```
Prefill entries accept an optional bootstrap port. PD mode merges prefill metadata with decode outputs and streams results back to the client.

### Multi-Model Inference Gateway
Enable IGW mode to route multiple models through a single router while applying per-model policies:
```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-a:8000",
        "model_id": "mistral",
        "priority": 10,
        "labels": {"tier": "gold"}
      }'

# Add another worker with a different model/policy hint
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-b:8000",
        "model_id": "llama3",
        "priority": 20,
        "labels": {"policy": "power_of_two", "tier": "silver"}
      }'

# Inspect registered workers
curl http://localhost:30000/workers
```
Sample response (http workers):
```json
{
  "workers": [
    {"id":"2f3a0c3e-3a7b-4c3f-8c70-1b7d4c3a6e1f","url":"http://0.0.0.0:31378","model_id":"mistral","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"},
    {"id":"9b0f6c2a-1c4f-4c2a-9f4a-1f2a6c0b9d3e","url":"http://0.0.0.0:34881","model_id":"llama3","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"}
  ],
  "total": 2,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 2
  }
}
```
Add more workers with the same API; include optional `labels` (for per-model policies) or `tokenizer_path` / `reasoning_parser` / `tool_parser` fields as needed. `/workers/{worker_id}` exposes queued job status while background jobs finalize registration.

### gRPC Routing
- **Rust binary**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls grpc://worker-grpc-0:31001 grpc://worker-grpc-1:31002 \
    --tokenizer-path /path/to/tokenizer.json \
    --reasoning-parser deepseek-r1 \
    --tool-call-parser json
  ```
- **Python router**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls grpc://127.0.0.1:20000 \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8080
  ```
The gRPC router tokenizes inputs locally, supports tool-call parsing, and streams responses. It supports both regular HTTP-equivalent serving and PD (prefill/decode) serving when the worker registry contains PD workers. Provide `--model-path` or `--tokenizer-path` (HuggingFace ID or local directory) whenever connection mode resolves to gRPC.
Use `--reasoning-parser` to select built-in reasoning pipelines (DeepSeek-R1, Qwen3, Step-3, GLM4, GLM4.7, etc.) and `--tool-call-parser` for JSON/Pythonic/XML tool contracts in streaming or non-streaming modes.

### OpenAI Backend Mode
Route requests to OpenAI or OpenAI-compatible endpoints:

```bash
# Route to OpenAI API
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \

# Route to custom OpenAI-compatible endpoint (Gemini, xAI, etc.)
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls http://my-openai-compatible-service:8000 \
```

**Notes**
- OpenAI backend mode acts as a proxy to a single remote endpoint; load balancing is not applied.
- Provide exactly one `--worker-urls` entry per router instance.
- The Rust binary supports the same flags (`./target/release/sgl-model-gateway --backend openai ...`).

### MCP Integration
The SGL Model Gateway provides native Model Context Protocol (MCP) client integration, enabling tool calling across STDIO, SSE, and Streamable transports. MCP servers are configured via a YAML configuration file and registered at startup through the workflow engine.

#### Basic Usage
```bash
# Rust binary
./target/release/sgl-model-gateway \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000

# Python launcher
python3 -m sglang_router.launch_router \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000
```

#### MCP Configuration File
Create an MCP configuration file to define servers, transports, and connection settings:

```yaml
servers:
  - name: "filesystem"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    protocol: "stdio"
    required: false

  - name: "github"
    url: "https://api.github.com/mcp"
    token: "ghp_xxxxx"
    protocol: "sse"
    required: false

  - name: "custom-tools"
    url: "https://tools.example.com/mcp"
    protocol: "streamable"
    required: true

pool:
  max_connections: 100
  idle_timeout: 300  # seconds

proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8443"
  no_proxy: "localhost,127.0.0.1,*.internal"

inventory:
  enable_refresh: true
  tool_ttl: 300  # seconds - how long tools are considered fresh
  refresh_interval: 300  # seconds - background refresh interval
```

#### Configuration Options

**Server Configuration** (`servers` array):
- `name`: Unique identifier for the MCP server
- `command` + `args`: For STDIO transport (local process execution)
- `url`: For SSE or Streamable transports (HTTP/HTTPS endpoints)
- `token`: Optional authentication token for HTTP-based transports
- `protocol`: Protocol type (`"sse"`, `"streamable"`, or `"stdio"`)
- `required`: If `true`, router fails to start if server is unreachable (default: `false`)
- `envs`: Environment variables for STDIO processes (optional)
- `proxy`: Per-server proxy override (set to `null` to bypass global proxy)

**Connection Pool** (`pool`):
- `max_connections`: Maximum pooled connections for dynamic servers (default: 100)
- `idle_timeout`: Idle connection timeout in seconds before cleanup (default: 300)

**Proxy Configuration** (`proxy`):
- `http`/`https`: Proxy URLs for MCP server connections (not LLM traffic)
- `no_proxy`: Comma-separated hosts to exclude from proxying (supports wildcards)
- **Note**: Proxy settings are currently ignored for `streamable` transport. Use STDIO or SSE transports if proxy support is required.

**Inventory Settings** (`inventory`):
- `enable_refresh`: Enable automatic background refresh of tool inventory (default: true)
- `tool_ttl`: Tool cache TTL in seconds - how long tools are considered fresh (default: 300)
- `refresh_interval`: Background refresh interval in seconds - proactive inventory refresh (default: 300)

#### Transport Types

**STDIO** (Local Process):
```yaml
name: "local-tools"
command: "python"
args: ["-m", "my_mcp_server"]
envs:
  API_KEY: "secret"
  DEBUG: "true"
```

**SSE** (Server-Sent Events):
```yaml
name: "remote-sse"
url: "https://mcp.example.com/events"
token: "bearer-token"
protocol: "sse"
```

**Streamable** (Bidirectional Streaming):
```yaml
name: "streaming-tools"
url: "https://mcp.example.com/stream"
protocol: "streamable"
required: true
```

#### Server Lifecycle
- MCP servers are registered via the workflow engine with retry logic (100 attempts, 2-hour timeout for STDIO servers)
- Discovery phase identifies tools, prompts, and resources
- Tool inventory is cached with configurable TTL and periodic refresh
- Failed optional servers log warnings; required servers halt startup
- Static servers (from config) are permanent; dynamic servers (per-request) use connection pooling

Check Prometheus metrics for MCP activity (`mcp_*` metrics) and workflow job status via the admin API.

### Python Launcher (Router + Workers)
Launch router and SGLang worker processes together; `launch_server` spins up workers (HTTP or gRPC) and the router in one shot.
```bash
python3 -m sglang_router.launch_server --host 0.0.0.0
```
Add flags as needed for production deployments:
```bash
python3 -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode
```
Omit `--grpc-mode` to start HTTP workers; the router automatically configures worker URLs and schedules them based on the provided DP size.

### Mini Load Balancer (Debug)
```bash
python3 -m sglang_router.launch_router \
  --mini-lb \
  --pd-disaggregation \
  --prefill http://localhost:30001 \
  --decode http://localhost:30011
```
MiniLB forwards PD requests using simple random routing and is intended for local debugging only.

### Running Worker Servers
Use upstream SGLang binaries to start dedicated worker processes.
- **Prefill worker server (gRPC mode)**:
  ```bash
  python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 20000 \
    --tp-size 1 \
    --grpc-mode
  ```
  Remove `--grpc-mode` for HTTP workers. Combine with the router commands above to register the worker via CLI flags or the control-plane API.

## Control Plane

### Worker Lifecycle & Job Queue
- `JobQueue` handles asynchronous add/remove operations to avoid blocking clients.
- `WorkerManager` inspects worker metadata (`/get_server_info`, `/get_model_info`), tracks load, and exposes `flush_cache` and `get_loads`.
- Per-worker circuit breakers and health probes keep the registry healthy; load monitor feeds metrics to cache-aware and power-of-two policies.

### Administrative & Worker APIs
| Method   | Path             | Description                                                                                                                                               |
|----------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `POST`   | `/workers`       | Queue worker registration (prefill/decode/regular). Body matches `WorkerConfigRequest`. Returns `202 Accepted` while the job queue processes the request. |
| `GET`    | `/workers`       | List workers with health, load, policy metadata, and queued job status.                                                                                   |
| `GET`    | `/workers/{worker_id}` | Inspect a specific worker or job queue entry (UUID).                                                                                                   |
| `PUT`    | `/workers/{worker_id}` | Queue worker update by UUID.                                                                                                                           |
| `DELETE` | `/workers/{worker_id}` | Queue worker removal by UUID.                                                                                                                          |
| `POST`   | `/flush_cache`   | Trigger cache flush across HTTP workers with success/failure breakdown.                                                                                   |
| `GET`    | `/get_loads`     | Sample current load reported by each worker.                                                                                                              |

All administrative routes inherit router API-key protection when `--api-key` is supplied. Job status includes `pending`, `processing`, and `failed` phases with timestamps.

### Service Discovery
Enable Kubernetes discovery to reconcile workers automatically:
```bash
./target/release/sgl-model-gateway \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace sglang-system \
  --service-discovery-port 8000
```
PD mode accepts dedicated selectors:
```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```
Prefill pods can expose bootstrap ports via the `sglang.ai/bootstrap-port` annotation. RBAC must allow `get`, `list`, and `watch` on pods.

## Data Plane

### Router Capabilities (HTTP & gRPC)
Both router stacks:
- Share load-balancing policies (random, round-robin, cache-aware, power-of-two) with DP-aware scheduling, retries, circuit breakers, and rate limiting.
- Record metrics per request, track running load, and integrate with the router-wide policy registry.

The HTTP router exposes the full OpenAI-compatible surface area (`/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/rerank`, etc.). The gRPC router delivers blazing-fast `/generate` and `/v1/chat/completions` today, with the remaining endpoints returning `501 Not Implemented` until their pipelines are finalised.

#### HTTP Router specifics
- **Regular router** handles classic single-stage workers with per-model policy overrides.
- **Prefill/Decode router** coordinates disaggregated prefill and decode workers, merges metadata, and manages streaming fan-in.

#### gRPC Router specifics
- Industry-first fully Rust implementation of an OpenAI-compatible gRPC inference gateway, including tokenizer, reasoning parser, and tool parser execution in-process for maximum throughput.
- Supports both single-stage and PD (prefill/decode) worker topologies; the router automatically selects the appropriate pipeline per model.
- Provides the same `/v1/*` APIs as the HTTP router while streaming tokenized requests/responses directly to SRT gRPC workers.
- Built-in reasoning parsers for DeepSeek, Qwen, Llama, Mistral, GPT-OSS, Step-3, GLM4, GLM4.7, Kimi K2, and other structured-thought models.
- Tool-call parsers for JSON, Pythonic, XML, and custom schemas with streaming and non-streaming execution loops.
- Tokenizer factory supporting HuggingFace models, local tokenizer.json files, and chat template overrides (see `src/tokenizer`).
- Explore the code paths in `src/reasoning_parser`, `src/tool_parser`, and `src/tokenizer` for the end-to-end Rust implementations that power gRPC mode.

### OpenAI Router
- Proxies OpenAI-compatible chat completions and responses APIs, preserving headers and SSE streams end-to-end.
- Supports `/v1/responses` background jobs with cancellation, deletion, and listing input items—enabling agentic, multi-turn orchestration without persisting data at remote vendor endpoints.
- Conversation APIs (`/v1/conversations` and `/v1/conversations/{id}/items`) interact with the configured conversation storage backend for compliant chat-history management. Conversation state lives at the router tier, so the same history can drive different models or MCP loops without leaking data to upstream vendors.
- Chat history, agentic multi-turn `/v1/responses`, and the native MCP client (STDIO/HTTP/SSE/Streamable transports) are designed to satisfy enterprise data-privacy requirements by keeping sensitive state within the router.

### Request Endpoints
| Endpoint                                                                         | Notes                                                      |
|----------------------------------------------------------------------------------|------------------------------------------------------------|
| `POST /generate`                                                                 | SGLang generate API.                                       |
| `POST /v1/chat/completions`                                                      | OpenAI-compatible chat. Supports streaming and tool calls. |
| `POST /v1/completions`                                                           | OpenAI-compatible text completions.                        |
| `POST /v1/responses`                                                             | Create background responses, returns response IDs.         |
| `GET /v1/responses/{id}`                                                         | Retrieve stored responses.                                 |
| Conversation endpoints (`/v1/conversations`, `/v1/conversations/{id}`, `/v1/conversations/{id}/items`) | Manage chat history.                                       |
| `POST /v1/embeddings`                                                            | Forward embedding requests (HTTP and gRPC).                |
| `POST /v1/rerank`, `POST /rerank`                                                | Ranking APIs.                                              |
| `POST /v1/classify`                                                              | Text classification endpoint.                              |

Public health endpoints (`/liveness`, `/readiness`, `/health`, `/health_generate`) reflect registry state; readiness ensures PD workers are paired and IGW has at least one healthy route.

### Tokenization Endpoints

The gateway provides HTTP endpoints for text tokenization, designed to mirror the SGLang Python tokenization API with support for batch operations.

| Endpoint                      | Method   | Description                                           |
|-------------------------------|----------|-------------------------------------------------------|
| `POST /v1/tokenize`           | `POST`   | Tokenize text to token IDs (single or batch).         |
| `POST /v1/detokenize`         | `POST`   | Convert token IDs back to text (single or batch).     |
| `POST /v1/tokenizers`         | `POST`   | Register a new tokenizer (async, returns job status). |
| `GET /v1/tokenizers`          | `GET`    | List all registered tokenizers.                       |
| `GET /v1/tokenizers/{id}`     | `GET`    | Get tokenizer info by UUID.                           |
| `GET /v1/tokenizers/{id}/status` | `GET` | Check async tokenizer loading status.                 |
| `DELETE /v1/tokenizers/{id}`  | `DELETE` | Remove a tokenizer from the registry.                 |

**Tokenize Request:**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

**Batch Tokenize Request:**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": ["Hello", "World", "How are you?"]
}
```

**Tokenize Response:**
```json
{
  "tokens": [15339, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

**Detokenize Request:**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [15339, 11, 1917, 0],
  "skip_special_tokens": true
}
```

**Add Tokenizer (async registration):**
```bash
# Register from HuggingFace
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'

# Check status
curl http://localhost:30000/v1/tokenizers/{tokenizer_id}/status
```

### Parser Endpoints

The gateway provides admin endpoints for parsing reasoning content and function calls from LLM outputs.

| Endpoint                 | Method | Description                                            |
|--------------------------|--------|--------------------------------------------------------|
| `POST /parse/reasoning`  | `POST` | Separate reasoning (`<think>`) from normal text.       |
| `POST /parse/function_call` | `POST` | Parse function/tool calls from text.                |

**Separate Reasoning Request:**
```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "parser": "deepseek-r1"
}
```

**Response:**
```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step..."
}
```

**Supported Reasoning Parsers:**
- `deepseek-r1` - DeepSeek-R1 (initial reasoning mode)
- `qwen3` - Qwen-3 models
- `qwen3-thinking` / `qwen-thinking` - Qwen thinking variant
- `kimi` - Kimi K2 with Unicode tokens
- `glm45` / `glm47` - GLM-4.5/4.6/4.7 models
- `step3` - Step-3 models
- `minimax` - MiniMax models

**Function Call Parsing:**
```json
{
  "text": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}",
  "parser": "json"
}
```

Supported tool parsers: `json`, `python`, `xml`.

## Conversations, Responses, and Data Connectors
- `--history-backend memory` (default) stores responses and conversations in-process.
- `--history-backend none` disables persistence while keeping APIs.
- `--history-backend oracle` uses Oracle Autonomous Database; provide credentials via flags or environment variables.
- `--history-backend postgres` uses PostgreSQL Database.
- Conversation item storage mirrors the history backend (Oracle or memory). The same storage powers OpenAI `/responses` and conversation APIs.

### History Backend (OpenAI Router Mode)
Store conversation and response data for tracking, debugging, or analytics.

> **Note:** History backends are currently supported only when running with `--backend openai`. gRPC mode support for the `/v1/responses` API is planned.

#### Available storage options
- **Memory** (default): In-memory storage, fast but ephemeral.
- **None**: No storage, minimal overhead.
- **Oracle**: Persistent storage backed by Oracle Autonomous Database.
- **Postgres**: Persistent storage backed by PostgreSQL Database.

```bash
# Memory backend (default)
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory

# No storage for maximum performance
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend none

# Oracle ATP backend (see configuration below)
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle

# PostgreSQL backend
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend postgres
```

#### Oracle configuration
Install the Oracle Instant Client and set `LD_LIBRARY_PATH` accordingly. Choose **one** connection method:
```bash
# Option 1: Full connection descriptor
export ATP_DSN="(description=(address=(protocol=tcps)(port=1522)(host=adb.region.oraclecloud.com))(connect_data=(service_name=service_name)))"

# Option 2: TNS alias (requires wallet)
export ATP_TNS_ALIAS="sglroutertestatp_high"
export ATP_WALLET_PATH="/path/to/wallet"
```
Provide database credentials and optional pool sizing:
```bash
export ATP_USER="admin"
export ATP_PASSWORD="YourPassword123"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32
```

Router flags map to these values:
- `--oracle-dsn` (env: `ATP_DSN`) or `--oracle-tns-alias` with `--oracle-wallet-path`.
- `--oracle-user` / `--oracle-password` (`ATP_USER` / `ATP_PASSWORD`).
- `--oracle-wallet-path` (`ATP_WALLET_PATH`) when using TNS alias.
- `--oracle-pool-min`, `--oracle-pool-max`, `--oracle-pool-timeout-secs`.

Only one of `--oracle-dsn` or `--oracle-tns-alias` should be supplied.

## Reliability & Flow Control
- **Retries**: Default max retries = 5 with exponential backoff (`--retry-max-retries`, `--retry-initial-backoff-ms`, `--retry-max-backoff-ms`, `--retry-backoff-multiplier`, `--retry-jitter-factor`). Retries trigger on 408/429/500/502/503/504.
- **Circuit Breakers**: Per worker thresholds (`--cb-failure-threshold`, `--cb-success-threshold`, `--cb-timeout-duration-secs`, `--cb-window-duration-secs`). Disable via `--disable-circuit-breaker`.
- **Rate Limiting**: Token bucket driven by `--max-concurrent-requests`. Set `--rate-limit-tokens-per-second` to override refill rate. Configure request queue via `--queue-size` and `--queue-timeout-secs`; queued requests observe FIFO order and respect cancellation.
- **Health Checks**: Runtime probes via `--health-check-interval-secs`, `--health-check-timeout-secs`, failure/success thresholds, and `--health-check-endpoint`.
- **Cache Management**: `/flush_cache` ensures LRU eviction when redeploying PD workers.

## Load Balancing Policies
- `random`: uniform random worker selection.
- `round_robin`: sequential rotation with atomic counters.
- `cache_aware`: maintains a prefix tree of prompts to route repeat traffic and evens load with configurable thresholds (`--cache-threshold`, `--balance-abs-threshold`, `--balance-rel-threshold`, `--eviction-interval`, `--max-tree-size`).
- `power_of_two`: chooses the lighter worker among two random candidates; integrates with `LoadMonitor`.
  Per-model overrides are available in PD mode (`--prefill-policy`, `--decode-policy`) and IGW mode via the worker registry.

## Observability

### Logging
Structured tracing through `tracing` with optional file sink (`--log-dir`) and `--log-level` (`debug`, `info`, `warn`, `error`).

### Prometheus Metrics
Enable with `--prometheus-host`/`--prometheus-port` (defaults to `0.0.0.0:29000`).

**Metric Categories (40+ metrics):**

| Layer | Metric Prefix | Description |
|-------|---------------|-------------|
| HTTP | `smg_http_*` | Request counts, duration, active connections, rate limiting |
| Router | `smg_router_*` | Requests by model/endpoint, latency, errors, upstream responses |
| Inference | `smg_router_ttft/tpot/tokens_*` | Time to first token, time per output token, token counts (gRPC) |
| Worker | `smg_worker_*` | Pool size, active connections, health checks, selection events |
| Circuit Breaker | `smg_worker_cb_*` | State (closed/open/half-open), transitions, outcomes |
| Retry | `smg_worker_retries_*` | Retry attempts, exhausted retries, backoff duration |
| Discovery | `smg_discovery_*` | K8s registrations, sync duration, workers discovered |
| MCP | `smg_mcp_*` | Tool calls, duration, active servers, iterations |
| Database | `smg_db_*` | Operations, duration, connections, items stored |

**Key Metrics:**
- `smg_router_ttft_seconds` - Time to first token histogram (gRPC mode)
- `smg_router_tpot_seconds` - Time per output token histogram (gRPC mode)
- `smg_router_tokens_total` - Total input/output tokens by model
- `smg_router_generation_duration_seconds` - End-to-end generation time
- `smg_worker_cb_state` - Circuit breaker state gauge (0=closed, 1=open, 2=half-open)

**Duration Buckets:**
1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 15s, 30s, 45s, 60s, 90s, 120s, 180s, 240s

### OpenTelemetry Tracing
Enable distributed tracing with OTLP export:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

**Features:**
- OTLP/gRPC exporter (default port 4317)
- W3C Trace Context propagation for HTTP and gRPC
- Batch span processing (500ms delay, 64 span batch size)
- Custom filtering to reduce noise (only exports relevant spans)
- Trace context injection into upstream worker requests

**Configuration:**
- `--enable-trace` - Enable OpenTelemetry tracing
- `--otlp-traces-endpoint <host:port>` - OTLP collector endpoint

### Request ID Propagation
Configure headers for request ID extraction:
```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```
Responses include `x-request-id` header for correlation.

### CORS
Set `--cors-allowed-origins` for browser access.

## Security

### Router and Worker API Keys
- **Router API key (`--api-key`)** protects client access to router endpoints; all protected routes expect `Authorization: Bearer <key>`.
- Workers listed in `--worker-urls` inherit the router API key automatically.
- When adding workers dynamically, provide explicit API keys via payload or query string; they do **not** inherit automatically.

```bash
# Router and initial workers share the same key
python3 -m sglang_router.launch_router \
  --api-key "shared-api-key" \
  --worker-urls http://worker1:8000 http://worker2:8000

# Adding a worker without key while router has one triggers a warning and leaves the worker unprotected
curl -X POST http://localhost:8080/add_worker?url=http://worker3:8000

# Add worker with explicit key
curl -X POST "http://localhost:8080/add_worker?url=http://worker3:8000&api_key=worker3-specific-key"
```

### Security Configurations
1. **No Authentication** (default): Router and workers accept requests without keys—use only in trusted environments.
2. **Router-only Authentication**: Provide `--api-key`; clients must present the key, router accesses workers without credentials.
3. **Worker-only Authentication**: Router open to clients; each worker requires its own key. Supply keys when calling `/workers` or `/add_worker`.
4. **Full Authentication**: Set router API key and provide per-worker keys. Example:
   ```bash
   python3 -m sglang_router.launch_router --api-key "router-key"
   curl -H "Authorization: Bearer router-key" \
     -X POST http://localhost:8080/add_worker?url=http://worker:8000&api_key=worker-key
   ```

### Important Notes
- Initial workers declared via CLI inherit the router key; dynamic workers must supply keys explicitly.
- Router logs a warning when a worker is registered without a key while the router expects authentication.
- When router and workers share the same key, still include the key when invoking dynamic registration APIs.

### TLS (HTTPS) for Gateway Server

Enable TLS to serve the gateway over HTTPS:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --tls-cert-path /path/to/server.crt \
  --tls-key-path /path/to/server.key
```

| Parameter | Description |
|-----------|-------------|
| `--tls-cert-path` | Path to server certificate (PEM format) |
| `--tls-key-path` | Path to server private key (PEM format) |

Both parameters must be provided together. The gateway uses rustls with the ring crypto provider for TLS termination. If TLS is not configured, the gateway falls back to plain HTTP.

### mTLS for Worker Communication

Enable mutual TLS (mTLS) for secure communication with workers in HTTP mode:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

| Parameter | Description |
|-----------|-------------|
| `--client-cert-path` | Path to client certificate for mTLS (PEM format) |
| `--client-key-path` | Path to client private key for mTLS (PEM format) |
| `--ca-cert-path` | Path to CA certificate for verifying worker TLS (PEM format) |

**Key Points:**
- Client certificate and key must be provided together
- Multiple CA certificates can be added with multiple `--ca-cert-path` flags
- Uses rustls backend when TLS is configured
- Single HTTP client is created for all workers (assumes single security domain)
- TCP keepalive (30 seconds) is enabled for long-lived connections

**Full TLS Example (Gateway HTTPS + Worker mTLS):**
```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key"
```

### Control Plane Authentication

The gateway supports role-based access control (RBAC) for control plane APIs (worker management, tokenizer registration, cache operations). Two authentication methods are available:

#### Authentication Methods

| Method | Use Case | Configuration |
|--------|----------|---------------|
| **API Keys** | Service accounts, internal services | `--control-plane-api-keys` |
| **JWT/OIDC** | User authentication via Identity Provider | `--jwt-issuer`, `--jwt-audience` |

Both methods can be used together. Requests are authenticated in order: API key → JWT token.

#### Roles

| Role | Access |
|------|--------|
| `admin` | Full access to all control plane APIs (workers, tokenizers, cache, etc.) |
| `user` | Inference/data plane APIs only (chat completions, embeddings, etc.) |

#### API Key Authentication

Static API keys for service accounts and automation:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --control-plane-api-keys 'svc1:CI Pipeline:admin:secret-key-123' \
                           'svc2:Monitoring:user:readonly-key-456' \
  --control-plane-audit-enabled
```

**Format:** `id:name:role:key`
- `id` - Unique identifier for the key
- `name` - Human-readable description
- `role` - Either `admin` or `user`
- `key` - The secret key (stored as SHA-256 hash internally)

**Usage:**
```bash
curl -H "Authorization: Bearer secret-key-123" \
  http://localhost:30000/workers
```

#### JWT/OIDC Authentication

Authenticate users via an external Identity Provider (Azure AD, Okta, Auth0, Keycloak, etc.):

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --jwt-issuer "https://login.microsoftonline.com/{tenant-id}/v2.0" \
  --jwt-audience "api://my-gateway-client-id" \
  --jwt-jwks-uri "https://login.microsoftonline.com/{tenant-id}/discovery/v2.0/keys" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user' \
  --control-plane-audit-enabled
```

| Parameter | Description |
|-----------|-------------|
| `--jwt-issuer` | OIDC issuer URL. Used to validate the `iss` claim and discover JWKS endpoint via `.well-known/openid-configuration`. |
| `--jwt-audience` | Expected audience (`aud` claim). Typically your application's client ID or API identifier (e.g., `api://client-id`). |
| `--jwt-jwks-uri` | (Optional) Explicit JWKS URI. If omitted, discovered automatically from the issuer's OIDC configuration. |
| `--jwt-role-mapping` | Map IDP group/role names to gateway roles. Format: `idp_role=gateway_role`. |

**How it works:**
1. User authenticates with Identity Provider (OAuth2/OIDC flow)
2. IDP issues a JWT token
3. User sends token to gateway: `Authorization: Bearer <jwt-token>`
4. Gateway validates the JWT:
   - Verifies signature against JWKS
   - Checks `iss` matches `--jwt-issuer`
   - Checks `aud` matches `--jwt-audience`
   - Validates expiration and other standard claims
   - Extracts role from `roles` claim (or `groups` as fallback)
   - Maps IDP role to gateway role via `--jwt-role-mapping`

**Example Azure AD Configuration:**
```bash
# Azure AD issues tokens with:
#   iss: https://login.microsoftonline.com/{tenant}/v2.0
#   aud: api://your-client-id (or the client ID itself)
#   roles: ["Gateway.Admins"] or groups: ["group-id"]

python3 -m sglang_router.launch_router \
  --jwt-issuer "https://login.microsoftonline.com/your-tenant-id/v2.0" \
  --jwt-audience "api://your-client-id" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user'
```

#### Audit Logging

Enable `--control-plane-audit-enabled` to log all control plane operations with:
- Timestamp
- Principal (API key ID or JWT subject)
- Role
- Action performed
- Success/failure status

#### Combined Authentication Example

Use both API keys and JWT for different use cases:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  # API keys for service accounts
  --control-plane-api-keys 'ci:CI/CD Pipeline:admin:ci-secret' \
  # JWT for human users via Azure AD
  --jwt-issuer "https://login.microsoftonline.com/{tenant}/v2.0" \
  --jwt-audience "api://gateway" \
  --jwt-role-mapping 'Platform.Admins=admin' 'Platform.Users=user' \
  # Enable audit logging
  --control-plane-audit-enabled
```

## Development & Testing
```bash
# Build Rust components (debug mode, fast)
cargo build

# Run Rust tests
cargo test

# Fast Python development (rebuilds and installs in debug mode)
cd bindings/python && maturin develop

# Run Python tests
cd ../..  # Back to sgl-model-gateway root
pytest py_test/
```
For production builds, use `maturin build --release --out dist` from the `bindings/python/` directory to create optimized wheels. During development, `maturin develop` rebuilds and installs instantly without creating wheel files. Use `python -m sglang_router.launch_server` to co-launch router and SGLang workers in small clusters for local validation.

### Build Caching

**Local development** uses incremental compilation by default (configured in `.cargo/config.toml`), which is optimal for the edit-compile-test cycle.

**For release builds or CI**, you can optionally use [sccache](https://github.com/mozilla/sccache) to cache compilation artifacts:

```bash
# Install sccache
cargo install sccache

# Option 1: Set environment variable (per-session)
export RUSTC_WRAPPER=sccache
cargo build --release

# Option 2: Add to your global cargo config (~/.cargo/config.toml)
# [build]
# rustc-wrapper = "sccache"
```

> **Note:** sccache and incremental compilation are mutually exclusive—sccache cannot cache incrementally compiled crates. The project defaults to incremental compilation for faster local iteration. Use sccache for clean/release builds where caching across builds matters more.

---

## Release Management

### Creating Gateway Releases

Create releases for the Gateway/Router component with filtered commits:

```bash
# Using make
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0

# Save to file
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 OUTPUT=RELEASE_NOTES.md

# Create draft release (requires gh CLI, DEFAULT behavior)
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

# Publish release immediately (requires gh CLI)
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
```

**Tag Naming**: Use `gateway-*` or `router-*` prefixes to avoid triggering unrelated CI workflows.

### Release Workflow

1. **Create and push tag**:
   ```bash
   git tag -a gateway-v1.0.0 <commit-hash> -m "Gateway release v1.0.0"
   git push origin gateway-v1.0.0
   ```

2. **Generate release notes** (automatically filters gateway-related commits):
   ```bash
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0
   ```

3. **Create GitHub release**:
   ```bash
   # Create draft (DEFAULT - review before publishing)
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

   # Or publish immediately (skip draft)
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
   ```

### Filtered Paths

Release notes only include commits touching:
- `sgl-model-gateway/` - Router codebase
- `python/sglang/srt/grpc/` - gRPC protocol
- `python/sglang/srt/entrypoints/grpc_server.py` - gRPC server

The script automatically extracts author attribution, PR links, and identifies new contributors.

---

SGLang Model Gateway continues to evolve alongside the core SGLang runtime. Contributions should keep CLI flags, documentation, and Python bindings in sync with the Rust implementation.
