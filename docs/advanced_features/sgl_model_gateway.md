# SGLang Model Gateway

SGLang Model Gateway is a high-performance model-routing gateway for large-scale LLM deployments. It centralizes worker lifecycle management, balances traffic across heterogeneous protocols (HTTP, gRPC, OpenAI-compatible), and provides enterprise-ready control over history storage, MCP tooling, and privacy-sensitive workflows. The gateway is deeply optimized for the SGLang serving runtime, but can route to any OpenAI-compatible backend.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Control Plane](#control-plane)
   - [Data Plane](#data-plane)
   - [Storage and Privacy](#storage-and-privacy)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Deployment Modes](#deployment-modes)
   - [Co-launch Router and Workers](#co-launch-router-and-workers)
   - [Separate Launch (HTTP)](#separate-launch-http)
   - [gRPC Launch](#grpc-launch)
   - [Prefill-Decode Disaggregation](#prefill-decode-disaggregation)
   - [OpenAI Backend Proxy](#openai-backend-proxy)
   - [Multi-Model Inference Gateway](#multi-model-inference-gateway)
6. [API Reference](#api-reference)
   - [Inference Endpoints](#inference-endpoints)
   - [Tokenization Endpoints](#tokenization-endpoints)
   - [Parser Endpoints](#parser-endpoints)
   - [Conversation and Response APIs](#conversation-and-response-apis)
   - [Worker Management APIs](#worker-management-apis)
   - [Admin and Health Endpoints](#admin-and-health-endpoints)
7. [Load Balancing Policies](#load-balancing-policies)
8. [Reliability and Flow Control](#reliability-and-flow-control)
   - [Retries](#retries)
   - [Circuit Breaker](#circuit-breaker)
   - [Rate Limiting and Queuing](#rate-limiting-and-queuing)
   - [Health Checks](#health-checks)
9. [Reasoning Parser Integration](#reasoning-parser-integration)
10. [Tool Call Parsing](#tool-call-parsing)
11. [Tokenizer Management](#tokenizer-management)
12. [MCP Integration](#mcp-integration)
13. [Service Discovery (Kubernetes)](#service-discovery-kubernetes)
14. [History and Data Connectors](#history-and-data-connectors)
15. [WASM Middleware](#wasm-middleware)
16. [Language Bindings](#language-bindings)
17. [Security and Authentication](#security-and-authentication)
    - [TLS (HTTPS) for Gateway Server](#tls-https-for-gateway-server)
    - [mTLS for Worker Communication](#mtls-for-worker-communication)
18. [Observability](#observability)
    - [Prometheus Metrics](#prometheus-metrics)
    - [OpenTelemetry Tracing](#opentelemetry-tracing)
    - [Logging](#logging)
19. [Production Recommendations](#production-recommendations)
    - [Security Best Practices](#security-best-practices)
    - [High Availability](#high-availability)
    - [Performance](#performance)
    - [Kubernetes Deployment](#kubernetes-deployment)
    - [Monitoring with PromQL](#monitoring-with-promql)
20. [Configuration Reference](#configuration-reference)
21. [Troubleshooting](#troubleshooting)

---

## Overview

- **Unified control plane** for registering, monitoring, and orchestrating regular, prefill, and decode workers across heterogeneous model fleets.
- **Multi-protocol data plane** that routes traffic across HTTP, PD (prefill/decode), gRPC, and OpenAI-compatible backends with shared reliability primitives.
- **Industry-first gRPC pipeline** with native Rust tokenization, reasoning parsers, and tool-call execution for high-throughput, OpenAI-compatible serving; supports both single-stage and PD topologies.
- **Inference Gateway Mode (`--enable-igw`)** dynamically instantiates multiple router stacks (HTTP regular/PD, gRPC) and applies per-model policies for multi-tenant deployments.
- **Conversation & responses connectors** centralize chat history inside the router so the same context can be reused across models and MCP loops without leaking data to upstream vendors (memory, none, Oracle ATP, PostgreSQL).
- **Enterprise privacy**: agentic multi-turn `/v1/responses`, native MCP client (STDIO/HTTP/SSE/Streamable), and history storage all operate within the router boundary.
- **Reliability core**: retries with jitter, worker-scoped circuit breakers, token-bucket rate limiting with queuing, background health checks, and cache-aware load monitoring.
- **Comprehensive observability**: 40+ Prometheus metrics, OpenTelemetry distributed tracing, structured logging, and request ID propagation.

---

## Architecture

### Control Plane

- **Worker Manager** discovers capabilities (`/get_server_info`, `/get_model_info`), tracks load, and registers/removes workers in the shared registry.
- **Job Queue** serializes add/remove requests and exposes status (`/workers/{worker_id}`) so clients can track onboarding progress.
- **Load Monitor** feeds cache-aware and power-of-two policies with live worker load statistics.
- **Health Checker** continuously probes workers and updates readiness, circuit breaker state, and router metrics.
- **Tokenizer Registry** manages dynamically registered tokenizers with async loading from HuggingFace or local paths.

### Data Plane

- **HTTP routers** (regular & PD) implement `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/classify`, `/v1/tokenize`, `/v1/detokenize`, and associated admin endpoints.
- **gRPC router** streams tokenized requests directly to SRT gRPC workers, running fully in Rust—tokenizer, reasoning parser, and tool parser all reside in-process. Supports both single-stage and PD routing, including embeddings.
- **OpenAI router** proxies OpenAI-compatible endpoints to external vendors (OpenAI, xAI, etc.) while keeping chat history and multi-turn orchestration local.

### Storage and Privacy

- Conversation and response history is stored at the router tier (memory, none, Oracle ATP, or PostgreSQL). The same history can power multiple models or MCP loops without sending data to upstream vendors.
- `/v1/responses` agentic flows, MCP sessions, and conversation APIs share the same storage layer, enabling compliance for regulated workloads.

---

## Installation

### Docker

Pre-built Docker images are available on Docker Hub with multi-architecture support (x86_64 and ARM64):

```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### Prerequisites

- **Rust and Cargo**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  rustc --version
  cargo --version
  ```
- **Python** with `pip` and virtualenv tooling available.

### Rust Binary

```bash
cd sgl-model-gateway
cargo build --release
```

### Python Package

```bash
pip install maturin

# Fast development mode
cd sgl-model-gateway/bindings/python
maturin develop

# Production build
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl
```

---

## Quick Start

### Regular HTTP Routing

```bash
# Rust binary
./target/release/sgl-model-gateway \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware

# Python launcher
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware
```

### gRPC Routing

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

---

## Deployment Modes

### Co-launch Router and Workers

Launch the router and a fleet of SGLang workers in one process:

```bash
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

Comprehensive example with router arguments (prefixed with `--router-`):

```bash
python -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode \
  --log-level debug \
  --router-prometheus-port 10001 \
  --router-tool-call-parser llama \
  --router-model-path meta-llama/Llama-3.1-8B-Instruct \
  --router-policy round_robin \
  --router-log-level debug
```

### Separate Launch (HTTP)

Run workers independently and point the router at their HTTP endpoints:

```bash
# Worker nodes
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001

# Router node
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

### gRPC Launch

Use SRT gRPC workers to unlock the highest throughput and access native reasoning/tool pipelines:

```bash
# Workers expose gRPC endpoints
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --grpc-mode \
  --port 20000

# Router
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

The gRPC router supports both regular HTTP-equivalent serving and PD (prefill/decode) serving. Provide `--tokenizer-path` or `--model-path` (HuggingFace ID or local directory) whenever connection mode resolves to gRPC.

### Prefill-Decode Disaggregation

Split prefill and decode workers for PD-aware caching and balancing:

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --decode http://decode1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

Prefill entries accept an optional bootstrap port. PD mode merges prefill metadata with decode outputs and streams results back to the client.

### OpenAI Backend Proxy

Proxy OpenAI-compatible endpoints while keeping history and MCP sessions local:

```bash
python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory
```

OpenAI backend mode expects exactly one `--worker-urls` entry per router instance.

### Multi-Model Inference Gateway

Enable IGW mode to route multiple models through a single router:

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
```

---

## API Reference

### Inference Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | SGLang generate API |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming/tool calls) |
| `POST` | `/v1/completions` | OpenAI-compatible text completions |
| `POST` | `/v1/embeddings` | Embedding generation (HTTP and gRPC) |
| `POST` | `/v1/rerank`, `/rerank` | Reranking requests |
| `POST` | `/v1/classify` | Text classification |

### Tokenization Endpoints

The gateway provides HTTP endpoints for text tokenization with batch support, designed to mirror the SGLang Python tokenization API.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/tokenize` | Tokenize text to token IDs (single or batch) |
| `POST` | `/v1/detokenize` | Convert token IDs back to text (single or batch) |
| `POST` | `/v1/tokenizers` | Register a new tokenizer (async, returns job status) |
| `GET` | `/v1/tokenizers` | List all registered tokenizers |
| `GET` | `/v1/tokenizers/{id}` | Get tokenizer info by UUID |
| `GET` | `/v1/tokenizers/{id}/status` | Check async tokenizer loading status |
| `DELETE` | `/v1/tokenizers/{id}` | Remove a tokenizer from the registry |

#### Tokenize Request

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

#### Batch Tokenize Request

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": ["Hello", "World", "How are you?"]
}
```

#### Tokenize Response

```json
{
  "tokens": [15339, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

#### Detokenize Request

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [15339, 11, 1917, 0],
  "skip_special_tokens": true
}
```

#### Detokenize Response

```json
{
  "text": "Hello, world!"
}
```

#### Add Tokenizer (Async)

```bash
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Tokenizer registration queued"
}
```

Check status:
```bash
curl http://localhost:30000/v1/tokenizers/550e8400-e29b-41d4-a716-446655440000/status
```

### Parser Endpoints

The gateway provides admin endpoints for parsing reasoning content and function calls from LLM outputs.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/parse/reasoning` | Separate reasoning (`<think>`) from normal text |
| `POST` | `/parse/function_call` | Parse function/tool calls from text |

#### Separate Reasoning Request

```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "parser": "deepseek-r1"
}
```

#### Response

```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step..."
}
```

#### Function Call Parsing

```json
{
  "text": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}",
  "parser": "json"
}
```

### Conversation and Response APIs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/responses` | Create background responses (agentic loops) |
| `GET` | `/v1/responses/{id}` | Retrieve stored response |
| `POST` | `/v1/responses/{id}/cancel` | Cancel background response |
| `DELETE` | `/v1/responses/{id}` | Delete response |
| `GET` | `/v1/responses/{id}/input_items` | List response input items |
| `POST` | `/v1/conversations` | Create conversation |
| `GET` | `/v1/conversations/{id}` | Get conversation |
| `POST` | `/v1/conversations/{id}` | Update conversation |
| `DELETE` | `/v1/conversations/{id}` | Delete conversation |
| `GET` | `/v1/conversations/{id}/items` | List conversation items |
| `POST` | `/v1/conversations/{id}/items` | Add items to conversation |
| `GET` | `/v1/conversations/{id}/items/{item_id}` | Get conversation item |
| `DELETE` | `/v1/conversations/{id}/items/{item_id}` | Delete conversation item |

### Worker Management APIs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workers` | Queue worker registration (returns 202 Accepted) |
| `GET` | `/workers` | List workers with health, load, and policy metadata |
| `GET` | `/workers/{worker_id}` | Inspect specific worker or job queue entry |
| `PUT` | `/workers/{worker_id}` | Queue worker update |
| `DELETE` | `/workers/{worker_id}` | Queue worker removal |

#### Add Worker

```bash
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"grpc://0.0.0.0:31000","worker_type":"regular"}'
```

#### List Workers

```bash
curl http://localhost:30000/workers
```

Response:
```json
{
  "workers": [
    {
      "id": "2f3a0c3e-3a7b-4c3f-8c70-1b7d4c3a6e1f",
      "url": "http://0.0.0.0:31378",
      "model_id": "mistral",
      "priority": 50,
      "cost": 1.0,
      "worker_type": "regular",
      "is_healthy": true,
      "load": 0,
      "connection_mode": "Http"
    }
  ],
  "total": 1,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 1
  }
}
```

### Admin and Health Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/liveness` | Health check (always returns OK) |
| `GET` | `/readiness` | Readiness check (checks healthy worker availability) |
| `GET` | `/health` | Alias for liveness |
| `GET` | `/health_generate` | Health generate test |
| `GET` | `/engine_metrics` | Engine-level metrics from workers |
| `GET` | `/v1/models` | List available models |
| `GET` | `/get_model_info` | Get model information |
| `GET` | `/get_server_info` | Get server information |
| `POST` | `/flush_cache` | Clear all caches |
| `GET` | `/get_loads` | Get all worker loads |
| `POST` | `/wasm` | Upload WASM module |
| `GET` | `/wasm` | List WASM modules |
| `DELETE` | `/wasm/{module_uuid}` | Remove WASM module |

---

## Load Balancing Policies

| Policy | Description | Usage |
|--------|-------------|-------|
| `random` | Uniform random selection | `--policy random` |
| `round_robin` | Cycles through workers in order | `--policy round_robin` |
| `power_of_two` | Samples two workers and picks the lighter one | `--policy power_of_two` |
| `cache_aware` | Combines cache locality with load balancing (default) | `--policy cache_aware` |
| `bucket` | Divides workers into load buckets with dynamic boundaries | `--policy bucket` |

### Cache-Aware Policy Tuning

```bash
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.5 \
--eviction-interval-secs 120 \
--max-tree-size 67108864
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cache-threshold` | 0.3 | Minimum prefix match ratio for cache hit |
| `--balance-abs-threshold` | 64 | Absolute load difference before rebalancing |
| `--balance-rel-threshold` | 1.5 | Relative load ratio before rebalancing |
| `--eviction-interval-secs` | 120 | Cache eviction cadence in seconds |
| `--max-tree-size` | 67108864 | Maximum nodes in cache tree |

---

## Reliability and Flow Control

### Retries

Configure exponential backoff retries:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --retry-max-retries 5 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 30000 \
  --retry-backoff-multiplier 1.5 \
  --retry-jitter-factor 0.2
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--retry-max-retries` | 5 | Maximum retry attempts |
| `--retry-initial-backoff-ms` | 50 | Initial backoff duration (ms) |
| `--retry-max-backoff-ms` | 5000 | Maximum backoff duration (ms) |
| `--retry-backoff-multiplier` | 2.0 | Exponential backoff multiplier |
| `--retry-jitter-factor` | 0.1 | Random jitter factor (0.0-1.0) |
| `--disable-retries` | false | Disable retries entirely |

**Retryable Status Codes:** 408, 429, 500, 502, 503, 504

### Circuit Breaker

Per-worker circuit breakers prevent cascading failures:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cb-failure-threshold` | 5 | Consecutive failures to open circuit |
| `--cb-success-threshold` | 2 | Successes to close from half-open |
| `--cb-timeout-duration-secs` | 30 | Time before half-open attempt |
| `--cb-window-duration-secs` | 60 | Failure counting window |
| `--disable-circuit-breaker` | false | Disable circuit breaker |

**Circuit Breaker States:**
- **Closed**: Normal operation, requests allowed
- **Open**: Failing, requests rejected immediately
- **Half-Open**: Testing recovery, limited requests allowed

### Rate Limiting and Queuing

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --max-concurrent-requests 256 \
  --rate-limit-tokens-per-second 512 \
  --queue-size 128 \
  --queue-timeout-secs 30
```

Requests beyond the concurrency limit wait in a FIFO queue. Returns:
- `429 Too Many Requests` when queue is full
- `408 Request Timeout` when queue timeout expires

### Health Checks

```bash
--health-check-interval-secs 30 \
--health-check-timeout-secs 10 \
--health-success-threshold 2 \
--health-failure-threshold 3 \
--health-check-endpoint /health
```

---

## Reasoning Parser Integration

The gateway includes built-in reasoning parsers for models that use Chain-of-Thought (CoT) reasoning with explicit thinking blocks.

### Supported Parsers

| Parser ID | Model Family | Think Tokens |
|-----------|--------------|--------------|
| `deepseek-r1` | DeepSeek-R1 | `<think>...</think>` (initial reasoning) |
| `qwen3` | Qwen-3 | `<think>...</think>` |
| `qwen3-thinking` | Qwen-3 Thinking | `<think>...</think>` (initial reasoning) |
| `kimi` | Kimi K2 | Unicode think tokens |
| `glm45` | GLM-4.5/4.6/4.7 | `<think>...</think>` |
| `step3` | Step-3 | `<think>...</think>` |
| `minimax` | MiniMax | `<think>...</think>` |

### Usage

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek-r1
```

The gRPC router automatically:
1. Detects reasoning blocks in streaming output
2. Separates reasoning content from normal text
3. Applies incremental streaming parsing with buffer management
4. Handles partial token detection for correct streaming behavior

---

## Tool Call Parsing

The gateway supports parsing function/tool calls from LLM outputs in multiple formats.

### Supported Formats

| Parser | Format | Description |
|--------|--------|-------------|
| `json` | JSON | Standard JSON tool calls |
| `python` | Pythonic | Python function call syntax |
| `xml` | XML | XML-formatted tool calls |

### Usage

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tool-call-parser json
```

---

## Tokenizer Management

### Tokenizer Sources

The gateway supports multiple tokenizer backends:
- **HuggingFace**: Load from HuggingFace Hub by model ID
- **Local**: Load from local `tokenizer.json` or directory
- **Tiktoken**: Auto-detect OpenAI GPT models (gpt-4, davinci, etc.)

### Configuration

```bash
# HuggingFace model
--model-path meta-llama/Llama-3.1-8B-Instruct

# Local tokenizer
--tokenizer-path /path/to/tokenizer.json

# With chat template override
--chat-template /path/to/template.jinja
```

### Tokenizer Caching

Two-level caching for optimal performance:

| Cache | Type | Description |
|-------|------|-------------|
| L0 | Exact match | Whole-string caching for repeated prompts |
| L1 | Prefix match | Prefix boundary matching for incremental prompts |

```bash
--enable-l0-cache \
--l0-max-entries 10000 \
--enable-l1-cache \
--l1-max-memory 52428800  # 50MB
```

---

## MCP Integration

The gateway provides native Model Context Protocol (MCP) client integration for tool execution.

### Supported Transports

| Transport | Description |
|-----------|-------------|
| STDIO | Local process execution |
| SSE | Server-Sent Events (HTTP) |
| Streamable | Bidirectional streaming |

### Configuration

```bash
python -m sglang_router.launch_router \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000
```

### MCP Configuration File

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
  idle_timeout: 300

proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8443"
  no_proxy: "localhost,127.0.0.1,*.internal"

inventory:
  enable_refresh: true
  tool_ttl: 300
  refresh_interval: 300
```

---

## Service Discovery (Kubernetes)

Enable automatic worker discovery via Kubernetes pod selectors:

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

### PD Mode Discovery

```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```

Prefill pods can expose bootstrap ports via the `sglang.ai/bootstrap-port` annotation. RBAC must allow `get`, `list`, and `watch` on pods.

---

## History and Data Connectors

| Backend | Description | Usage |
|---------|-------------|-------|
| `memory` | In-memory storage (default) | `--history-backend memory` |
| `none` | No persistence | `--history-backend none` |
| `oracle` | Oracle Autonomous Database | `--history-backend oracle` |
| `postgres` | PostgreSQL Database | `--history-backend postgres` |

### Oracle Configuration

```bash
# Connection descriptor
export ATP_DSN="(description=(address=(protocol=tcps)(port=1522)(host=adb.region.oraclecloud.com))(connect_data=(service_name=service_name)))"

# Or TNS alias (requires wallet)
export ATP_TNS_ALIAS="sglroutertestatp_high"
export ATP_WALLET_PATH="/path/to/wallet"

# Credentials
export ATP_USER="admin"
export ATP_PASSWORD="secret"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle
```

### PostgreSQL Configuration

```bash
export POSTGRES_DB_URL="postgres://user:password@host:5432/dbname"

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend postgres
```

---

## WASM Middleware

The gateway supports WebAssembly (WASM) middleware modules for custom request/response processing. This enables organization-specific logic for authentication, rate limiting, billing, logging, and more—without modifying or recompiling the gateway.

### Overview

WASM middleware runs in a sandboxed environment with memory isolation, no network/filesystem access, and configurable resource limits.

| Attach Point | When Executed | Use Cases |
|--------------|---------------|-----------|
| `OnRequest` | Before forwarding to workers | Auth, rate limiting, request modification |
| `OnResponse` | After receiving worker response | Logging, response modification, error handling |

| Action | Description |
|--------|-------------|
| `Continue` | Proceed without modification |
| `Reject(status)` | Reject request with HTTP status code |
| `Modify(...)` | Modify headers, body, or status |

### Examples

Complete working examples are available in `examples/wasm/`:

| Example | Description |
|---------|-------------|
| `auth/` | API key authentication for protected routes |
| `rate_limit/` | Per-client rate limiting (requests/minute) |
| `logging/` | Request tracking headers and response modification |

The interface definition is located at `src/wasm/interface`.

### Building Modules

```bash
# Prerequisites
rustup target add wasm32-wasip2
cargo install wasm-tools

# Build
cargo build --target wasm32-wasip2 --release

# Convert to component format
wasm-tools component new \
  target/wasm32-wasip2/release/my_middleware.wasm \
  -o my_middleware.component.wasm
```

### Deploying Modules

```bash
# Enable WASM support
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-wasm

# Upload module
curl -X POST http://localhost:30000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "auth-middleware",
      "file_path": "/absolute/path/to/auth.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'

# List modules
curl http://localhost:30000/wasm

# Remove module
curl -X DELETE http://localhost:30000/wasm/{module_uuid}
```

### Runtime Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_memory_pages` | 1024 (64MB) | Maximum WASM memory |
| `max_execution_time_ms` | 1000 | Execution timeout |
| `max_stack_size` | 1MB | Stack size limit |
| `module_cache_size` | 10 | Cached modules per worker |

**Note:** Rate limiting state is per-worker thread and not shared across gateway replicas. For production, consider implementing rate limiting at a shared layer (e.g., Redis)

---

## Language Bindings

SGLang Model Gateway provides official language bindings for Python and Go, enabling integration with different technology stacks and organizational requirements.

### Python Bindings

The Python bindings provide a PyO3-based wrapper around the Rust gateway library. This is a straightforward binding that calls the gateway server startup from Python.

#### Installation

```bash
# From PyPI
pip install sglang-router

# Development build
cd sgl-model-gateway/bindings/python
pip install maturin && maturin develop --features vendored-openssl
```

#### Usage

The Python bindings are used throughout this documentation. See the [Quick Start](#quick-start) and [Deployment Modes](#deployment-modes) sections for detailed examples.

Key components:
- `RouterArgs` dataclass with 50+ configuration options
- `Router.from_args()` for programmatic startup
- CLI commands: `smg launch`, `smg server`, `python -m sglang_router.launch_router`

### Go Bindings

The Go bindings provide a high-performance gRPC client library for organizations with Go-based infrastructure. This is ideal for:

- Integration with internal Go services and tooling
- High-performance client applications
- Building custom OpenAI-compatible proxy servers

#### Architecture

```
┌─────────────────────────────────────────┐
│         High-Level Go API               │
│   (client.go - OpenAI-style interface)  │
├─────────────────────────────────────────┤
│         gRPC Layer                      │
├─────────────────────────────────────────┤
│         Rust FFI Layer                  │
│   (Tokenization, Parsing, Conversion)   │
└─────────────────────────────────────────┘
```

**Key Features:**
- Native Rust tokenization via FFI (thread-safe, lock-free)
- Full streaming support with context cancellation
- Configurable channel buffer sizes for high concurrency
- Built-in tool call parsing and chat template application

#### Installation

```bash
# Build the FFI library first
cd sgl-model-gateway/bindings/golang
make build && make lib

# Then use in your Go project
go get github.com/sgl-project/sgl-go-sdk
```

**Requirements:** Go 1.24+, Rust toolchain

#### Examples

Complete working examples are available in `bindings/golang/examples/`:

| Example | Description |
|---------|-------------|
| `simple/` | Non-streaming chat completion |
| `streaming/` | Streaming chat completion with SSE |
| `oai_server/` | Full OpenAI-compatible HTTP server |

```bash
# Run examples
cd sgl-model-gateway/bindings/golang/examples/simple && ./run.sh
cd sgl-model-gateway/bindings/golang/examples/streaming && ./run.sh
cd sgl-model-gateway/bindings/golang/examples/oai_server && ./run.sh
```

#### Testing

```bash
cd sgl-model-gateway/bindings/golang

# Unit tests
go test -v ./...

# Integration tests (requires running SGLang server)
export SGL_GRPC_ENDPOINT=grpc://localhost:20000
export SGL_TOKENIZER_PATH=/path/to/tokenizer
go test -tags=integration -v ./...
```

### Comparison

| Feature | Python | Go |
|---------|--------|-----|
| **Primary Use** | Gateway server launcher | gRPC client library |
| **CLI Support** | Full CLI (smg, sglang-router) | Library only |
| **K8s Discovery** | Native support | N/A (client library) |
| **PD Mode** | Built-in | N/A (client library) |

**When to Use Python:** Launching and managing the gateway server, service discovery, PD disaggregation.

**When to Use Go:** Building custom client applications, integration with Go microservices, OpenAI-compatible proxy servers

---

## Security and Authentication

### Router API Key

```bash
python -m sglang_router.launch_router \
  --api-key "your-router-api-key" \
  --worker-urls http://worker1:8000
```

Clients must supply `Authorization: Bearer <key>` for protected endpoints.

### Worker API Keys

```bash
# Add worker with explicit key
curl -H "Authorization: Bearer router-key" \
  -X POST http://localhost:8080/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://worker:8000","api_key":"worker-key"}'
```

### Security Configurations

1. **No Authentication** (default): Use only in trusted environments
2. **Router-only Authentication**: Clients authenticate to router
3. **Worker-only Authentication**: Router open, workers require keys
4. **Full Authentication**: Both router and workers protected

### TLS (HTTPS) for Gateway Server

Enable TLS to serve the gateway over HTTPS:

```bash
python -m sglang_router.launch_router \
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
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

| Parameter | Description |
|-----------|-------------|
| `--client-cert-path` | Path to client certificate for mTLS (PEM format) |
| `--client-key-path` | Path to client private key for mTLS (PEM format) |
| `--ca-cert-path` | Path to CA certificate for verifying worker TLS (PEM format, repeatable) |

**Key Points:**
- Client certificate and key must be provided together
- Multiple CA certificates can be added with multiple `--ca-cert-path` flags
- Uses rustls backend when TLS is configured
- Single HTTP client is created for all workers (assumes single security domain)
- TCP keepalive (30 seconds) is enabled for long-lived connections

### Full TLS Configuration Example

Gateway HTTPS + Worker mTLS + API Key authentication:

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key" \
  --policy cache_aware
```

---

## Observability

### Prometheus Metrics

Enable with `--prometheus-host`/`--prometheus-port` (defaults to `0.0.0.0:29000`).

#### Metric Categories (40+ metrics)

| Layer | Prefix | Metrics |
|-------|--------|---------|
| HTTP | `smg_http_*` | `requests_total`, `request_duration_seconds`, `responses_total`, `connections_active`, `rate_limit_total` |
| Router | `smg_router_*` | `requests_total`, `request_duration_seconds`, `request_errors_total`, `stage_duration_seconds`, `upstream_responses_total` |
| Inference | `smg_router_*` | `ttft_seconds`, `tpot_seconds`, `tokens_total`, `generation_duration_seconds` |
| Worker | `smg_worker_*` | `pool_size`, `connections_active`, `requests_active`, `health_checks_total`, `selection_total`, `errors_total` |
| Circuit Breaker | `smg_worker_cb_*` | `state`, `transitions_total`, `outcomes_total`, `consecutive_failures`, `consecutive_successes` |
| Retry | `smg_worker_*` | `retries_total`, `retries_exhausted_total`, `retry_backoff_seconds` |
| Discovery | `smg_discovery_*` | `registrations_total`, `deregistrations_total`, `sync_duration_seconds`, `workers_discovered` |
| MCP | `smg_mcp_*` | `tool_calls_total`, `tool_duration_seconds`, `servers_active`, `tool_iterations_total` |
| Database | `smg_db_*` | `operations_total`, `operation_duration_seconds`, `connections_active`, `items_stored` |

#### Key Inference Metrics (gRPC mode)

| Metric | Type | Description |
|--------|------|-------------|
| `smg_router_ttft_seconds` | Histogram | Time to first token |
| `smg_router_tpot_seconds` | Histogram | Time per output token |
| `smg_router_tokens_total` | Counter | Total tokens (input/output) |
| `smg_router_generation_duration_seconds` | Histogram | End-to-end generation time |

#### Duration Buckets

1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 15s, 30s, 45s, 60s, 90s, 120s, 180s, 240s

### OpenTelemetry Tracing

Enable distributed tracing with OTLP export:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

#### Features

- OTLP/gRPC exporter (default port 4317)
- W3C Trace Context propagation for HTTP and gRPC
- Batch span processing (500ms delay, 64 span batch size)
- Custom filtering to reduce noise
- Trace context injection into upstream worker requests
- Service name: `sgl-router`

### Logging

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs
```

Structured tracing with optional file sink. Log levels: `debug`, `info`, `warn`, `error`.

### Request ID Propagation

```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```

Responses include `x-request-id` header for correlation.

---

## Production Recommendations

This section provides guidance for deploying SGLang Model Gateway in production environments.

### Security Best Practices

**Always enable TLS in production:**

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "${ROUTER_API_KEY}"
```

**Security Checklist:**
- Enable TLS for gateway HTTPS termination
- Enable mTLS for worker communication when workers are on untrusted networks
- Set `--api-key` to protect router endpoints
- Use Kubernetes Secrets or a secrets manager for credentials
- Rotate certificates and API keys periodically
- Restrict network access with firewalls or network policies

### High Availability

**Scaling Strategy:**

The gateway supports running multiple replicas behind a load balancer for high availability. However, there are important considerations:

| Component | Shared Across Replicas | Impact |
|-----------|----------------------|--------|
| Worker Registry | No (independent) | Each replica discovers workers independently |
| Radix Cache Tree | No (independent) | Cache hits may decrease by 10-20% |
| Circuit Breaker State | No (independent) | Each replica tracks failures independently |
| Rate Limiting | No (independent) | Limits apply per-replica, not globally |

**Recommendations:**

1. **Prefer horizontal scaling over vertical scaling**: Deploy multiple smaller gateway replicas rather than one large instance with excessive CPU and memory. This provides:
   - Better fault tolerance (single replica failure doesn't take down the gateway)
   - More predictable resource usage
   - Easier capacity planning

2. **Use Kubernetes Service Discovery**: Let the gateway automatically discover and manage workers:
   ```bash
   python -m sglang_router.launch_router \
     --service-discovery \
     --selector app=sglang-worker \
     --service-discovery-namespace production
   ```

3. **Accept cache efficiency trade-off**: With multiple replicas, the cache-aware routing policy's radix tree is not synchronized across replicas. This means:
   - Each replica builds its own cache tree
   - Requests from the same user may hit different replicas
   - Expected cache hit rate reduction: **10-20%**
   - This is often acceptable given the HA benefits

4. **Configure session affinity (optional)**: If cache efficiency is critical, configure your load balancer for session affinity based on a consistent hash of the request (e.g., user ID or API key).

**Example HA Architecture:**
```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (L4/L7)       │
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Gateway  │  │  Gateway  │  │  Gateway  │
        │ Replica 1 │  │ Replica 2 │  │ Replica 3 │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Worker   │  │  Worker   │  │  Worker   │
        │  Pod 1    │  │  Pod 2    │  │  Pod N    │
        └───────────┘  └───────────┘  └───────────┘
```

### Performance

**Use gRPC mode for high throughput:**

gRPC mode provides the highest performance for SGLang workers:

```bash
# Start workers in gRPC mode
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --grpc-mode \
  --port 20000

# Configure gateway for gRPC
python -m sglang_router.launch_router \
  --worker-urls grpc://worker1:20000 grpc://worker2:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --policy cache_aware
```

**Performance Benefits of gRPC:**
- Native Rust tokenization (no Python overhead)
- Streaming with lower latency
- Built-in reasoning parser execution
- Tool call parsing in the gateway
- Reduced serialization overhead

**Tuning Recommendations:**

| Parameter | Recommendation | Reason |
|-----------|---------------|--------|
| `--policy` | `cache_aware` | Best for repeated prompts, ~30% latency reduction |
| `--max-concurrent-requests` | 2-4x worker count | Prevent overload while maximizing throughput |
| `--queue-size` | 2x max-concurrent | Buffer for burst traffic |
| `--request-timeout-secs` | Based on max generation length | Prevent stuck requests |

### Kubernetes Deployment

**Pod Labeling for Service Discovery:**

For the gateway to discover workers automatically, label your worker pods consistently:

```yaml
# Worker Deployment (Regular Mode)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-worker
  namespace: production
spec:
  replicas: 4
  selector:
    matchLabels:
      app: sglang-worker
      component: inference
  template:
    metadata:
      labels:
        app: sglang-worker
        component: inference
        model: llama-3-8b
    spec:
      containers:
      - name: worker
        image: lmsysorg/sglang:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 20000
          name: grpc
```

**Gateway configuration for discovery:**
```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker component=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

**PD (Prefill/Decode) Mode Labeling:**

```yaml
# Prefill Worker
metadata:
  labels:
    app: sglang-worker
    component: prefill
  annotations:
    sglang.ai/bootstrap-port: "9001"

# Decode Worker
metadata:
  labels:
    app: sglang-worker
    component: decode
```

**Gateway configuration for PD discovery:**
```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --pd-disaggregation \
  --prefill-selector app=sglang-worker component=prefill \
  --decode-selector app=sglang-worker component=decode \
  --service-discovery-namespace production
```

**RBAC Requirements:**

The gateway needs permissions to watch pods:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-gateway
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-gateway
  namespace: production
subjects:
- kind: ServiceAccount
  name: sglang-gateway
  namespace: production
roleRef:
  kind: Role
  name: sglang-gateway
  apiGroup: rbac.authorization.k8s.io
```

### Monitoring with PromQL

Configure Prometheus to scrape the gateway metrics endpoint (default: `:29000/metrics`).

**Essential Dashboards:**

**1. Request Rate and Latency:**
```promql
# Request rate by endpoint
sum(rate(smg_http_requests_total[5m])) by (path, method)

# P50 latency
histogram_quantile(0.50, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# P99 latency
histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(smg_http_responses_total{status=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))
```

**2. Worker Health:**
```promql
# Healthy workers
sum(smg_worker_pool_size)

# Active connections per worker
smg_worker_connections_active

# Worker health check failures
sum(rate(smg_worker_health_checks_total{result="failure"}[5m])) by (worker_id)
```

**3. Circuit Breaker Status:**
```promql
# Circuit breaker states (0=closed, 1=open, 2=half-open)
smg_worker_cb_state

# Circuit breaker transitions
sum(rate(smg_worker_cb_transitions_total[5m])) by (worker_id, from_state, to_state)

# Workers with open circuits
count(smg_worker_cb_state == 1)
```

**4. Inference Performance (gRPC mode):**
```promql
# Time to first token (P50)
histogram_quantile(0.50, sum(rate(smg_router_ttft_seconds_bucket[5m])) by (le, model))

# Time per output token (P99)
histogram_quantile(0.99, sum(rate(smg_router_tpot_seconds_bucket[5m])) by (le, model))

# Token throughput
sum(rate(smg_router_tokens_total[5m])) by (model, direction)

# Generation duration P95
histogram_quantile(0.95, sum(rate(smg_router_generation_duration_seconds_bucket[5m])) by (le))
```

**5. Rate Limiting and Queuing:**
```promql
# Rate limit rejections
sum(rate(smg_http_rate_limit_total{decision="rejected"}[5m]))

# Queue depth (if using concurrency limiting)
smg_worker_requests_active

# Retry attempts
sum(rate(smg_worker_retries_total[5m])) by (worker_id)

# Exhausted retries (failures after all retries)
sum(rate(smg_worker_retries_exhausted_total[5m]))
```

**6. MCP Tool Execution:**
```promql
# Tool call rate
sum(rate(smg_mcp_tool_calls_total[5m])) by (server, tool)

# Tool latency P95
histogram_quantile(0.95, sum(rate(smg_mcp_tool_duration_seconds_bucket[5m])) by (le, tool))

# Active MCP server connections
smg_mcp_servers_active
```

**Alerting Rules Example:**

```yaml
groups:
- name: sglang-gateway
  rules:
  - alert: HighErrorRate
    expr: |
      sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
      / sum(rate(smg_http_responses_total[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate on SGLang Gateway"

  - alert: CircuitBreakerOpen
    expr: count(smg_worker_cb_state == 1) > 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Worker circuit breaker is open"

  - alert: HighLatency
    expr: |
      histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le)) > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 latency exceeds 30 seconds"

  - alert: NoHealthyWorkers
    expr: sum(smg_worker_pool_size) == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "No healthy workers available"
```

---

## Configuration Reference

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--host` | str | 127.0.0.1 | Router host |
| `--port` | int | 30000 | Router port |
| `--worker-urls` | list | [] | Worker URLs (HTTP or gRPC) |
| `--policy` | str | cache_aware | Routing policy |
| `--max-concurrent-requests` | int | -1 | Concurrency limit (-1 disables) |
| `--request-timeout-secs` | int | 600 | Request timeout |
| `--max-payload-size` | int | 256MB | Maximum request payload |

### Prefill/Decode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pd-disaggregation` | flag | false | Enable PD mode |
| `--prefill` | list | [] | Prefill URLs + optional bootstrap ports |
| `--decode` | list | [] | Decode URLs |
| `--prefill-policy` | str | None | Override policy for prefill nodes |
| `--decode-policy` | str | None | Override policy for decode nodes |
| `--worker-startup-timeout-secs` | int | 600 | Worker init timeout |

### Kubernetes Discovery

| Parameter | Type | Description |
|-----------|------|-------------|
| `--service-discovery` | flag | Enable discovery |
| `--selector` | list | Label selectors (key=value) |
| `--prefill-selector` / `--decode-selector` | list | PD mode selectors |
| `--service-discovery-namespace` | str | Namespace to watch |
| `--service-discovery-port` | int | Worker port (default 80) |
| `--bootstrap-port-annotation` | str | Annotation for bootstrap ports |

### TLS Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `--tls-cert-path` | str | Server certificate for gateway HTTPS (PEM) |
| `--tls-key-path` | str | Server private key for gateway HTTPS (PEM) |
| `--client-cert-path` | str | Client certificate for worker mTLS (PEM) |
| `--client-key-path` | str | Client private key for worker mTLS (PEM) |
| `--ca-cert-path` | str | CA certificate for verifying workers (PEM, repeatable) |

---

## Troubleshooting

### Workers Never Ready

Increase `--worker-startup-timeout-secs` or ensure health probes respond before router startup.

### Load Imbalance / Hot Workers

Inspect `smg_router_requests_total` by worker and tune cache-aware thresholds (`--balance-*`, `--cache-threshold`).

### Circuit Breaker Flapping

Increase `--cb-failure-threshold` or extend the timeout/window durations. Consider temporarily disabling retries.

### Queue Overflow (429)

Increase `--queue-size` or reduce client concurrency. Ensure `--max-concurrent-requests` matches downstream capacity.

### Memory Growth

Reduce `--max-tree-size` or lower `--eviction-interval-secs` for more aggressive cache pruning.

### Debugging

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs
```

### gRPC Connection Issues

Ensure workers are started with `--grpc-mode` and verify `--model-path` or `--tokenizer-path` is provided to the router.

### Tokenizer Loading Failures

Check HuggingFace Hub credentials (`HF_TOKEN` environment variable) for private models. Verify local paths are accessible.

---

SGLang Model Gateway continues to evolve alongside the SGLang runtime. Keep CLI flags, integrations, and documentation aligned when adopting new features or contributing improvements.
