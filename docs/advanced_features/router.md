# SGLang Model Gateway (formerly SGLang Router)

SGLang Model Gateway is a high-performance model-routing gateway for large-scale LLM deployments. It centralizes worker lifecycle management, balances traffic across heterogeneous protocols (HTTP, gRPC, OpenAI-compatible), and provides enterprise-ready control over history storage, MCP tooling, and privacy-sensitive workflows. The router is deeply optimized for the SGLang serving runtime, but can route to any OpenAI-compatible backend.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Control Plane](#control-plane)
   - [Data Plane](#data-plane)
   - [Storage & Privacy](#storage--privacy)
3. [Deployment Modes](#deployment-modes)
   - [Co-launch Router + Workers](#co-launch-router--workers)
   - [Separate Launch (HTTP)](#separate-launch-http)
   - [gRPC Launch](#grpc-launch)
   - [Prefill/Decode Disaggregation](#prefilldecode-disaggregation)
   - [OpenAI Backend Proxy](#openai-backend-proxy)
4. [Worker Lifecycle & Dynamic Scaling](#worker-lifecycle--dynamic-scaling)
5. [Reliability & Flow Control](#reliability--flow-control)
6. [Load Balancing Policies](#load-balancing-policies)
7. [Service Discovery (Kubernetes)](#service-discovery-kubernetes)
8. [Security & Authentication](#security--authentication)
9. [History & Data Connectors](#history--data-connectors)
10. [MCP & Advanced Tooling](#mcp--advanced-tooling)
11. [API Surface](#api-surface)
12. [Configuration Reference](#configuration-reference)
13. [Observability](#observability)
14. [Troubleshooting](#troubleshooting)

---

## Overview
- **Unified control plane** for registering, monitoring, and orchestrating regular, prefill, and decode workers across heterogeneous model fleets.
- **Multi-protocol data plane** that routes traffic across HTTP, PD (prefill/decode), gRPC, and OpenAI-compatible backends with shared reliability primitives.
- **Industry-first gRPC pipeline** with native Rust tokenization, reasoning parsers, and tool-call execution for high-throughput, OpenAI-compatible serving; supports both single-stage and PD topologies.
- **Inference Gateway Mode (`--enable-igw`)** dynamically instantiates multiple router stacks (HTTP regular/PD, gRPC) and applies per-model policies for multi-tenant deployments.
- **Conversation & responses connectors** centralize chat history inside the router so the same context can be reused across models and MCP loops without leaking data to upstream vendors (memory, none, Oracle ATP).
- **Enterprise privacy**: agentic multi-turn `/v1/responses`, native MCP client (STDIO/HTTP/SSE/Streamable), and history storage all operate within the router boundary.
- **Reliability core**: retries with jitter, worker-scoped circuit breakers, token-bucket rate limiting with queuing, background health checks, and cache-aware load monitoring.
- **Observability**: Prometheus metrics, structured tracing, request ID propagation, and detailed job queue stats.

---

## Architecture

### Control Plane
- **Worker Manager** discovers capabilities (`/get_server_info`, `/get_model_info`), tracks load, and registers/removes workers in the shared registry.
- **Job Queue** serializes add/remove requests and exposes status (`/workers/{url}`) so clients can track onboarding progress.
- **Load Monitor** feeds cache-aware and power-of-two policies with live worker load statistics.
- **Health Checker** continuously probes workers and updates readiness, circuit breaker state, and router metrics.

### Data Plane
- **HTTP routers** (regular & PD) implement `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, and associated admin endpoints.
- **gRPC router** streams tokenized requests directly to SRT gRPC workers, running fully in Rust—tokenizer, reasoning parser, and tool parser all reside in-process. Supports both single-stage and PD routing.
- **OpenAI router** proxies OpenAI-compatible endpoints to external vendors (OpenAI, xAI, etc.) while keeping chat history and multi-turn orchestration local.

### Storage & Privacy
- Conversation and response history is stored at the router tier (memory, none, or Oracle ATP). The same history can power multiple models or MCP loops without sending data to upstream vendors.
- `/v1/responses` agentic flows, MCP sessions, and conversation APIs share the same storage layer, enabling compliance for regulated workloads.

---

## Deployment Modes

### Co-launch Router + Workers
Launch the router and a fleet of SGLang workers in one process (ideal for single-node or quick starts). The CLI accepts two namespaces of arguments:
- **Worker arguments** (no prefix) configure the SGLang runtime (`--model`, `--tp-size`, `--dp-size`, `--grpc-mode`, etc.).
- **Router arguments** are prefixed with `--router-` and map directly to `launch_router` flags (`--router-policy`, `--router-model-path`, `--router-log-level`, ...).

```bash
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

Comprehensive example:
```bash
python3 -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode \
  --log-level debug \
  --router-prometheus-port 10001 \
  --router-tool-call-parser llama \
  --router-health-success-threshold 2 \
  --router-health-check-timeout-secs 6000 \
  --router-health-check-interval-secs 60 \
  --router-model-path meta-llama/Llama-3.1-8B-Instruct \
  --router-policy round_robin \
  --router-log-level debug
```

### Separate Launch (HTTP)
Run workers independently and point the router at their HTTP endpoints.

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
Use SRT gRPC workers to unlock the highest throughput and access native reasoning/tool pipelines.

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

> gRPC router supports both single-stage and PD serving. Provide `--tokenizer-path` or `--model-path` (HF repo or local directory) plus optional `--chat-template`.

### Prefill/Decode Disaggregation
Split prefill and decode workers for PD-aware caching and balancing. Specifying `--policy A` is equivalent to `--prefill-policy A --decode-policy A`.

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --decode http://decode1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

### OpenAI Backend Proxy
Proxy OpenAI-compatible endpoints (OpenAI, xAI, etc.) while keeping history and MCP sessions local.

```bash
python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory
```

> OpenAI backend mode expects exactly one `--worker-urls` entry per router instance.

---

## Worker Lifecycle & Dynamic Scaling

Add or remove workers at runtime using the REST APIs. Jobs are queued and tracked for eventual consistency.

```bash
# Add a worker (HTTP or gRPC)
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"grpc://0.0.0.0:31000","worker_type":"regular"}'

# Inspect registry
curl http://localhost:30000/workers

# Remove a worker
curl -X DELETE http://localhost:30000/workers/grpc%3A%2F%2F0.0.0.0%3A31000
```

Legacy endpoints (`/add_worker`, `/remove_worker`, `/list_workers`) remain available but will be deprecated. `/workers/{url}` returns both registry data and queued job status. The worker url in the removal request should be escaped.

---

## Reliability & Flow Control

### Retries
```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --retry-max-retries 5 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 30000 \
  --retry-backoff-multiplier 1.5 \
  --retry-jitter-factor 0.2
```

### Circuit Breaker
```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

### Rate Limiting & Queuing
```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --max-concurrent-requests 256 \
  --rate-limit-tokens-per-second 512 \
  --queue-size 128 \
  --queue-timeout-secs 30
```

Requests beyond the concurrency limit wait in a FIFO queue (up to `queue-size`). A `429` is returned when the queue is full; `408` is returned when `queue-timeout-secs` expires.

---

## Load Balancing Policies

| Policy             | Description                                                                                      | Usage                         |
|--------------------|--------------------------------------------------------------------------------------------------|-------------------------------|
| `random`           | Uniform random selection.                                                                        | `--policy random`             |
| `round_robin`      | Cycles through workers in order.                                                                 | `--policy round_robin`        |
| `power_of_two`     | Samples two workers and picks the lighter one (requires Load Monitor).                           | `--policy power_of_two`       |
| `cache_aware`      | Default policy; combines cache locality with load balancing, falling back to shortest queue.     | `--policy cache_aware` + tuning flags |

Key tuning flags:
```bash
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.5 \
--eviction-interval-secs 120 \
--max-tree-size 67108864
```

---

## Service Discovery (Kubernetes)

Enable automatic worker discovery via Kubernetes pod selectors.

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

PD deployments can specify `--prefill-selector` and `--decode-selector` plus the `sglang.ai/bootstrap-port` annotation for prefill bootstrap ports. Ensure RBAC grants `get/list/watch` on pods.

---

## Security & Authentication

- **Router API key (`--api-key`)**: clients must supply `Authorization: Bearer <key>`.
- **Worker API keys**: when adding workers dynamically, include `api_key` in the payload; workers listed via CLI inherit the router key.
- **Full-stack auth**: start router with `--api-key`, then add workers with their own keys:
  ```bash
  curl -H "Authorization: Bearer router-key" \
    -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url":"http://worker:8000","api_key":"worker-key"}'
  ```
- **Privacy**: All conversation history, `/v1/responses` state, and MCP sessions stay inside the router. Nothing is persisted at remote model vendors unless explicitly proxied.

---

## History & Data Connectors

| Backend | Description | Usage |
|---------|-------------|-------|
| `memory` (default) | In-memory storage for quick prototyping. | `--history-backend memory` |
| `none` | No persistence; APIs operate but store nothing. | `--history-backend none` |
| `oracle` | Oracle Autonomous Database-backed storage (pooled connections). | `--history-backend oracle` |
| `postgres` | PostgreSQL Database-backed storage (pooled connections). | `--history-backend postgres` |

Oracle configuration (choose DSN *or* TNS alias):
Install the Oracle Instant Client and set `LD_LIBRARY_PATH` accordingly.
Choose **one** connection method:
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
export ATP_PASSWORD="secret"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle
```

> History backends currently apply to OpenAI router mode. gRPC parity for `/v1/responses` is on the roadmap.

---

## MCP & Advanced Tooling

- Native MCP client supports **STDIO**, **HTTP**, **SSE**, and **Streamable** transports—no external config files required.
- Tool-call parsers cover JSON, Pythonic, XML, and custom schemas with streaming/non-streaming execution loops.
- Reasoning parsers ship for DeepSeek-R1, Qwen3, Step-3, GLM4, Llama families, Kimi K2, GPT-OSS, Mistral, and more (`src/reasoning_parser`).
- Tokenizer factory accepts HuggingFace IDs, local directories, and explicit `tokenizer.json` files with chat template overrides (`src/tokenizer`).

Use CLI flags to select parsers:
```bash
--reasoning-parser deepseek-r1 \
--tool-call-parser json \
--chat-template /path/to/template.json
```

---

## API Surface

| Method                | Path                                     | Description                                    |
|-----------------------|------------------------------------------|------------------------------------------------|
| `POST`                | `/generate`                              | SGLang generate API.                           |
| `POST`                | `/v1/chat/completions`                   | OpenAI-compatible chat (streaming/tool calls). |
| `POST`                | `/v1/completions`                        | OpenAI-compatible text completions.            |
| `POST`                | `/v1/responses`                          | Create background responses (agentic loops).   |
| `GET`                 | `/v1/responses/{id}`                     | Retrieve stored responses.                     |
| `POST`                | `/v1/embeddings`                         | Forward embedding requests.                    |
| `POST`                | `/v1/rerank`                             | Ranking endpoint (`/rerank` synonym).          |
| `POST`                | `/v1/conversations`                      | Create conversation metadata.                  |
| `GET`/`POST`/`DELETE` | `/v1/conversations/{id}`                 | Get/update/delete conversation.                |
| `GET`/`POST`          | `/v1/conversations/{id}/items`           | List or append conversation items.             |
| `GET`/`DELETE`        | `/v1/conversations/{id}/items/{item_id}` | Inspect/delete conversation item.              |
| `GET`                 | `/workers`                               | List registered workers with health/load.      |
| `POST`                | `/workers`                               | Queue worker registration.                     |
| `DELETE`              | `/workers/{url}`                         | Queue worker removal.                          |
| `POST`                | `/flush_cache`                           | Flush worker caches (HTTP workers).            |
| `GET`                 | `/get_loads`                             | Retrieve worker load snapshot.                 |
| `GET`                 | `/liveness` / `/readiness` / `/health`   | Health probes.                                 |

---

## Configuration Reference

### Core Settings

| Parameter                   | Type | Default     | Description                                                              |
|-----------------------------|------|-------------|--------------------------------------------------------------------------|
| `--host`                    | str  | 127.0.0.1   | Router host.                                                             |
| `--port`                    | int  | 30000       | Router port.                                                             |
| `--worker-urls`             | list | []          | Worker URLs (HTTP or gRPC).                                              |
| `--policy`                  | str  | cache_aware | Routing policy (`random`, `round_robin`, `cache_aware`, `power_of_two`). |
| `--max-concurrent-requests` | int  | -1          | Concurrency limit (-1 disables rate limiting).                           |
| `--request-timeout-secs`    | int  | 600         | Request timeout.                                                         |
| `--max-payload-size`        | int  | 256MB       | Maximum request payload.                                                 |

### Cache-Aware Tuning

| Parameter                  | Type  | Default  | Description                 |
|----------------------------|-------|----------|-----------------------------|
| `--cache-threshold`        | float | 0.3      | Minimum prefix match ratio. |
| `--balance-abs-threshold`  | int   | 64       | Absolute load threshold.    |
| `--balance-rel-threshold`  | float | 1.5      | Relative load ratio.        |
| `--eviction-interval-secs` | int   | 120      | Cache eviction cadence.     |
| `--max-tree-size`          | int   | 67108864 | Max nodes in cache tree.    |

### Fault Tolerance

| Parameter                    | Type  | Default | Description                      |
|------------------------------|-------|---------|----------------------------------|
| `--retry-max-retries`        | int   | 5       | Max retries.                     |
| `--retry-initial-backoff-ms` | int   | 50      | Initial backoff (ms).            |
| `--retry-max-backoff-ms`     | int   | 30000   | Max backoff (ms).                |
| `--retry-backoff-multiplier` | float | 1.5     | Backoff multiplier.              |
| `--retry-jitter-factor`      | float | 0.2     | Retry jitter (0.0-1.0).          |
| `--disable-retries`          | flag  | False   | Disable retries.                 |
| `--cb-failure-threshold`     | int   | 5       | Failures before opening circuit. |
| `--cb-success-threshold`     | int   | 2       | Successes to close circuit.      |
| `--cb-timeout-duration-secs` | int   | 30      | Cooldown period.                 |
| `--cb-window-duration-secs`  | int   | 60      | Window size.                     |
| `--disable-circuit-breaker`  | flag  | False   | Disable circuit breaker.         |

### Prefill/Decode

| Parameter                         | Type | Default | Description                              |
|-----------------------------------|------|---------|------------------------------------------|
| `--pd-disaggregation`             | flag | False   | Enable PD mode.                          |
| `--prefill`                       | list | []      | Prefill URLs + optional bootstrap ports. |
| `--decode`                        | list | []      | Decode URLs.                             |
| `--prefill-policy`                | str  | None    | Override policy for prefill nodes.       |
| `--decode-policy`                 | str  | None    | Override policy for decode nodes.        |
| `--worker-startup-timeout-secs`   | int  | 600     | Worker init timeout.                     |
| `--worker-startup-check-interval` | int  | 30      | Polling interval.                        |

### Kubernetes Discovery

| Parameter                                  | Type | Description                                                        |
|--------------------------------------------|------|--------------------------------------------------------------------|
| `--service-discovery`                      | flag | Enable discovery.                                                  |
| `--selector key=value ...`                 | list | Label selectors (regular mode).                                    |
| `--prefill-selector` / `--decode-selector` | list | Label selectors for PD mode.                                       |
| `--service-discovery-namespace`            | str  | Namespace to watch.                                                |
| `--service-discovery-port`                 | int  | Worker port (default 80).                                          |
| `--bootstrap-port-annotation`              | str  | Prefill bootstrap annotation (default `sglang.ai/bootstrap-port`). |

---

## Observability

Enable Prometheus metrics:
```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --prometheus-host 0.0.0.0 \
  --prometheus-port 29000
```

Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `sgl_router_requests_total` | Counter | Total requests by endpoint/method. |
| `sgl_router_processed_requests_total` | Counter | Requests processed per worker. |
| `sgl_router_active_workers` | Gauge | Healthy worker count. |
| `sgl_router_running_requests` | Gauge | In-flight requests per worker. |
| `sgl_router_cache_hits_total` / `misses_total` | Counter | Cache-aware routing hits/misses. |
| `sgl_router_generate_duration_seconds` | Histogram | Request latency distribution. |

Enable request ID propagation:
```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --request-id-headers x-request-id x-trace-id
```

Enable opentelmetry tracing:
```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint 0.0.0.0:4317
```

---

## Troubleshooting

1. **Workers never ready**
   Increase `--worker-startup-timeout-secs` or ensure health probes respond before router startup.

2. **Load imbalance / hot workers**
   Inspect `sgl_router_processed_requests_total` and tune cache-aware thresholds (`--balance-*`, `--cache-threshold`).

3. **Circuit breaker flapping**
   Increase `--cb-failure-threshold` or extend the timeout/window durations. Consider temporarily disabling retries.

4. **Queue overflow (429)**
   Increase `--queue-size` or reduce client concurrency. Ensure `--max-concurrent-requests` matches downstream capacity.

5. **Memory growth**
   Reduce `--max-tree-size` or lower `--eviction-interval-secs` for more aggressive cache pruning.

6. **Debugging**
   ```bash
   python -m sglang_router.launch_router \
     --worker-urls http://worker1:8000 \
     --log-level debug \
     --log-dir ./router_logs
   ```

---

SGLang Model Gateway continues to evolve alongside the SGLang runtime. Keep CLI flags, integrations, and documentation aligned when adopting new features or contributing improvements.
