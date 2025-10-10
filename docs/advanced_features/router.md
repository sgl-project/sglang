# SGLang Router

The SGLang Router is a high-performance request distribution system that routes inference requests across multiple SGLang runtime instances. It features cache-aware load balancing, fault tolerance, and support for advanced deployment patterns including data parallelism and prefill-decode disaggregation.

## Key Features

- **Cache-Aware Load Balancing**: Optimizes cache utilization while maintaining balanced load distribution
- **Multiple Routing Policies**: Choose from random, round-robin, cache-aware, or power-of-two policies
- **Fault Tolerance**: Automatic retry and circuit breaker mechanisms for resilient operation
- **Dynamic Scaling**: Add or remove workers at runtime without service interruption
- **Kubernetes Integration**: Native service discovery and pod management
- **Prefill-Decode Disaggregation**: Support for disaggregated serving load balancing
- **Prometheus Metrics**: Built-in observability and monitoring

## Installation

```bash
pip install sglang-router
```

## Quick Start

To see all available options:

```bash
python -m sglang_router.launch_server --help  # Co-launch router and workers
python -m sglang_router.launch_router --help  # Launch router only
```

## Deployment Modes

The router supports three primary deployment patterns:

1. **Co-launch Mode**: Router and workers launch together (simplest for single-node deployments)
2. **Separate Launch Mode**: Router and workers launch independently (best for multi-node setups)
3. **Prefill-Decode Disaggregation**: Specialized mode for disaggregated serving

### Mode 1: Co-launch Router and Workers

This mode launches both the router and multiple worker instances in a single command. It's the simplest deployment option and replaces the `--dp-size` argument of SGLang Runtime.

```bash
# Launch router with 4 workers
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

#### Sending Requests

Once the server is ready, send requests to the router endpoint:

```python
import requests

# Using the /generate endpoint
url = "http://localhost:30000/generate"
data = {
    "text": "What is the capital of France?",
    "sampling_params": {
        "temperature": 0.7,
        "max_new_tokens": 100
    }
}

response = requests.post(url, json=data)
print(response.json())

# OpenAI-compatible endpoint
url = "http://localhost:30000/v1/chat/completions"
data = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}

response = requests.post(url, json=data)
print(response.json())
```

### Mode 2: Separate Launch Mode

This mode is ideal for multi-node deployments where workers run on different machines.

#### Step 1: Launch Workers

On each worker node:

```bash
# Worker node 1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000

# Worker node 2
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8001
```

#### Step 2: Launch Router

On the router node:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8001 \
    --host 0.0.0.0 \
    --port 30000 \
    --policy cache_aware  # or random, round_robin, power_of_two
```

### Mode 3: Prefill-Decode Disaggregation

This advanced mode separates prefill and decode operations for optimized performance:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://prefill1:8000 9000 \
    --prefill http://prefill2:8001 9001 \
    --decode http://decode1:8002 \
    --decode http://decode2:8003 \
    --prefill-policy cache_aware \
    --decode-policy round_robin
```

#### Understanding --prefill Arguments

The `--prefill` flag accepts URLs with optional bootstrap ports:
- `--prefill http://server:8000` - No bootstrap port
- `--prefill http://server:8000 9000` - Bootstrap port 9000
- `--prefill http://server:8000 none` - Explicitly no bootstrap port

#### Policy Inheritance in PD Mode

The router intelligently handles policy configuration for prefill and decode nodes:

1. **Only `--policy` specified**: Both prefill and decode nodes use this policy
2. **`--policy` and `--prefill-policy` specified**: Prefill nodes use `--prefill-policy`, decode nodes use `--policy`
3. **`--policy` and `--decode-policy` specified**: Prefill nodes use `--policy`, decode nodes use `--decode-policy`
4. **All three specified**: Prefill nodes use `--prefill-policy`, decode nodes use `--decode-policy` (main `--policy` is ignored)

Example with mixed policies:
```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://prefill1:8000
    --prefill http://prefill2:8000 \
    --decode http://decode1:8001
    --decode http://decode2:8001 \
    --policy round_robin \
    --prefill-policy cache_aware  # Prefill uses cache_aware and decode uses round_robin from --policy
```

#### PD Mode with Service Discovery

For Kubernetes deployments with separate prefill and decode server pools:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --service-discovery \
    --prefill-selector app=prefill-server tier=gpu \
    --decode-selector app=decode-server tier=cpu \
    --service-discovery-namespace production \
    --prefill-policy cache_aware \
    --decode-policy round_robin
```

## Dynamic Scaling

The router supports runtime scaling through REST APIs:

### Adding Workers

```bash
# Launch a new worker
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 30001

# Add it to the router
curl -X POST "http://localhost:30000/add_worker?url=http://127.0.0.1:30001"
```

### Removing Workers

```bash
curl -X POST "http://localhost:30000/remove_worker?url=http://127.0.0.1:30001"
```

**Note**: When using cache-aware routing, removed workers are cleanly evicted from the routing tree and request queues.

## Fault Tolerance

The router includes comprehensive fault tolerance mechanisms:

### Retry Configuration

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8001 \
    --retry-max-retries 3 \
    --retry-initial-backoff-ms 100 \
    --retry-max-backoff-ms 10000 \
    --retry-backoff-multiplier 2.0 \
    --retry-jitter-factor 0.1
```

### Circuit Breaker

Protects against cascading failures:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8001 \
    --cb-failure-threshold 5 \
    --cb-success-threshold 2 \
    --cb-timeout-duration-secs 30 \
    --cb-window-duration-secs 60
```

**Behavior**:
- Worker is marked unhealthy after `cb-failure-threshold` consecutive failures
- Returns to service after `cb-success-threshold` successful health checks
- Circuit breaker can be disabled with `--disable-circuit-breaker`

## Routing Policies

The router supports multiple routing strategies:

### 1. Random Routing
Distributes requests randomly across workers.

```bash
--policy random
```

### 2. Round-Robin Routing
Cycles through workers in order.

```bash
--policy round_robin
```

### 3. Power of Two Choices
Samples two workers and routes to the less loaded one.

```bash
--policy power_of_two
```

### 4. Cache-Aware Load Balancing (Default)

The most sophisticated policy that combines cache optimization with load balancing:

```bash
--policy cache_aware \
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.0001
```

#### How It Works

1. **Load Assessment**: Checks if the system is balanced
   - Imbalanced if: `(max_load - min_load) > balance_abs_threshold` AND `max_load > balance_rel_threshold * min_load`

2. **Routing Decision**:
   - **Balanced System**: Uses cache-aware routing
     - Routes to worker with highest prefix match if match > `cache_threshold`
     - Otherwise routes to worker with most available cache capacity
   - **Imbalanced System**: Uses shortest queue routing to the least busy worker

3. **Cache Management**:
   - Maintains approximate radix trees per worker
   - Periodically evicts LRU entries based on `--eviction-interval-secs` and `--max-tree-size`

### Data Parallelism Aware Routing

Enables fine-grained control over data parallel replicas:

```bash
--dp-aware \
--api-key your_api_key  # Required for worker authentication
```

This mode coordinates with SGLang's DP controller for optimized request distribution across data parallel ranks.

## Configuration Reference

### Core Settings

| Parameter                   | Type | Default     | Description                                                     |
| --------------------------- | ---- | ----------- | --------------------------------------------------------------- |
| `--host`                    | str  | 127.0.0.1   | Router server host address                                      |
| `--port`                    | int  | 30000       | Router server port                                              |
| `--worker-urls`             | list | []          | Worker URLs for separate launch mode                            |
| `--policy`                  | str  | cache_aware | Routing policy (random, round_robin, cache_aware, power_of_two) |
| `--max-concurrent-requests` | int  | 64          | Maximum concurrent requests (rate limiting)                     |
| `--request-timeout-secs`    | int  | 600         | Request timeout in seconds                                      |
| `--max-payload-size`        | int  | 256MB       | Maximum request payload size                                    |

### Cache-Aware Routing Parameters

| Parameter                  | Type  | Default  | Description                                            |
| -------------------------- | ----- | -------- | ------------------------------------------------------ |
| `--cache-threshold`        | float | 0.5      | Minimum prefix match ratio for cache routing (0.0-1.0) |
| `--balance-abs-threshold`  | int   | 32       | Absolute load difference threshold                     |
| `--balance-rel-threshold`  | float | 1.0001   | Relative load ratio threshold                          |
| `--eviction-interval-secs` | int   | 60       | Seconds between cache eviction cycles                  |
| `--max-tree-size`          | int   | 16777216 | Maximum nodes in routing tree                          |

### Fault Tolerance Parameters

| Parameter                    | Type  | Default | Description                           |
| ---------------------------- | ----- | ------- | ------------------------------------- |
| `--retry-max-retries`        | int   | 3       | Maximum retry attempts per request    |
| `--retry-initial-backoff-ms` | int   | 100     | Initial retry backoff in milliseconds |
| `--retry-max-backoff-ms`     | int   | 10000   | Maximum retry backoff in milliseconds |
| `--retry-backoff-multiplier` | float | 2.0     | Backoff multiplier between retries    |
| `--retry-jitter-factor`      | float | 0.1     | Random jitter factor for retries      |
| `--disable-retries`          | flag  | False   | Disable retry mechanism               |
| `--cb-failure-threshold`     | int   | 5       | Failures before circuit opens         |
| `--cb-success-threshold`     | int   | 2       | Successes to close circuit            |
| `--cb-timeout-duration-secs` | int   | 30      | Circuit breaker timeout duration      |
| `--cb-window-duration-secs`  | int   | 60      | Circuit breaker window duration       |
| `--disable-circuit-breaker`  | flag  | False   | Disable circuit breaker               |

### Prefill-Decode Disaggregation Parameters

| Parameter                         | Type | Default | Description                                           |
| --------------------------------- | ---- | ------- | ----------------------------------------------------- |
| `--pd-disaggregation`             | flag | False   | Enable PD disaggregated mode                          |
| `--prefill`                       | list | []      | Prefill server URLs with optional bootstrap ports     |
| `--decode`                        | list | []      | Decode server URLs                                    |
| `--prefill-policy`                | str  | None    | Routing policy for prefill nodes (overrides --policy) |
| `--decode-policy`                 | str  | None    | Routing policy for decode nodes (overrides --policy)  |
| `--worker-startup-timeout-secs`   | int  | 300     | Timeout for worker startup                            |
| `--worker-startup-check-interval` | int  | 10      | Interval between startup checks                       |

### Kubernetes Integration

| Parameter                       | Type | Default                  | Description                                          |
| ------------------------------- | ---- | ------------------------ | ---------------------------------------------------- |
| `--service-discovery`           | flag | False                    | Enable Kubernetes service discovery                  |
| `--selector`                    | list | []                       | Label selector for workers (key1=value1 key2=value2) |
| `--prefill-selector`            | list | []                       | Label selector for prefill servers in PD mode        |
| `--decode-selector`             | list | []                       | Label selector for decode servers in PD mode         |
| `--service-discovery-port`      | int  | 80                       | Port for discovered pods                             |
| `--service-discovery-namespace` | str  | None                     | Kubernetes namespace to watch                        |
| `--bootstrap-port-annotation`   | str  | sglang.ai/bootstrap-port | Annotation for bootstrap ports                       |

### Observability

| Parameter              | Type | Default   | Description                                           |
| ---------------------- | ---- | --------- | ----------------------------------------------------- |
| `--prometheus-port`    | int  | 29000     | Prometheus metrics port                               |
| `--prometheus-host`    | str  | 127.0.0.1 | Prometheus metrics host                               |
| `--log-dir`            | str  | None      | Directory for log files                               |
| `--log-level`          | str  | info      | Logging level (debug, info, warning, error, critical) |
| `--request-id-headers` | list | None      | Custom headers for request tracing                    |

### CORS Configuration

| Parameter                | Type | Default | Description          |
| ------------------------ | ---- | ------- | -------------------- |
| `--cors-allowed-origins` | list | []      | Allowed CORS origins |

## Advanced Features

### Kubernetes Service Discovery

Automatically discover and manage workers in Kubernetes:

#### Standard Mode
```bash
python -m sglang_router.launch_router \
    --service-discovery \
    --selector app=sglang-worker env=prod \
    --service-discovery-namespace production \
    --service-discovery-port 8000
```

#### Prefill-Decode Disaggregation Mode
```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --service-discovery \
    --prefill-selector app=prefill-server env=prod \
    --decode-selector app=decode-server env=prod \
    --service-discovery-namespace production
```

**Note**: The `--bootstrap-port-annotation` (default: `sglang.ai/bootstrap-port`) is used to discover bootstrap ports for prefill servers in PD mode. Prefill pods should have this annotation set to their bootstrap port value.

### Prometheus Metrics

Expose metrics for monitoring:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8001 \
    --prometheus-port 29000 \
    --prometheus-host 0.0.0.0
```

Metrics available at `http://localhost:29000/metrics`

### Request Tracing

Enable request ID tracking:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8001 \
    --request-id-headers x-request-id x-trace-id
```

## Observability

When Prometheus is enabled, the router provides several key metrics for observability.

| Metric Name                            | Type      | Description                                                                                          |
|:---------------------------------------|:----------|:-----------------------------------------------------------------------------------------------------|
| `sgl_router_requests_total`            | Counter   | Total number of requests received by the router's API endpoint. Useful for tracking overall traffic. |
| `sgl_router_processed_requests_total`  | Counter   | Total requests processed, labeled by `worker`. Critical for spotting load imbalances.                |
| `sgl_router_active_workers`            | Gauge     | The current number of healthy workers in the routing pool. Essential for alerting.                   |
| `sgl_router_running_requests`          | Gauge     | The number of currently in-flight requests, labeled by `worker`. For monitoring real-time load.      |
| `sgl_router_cache_hits_total`          | Counter   | Total requests routed to a worker with a matching prefix cache.                                      |
| `sgl_router_cache_misses_total`        | Counter   | Total requests that could not be routed based on cache locality.                                     |
| `sgl_router_generate_duration_seconds` | Histogram | Tracks end-to-end request latency. Use this to monitor performance (e.g., p95/p99).                  |

## Troubleshooting

### Common Issues

1. **Workers not connecting**: Ensure workers are fully initialized before starting the router. Use `--worker-startup-timeout-secs` to increase wait time.

2. **High latency**:
   - **A common cause**: Load Imbalanced.
   - Check the `sgl_router_processed_requests_total` metric grouped by `worker`.
   - Cache-aware routing might be prioritizing cache hits too aggressively.
   - Try adjusting `--balance-abs-threshold` and `--balance-rel-threshold`.

3. **Memory growth**: Reduce `--max-tree-size` or decrease `--eviction-interval-secs` for more aggressive cache cleanup.

4. **Circuit breaker triggering frequently**: Increase `--cb-failure-threshold` or extend `--cb-window-duration-secs`.

### Debug Mode

Enable detailed logging:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8001 \
    --log-level debug \
    --log-dir ./router_logs
```
