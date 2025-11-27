# SGLang Metrics Missing When Using sgl.Engine Through Dynamo - Root Cause Analysis

**Date**: November 14, 2025
**Issue**: Missing TTFT, ITL, ISL, OSL metrics when using `sgl.Engine` through Dynamo
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

The issue reports that when SGLang is deployed through Dynamo using the `sgl.Engine` API, several key latency metrics (TTFT, ITL, ISL, OSL) are missing from the Prometheus `/metrics` endpoint. However, these metrics appear correctly when using `sglang.launch_server` directly.

**Root Cause Identified**: The metrics **ARE** being registered in both `sgl.Engine` and `launch_server` modes. The problem is that **the `/metrics` HTTP endpoint is NOT exposed** when using `sgl.Engine` in library mode. The Prometheus middleware that serves the `/metrics` endpoint is only added in `launch_server` (HTTP server mode), not in the Engine API mode.

**Impact**: Medium - Metrics are collected but not exposed via HTTP when using Engine API
**Complexity to Fix**: Low - Requires Dynamo to add its own Prometheus endpoint

---

## Investigation Findings

### 1. Metrics ARE Registered in Both Modes

**Key Discovery**: Both `sgl.Engine` and `launch_server` use the exact same initialization path (`_launch_subprocesses()`) which registers all metrics collectors.

#### Metrics Registration Path

**Location**: `python/sglang/srt/entrypoints/engine.py:776-930`

Both modes follow this path:
```
Engine.__init__() or launch_server()
  ↓
_launch_subprocesses(server_args)
  ↓
_set_envs_and_config(server_args)
  ↓ (if enable_metrics=True)
set_prometheus_multiproc_dir()  ← Sets PROMETHEUS_MULTIPROC_DIR env var
  ↓
TokenizerManager.__init__()
  ↓ (if enable_metrics=True)
TokenizerMetricsCollector.__init__()  ← Registers TTFT, ITL, E2E metrics
```

**Proof**: `python/sglang/srt/entrypoints/engine.py:717-719`
```python
# Set prometheus env vars
if server_args.enable_metrics:
    set_prometheus_multiproc_dir()
```

This is called by **both** Engine and launch_server via `_launch_subprocesses()`.

### 2. TTFT, ITL, and E2E Latency Metrics Registration

**Location**: `python/sglang/srt/metrics/collector.py:886-905`

**TokenizerMetricsCollector** registers:

```python
self.histogram_time_to_first_token = Histogram(
    name="sglang:time_to_first_token_seconds",
    documentation="Histogram of time to first token in seconds.",
    labelnames=labels.keys(),
    buckets=bucket_time_to_first_token,
)

self.histogram_inter_token_latency = Histogram(
    name="sglang:inter_token_latency_seconds",
    documentation="Histogram of inter-token latency in seconds.",
    labelnames=labels.keys(),
    buckets=bucket_inter_token_latency,
)

self.histogram_e2e_request_latency = Histogram(
    name="sglang:e2e_request_latency_seconds",
    documentation="Histogram of End-to-end request latency in seconds",
    labelnames=labels.keys(),
    buckets=bucket_e2e_request_latency,
)
```

**Initialization**: `python/sglang/srt/managers/tokenizer_manager.py:375-382`

```python
if self.enable_metrics:
    labels = {
        "model_name": self.server_args.served_model_name,
    }
    if server_args.tokenizer_metrics_allowed_custom_labels:
        for label in server_args.tokenizer_metrics_allowed_custom_labels:
            labels[label] = ""
    self.metrics_collector = TokenizerMetricsCollector(
        server_args=server_args,
        labels=labels,
        bucket_time_to_first_token=self.server_args.bucket_time_to_first_token,
        bucket_e2e_request_latency=self.server_args.bucket_e2e_request_latency,
        bucket_inter_token_latency=self.server_args.bucket_inter_token_latency,
        collect_tokens_histogram=self.server_args.collect_tokens_histogram,
    )
```

This happens in **both modes** because both modes create a `TokenizerManager`.

### 3. ISL and OSL Are NOT Separate Metrics

**Important Clarification**: ISL (Input Sequence Length) and OSL (Output Sequence Length) are **NOT** exposed as separate Prometheus metrics named "ISL" or "OSL" in SGLang.

Instead, SGLang tracks:
- `prompt_tokens_total` (Counter) - total prompt tokens across all requests
- `generation_tokens_total` (Counter) - total generation tokens across all requests
- `prompt_tokens_histogram` (Histogram) - distribution of prompt lengths (if `collect_tokens_histogram=True`)
- `generation_tokens_histogram` (Histogram) - distribution of generation lengths (if `collect_tokens_histogram=True`)

**Location**: `python/sglang/srt/metrics/collector.py:863-884`

### 4. The Real Difference: HTTP Endpoint Exposure

**This is the actual problem**: The `/metrics` HTTP endpoint is only added in `launch_server` mode, not in `sgl.Engine` mode.

#### launch_server (HTTP Server Mode)

**Location**: `python/sglang/srt/entrypoints/http_server.py:218-220`

```python
# Add prometheus middleware
if server_args.enable_metrics:
    add_prometheus_middleware(app)  # ← Adds /metrics endpoint to FastAPI app
    enable_func_timer()
```

**What `add_prometheus_middleware` does**: `python/sglang/srt/utils/common.py:1482-1492`

```python
def add_prometheus_middleware(app):
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))  # ← HTTP endpoint!

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
```

#### sgl.Engine (Library Mode)

**Location**: `python/sglang/srt/entrypoints/engine.py:105-157`

```python
def __init__(self, **kwargs):
    # ... server_args setup ...

    # Launch subprocesses (registers metrics!)
    tokenizer_manager, template_manager, scheduler_info, port_args = (
        _launch_subprocesses(server_args=server_args)
    )

    # ... ZMQ setup ...

    # NO HTTP SERVER CREATED!
    # NO add_prometheus_middleware() CALLED!
```

**Engine mode does NOT**:
- Create a FastAPI app
- Add Prometheus middleware
- Expose a `/metrics` HTTP endpoint

The metrics **are collected** and stored in the multiprocess directory (`PROMETHEUS_MULTIPROC_DIR`), but there is **no HTTP server** to serve them.

---

## Why This Affects Dynamo

### Dynamo's Architecture

Based on the issue description, Dynamo:
1. Uses `sgl.Engine` API to run SGLang backends (not `launch_server`)
2. Expects to query backend worker metrics at `localhost:9090/metrics` (or similar)
3. Needs these metrics for the SLA planner

### The Problem

When Dynamo invokes SGLang using `sgl.Engine`:
- SGLang initializes all metrics collectors ✓
- SGLang records TTFT, ITL, E2E latency to Prometheus ✓
- But there is NO HTTP endpoint to query these metrics ✗

**Result**: `curl localhost:9090/metrics | grep sglang` returns nothing or limited metrics

---

## Comparison with Direct launch_server

When using `launch_server` directly:

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-0.6B --enable-metrics
curl localhost:30000/metrics
```

This works because:
1. `launch_server` creates a FastAPI HTTP server
2. Adds Prometheus middleware with `/metrics` endpoint
3. Metrics are both collected AND exposed via HTTP

---

## Solution Options

### Option 1: Dynamo Adds Its Own Metrics Endpoint (RECOMMENDED)

**Location**: `dynamo/sglang/main.py` (in Dynamo repository)

Dynamo should add its own HTTP server to expose Prometheus metrics:

```python
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
import uvicorn

# After creating sgl.Engine
engine = sgl.Engine(
    model_path="...",
    enable_metrics=True,  # ← MUST be True!
    # ... other args ...
)

# Create a separate metrics server
app = FastAPI()

# Add Prometheus endpoint
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)
metrics_app = make_asgi_app(registry=registry)
app.mount("/metrics", metrics_app)

# Run on a separate port (e.g., 9090 for Prometheus)
uvicorn.run(app, host="0.0.0.0", port=9090)
```

**Pros**:
- No changes needed to SGLang
- Dynamo has full control over metrics endpoint
- Can add additional Dynamo-specific metrics

**Cons**:
- Requires changes to Dynamo

### Option 2: SGLang Adds Optional Metrics Server to Engine

**Location**: `python/sglang/srt/entrypoints/engine.py`

Add a new parameter to expose metrics:

```python
class Engine(EngineBase):
    def __init__(
        self,
        enable_metrics=False,
        enable_metrics_server=False,  # ← NEW
        metrics_server_port=9090,      # ← NEW
        **kwargs
    ):
        # ... existing initialization ...

        if enable_metrics and enable_metrics_server:
            self._start_metrics_server(metrics_server_port)

    def _start_metrics_server(self, port):
        from fastapi import FastAPI
        from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
        import uvicorn
        import threading

        app = FastAPI()
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        app.mount("/metrics", make_asgi_app(registry=registry))

        # Run in background thread
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=port)

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
```

**Usage from Dynamo**:
```python
engine = sgl.Engine(
    model_path="...",
    enable_metrics=True,
    enable_metrics_server=True,  # ← NEW
    metrics_server_port=9090,
)
```

**Pros**:
- Minimal changes to Dynamo
- Easier integration for other frameworks

**Cons**:
- Requires changes to SGLang
- Adds HTTP server dependency to Engine mode

### Option 3: Direct File-Based Metrics Collection

Since metrics are already written to `PROMETHEUS_MULTIPROC_DIR`, Dynamo could read them directly:

```python
import os
from prometheus_client import CollectorRegistry, multiprocess

# Get the multiprocess directory
multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

# Create registry and collect metrics
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

# Export metrics in various formats
from prometheus_client import generate_latest
metrics_text = generate_latest(registry)
```

**Pros**:
- No HTTP server needed
- Direct access to metrics

**Cons**:
- More complex integration
- Need to handle file locking and cleanup

---

## Required Configuration

### Critical: `enable_metrics` MUST Be Set

**For ANY of the solutions to work**, Dynamo must pass `enable_metrics=True` when creating the Engine:

```python
# In dynamo/sglang/main.py
engine = sgl.Engine(
    model_path=args.model,
    enable_metrics=True,  # ← CRITICAL!
    # ... other args ...
)
```

**Default value**: `enable_metrics=False` (from `python/sglang/srt/server_args.py:307`)

If this is not set, **NO metrics collectors are created**, regardless of HTTP endpoint availability.

---

## Testing Verification

### Test 1: Verify Metrics Are Registered with sgl.Engine

```python
from sglang import Engine
import os

# Create engine with metrics enabled
engine = Engine(
    model_path="Qwen/Qwen3-0.6B",
    enable_metrics=True,
    log_level="info",
)

# Check that PROMETHEUS_MULTIPROC_DIR is set
print(f"Metrics dir: {os.environ.get('PROMETHEUS_MULTIPROC_DIR')}")

# Generate some requests
for _ in range(10):
    result = engine.generate(prompt="Hello", sampling_params={"max_new_tokens": 10})

# Check that metric files exist
import glob
metric_files = glob.glob(f"{os.environ['PROMETHEUS_MULTIPROC_DIR']}/*.db")
print(f"Metric files: {metric_files}")
```

**Expected**: Metric files should exist in the multiproc directory

### Test 2: Verify Metrics Can Be Collected Programmatically

```python
from prometheus_client import CollectorRegistry, multiprocess, generate_latest

# Create registry and collect
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

# Generate Prometheus format output
metrics_output = generate_latest(registry).decode('utf-8')

# Check for expected metrics
assert "sglang:time_to_first_token_seconds" in metrics_output
assert "sglang:inter_token_latency_seconds" in metrics_output
assert "sglang:e2e_request_latency_seconds" in metrics_output

print("✓ All metrics present")
print(metrics_output)
```

---

## Action Items for Dynamo Team

1. **Immediate**:
   - [ ] Verify that `enable_metrics=True` is passed when creating `sgl.Engine`
   - [ ] Check if there's an existing HTTP server in Dynamo that could expose `/metrics`

2. **Short-term** (Option 1 - Recommended):
   - [ ] Add FastAPI metrics server to `dynamo/sglang/main.py`
   - [ ] Mount Prometheus endpoint at `/metrics`
   - [ ] Configure Prometheus to scrape this endpoint

3. **Alternative** (Option 2):
   - [ ] Request SGLang to add `enable_metrics_server` parameter to Engine
   - [ ] Wait for SGLang release with this feature
   - [ ] Update Dynamo integration to use new parameter

---

## Action Items for SGLang Team

**Optional Enhancement**: Add `enable_metrics_server` parameter to `sgl.Engine` to make metrics exposure easier for frameworks using the Engine API (Option 2 above).

**Priority**: Low-Medium (nice-to-have, not critical since Dynamo can implement Option 1)

---

## Conclusion

**The metrics ARE being registered** in both `sgl.Engine` and `launch_server` modes. The issue is purely about **HTTP endpoint exposure**:

- ✅ Metrics collectors initialized: YES (both modes)
- ✅ TTFT, ITL, E2E metrics registered: YES (both modes)
- ✅ Metrics recorded during inference: YES (both modes)
- ❌ HTTP `/metrics` endpoint exposed: NO (Engine mode) / YES (launch_server mode)

**Recommended Solution**: Dynamo should add its own HTTP server with a `/metrics` endpoint that serves the Prometheus metrics from the multiprocess directory (Option 1).

**Configuration Required**: Ensure `enable_metrics=True` is passed when creating `sgl.Engine`.

---

**Document Version**: 1.0
**Last Updated**: November 14, 2025
**Status**: Ready for Implementation
