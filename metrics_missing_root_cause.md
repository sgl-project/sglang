# SGLang Metrics Missing in Dynamo - ACTUAL Root Cause Analysis

**Date**: November 16, 2025
**Issue**: Missing TTFT, ITL, ISL, OSL, and per-stage metrics when using `sgl.Engine` through Dynamo
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

You are correct - some metrics ARE missing when using `sgl.Engine`. The issue is NOT about HTTP endpoint exposure (my initial analysis was wrong). The actual root cause is:

**Scheduler metrics (including per-stage latencies like ISL/OSL) are only collected from rank 0 by default**, even when `enable_metrics=True`. In disaggregated mode with multiple workers, only `attn_tp_rank==0` collects SchedulerMetricsCollector metrics unless `enable_metrics_for_all_schedulers=True` is also set.

---

## What's Actually Happening

### Two Different Metrics Collectors

SGLang has **two separate** metrics collectors:

#### 1. TokenizerMetricsCollector (Always enabled when enable_metrics=True)
**Location**: `python/sglang/srt/metrics/collector.py:708-993`
**Initialized in**: `python/sglang/srt/managers/tokenizer_manager.py:375-382`

**Metrics exposed**:
- ✅ `sglang:time_to_first_token_seconds` (TTFT)
- ✅ `sglang:inter_token_latency_seconds` (ITL)
- ✅ `sglang:e2e_request_latency_seconds`
- ✅ `prompt_tokens_total`
- ✅ `generation_tokens_total`
- ✅ `cached_tokens_total`
- ✅ `num_requests_total`

**Condition**: Created when `enable_metrics=True`
**Available in**: ALL modes (Engine and launch_server)

#### 2. SchedulerMetricsCollector (Conditionally enabled)
**Location**: `python/sglang/srt/metrics/collector.py:218-706`
**Initialized in**: `python/sglang/srt/managers/scheduler_metrics_mixin.py:60-71`

**Metrics exposed**:
- ❌ `sglang:per_stage_req_latency_seconds` (includes ISL, OSL, and all stage latencies)
- ❌ `sglang:queue_time_seconds`
- ❌ `sglang:num_running_reqs`
- ❌ `sglang:num_used_tokens`
- ❌ `sglang:token_usage`
- ❌ `sglang:gen_throughput`
- ❌ `sglang:num_queue_reqs`
- ❌ `sglang:cache_hit_rate`
- ❌ `sglang:kv_transfer_speed_gb_s`
- ❌ `sglang:kv_transfer_latency_ms`
- ❌ And many more scheduler-level metrics

**Condition**: Created when `enable_metrics=True` BUT only rank 0 actually collects/logs these metrics
**The Problem**: Even when created, metrics are only collected if `current_scheduler_metrics_enabled()` returns True

---

## The Actual Bug

### Code Location: scheduler.py:2588-2589

```python
def current_scheduler_metrics_enabled(self):
    return self.attn_tp_rank == 0 or self.enable_metrics_for_all_schedulers
```

This function is called before **every** scheduler metrics collection operation:

**Examples**:

1. **scheduler.py:641-644** (Initialization):
```python
if self.current_scheduler_metrics_enabled():
    self.send_metrics_from_scheduler = get_zmq_socket(
        context, zmq.PUSH, port_args.metrics_ipc_name, False
    )
```

2. **scheduler.py:1873-1874** (Prefill stats):
```python
if self.current_scheduler_metrics_enabled():
    self.log_prefill_stats(adder, can_run_list, running_bs, 0)
```

3. **scheduler_output_processor_mixin.py:387-390** (Decode stats):
```python
if (
    self.current_scheduler_metrics_enabled()
    and self.forward_ct_decode % self.server_args.decode_log_interval == 0
):
    self.log_decode_stats(can_run_cuda_graph, running_batch=batch)
```

### What This Means

In a Dynamo disaggregated setup with multiple workers:
- Worker with `attn_tp_rank==0`: ✅ Collects ALL metrics (Tokenizer + Scheduler)
- Workers with `attn_tp_rank!=0`: ✅ Collects Tokenizer metrics, ❌ **SKIPS** Scheduler metrics

**Result**: You see TTFT and ITL (from TokenizerMetricsCollector), but NOT per-stage latencies or queue metrics (from SchedulerMetricsCollector).

---

## ISL and OSL Explained

ISL (Input Sequence Length) and OSL (Output Sequence Length) are NOT standalone metrics. They are **stage labels** in the `per_stage_req_latency_seconds` histogram.

### Request Stages

**Location**: `python/sglang/srt/managers/schedule_batch.py:405-432`

```python
class RequestStage(str, enum.Enum):
    # Tokenizer
    TOKENIZE = "tokenize"              # ← This includes input tokenization (ISL)
    TOKENIZER_DISPATCH = "dispatch"

    # DP controller
    DC_DISPATCH = "dc_dispatch"

    # common/non-disaggregation
    PREFILL_WAITING = "prefill_waiting"
    REQUEST_PROCESS = "request_process"
    DECODE_LOOP = "decode_loop"        # ← This includes output generation (OSL)
    PREFILL_FORWARD = "prefill_forward"
    PREFILL_CHUNKED_FORWARD = "chunked_prefill"

    # disaggregation prefill
    PREFILL_PREPARE = "prefill_prepare"
    PREFILL_BOOTSTRAP = "prefill_bootstrap"
    PREFILL_TRANSFER_KV_CACHE = "prefill_transfer_kv_cache"

    # disaggregation decode
    DECODE_PREPARE = "decode_prepare"
    DECODE_BOOTSTRAP = "decode_bootstrap"
    DECODE_WAITING = "decode_waiting"
    DECODE_TRANSFERRED = "decode_transferred"
    DECODE_FAKE_OUTPUT = "fake_output"
    DECODE_QUICK_FINISH = "quick_finish"
```

### How Per-Stage Metrics Work

**Metric**: `sglang:per_stage_req_latency_seconds`
**Labels**: `model_name`, `engine_type`, `tp_rank`, `pp_rank`, `dp_rank` (if applicable), **`stage`**

**Example Prometheus output**:
```
sglang:per_stage_req_latency_seconds{model_name="Qwen3-0.6B",stage="tokenize"} 0.002
sglang:per_stage_req_latency_seconds{model_name="Qwen3-0.6B",stage="prefill_forward"} 0.015
sglang:per_stage_req_latency_seconds{model_name="Qwen3-0.6B",stage="decode_loop"} 0.003
sglang:per_stage_req_latency_seconds{model_name="Qwen3-0.6B",stage="prefill_bootstrap"} 0.050
```

**Recorded by**: `SchedulerMetricsCollector.observe_per_stage_req_latency(stage, latency)`

**The problem**: This is only called when `current_scheduler_metrics_enabled()` returns True!

---

## Why This Happens in Dynamo

### Typical Dynamo Setup

Based on the issue description:
```yaml
# disagg_planner.yaml
prefill_workers:
  - host: worker1, attn_tp_rank: 0  # ✅ Collects scheduler metrics
  - host: worker2, attn_tp_rank: 1  # ❌ Skips scheduler metrics
  - host: worker3, attn_tp_rank: 2  # ❌ Skips scheduler metrics

decode_workers:
  - host: worker4, attn_tp_rank: 0  # ✅ Collects scheduler metrics
  - host: worker5, attn_tp_rank: 1  # ❌ Skips scheduler metrics
  - host: worker6, attn_tp_rank: 2  # ❌ Skips scheduler metrics
```

When you query `curl localhost:9090/metrics | grep sglang`:
- You get TTFT/ITL from ALL workers (because TokenizerMetricsCollector doesn't check rank)
- You get per-stage/queue metrics from ONLY attn_tp_rank==0 workers
- If you're querying a non-rank-0 worker, you see **no scheduler metrics at all**

---

## The Solution

### Option 1: Set enable_metrics_for_all_schedulers=True (RECOMMENDED)

When creating the Engine in Dynamo, add this parameter:

```python
# In dynamo/sglang/main.py
engine = sgl.Engine(
    model_path=args.model,
    enable_metrics=True,
    enable_metrics_for_all_schedulers=True,  # ← ADD THIS!
    # ... other args ...
)
```

**Effect**: All workers will collect scheduler metrics, not just rank 0.

**server_args.py definition** (line 308):
```python
enable_metrics_for_all_schedulers: bool = False
```

**Pros**:
- Simple one-line fix
- Gets you all metrics from all workers
- No code changes to SGLang needed

**Cons**:
- Slightly higher overhead (more workers writing metrics)
- More metric data to aggregate

### Option 2: Aggregate Metrics from Rank 0 Only

If you only need metrics from one worker per TP group:

1. Configure Dynamo to query only rank 0 workers for scheduler metrics
2. Keep `enable_metrics_for_all_schedulers=False` (default)
3. In your SLA planner, combine:
   - Tokenizer metrics from any worker
   - Scheduler metrics from rank 0 workers only

**Pros**:
- Lower overhead
- Less metric data

**Cons**:
- More complex aggregation logic
- Need to know which workers are rank 0

### Option 3: Patch SGLang to Always Enable Scheduler Metrics

Modify `scheduler.py:2588-2589`:

```python
def current_scheduler_metrics_enabled(self):
    # OLD:
    # return self.attn_tp_rank == 0 or self.enable_metrics_for_all_schedulers

    # NEW:
    return True  # Always enable when enable_metrics=True
```

**Pros**:
- Simplest for users
- Consistent behavior

**Cons**:
- Requires SGLang code change
- May have performance implications (original design intentionally limited to rank 0)

---

## Testing Verification

### Test 1: Verify Current Behavior

```bash
# In Dynamo, deploy with current config
# Query a non-rank-0 worker
curl localhost:9090/metrics | grep -E "sglang:(time_to_first_token|per_stage_req_latency)"

# Expected:
sglang:time_to_first_token_seconds{...} 0.05    # ✅ Present
sglang:per_stage_req_latency_seconds{...} ...   # ❌ Missing!
```

### Test 2: Verify Fix with enable_metrics_for_all_schedulers

```python
# In dynamo/sglang/main.py
engine = sgl.Engine(
    model_path="Qwen/Qwen3-0.6B",
    enable_metrics=True,
    enable_metrics_for_all_schedulers=True,  # ← Added
)
```

```bash
# Query the same non-rank-0 worker
curl localhost:9090/metrics | grep -E "sglang:(time_to_first_token|per_stage_req_latency)"

# Expected:
sglang:time_to_first_token_seconds{...} 0.05              # ✅ Present
sglang:per_stage_req_latency_seconds{stage="tokenize"} ... # ✅ Now present!
sglang:per_stage_req_latency_seconds{stage="decode_loop"} ... # ✅ Now present!
```

### Test 3: Verify All Stages Are Present

```bash
curl localhost:9090/metrics | grep "sglang:per_stage_req_latency_seconds" | cut -d'{' -f2 | cut -d',' -f1

# Expected output (example):
stage="tokenize"
stage="dispatch"
stage="prefill_waiting"
stage="request_process"
stage="prefill_forward"
stage="decode_loop"
stage="prefill_bootstrap"
stage="decode_bootstrap"
# etc.
```

---

## Additional Notes on HTTP Endpoint

My original analysis about HTTP endpoint exposure was partially correct:

- **Engine mode**: No HTTP server by default, but Prometheus metrics ARE written to `PROMETHEUS_MULTIPROC_DIR`
- **Dynamo**: Must expose its own `/metrics` endpoint to serve these metrics

So you still need to ensure Dynamo has a metrics HTTP server (as described in my first analysis), but the BIGGER issue is that even with the endpoint, you won't see scheduler metrics without `enable_metrics_for_all_schedulers=True`.

---

## Summary of Fixes Needed

### In Dynamo Integration

1. **Critical**: Add `enable_metrics_for_all_schedulers=True` when creating Engine:
```python
engine = sgl.Engine(
    model_path=args.model,
    enable_metrics=True,
    enable_metrics_for_all_schedulers=True,  # ← Fix missing metrics
    # ... other args ...
)
```

2. **Also needed** (from original analysis): Expose metrics via HTTP endpoint:
```python
# Add Prometheus endpoint to Dynamo
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

app = FastAPI()
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)
app.mount("/metrics", make_asgi_app(registry=registry))

# Run on port 9090 or similar
uvicorn.run(app, host="0.0.0.0", port=9090)
```

### Both Fixes Are Required

- **Fix 1** (enable_metrics_for_all_schedulers): Makes scheduler metrics actually get collected
- **Fix 2** (HTTP endpoint): Makes collected metrics accessible via HTTP

Without Fix 1, you won't see per-stage latencies, queue metrics, etc.
Without Fix 2, you won't be able to query metrics via HTTP at all.

---

## Metrics Reference

### Always Available (TokenizerMetricsCollector)

| Metric Name | Description | When Collected |
|-------------|-------------|----------------|
| `sglang:time_to_first_token_seconds` | TTFT histogram | All ranks with enable_metrics=True |
| `sglang:inter_token_latency_seconds` | ITL histogram | All ranks with enable_metrics=True |
| `sglang:e2e_request_latency_seconds` | End-to-end latency | All ranks with enable_metrics=True |
| `prompt_tokens_total` | Total prompt tokens | All ranks with enable_metrics=True |
| `generation_tokens_total` | Total generation tokens | All ranks with enable_metrics=True |

### Conditionally Available (SchedulerMetricsCollector)

| Metric Name | Description | When Collected |
|-------------|-------------|----------------|
| `sglang:per_stage_req_latency_seconds{stage=...}` | Per-stage latencies (includes ISL/OSL stages) | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |
| `sglang:queue_time_seconds` | Time spent in queue | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |
| `sglang:num_running_reqs` | Running request count | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |
| `sglang:num_queue_reqs` | Queued request count | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |
| `sglang:gen_throughput` | Generation throughput | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |
| `sglang:cache_hit_rate` | Prefix cache hit rate | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |
| `sglang:kv_transfer_speed_gb_s` | KV transfer speed | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |
| `sglang:kv_transfer_latency_ms` | KV transfer latency | Only attn_tp_rank==0 OR enable_metrics_for_all_schedulers=True |

---

## Conclusion

The missing metrics issue is caused by:

1. ✅ **Primary cause**: `enable_metrics_for_all_schedulers=False` (default) - only rank 0 collects scheduler metrics
2. ✅ **Secondary cause**: No HTTP /metrics endpoint when using Engine mode

**Immediate action**: Add `enable_metrics_for_all_schedulers=True` to your Engine initialization in Dynamo.

---

**Document Version**: 2.0 (Corrected)
**Last Updated**: November 16, 2025
**Status**: Ready for Implementation
