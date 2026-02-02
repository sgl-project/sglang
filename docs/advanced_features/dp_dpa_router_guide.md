# Comprehensive Guide: Navigating DP, DPA, and Router Best Practices

This guide serves as the definitive "Source of Truth" for understanding and optimizing parallelism strategies in SGLang. As SGLang expands its support for advanced model architectures (such as DeepSeek), choosing the right parallelism strategy becomes crucial for achieving high-throughput, large-scale inference.

## Table of Contents

1. [Overview: Parallelism Strategies in SGLang](#overview-parallelism-strategies-in-sglang)
2. [Understanding Data Parallelism (DP)](#understanding-data-parallelism-dp)
3. [Understanding DPA (Data Parallelism Attention)](#understanding-dpa-data-parallelism-attention)
   - [What is DPA?](#what-is-dpa)
   - [DPA with Expert Parallelism for MoE](#dpa-with-expert-parallelism-for-moe)
   - [Target Models](#target-models)
   - [Activation Logic](#activation-logic)
4. [Native DP vs. Router-Based DP](#native-dp-vs-router-based-dp)
   - [Native DP Mode](#native-dp-mode)
   - [Router-Based DP (Recommended)](#router-based-dp-recommended)
   - [Comparison Summary](#comparison-summary)
5. [Practical Implementation: DP Routing via Router](#practical-implementation-dp-routing-via-router)
   - [Quick Start](#quick-start)
   - [Load Balancing Policies](#load-balancing-policies)
   - [Best Practices](#best-practices)
   - [Verifying Traffic Distribution](#verifying-traffic-distribution)
6. [Quick Reference](#quick-reference)

---

## Overview: Parallelism Strategies in SGLang

SGLang supports multiple parallelism strategies that can be combined for optimal performance:

| Strategy | Component | Description |
|----------|-----------|-------------|
| **TP (Tensor Parallelism)** | All layers | Splits model weights across GPUs |
| **DP (Data Parallelism)** | Full model | Replicates model, processes different batches |
| **DPA (DP Attention)** | Attention layers | DP for attention, avoids KV cache duplication |
| **EP (Expert Parallelism)** | MoE layers | Distributes expert weights across GPUs |
| **PP (Pipeline Parallelism)** | Across layers | Distributes layers across pipeline stages |

For **DeepSeek models** (V2/V3/R1), the recommended strategy combines:
- **DPA** for attention layers (eliminates KV cache duplication)
- **EP** for MoE layers (enables large-scale expert distribution)
- **DP** for dense FFN and LM head layers

This achieves up to **5x throughput improvement** compared to vanilla tensor parallelism.

---

## Understanding Data Parallelism (DP)

**Data Parallelism (DP)** is a parallelism strategy that replicates the entire model across multiple GPU sets and processes different batches of requests in parallel. Each GPU set handles independent requests, effectively multiplying the throughput of your serving system.

**Key characteristics:**
- Each worker has a full copy of the model
- Requests are distributed across workers
- No inter-worker communication during inference (for simple DP)
- Linear throughput scaling with the number of workers

---

## Understanding DPA (Data Parallelism Attention)

### What is DPA?

**Data Parallelism Attention (DPA)**, also known as DP Attention, is an advanced parallelism strategy specifically designed for models with **Multi-Head Latent Attention (MLA)** architecture, such as DeepSeek models.

#### The Problem with Tensor Parallelism for MLA Models

The most common parallelism strategy for inference is **Tensor Parallelism (TP)**. However, TP might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have **one KV head**. If we use tensor parallelism on 8 GPUs, it will lead to:

- **Duplicated KV cache** across all GPUs
- **Unwanted memory usage** that limits batch size
- **Reduced throughput** due to memory constraints

#### How DPA Works

DPA addresses these limitations by applying **data parallelism specifically to the attention component**. Each DP worker:
- Processes different batches independently
- Maintains its own KV cache (no duplication)
- Enables 4x larger batch sizes due to memory savings

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  DeepSeek Parallelism Architecture (DPA + EP)                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐            │
│   │  Worker 0 │   │  Worker 1 │   │  Worker 2 │   │  Worker 3 │   ...      │
│   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘            │
│         │               │               │               │                  │
│         ▼               ▼               ▼               ▼                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              Attention Layer (Data Parallel - DPA)                  │  │
│   │  • Each worker processes different batches independently            │  │
│   │  • KV cache is NOT duplicated (significant memory savings)          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                 │                                          │
│                         Reduce-Scatter / All-Gather                        │
│                                 │                                          │
│                                 ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              MoE Layers (Expert Parallelism - EP)                   │  │
│   │  • Expert weights distributed across all GPUs                       │  │
│   │  • Critical for large-scale MoE models (256+ experts)               │  │
│   │  • See Expert Parallelism documentation for details                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                 │                                          │
│                          Redistribute                                      │
│                                 │                                          │
│                                 ▼                                          │
│                    (Back to DPA for next layer)                            │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key benefits of DPA:**

1. **Significantly reduced KV cache memory**: Each DP worker only stores KV cache for its own batches
2. **Larger batch sizes**: Memory savings allow for 4x larger batch sizes
3. **Up to 1.9x decoding throughput improvement**: Demonstrated on DeepSeek models
4. **Independent batch handling**: Each DP worker handles different types of batches (prefill, decode, idle) independently

### DPA with Expert Parallelism for MoE

For MoE models like DeepSeek, **DPA must be combined with Expert Parallelism (EP)** for optimal performance. While DPA handles the attention layers efficiently, the MoE layers require EP to:

- Distribute 256+ expert weights across GPUs (cannot fit on single GPU)
- Enable efficient all-to-all token routing via DeepEP
- Scale to large clusters (up to 5x throughput improvement over vanilla TP)

**Recommended setup for DeepSeek:**

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --ep 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

For detailed EP configuration (DeepEP, Two-Batch Overlap, EPLB), see [Expert Parallelism](expert_parallelism.md).

### Target Models

DPA is specifically designed for and **automatically enabled** for the following model architectures:

| Model Family | Architecture Feature | Auto-Enable Condition |
|--------------|---------------------|----------------------|
| **DeepSeek-V2** | MLA with single KV head | `tp_size > 1` and `dp_size > 1` |
| **DeepSeek-V3** | MLA with single KV head | `tp_size > 1` and `dp_size > 1` |
| **DeepSeek-R1** | MLA with single KV head | `tp_size > 1` and `dp_size > 1` |
| **DeepSeek-Coder-V2** | MLA with single KV head | `tp_size > 1` and `dp_size > 1` |

**Note**: DPA is optimized for MLA-based architectures. For standard multi-head attention models (like Llama, Qwen, etc.), standard DP or TP is recommended.

### Activation Logic

DPA can be enabled in two ways:

#### 1. Automatic Activation (Recommended for DeepSeek models)

SGLang automatically enables DPA for supported models when the following conditions are met:

```python
# Automatic activation logic (simplified)
if model_architecture in ["DeepSeekV2", "DeepSeekV3"]:
    if tp_size > 1 and dp_size > 1:
        enable_dp_attention = True  # Auto-enabled
```

#### 2. Manual Activation

You can explicitly enable DPA using the `--enable-dp-attention` flag:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --enable-dp-attention
```

---

## Native DP vs. Router-Based DP

### Native DP Mode

Native DP (built-in Data Parallelism) in SGLang creates multiple worker processes within a single server instance:

```bash
# Native DP mode
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

**Characteristics:**
- Single server process manages all workers
- Simple round-robin or basic load balancing
- Workers share the same memory pool configuration
- Limited load balancing intelligence
- No cache-aware routing

### Router-Based DP (Recommended)

**We strongly recommend using the SGLang Router for production-grade Data Parallelism.** The Router provides significant advantages over native DP mode.

```bash
# Router-based DP mode (Recommended)
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

**Advantages of Router-Based DP:**

| Feature | Native DP | Router-Based DP |
|---------|-----------|-----------------|
| **Load Balancing** | Basic round-robin | Advanced policies (cache-aware, power-of-two, etc.) |
| **Cache Awareness** | ❌ No | ✅ Yes - up to 3.8x higher cache hit rate |
| **Throughput** | Baseline | Up to 1.9x improvement |
| **Multi-Node Support** | Limited | ✅ Full support |
| **Worker Health Monitoring** | Basic | ✅ Circuit breakers, health checks |
| **Reliability** | Basic | ✅ Retries, rate limiting, queuing |
| **Observability** | Basic metrics | ✅ 40+ Prometheus metrics, OpenTelemetry |
| **Hot Worker Add/Remove** | ❌ No | ✅ Yes |

### Comparison Summary

#### Cache-Aware Router Performance

The cache-aware routing policy in SGLang Router significantly improves performance for workloads with shared prefixes:

| Metric | Without Cache-Aware | With Cache-Aware Router |
|--------|---------------------|-------------------------|
| Throughput (token/s) | 82,665 | 158,596 (+92%) |
| Cache Hit Rate | 20% | 75% (+275%) |

*Benchmark from [SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), workload with multiple long prefix groups, 8x A100 80GB GPUs, dp-size=8*

#### When to Use Each

**Use Native DP when:**
- Quick prototyping or development
- Single-node deployments with simple workloads
- You don't need advanced load balancing

**Use Router-Based DP when:**
- Production deployments
- Multi-node distributed setups
- Workloads with shared prefixes (high cache reuse potential)
- You need high availability and reliability features
- You require detailed observability and metrics

---

## Practical Implementation: DP Routing via Router

### Quick Start

#### Installation

```bash
pip install sglang-router
# or
pip install "sglang[all]"
```

#### Option 1: Co-launch Workers and Router (Simplest)

This is the easiest way to get started - the router and workers are launched together:

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

#### Option 2: Separate Launch (Multi-Node)

For distributed deployments across multiple machines:

**Step 1: Launch workers on each node**

```bash
# Node 1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000

# Node 2
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000
```

**Step 2: Launch router pointing to workers**

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:8000 http://node2:8000 \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000
```

#### Option 3: Dynamic Worker Registration

For elastic deployments where workers can be added/removed dynamically:

```bash
# Launch router first
python -m sglang_router.launch_router \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker1:8000"}'

curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker2:8000"}'
```

### Load Balancing Policies

The router supports multiple load balancing policies:

| Policy | Description | Best For |
|--------|-------------|----------|
| `cache_aware` | Combines cache locality with load balancing | **Recommended for most workloads** |
| `round_robin` | Cycles through workers in order | Simple, predictable distribution |
| `random` | Random worker selection | Baseline, testing |
| `power_of_two` | Samples two workers, picks lighter one | Low latency requirements |
| `bucket` | Divides workers into load buckets | Variable workload sizes |

#### Cache-Aware Policy (Default, Recommended)

The cache-aware policy provides the best performance for most workloads:

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 32 \
    --balance-rel-threshold 1.5 \
    --eviction-interval-secs 120 \
    --max-tree-size 67108864
```

**How it works:**
1. Maintains an approximate radix tree for each worker based on request history
2. Routes requests to workers with the highest prefix match (cache hit)
3. Falls back to shortest-queue routing when load is imbalanced
4. Automatically evicts old entries to prevent memory overflow

### Best Practices

1. **Start with `cache_aware` policy** - It provides the best balance between cache locality and load distribution for most workloads
2. **Use Router-Based DP for production** - Prefer `sglang_router.launch_server` over `--dp-size` for better reliability and observability
3. **Enable health checks** - Configure `--health-check-interval-secs` to detect and remove unhealthy workers automatically

**Recommended command with best practices applied:**

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --router-policy cache_aware \
    --health-check-interval-secs 30 \
    --router-prometheus-port 10001 \
    --host 0.0.0.0 \
    --port 30000
```

For advanced configuration (circuit breakers, retries, Prometheus metrics, K8s integration), see [SGLang Model Gateway](sgl_model_gateway.md).

### Verifying Traffic Distribution

After launching your router, verify that traffic is being distributed correctly:

**1. Check worker status:**

```bash
curl http://localhost:30000/workers
```

**2. Check load distribution:**

```bash
curl http://localhost:30000/get_loads
```

**3. Monitor metrics (if Prometheus enabled):**

```bash
# Key metrics to check
sglang_router_requests_total{worker="..."}
sglang_router_cache_hit_rate
sglang_router_worker_load{worker="..."}
```

For detailed metrics and monitoring setup, see [SGLang Model Gateway](sgl_model_gateway.md).

---

## Quick Reference

| Strategy | Use Case | Key Benefit |
|----------|----------|-------------|
| **Native DP** (`--dp-size`) | Development, simple workloads | Easy to configure, single process |
| **Router-Based DP** | Production, multi-node | Cache-aware routing, high availability |
| **DPA** | DeepSeek/MLA models | Eliminates KV cache duplication, 1.9x throughput |
| **DPA + EP** | DeepSeek MoE models | Up to 5x throughput vs vanilla TP |

**Recommended production setup for DeepSeek:**
1. Enable **DPA** for attention layers (`--enable-dp-attention`)
2. Enable **EP** for MoE layers (`--ep 8 --moe-a2a-backend deepep`)
3. Use **Router-Based DP** with **cache_aware** policy

**Related documentation:**
- [Expert Parallelism](expert_parallelism.md) - DeepEP, Two-Batch Overlap, EPLB
- [SGLang Model Gateway](sgl_model_gateway.md) - Router configuration & troubleshooting
- [Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - 96 GPU deployment guide
