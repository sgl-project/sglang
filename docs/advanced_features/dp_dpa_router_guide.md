# Comprehensive Guide: DP, DPA, and Router

This guide explains the difference between Data Parallelism (DP) and Data Parallelism Attention (DPA), how to enable each mode correctly, and how to use the router for production-grade DP deployments.

## Data Parallelism (DP)

**Data Parallelism (DP)** is the most common parallelism strategy that replicates the entire model across multiple GPU sets and processes different batches of requests in parallel. Each GPU set handles independent requests. With dedicated routing strategies, as we will introduce later, the throughput of your serving system could be multiplied near linearly.

**Key characteristics:**
- Each replica has a full copy of the model
- Requests are distributed across replicas
- No inter-replica communication during one request's inference (for simple DP)

> **Note for MLA models (e.g., DeepSeek)**: Standard DP works for MLA models too. If you want independent replicas with duplicated KV cache (simpler setup), just use `--dp-size` without `--enable-dp-attention`. DPA is only needed when you want to eliminate KV cache duplication for better memory efficiency.


## Data Parallelism Attention (DPA)

**Data Parallelism Attention (DPA)**, also known as DP Attention, is an advanced parallelism strategy specifically designed for models with **Multi-Head Latent Attention (MLA)** architecture, such as DeepSeek, MiniMax, Kimi-K2, and other MLA-based models.

**The Problem with Tensor Parallelism for MLA Models**

The most common parallelism strategy for inference is **Tensor Parallelism (TP)**. However, TP might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have **one KV head**. If we use tensor parallelism on 8 GPUs, it will lead to:

- **Duplicated KV cache** across all GPUs
- **Unwanted memory usage** that limits batch size
- **Reduced throughput** due to memory constraints

**How DPA Works**

DPA addresses these limitations by applying **data parallelism specifically to the attention component**. Each DP replica:
- Processes different batches independently (can be in different forward modes: prefill, decode, or idle)
- Maintains its own KV cache (no duplication)
- Enables significantly larger batch sizes due to memory savings

![DPA + EP Architecture](../_static/image/dpa.png)

**Communication patterns in DPA + EP:**

- **All2All (Dispatch)**: Routes tokens to expert sub-groups based on router decisions - each token is sent to the GPU(s) hosting its assigned expert(s)
- **All2All (Combine)**: Gathers computed results from experts back to original token positions

**Key benefits of DPA:**

1. **Significantly reduced KV cache memory**: Each DP replica only stores KV cache for its own batches
2. **Larger batch sizes**: Memory savings enable larger batch sizes
3. **Improved decoding throughput**: Significant throughput gains for MLA-based models
4. **Independent forward modes**: Each DP replica can be in different forward modes (prefill, decode, or idle) and handles its assigned batches independently during attention computation

### DPA with Expert Parallelism for MoE

For MoE models like DeepSeek, DPA is **often** paired with Expert Parallelism (EP) for best throughput at scale. However, **DPA does not require EP**: you can enable DPA without EP if your deployment does not need expert sharding.

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

**DPA is specifically optimized for MLA (Multi-Head Latent Attention) based models**, including:
- DeepSeek family (DeepSeek-V2, DeepSeek-V3, DeepSeek-R1)
- MiniMax models
- Kimi-K2
- Other models using MLA architecture

For standard multi-head attention models (like Llama, Qwen, etc.), standard DP or TP is recommended.

To enable DPA, add `--enable-dp-attention` to your server launch command.

### Activation Logic

DPA is enabled explicitly via server arguments (CLI or config). You can enable DPA using the `--enable-dp-attention` flag:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --enable-dp-attention
```

---

## Native DP vs. Router

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

**We strongly recommend using the Router for production-grade Data Parallelism.** The Router provides significant advantages over native DP mode.

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
| **Cache Awareness** | ❌ No | ✅ Yes - significantly higher cache hit rate |
| **Throughput** | Baseline | Significant improvement |
| **Multi-Node Support** | Limited | ✅ Full support |
| **Worker Health Monitoring** | Basic | ✅ Circuit breakers, health checks |
| **Reliability** | Basic | ✅ Retries, rate limiting, queuing |
| **Observability** | Basic metrics | ✅ 40+ Prometheus metrics, OpenTelemetry |
| **Hot Worker Add/Remove** | ❌ No | ✅ Yes |

### Comparison Summary

**Cache-Aware Routing Performance**

The cache-aware routing policy in the Router significantly improves performance for workloads with shared prefixes:

| Metric | Without Cache-Aware | With Cache-Aware Router |
|--------|---------------------|-------------------------|
| Throughput (token/s) | 82,665 | 158,596 (+92%) |
| Cache Hit Rate | 20% | 75% (+275%) |

*Benchmark from [SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), workload with multiple long prefix groups, 8x A100 80GB GPUs, dp-size=8*

**When to Use Each**

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

**Installation**

```bash
pip install sglang-router
# or
pip install "sglang[all]"
```

**Option 1: Co-launch Workers and Router (Simplest)**

This is the easiest way to get started - Router and workers are launched together:

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

**Option 2: Separate Launch (Multi-Node)**

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

**Step 2: Launch Router pointing to workers**

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:8000 http://node2:8000 \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000
```

**Option 3: Dynamic Worker Registration**

For elastic deployments where workers can be added/removed dynamically:

```bash
# Launch Router first
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

The Router supports multiple load balancing policies:

| Policy | Description | Best For |
|--------|-------------|----------|
| `cache_aware` | Combines cache locality with load balancing | **Recommended for most workloads** |
| `round_robin` | Cycles through workers in order | Simple, predictable distribution |
| `random` | Random worker selection | Baseline, testing |
| `power_of_two` | Samples two workers, picks lighter one | Low latency requirements |

**Cache-Aware Policy (Default, Recommended)**

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
2. **Use Router for production** - Prefer `sglang_router.launch_server` over `sglang.launch_server` for better reliability and observability
3. **Enable health checks** - Configure `--router-health-check-interval-secs` to detect and remove unhealthy workers automatically

**Recommended command with best practices applied:**

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --router-policy cache_aware \
    --router-health-check-interval-secs 30 \
    --router-prometheus-port 10001 \
    --host 0.0.0.0 \
    --port 30000
```

For advanced configuration (circuit breakers, retries, Prometheus metrics, K8s integration), see [Router Documentation](sgl_model_gateway.md).

### Verifying Traffic Distribution

After launching the Router, verify that traffic is being distributed correctly:

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

For detailed metrics and monitoring setup, see [Router Documentation](sgl_model_gateway.md).

---

## Quick Reference

| Strategy | Use Case | Key Benefit |
|----------|----------|-------------|
| **Native DP** (`--dp-size`) | Development, simple workloads | Easy to configure, single process |
| **Router-Based DP** | Production, multi-node | Cache-aware routing, high availability |
| **DPA** | DeepSeek/MLA models | Eliminates KV cache duplication, improved throughput |
| **DPA + EP** | DeepSeek MoE models | Significant throughput improvement vs vanilla TP |

**Recommended production setup for DeepSeek:**
1. Enable **DPA** for attention layers (`--enable-dp-attention`)
2. Enable **EP** for MoE layers (`--ep 8 --moe-a2a-backend deepep`)
3. Use **Router** with **cache_aware** policy

**Related documentation:**
- [Expert Parallelism](expert_parallelism.md) - DeepEP, Two-Batch Overlap, EPLB
- [Router Documentation](sgl_model_gateway.md) - Router configuration & troubleshooting
- [Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - 96 GPU deployment guide
