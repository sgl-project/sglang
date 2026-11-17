---
title: "DP, DPA and sgl-router"
metatags:
    description: "Learn the differences between Data Parallelism, Data Parallelism Attention, and sgl-router routing for production DP deployments."
---

This guide explains the difference between Data Parallelism (DP) and Data Parallelism Attention (DPA), how to enable each mode correctly, and how to use `sgl-router`, SGLang's router, for production-grade DP deployments.

## Data Parallelism (DP)

**Data Parallelism (DP)** is the most common parallelism strategy that replicates the entire model across multiple GPU sets and processes different batches of requests in parallel. Each GPU set handles independent requests. With dedicated routing strategies, as we will introduce later, with those proper routing algorithms in `sgl-router`, the throughput of your serving system could be multiplied nearly linearly.

### Key characteristics

- Each replica has a full copy of the model
- Requests are distributed/scattered across replicas
- No inter-replica communication during one request's inference (for simple DP)

## Data Parallelism Attention (DPA)

**Data Parallelism Attention (DPA)**, also known as DP Attention, is an advanced parallelism strategy. While DPA provides the most significant benefits for **Multi-Head Latent Attention (MLA)** models (such as DeepSeek, MiniMax, Kimi-K2), it also supports **standard attention models** like Qwen.

### The Problem with Tensor Parallelism for MLA Models

The most common parallelism strategy for inference is **Tensor Parallelism (TP)**. However, TP might not be the most efficient strategy for certain models. For example, DeepSeek models use MLA and only have **one KV head**. If we use tensor parallelism on 8 GPUs, it will lead to:

- **Duplicated KV cache** across all GPUs
- **Unwanted memory usage** that limits batch size
- **Reduced throughput** due to memory constraints

### How DPA Works

DPA addresses these limitations by applying **data parallelism specifically to the attention component**.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", verticalAlign: "top", backgroundColor: "rgba(255,255,255,0.02)"}}>
        <img src="/images/dpa.png" alt="DPA + EP Architecture" style={{width: "100%", height: "auto"}} />
      </td>
      <td style={{padding: "9px 12px", verticalAlign: "top", backgroundColor: "rgba(255,255,255,0.05)"}}>
        <p><strong>Each DP replica:</strong></p>
        <ul>
          <li>Processes different batches independently (can be in different forward modes: prefill, decode, or idle)</li>
          <li>Maintains its own KV cache (no duplication)</li>
          <li>Enables significantly larger batch sizes due to memory savings</li>
        </ul>
        <p><strong>Communication patterns in DPA + EP:</strong></p>
        <ul>
          <li><strong>All2All (Dispatch)</strong>: Routes tokens to expert sub-groups based on gating decisions</li>
          <li><strong>All2All (Combine)</strong>: Gathers computed results from experts back to original token positions</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

### Key benefits of DPA

1. **Significantly reduced KV cache memory**: Each DP replica only stores KV cache for its own batches
2. **Larger batch sizes**: Memory savings enable larger batch sizes
3. **Improved decoding throughput**: Significant throughput gains for MLA-based models
4. **Independent forward modes**: Each DP replica can be in different forward modes (prefill, decode, or idle) and handles its assigned batches independently during attention computation

### DPA with Expert Parallelism for MoE

For MoE models like DeepSeek, DPA is **often** paired with Expert Parallelism (EP) for best throughput at scale. However, **DPA does not require EP**: you can enable DPA without EP if your deployment does not need expert sharding.

- Distribute 256+ expert weights across GPUs (cannot fit on a single GPU)
- Enable efficient all-to-all token routing via DeepEP
- Scale to large clusters (up to 5x throughput improvement over vanilla TP)

### Recommended setup for DeepSeek

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --attn-dp-size 8 \
    --ep 8 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

> **Note**: `--attn-dp-size` sets the number of attention data-parallel groups inside the TP group. If `attn_dp_size` is 1 (default), DPA is disabled.

For detailed EP configuration (DeepEP, Two-Batch Overlap, EPLB), see [Expert Parallelism](/docs/advanced_features/expert_parallelism).

### Target Models

DPA supports the following model architectures:

- **MLA (Multi-Head Latent Attention) models** - where DPA provides the most significant benefits:
  - DeepSeek family (DeepSeek-V2, DeepSeek-V3, DeepSeek-R1)
  - MiniMax models
  - Kimi-K2
  - Other models using MLA architecture

- **Standard attention models** - also supported:
  - Qwen models (see [PR #6121](https://github.com/sgl-project/sglang/pull/6121))

For models like Llama, with standard GQA, standard DP, or TP is typically recommended.

To enable DPA, add `--attn-dp-size N` (N > 1) to your server launch command.

### Activation Logic

DPA is enabled explicitly via server arguments (CLI or config) with `--attn-dp-size` (alias `--attention-data-parallel-size`). It creates N attention data-parallel groups inside the TP group, one scheduler each, while the FFN stays tensor parallel:

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --attn-dp-size 8
```

**Important**: `--attn-dp-size` must be greater than 1 for DPA to work. When `attn_dp_size == 1` (default), DPA is disabled. The constraint `tp_size % attn_dp_size == 0` must also be satisfied. `--dp-size` only sets the number of data-parallel replicas of the model; combining replicas (`--dp-size` > 1) with DPA is not supported.

> **Deprecated**: `--enable-dp-attention` still works: `--dp-size N --enable-dp-attention` resolves to `--attn-dp-size N --dp-size 1` and prints a deprecation warning. Use `--attn-dp-size N` instead. The flag will be removed after 2026-12-31.
>
> Under DeepSeek context parallelism (DSA, MLA, and DeepSeek-V4 prefill CP), `--dp-size N` alone still runs N attention-DP groups, and logs a deprecation warning; it will be rejected after 2026-12-31. Use `--attn-dp-size N` there too.

**Elastic EP scale joiners.** A rank that joins an existing DPA deployment through elastic EP scale-up runs DPA even when it is launched with `--attn-dp-size 1`: its own attention DP group is one rank wide.

**DP rank count.** The number of DP ranks (one scheduler each) is `dp_size × attn_dp_size`; one of the two is always 1.
- `/server_info` reports `dp_size` (replicas) and `attn_dp_size` separately, so under DPA its `dp_size` is 1. Until the flag is removed, its deprecated `enable_dp_attention` field still reports whether DPA runs; new clients should read `attn_dp_size`.
- Wire formats that predate `--attn-dp-size` keep the key `dp_size` for the DP rank count: KV-event descriptors, the PD prefill server info, and load snapshots.

### TP LMHead with all-to-all

`dp-lm-head` keeps full weight and computes the local batch, making the gemm inefficient when batchsize per-rank is small. The existing `tp-lm-head` improves the gemm efficiency but introduce heavy allgather communication on full vocabulary.

Therefore, all-to-all can be applied to eliminate the heavy allgather while retaining the gemm efficiency under small per-rank batchsize. Each rank will send and receive only the local rows with sharded vocab, which is more lightweight.

This optimization is **enabled by default for decode node and pure DPA**. Users can use `--no-enable-tp-lm-head-all-to-all` to opt out, or `--enable-tp-lm-head-all-to-all` for force enabling.

```bash
python -m sglang.launch_server \
    --model-path MODEL_PATH \
    --disaggregation-mode decode \
    --tp 8 \
    --attn-dp-size 8
```

Prefill / colocated workers will fallback to TP LMHead with allgather, as well as other parallel strategies. More details can be found in [PR #32313](https://github.com/sgl-project/sglang/pull/32313).

### Standard DP for MLA models

Note that MLA models, of course, also support DP. Suppose you want to enable standard DP for MLA models. First, launch each MLA model's replica independently. You may launch these replicas one by one with DPA enabled. After launching each MLA model's replica, launch `sgl-router` and connect all the replicas to it. A detailed explanation of `sgl-router` is as follows.

## Data Parallelism with sgl-router

### Native DP Mode

Native DP (built-in Data Parallelism) in SGLang creates multiple worker processes within a single SGLang instance, under the control of `DataParallelController` with the launching parameter of `dp-size`.

```bash
# Native DP mode
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

**Limitations:**

- Built-in in-process load balancing only (e.g., `round_robin`, `total_requests`, `total_tokens`)
- No cache-aware routing
- Limited observability and metrics
- No fault tolerance or circuit breakers
- Not suitable for production workloads

⚠️ Native DP is **highly not recommended for use right now**. It is only used in some ancient/outdated RL frameworks. You can use `sgl-router` to power up your data parallelism in any use case.

### sgl-router-Based DP (Recommended)

`sgl-router` is SGLang's router, written in Rust to keep routing overhead low. It serves one model and spreads its requests across that model's SGLang servers, picking a server per request from live load, KV-cache prefix matches, or session affinity. It also routes prefill/decode disaggregation (see [PD Disaggregation](./pd_disaggregation#router-integration)); this doc only discusses its usage in DP routing.

**We strongly recommend running separate SGLang servers behind `sgl-router` for production-grade Data Parallelism.** This provides significant advantages over native DP mode.

**Advantages of sgl-router-Based DP:**

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "28%"}} />
    <col style={{width: "34%"}} />
    <col style={{width: "38%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Feature</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Native DP</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>sgl-router-Based DP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Load Balancing</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Built-in in-process methods</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Advanced policies (cache-aware, power-of-two, etc.)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Cache Awareness</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>❌ No</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes - significantly higher cache hit rate</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Throughput</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Baseline</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Significant improvement</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Multi-Node Support</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Limited</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Full support</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Worker Health Monitoring</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Basic</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Circuit breakers</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Reliability</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Basic</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Retries, admission limits</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Observability</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Basic metrics</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Prometheus metrics</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Hot Worker Add/Remove</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>❌ No</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes (Kubernetes service discovery)</td>
    </tr>
  </tbody>
</table>

### Cache-Aware Routing Performance

Cache-aware routing significantly improves performance for workloads with shared prefixes:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "33%"}} />
    <col style={{width: "33%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Metric</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Without Cache-Aware</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>With Cache-Aware Routing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Throughput (token/s)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>82,665</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>158,596 (+92%)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Cache Hit Rate</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>20%</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>75% (+275%)</td>
    </tr>
  </tbody>
</table>

*Benchmark from [SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), workload with multiple long prefix groups, 8x A100 80GB GPUs, dp-size=8*

### When to Use Each

**Use Native DP when:**

- ~Never use Native/Naive DP~
- Learning material of DP routing

**Use sgl-router-Based DP when:**

- In any case, when you think DP is needed
- Production deployments
- Multi-node distributed setups
- Workloads with shared prefixes (high cache reuse potential)
- You need high availability and reliability features
- You require detailed observability and metrics
- You want to have highly efficient RL rollout systems

Note that for RL rollout systems, **there are four crucial reasons that sgl-router-based DP is far better than naive DP routing**. Details can be found at [Load Balancing Router in RL](/docs/advanced_features/sglang_for_rl#load-balancing-router).

### Quick Start For sgl-router

**Installation**

`sgl-router` is pre-installed on the `PATH` of the SGLang Docker images. If you installed SGLang from source, build it from the `sgl-router` directory of the SGLang repository (this needs a Rust toolchain, e.g., from [rustup](https://rustup.rs)) and put it on your `PATH`:

```bash
cd sgl-router
cargo build --release --bin sgl-router
cp target/release/sgl-router /usr/local/bin/
```

**1. Launch workers**

Launch one SGLang server per replica, on one or more nodes. `--kv-events-config` publishes the KV-cache events that the `cache_aware` policy builds its prefix tree from, and `--load-publish-endpoint auto` publishes the load reports that load-aware routing reads. On a single node, give each server its own GPUs (`--base-gpu-id`), `--port`, and KV-event endpoint port.

```bash
# Node 1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv"}' \
    --load-publish-endpoint auto \
    --port 8000

# Node 2
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv"}' \
    --load-publish-endpoint auto \
    --port 8000
```

**2. Launch sgl-router pointing to workers**

```bash
sgl-router \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --worker-urls http://node1:8000 http://node2:8000 \
    --chat-routing reorg \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000
```

`--model-id` must match the servers' model name, which is their `--served-model-name`, or their `--model-path` when that flag is not set. The router loads the tokenizer for `--model-id` from Hugging Face; for a local checkpoint, pass `--tokenizer-path /path/to/checkpoint/tokenizer.json`. Wait for `curl http://localhost:30000/readiness` to return HTTP 200, which means the workers are registered.

**Elastic deployments on Kubernetes**

For deployments where workers are added and removed dynamically, `sgl-router` can discover them from Kubernetes EndpointSlices instead of a static `--worker-urls` list. `--selector` matches the labels of the workers' Service:

```bash
sgl-router \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --chat-routing reorg \
    --policy cache_aware \
    --service-discovery \
    --service-discovery-namespace default \
    --selector app=sglang-llama \
    --host 0.0.0.0 \
    --port 30000
```

### Load Balancing Policies

`sgl-router` supports multiple load balancing policies:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "24%"}} />
    <col style={{width: "42%"}} />
    <col style={{width: "34%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Policy</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Description</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><code>cache_aware</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Combines cache locality with load balancing</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Recommended for most workloads</strong></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><code>round_robin</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Cycles through workers in order</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Simple, predictable distribution</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><code>random</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Random worker selection</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Baseline, testing</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><code>power_of_two</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Samples two workers, picks lighter one</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Low latency requirements</td>
    </tr>
  </tbody>
</table>

**Cache-Aware Policy (Recommended)**

The cache-aware policy provides the best performance for most workloads:

```bash
sgl-router \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --chat-routing reorg \
    --policy cache_aware \
    --cache-affinity-min-matched-tokens 1024 \
    --affinity-mode balanced
```

**How it works:**

1. Maintains a radix tree of each worker's cached prefixes, built from the KV-cache events the workers publish
2. Routes requests to the worker with the longest cached prefix of the prompt, if it matches at least `--cache-affinity-min-matched-tokens` tokens
3. Falls back to `power_of_two` when no worker has such a prefix or the preferred worker is at an admission limit; with `--affinity-mode balanced`, it also switches away from the preferred worker when that worker is much more loaded than an alternative (`--affinity-load-factor`, `--affinity-load-gap`)

### Best Practices

1. **Start with `cache_aware` policy** - It provides the best balance between cache locality and load distribution for most workloads
2. **Use sgl-router for production** - Prefer separate servers behind `sgl-router` over a single `sglang.launch_server --dp-size` for better reliability and observability
3. **Enable retries and keep the circuit breaker** - `--retry-max-attempts` resends a request that failed before any response to another worker, and a worker leaves routing after `--cb-threshold` consecutive failures until it recovers

**Recommended command with best practices applied:**

```bash
sgl-router \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --worker-urls http://node1:8000 http://node2:8000 \
    --chat-routing reorg \
    --policy cache_aware \
    --retry-max-attempts 3 \
    --cb-threshold 3 \
    --host 0.0.0.0 \
    --port 30000
```

For advanced configuration (admission limits, session affinity, DP-rank routing, Kubernetes integration), see `sgl-router --help` and the [sgl-router README](https://github.com/sgl-project/sglang/blob/main/sgl-router/README.md).

### Verifying Traffic Distribution

After launching `sgl-router`, verify that traffic is being distributed correctly:

**1. Check that the workers are registered:**

```bash
curl http://localhost:30000/readiness
```

**2. Check load distribution in the router's Prometheus metrics:**

```bash
curl http://localhost:30000/metrics
```

```text
# Key metrics to check
sgl_router_workers{mode="..."}
sgl_router_worker_requests_total{worker_url="..."}
sgl_router_worker_inflight_requests{worker_url="..."}
```

For detailed metrics and monitoring setup, see the [sgl-router monitoring guide](https://github.com/sgl-project/sglang/blob/main/sgl-router/monitoring/README.md).

## Reference

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "33%"}} />
    <col style={{width: "33%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Strategy</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Use Case</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Key Benefit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>Native DP</strong> (<code>--dp-size</code>)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Never</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Easy to understand, not rust based</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>sgl-router-Based DP</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><strong>Production (recommended)</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Cache-aware routing, high availability</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>DPA</strong> (<code>--attn-dp-size N</code>)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>DeepSeek/MLA models</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Eliminates KV cache duplication, improved throughput</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong>DPA + EP</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>DeepSeek MoE models</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Significant throughput improvement vs vanilla TP</td>
    </tr>
  </tbody>
</table>

**Recommended production setup for DeepSeek:**
1. Enable **DPA** for attention layers (`--attn-dp-size 8`)
2. Enable **EP** for MoE layers (`--ep 8 --moe-a2a-backend deepep`)
3. Use **sgl-router** with **cache_aware** policy

**Related documentation:**
- [Expert Parallelism](./expert_parallelism) - DeepEP, Two-Batch Overlap, EPLB
- [sgl-router README](https://github.com/sgl-project/sglang/blob/main/sgl-router/README.md) - router configuration & routing policies
- [Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - 96 GPU deployment guide
