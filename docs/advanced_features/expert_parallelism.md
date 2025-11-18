# Expert Parallelism

## Overview and Rationale

Expert Parallelism (EP) is a vital parallelization strategy in SGLang for high-performance serving of MoEs, such as DeepSeek-V3. SGLang's EP overcomes two key challenges inherent to MoE serving:

### Memory and Communication Bottlenecks

MoE models' vast expert weights exceed single GPU memory, and routing tokens across devices incurs complex All-to-All communication overhead. EP partitions expert weights across GPUs to eliminate memory bottlenecks. SGLang integrates [DeepEP](https://github.com/deepseek-ai/DeepEP), a specialized communication library, with Two-Batch Overlap (TBO) to aggressively overlap All-to-All communication latency with expert computation, minimizing overhead and boosting throughput.

### Dynamic Expert Load Imbalance

Real-world inference traffic often leads to uneven expert usage. SGLang's Expert Parallelism Load Balancer ([EPLB](https://github.com/deepseek-ai/eplb)) dynamically monitors expert usage, reassigns experts to balance workload, and maximizes GPU utilization for consistent high performance.

## 3. Synergy with Prefill-Decode (PD) Disaggregation

In large-scale MoE deployments, Expert Parallelism (EP) synergizes with Prefill-Decode (PD) Disaggregation, SGLang's architectural design that separates LLM inference into two distinct phases:

-   **Prefill Phase**: The computation-intensive stage for processing the initial input prompt, optimized for high throughput.
-   **Decode Phase**: The memory-intensive stage for iterative token generation, optimized for low latency.

This separation allows SGLang to apply tailored parallelization strategies:

-   The Prefill Server can be tuned for throughput, potentially utilizing larger EP groups.
-   The Decode Server can be tuned for latency, ensuring a fast Time-to-First-Token crucial for user experience.

This combined approach enables SGLang to achieve state-of-the-art performance, replicating 52.3k input tokens/s and 22.3k output tokens/s per node in multi-node MoE benchmarks. For more details, refer to [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

## 4. Practical Deployment Guide

To enable Expert Parallelism in SGLang, you use specific flags in your server launch command. EP is automatically calculated in conjunction with Data Parallelism (DP) for the FFN layers.

### Example Deployment Command

The following command deploys an MoE model across 8 GPUs, enabling EP, DP Attention, and the EPLB for dynamic load balancing:

```bash
# Assuming an 8-GPU node configuration
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-67B \ # Specifies the path to the MoE model
  --ep-size 8 \ # Sets the Expert Parallelism size to 8 GPUs
  --enable-dp-attention \ # Activates Data Parallelism for Attention layers
  --enable-eplb \ # Enables the Expert Parallelism Load Balancer
  --expert-distribution-recorder-mode stat \ # Configures EPLB to use aggregated statistics
  --trust-remote-code # Required for custom model architectures
```
### Key Configuration Flags

| Flag | Description |
|---|---|
| --ep-size N | Expert Parallelism size. The number of GPUs over which the expert weights are distributed. This is often the total number of GPUs available in a single-node setup. |
| --enable-eplb | Activates the Expert Parallelism Load Balancer (EPLB) to dynamically re-map expert locations for mitigating workload skew and maximizing utilization. |
| --expert-distribution-recorder-mode [stat, per_pass, per_token] | Defines how EPLB collects expert usage data. **stat** (default, aggregated moving average for stable load balancing), **per_pass** (finer-grained data per forward pass), **per_token** (most granular data per token, higher overhead). |
| --enable-dp-attention | Enables Data Parallelism (DP) for the Multi-head Latent Attention (MLA) layers, a specific attention mechanism in DeepSeek models. This helps eliminate KV cache duplication (where identical Key-Value caches are unnecessarily stored across multiple GPUs) and reduces memory overhead. |
| --trust-remote-code | Required for models that use custom architectures like DeepSeek. |
