# Expert Parallelism

Expert Parallelism (EP) is a vital parallelization strategy in SGLang for high-performance serving of MoEs, such as DeepSeek-V3. SGLang's EP overcomes two key challenges inherent to MoE serving:

1. MoE models' vast expert weights exceed single GPU memory, and routing tokens across devices incurs complex All-to-All communication overhead. EP partitions expert weights across GPUs to eliminate memory bottlenecks. SGLang integrates [DeepEP](https://github.com/deepseek-ai/DeepEP), a specialized communication library, with Two-Batch Overlap (TBO) to aggressively overlap All-to-All communication latency with expert computation, minimizing overhead and boosting throughput.
2. Real-world inference traffic often leads to uneven expert usage. SGLang's Expert Parallelism Load Balancer ([EPLB](https://github.com/deepseek-ai/eplb)) dynamically monitors expert usage, reassigns experts to balance workload, and maximizes GPU utilization for consistent high performance.

##  Synergy with Prefill-Decode (PD) Disaggregation

SGLang synergizes Expert Parallelism with Prefill-Decode Disaggregation by applying tailored parallelization strategies:

-   The Prefill Server is tuned for throughput, utilizing larger EP groups.
-   The Decode Server is tuned for latency, with EP ensuring a fast Time-to-First-Token, crucial for user experience.

This combined approach enables SGLang to achieve state-of-the-art performance, replicating 52.3k input tokens/s and 22.3k output tokens/s per node in multi-node MoE benchmarks. For more details, refer to [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

## Deployment Guide

To enable Expert Parallelism in SGLang, use the relevant flags in your server launch command, which are fully documented in the [Server Arguments page](../advanced_features/server_arguments.md#expert-parallelism). 

### Example

The following command deploys an MoE model across 8 GPUs, enabling EP, DP Attention, and the EPLB for dynamic load balancing:

#### 1. deepseek-ai/DeepSeek-V3

```bash
# Assuming an 8-GPU node configuration
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --ep-size 8 \ # Sets the Expert Parallelism size to 8 GPUs
  --enable-dp-attention \ # Activates Data Parallelism for Attention layers
  --enable-eplb \ # Enables the Expert Parallelism Load Balancer
  --expert-distribution-recorder-mode stat \ # Configures EPLB to use aggregated statistics
  --trust-remote-code
```

#### 2. moonshotai/Kimi-K2-Thinking

```bash
# Assuming an 8-GPU node configuration
python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2-Thinking \
  --ep-size 8 \ # Sets the Expert Parallelism size to 8 GPUs
  --enable-dp-attention \ # Activates Data Parallelism for Attention layers
  --enable-eplb \ # Enables the Expert Parallelism Load Balancer
  --expert-distribution-recorder-mode stat \ # Configures EPLB to use aggregated statistics
  --trust-remote-code
```

#### 3. Qwen/Qwen3-VL-235B-A22B-Instruct

```bash
# Assuming an 8-GPU node configuration
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct \
  --ep-size 8 \ # Sets the Expert Parallelism size to 8 GPUs
  --enable-dp-attention \ # Activates Data Parallelism for Attention layers
  --enable-eplb \ # Enables the Expert Parallelism Load Balancer
  --expert-distribution-recorder-mode stat \ # Configures EPLB to use aggregated statistics
  --trust-remote-code
```

**Note:** These examples are demonstrated on a single H200 GPU node.

