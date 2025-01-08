# DeepSeek Model Optimizations in SGLang

SGLang provides several optimizations specifically designed for the DeepSeek model to boost its inference speed. This document outlines current optimizations for DeepSeek. Additionally, the SGLang team is actively developing enhancements for [DeepSeekV3](https://github.com/sgl-project/sglang/issues/2591).


## Multi-head Latent Attention (MLA) Throughput Optimizations

**Description**: [MLA](https://arxiv.org/pdf/2405.04434) is an innovative attention mechanism introduced by the DeepSeek team, aimed at improving inference efficiency. SGLang has implemented specific optimizations for this, including:

- **Weight Absorption**: Absorbing Key matrix into query and value matrix into output
- **MLA Triton optimization**  Optimize memory access for MLA decoding.
- **Grouped Decoding Kernels**
- **FP8 Batched Matrix Multiplication (MatMul)**: Speed up Mixture-of-Experts (MoE) computation by lower precision
- **FP8 KV Cache Quantization**: Reduces memory usage and increases throughput

**Usage**: MLA optimization is enabled by defalut, to disable, use `--disable-mla`

**Reference**: [Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations)

## Data Parallelism Attention

**Description**: This optimization involves data parallelism (DP) for the attention mechanism of DeepSeek models, which allows for a significant reduction in the KV cache size, enabling larger batch sizes. Each DP worker independently handles different types of batches (prefill, decode, idle), which are then synchronized before and after processing through the Mixture-of-Experts (MoE) layer.
![Data Parallelism Attention for DeepSeek Models](https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg)

**Usage**: Data Parallelism Attention optimization can be enabeld by `--enable-dp-attention` for DeepSeek models

**Reference**: [Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models)

## Multi Node Tensor Parallelism
**Description**: For users with limited memory on a single node, SGLang supports serving DeepSeek models, including DeepSeek V3, across multiple nodes using tensor parallelism. This approach partitions the model parameters across multiple GPUs or nodes to handle models that are too large for one node's memory.

**Usage**: [Example](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208)
