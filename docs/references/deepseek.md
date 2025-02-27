# DeepSeek Usage

SGLang provides several optimizations specifically designed for the DeepSeek model to boost its inference speed. This document outlines current optimizations for DeepSeek. Additionally, the SGLang team is actively developing enhancements for [DeepSeek V3](https://github.com/sgl-project/sglang/issues/2591).

Special thanks to Meituan's Search & Recommend Platform Team and Baseten's Model Performance Team for implementing the model, and DataCrunch for providing GPU resources.

## Launch DeepSeek V3 with SGLang

SGLang is recognized as one of the top engines for [DeepSeek model inference](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3). To run DeepSeek V3/R1 models, the requirements are as follows:

| Weight Type | Configuration |
|------------|-------------------|
| **Full precision FP8**<br>*(recommended)* | 8 x H200 |
| | 8 x MI300X |
| | 2 x 8 x H100/800/20 |
| **Full precision BF16** | 2 x 8 x H200 |
| | 2 x 8 x MI300X |
| | 4 x 8 x H100/800/20 |
| | 4 x 8 x A100/A800 |
| **Quantized weights (AWQ)** | 8 x H100/800/20 |
| | 8 x A100/A800 |

<style>
.md-typeset__table {
  width: 100%;
}

.md-typeset__table table {
  border-collapse: collapse;
  margin: 1em 0;
  border: 2px solid var(--md-typeset-table-color);
  table-layout: fixed;
}

.md-typeset__table th {
  border: 1px solid var(--md-typeset-table-color);
  border-bottom: 2px solid var(--md-typeset-table-color);
  background-color: var(--md-default-bg-color--lighter);
  padding: 12px;
}

.md-typeset__table td {
  border: 1px solid var(--md-typeset-table-color);
  padding: 12px;
}

.md-typeset__table tr:nth-child(2n) {
  background-color: var(--md-default-bg-color--lightest);
}
</style>

Detailed commands for reference:

- [8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)
- [8 x MI300X](https://docs.sglang.ai/references/amd.html#running-deepseek-v3)
- [2 x 8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes)
- [4 x 8 x A100](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)
- [8 x A100 (AWQ)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-8-a100a800-with-awq-quantization)

### Download Weights

If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded. Please refer to [DeepSeek V3]([https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#installation--launch](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only)) offical guide to download the weights.

### Caching `torch.compile`

The DeepSeek series have huge model weights, it takes some time to compile the model with `torch.compile` for the first time if you have added the flag `--enable-torch-compile`. By default, `torch.compile` will automatically cache the FX graph and Triton in `/tmp/torchinductor_root`, which might be cleared according to the [system policy](https://serverfault.com/questions/377348/when-does-tmp-get-cleared). You can export the environment variable `TORCHINDUCTOR_CACHE_DIR` to save compilation cache in your desired directory to avoid unwanted deletion. You can also share the cache with other machines to reduce the compilation time. You may refer to the [PyTorch official documentation](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) and [SGLang Documentation](./torch_compile_cache.md) for more details.

### Launch with One node of 8 H200

Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended). **Note that Deepseek V3 is already in FP8. So we should not run it with any quantization arguments like `--quantization fp8 --kv-cache-dtype fp8_e5m2`.** Also, `--enable-dp-attention` can be useful to improve for Deepseek V3/R1's throughput. Please refer to [Data Parallelism Attention](https://docs.sglang.ai/references/deepseek.html#multi-head-latent-attention-mla-throughput-optimizations) for detail.

### Running examples on Multi-node

- [Serving with two H20*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes).

- [Serving with two H200*8 nodes and docker](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker).

- [Serving with four A100*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes).

## Optimizations

### Multi-head Latent Attention (MLA) Throughput Optimizations

**Description**: [MLA](https://arxiv.org/pdf/2405.04434) is an innovative attention mechanism introduced by the DeepSeek team, aimed at improving inference efficiency. SGLang has implemented specific optimizations for this, including:

- **Weight Absorption**: By applying the associative law of matrix multiplication to reorder computation steps, this method balances computation and memory access and improves efficiency in the decoding phase.

- **Flashinfer MLA Wrapper**: By providing `--enable-flashinfer-mla` argument, the server will use MLA kernels customized by Flashinfer. This optimization can be significant under long context scenarios. More details can be referred to [this document](https://docs.flashinfer.ai/api/mla.html).

- **FP8 Quantization**: W8A8 FP8 and KV Cache FP8 quantization enables efficient FP8 inference. Additionally, we have implemented Batched Matrix Multiplication (BMM) operator to facilitate FP8 inference in MLA with weight absorption.

- **CUDA Graph & Torch.compile**: Both MLA and Mixture of Experts (MoE) are compatible with CUDA Graph and Torch.compile, which reduces latency and accelerates decoding speed for small batch sizes.

Overall, with these optimizations, we have achieved up to a 7x acceleration in output throughput compared to the previous version.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_3/deepseek_mla.svg" alt="Multi-head Latent Attention for DeepSeek Series Models">
</p>

**Usage**: MLA optimization is enabled by default, to disable, use `--disable-mla`.

**Reference**: Check [Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) and [Slides](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf) for more details.

### Data Parallelism Attention

**Description**: This optimization involves data parallelism (DP) for the MLA attention mechanism of DeepSeek Series Models, which allows for a significant reduction in the KV cache size, enabling larger batch sizes. Each DP worker independently handles different types of batches (prefill, decode, idle), which are then synchronized before and after processing through the Mixture-of-Experts (MoE) layer.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg" alt="Data Parallelism Attention for DeepSeek Series Models">
</p>

**Usage**: This optimization is aimed at improving throughput and should be used for scenarios with high QPS (Queries Per Second). Data Parallelism Attention optimization can be enabled by `--enable-dp-attention` for DeepSeek Series Models.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/deepseek_coder_v2.svg" alt="Data Parallelism Attention Performance Comparison">
</p>

**Reference**: Check [Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models).

### Multi Node Tensor Parallelism

**Description**: For users with limited memory on a single node, SGLang supports serving DeepSeek Series Models, including DeepSeek V3, across multiple nodes using tensor parallelism. This approach partitions the model parameters across multiple GPUs or nodes to handle models that are too large for one node's memory.

**Usage**: Check [here](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208) for usage examples.

### Block-wise FP8

**Description**: SGLang implements block-wise FP8 quantization with two key optimizations:

- **Activation**: E4M3 format using per-token-per-128-channel sub-vector scales with online casting.

- **Weight**: Per-128x128-block quantization for better numerical stability.

**Usage**: turn on by default for DeepSeek V3 models.

## FAQ

1. **Question**: What should I do if model loading takes too long and NCCL timeout occurs?

    **Answer**: You can try to add `--dist-timeout 3600` when launching the model, this allows for 1-hour timeout.
