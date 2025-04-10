# DeepSeek Usage

SGLang provides several optimizations specifically designed for the DeepSeek model to boost its inference speed. This document outlines current optimizations for DeepSeek. Additionally, the SGLang team is actively developing enhancements for [DeepSeek V3](https://github.com/sgl-project/sglang/issues/2591).

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
| **Quantized weights (int8)** | 16 x A100/800 |
| | 32 x L40S |

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
- [16 x A100 (int8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-16-a100a800-with-int8-quantization)
- [32 x L40S (int8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-32-l40s-with-int8-quantization)

### Download Weights

If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded. Please refer to [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only) official guide to download the weights.

### Caching `torch.compile`

The DeepSeek series have huge model weights, it takes some time to compile the model with `torch.compile` for the first time if you have added the flag `--enable-torch-compile`. You can refer [here](https://docs.sglang.ai/backend/hyperparameter_tuning.html#try-advanced-options) to optimize the caching of compilation results, so that the cache can be used to speed up the next startup.
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

- **MLA Attention Backends**: Currently SGLang supports different optimized MLA attention backends, including FlashAttention3, [Flashinfer](https://docs.flashinfer.ai/api/mla.html) and Triton backends. It can be set with `--attention-backend` argument.

- **FP8 Quantization**: W8A8 FP8 and KV Cache FP8 quantization enables efficient FP8 inference. Additionally, we have implemented Batched Matrix Multiplication (BMM) operator to facilitate FP8 inference in MLA with weight absorption.

- **CUDA Graph & Torch.compile**: Both MLA and Mixture of Experts (MoE) are compatible with CUDA Graph and Torch.compile, which reduces latency and accelerates decoding speed for small batch sizes.

Overall, with these optimizations, we have achieved up to **7x** acceleration in output throughput compared to the previous version.

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

With data parallelism attention enabled, we have achieved up to **1.9x** decoding throughput improvement compared to the previous version.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/deepseek_coder_v2.svg" alt="Data Parallelism Attention Performance Comparison">
</p>

**Usage**:
- This optimization is aimed at improving throughput and should be used for scenarios with high QPS (Queries Per Second). It can be enabled by `--enable-dp-attention` for DeepSeek models.
- Since v0.4.4, DP and TP attention can be flexibly combined. For example, to deploy DeepSeek-V3/R1 on 2 node with 8*H100, you can specify `--tp 16` and `--dp 2`, which means for attention part there are 2 DP groups, and in each DP group there are 8 TP groups.

**Reference**: Check [Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models).

### Multi Node Tensor Parallelism

**Description**: For users with limited memory on a single node, SGLang supports serving DeepSeek Series Models, including DeepSeek V3, across multiple nodes using tensor parallelism. This approach partitions the model parameters across multiple GPUs or nodes to handle models that are too large for one node's memory.

**Usage**: Check [here](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208) for usage examples.

### Block-wise FP8

**Description**: SGLang implements block-wise FP8 quantization with two key optimizations:

- **Activation**: E4M3 format using per-token-per-128-channel sub-vector scales with online casting.

- **Weight**: Per-128x128-block quantization for better numerical stability.

**Usage**: Turn on by default for DeepSeek V3 models.

### Multi-token Prediction
**Description**: SGLang implements DeepSeek V3 Multi-Token Prediction (MTP) based on [EAGLE speculative decoding](https://docs.sglang.ai/backend/speculative_decoding.html#EAGLE-Decoding). With this optimization, the decoding speed can be improved by **1.8x** for batch size 1 and **1.5x** for batch size 32 respectively on H200 TP8 setting.

**Usage**:
Add arguments `--speculative-algorithm`, `--speculative-draft-model-path`,
`--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` to enable this feature. For example:
```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3-0324 --speculative-algorithm EAGLE --speculative-draft-model-path lmsys/DeepSeek-V3-0324-NextN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --trust-remote-code --tp 8
```
- The draft model are available at huggingface: [lmsys/DeepSeek-V3-0324-NextN](https://huggingface.co/lmsys/DeepSeek-V3-0324-NextN), [lmsys/DeepSeek-R1-NextN](https://huggingface.co/lmsys/DeepSeek-R1-NextN). It can also be exported from original DeepSeek-V3/R1 model with [export_deepseek_nextn.py](https://github.com/sgl-project/sglang/blob/main/scripts/export_deepseek_nextn.py) script.
- The best configuratin for `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` can be searched with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) script for given batch size. The minimum configuration is `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`, which can achieve speedup for larger batch sizes.
When using FlashInfer MLA wrapper (`--attention-backend flashinfer`) with speculative decoding, set the `--speculative-eagle-topk` parameter to `1`. The FlashAttention 3 backend also only supports `--speculative-eagle-topk 1`.
- To enable DeepSeek MTP for large batch sizes (>32), there are some parameters should be changed (Reference [this discussion](https://github.com/sgl-project/sglang/issues/4543#issuecomment-2737413756)):
  - Adjust `--max-running-requests` to a larger number. The default value is `32` for MTP. For larger batch sizes, you should increase this value beyond the default value.
  - Set `--cuda-graph-bs`. It's a list of batch sizes for cuda graph capture. The default captured batch sizes for speculative decoding is set [here](https://github.com/sgl-project/sglang/blob/49420741746c8f3e80e0eb17e7d012bfaf25793a/python/sglang/srt/model_executor/cuda_graph_runner.py#L126). You can include more batch sizes into it.


### Reasoning Content for DeepSeek R1

See [Separate Reasoning](https://docs.sglang.ai/backend/separate_reasoning.html).

## FAQ

1. **Question**: What should I do if model loading takes too long and NCCL timeout occurs?

    **Answer**: You can try to add `--dist-timeout 3600` when launching the model, this allows for 1-hour timeout.
