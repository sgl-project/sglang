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

- **Flashinfer MLA Wrapper**: By providing `--enable-flashinfer-mla` argument, the server will use MLA kernels customized by Flashinfer. More details can be referred to [this document](https://docs.flashinfer.ai/api/mla.html). Under long input scenarios, flashinfer mla can improve performance significantly. Optimized triton kernels will be used when flashinfer mla is turned off.

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

### Data Parallelism MLA

**Description**: This optimization is similar to data parallelism attention, but it applies to MLA core instead of the entire attention. Compared with data parallelism attention, it does not additionally increase the memory occupied by weights. It allows for a significant reduction in the KV cache size and enables larger batch sizes.

On an 8×H20 (96GB) node, with data parallelism MLA enabled, we have achieved up to **1.85x(dp=4)** ~ **2.27x(dp=8)** decoding throughput improvement compared to the previous version. And, the number of kvcaches has been increased by **3.5x(dp=4)** and **7x(dp=8)**.

In wgmma computation, the number of rows M is fixed at 64. When TP equals 8 and the head count is 16, both Flash MLA and Flash Infer MLA operate at only 25% of their computing capacity under large batch or long sequence scenarios. If MLA DP = 4 is adopted (in this case, the number of heads is 128 / 2 = 64), low latency can be maintained. If MLA DP = 8 (the number of heads is 128), although the MLA computation time increases by 50% - 100%, a larger KV Cache capacity can be obtained.

**Memory**:

| |default mem=0.95	| dp mla, dp-size=4	| dp mla, dp-size=8 |
| ----|---- | ----|-------------------|
| mem-fraction-static |	0.95	| 0.94	| 0.94 |
| distributed ends. mem usage |	1.81	| 2.09	| 2.09 |
|Load weight end. avail mem/mem usage| 13.29/79.59 |	13.12/79.48	| 13.12/79.48       |
| Capture cuda graph end. avail mem/mem usage |	3.43/0.75 |	3.81/1.26	| 3.42/1.63         |
| max_total_num_tokens  | 130060 | 113450*4 | 113450*8          |

**Performance Comparison**:

| input len | batch size | tp8 latency | output throughput | dp4 latency | output throughput | dp8 latency | output throughput |
|-----------|------------|-------------|-------------------|-------------| ----------------- |-------------|-------------------|
| 256       | 1          | 26.62       | 38.47             | 29.00       | 35.31             | 29.26       | 34.99             |
| 256       | 4          | 27.97       | 146.45            | 29.92       | 136.89            | 30.45       | 134.52            |
| 256       | 8          | 37.27       | 219.78            | 39.18       | 209.06            | 38.95       | 210.34            |
| 256       | 16         | 47.16       | 347.45            | **43.88**   | **373.35**        | 49.46       | 331.27            |
| 256       | 32         | 58.05       | 564.52            | 61.79       | 530.34            | 64.13       | 511.00            |
| 256       | 48         | 78.02       | 630.00            | 79.91       | 615.11            | 86.57       | 567.78            |
| 256       | 64         | 125.38      | 522.70            | **91.21**   | **718.55**        | 90.01       | 728.09            |
| 256       | 96         | 156.59      | 627.79            | **101.20**  | **971.42**        | 105.25      | 934.05            |
| 256       | 128        | 219.12      | 598.16            | **112.57**  | **1164.37**       | 115.85      | 1131.37           |
| 256       | 160        |             |                   |             |                   | **131.08**  | **1249.90**       |
| 256       | 192        |             |                   |             |                   | **137.61**  | **1428.69**       |
| 1024      | 1          | 26.89       | 38.08             | 29.13       | 35.15             | 29.74       | 34.43             |
| 1024      | 4          | 28.73       | 142.55            | 29.77       | 137.59            | 31.06       | 131.87            |
| 1024      | 8          | 39.29       | 208.52            | 39.66       | 206.55            | 38.12       | 214.93            |
| 1024      | 16         | 47.60       | 344.23            | 50.66       | 323.41            | 49.33       | 332.11            |
| 1024      | 32         | 63.61       | 515.12            | 65.86       | 497.56            | 71.41       | 458.86            |
| 1024      | 48         | 85.67       | 573.71            | **84.92**   | **578.80**        | 88.22       | 557.18            |
| 1024      | 64         | 133.73      | 490.07            | **93.03**   | **704.48**        | 97.82       | 669.96            |
| 1024      | 96         | 169.32      | 580.57            | **111.41**  | **882.40**        | 116.40      | 844.51            |
| 1024      | 128        | 236.02      | 555.34            | **122.51**  | **1069.86**       | 131.31      | 998.22            |
| 1024      | 160        |             |                   |             |                   | **150.28**  | **1090.25**       |
| 1024      | 192        |             |                   |             |                   | **166.73**  | **1179.17**       |
| 4096      | 1          | 27.78       | 36.86             | 30.77       | 33.28             | 32.30       | 31.71             |
| 4096      | 4          | 31.52       | 129.93            | 32.94       | 124.33            | 35.24       | 116.24            |
| 4096      | 8          | 44.91       | 182.40            | 45.91       | 178.42            | **41.51**   | **197.36**        |
| 4096      | 16         | 60.06       | 272.80            | 60.46       | 270.98            | **59.85**   | **273.73**        |
| 4096      | 32         | 125.42      | 261.27            | 90.94       | 360.34            | **88.19**   | **371.58**        |
| 4096      | 48         | 159.55      | 308.07            | 117.70      | 417.59            | **119.88**  | **410.02**        |
| 4096      | 64         | 220.50      | 297.22            | 143.10      | 457.98            | **142.32**  | **460.49**        |
| 4096      | 96         | 321.90      | 305.38            | 196.73      | 499.68            | **182.11**  | **539.81**        |
| 4096      | 128        | 415.31      | 315.60            | 280.51      | 467.27            | **218.63**  | **599.53**        |
| 4096      | 160        |             |                   |             |                   | **264.45**  | **619.55**        |
| 4096      | 192        |             |                   |             |                   | **322.20**  | **610.20**        |

**Usage**:

***Restrictions***:
- The currently implemented custom all-to-all operation is only restricted within a single node. Cross-node is not supported for the time being.
- The `chunked-prefill-size` needs to be less than or equal to `16384`.

On an 8×H20 (96GB) node, on top of `--enable-flashinfer-mla`, additional parameters:

```shell
# dp=4:
--max-running-requests 128 --mem-fraction-static 0.94 --enable-dp-mla
```

```shell
# dp=8:
--max-running-requests 192 --mem-fraction-static 0.94 --enable-dp-mla --dp-size=8 --cuda-graph-max-bs 192
```

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
- Currently when using flashinfer mla wrapper (`--enable-flashinfer-mla`) and speculative decoding together, the `--speculative-eagle-topk` parameter should be set to `1`.
- To enable DeepSeek MTP for large batch sizes (>32), there are some parameters should be changed (Reference [this discussion](https://github.com/sgl-project/sglang/issues/4543#issuecomment-2737413756)):
  - Adjust `--max-running-requests` to a larger number. The default value is `32` for MTP. For larger batch sizes, you should increase this value beyond the default value.
  - Set `--cuda-graph-bs`. It's a list of batch sizes for cuda graph capture. The default captured batch sizes for speculative decoding is set [here](https://github.com/sgl-project/sglang/blob/49420741746c8f3e80e0eb17e7d012bfaf25793a/python/sglang/srt/model_executor/cuda_graph_runner.py#L126). You can include more batch sizes into it.


### Reasoning Content for DeepSeek R1

See [Separate Reasoning](https://docs.sglang.ai/backend/separate_reasoning.html).

## FAQ

1. **Question**: What should I do if model loading takes too long and NCCL timeout occurs?

    **Answer**: You can try to add `--dist-timeout 3600` when launching the model, this allows for 1-hour timeout.
