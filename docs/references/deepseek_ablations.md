# DeepSeek Optimization Ablations

## Overview

As of 2024-03-04, SGLang provides the following optimizations for DeepSeek V3/R1 models:

| Name                                        | Description                                                                                                                                                                                                                                     | Enabled by Default | Enable/Disable Argument                                                                                                                                   |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| MLA Optimization                            | SGLang custom tailored optimizations, including<br>  - *Weight Absorption*,<br>- *Flashinfer MLA Wrapper*,<br> - *FP8 Quantization*,<br> - *CUDA Graph & Torch.compile suuport*                                                                 | ✅               | `--disable-mla`                                                                                                                                           |
| CUDA Graph                                  | Capturing and replaying entire sequences of GPU operations as a single graph, thereby reducing kernel launch overhead and synchronization delays                                                                                                | ✅               | `--disable-cuda-graph`                                                                                                                                    |
| Torch Compile                               | Dynamically converting PyTorch models into optimized execution graphs, reducing runtime overhead and enhancing GPU performance                                                                                                                  | ❌              | `--enable-torch-compile`                                                                                                                                 |
| Radix Cache                                 | Organizes the KV cache in a radix tree, enabling automatic detection and reuse of shared prompt prefixes across multiple generation calls, thereby reducing redundant computations                                                              | ✅               | `--disable-radix-cache`                                                                                                                                   |
| Flashinfer MLA                              | Multi-latent Attention implemented by Flashinfer that replaces the default Triton backend                                                                                                                                                       | ❌              | `--enable-flashinfer-mla`                                                                                                                                 |
| Speculative Decoding (`Next-N`)             | Dynamically generating a context-aware draft token tree with a smaller, well-calibrated model and then verifying these tokens in parallel with the original LLM, thereby reducing expensive forward passes while preserving output quality.     | ❌              | `--speculative-algorithm`,<br> `--speculative-draft`,<br> `--speculative-num-steps`,<br> `--speculative-eagle-topk`,<br> `--speculative-num-draft-tokens` |
| Tensor Parallelism (`tp`)                   | Splitting the heavy tensor operations—such as the matrix multiplications in self-attention and feedforward layers—across multiple GPUs, thereby lowering the per-device memory burden and enabling simultaneous computation for reduced latency | ✅ (=1)         | `--tp-size`                                                                                                                                               |
| Expert Parallelism (`EP-MoE`)               | Distributing the computation of different expert subnetworks across multiple devices, thereby reducing memory constraints and communication overhead while enabling simultaneous, efficient processing of input tokens.                         | ❌              | `--enable-ep-moe`,<br> `--ep-size`                                                                                                                        |
| Data Parallelism Attention (`DP-Attention`) | Partitioning the MLA attention across DP workers—each handling independent prefill, decode, and idle batches—to significantly reduce per-worker KV cache size and enable larger, more efficient batch processing                                | ❌              | `--enable-dp-attention`                                                                                                                                   |

## General Advice

* Speculative Decoding is great for small concurrency (less than 32), but its performance degrades quickly as the concurrency increases.
* `CUDA Graph` boosts inference performance significantly, at the cost of increased memory usage. Sometimes it's a good trade-off to disable `CUDA Graph` to further increase concurrency to get better throughput.
* `DP-Attention` is a must for large concurrency (greater than 256), but it hurts per-request decoding speed.

## Known Issues

* Speculative Decoding is not compatible with:
  - `Flashinfer-mla`
  - `Radix Cache`
  - `DP-Attention`
  - Both `CUDA Graph` and `Torch Compile` enabled simultaneously
* `EP-MoE` is not supported with both `CUDA Graph` and `Torch Compile` enabled
* To run `DP-Attention` with large concurrency, you must first run a warmup phase with small concurrency (e.g. `bs=16`, `total req=32`) to avoid CUDA out of memory error.

## Optimization Ablations

### Test Environment

* SGLang version: 0.4.3.post2@[110e006](https://github.com/sgl-project/sglang/commit/110e0066735a3bd431c2640ae168fc040d7c0806)
* Flashinfer version: 0.2.2.post1
* Hardware: 2 nodes of H20 (AMD EPYC 9K84 * 2, 2.20 TiB memory, 8 * H20 96GiB each)
* Model: DeepSeek-R1
* Model Max Length: 3200 (modified in both model and NextN's `tokenizer_config.json`)
* CUDA Version: 12.2
* Operating System: Rocky Linux release 9.2 (Blue Onyx)

### Single Query Performance

* Test query: `一个汉字具有左右结构，左边是木，右边是乞。这个字是什么？只需回答这个字即可。`
* Expected output: `杚`[1]

| Runnable           | TPS@1[2] | Torch Compile | Cuda Graph | Radix Cache | Flashinfer-mla | Next-N | EP-MoE | DP-Attention |
|--------------------|-----------|---------------|------------|-------------|----------------|--------|--------|--------------|
| ✅                 | 37.0[11] |       ✅       |      ✅     |      ✅      |        ➖       |    ➖   |    ➖   |       ➖      |
| ✅                 | 33.6     |       ✅       |      ✅     |      ✅      |        ✅       |    ➖   |    ➖   |       ➖      |
| ✅                 | 19.1     |       ✅       |      ✅     |      ✅      |        ✅       |    ➖   |    ➖   |       ✅      |
| ❌ [3]            | N/A      |       ✅       |      ✅     |      ✅      |        ✅       |    ➖   |    ✅   |       ✅      |
| ❌ [3]            | N/A      |       ✅       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ➖      |
| ✅                 | 6.5      |       ✅       |      ➖     |      ✅      |        ➖       |    ➖   |    ✅   |       ➖      |
| ✅                 | 24.4     |       ➖       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ➖      |
| ✅                 | 23.6     |       ➖       |      ✅     |      ✅      |        ✅       |    ➖   |    ✅   |       ➖      |
| ✅                 | 13.0     |       ➖       |      ➖     |      ➖      |        ➖       |    ✅   |    ✅   |       ➖      |
| ❌ [4] <br> ✅ [5] | 41.0     |       ➖       |      ✅     |      ➖      |        ➖       |    ✅   |    ✅   |       ➖      |
| ❌ [3]            | N/A      |       ✅       |      ✅     |      ➖      |        ➖       |    ✅   |    ✅   |       ➖      |
| ✅ [5]            | 16.0     |       ➖       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ✅      |
| ❌ [3]            | N/A      |       ✅       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ✅      |
| ✅ [5]            | 15.8     |       ➖       |      ✅     |      ✅      |        ✅       |    ➖   |    ✅   |       ✅      |
| ❌ [3]            | N/A      |       ➖       |      ✅     |      ✅      |        ➖       |    ✅   |    ✅   |       ✅      |
| ❌ [6]            | N/A      |       ➖       |      ➖     |      ➖      |        ➖       |    ✅   |    ➖   |       ✅      |

### Offline Batch Performance

* Test bench: ThreadPool with AsyncOpenAI client
* Avg input length = 760 tokens
* Avg output length = 460 tokens

| Runnable | Torch Compile | Cuda Graph  | Radix Cache  | Flashinfer-mla | Next-N |  EP-MoE  | DP-Attn | Client Concurrency [7]        | Avg Throughput<br><sub><sup>(p+d, token/s)</sup></sub> [8]                 | Per-req Throughput<br><sub><sup>(d, token/s)</sup></sub> [9] |   Total Req    | Max-running-req [10] |
|----------|---------------|-------------|--------------|----------------|---------|---------|--------------|--------------------------------|----------------------------------------------------|-----------------------------------------------|---------------------|--------------------------|
| ✅       |       ✅       |      ✅     |      ✅      |        ✅       |    ➖   |    ➖   |       ➖      | 768                            | 3909.04                                            | 3.28                                          | 1024                | 768                      |
| ✅       |       ✅       |      ✅     |      ✅      |        ✅       |    ➖   |    ➖   |       ✅      | 16<br>512<br>768               | 306.18<br>4329.32<br>5457.14                       | 12.96<br>5.69<br>5.38                         | 32<br>1024<br>1024  | 768                      |
| ❌[3]  |       ✅       |      ✅     |      ✅      |        ✅       |    ➖   |    ✅   |       ✅      | N/A                            | N/A                                                | N/A                                           | N/A                 | 768                      |
| ❌[3]  |       ✅       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ➖      | N/A                            | N/A                                                | N/A                                           | N/A                 | 768                      |
| ✅       |       ✅       |      ➖     |      ✅      |        ➖       |    ➖   |    ✅   |       ➖      | 768                            | 2100.85                                            | 2.79                                          | 1024                | 768                      |
| ✅       |       ➖       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ➖      | 256<br>512<br>768              | 2134.99<br>3842.52<br>3453.49                      | 5.16<br>4.05<br>3.15                          | 512<br>1024<br>1024 | 768                      |
| ✅       |       ➖       |      ✅     |      ✅      |        ✅       |    ➖   |    ✅   |       ➖      | 256<br>512<br>768              | 2220.56<br>3583.08<br>3556.76                      | 5.12<br>4.07<br>3.52                          | 512<br>1024<br>1024 | 768                      |
| ✅       |       ➖       |      ➖     |      ➖      |        ➖       |    ✅   |    ✅   |       ➖      | N/A                            | N/A                                                | N/A                                           | N/A                 | 768                      |
| ✅[5]  |       ➖       |      ✅     |      ➖      |        ➖       |    ✅   |    ✅   |       ➖      | 16<br>32                       | 732.22<br>1227.72                                  | 19.93<br>15.14                                | 256                 | 768                      |
| ❌[3]  |       ✅       |      ✅     |      ➖      |        ➖       |    ✅   |    ✅   |       ➖      | N/A                            | N/A                                                | N/A                                           | N/A                 | 768                      |
| ✅[5]  |       ➖       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ✅      | 16<br>128<br>256<br>512<br>768 | 862.10<br>1598.17<br>2664.40<br>4098.18<br>❌[4] | 9.20<br>8.22<br>6.70<br>5.48<br>❌[4]        | 128<br>256<br>512<br>1024<br>1024 | 768        |
| ❌[3]  |       ✅       |      ✅     |      ✅      |        ➖       |    ➖   |    ✅   |       ✅      | N/A                            | N/A                                                | N/A                                           | N/A                 | 768                      |
| ✅[5]  |       ➖       |      ✅     |      ✅      |        ✅       |    ➖   |    ✅   |       ✅      | 16<br>512<br>768               | 406.29<br>3633.20<br>❌[4]                       | 12.29<br>5.74<br>❌[4]                       | 32<br>1024<br>1024  | 768                     |
| ❌[3]  |       ➖       |      ✅     |      ➖      |        ➖       |    ✅   |    ✅   |       ✅      | N/A                            | N/A                                                | N/A                                           | N/A                 | 768                      |
| ❌[6]  |       ➖       |      ➖     |      ➖      |        ➖       |    ✅   |    ➖   |       ✅      | N/A                            | N/A                                                | N/A                                           | N/A                 | 768                      |

[^1]: DeepSeek-R1 cannot give the correct output if quantization is used or has precision issues (fixed in [b110084](https://github.com/sgl-project/sglang/commit/b110084654a1986f0148901190e2f280c605476f))
[^2]: TPS@1 (Tokens Per Second for single request) is read directly from SGLang's logging.
[^3]: CUDA error at graph capture
[^4]: CUDA out of memory
[^5]: Requires setting `mem-fraction-static=0.7` to avoid OOM errors
[^6]: TypeError: object of type 'NoneType' has no len()
[^7]: All statistics are collected from the test bench. Token count is calculated using the same tokenizer used in inference.
[^8]: Average Throughput(prefill+decode, token/s) = (total tokens)/(total time)
[^9]: Average Decoding Throughput = (sum of (output tokens/duration) for each successful request)/(number of successful requests)
[^10]: The maximum number of requests to run concurrently at a SGLang backend, controlled by `--max-running-requests`
[^11]: Tested by [Lzhang-Hub](https://github.com/sgl-project/sglang/issues/3956#issuecomment-2700514223)
