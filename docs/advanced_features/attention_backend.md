# Attention Backend

SGLang supports a large variety of attention backends. Each of them has different pros and cons.
You can test them according to your needs.

```{important}
Selecting an optimal attention backend is crucial for maximizing your performance. Different backends excel in various scenarios, so choose based on your model, hardware, and use case. Not all backends are supported on all platforms and model architectures.

If you don't specify `--attention-backend`, SGLang makes a best effort to automatically select the most performant backend based on your hardware and model architecture.
```

## Support Matrix

The support matrix is split into two parts: MHA (standard attention) and MLA (multi-head latent attention). For an explanation of the key differences between MHA and MLA, please see the [SGLang documentation on DeepSeek MLA](../basic_usage/deepseek_v3.md#multi-head-latent-attention-mla-throughput-optimizations) and the original [DeepSeek MLA paper](https://arxiv.org/pdf/2405.04434).

### MHA Backends

| **Backend**                     | **Page Size > 1 (native)** | **FP8 KV Cache** | **FP4 KV Cache** | **Spec topk=1** | **Spec topk>1** | **Sliding Window** | **MultiModal** |
|---------------------------------|-----------------------------|------------------|-----------------|-----------------|-----------------|--------------------|----------------|
| **FlashInfer**                  | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ❌             |
| **FA3 (FlashAttention 3)**      | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ✅             |
| **FA4 (FlashAttention 4)**      | 128                         | ❌               | ✅              | ❌              | ❌              | ❌                 | ❌             |
| **Triton**                      | ❌                          | ❌               | ✅              | ✅              | ✅              | ✅                 | ✅             |
| **Torch Native (SDPA)**         | ❌                          | ❌               | ✅              | ❌              | ❌              | ❌                 | ✅             |
| **FlexAttention (PyTorch)**     | ❌                          | ❌               | ✅              | ❌              | ❌              | ❌                 | ❌             |
| **TRTLLM MHA**                  | 16, 32 or 64                | ✅               | ✅              | ✅              | ❌              | ✅                 | ❌             |
| **Dual Chunk FlashAttention**   | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **AITER (ROCm)**                | ✅                          | ❌               | ❌              | ✅              | ✅              | ❌                 | ✅             |
| **Wave (ROCm)**                 | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **Ascend (NPU)**                | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ✅             |
| **Intel XPU**                   | ✅                          | ❌               | ❌              | ❌              | ❌              | ✅                 | ❌             |
| **Intel AMX (CPU)**             | ❌                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |

### MLA Backends

| **Backend**                | **Native Page Sizes**     | **FP8 KV Cache** | **FP4 KV Cache** | **Chunked Prefix Cache** | **Spec topk=1** | **Spec topk>1** |
|----------------------------|---------------------------|------------------|------------------|--------------------------|-----------------|-----------------|
| **FlashInfer MLA**         | 1                         | ❌               | ✅               | ✅                       | ✅              | ❌              |
| **FlashMLA**               | 64                        | ✅               | ❌               | ✅                       | ✅              | ❌              |
| **Cutlass MLA**            | 128                       | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **TRTLLM MLA (Blackwell)** | 32 or 64                  | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **FA3 (FlashAttention 3)** | n/a                       | ❌               | ❌               | ✅                       | ✅              | ⚠️ (page_size=1 only) |
| **Triton**                 | n/a                       | ❌               | ❌               | ❌                       | ✅              | ⚠️ (page_size=1 only) |
| **FA4**                    | 1                         | ❌               | ✅               | ❌                       | ❌              | ❌              |
| **Ascend MLA (NPU)**       | 128                       | ❌               | ❌               | ❌                       | ❌              | ❌              |

```{note}
Multimodal attention is selected by `--mm-attention-backend`. The "MultiModal" column indicates whether a corresponding multimodal implementation exists for that backend family.
```

```{note}
- FlashAttention 4 is prefill-only for now.
- NSA is specifically designed for [DeepSeek V3.2 DSA](https://lmsys.org/blog/2025-09-29-deepseek-V32/).
```

```{note}
For the KV4 FA4 scenario, FA4 requires using a different --decode-attention-backend to run. Except for trtllm_mha being incompatible with FA4, all other decode backends behave as shown in the table.
```

```{tip}
Speculative decoding topk: `topk` is the number of draft tokens sampled per step from the draft model. `topk = 1` follows classic EAGLE; `topk > 1` explores multiple branches and requires backend support in both draft and verification paths.
```

```{tip}
Page size controls how many tokens are grouped into a KV cache block. For the prefix cache to take effect, the number of tokens must fill at least one complete page. For example, if your prompt is only 32 tokens and `page_size = 64`, it won't fill a complete page and cannot be matched in the prefix cache (pages cannot be padded). With 65 tokens and `page_size = 64`, only the first page of 64 tokens will be cached and matched; the remaining 1 token is discarded. Use `page_size = 1` for maximum prefix reuse (token-level matching).
```

Many backends that do not natively operate on pages can emulate `page_size > 1` at the wrapper layer by expanding page tables to per-token indices. The "Page Size > 1 (native)" column indicates true in-kernel paging. Some backends require fixed native page sizes and cannot be reduced/emulated differently: TRTLLM MHA (16/32/64), TRTLLM MLA (32/64), FlashMLA (64), Cutlass MLA (128), Ascend (128).

MLA page-size constraints:
- FlashInfer MLA: page_size = 1.
- FlashMLA: page_size = 64.
- Cutlass MLA: page_size = 128.
- TRTLLM MLA: page_size ∈ {32, 64}.

### Hybrid attention (different backends for prefill vs decode) (Experimental)

```{warning}
Hybrid attention is an experimental feature.
```

You can mix-and-match attention backends for prefill and decode. This is useful when one backend excels at prefill and another excels at decode. For the implementation details, please see `python/sglang/srt/layers/attention/hybrid_attn_backend.py`.

```bash
# Example: Prefill with FA4, Decode with TRTLLM MLA (Blackwell)
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-FP4 \
  --tp 8 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --quantization modelopt_fp4 \
  --prefill-attention-backend fa4
```

#### Speculative decoding with hybrid attention

Hybrid attention also works with speculative decoding. The backend used for draft decoding and target verification depends on `--speculative-attention-mode`:

- `--speculative-attention-mode decode` (recommended): draft/verify use the decode backend.
- `--speculative-attention-mode prefill` (default): draft/verify use the prefill backend.

Constraints when combining hybrid attention with speculative decoding:

- If any attention backend is `trtllm_mha`, speculative decoding supports only `--speculative-eagle-topk 1`.
- For paged MHA backends with `--page-size > 1` and `--speculative-eagle-topk > 1`, only `flashinfer` is supported.
- CUDA Graph: the decode backend is always captured; the prefill backend is captured only when `--speculative-attention-mode prefill`.


```{tip}
If you set only one of `--prefill-attention-backend` or `--decode-attention-backend`, the unspecified phase inherits `--attention-backend`.
If both are specified and differ, SGLang automatically enables a hybrid wrapper to dispatch to the chosen backend per phase.
```

## User Guide

### Launch Command for Different Attention Backends

- FlashInfer (Default for Non-Hopper Machines, e.g., A100, A40)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flashinfer
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend flashinfer \
  --trust-remote-code
```

- FlashAttention 3 (Default for Hopper Machines, e.g., H100, H200, H20)
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend fa3
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
  --attention-backend fa3
```

- Triton
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend triton
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend triton \
  --trust-remote-code
```

- FlashMLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --trust-remote-code
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- TRTLLM MLA (Optimized for Blackwell Architecture, e.g., B200)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --trust-remote-code
```

- TRTLLM MLA with FP8 KV Cache (Higher concurrency, lower memory footprint)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- FlashAttention 4 (MHA & MLA)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --prefill-attention-backend fa4 \
  --trust-remote-code
```

- Cutlass MLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend cutlass_mla \
  --trust-remote-code
```

- Ascend
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend ascend
```

- Intel XPU
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend intel_xpu
```

- Wave
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend wave
```

- FlexAttention
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flex_attention
```

- Dual Chunk FlashAttention
```bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --attention-backend dual_chunk_flash_attn
```

- Torch Native
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend torch_native
```

## Steps to add a new attention backend
To add a new attention backend, you can learn from the existing backends
(`python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`)
and follow the steps below.

1. Run without cuda graph. Support the two forward functions
    - forward_extend
        - Will be used for prefill, prefill with KV cache, and target verification
        - It will be called once per layer
    - forward_decode
        - Will be used for normal decode, and draft decode
        - It will be called once per layer
    - init_forward_metadata
        - Initialize the class and common metadata shared by all layers
        - Call the plan function for optimizations like split_kv
        - It will be called once per forward
2. Run with cuda graph. It has two phases (capture and replay) and you need to implement three functions
    - init_cuda_graph_state
        - It will be called once during life time
        - Create all common shared buffers
    - init_forward_metadata_capture_cuda_graph
        - It will be called before capturing a cuda graph
        - It is similar to init_forward_metadata but write the medatada to some pre-defined buffers
    - init_forward_metadata_replay_cuda_graph
        - It will be called before replaying a cuda graph
        - This function is in the critical path and needs to be fast
