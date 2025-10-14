# Attention Backend

SGLang supports a large variety of attention backends. Each of them has different pros and cons.
You can test them according to your needs.

```{important}
Selecting an optimal attention backend is crucial for maximizing your performance. Different backends excel in various scenarios, so choose based on your model, hardware, and use case. Not all backends are supported on all platforms and model architectures.
```

## Supporting Matrix

The support matrix is split into two parts: MHA (standard attention) and MLA (multi-head latent attention). For an explanation of the key differences between MHA and MLA, please see the [SGLang documentation on DeepSeek MLA](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek.md#multi-head-latent-attention-mla) and the original [DeepSeek MLA paper](https://arxiv.org/pdf/2405.04434).

### MHA Backends

| **Backend**                     | **Page Size > 1 (native)** | **FP8 KV Cache** | **Spec topk=1** | **Spec topk>1** | **Sliding Window** | **MultiModal** |
|---------------------------------|-----------------------------|------------------|-----------------|-----------------|--------------------|----------------|
| **FlashInfer**                  | ✅                          | ✅               | ✅              | ✅              | ✅                 | ❌             |
| **FA3 (FlashAttention 3)**     | ✅                          | ✅               | ✅              | ⚠️ (limited)    | ✅                 | ✅             |
| **FA4 (FlashAttention 4)**     | ❌ (prefill-only)           | n/a              | ❌              | ❌              | ❌                 | ❌             |
| **Triton**                      | ❌                          | ❌               | ✅              | ✅              | ✅                 | ❌             |
| **TRTLLM MHA (Blackwell)**      | ✅                          | ✅               | ✅              | ❌              | ✅                 | ❌             |
| **Torch Native (SDPA)**         | ❌                          | ❌               | ❌              | ❌              | ❌                 | ❌             |
| **FlexAttention (PyTorch)**     | ❌                          | ❌               | ❌              | ❌              | ❌                 | ❌             |
| **Dual Chunk FlashAttention**   | ✅                          | ❌               | ✅              | ✅              | ❌                 | ❌             |
| **AITER (ROCm)**                | ✅                          | ❌               | ✅              | ✅              | ❌                 | ❌             |
| **Wave (ROCm)**                 | ✅                          | ❌               | ❌              | ❌              | ❌                 | ❌             |
| **Ascend (NPU)**                | ✅                          | ❌               | ❌              | ❌              | ❌                 | ❌             |

### MLA Backends

| **Backend**                | **Native Page Sizes**     | **FP8 KV Cache** | **Chunked Prefix Cache** | **Spec topk=1** | **Spec topk>1** |
|----------------------------|---------------------------|------------------|--------------------------|-----------------|-----------------|
| **FlashInfer MLA**         | 1                         | ✅               | ✅                       | ✅              | ❌              |
| **FlashMLA**               | 64                        | ⚠️ (caveat)      | ✅                       | ✅              | ❌              |
| **Cutlass MLA**            | 128                       | ✅               | ✅                       | ✅              | ❌              |
| **TRTLLM MLA (Blackwell)** | 32 or 64                  | ✅               | ✅                       | ✅              | ✅              |
| **FA4 (MLA prefill-only)** | n/a                       | n/a              | n/a                     | ❌              | ❌              |
| **Ascend MLA (NPU)**             | 128                       | ❌               | ✅                       | ❌              | ❌              |

```{note}
- FlashAttention 4 (fa4) is prefill-only for now and MLA-only in SGLang.
- FlexAttention and Torch Native do not currently support MLA in SGLang.
- Dual Chunk FlashAttention does not support multimodal attention.
- NSA is designed for DeepSeek NSA models and enables sparse attention via indexing.
```

```{important}
FlashInfer vs FlashInfer MLA:
- FlashInfer (MHA) supports native page_size > 1 and speculative decoding with topk > 1.
- FlashInfer MLA uses MLA kernels and currently supports page_size = 1 only and speculative decoding with topk = 1.
```

```{tip}
Speculative decoding topk: "topk" is the number of draft tokens sampled per step from the draft model. topk = 1 follows classic EAGLE; topk > 1 explores multiple branches and requires backend support in both draft and verification paths.
```

Note: Every kernel backend is compatible with a page size > 1 via page-table emulation, but some operate natively at page sizes > 1. The "Page Size > 1 (native)" column indicates native support.

MLA page-size constraints:
- FlashInfer MLA: page_size = 1.
- FlashMLA: page_size = 64.
- Cutlass MLA: page_size = 128.
- TRTLLM MLA: page_size ∈ {32, 64}.

FP8 KV cache:
- TRTLLM MLA requires `--kv-cache-dtype fp8_e4m3` or `auto` (will choose fp8_e4m3 on supported GPUs).
- FlashMLA has known FP8 caveats.

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
- `flex_attention` is not supported with speculative decoding.
- For MLA backends, `trtllm_mla` supports `topk > 1`; `flashmla` and `flashinfer_mla` support only `topk = 1`.
- CUDA Graph: the decode backend is always captured; the prefill backend is captured only when `--speculative-attention-mode prefill`.


```{tip}
If you set only one of `--prefill-attention-backend` or `--decode-attention-backend`, the unspecified phase inherits `--attention-backend`.
If both are specified and differ, SGLang automatically enables a hybrid wrapper to dispatch to the chosen backend per phase.
```

## User guide

### Launch command for different attention backends.

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

- Torch Native
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend torch_native
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

- Ascend
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend ascend
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

- Dual Chunk FlashAttention (MHA-only)
```bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --attention-backend dual_chunk_flash_attn
```

- Cutlass MLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend cutlass_mla \
  --trust-remote-code
```

- FlashAttention 4 (MLA-only prefill)
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend fa4 \
  --trust-remote-code
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
