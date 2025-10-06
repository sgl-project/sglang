# Attention Backend

SGLang supports a large variety of attention backends. Each of them has different pros and cons.
You can test them according to your needs.

```{important}
Selecting an optimal attention backend is crucial for maximizing your performance. Different backends excel in various scenarios, so choose based on your model, hardware, and use case.
```

## Supporting Matrix

| **Backend**               | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** | **MultiModal** |
|---------------------------|-------------------|-------------------|---------|--------------------|----------------|
| **FlashInfer**            | ❌                | ✅                 | ✅      | ✅                 | ✅              |
| **FA3**                   | ✅                | ✅                 | ✅      | ✅                 | ✅              |
| **FA4**                   | ❌                | ❌                 | ✅      | ❌                 | ❌              |
| **Triton**                | ❌                | ✅                 | ✅      | ✅                 | ❌              |
| **NSA**                   | ✅                | ❌                 | ✅      | ❌                 | ❌              |
| **FlashMLA**              | ✅                | ✅                 | ✅      | ❌                 | ❌              |
| **Cutlass MLA**           | ✅                | ✅                 | ✅      | ❌                 | ❌              |
| **TRTLLM MLA**            | ✅                | ❌                 | ✅      | ✅                 | ❌              |
| **TRTLLM MHA**            | ✅                | ✅                 | ❌      | ✅                 | ❌              |
| **Torch Native**          | ❌                | ❌                 | ✅      | ❌                 | ❌              |
| **FlexAttention**         | ❌                | ❌                 | ✅      | ❌                 | ❌              |
| **Dual Chunk FlashAttention** | ✅                | ✅                 | ✅      | ❌                 | ✅              |
| **AITER**                 | ✅                | ✅                 | ✅      | ❌                 | ❌              |
| **Wave**                  | ✅                | ❌                 | ❌      | ❌                 | ❌              |
| **Ascend**                | ✅                | ❌                 | ✅      | ❌                 | ❌              |

```{note}
FlashAttention v4 (fa4) is prefill-only for now.

TRTLLM MLA only implements decode operations. For prefill operations, it falls back to FlashInfer MLA backend.

TRTLLM MHA supports speculative decoding with topk ≤ 1 only.

NSA is designed for DeepSeek NSA models and enables sparse attention via indexing.
```

Note: Every kernel backend is compatible with a page size > 1 by specifying an argument such as `--page-size 16`.
This is because a page size of 16 can be converted to a page size of 1 in the kernel backend.
The "❌" and "✅" symbols in the table above under "Page Size > 1" indicate whether the kernel actually operates with a page size greater than 1, rather than treating a page size of 16 as a page size of 1.

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
- For paged backends with `--page-size > 1` and `--speculative-eagle-topk > 1`, only `flashinfer` is supported.
- `flex_attention` is not supported with speculative decoding.
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
