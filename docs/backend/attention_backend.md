# Attention Backend

## Supporting matrix for different attention backends

| **Backend**              | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** | **MultiModal** |
|--------------------------|-------------------|-------------------|---------|--------------------|----------------|
| **FlashInfer**           | ❌                | ✅                 | ✅      | ✅                 | ✅              |
| **FA3**                  | ✅                | ✅                 | ✅      | ✅                 | ✅              |
| **Triton**               | ❌                | ✅                 | ✅      | ✅                 | ❌              |
| **Torch Native**         | ❌                | ❌                 | ❌      | ❌                 | ❌              |
| **FlashMLA**             | ✅                | ✅                 | ✅      | ❌                 | ❌              |
| **TRTLLM MLA**           | ✅                | ❌                 | ✅      | ✅                 | ❌              |
| **Ascend**               | ✅                | ❌                 | ❌      | ❌                 | ❌              |

**Notes:**
- TRTLLM MLA only implements decode operations. For prefill operations (including multimodal inputs), it falls back to FlashInfer MLA backend.

Note: Every kernel backend is compatible with a page size > 1 by specifying an argument such as `--page-size 16`.
This is because a page size of 16 can be converted to a page size of 1 in the kernel backend.
The "❌" and "✅" symbols in the table above under "Page Size > 1" indicate whether the kernel actually operates with a page size greater than 1, rather than treating a page size of 16 as a page size of 1.

## User guide

#### Launch command for different attention backends.

- FlashInfer (Default for Non-Hopper Machines, e.g., A100, A40)
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend flashinfer
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --attention-backend flashinfer --trust-remote-code
```

- FlashAttention 3 (Default for Hopper Machines, e.g., H100, H200, H20)
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend fa3
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --trust-remote-code --attention-backend fa3
```

- Triton
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend triton
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --attention-backend triton --trust-remote-code
```

- Torch Native
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend torch_native
```

- FlashMLA
```bash
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-R1 --attention-backend flashmla --trust-remote-code
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-R1 --attention-backend flashmla --kv-cache-dtype fp8_e4m3 --trust-remote-code
```

- TRTLLM MLA (Optimized for Blackwell Architecture, e.g., B200)
```bash
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-R1 --attention-backend trtllm_mla --trust-remote-code
```

- Ascend
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend ascend
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
