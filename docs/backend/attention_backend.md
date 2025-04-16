# Attention Backend

## Supporting matrix for different attention backend

| **Backend**              | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** | **MultiModal** |
|--------------------------|-------------------|-------------------|--------|--------------------|------------|
| **FlashInfer** | ✅                | ✅                | ✅     | ✅                 | ✅ |
| **FA3**                  | ✅                | ✅                | ✅     | ✅                 | ✅ |
| **Triton**               | ❌                | ✅                | ✅     | ❌                 | ❌ |
| **Torch Native**         | ❌                | ❌                | ❌     | ❌                 | ❌ |


## User guide

#### Launch command for different attention backend.

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
