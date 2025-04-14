# Attention Backend

## Supporting matrix for different attention backend

| **Backend**              | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** | **MultiModal** |
|--------------------------|-------------------|-------------------|--------|--------------------|------------|
| **FlashInfer (Default)** | ✅                | ✅                | ✅     | ✅                 | ✅ |
| **FA3**                  | ✅                | ✅                | ✅     | ✅                 | ✅ |
| **Triton**               | ❌                | ✅                | ✅     | ❌                 | ❌ |
| **Torch Native**         | ❌                | ❌                | ❌     | ❌                 | ❌ |


## User guide

#### Launch command for different attention backend.

- FlashInfer (Default backend if not use MLA)
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --attention-backend flashinfer --trust-remote-code
```

- FlashAttention 3 (Default Backend for MLA)
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend fa3
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --trust-remote-code
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
