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

- FlashInfer (Default)
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3
```

- FlashAttention 3
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend fa3
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --attention-backend fa3
```

- Triton
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend triton
python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --attention-backend triton

```

- Torch Native
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend torch_native
```
