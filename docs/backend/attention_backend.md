# Attention Backend

## Supporting matrix for different attention backend

| **Backend**              | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** |
|--------------------------|-------------------|-------------------|--------|--------------------|
| **FlashInfer (Default)** | ✅                | ✅                | ✅     | ✅                 |
| **FA3**                  | ✅                | ✅                | ✅     | ✅                 |
| **FlashMLA**             | ✅                | ❌                | ✅     | ❌                 |
| **Triton**               | ❌                | ✅                | ❌     | ❌                 |
| **Torch Native**         | ❌                | ❌                | ❌     | ❌                 |

*Note: FlashMLA only supports page size = 64 case.*

## User guide

#### Launch command for different attention backend.

- FlashInfer (Default)
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

- FlashAttention 3
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend fa3
```

- FlashMLA
```bash
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --enable-flashmla
```

- Triton
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend triton
```

- Torch Native
```bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend torch_native
```
