# HiSparse: Hierarchical Sparse Attention

HiSparse reduces per-request GPU memory consumption during the decode phase by maintaining only a small "hot" KV buffer on GPU while keeping complete KV data in CPU pinned memory. Combined with PD disaggregation, it enables significantly higher decode concurrency.

> **Prerequisites**: HiSparse only works with models that use **DeepSeek Sparse Attention (DSA)**  architectures (e.g., DeepSeek-V3.2, GLM-5). These models natively select a subset of tokens for attention, making it possible to keep only the top-k KV on GPU while storing the full KV in host memory — without accuracy loss.  Additionally, HiSparse currently requires **PD disaggregation mode** and is enabled on the **decode instance** only.

## Why HiSparse?

In long-context LLM inference, each decoding request holds a full-length KV cache on GPU, limiting the number of concurrent requests a decode instance can serve. HiSparse addresses this by:

- **Reducing GPU memory per request**: Each request occupies only a fixed-size device buffer (e.g., 4KB tokens) instead of the full sequence length.
- **On-demand swap-in**: A CUDA kernel dynamically loads the top-k most relevant KV entries from host memory based on attention scores.
- **Transparent to prefill**: HiSparse is entirely a decode-side optimization; the prefill instance requires no changes.

## Design Overview

### Decode Workflow

Each decode step follows this flow:

1. **Forward decode** — generate the next token
2. **Top-k selection** — select the most relevant token positions via attention scores
3. **Swap-in** — the CUDA kernel loads top-k KV entries from host to device buffer:
   - *Short sequences* (`seq_len ≤ device_buffer_size`): fast path, all KV already in buffer
   - *Long sequences*: hit detection → LRU reordering → miss handling (host → device copy)
4. **Decode attention** — compute attention using the top-k device locations
5. **Eager backup** — asynchronously copy the previous token's KV from device to host

### PD Disaggregation Integration (Direct-to-Host)

In PD disaggregation mode, the prefill instance transfers KV cache directly into the decode instance's host pool via RDMA, bypassing the GPU entirely on the decode side. This eliminates the transient GPU memory spike during KV transfer and removes the staging DMA step.

```
Prefill GPU  ──RDMA──▶  Decode Host Pool (CPU pinned memory)
                              │
                              ▼
                     alloc device buffer (4KB)
                              │
                              ▼
                     swap-in kernel (on-demand top-k)
```

## Server Arguments

| Argument | Type / Default | Description |
|----------|---------------|-------------|
| `--enable-hisparse` | flag; default: disabled | Enable HiSparse on the decode instance |
| `--hisparse-config` | JSON string | Configuration for HiSparse (see below) |

### HiSparse Config Parameters

Pass as a JSON string via `--hisparse-config`:

| Parameter | Type / Default | Description |
|-----------|---------------|-------------|
| `top_k` | int | Number of topk entries |
| `device_buffer_size` | int | Number of token slots in the per-request GPU device buffer |
| `host_to_device_ratio` | int | Ratio of logical pool size to device pool size, determining host memory capacity |

Example: `--hisparse-config='{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}'`

## Deployment

HiSparse currently requires **PD disaggregation mode** and is enabled only on the **decode instance**.

### Prefill Instance

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/model \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --context-length 81920 \
    --chunked-prefill-size 65536 \
    --tp-size 8 --dp-size 8 --enable-dp-attention \
    --page-size 64 \
    --mem-fraction-static 0.85 \
    --disaggregation-mode prefill \
    --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
    --nnodes 1 --node-rank 0
```

### Decode Instance (with HiSparse)

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/model \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --context-length 81920 \
    --tp-size 8 --dp-size 8 --enable-dp-attention \
    --page-size 64 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype bfloat16 \
    --nsa-decode-backend flashmla_sparse \
    --disaggregation-mode decode \
    --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
    --dist-init-addr 127.0.0.1:5757 \
    --nnodes 1 --node-rank 0 \
    --enable-hisparse \
    --hisparse-config='{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 5}'
```

### Key Notes

- The prefill instance does not need `--enable-hisparse`; it is unaware of HiSparse.
- On the decode instance, the following flags are **required** for HiSparse:
  - `--kv-cache-dtype bfloat16` — currently only bfloat16 KV cache is supported (more dtypes planned).
  - `--nsa-decode-backend flashmla_sparse` — currently only `flashmla_sparse` backend is supported.
  - `--enable-hisparse` — enables HiSparse.
  - `--hisparse-config` — HiSparse configuration (top_k, device_buffer_size, host_to_device_ratio).
