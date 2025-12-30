# Attention Backends

This document describes the attention backends available in sglang diffusion (`sglang.multimodal_gen`) and how to select them.

## Overview

Attention backends are defined by `AttentionBackendEnum` (`sglang.multimodal_gen.runtime.platforms.interface.AttentionBackendEnum`) and selected via the CLI flag `--attention-backend`.

Backend selection is performed by the shared attention layers (e.g. `LocalAttention` / `USPAttention` / `UlyssesAttention` in `sglang.multimodal_gen.runtime.layers.attention.layer`) and therefore applies to any model component using these layers (e.g. diffusion transformer / DiT and encoders).

- **CUDA**: prefers FlashAttention (FA3/FA4) when supported; otherwise falls back to PyTorch SDPA.
- **ROCm**: uses FlashAttention when available; otherwise falls back to PyTorch SDPA.
- **MPS**: always uses PyTorch SDPA.

## Backend options

The CLI accepts the lowercase names of `AttentionBackendEnum`. The table below lists the backends implemented by the built-in platforms. `fa3`/`fa4` are accepted as aliases for `fa`.

| CLI value | Enum value | Notes |
|---|---|---|
| `fa` / `fa3` / `fa4` | `FA` | FlashAttention. `fa3/fa4` are normalized to `fa` during argument parsing (`ServerArgs.__post_init__`). |
| `torch_sdpa` | `TORCH_SDPA` | PyTorch `scaled_dot_product_attention`. |
| `sage_attn` | `SAGE_ATTN` | Requires `sageattention`. Upstream SageAttention CUDA extensions target SM80/SM86/SM89/SM90/SM120 (compute capability 8.0/8.6/8.9/9.0/12.0); see upstream `setup.py`: https://github.com/thu-ml/SageAttention/blob/main/setup.py. |
| `sage_attn_3` | `SAGE_ATTN_3` | Requires SageAttention3 installed per upstream instructions. |
| `aiter` | `AITER` | Requires `aiter`. |

## Selection priority

The selection order in `runtime/layers/attention/selector.py` is:

1. `global_force_attn_backend(...)` / `global_force_attn_backend_context_manager(...)`
2. CLI `--attention-backend` (`ServerArgs.attention_backend`)
3. Auto selection (platform capability, dtype, and installed packages)

## Platform support matrix

| Backend | CUDA | ROCm | MPS | Notes |
|---|---:|---:|---:|---|
| `fa` | ✅ | ✅ | ❌ | CUDA requires SM80+ and fp16/bf16. FlashAttention is only used when the required runtime is installed; otherwise it falls back to `torch_sdpa`. |
| `torch_sdpa` | ✅ | ✅ | ✅ | Most compatible option across platforms. |
| `sage_attn` | ✅ | ❌ | ❌ | CUDA-only (optional dependency). |
| `sage_attn_3` | ✅ | ❌ | ❌ | CUDA-only (optional dependency). |
| `aiter` | ✅ | ❌ | ❌ | Requires `aiter`. |

## Usage

### Select a backend via CLI

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend fa
```

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend torch_sdpa
```

### Notes for ROCm / MPS

- ROCm: use `--attention-backend torch_sdpa` or `fa` depending on what is available in your environment.
- MPS: the platform implementation always uses `torch_sdpa`.
