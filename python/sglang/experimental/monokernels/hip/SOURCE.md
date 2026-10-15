# Sources of derived code

| Files | Derived from | License |
|---|---|---|
| `common/__init__.py`, `common/mx.py`, `common/ops.py`, `common/peer_memory.py`, `common/sync.py` | ROCm/ATOM at `97359d6`: `atom/mono/__init__.py`, `atom/mono/device/{mx,ops,ranks,stamps,sync}.py`, `atom/mono/runtime/{consensus,peer_memory}.py` | MIT, Copyright (C) 2026 Advanced Micro Devices, Inc. |
| `deepseek_v41/gfx950/attention/device.py` | ROCm/ATOM's mono decode framework | MIT, Copyright (C) 2026 Advanced Micro Devices, Inc. |
| `deepseek_v41/gfx950/attention/device.py`, `deepseek_v41/gfx950/attention/front.py` (MXFP8 scale rule) | vLLM's ROCm MXFP8 quantization | Apache-2.0, Copyright contributors to the vLLM project |
| every file with the vLLM SPDX header | first written for vLLM | Apache-2.0, Copyright contributors to the vLLM project |

The guarded wait (`common/sync.py`, `common/guard.py`) follows RFC #43949 section 8. No code is taken from
sglang#44406 or sglang#44407.
