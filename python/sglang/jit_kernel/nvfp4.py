from __future__ import annotations

import torch

_FLOAT4_E2M1_MAX = 6.0
_FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def suggest_nvfp4_global_scale(x: torch.Tensor) -> torch.Tensor:
    """Utility for tests/benchmarks: return global scale used by NVFP4 quantization."""
    tensor_amax = torch.abs(x).max().to(torch.float32)
    return _FLOAT8_E4M3_MAX * _FLOAT4_E2M1_MAX / tensor_amax
