# SPDX-License-Identifier: Apache-2.0
"""Skinny (M == 1) fp8 PTPC GEMV over aiter (16,16)-preshuffled weights.

Adapted from vLLM's wvSplitKQ (csrc/rocm/skinny_gemms.cu) to the
preshuffled layout and per-channel weight scales (see the .cuh). Replaces
``aiter.gemm_a8w8_bpreshuffle`` for decode's M=1 qkv projections on gfx950,
where the tuned tile schemes are read-latency bound. Dispatch is gated to
the shapes where it measures faster; everything else keeps the aiter path.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils import is_gfx95_supported


@cache_once
def _jit_module():
    return load_jit(
        "skinny_ptpc_gemv",
        cuda_files=["gemm/skinny_ptpc_gemv.cuh"],
        cuda_wrappers=[("skinny_ptpc_gemv", "skinny_ptpc_gemv")],
    )


def skinny_ptpc_gemv_supported(m: int, n: int, k: int) -> bool:
    # Only the shapes where this measured faster than the aiter dispatch cold
    # (M3 qkv projections: N 1280/1536 at TP8, 2304/2560 at TP4).
    return (
        m == 1
        and k == 6144
        and 1280 <= n <= 2560
        and n % 64 == 0
        and is_gfx95_supported()
    )


def skinny_ptpc_gemv(
    q_input: torch.Tensor,  # [1, K] fp8
    w_shuf: torch.Tensor,  # (16,16)-preshuffled fp8, N*K bytes
    x_scale: torch.Tensor,  # [1] or [1, 1] fp32
    w_scale: torch.Tensor,  # [N] or [N, 1] fp32
) -> torch.Tensor:
    n = w_scale.numel()
    out = torch.empty(1, n, dtype=torch.bfloat16, device=q_input.device)
    _jit_module().skinny_ptpc_gemv(
        q_input.view(torch.uint8).view(-1),
        w_shuf.view(torch.uint8).view(-1),
        x_scale.view(-1),
        w_scale.view(-1),
        out,
    )
    return out
