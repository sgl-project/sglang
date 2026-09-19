"""Fused SiLU+Mul + MXFP4/FP8 quantization kernels (PPU-only, JIT via tvm_ffi)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.common import is_ppu

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_PPU_ONLY_MSG = "{name} is PPU-only (__ppu_sgmdf); use the generic fallback on non-PPU."

_TP_BLOCK_N_FAST = 512  # full-warp (32 threads), for H divisible by 512
_TP_BLOCK_N_MID = 256  # half-warp (16 threads), for H divisible by 256
_TP_BLOCK_N_SAFE = 128  # 1/4-warp (8 threads), for any even H
_EP_BLOCK_N = 256
_FP8_BLOCK_THREADS = 256


@cache_once
def _jit_tp_module(apply_clamp: bool, block_n: int) -> Module:
    if not is_ppu():
        raise RuntimeError(_PPU_ONLY_MSG.format(name="silu_and_mul_post_quant_mxfp4"))
    clamp_str = "true" if apply_clamp else "false"
    return load_jit(
        "silu_and_mul_post_quant_mxfp4",
        str(block_n),
        clamp_str,
        cuda_files=["elementwise/silu_and_mul_post_quant_mxfp4.cuh"],
        cuda_wrappers=[
            ("run", f"SiluMulMxfp4TP<{block_n},{clamp_str}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_post_quant_mxfp4(
    gateup: torch.Tensor,
    swiglu_limit: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gateup.ndim == 2, "input must be 2D (N, 2H)"
    assert gateup.dtype == torch.bfloat16, "input must be bfloat16"
    gateup = gateup.contiguous()

    N, two_H = gateup.shape
    H = two_H // 2
    assert H > 0 and H % 2 == 0, f"H must be positive and even, got H={H}"

    # Dynamic block_n: prefer largest aligned block for better warp utilization
    if H % _TP_BLOCK_N_FAST == 0:
        block_n = _TP_BLOCK_N_FAST
    elif H % _TP_BLOCK_N_MID == 0:
        block_n = _TP_BLOCK_N_MID
    else:
        block_n = _TP_BLOCK_N_SAFE

    # H_padded: H rounded up to block_n; kernel writes scale for all groups in [0, H_padded)
    H_padded = ((H + block_n - 1) // block_n) * block_n
    S_alloc = H_padded // 64
    S_valid = (H + 63) // 64

    out_quant = torch.empty((N, H // 2), dtype=torch.uint8, device=gateup.device)
    out_scale = torch.empty((S_alloc, N), dtype=torch.uint16, device=gateup.device)

    if N == 0:
        return out_quant, out_scale[:S_valid, :].t()

    apply_clamp = swiglu_limit is not None
    limit_val = float(swiglu_limit) if apply_clamp else 0.0

    module = _jit_tp_module(apply_clamp, block_n)
    module.run(gateup, out_quant, out_scale, limit_val)

    return out_quant, out_scale[:S_valid, :].t()


@cache_once
def _jit_fp8_module(apply_clamp: bool) -> Module:
    if not is_ppu():
        raise RuntimeError(
            _PPU_ONLY_MSG.format(name="silu_and_mul_post_per_token_quant_fp8")
        )
    clamp_str = "true" if apply_clamp else "false"
    return load_jit(
        "silu_and_mul_post_per_token_quant_fp8",
        clamp_str,
        cuda_files=["elementwise/silu_and_mul_post_per_token_quant_fp8.cuh"],
        cuda_wrappers=[
            ("run", f"SiluMulFp8TP<{_FP8_BLOCK_THREADS},{clamp_str}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_post_per_token_quant_fp8(
    gateup: torch.Tensor,
    swiglu_limit: Optional[float] = None,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gateup.ndim == 2, "input must be 2D (N, 2H)"
    assert gateup.dtype == torch.bfloat16, "input must be bfloat16"
    gateup = gateup.contiguous()

    N, two_H = gateup.shape
    H = two_H // 2
    assert two_H % 2 == 0, "input last dim must be even"
    assert H % 8 == 0, f"H must be multiple of 8, got H={H}"

    output = torch.empty((N, H), dtype=torch.float8_e4m3fn, device=gateup.device)
    output_scale = torch.empty((N, 1), dtype=torch.float32, device=gateup.device)

    if N == 0:
        return output, output_scale

    apply_clamp = swiglu_limit is not None
    limit_val = float(swiglu_limit) if apply_clamp else 0.0

    module = _jit_fp8_module(apply_clamp)
    module.run(gateup, output, output_scale, limit_val, float(eps))

    return output, output_scale


@cache_once
def _jit_ep_module(apply_clamp: bool) -> Module:
    if not is_ppu():
        raise RuntimeError(
            _PPU_ONLY_MSG.format(name="silu_and_mul_masked_post_quant_mxfp4")
        )
    clamp_str = "true" if apply_clamp else "false"
    return load_jit(
        "silu_and_mul_masked_post_quant_mxfp4",
        clamp_str,
        cuda_files=["elementwise/silu_and_mul_masked_post_quant_mxfp4.cuh"],
        cuda_wrappers=[
            ("run", f"SiluMulMxfp4EP<{_EP_BLOCK_N},{clamp_str}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_masked_post_quant_mxfp4(
    input: torch.Tensor,
    masked_m: torch.Tensor,
    swiglu_limit: Optional[float] = None,
    expected_m: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim == 3, "input must be 3D (E, T, 2H)"
    assert input.dtype == torch.bfloat16, "input must be bfloat16"
    assert masked_m.dtype == torch.int32, "masked_m must be int32"

    E, T, two_H = input.shape
    H = two_H // 2
    assert H > 0 and H % 2 == 0, f"H must be positive and even, got H={H}"

    # H_padded: H rounded up to block_n; kernel writes scale for all groups in [0, H_padded)
    H_padded = ((H + _EP_BLOCK_N - 1) // _EP_BLOCK_N) * _EP_BLOCK_N
    S_alloc = H_padded // 64
    S_valid = (H + 63) // 64

    out_quant = torch.empty((E, T, H // 2), dtype=torch.uint8, device=input.device)
    out_scale = torch.empty((E, S_alloc, T), dtype=torch.uint16, device=input.device)

    apply_clamp = swiglu_limit is not None
    limit_val = float(swiglu_limit) if apply_clamp else 0.0

    max_masked_m = int(expected_m) if expected_m is not None else int(T)

    module = _jit_ep_module(apply_clamp)
    module.run(input, out_quant, out_scale, masked_m, limit_val, max_masked_m)

    return out_quant, out_scale[:, :S_valid, :]
