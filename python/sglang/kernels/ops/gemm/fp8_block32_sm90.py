"""SM90 block-[32, 32] FP8 GEMM for NVIDIA H20-3e.

WGMMA atom K is 32. The stock CUTLASS SM90 blockwise mainloop scales only
after four atoms (K=128). This kernel multiplies the fp32 group scale after
every atom, matching the Triton ``w8a8_block_fp8_matmul`` fallback.

The serving dispatch enables it only on NVIDIA H20-3e. Other SM90 GPUs keep
the Triton kernel. Decode-sized M uses an 8-token tile plus K split so the
grid fills an H20-3e (78 SMs). Prefill-sized M uses a 64x128 tile; the
8-token tile reloads the weight once per 8 rows and loses to Triton there.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fp8_block32_sm90_module() -> Module:
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("fp8_block32_sm90 requires an SM90 GPU")
    return load_jit(
        "fp8_block32_sm90",
        cuda_files=["gemm/fp8_block32_sm90.cuh"],
        cuda_wrappers=[
            ("run", "fp8_block32_sm90"),
            ("run_large", "fp8_block32_sm90_large"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
        extra_dependencies=["cutlass"],
    )


def legal_split_k(k: int, split_k: int) -> bool:
    """True when every split is a non-empty multiple of the group size 32."""
    return split_k >= 1 and k % split_k == 0 and (k // split_k) % 32 == 0


def choose_split_k(n: int, k: int, sm_count: int, m: int = 1) -> int:
    """Largest legal split at or below ~8 grid-waves, with at least 128 K per split.

    On H20-3e the K=5120 square GEMV was bandwidth-bound at one CTA per 64 rows
    (split 1, ~80 CTAs). Splitting K into 8 raised the grid to ~8 waves and
    moved device time from the mid-30 us range to ~10 us. Smaller shapes are
    capped so a split still owns 128 K elements (four WGMMA atoms). Token tiles
    of 8 already add grid z, so a wide M does not also split K.
    """
    groups = k // 32
    tiles_n = max(1, (n + 63) // 64)
    tiles_m = max(1, (m + 7) // 8)
    tiles = tiles_n * tiles_m
    target = max(1, (sm_count * 8 + tiles - 1) // tiles)
    limit = min(target, max(1, k // 128), groups)
    best = 1
    for split in range(1, limit + 1):
        if groups % split == 0:
            best = split
    return best


@lru_cache(maxsize=8)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_H20_3E_NAME = "NVIDIA H20-3e"


@lru_cache(maxsize=8)
def _is_h20_3e(device_index: int) -> bool:
    """True only for the H20-3e SKU. Other SM90 parts keep Triton."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_name(device_index) == _H20_3E_NAME


def use_fp8_block32_large_tile(m: int, n: int, k: int, device_index: int = 0) -> bool:
    """True when the 64x128 tile beats the 8-token split-K kernel on H20-3e.

    The wide tile always computes 64 token rows. At M=1 that is pure waste, and
    on the decode GEMMs it loses: wq_b is N=4096 K=1280 (8 µs vs 16 µs) and
    wo_b is N=5120 K=1024 (7 µs vs 14 µs). Those two run every layer, so routing
    them to the wide tile regresses TPOT even though the same tile wins prefill
    (M≈1k, about 3×). Split-K stays on the decode side until M fills most of
    the 64-row tile and the grid covers about half the SMs.
    """
    del k
    if m < 48:
        return False
    tiles = ((m + 63) // 64) * ((n + 127) // 128)
    sm = _sm_count(device_index)
    return tiles >= max(sm // 2, 1)


def select_fp8_block32_variant(m: int, n: int, k: int, device_index: int = 0) -> str:
    """``64x128`` or ``m8`` for logs."""
    if use_fp8_block32_large_tile(m, n, k, device_index):
        return "64x128"
    return "m8"


def fp8_block32_gemm_large(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """64x128 block-[32, 32] GEMM. One warpgroup, four K-pipeline stages."""
    if act.ndim == 1:
        act = act.view(1, -1)
    m, _k = act.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=act.device)
    module = _jit_fp8_block32_sm90_module()
    module.run_large(act, weight, act_scale, weight_scale, out)
    return out


def fp8_block32_gemm(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    split_k: int | None = None,
) -> torch.Tensor:
    """``out[M, N] = act[M, K] @ weight[N, K].T`` with block-[32, 32] fp32 scales.

    ``act`` and ``weight`` are fp8 e4m3. Scales are fp32, already decoded from
    UE8M0 when the checkpoint stores powers of two. ``weight_scale`` is
    ``[ceil(N/32), K/32]`` and ``act_scale`` is ``[M, K/32]``. One CTA consumes
    8 tokens; M is not limited to 1.
    """
    if act.ndim == 1:
        act = act.view(1, -1)
    assert act.ndim == 2 and act.shape[0] >= 1
    assert weight.ndim == 2 and act.shape[1] == weight.shape[1]
    m, k = act.shape
    n = weight.shape[0]
    if split_k is None:
        device_index = act.device.index or 0
        if use_fp8_block32_large_tile(m, n, k, device_index):
            return fp8_block32_gemm_large(act, weight, act_scale, weight_scale)
        split_k = choose_split_k(n, k, _sm_count(device_index), m)
    assert legal_split_k(k, split_k), (k, split_k)

    out = torch.empty((m, n), dtype=torch.bfloat16, device=act.device)
    # split_k == 1 stores bf16 from the mainloop; the fp32 partial is unused.
    partial = (
        torch.empty((split_k, m, n), dtype=torch.float32, device=act.device)
        if split_k > 1
        else torch.empty(0, dtype=torch.float32, device=act.device)
    )
    module = _jit_fp8_block32_sm90_module()
    module.run(act, weight, act_scale, weight_scale, partial, out, split_k)
    return out


def use_fp8_block32_gemm(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> bool:
    """True when this kernel should replace Triton. NVIDIA H20-3e only."""
    if output_dtype != torch.bfloat16 or list(block_size) != [32, 32]:
        return False
    if act.ndim != 2 or act.shape[0] < 1 or weight.ndim != 2:
        return False
    if not act.is_cuda or torch.cuda.get_device_capability(act.device)[0] != 9:
        return False
    device_index = act.device.index if act.device.index is not None else 0
    if not _is_h20_3e(device_index):
        return False
    n, k = weight.shape
    if k % 32 != 0 or k == 0 or n == 0 or act.shape[1] != k:
        return False
    if act.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        return False
    if act_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        return False
    if not act.is_contiguous() or not weight.is_contiguous():
        return False
    scale_k = k // 32
    m = act.shape[0]
    if act_scale.shape != (m, scale_k) or weight_scale.shape != (
        (n + 31) // 32,
        scale_k,
    ):
        return False
    return act_scale.is_contiguous() and weight_scale.is_contiguous()
