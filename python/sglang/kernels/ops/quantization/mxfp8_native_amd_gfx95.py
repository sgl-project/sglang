# SPDX-License-Identifier: Apache-2.0
"""The gfx950 native MXFP8 dense route for 32x32-block ue8m0 fp8 checkpoints.

The weight is kept in fp8, permuted once at load into the scaled-MFMA lane order
(``shuffle_mxfp8_weight``, ``[N/16, K/128, 2048]``) with its block scales as ue8m0 exponent bytes
``[N/32, K/32]``; a bf16 copy is kept only for a shape whose ``large_m_plan`` picks hipBLASLt.
Every path multiplies the CUDA MXFP8 route's operands (fp8 e4m3 x fp8 e4m3, per-32 ue8m0 scales,
fp32 accumulation) with a summation order fixed by its tile, so repeated calls are bitwise
identical and rows are batch-invariant inside one M range. ``M <= 32`` runs ``mxfp8_gemv`` with
the per-shape config from ``mxfp8_gemv_gfx95_configs.json``; larger M runs the Triton
``tl.dot_scaled`` GEMM over the shuffled weight or hipBLASLt bf16, per ``large_m_plan``.
Skinny kernel limits: 1 <= M <= 32, K % 128 == 0, N % 32 == 0; gfx950 only.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    dequant_block_fp8_weight_to_bf16,
    fake_quant_fp8_activation,
    mxfp8_e4m3_quantize,
)

# M <= 32: the scaled-MFMA skinny kernel (deepseek_v4/mxfp8_gemv_gfx95.cuh)
MXFP8_GEMV_MAX_TOKENS = 32
_TILE_N = 16
_STEP_K = 128
_LANES = 64
_LANE_BYTES = 32
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "mxfp8_gemv_gfx95_configs.json")
# Token-count buckets of the config table: a config tuned at the bucket's upper bound.
M_BUCKETS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class GemvConfig:
    waves: int = 8  # waves per workgroup: 4, 8, 16
    steps: int = 1  # 128-K steps in flight per wave: 1, 2, 4
    rows: int = 16  # weight rows per wave tile: 16, 32
    tokens: int = 16  # token columns per wave tile: 16, 32 (M <= tokens)
    ksplit: bool = True  # waves split K and reduce through LDS; else one tile per wave

    def key(self) -> str:
        return f"w{self.waves}s{self.steps}r{self.rows}t{self.tokens}{'k' if self.ksplit else 'n'}"

    @staticmethod
    def parse(key: str) -> GemvConfig:
        import re

        m = re.fullmatch(r"w(\d+)s(\d+)r(\d+)t(\d+)([kn])", key)
        assert m, key
        return GemvConfig(int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5] == "k")

    def valid_for(self, m: int, n: int, k: int) -> bool:
        return (
            self.waves in (4, 8, 16)
            and self.steps in (1, 2, 4)
            and self.rows in (16, 32)
            and self.tokens in (16, 32)
            and m <= self.tokens
            and n % self.rows == 0
        )


ALL_CONFIGS = tuple(
    GemvConfig(w, s, r, t, ks)
    for w in (4, 8, 16)
    for s in (1, 2, 4)
    for r in (16, 32)
    for t in (16, 32)
    for ks in (True, False)
)


def default_config(m: int, n: int, k: int) -> GemvConfig:
    """Heuristic for shapes without a tuned row."""
    tokens = 16 if m <= 16 else 32
    if k // _STEP_K > 40:
        return GemvConfig(4, 2, 16, tokens, True)
    return GemvConfig(8, 1, 16, tokens, True)


def m_bucket(m: int) -> int:
    for b in M_BUCKETS:
        if m <= b:
            return b
    raise ValueError(
        f"M={m} exceeds the skinny kernel's {MXFP8_GEMV_MAX_TOKENS} tokens"
    )


@functools.lru_cache(maxsize=1)
def _load_config_table() -> Dict[str, str]:
    """{'gfx950:N:K:Mbucket': config key} from the JSON next to this file."""
    try:
        with open(CONFIG_FILE) as f:
            table = json.load(f)
    except (OSError, ValueError):
        return {}
    return {str(key): str(value) for key, value in table.get("configs", {}).items()}


@functools.lru_cache(maxsize=None)
def gfx_name() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def select_config(m: int, n: int, k: int) -> GemvConfig:
    """The tuned configuration for (gfx, N, K, M bucket), else the heuristic."""
    table = _load_config_table()
    key = f"{gfx_name()}:{n}:{k}:{m_bucket(m)}"
    if key in table:
        cfg = GemvConfig.parse(table[key])
        if cfg.valid_for(m, n, k):
            return cfg
    return default_config(m, n, k)


@cache_once
def _jit_module(cfg: GemvConfig, x_bf16: bool):
    args = make_cpp_args(cfg.waves, cfg.steps, cfg.rows, cfg.tokens, cfg.ksplit, x_bf16)
    return load_jit(
        "dpsk_v4_mxfp8_gemv_gfx95",
        *args,
        cuda_files=["deepseek_v4/mxfp8_gemv_gfx95.cuh"],
        cuda_wrappers=[("run", f"Mxfp8GemvGfx950Kernel<{args}>::run")],
    )


def shuffle_mxfp8_weight(weight: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 ``[N, K]`` -> ``[N/16, K/128, 2048]`` uint8 in the gfx950 16x16x128 scaled-MFMA
    lane order: for tile ``t``, K step ``s`` and lane ``l = 16 * g + r``, the lane's 32 bytes are
    ``W[16t + r][128s + 32(g/2) + 16(g%2) : +16]`` followed by the same 16 bytes 64 K later."""
    assert weight.dtype == torch.float8_e4m3fn, weight.dtype
    n, k = weight.shape
    assert n % _TILE_N == 0 and k % _STEP_K == 0, (n, k)
    t = weight.view(torch.uint8).view(n // _TILE_N, _TILE_N, k // _STEP_K, 2, 2, 2, 16)
    # (tile, row, step, half, g1, g0, 16 B) -> (tile, step, g1, g0, row, half, 16 B)
    return (
        t.permute(0, 2, 4, 5, 1, 3, 6)
        .contiguous()
        .view(n // _TILE_N, k // _STEP_K, _LANES * _LANE_BYTES)
    )


def ue8m0_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two block scales ``[N/32, K/32]`` -> ue8m0 exponent bytes (exact)."""
    s = weight_scale.float().contiguous()
    bits = s.view(torch.int32)
    assert bool(((bits & 0x807FFFFF) == 0).all()), (
        "weight scales must be positive powers of two"
    )
    return (bits >> 23).to(torch.uint8)


def mxfp8_gemv(
    x: torch.Tensor,
    weight_shuffled: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    config: Optional[GemvConfig] = None,
) -> torch.Tensor:
    """``out[M, N] bf16 = x[M, K] . W^T`` on the gfx950 scaled matrix core.

    ``x`` is fp8 e4m3 with ``x_scale`` ue8m0 ``[M, K/32]``, or bf16 (quantized in-kernel with
    the same rule; ``x_scale`` unused).  ``weight_shuffled`` / ``weight_scale_ue8m0`` come from
    ``shuffle_mxfp8_weight`` / ``ue8m0_weight_scale``.  ``config`` overrides the table lookup."""
    assert x.dim() == 2 and x.is_contiguous(), x.shape
    m, k = x.shape
    n = weight_shuffled.shape[0] * _TILE_N
    assert 1 <= m <= MXFP8_GEMV_MAX_TOKENS, m
    x_bf16 = x.dtype == torch.bfloat16
    if x_bf16:
        x_scale = torch.empty(0, dtype=torch.uint8, device=x.device)
    else:
        assert x.dtype == torch.float8_e4m3fn and x_scale is not None, x.dtype
        assert x_scale.dtype == torch.uint8 and x_scale.shape == (m, k // 32), (
            x_scale.shape
        )
        x_scale = x_scale.contiguous()
        x = x.view(
            torch.uint8
        )  # tvm-ffi has no fp8 dtype on ROCm; the kernel takes the bytes
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    cfg = config or select_config(m, n, k)
    assert cfg.valid_for(m, n, k), (cfg, m, n, k)
    _jit_module(cfg, x_bf16).run(weight_shuffled, weight_scale_ue8m0, x, x_scale, out)
    return out


# M > 32: hipBLASLt bf16 on a bf16 copy or the Triton dot_scaled tile, per the table's "large_m" section
LARGE_M_BUCKETS = (64, 128, 256, 1024, 4096, 8192, 16384)
HIPBLASLT_BF16 = "hipblaslt_bf16"


def large_m_bucket(m: int) -> int:
    for b in LARGE_M_BUCKETS:
        if m <= b:
            return b
    return LARGE_M_BUCKETS[-1]


@functools.lru_cache(maxsize=2)
def _large_m_table(fp8_in: bool) -> Dict[str, str]:
    """{'gfx950:N:K:bucket': 'hipblaslt_bf16' | 'ds:BM,BN,BK,warps,splitk'} from the table's "large_m" section."""
    try:
        with open(CONFIG_FILE) as f:
            table = json.load(f)
    except (OSError, ValueError):
        return {}
    section = table.get("large_m_fp8in" if fp8_in else "large_m", {})
    return {str(k): str(v) for k, v in section.items()}


def large_m_plan(
    m: int, n: int, k: int, fp8_in: bool = False
) -> Optional[Tuple[int, int, int, int, int]]:
    """The dot_scaled tile (BM, BN, BK, warps, splitk) for ``m`` rows, or None when hipBLASLt
    bf16 is the measured winner or the shape has no row (the caller then needs a bf16 copy).
    ``fp8_in``: the activation arrives as fp8 + ue8m0 (no quant to pay)."""
    entry = _large_m_table(fp8_in).get(f"{gfx_name()}:{n}:{k}:{large_m_bucket(m)}")
    if entry is None or entry == HIPBLASLT_BF16:
        return None
    assert entry.startswith("ds:"), entry
    bm, bn, bk, warps, sk = (int(v) for v in entry[3:].split(","))
    return bm, bn, bk, warps, sk


def weight_needs_bf16_copy(n: int, k: int) -> bool:
    """Some M > 32 bucket of this shape runs hipBLASLt bf16 for either activation encoding
    (or the shape is untuned)."""
    keys = [f"{gfx_name()}:{n}:{k}:{b}" for b in LARGE_M_BUCKETS]
    for fp8_in in (False, True):
        table = _large_m_table(fp8_in)
        if not all(key in table for key in keys):
            return True
        if any(table[key] == HIPBLASLT_BF16 for key in keys):
            return True
    return False


def native_consumer_wants_fp8(m: int, n: int, k: int) -> bool:
    """Whether a fused producer should hand the native route fp8 + ue8m0 for ``m`` tokens of
    a consumer with weight ``[n, k]``: the skinny kernel's range, or an M bucket whose measured
    winner with a free fp8 input is the dot_scaled tile (hipBLASLt bf16 wants bf16)."""
    if m <= MXFP8_GEMV_MAX_TOKENS:
        return True
    return large_m_plan(m, n, k, fp8_in=True) is not None


def native_route_supports(n: int, k: int) -> bool:
    """Shapes the native route serves: 16-row tiles, 32-row block scales, 128-wide K steps."""
    return n % 32 == 0 and k % 128 == 0


def prepare_mxfp8_native_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_size
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """fp8 ``[N, K]`` + fp32 block scales -> (shuffled fp8 bytes ``[N/16, K/128, 2048]``,
    ue8m0 scale bytes ``[N/32, K/32]``, bf16 copy or None)."""
    n, k = weight.shape
    assert tuple(block_size) == (32, 32), block_size
    assert native_route_supports(n, k), (n, k)
    shuffled = shuffle_mxfp8_weight(weight.contiguous())
    scale_e8m0 = ue8m0_weight_scale(weight_scale)
    weight_bf16 = None
    if weight_needs_bf16_copy(n, k):
        weight_bf16 = dequant_block_fp8_weight_to_bf16(weight, weight_scale, block_size)
    return shuffled, scale_e8m0, weight_bf16


# Triton tl.dot_scaled GEMM over the shuffled weight
@triton.jit
def _mxfp8_shuffled_gemm_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xsm,
    stride_wsn,
    stride_om,
    k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_F32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(
        2
    )  # split-K partition; partials are summed in fixed order outside
    k0 = pid_k * k_per_split
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # a 2 KB lane-order block is (g1, g0, row, half, 16 B) with lane 16 * (2 g1 + g0) + row holding
    # K [32 g1 + 16 g0 + 64 half, +16); put it back into row-major [BLOCK_N, BLOCK_K] in registers
    T: tl.constexpr = BLOCK_N // 16
    S: tl.constexpr = BLOCK_K // 128
    nsteps = K // 128
    blk = tl.arange(0, T * S)
    blk_tile = pid_n * T + blk // S
    blk_mask = blk_tile < N // 16  # the last N block may hold fewer 16-row tiles
    blk_off = blk_tile * (nsteps * 2048) + (blk % S + k0 // 128) * 2048
    w_ptrs = w_ptr + blk_off[:, None] + tl.arange(0, 2048)[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k0 + offs_k[None, :]
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + k0 // 32 + offs_sk[None, :]
    ws_ptrs = (
        ws_ptr + (offs_n // 32)[:, None] * stride_wsn + k0 // 32 + offs_sk[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_per_split // BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0)
        w_raw = tl.load(w_ptrs, mask=blk_mask[:, None], other=0)  # [T * S, 2048]
        w7 = tl.reshape(w_raw, (T, S, 2, 2, 16, 2, 16))  # (t, s, g1, g0, row, half, e)
        w7 = tl.permute(w7, (0, 4, 1, 5, 2, 3, 6))  # (t, row, s, half, g1, g0, e)
        w = tl.reshape(w7, (BLOCK_N, BLOCK_K))
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=127)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=127)
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc)
        x_ptrs += BLOCK_K
        xs_ptrs += BLOCK_K // 32
        w_ptrs += S * 2048
        ws_ptrs += BLOCK_K // 32

    o_ptrs = (
        out_ptr
        + pid_k.to(tl.int64) * M * N
        + offs_m[:, None] * stride_om
        + offs_n[None, :]
    )
    if OUT_F32:
        tl.store(o_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
    else:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def mxfp8_shuffled_gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_shuffled: torch.Tensor,
    weight_scale_e8m0: torch.Tensor,
    tile: Tuple[int, int, int, int],
    splitk: int,
) -> torch.Tensor:
    """``[M, N] bf16 = xq[M, K] fp8 . W^T`` over the shuffled fp8 weight with the table's
    ``tile`` (BM, BN, BK, warps).  With ``splitk > 1`` the K partitions' fp32 partials are
    summed in partition order (deterministic)."""
    m, k = xq.shape
    n = weight_shuffled.shape[0] * 16
    bm, bn, bk, warps = tile
    if k % bk != 0:
        bk = 128
    assert (k // bk) % splitk == 0, (k, bk, splitk)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), splitk)
    if splitk == 1:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=xq.device)
    else:
        out = torch.empty(splitk, m, n, dtype=torch.float32, device=xq.device)
    _mxfp8_shuffled_gemm_kernel[grid](
        xq.view(torch.uint8),
        xs,
        weight_shuffled,
        weight_scale_e8m0,
        out,
        m,
        n,
        k,
        xq.stride(0),
        xs.stride(0),
        weight_scale_e8m0.stride(0),
        n,
        k // splitk,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        OUT_F32=splitk > 1,
        num_warps=warps,
        num_stages=2,
    )
    if splitk > 1:
        out = out.sum(0).to(torch.bfloat16)  # fixed partition order
    return out


# The route
def native_route_plan(
    m: int, n: int, k: int, has_bf16_copy: bool, fp8_in: bool = False
) -> str:
    """Which kernel serves ``m`` tokens: 'gemv', 'hipblaslt_bf16' or 'dot_scaled'."""
    if m <= MXFP8_GEMV_MAX_TOKENS:
        return "gemv"
    if has_bf16_copy and large_m_plan(m, n, k, fp8_in) is None:
        return HIPBLASLT_BF16
    return "dot_scaled"


def mxfp8_native_blockscaled_linear(
    input: torch.Tensor,
    weight_shuffled: torch.Tensor,
    weight_scale_e8m0: torch.Tensor,
    weight_bf16: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    input_on_fp8_grid: bool = False,
) -> torch.Tensor:
    """Dense linear of the native route.  ``input`` is bf16 (plain, or on the fp8 grid
    when ``input_on_fp8_grid``), or fp8 e4m3 with ``input_scale`` ue8m0 ``[M, K/32]``."""
    input_2d = input.view(-1, input.shape[-1])
    m, k = input_2d.shape
    n = weight_shuffled.shape[0] * 16
    if m == 0:
        out = input_2d.new_empty((0, n), dtype=torch.bfloat16)
    else:
        plan = native_route_plan(
            m, n, k, weight_bf16 is not None, input_scale is not None
        )
        if input_scale is not None:
            assert input_2d.dtype == torch.float8_e4m3fn, input_2d.dtype
            xq, xs = input_2d.contiguous(), input_scale
        else:
            xq = xs = None
            input_2d = input_2d.to(torch.bfloat16).contiguous()
        if plan == "gemv":
            if xq is not None:
                out = mxfp8_gemv(xq, weight_shuffled, weight_scale_e8m0, xs)
            else:
                out = mxfp8_gemv(input_2d, weight_shuffled, weight_scale_e8m0)
        elif plan == HIPBLASLT_BF16:
            if xq is not None:
                x = (
                    xq.float() * torch.exp2(xs.float() - 127).repeat_interleave(32, 1)
                ).to(torch.bfloat16)
            elif input_on_fp8_grid:
                x = input_2d
            else:
                x = fake_quant_fp8_activation(input_2d)
            out = torch.nn.functional.linear(x, weight_bf16)
        else:
            fp8_in = xq is not None
            if xq is None:
                xq, xs = mxfp8_e4m3_quantize(input_2d)
            # a weight without a bf16 copy has a table row for every bucket, so the plan is never None here
            tile = large_m_plan(m, n, k, fp8_in)
            assert tile is not None, (m, n, k, fp8_in)
            out = mxfp8_shuffled_gemm(
                xq, xs, weight_shuffled, weight_scale_e8m0, tile[:4], tile[4]
            )
    if bias is not None:
        out = out + bias
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out.view(*input.shape[:-1], n)


__all__ = [
    "ALL_CONFIGS",
    "CONFIG_FILE",
    "Fp8GridActivation",
    "GemvConfig",
    "MXFP8_GEMV_MAX_TOKENS",
    "M_BUCKETS",
    "default_config",
    "gfx_name",
    "mxfp8_gemv",
    "select_config",
    "shuffle_mxfp8_weight",
    "ue8m0_weight_scale",
    "HIPBLASLT_BF16",
    "LARGE_M_BUCKETS",
    "large_m_bucket",
    "large_m_plan",
    "native_consumer_wants_fp8",
    "weight_needs_bf16_copy",
    "mxfp8_native_blockscaled_linear",
    "mxfp8_shuffled_gemm",
    "native_route_plan",
    "native_route_supports",
    "prepare_mxfp8_native_weight",
]
