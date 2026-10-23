# SPDX-License-Identifier: Apache-2.0
"""gfx950 native MXFP8 dense route for 32x32-block ue8m0 fp8 checkpoints: the weight stays fp8 in
scaled-MFMA lane order, K zero-padded to a multiple of 128; M <= 32 runs the skinny gemv kernel,
larger M the tl.dot_scaled GEMM with the tiles tuned in mxfp8_gemv_gfx95_configs.json."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Sequence, Tuple

import msgspec
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, empty_sentinel, load_jit, make_cpp_args
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_quantize

# M <= 32: the scaled-MFMA skinny kernel (deepseek_v4/mxfp8_gemv_gfx95.cuh)
_GEMV_MAX_TOKENS = 32
_TILE_N = 16
_STEP_K = 128
_LANES = 64
_LANE_BYTES = 32
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "mxfp8_gemv_gfx95_configs.json")
# Token-count buckets of the config table: a config tuned at the bucket's upper bound.
_M_BUCKETS = (1, 2, 4, 8, 16, 32)


class _GemvConfig(msgspec.Struct, frozen=True):
    waves: int  # waves per workgroup: 4, 8, 16
    steps: int  # 128-K steps in flight per wave: 1, 2, 4
    rows: int  # weight rows per wave tile: 16, 32
    tokens: int  # token columns per wave tile: 16, 32 (M <= tokens)

    @staticmethod
    def parse(key: str) -> _GemvConfig:
        # trailing k: the waves split K and reduce through LDS, the regime the kernel implements
        m = re.fullmatch(r"w(\d+)s(\d+)r(\d+)t(\d+)k", key)
        assert m, key
        return _GemvConfig(int(m[1]), int(m[2]), int(m[3]), int(m[4]))

    def valid_for(self, m: int, n: int) -> bool:
        return (
            self.waves in (4, 8, 16)
            and self.steps in (1, 2, 4)
            and self.rows in (16, 32)
            and self.tokens in (16, 32)
            and m <= self.tokens
            and n % self.rows == 0
        )


def _default_config(m: int) -> _GemvConfig:
    """Shapes without a tuned row. Measured on gfx950 over K 2048 to 16384 and M 1 to 32:
    8 waves, 1 step stays within ~10% of each shape's best 16-row config."""
    return _GemvConfig(8, 1, 16, 16 if m <= 16 else 32)


def _m_bucket(m: int) -> int:
    for b in _M_BUCKETS:
        if m <= b:
            return b
    raise ValueError(f"M={m} exceeds the skinny kernel's {_GEMV_MAX_TOKENS} tokens")


@cache_once
def _config_table(section: str) -> Dict[str, str]:
    """One {'gfx950:N:K:M_bucket': entry} section of _CONFIG_FILE."""
    with open(_CONFIG_FILE) as f:
        return json.load(f)[section]


@cache_once
def _gfx_name() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


@cache_once
def _select_config(bucket: int, n: int, k: int) -> _GemvConfig:
    """The tuned configuration for (gfx, N, K, M bucket), else the heuristic."""
    key = f"{_gfx_name()}:{n}:{k}:{bucket}"
    entry = _config_table("configs").get(key)
    if entry is None:
        return _default_config(bucket)
    cfg = _GemvConfig.parse(entry)
    assert cfg.valid_for(bucket, n), f"tuned entry {key}: {entry} cannot serve it"
    return cfg


@cache_once
def _jit_mxfp8_gemv_module(cfg: _GemvConfig, x_bf16: bool):
    args = make_cpp_args(cfg.waves, cfg.steps, cfg.rows, cfg.tokens, x_bf16)
    return load_jit(
        "dpsk_v4_mxfp8_gemv_gfx95",
        *args,
        cuda_files=["deepseek_v4/mxfp8_gemv_gfx95.cuh"],
        cuda_wrappers=[("run", f"Mxfp8GemvGfx950Kernel<{args}>::run")],
    )


def shuffle_mxfp8_weight(weight: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 [N, K] -> [N/16, K/128, 2048] uint8 in the gfx950 16x16x128 scaled-MFMA
    lane order: for tile t, K step s and lane l = 16 * g + r, the lane's 32 bytes are
    W[16t + r][128s + 32(g/2) + 16(g%2) : +16] followed by the same 16 bytes 64 K later."""
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
    """fp32 power-of-two block scales [N/32, K/32] -> ue8m0 exponent bytes (exact)."""
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
) -> torch.Tensor:
    """out[M, N] bf16 = x[M, K] . W^T on the gfx950 scaled matrix core; x is fp8 e4m3 with
    x_scale ue8m0 [M, K/32] or bf16 quantized in-kernel."""
    # shapes, contiguity and the M range are checked by the launcher's TensorMatcher
    m, k = x.shape
    n = weight_shuffled.shape[0] * _TILE_N
    x_bf16 = x.dtype == torch.bfloat16
    if x_bf16:
        x_scale = empty_sentinel(x.device, torch.uint8)
    else:
        # the kernel takes the fp8 bytes, so the launcher cannot tell fp8 from uint8
        assert x.dtype == torch.float8_e4m3fn and x_scale is not None, x.dtype
        x_scale = x_scale.contiguous()
        x = x.view(torch.uint8)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    config = _select_config(_m_bucket(m), n, k)
    _jit_mxfp8_gemv_module(config, x_bf16).run(
        weight_shuffled, weight_scale_ue8m0, x, x_scale, out
    )
    return out


# M > 32: the Triton dot_scaled GEMM with the tile the "large_m" table gives the M bucket
_LARGE_M_BUCKETS = (64, 128, 256, 1024, 4096, 8192, 16384)
# Shapes without a tuned row. BK 128 divides every K the route admits; the large-M tile won
# every tuned shape from 4096 rows up, the small-M one is untuned.
_DEFAULT_SMALL_M_TILE = (128, 64, 128, 4, 1)
_DEFAULT_LARGE_M_TILE = (128, 256, 128, 8, 1)


def _large_m_bucket(m: int) -> int:
    for b in _LARGE_M_BUCKETS:
        if m <= b:
            return b
    return _LARGE_M_BUCKETS[-1]


def _large_m_tile(m: int, n: int, k: int) -> Tuple[int, int, int, int, int]:
    """The dot_scaled tile (BM, BN, BK, warps, split_k) of m's bucket."""
    bucket = _large_m_bucket(m)
    entry = _config_table("large_m").get(f"{_gfx_name()}:{n}:{k}:{bucket}")
    if entry is None:
        return _DEFAULT_SMALL_M_TILE if bucket <= 1024 else _DEFAULT_LARGE_M_TILE
    return tuple(int(v) for v in entry.split(","))


def native_route_supports(n: int, k: int) -> bool:
    """Shapes the native route serves: whole 32-row / 32-column scale blocks (a K tail short
    of the 128-wide step is zero-padded in the weight and masked in the activation)."""
    return n % 32 == 0 and k % 32 == 0


def prepare_mxfp8_native_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_size: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp8 [N, K] + fp32 block scales -> (shuffled fp8 bytes [N/16, KP/128, 2048],
    ue8m0 scale bytes [N/32, KP/32]), KP = K rounded up to 128 with zero weight."""
    n, k = weight.shape
    assert tuple(block_size) == (32, 32), block_size
    assert native_route_supports(n, k), (n, k)
    pad = -k % _STEP_K
    weight = F.pad(weight.contiguous().view(torch.uint8), (0, pad))
    # a padded block's weight is zero, so its scale only needs to be finite
    weight_scale = F.pad(weight_scale.float(), (0, pad // 32), value=1.0)
    return (
        shuffle_mxfp8_weight(weight.view(torch.float8_e4m3fn)),
        ue8m0_weight_scale(weight_scale),
    )


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
    K_PAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_F32: tl.constexpr,
    K_TAIL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # split-K partition; partials are summed in fixed order outside
    pid_k = tl.program_id(2)
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
    nsteps = K_PAD // 128
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
    for it in range(0, k_per_split // BLOCK_K):
        if K_TAIL:
            # the weight's zero-padded K tail meets a masked activation
            k_cur = k0 + it * BLOCK_K
            x_mask = m_mask[:, None] & (k_cur + offs_k < K)[None, :]
            xs_mask = m_mask[:, None] & (k_cur // 32 + offs_sk < K // 32)[None, :]
        else:
            x_mask = m_mask[:, None]
            xs_mask = m_mask[:, None]
        x = tl.load(x_ptrs, mask=x_mask, other=0)
        w_raw = tl.load(w_ptrs, mask=blk_mask[:, None], other=0)  # [T * S, 2048]
        w7 = tl.reshape(w_raw, (T, S, 2, 2, 16, 2, 16))  # (t, s, g1, g0, row, half, e)
        w7 = tl.permute(w7, (0, 4, 1, 5, 2, 3, 6))  # (t, row, s, half, g1, g0, e)
        w = tl.reshape(w7, (BLOCK_N, BLOCK_K))
        xs = tl.load(xs_ptrs, mask=xs_mask, other=127)
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


def _mxfp8_shuffled_gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_shuffled: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    tile: Tuple[int, int, int, int],
    split_k: int,
) -> torch.Tensor:
    """[M, N] bf16 = xq[M, K] fp8 . W^T over the shuffled fp8 weight with the table's
    tile (BM, BN, BK, warps). With split_k > 1 the K partitions' fp32 partials are
    summed in partition order (deterministic)."""
    m, k = xq.shape
    n = weight_shuffled.shape[0] * 16
    k_pad = weight_shuffled.shape[1] * _STEP_K
    bm, bn, bk, warps = tile
    assert k_pad % bk == 0 and (k_pad // bk) % split_k == 0, (k_pad, bk, split_k)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), split_k)
    if split_k == 1:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=xq.device)
    else:
        out = torch.empty(split_k, m, n, dtype=torch.float32, device=xq.device)
    _mxfp8_shuffled_gemm_kernel[grid](
        xq.view(torch.uint8),
        xs,
        weight_shuffled,
        weight_scale_ue8m0,
        out,
        m,
        n,
        k,
        xq.stride(0),
        xs.stride(0),
        weight_scale_ue8m0.stride(0),
        n,
        k_pad // split_k,
        k_pad,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        OUT_F32=split_k > 1,
        K_TAIL=k != k_pad,
        num_warps=warps,
        num_stages=2,
    )
    if split_k > 1:
        out = out.sum(0).to(torch.bfloat16)  # fixed partition order
    return out


def mxfp8_native_blockscaled_linear(
    input: torch.Tensor,
    weight_shuffled: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dense linear of the native route. input is bf16 (a bf16 input already on the fp8 grid
    re-encodes exactly), or fp8 e4m3 with input_scale ue8m0 [M, K/32]."""
    input_2d = input.view(-1, input.shape[-1])
    m, k = input_2d.shape
    n = weight_shuffled.shape[0] * 16
    assert k % 32 == 0 and weight_shuffled.shape[1] == -(-k // _STEP_K), (
        f"input K {k} does not match the weight's {weight_shuffled.shape[1]} K steps"
    )
    if input_scale is not None:
        assert input_2d.dtype == torch.float8_e4m3fn, input_2d.dtype
        xq, xs = input_2d.contiguous(), input_scale
    else:
        input_2d = input_2d.to(torch.bfloat16).contiguous()
    if m == 0:
        out = input_2d.new_empty((0, n), dtype=torch.bfloat16)
    elif m <= _GEMV_MAX_TOKENS:
        if input_scale is not None:
            out = mxfp8_gemv(xq, weight_shuffled, weight_scale_ue8m0, xs)
        else:
            out = mxfp8_gemv(input_2d, weight_shuffled, weight_scale_ue8m0)
    else:
        if input_scale is None:
            xq, xs = fp8_grid_quantize(input_2d)
        *tile, split_k = _large_m_tile(m, n, k)
        out = _mxfp8_shuffled_gemm(
            xq, xs, weight_shuffled, weight_scale_ue8m0, tuple(tile), split_k
        )
    if bias is not None:
        out = out + bias
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out.view(*input.shape[:-1], n)
