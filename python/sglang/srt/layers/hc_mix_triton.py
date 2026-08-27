"""Fused HC low-rank mix for decode-size batches.

`GatedResidual._mix_compute` lowers to a five-kernel chain per call
(down GEMV + splitK reduce, silu, up GEMV, sigmoid-mul-mean); at bs=1
speculative decode that chain runs ~100 times per iteration on the GPU
critical path between allreduces, and at these sizes every kernel is
latency-bound, so the win comes from kernel count, not bandwidth.

One persistent kernel replaces the chain. The launch grid is capped at the
device SM count, which makes the software grid barrier deadlock-free:

* phase 0 — zero the fp32 accumulator ``t_raw`` (strided across CTAs)
* phase A — grid-strided (n-block, k-chunk) tiles of ``x @ W_down^T``
  accumulated into ``t_raw`` with device-scope atomics
* phase B — grid-strided output blocks: ``t = silu(t_raw / hc)`` on the
  fly, one ``tl.dot`` covering all hc groups, then
  ``out_j = mean_g(sigmoid(t @ W_up[g,j]^T) * x[g,j])``

The barrier counters are reset by the last CTA to finish, so a captured
CUDA graph replays with the buffers back in their initial state.

SM120 uses tuned launch geometry through 64 rows. Other devices keep the
16-row limit. Larger prefill batches use the torch.compile path, which uses
proper GEMM kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

_FUSED_MIX_MAX_ROWS = 16
_SM120_FUSED_MIX_MAX_ROWS = 64


@dataclass(frozen=True)
class _HCMixConfig:
    max_rows: int
    rows_pad: int
    num_ctas: int
    block_k: int
    num_warps: int
    static_dims: bool


def _select_hc_mix_config(
    num_rows: int, capability: tuple[int, int], sm_count: int
) -> _HCMixConfig | None:
    if capability == (12, 0):
        num_ctas = min(80, sm_count)
        if num_rows <= 1:
            return _HCMixConfig(1, 16, num_ctas, 256, 4, True)
        if num_rows <= 16:
            return _HCMixConfig(16, 16, num_ctas, 256, 4, True)
        if num_rows <= _SM120_FUSED_MIX_MAX_ROWS:
            return _HCMixConfig(_SM120_FUSED_MIX_MAX_ROWS, 64, num_ctas, 128, 8, True)
        return None
    if num_rows <= _FUSED_MIX_MAX_ROWS:
        return _HCMixConfig(
            _FUSED_MIX_MAX_ROWS,
            _FUSED_MIX_MAX_ROWS,
            sm_count,
            256,
            8,
            False,
        )
    return None


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _hc_mix_persistent_kernel(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    num_rows,
    num_ctas,
    inv_hc,
    K,
    LOWRANK,
    HS,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    zero_span = ROWS * LOWRANK
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, num_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    _grid_barrier(counters_ptr + 0, num_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = tl.dot(xt, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            )
            acc = tl.dot(t, tl.trans(w), acc)
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


@triton.jit
def _hc_mix_persistent_kernel_sm120(
    x_ptr,
    w_down_ptr,
    w_up_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    num_rows,
    num_ctas,
    inv_hc,
    K: tl.constexpr,
    LOWRANK: tl.constexpr,
    HS: tl.constexpr,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    _hc_mix_persistent_kernel(
        x_ptr,
        w_down_ptr,
        w_up_ptr,
        t_raw_ptr,
        out_ptr,
        counters_ptr,
        num_rows,
        num_ctas,
        inv_hc,
        K,
        LOWRANK,
        HS,
        ROWS,
        HC,
        BLOCK_N,
        BLOCK_K,
        BLOCK_J,
        BLOCK_R,
    )


_counters_cache = {}


def _get_counters(device: torch.device) -> torch.Tensor:
    buf = _counters_cache.get(device)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[device] = buf
    return buf


@lru_cache(maxsize=None)
def _get_hc_mix_config(num_rows: int, device: torch.device) -> _HCMixConfig | None:
    props = torch.cuda.get_device_properties(device)
    return _select_hc_mix_config(
        num_rows,
        (props.major, props.minor),
        props.multi_processor_count,
    )


_deterministic_inference_cached = None


def _deterministic_inference() -> bool:
    global _deterministic_inference_cached
    if _deterministic_inference_cached is None:
        try:
            from sglang.srt.server_args import get_global_server_args

            _deterministic_inference_cached = bool(
                get_global_server_args().enable_deterministic_inference
            )
        except Exception:
            _deterministic_inference_cached = False
    return _deterministic_inference_cached


def fused_hc_mix_supported(
    hyper_input_normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    # The persistent kernel accumulates the down projection with
    # device-scope atomics, so summation order varies across replays.
    if _deterministic_inference():
        return False
    if not hyper_input_normed.is_cuda or hyper_input_normed.dim() != 2:
        return False
    device = hyper_input_normed.device
    return (
        hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_down.dtype == hyper_input_normed.dtype
        and w_up.dtype == hyper_input_normed.dtype
        and _get_hc_mix_config(hyper_input_normed.shape[0], device) is not None
        and hyper_input_normed.shape[1] % 2048 == 0
        and hyper_input_normed.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


def fused_hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    device = hyper_input_normed.device
    config = _get_hc_mix_config(rows, device)
    if config is None:
        raise ValueError(f"unsupported HC mix row count: {rows}")
    t_raw = torch.empty((config.rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    kernel = (
        _hc_mix_persistent_kernel_sm120
        if config.static_dims
        else _hc_mix_persistent_kernel
    )
    kernel[(config.num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        t_raw,
        out,
        _get_counters(device),
        rows,
        config.num_ctas,
        1.0 / hc,
        K=k,
        LOWRANK=lowrank,
        HS=hs,
        ROWS=config.rows_pad,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=config.block_k,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=config.num_warps,
    )
    return out
