"""Fused HC low-rank mix for decode-size batches.

`GatedResidual._mix_compute` lowers to a five-kernel chain per call
(down GEMV + splitK reduce, silu, up GEMV, sigmoid-mul-mean); at bs=1
speculative decode that chain runs ~100 times per iteration on the GPU
critical path between allreduces, and at these sizes every kernel is
latency-bound, so the win comes from kernel count, not bandwidth.

One persistent kernel replaces the chain.  A grid of one CTA per SM is
resident by construction, which makes the software grid barrier
deadlock-free:

* phase 0 — zero the fp32 accumulator ``t_raw`` (strided across CTAs)
* phase A — grid-strided (n-block, k-chunk) tiles of ``x @ W_down^T``
  accumulated into ``t_raw`` with device-scope atomics
* phase B — grid-strided output blocks: ``t = silu(t_raw / hc)`` on the
  fly, one ``tl.dot`` covering all hc groups, then
  ``out_j = mean_g(sigmoid(t @ W_up[g,j]^T) * x[g,j])``

The barrier counters are reset by the last CTA to finish, so a captured
CUDA graph replays with the buffers back in their initial state.

Row counts beyond ``_FUSED_MIX_MAX_ROWS`` (prefill) keep the
torch.compile path, which uses proper GEMM kernels.

V100 uses a separate two-kernel decode path for Qwen's exact HC shape.  The
first kernel replaces cuBLAS's split-K down GEMV and fuses SiLU; the second
computes the up projection by hidden coordinate and fuses sigmoid-mul-mean.
This avoids both split-K reduction and pointwise launches without global
barriers or atomics.  It covers ``1 <= M <= _SM70_MAX_MIX_ROWS`` so it also
runs in MTP verify, which is where most decode time goes: at bs=1 a decode
round is 96 mix calls at M=4 (48 layers x 2 modules) against 6 at M=1.
The exact FP16 shape uses native CUDA with vector loads and four parallel
branch reductions per warp. Two/four-row verification reuses each weight
load across the rows. SGLANG_SM70_HC_NATIVE=0 selects the previous paths;
SGLANG_SM70_MTP_HC=0 disables only the batched native path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.srt.environ import envs

_FUSED_MIX_MAX_ROWS = 16


@triton.jit
def _sm70_hc_down_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    inv_hc,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-CTA FP16 GEMV + SiLU for Qwen HC's fixed down projection."""
    output_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offsets
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + output_id * K + k, mask=k < K, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)
    # Match F.linear's FP16 output boundary before the original divide + SiLU.
    down = acc.to(tl.float16).to(tl.float32) * inv_hc
    tl.store(out_ptr + output_id, down * tl.sigmoid(down))


@triton.jit
def _sm70_hc_down_gemv_rows_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    inv_hc,
    num_rows,
    K: tl.constexpr,
    LOWRANK: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Multi-row form of the above, for MTP verify.

    Weight-stationary: the weight row is read once and dotted against all ROWS
    rows, so the extra rows add arithmetic but no traffic. The products are
    accumulated elementwise and reduced once at the end rather than per
    k-chunk; the per-chunk form costs one cross-thread tree reduction per row
    per chunk, which is what makes the naive widening slower than cuBLAS.
    Measured on V100 at M=4: 16.12 us per-chunk vs 13.53 us deferred.
    """
    output_id = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((ROWS, BLOCK_K), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        mask_k = k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + output_id * K + k, mask=mask_k, other=0.0).to(tl.float32)
        acc += x * w[None, :]
    # Match F.linear's FP16 output boundary before the original divide + SiLU.
    down = tl.sum(acc, axis=1).to(tl.float16).to(tl.float32) * inv_hc
    tl.store(
        out_ptr + offs_m * LOWRANK + output_id,
        down * tl.sigmoid(down),
        mask=mask_m,
    )


# Above 4 rows the single-CTA-per-output-row shape loses to cuBLAS, which is
# nearly M-independent here. Measured on V100 for the down+up pair against the
# compiled fallback: M=1 21.4 us vs 32.8, M=4 25.8 vs 31.7, M=12 124.4 vs 33.5.
# bs=1 verify is M=4, which is 95% of production decode.
_SM70_MAX_MIX_ROWS = 4


def sm70_hc_down_gemv_silu(
    x: torch.Tensor, w_down: torch.Tensor, hc_count: int
) -> torch.Tensor:
    if (
        hc_count == 4
        and x.ndim == 2
        and x.shape[0] in (1, 2, 4)
        and x.shape[1] == 10240
        and w_down.shape == (320, 10240)
        and x.data_ptr() % 16 == 0
        and w_down.data_ptr() % 16 == 0
        and envs.SGLANG_SM70_HC_NATIVE.get()
        and (x.shape[0] == 1 or envs.SGLANG_SM70_MTP_HC.get())
    ):
        from sglang.kernels.ops.elementwise.sm70_hc_mix import hc_down

        return hc_down(x, w_down)
    rows = x.shape[0]
    out = torch.empty((rows, w_down.shape[0]), dtype=x.dtype, device=x.device)
    if rows == 1:
        _sm70_hc_down_gemv_kernel[(w_down.shape[0],)](
            x,
            w_down,
            out,
            1.0 / hc_count,
            K=x.shape[1],
            BLOCK_K=2048,
            num_warps=4,
        )
        return out
    _sm70_hc_down_gemv_rows_kernel[(w_down.shape[0],)](
        x,
        w_down,
        out,
        1.0 / hc_count,
        rows,
        K=x.shape[1],
        LOWRANK=w_down.shape[0],
        ROWS=triton.next_power_of_2(rows),
        BLOCK_K=2048,
        num_warps=8,
    )
    return out


def sm70_hc_down_gemv_silu_supported(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (7, 0)
        and x.dtype == torch.float16
        and w_down.dtype == torch.float16
        and w_up.dtype == torch.float16
        and x.dim() == 2
        and 1 <= x.shape[0] <= _SM70_MAX_MIX_ROWS
        and x.shape[1] == 10240
        and w_down.shape == (320, 10240)
        and w_up.shape == (10240, 320)
        and x.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


@triton.jit
def _sm70_hc_up_gemv_reduce_kernel(
    activated_down_ptr,
    x_ptr,
    w_up_ptr,
    out_ptr,
    LOWRANK: tl.constexpr,
    HS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    hidden_id = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    activated_down = tl.load(activated_down_ptr + r, mask=r < LOWRANK, other=0.0).to(
        tl.float32
    )
    out = 0.0
    for branch in range(HC):
        row = branch * HS + hidden_id
        w = tl.load(w_up_ptr + row * LOWRANK + r, mask=r < LOWRANK, other=0.0).to(
            tl.float32
        )
        up = tl.sum(activated_down * w, axis=0).to(tl.float16).to(tl.float32)
        gate = tl.sigmoid(up)
        x = tl.load(x_ptr + row).to(tl.float32)
        out += gate * x
    tl.store(out_ptr + hidden_id, out / HC)


@triton.jit
def _sm70_hc_up_gemv_reduce_rows_kernel(
    activated_down_ptr,
    x_ptr,
    w_up_ptr,
    out_ptr,
    num_rows,
    LOWRANK: tl.constexpr,
    HS: tl.constexpr,
    HC: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Multi-row form: one CTA per hidden coordinate, ROWS rows in registers."""
    hidden_id = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    r = tl.arange(0, BLOCK_R)
    mask_r = r < LOWRANK
    activated_down = tl.load(
        activated_down_ptr + offs_m[:, None] * LOWRANK + r[None, :],
        mask=mask_m[:, None] & mask_r[None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.zeros((ROWS,), dtype=tl.float32)
    for branch in range(HC):
        row = branch * HS + hidden_id
        w = tl.load(w_up_ptr + row * LOWRANK + r, mask=mask_r, other=0.0).to(tl.float32)
        up = tl.sum(activated_down * w[None, :], axis=1).to(tl.float16).to(tl.float32)
        gate = tl.sigmoid(up)
        x = tl.load(x_ptr + offs_m * (HC * HS) + row, mask=mask_m, other=0.0).to(
            tl.float32
        )
        out += gate * x
    tl.store(out_ptr + offs_m * HS + hidden_id, out / HC, mask=mask_m)


def sm70_hc_up_gemv_reduce(
    activated_down: torch.Tensor,
    x: torch.Tensor,
    w_up: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    if (
        hc_count == 4
        and hidden_size == 2560
        and x.ndim == 2
        and x.shape[0] in (1, 2, 4)
        and x.shape[1] == 10240
        and activated_down.shape == (x.shape[0], 320)
        and w_up.shape == (10240, 320)
        and activated_down.data_ptr() % 16 == 0
        and w_up.data_ptr() % 16 == 0
        and envs.SGLANG_SM70_HC_NATIVE.get()
        and (x.shape[0] == 1 or envs.SGLANG_SM70_MTP_HC.get())
    ):
        from sglang.kernels.ops.elementwise.sm70_hc_mix import hc_up

        return hc_up(activated_down, x, w_up)
    rows = x.shape[0]
    out = torch.empty((rows, hidden_size), dtype=x.dtype, device=x.device)
    if rows == 1:
        _sm70_hc_up_gemv_reduce_kernel[(hidden_size,)](
            activated_down,
            x,
            w_up,
            out,
            LOWRANK=activated_down.shape[1],
            HS=hidden_size,
            HC=hc_count,
            BLOCK_R=512,
            num_warps=1,
        )
        return out
    _sm70_hc_up_gemv_reduce_rows_kernel[(hidden_size,)](
        activated_down,
        x,
        w_up,
        out,
        rows,
        LOWRANK=activated_down.shape[1],
        HS=hidden_size,
        HC=hc_count,
        ROWS=triton.next_power_of_2(rows),
        BLOCK_R=512,
        num_warps=1,
    )
    return out


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
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
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


_counters_cache = {}


def _get_counters(device: torch.device) -> torch.Tensor:
    buf = _counters_cache.get(device)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[device] = buf
    return buf


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
    # SM70 is a pathological case for this persistent implementation.  On a
    # V100 the device-wide atomics and software grid barriers make one Qwen
    # 3.8 decode-size mix take ~1.12 ms, while the ordinary two-GEMV chain is
    # ~36 us under CUDA graph replay.  Keep V100 on the compiled GEMV path.
    if hyper_input_normed.is_cuda and torch.cuda.get_device_capability(
        hyper_input_normed.device
    ) == (7, 0):
        return False
    # The persistent kernel accumulates the down projection with
    # device-scope atomics, so summation order varies across replays.
    if _deterministic_inference():
        return False
    return (
        hyper_input_normed.is_cuda
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_down.dtype == hyper_input_normed.dtype
        and w_up.dtype == hyper_input_normed.dtype
        and hyper_input_normed.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input_normed.dim() == 2
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
    rows_pad = 16
    device = hyper_input_normed.device
    num_ctas = current_arch().num_sms
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=hyper_input_normed.dtype, device=device)
    if rows == 0:
        return out
    _hc_mix_persistent_kernel[(num_ctas,)](
        hyper_input_normed,
        w_down,
        w_up,
        t_raw,
        out,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=32,
        BLOCK_R=64,
        num_warps=8,
    )
    return out
