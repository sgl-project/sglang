"""HIP fused mHC sublayer boundary: hc_post + collapse + mixing statistics in one launch, with the
reduce + sinkhorn launched alone or hosted by the layer's next RMSNorm (``HcCoefficients``)."""

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.layernorm.mhc import _HC_MIX_DOT_PRECISION, _is_hip
from sglang.kernels.ops.quantization.rmsnorm_fake_quant_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    _row_major_2d,
    rmsnorm_fake_quant_row,
    rmsnorm_row_chunk,
)
from sglang.srt.utils.common import is_gfx95_supported

# a CTA owns every HC copy of its tile: a row's fp32 operation order depends on (H, HC), not on M
_HC_BOUNDARY_BLOCK_M = 16
_HC_BOUNDARY_BLOCK_K = 64
_HC_BOUNDARY_NUM_WARPS = 2
_HC_BOUNDARY_NUM_STAGES = 1
# the reduce + sinkhorn row uses the norm kernel's warp count whether hosted there or launched alone
_HC_SINKHORN_NUM_WARPS = 4


@triton.jit
def _hc_mix_reduce_sinkhorn_row(
    row,
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """``_hc_mix_reduce_sinkhorn_kernel`` for one row with a tree fixed by NUM_SLICES; ``scratch_ptr``
    ([m, 32] fp32) round-trips the reduced mixes so the sinkhorn starts from a plain layout."""
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]

    s_idx = tl.arange(0, SLICES_PAD)
    s_ok = s_idx < NUM_SLICES
    off = (s_idx * m + row) * MIX
    a_pre = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + j[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_post = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + HC + j[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_comb = tl.sum(
        tl.load(
            part_mix_ptr
            + off[:, None, None]
            + 2 * HC
            + jj[None, :, :] * HC
            + kk[None, :, :],
            mask=s_ok[:, None, None],
            other=0.0,
        ),
        axis=0,
    )
    sq = tl.sum(tl.load(part_sq_ptr + s_idx * m + row, mask=s_ok, other=0.0), axis=0)
    sp = scratch_ptr + row * 32
    tl.store(sp + j, a_pre)
    tl.store(sp + HC + j, a_post)
    tl.store(sp + 2 * HC + jj * HC + kk, a_comb)
    tl.debug_barrier()
    a_pre = tl.load(sp + j)
    a_post = tl.load(sp + HC + j)
    a_comb = tl.load(sp + 2 * HC + jj * HC + kk)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + rms_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + j)) + EPS
    tl.store(pre_ptr + row * HC + j, pre)
    post = 2.0 * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + j))
    tl.store(post_ptr + row * HC + j, post)

    comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)


@triton.jit
def _hc_mix_reduce_sinkhorn_vec_kernel(
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """One program per row of ``_hc_mix_reduce_sinkhorn_row``."""
    _hc_mix_reduce_sinkhorn_row(
        tl.program_id(0),
        part_mix_ptr,
        part_sq_ptr,
        scratch_ptr,
        scale_ptr,
        base_ptr,
        pre_ptr,
        post_ptr,
        comb_ptr,
        m,
        inv_k,
        rms_eps,
        MIX=MIX,
        HC=HC,
        NUM_SLICES=NUM_SLICES,
        SLICES_PAD=SLICES_PAD,
        ITERS=ITERS,
        EPS=EPS,
    )


@triton.jit
def _rmsnorm_sinkhorn_kernel(
    x_ptr,
    w_ptr,
    out_fq_ptr,
    out_norm_ptr,
    out_scale_ptr,
    K,
    stride_xm,
    stride_fm,
    stride_nm,
    stride_sm,
    eps,
    quant_eps,
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    M,
    m,
    inv_k,
    rms_eps,
    WRITE_NORM: tl.constexpr,
    EMIT_FP8: tl.constexpr,
    FAKE_QUANT: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Grid (M + m,): programs [0, M) run ``rmsnorm_fake_quant_row`` on the norm's rows, programs
    [M, M + m) ``_hc_mix_reduce_sinkhorn_row`` on the pending boundary's rows; each row keeps its
    standalone kernel's arithmetic (same tiles, same warps)."""
    pid = tl.program_id(0)
    if pid < M:
        rmsnorm_fake_quant_row(
            pid.to(tl.int64),
            x_ptr,
            w_ptr,
            x_ptr,
            out_fq_ptr,
            out_norm_ptr,
            out_scale_ptr,
            K,
            stride_xm,
            0,
            stride_fm,
            stride_nm,
            stride_sm,
            eps,
            quant_eps,
            HAS_RESIDUAL=False,
            WRITE_NORM=WRITE_NORM,
            EMIT_FP8=EMIT_FP8,
            FAKE_QUANT=FAKE_QUANT,
            CHUNK=CHUNK,
            NUM_CHUNKS=NUM_CHUNKS,
        )
    else:
        _hc_mix_reduce_sinkhorn_row(
            pid - M,
            part_mix_ptr,
            part_sq_ptr,
            scratch_ptr,
            scale_ptr,
            base_ptr,
            pre_ptr,
            post_ptr,
            comb_ptr,
            m,
            inv_k,
            rms_eps,
            MIX=MIX,
            HC=HC,
            NUM_SLICES=NUM_SLICES,
            SLICES_PAD=SLICES_PAD,
            ITERS=ITERS,
            EPS=EPS,
        )


def hc_mix_reduce_sinkhorn_vec(
    part_mix: torch.Tensor,
    part_sq: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    k: int,
    rms_eps: float,
    mix: int,
    hc_mult: int,
    num_slices: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> None:
    """Reduce the [num_slices, m, mix] partials and run the sinkhorn into pre/post/comb."""
    m = part_sq.shape[1]
    # layout round trip for the reduce + sinkhorn row (see its docstring)
    scratch = torch.empty((m, 32), dtype=torch.float32, device=part_mix.device)
    _hc_mix_reduce_sinkhorn_vec_kernel[(m,)](
        part_mix,
        part_sq,
        scratch,
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        pre,
        post,
        comb,
        m,
        1.0 / k,
        rms_eps,
        MIX=mix,
        HC=hc_mult,
        NUM_SLICES=num_slices,
        SLICES_PAD=triton.next_power_of_2(num_slices),
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        num_warps=_HC_SINKHORN_NUM_WARPS,
    )


class HcCoefficients:
    """One boundary's mixing coefficients, held as split-K partials until ``rmsnorm_with_sinkhorn``
    hosts their reduce + sinkhorn or the first access of ``pre`` / ``post`` / ``comb`` launches it."""

    def __init__(
        self,
        part_mix: torch.Tensor,
        part_sq: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        k: int,
        rms_eps: float,
        mix: int,
        hc_mult: int,
        num_slices: int,
        sinkhorn_iters: int,
        hc_eps: float,
    ):
        m = part_sq.shape[1]
        dev = part_mix.device
        self.part_mix = part_mix
        self.part_sq = part_sq
        self.hc_scale = hc_scale.float().contiguous()
        self.hc_base = hc_base.float().contiguous()
        self.k = k
        self.rms_eps = rms_eps
        self.mix = mix
        self.hc_mult = hc_mult
        self.num_slices = num_slices
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.num_rows = m
        self._pre = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
        self._post = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
        self._comb = torch.empty(m, hc_mult, hc_mult, dtype=torch.float32, device=dev)
        # layout round trip for the reduce + sinkhorn row (see its docstring)
        self.scratch = torch.empty((m, 32), dtype=torch.float32, device=dev)
        self.materialized = m == 0

    def materialize(self) -> None:
        """Run the reduce + sinkhorn alone unless a norm launch already hosted it."""
        if self.materialized:
            return
        self.materialized = True
        hc_mix_reduce_sinkhorn_vec(
            self.part_mix,
            self.part_sq,
            self.hc_scale,
            self.hc_base,
            self._pre,
            self._post,
            self._comb,
            k=self.k,
            rms_eps=self.rms_eps,
            mix=self.mix,
            hc_mult=self.hc_mult,
            num_slices=self.num_slices,
            sinkhorn_iters=self.sinkhorn_iters,
            hc_eps=self.hc_eps,
        )

    @property
    def pre(self) -> torch.Tensor:
        self.materialize()
        return self._pre

    @property
    def post(self) -> torch.Tensor:
        self.materialize()
        return self._post

    @property
    def comb(self) -> torch.Tensor:
        self.materialize()
        return self._comb

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pre, self.post, self.comb


def rmsnorm_with_sinkhorn(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    coefficients: HcCoefficients,
    *,
    quant_eps: float = 1e-10,
    emit_fp8: bool = False,
    fake_quant: bool = True,
) -> Tuple[Union[Fp8GridActivation, Mxfp8Activation, None], torch.Tensor]:
    """``rmsnorm_fake_quant_fp8(x, weight, eps)`` (the plain bf16 RMSNorm when ``fake_quant`` is
    False) with the pending reduce + sinkhorn of ``coefficients`` in the same launch; returns
    ``(fake_quant or None, norm)``."""
    assert x.dim() == 2 and x.shape[-1] % 32 == 0, x.shape
    assert weight.dim() == 1 and weight.shape[0] == x.shape[-1], weight.shape
    assert weight.dtype == x.dtype, (weight.dtype, x.dtype)
    x = _row_major_2d(x)
    weight = weight.contiguous()
    M, K = x.shape
    assert coefficients.num_rows == M, (
        f"coefficients hold {coefficients.num_rows} rows, the norm input {M}"
    )
    dev = x.device
    out_fq = (
        torch.empty(
            (M, K), dtype=torch.float8_e4m3fn if emit_fp8 else x.dtype, device=dev
        )
        if fake_quant
        else None
    )
    out_scale = (
        torch.empty((M, K // 32), dtype=torch.uint8, device=dev)
        if fake_quant and emit_fp8
        else None
    )
    out_norm = torch.empty((M, K), dtype=x.dtype, device=dev)
    if fake_quant:
        quant = (
            Mxfp8Activation(out_fq, out_scale)
            if emit_fp8
            else Fp8GridActivation(out_fq)
        )
    else:
        quant = None
    if M == 0:
        return quant, out_norm
    m = 0 if coefficients.materialized else M
    coefficients.materialized = True
    CHUNK = rmsnorm_row_chunk(K)
    _rmsnorm_sinkhorn_kernel[(M + m,)](
        x,
        weight,
        out_fq if out_fq is not None else x,
        out_norm,
        out_scale if out_scale is not None else x,
        K,
        x.stride(0),
        out_fq.stride(0) if out_fq is not None else 0,
        out_norm.stride(0),
        out_scale.stride(0) if out_scale is not None else 0,
        eps,
        quant_eps,
        coefficients.part_mix,
        coefficients.part_sq,
        coefficients.scratch,
        coefficients.hc_scale,
        coefficients.hc_base,
        coefficients._pre,
        coefficients._post,
        coefficients._comb,
        M,
        m,
        1.0 / coefficients.k,
        coefficients.rms_eps,
        WRITE_NORM=True,
        EMIT_FP8=emit_fp8,
        FAKE_QUANT=fake_quant,
        CHUNK=CHUNK,
        NUM_CHUNKS=triton.cdiv(K, CHUNK),
        MIX=coefficients.mix,
        HC=coefficients.hc_mult,
        NUM_SLICES=coefficients.num_slices,
        SLICES_PAD=triton.next_power_of_2(coefficients.num_slices),
        ITERS=coefficients.sinkhorn_iters,
        EPS=coefficients.hc_eps,
        num_warps=_HC_SINKHORN_NUM_WARPS,
    )
    return quant, out_norm


@triton.jit
def _hc_boundary_partial_kernel(
    x_ptr,
    res_ptr,
    post_in_ptr,
    comb_in_ptr,
    pre_prev_ptr,
    w_ptr,
    res_out_ptr,
    y_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    H,
    x_stride_m,
    res_stride_m,
    w_stride_n,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_POST: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
):
    """Grid (cdiv(M, BLOCK_M), H // BLOCK_K), one hidden slice of every copy per program. HAS_POST:
    ``res_out[k] = post[k]*x + sum_j comb[j,k]*res[j]`` in aiter::mhc_post's order, read back as bf16
    for the statistics; HAS_COMBINE: ``y = sum_k pre_prev[k] * copy_k`` in _hc_combine_kernel's order."""
    tl.static_assert(HC == 4, "the weight tiles are prefetched by name")
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_h = pid_t * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, MIX_PAD)
    mask_n = offs_n < MIX
    m2 = mask_m[:, None]
    w_off = offs_n[None, :] * w_stride_n + offs_h[:, None]
    # Prefetching only reorders memory issue; each tile still feeds its own dot.
    w0 = tl.load(w_ptr + w_off + 0 * H, mask=mask_n[None, :], other=0.0)
    w1 = tl.load(w_ptr + w_off + 1 * H, mask=mask_n[None, :], other=0.0)
    w2 = tl.load(w_ptr + w_off + 2 * H, mask=mask_n[None, :], other=0.0)
    w3 = tl.load(w_ptr + w_off + 3 * H, mask=mask_n[None, :], other=0.0)
    if HAS_POST:
        xa = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_h[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        r_off = offs_m[:, None] * res_stride_m + offs_h[None, :]
        r0 = tl.load(res_ptr + r_off + 0 * H, mask=m2, other=0.0).to(tl.float32)
        r1 = tl.load(res_ptr + r_off + 1 * H, mask=m2, other=0.0).to(tl.float32)
        r2 = tl.load(res_ptr + r_off + 2 * H, mask=m2, other=0.0).to(tl.float32)
        r3 = tl.load(res_ptr + r_off + 3 * H, mask=m2, other=0.0).to(tl.float32)
        # one tile load per coefficient tensor; a one-hot sum extracts a column exactly
        cj = tl.arange(0, HC)
        post_tile = tl.load(
            post_in_ptr + offs_m[:, None] * HC + cj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        cc = tl.arange(0, HC * HC)
        comb_tile = tl.load(
            comb_in_ptr + offs_m[:, None] * (HC * HC) + cc[None, :], mask=m2, other=0.0
        ).to(tl.float32)
    if HAS_COMBINE:
        pj = tl.arange(0, HC)
        pre_tile = tl.load(
            pre_prev_ptr + offs_m[:, None] * HC + pj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        y = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in tl.static_range(HC):
        if HAS_POST:
            pk = tl.sum(tl.where((cj == k)[None, :], post_tile, 0.0), axis=1)
            c0 = tl.sum(tl.where((cc == 0 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c1 = tl.sum(tl.where((cc == 1 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c2 = tl.sum(tl.where((cc == 2 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c3 = tl.sum(tl.where((cc == 3 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            v = xa * pk[:, None]
            v += r0 * c0[:, None]
            v += r1 * c1[:, None]
            v += r2 * c2[:, None]
            v += r3 * c3[:, None]
            vb = v.to(res_out_ptr.dtype.element_ty)
            tl.store(
                res_out_ptr + offs_m[:, None] * (HC * H) + k * H + offs_h[None, :],
                vb,
                mask=m2,
            )
            xt = vb.to(tl.float32)
        else:
            xt = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + k * H + offs_h[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
        if HAS_COMBINE:
            pp = tl.sum(tl.where((pj == k)[None, :], pre_tile, 0.0), axis=1)
            y += pp[:, None] * xt
        if k == 0:
            w_tile = w0
        elif k == 1:
            w_tile = w1
        elif k == 2:
            w_tile = w2
        else:
            w_tile = w3
        acc += tl.dot(xt, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(xt * xt, axis=1)
    tl.store(
        part_mix_ptr + (pid_t * M + offs_m[:, None]) * MIX + offs_n[None, :],
        acc,
        mask=m2 & mask_n[None, :],
    )
    tl.store(part_sq_ptr + pid_t * M + offs_m, sq, mask=mask_m)
    if HAS_COMBINE:
        tl.store(
            y_ptr + offs_m[:, None] * H + offs_h[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=m2,
        )


# the gfx950 prefill kernel is bitwise the Triton kernel per row: the switch below is a speed choice
_HC_BOUNDARY_PREFILL_MIN_M = 1024


def _hc_boundary_prefill_available() -> bool:
    """The prefill kernel needs gfx950 (v_permlane*_swap, 16-byte LDS DMA)."""
    return _is_hip and torch.cuda.is_available() and is_gfx95_supported()


@cache_once
def _hc_boundary_prefill_module():
    from sglang.kernels.jit.utils import load_jit

    kernel = "mhc_boundary_hip::HcBoundaryPrefillKernel"
    return load_jit(
        "hc_boundary_prefill_hip",
        cuda_files=["deepseek_v4/mhc_boundary_hip.cuh"],
        cuda_wrappers=[
            ("post_combine", f"{kernel}<true, true>::run"),
            ("stats_only", f"{kernel}<false, false>::run"),
        ],
        # the hc_post chain must stay separate multiplies and adds, as in the Triton binary
        extra_cuda_cflags=["-ffp-contract=off"],
    )


def _hc_boundary_prefill_ctas(num_row_blocks: int) -> int:
    """CTAs per hidden slice (8 waves each, one CTA per CU by LDS)."""
    return max(6, min(16, num_row_blocks // 64))


def _hc_boundary_use_prefill(
    m: int, has_post: bool, has_combine: bool, x: Optional[torch.Tensor]
) -> bool:
    if m < _HC_BOUNDARY_PREFILL_MIN_M or not _hc_boundary_prefill_available():
        return False
    if (has_post, has_combine) not in ((True, True), (False, False)):
        return False
    # The LDS DMA moves 16-byte chunks: rows of x must stay 16-byte aligned.
    return x is None or x.stride(0) % 8 == 0


def _hc_boundary_partials(
    x: Optional[torch.Tensor],
    residual: torch.Tensor,
    post_in: Optional[torch.Tensor],
    comb_in: Optional[torch.Tensor],
    pre_prev: Optional[torch.Tensor],
    hc_fn: torch.Tensor,
    residual_out: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    *,
    hc_mult: int,
    prefill: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The boundary's first launch: writes ``residual_out`` / ``y`` when given
    and returns the ``[slices, M, MIX]`` mixing partials and ``[slices, M]`` row
    sums of squares. ``prefill`` forces a regime (tests); None selects by M."""
    m, _, h = residual.shape
    mix = hc_fn.shape[0]
    dev = residual.device
    has_post = x is not None
    has_combine = pre_prev is not None
    num_slices = h // _HC_BOUNDARY_BLOCK_K
    part_mix = torch.empty((num_slices, m, mix), dtype=torch.float32, device=dev)
    part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=dev)
    if m == 0:
        return part_mix, part_sq
    if prefill is None:
        prefill = _hc_boundary_use_prefill(m, has_post, has_combine, x)
    placeholder = part_sq
    if prefill:
        mod = _hc_boundary_prefill_module()
        ctas = _hc_boundary_prefill_ctas(triton.cdiv(m, _HC_BOUNDARY_BLOCK_M))
        fn = mod.post_combine if has_post else mod.stats_only
        fn(
            x if has_post else placeholder,
            residual,
            post_in if has_post else placeholder,
            comb_in if has_post else placeholder,
            pre_prev if has_combine else placeholder,
            hc_fn,
            residual_out if has_post else placeholder,
            y if has_combine else placeholder,
            part_mix,
            part_sq,
            ctas,
        )
        return part_mix, part_sq
    mix_pad = max(16, triton.next_power_of_2(mix))
    block_m = _HC_BOUNDARY_BLOCK_M
    grid_m = triton.cdiv(m, block_m)
    _hc_boundary_partial_kernel[(grid_m, num_slices)](
        x if has_post else residual,
        residual,
        post_in if has_post else placeholder,
        comb_in if has_post else placeholder,
        pre_prev if has_combine else placeholder,
        hc_fn,
        residual_out if has_post else placeholder,
        y if has_combine else placeholder,
        part_mix,
        part_sq,
        m,
        h,
        x.stride(0) if has_post else residual.stride(0),
        residual.stride(0),
        hc_fn.stride(0),
        HC=hc_mult,
        MIX=mix,
        MIX_PAD=mix_pad,
        BLOCK_M=block_m,
        BLOCK_K=_HC_BOUNDARY_BLOCK_K,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        HAS_POST=has_post,
        HAS_COMBINE=has_combine,
        num_warps=_HC_BOUNDARY_NUM_WARPS,
        num_stages=_HC_BOUNDARY_NUM_STAGES,
    )
    return part_mix, part_sq


def hc_boundary_fused_deferred(
    x: Optional[torch.Tensor],
    residual: torch.Tensor,
    post_in: Optional[torch.Tensor],
    comb_in: Optional[torch.Tensor],
    pre_prev: Optional[torch.Tensor],
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], HcCoefficients]:
    """``hc_boundary_fused`` with the reduce + sinkhorn left pending: returns
    ``(residual_out, y, coefficients)``; see ``HcCoefficients`` for how the last launch is
    hosted by the norm that follows the boundary."""
    assert _is_hip, "hc_boundary_fused_deferred launches HIP-only kernels"
    assert hc_mult == 4 and residual.dim() == 3 and residual.shape[1] == hc_mult
    assert residual.stride(2) == 1 and residual.stride(1) == residual.shape[2]
    assert hc_fn.dtype == torch.float32 and hc_fn.stride(1) == 1
    m, _, h = residual.shape
    k = hc_mult * h
    mix = hc_fn.shape[0]
    assert mix == (2 + hc_mult) * hc_mult and hc_fn.shape[1] == k
    assert h % _HC_BOUNDARY_BLOCK_K == 0, h
    dev = residual.device
    has_post = x is not None
    has_combine = pre_prev is not None
    if has_post:
        assert post_in is not None and comb_in is not None
        assert x.shape == (m, h) and x.stride(1) == 1
        post_in = post_in.contiguous().float()
        comb_in = comb_in.contiguous().float()
        residual_out = torch.empty_like(residual)
    else:
        residual_out = None
    if has_combine:
        pre_prev = pre_prev.contiguous()
        y = torch.empty((m, h), dtype=residual.dtype, device=dev)
    else:
        y = None
    part_mix, part_sq = _hc_boundary_partials(
        x, residual, post_in, comb_in, pre_prev, hc_fn, residual_out, y, hc_mult=hc_mult
    )
    coefficients = HcCoefficients(
        part_mix,
        part_sq,
        hc_scale,
        hc_base,
        k=k,
        rms_eps=rms_eps,
        mix=mix,
        hc_mult=hc_mult,
        num_slices=h // _HC_BOUNDARY_BLOCK_K,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
    )
    return residual_out, y, coefficients


def hc_boundary_fused(
    x: Optional[torch.Tensor],
    residual: torch.Tensor,
    post_in: Optional[torch.Tensor],
    comb_in: Optional[torch.Tensor],
    pre_prev: Optional[torch.Tensor],
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """HIP mHC sublayer boundary in two launches: ``residual_out = hc_post(x, residual, post_in,
    comb_in)`` when ``x`` is given, ``y = sum_k pre_prev[k] * copy_k`` when ``pre_prev`` is, and the
    mixing coefficients of the (new) residual. Returns ``(residual_out, y, pre, post, comb)``."""
    residual_out, y, coefficients = hc_boundary_fused_deferred(
        x,
        residual,
        post_in,
        comb_in,
        pre_prev,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        rms_eps,
        hc_eps,
    )
    return (residual_out, y, *coefficients.tensors())
