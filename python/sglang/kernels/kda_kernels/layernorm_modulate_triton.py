# SPDX-License-Identifier: Apache-2.0
# KDA extension provenance: FLUX.2 FP8 producer fusion from SGLang PR #37162.
"""Fused LayerNorm + adaLN modulate Triton kernels for bf16 activations.

Two fusions, each replacing an eager multi-kernel chain with a single
launch while reproducing the eager results bit for bit (``torch.equal``),
so callers need no quality gate:

- ``fused_layernorm_modulate``: ``LN(x) * (1 + scale) + shift``
- ``fused_qk_head_layernorm``: per-head ``LN(q)`` / ``LN(k)``

Numerics contract (replicates torch 2.11's
``at::native::vectorized_layer_norm_kernel<c10::BFloat16, float, false>``,
which ``nn.LayerNorm`` dispatches to for bf16 rows with ``N % 4 == 0`` and
16-byte-aligned buffers; SASS-level derivation in PR #34008):

- 128 aten threads per row; thread ``t`` serially Welford-pushes the
  4-element vectors ``t, t+128, ... < N/4`` (for ``N % 512 != 0`` the
  trailing threads stop one iteration early and enter the fold with a
  smaller count), each scalar as
  ``mean' = fma(delta, rcp(count+1), mean)``,
  ``m2' = fma(delta, val - mean', m2)``, with ``rcp`` being nvcc's
  guarded-reciprocal fast path (``_rcp4``).
- Lane states fold via shfl.down offsets 16, 8, 4, 2, 1 through
  ``cuWelfordCombine`` (argument order: self = lower lane, other = upper;
  non-positive counts fold to zeros); the 4 warp results then combine
  pairwise (0,2), (1,3), (0,1).
- ``rstd = rsqrtf(div.rn.f32(m2, N) + eps)``: correctly-rounded fp32
  division, then ``MUFU.RSQ`` with a 2^24/2^12 rescale for subnormals.
- The normalized output is ``cvt.rn.bf16(rstd * (x - mean))`` (mul then
  add, no fma); the modulate chain then rounds to bf16 after every op:
  ``round(1 + scale)``, ``round(y * that)``, ``round(prod + shift)``.

For the qk kernel (per-head rows, ``dim_head % 4 == 0``, ``dim_head <=
128``) only the first ``dim_head / 4`` of the 128 aten threads carry data;
the remaining lanes and warps enter the fold with ``count == 0`` and are
reproduced faithfully.

Bit-exactness holds only for the dispatch above, so callers must verify
``torch.equal`` against the live eager chain once at runtime and fall back
on mismatch (see ``glm_image.py``).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.jit.utils import get_jit_cuda_arch
from sglang.kernels.numerics import (
    cuda_rsqrtf,
    div_rn_f32,
    round_bf16_to_fp32,
)
from sglang.kernels.ops.diffusion.common.fast_launch import replay_launches
from sglang.kernels.ops.diffusion.common.platform import is_cuda
from sglang.kernels.ops.layernorm.welford import (
    _fold_tree_32,
    _push_vec4,
    _welford_combine,
    layernorm_welford,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max, fp8_min
from sglang.kernels.ops.quantization.fp8_utils import fp8_dtype_to_triton
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit(do_not_specialize=["seq_len", "n_rows"])
def _layernorm_modulate_kernel(
    y_ptr,
    y_q_ptr,
    x_ptr,
    scale_ptr,
    shift_ptr,
    input_scale_ptr,
    seq_len,
    n_rows,
    scale_row_stride,
    eps,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    STORE_BF16: tl.constexpr,
    QUANTIZE_FP8: tl.constexpr,
    HAS_SHIFT: tl.constexpr = True,
    POST_OP: tl.constexpr = "bf16_modulate",
):
    pid = tl.program_id(0).to(tl.int64)
    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < n_rows
    row_base = row_offs * D

    mean, m2 = layernorm_welford(x_ptr, row_base, row_mask, D, ROWS)

    denom = tl.zeros((ROWS, 1), dtype=tl.float32) + D
    rstd = cuda_rsqrtf(div_rn_f32(m2, denom) + eps)  # (ROWS, 1)

    batch = row_offs // seq_len

    # pass 2: normalize + modulate, in aten's rounding order.
    for i in tl.static_range((D + 511) // 512):
        cols = i * 512 + tl.arange(0, 512)
        mask = row_mask[:, None]
        if (i + 1) * 512 > D:
            mask = mask & (cols < D)[None, :]
        x = tl.load(
            x_ptr + row_base[:, None] + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        y = rstd * (x - mean)
        if POST_OP == "bf16_modulate":
            y = round_bf16_to_fp32(y)
        sc = tl.load(
            scale_ptr + batch[:, None] * scale_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if POST_OP == "bf16_modulate":
            one_plus = round_bf16_to_fp32(1.0 + sc)
            y = round_bf16_to_fp32(y * one_plus)
            if HAS_SHIFT:
                sh = tl.load(
                    shift_ptr + batch[:, None] * scale_row_stride + cols[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                y = y + sh
        else:
            sh = tl.load(
                shift_ptr + batch[:, None] * scale_row_stride + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if POST_OP == "fp32_affine":
                # Native FP32 LayerNorm contracts gamma * normalized + beta.
                y = tl.fma(y, sc, sh)
            else:
                # Native FP32 LN, separate FP32 multiply/add, one BF16 store.
                # The caller disables implicit FMA contraction.
                y = y * (1.0 + sc) + sh
        if STORE_BF16:
            tl.store(y_ptr + row_base[:, None] + cols[None, :], y, mask=mask)
        if QUANTIZE_FP8:
            # The standalone static quantizer reads the just-written BF16
            # modulation output, so reproduce that final store/load rounding
            # before applying its exact per-tensor scale expression.
            y = round_bf16_to_fp32(y)
            input_scale = tl.load(input_scale_ptr).to(tl.float32)
            input_scale_inv = 1.0 / input_scale
            y_q = tl.clamp(y * input_scale_inv, FP8_MIN, FP8_MAX).to(FP8_DTYPE)
            tl.store(
                y_q_ptr + row_base[:, None] + cols[None, :],
                y_q.to(tl.uint8, bitcast=True),
                mask=mask,
            )


@triton.jit
def _qk_ln_head_one(
    dst,
    src,
    pid,
    n_rows,
    eps,
    D: tl.constexpr,
    D_POW2: tl.constexpr,
    ROWS: tl.constexpr,
):
    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < n_rows
    row_base = row_offs * D

    lanes = tl.arange(0, 32)
    lane_valid = (lanes < D // 4)[None, :]
    mean = tl.zeros((ROWS, 32), dtype=tl.float32)
    m2 = tl.zeros((ROWS, 32), dtype=tl.float32)
    cnt = tl.zeros((ROWS, 32), dtype=tl.float32)

    cols = lanes[:, None] * 4 + tl.arange(0, 4)[None, :]
    x4 = tl.load(
        src + row_base[:, None, None] + cols[None, :, :],
        mask=row_mask[:, None, None] & lane_valid[:, :, None],
        other=0.0,
    ).to(tl.float32)
    mean, m2, cnt = _push_vec4(x4, mean, m2, cnt, lane_valid, ROWS, 32, MASKED=True)

    mean, m2, cnt = _fold_tree_32(mean, m2, cnt, ROWS)
    # inter-warp combines with the all-zero warps 1..3 of the aten block:
    # (0,2) with zero, then (0,1) where warp 1 combined two zero warps.
    zero = tl.zeros((ROWS, 1), dtype=tl.float32)
    mean, m2, cnt = _welford_combine(mean, m2, cnt, zero, zero, zero)
    mean, m2, cnt = _welford_combine(mean, m2, cnt, zero, zero, zero)

    denom = tl.zeros((ROWS, 1), dtype=tl.float32) + D
    rstd = cuda_rsqrtf(div_rn_f32(m2, denom) + eps)

    cols2 = tl.arange(0, D_POW2)
    out_mask = row_mask[:, None] & (cols2 < D)[None, :]
    x = tl.load(src + row_base[:, None] + cols2[None, :], mask=out_mask, other=0.0).to(
        tl.float32
    )
    y = rstd * (x - mean)
    tl.store(dst + row_base[:, None] + cols2[None, :], y, mask=out_mask)


@triton.jit(do_not_specialize=["n_rows"])
def _qk_ln_head_kernel(
    q_out_ptr,
    k_out_ptr,
    q_ptr,
    k_ptr,
    n_rows,
    eps,
    D: tl.constexpr,
    D_POW2: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 0:
        _qk_ln_head_one(q_out_ptr, q_ptr, pid, n_rows, eps, D, D_POW2, ROWS)
    else:
        _qk_ln_head_one(k_out_ptr, k_ptr, pid, n_rows, eps, D, D_POW2, ROWS)


def is_plain_layer_norm(norm: torch.nn.Module, hidden: int) -> bool:
    """True for a bare ``nn.LayerNorm((hidden,))`` without affine params."""
    return (
        type(norm) is torch.nn.LayerNorm
        and norm.weight is None
        and norm.bias is None
        and tuple(norm.normalized_shape) == (hidden,)
    )


def _is_bf16_cuda(t: torch.Tensor) -> bool:
    # `is_cuda` also covers ROCm, where the inline PTX below cannot compile:
    # LLVM makes the unusable `=f` constraint a fatal error that kills the
    # process, so this has to reject before the first launch rather than let
    # the caller's try/except fall back.
    return is_cuda() and t.is_cuda and t.dtype is torch.bfloat16


def _qk_head_launch_config() -> tuple[int, int]:
    arch = get_jit_cuda_arch()
    if arch.major == 10 and arch.minor == 3:
        return 32, 1
    if arch.major * 10 + arch.minor >= 120:
        return 8, 4
    return 64, 2


def _mod_row_stride(t: torch.Tensor, batch: int, hidden: int) -> int | None:
    # (batch, hidden) modulation rows, possibly strided views of a chunked
    # adaLN projection; the last dim must be packed.
    if t.shape != (batch, hidden) or t.stride(1) != 1:
        return None
    return t.stride(0) if batch > 1 else hidden


def can_use_fused_layernorm_modulate(dtype: torch.dtype, hidden: int) -> bool:
    """Select the BF16 vectorized LayerNorm reduction replicated below."""
    return (
        is_cuda() and dtype is torch.bfloat16 and 0 < hidden <= 8192 and hidden % 4 == 0
    )


def _validate_layernorm_modulate(x, scale, shift) -> tuple[int, int, int, int]:
    shape = x.shape
    if not (
        _is_bf16_cuda(x)
        and len(shape) == 3
        and x.numel() > 0
        and x.is_contiguous()
        and 0 < shape[-1] <= 8192
        and shape[-1] % 4 == 0
    ):
        raise RuntimeError(
            "LayerNorm modulation expects contiguous BF16 CUDA [B, S, D], D divisible by 4 and <= 8192"
        )
    batch, seq_len, hidden = shape
    device = x.device
    stride = _mod_row_stride(scale, batch, hidden)
    if not (
        stride is not None and scale.dtype is torch.bfloat16 and scale.device == device
    ):
        raise RuntimeError("scale must have packed BF16 [B, D] rows on x's device")
    if shift is not None and not (
        shift.dtype is torch.bfloat16
        and shift.device == device
        and _mod_row_stride(shift, batch, hidden) == stride
    ):
        raise RuntimeError("shift must match scale's row layout and dtype/device")
    return batch, seq_len, hidden, stride


def _fake_ln_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor | None, eps: float
) -> torch.Tensor:
    return torch.empty_like(x)


def try_fused_fp32_layernorm_bf16(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    *,
    affine: bool = False,
) -> torch.Tensor | None:
    """BF16 input/output with native FP32 LN and FP32 affine or adaLN math.

    Replicate ``layer_norm(x.float()).bfloat16()`` with FP32 affine rows,
    or affine-free FP32 LN followed by ``* (1 + scale) + shift``. Callers
    must verify the live Torch reduction dispatch before enabling this path.
    """
    if (
        not _is_bf16_cuda(x)
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or x.ndim != 3
        or x.numel() == 0
        or not x.is_contiguous()
        or x.shape[-1] % 4 != 0
        or x.shape[-1] > 8192
        or scale.dtype != torch.float32
        or shift.dtype != torch.float32
        or scale.device != x.device
        or shift.device != x.device
    ):
        return None
    batch, tokens, channels = x.shape
    stride = _mod_row_stride(scale, batch, channels)
    if stride is None or _mod_row_stride(shift, batch, channels) != stride:
        return None
    output = torch.empty_like(x)
    rows = 2
    _layernorm_modulate_kernel[(triton.cdiv(batch * tokens, rows),)](
        output,
        output,
        x,
        scale,
        shift,
        scale,
        tokens,
        batch * tokens,
        stride,
        eps,
        D=channels,
        ROWS=rows,
        FP8_DTYPE=fp8_dtype_to_triton(fp8_dtype),
        FP8_MIN=fp8_min,
        FP8_MAX=fp8_max,
        STORE_BF16=True,
        QUANTIZE_FP8=False,
        POST_OP="fp32_affine" if affine else "fp32_modulate",
        num_warps=4 if channels >= 2048 else 2,
        enable_fp_fusion=False,
    )
    return output


@replay_launches
def fused_layernorm_modulate_raw(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """``LN(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)``, bit-exact
    vs the eager aten chain (LayerNorm without affine).

    With ``shift=None``, omit the addition, preserving signed zeros in
    scale-only modulation.

    Direct-call variant without the ``torch.ops`` dispatch (which costs tens
    of microseconds per call); use it on CPU-launch-bound eager hot paths
    (e.g. Sana), and the registered custom op under ``torch.compile``.
    """
    batch, seq_len, hidden, stride = _validate_layernorm_modulate(x, scale, shift)
    n_rows = batch * seq_len
    rows = 2
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _layernorm_modulate_kernel[(triton.cdiv(n_rows, rows),)](
            out,
            out,
            x,
            scale,
            shift,
            scale,
            seq_len,
            n_rows,
            stride,
            eps,
            D=hidden,
            ROWS=rows,
            FP8_DTYPE=fp8_dtype_to_triton(fp8_dtype),
            FP8_MIN=fp8_min,
            FP8_MAX=fp8_max,
            STORE_BF16=True,
            QUANTIZE_FP8=False,
            HAS_SHIFT=shift is not None,
            # H200-tuned: 38.5us at (1, 4096, 4096) vs the 121.8us eager
            # chain, 14.3us at Sana's (2, 1024, 2240) vs 43.1us.  ROWS=1 +
            # 4 warps triggers pathological Triton layout conversions in
            # the fold stage (47-58us).
            num_warps=4 if hidden >= 2048 else 2,
        )
    return out


def fused_layernorm_modulate_fp8_quant_raw(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse FLUX.2 LayerNorm, adaLN modulation, and static FP8 quantization.

    The returned tensor is byte-identical to running
    :func:`fused_layernorm_modulate_raw` followed by ``static_quant_fp8``.
    Unlike that two-op chain, this path does not materialize the intermediate
    BF16 activation because FLUX.2 feeds it directly into an FP8 projection.
    """
    batch, seq_len, hidden, stride = _validate_layernorm_modulate(x, scale, shift)
    n_rows = batch * seq_len
    rows = 2
    out = torch.empty_like(x, dtype=fp8_dtype)
    with torch.cuda.device(x.device):
        _layernorm_modulate_kernel[(triton.cdiv(n_rows, rows),)](
            out,
            out.view(torch.uint8),
            x,
            scale,
            shift,
            input_scale,
            seq_len,
            n_rows,
            stride,
            eps,
            D=hidden,
            ROWS=rows,
            FP8_DTYPE=fp8_dtype_to_triton(fp8_dtype),
            FP8_MIN=fp8_min,
            FP8_MAX=fp8_max,
            STORE_BF16=False,
            QUANTIZE_FP8=True,
            num_warps=4 if hidden >= 2048 else 2,
        )
    return out


fused_layernorm_modulate = register_custom_op(
    fused_layernorm_modulate_raw,
    op_name="triton_fused_layernorm_modulate",
    mutates_args=[],
    fake_impl=_fake_ln_modulate,
)


def can_use_fused_qk_head_layernorm(dtype: torch.dtype, head_dim: int) -> bool:
    """Select the BF16 per-head LayerNorm reduction."""
    return (
        is_cuda()
        and dtype is torch.bfloat16
        and 0 < head_dim <= 128
        and head_dim % 4 == 0
    )


def _fake_qk_ln(
    q: torch.Tensor, k: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


@register_custom_op(
    op_name="triton_fused_qk_head_layernorm",
    mutates_args=[],
    fake_impl=_fake_qk_ln,
)
def fused_qk_head_layernorm(
    q: torch.Tensor, k: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head ``nn.LayerNorm(dim_head)`` (no affine) over q and k in one
    launch, bit-exact vs the eager aten kernel."""
    shape = q.shape
    if not (
        _is_bf16_cuda(q)
        and len(shape) == 4
        and q.numel() > 0
        and 0 < shape[-1] <= 128
        and shape[-1] % 4 == 0
        and k.dtype is torch.bfloat16
        and k.device == q.device
        and k.shape == shape
        and q.is_contiguous()
        and k.is_contiguous()
    ):
        raise RuntimeError(
            "QK LayerNorm expects matching contiguous BF16 CUDA [B, S, H, D], D divisible by 4 and <= 128"
        )
    head_dim = shape[-1]
    n_rows = q.numel() // head_dim
    # Architecture sweeps at the production GLM shape select 32 rows / 1 warp
    # on B300 (SM103) and 8 rows / 4 warps on RTX 5090 (SM120). Preserve the
    # independently tuned H100/H200 launch on all other architectures.
    rows, num_warps = _qk_head_launch_config()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    with torch.cuda.device(q.device):
        _qk_ln_head_kernel[(triton.cdiv(n_rows, rows), 2)](
            q_out,
            k_out,
            q,
            k,
            n_rows,
            eps,
            D=head_dim,
            D_POW2=triton.next_power_of_2(head_dim),
            ROWS=rows,
            # H200-tuned: 62us at (1, 4360, 32, 128) vs the 301us of the two
            # aten launches (one 128-thread block per head_dim-element row).
            num_warps=num_warps,
        )
    return q_out, k_out
