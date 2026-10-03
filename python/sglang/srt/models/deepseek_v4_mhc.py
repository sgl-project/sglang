"""DeepSeek-V4.1 manifold-constrained hyper-connections (mHC).

The residual is ``R [T, hc_mult, hidden]`` -- hc_mult parallel streams per token --
and each sublayer is wrapped in a hyper-connection (DeepSeek's ``attn_hc`` and
``ffn_hc``) carrying a ``(pre, post, comb)`` triplet. Per hyper-connection:

    1. mix stats  predict ``(pre, post, comb)`` from ``R``
    2. combine    collapse the streams with the PREVIOUS triplet's ``pre``, then norm
    3. sublayer   attention or MoE, including its tensor-parallel reduction
    4. post       ``R'[i] = post[i] * y + sum_j comb[j][i] * R[j]``

Step 1's ``pre`` is consumed by step 2 of the *next* hyper-connection -- the lag
that lets step 1 run beside steps 2-3 and makes V4's fused post+pre inexpressible.
``attn_hc -> ffn_hc`` stays inside a layer; ``ffn_hc -> attn_hc`` crosses layers,
where an Engram or a row selection can replace ``R`` (`with_residual`/`take_rows`).

The layer owns parameters, parallelism and token-range policy; this module owns
the kernel dispatch.

TODO: move dsv4 MHC into the same abstraction
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn.functional as F

from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import compile_in_capture_mode
from sglang.srt.models.deepseek_v2 import MoEOutput, _is_hip, _is_npu, _is_xpu
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.srt.utils import is_gfx95_supported

_is_gfx95_supported = is_gfx95_supported()

# ---------------------------------------------------------------------------
# What a layer hands this module
# ---------------------------------------------------------------------------

HcTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class HcConfig(NamedTuple):
    """Everything about a layer's mHC that is fixed for the model's lifetime."""

    mult: int
    sinkhorn_iters: int
    eps: float
    rms_eps: float
    hidden: int
    pre_from_prev: bool
    cp_prefill: bool


class HcSubLayer(NamedTuple):
    """One hyper-connection's fixed material. Holds the layer's Parameters by
    reference; a plain tuple registers nothing, so checkpoint keys stay put."""

    cfg: HcConfig
    fn: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    norm: RMSNorm
    tf32_parts: Optional[Any] = None
    bf16_parts: Optional[Any] = None


# ---------------------------------------------------------------------------
# How far the previous hyper-connection got toward the next sublayer's input
# ---------------------------------------------------------------------------


class HcNormed(NamedTuple):
    """Summed and normed; the sublayer can take these rows as they are."""

    rows: torch.Tensor


class HcQuantized(NamedTuple):
    """Normed, and already swizzled for the sublayer's first projection."""

    rows: torch.Tensor
    swizzled: Mxfp8SwizzledInput


HcPreOutput: TypeAlias = Union[HcNormed, HcQuantized]
"""The next sublayer's input, built by the previous post; a receiver skips step 2
(and ``pre``) entirely. Always normed -- combine and norm travel together."""


class AttnOutput(NamedTuple):
    """Attention's wo_b rows with the TP reduction left to the fused post."""

    partial: torch.Tensor


# ---------------------------------------------------------------------------
# The real parameters: what crosses from one hyper-connection to the next
# ---------------------------------------------------------------------------


class HcPending(NamedTuple):
    """A post not run yet; the residual it would produce is implied. The backward
    counterpart of `HcPreOutput`: the HIP boundary kernel takes both and runs the
    previous post, this triplet and this collapse in one launch."""

    rows: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


class HcState(NamedTuple):
    """The mHC state of the stream between two hyper-connections.

    ``streams`` is the residual ``R``, or -- out of the HIP fused boundary -- the
    post that would rebuild it, still outstanding.
    """

    streams: Union[torch.Tensor, HcPending]
    pre: Optional[torch.Tensor] = None
    input: Optional[HcPreOutput] = None

    @property
    def residual(self) -> torch.Tensor:
        """The materialized streams; run an outstanding post first (`materialized`)."""
        assert isinstance(self.streams, torch.Tensor)
        return self.streams

    def materialized(self, cfg: HcConfig) -> HcState:
        """Run the outstanding post, if any; residual readers call this first."""
        if not isinstance(self.streams, HcPending):
            return self
        p = self.streams
        return HcState(post(cfg, p.rows, p.residual, p.post, p.comb), self.pre)

    def with_residual(self, residual: torch.Tensor) -> HcState:
        """The residual was rewritten: drop ``input``; ``pre`` weights streams, not
        rows, so it survives."""
        return HcState(residual, self.pre)

    def take_rows(self, rows: Callable[[torch.Tensor], torch.Tensor]) -> HcState:
        """Keep only the rows ``rows`` selects -- the late-layer tail's narrowing."""
        pre = None if self.pre is None else rows(self.pre)
        if isinstance(self.streams, HcPending):
            return HcState(HcPending(*(rows(t) for t in self.streams)), pre)
        return HcState(rows(self.streams), pre)


# ---------------------------------------------------------------------------
# The plan: how a hyper-connection intends to run its post
# ---------------------------------------------------------------------------


class HcNextBoundary(NamedTuple):
    """What the next hyper-connection can take off this one's post; resolved after
    weights load. None where the seam is closed (last layer, Engram)."""

    norm: RMSNorm
    accepts_mxfp8: bool
    norm_fusable: bool


def is_deferred_finalize(routed) -> bool:
    """Whether a `MoEOutput` really carries an unfinalized handle for the post."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

    return isinstance(routed, FlashInferTrtllmDeferredFinalizeOutput)


def make_boundary(norm: RMSNorm, *, accepts_mxfp8: bool) -> HcNextBoundary:
    """Can this hyper-connection's norm fold into the previous post, and does its
    sublayer take a pre-quantized input."""
    return HcNextBoundary(
        norm=norm,
        accepts_mxfp8=accepts_mxfp8,
        norm_fusable=(
            not norm.cast_x_before_out_mul
            and norm.variance_size_override is None
            and norm.weight.dtype == torch.bfloat16
            # TODO: support other norm shape
            and norm.weight.shape == (5120,)
            and norm.weight.is_contiguous()
        ),
    )


# ---------------------------------------------------------------------------
# Step 1 -- the mix GEMM and the triplet it splits into
# ---------------------------------------------------------------------------


def use_stats_stream(cfg: HcConfig, forward_batch: ForwardBatch, x: torch.Tensor):
    if not cfg.pre_from_prev:
        return False
    decode_like = forward_batch.forward_mode.is_decode() or (
        forward_batch.forward_mode.is_target_verify() and x.shape[0] > 0
    )
    return decode_like and (
        not get_platform().is_sm90
        or x.shape[0] == 1
        or (forward_batch.forward_mode.is_decode() and 1 < x.shape[0] <= 64)
    )


def mix_stats(
    hc: HcSubLayer, x: torch.Tensor, stats_stream: Optional[torch.cuda.Stream] = None
) -> HcTriplet:
    """Predict the triplet on the caller's stream or the stats stream."""
    if stats_stream is None:
        return _mix_stats_impl(hc, x)
    main_stream = torch.cuda.current_stream()
    x.record_stream(stats_stream)
    with torch.cuda.stream(stats_stream):
        coefficients = _mix_stats_impl(hc, x)
    for coefficient in coefficients:
        coefficient.record_stream(main_stream)
    return coefficients


def _mix_stats_impl(hc: HcSubLayer, x: torch.Tensor) -> HcTriplet:
    from sglang.kernels.ops.layernorm.mhc import (
        hc_mix_stats,
        hc_mix_stats_sinkhorn,
        hc_split_sinkhorn,
    )

    cfg = hc.cfg

    x_flat = x.flatten(1)

    from sglang.srt.batch_invariant_ops import (
        is_batch_invariant_mode_enabled,
    )

    parts = bf16_parts = None
    hopper_medium = get_platform().is_sm90 and 32 <= x_flat.shape[0] < 4096
    if (
        x.is_cuda
        and (x_flat.shape[0] >= 128 or hopper_medium)
        and x_flat.is_contiguous()
        and (get_platform().is_sm100 or get_platform().is_sm90)
        and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        and not is_batch_invariant_mode_enabled()
    ):
        parts, bf16_parts = hc.tf32_parts, hc.bf16_parts

    use_bf16_projection = bf16_parts is not None and (
        hopper_medium or 4096 <= x_flat.shape[0] <= 65536
    )
    hopper_fused_stats = get_platform().is_sm90 and (
        x.shape[0] == 1 or (bf16_parts is not None and 32 <= x.shape[0] <= 65536)
    )
    # gfx950 uses the fused Triton port at every row count (MI350X).
    use_fused_stats = _is_gfx95_supported or (
        torch.version.cuda is not None
        and (get_platform().is_blackwell or hopper_fused_stats)
    )
    if x.is_cuda and use_fused_stats and x.dtype == torch.bfloat16:
        # The default split-K/Sinkhorn fusion preserves batch invariance;
        # compensated projections above are disabled in batch-invariant mode.
        if use_bf16_projection:
            from sglang.kernels.ops.layernorm.mhc import (
                hc_mix_stats_sinkhorn_bf16x3,
            )

            pre, post, comb = hc_mix_stats_sinkhorn_bf16x3(
                x_flat,
                bf16_parts,
                hc.scale,
                hc.base,
                cfg.sinkhorn_iters,
                cfg.rms_eps,
                cfg.eps,
            )
        elif parts is not None:
            from sglang.kernels.ops.layernorm.mhc import (
                hc_mix_stats_sinkhorn_deepgemm,
            )

            pre, post, comb = hc_mix_stats_sinkhorn_deepgemm(
                x_flat,
                parts,
                hc.scale,
                hc.base,
                cfg.sinkhorn_iters,
                cfg.rms_eps,
                cfg.eps,
            )
        else:
            pre, post, comb = hc_mix_stats_sinkhorn(
                x_flat,
                hc.fn,
                hc.scale,
                hc.base,
                cfg.mult,
                cfg.sinkhorn_iters,
                cfg.rms_eps,
                cfg.eps,
            )
        return pre, post, comb
    if x.is_cuda and torch.version.cuda is not None:
        # cuBLAS/torch reductions can change order with num_tokens; this kernel
        # keeps the mixing and RMS reductions batch-invariant.
        mixes = hc_mix_stats(x_flat, hc.fn, cfg.rms_eps).unsqueeze(1)
    else:
        x_flat = x_flat.float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + cfg.rms_eps)
        mixes = (F.linear(x_flat, hc.fn) * rsqrt).unsqueeze(1)
    pre, post, comb = hc_split_sinkhorn(
        mixes,
        hc.scale,
        hc.base,
        cfg.mult,
        cfg.sinkhorn_iters,
        cfg.eps,
    )
    return pre.squeeze(1), post.squeeze(1), comb.squeeze(1)


# ---------------------------------------------------------------------------
# Step 2 -- collapse the streams and normalize
# ---------------------------------------------------------------------------


def combine(
    hc: HcSubLayer,
    state: HcState,
    stats_stream: Optional[torch.cuda.Stream] = None,
    quantized: Optional[list] = None,
) -> torch.Tensor:
    """The sublayer's input: ``norm(sum_k pre[k] * R[k])``, short-circuited by
    whatever the previous post left in ``state.input``."""
    from sglang.kernels.ops.layernorm.mhc import hc_combine

    cfg, norm = hc.cfg, hc.norm
    x, apply_pre, shortcut = state.residual, state.pre, state.input
    quantize = quantized is not None
    x_flat = x.flatten(1)
    tiny = 0 < x.shape[0] <= 8
    if stats_stream is not None and not tiny:
        stats_stream.wait_stream(torch.cuda.current_stream())

    def combine_and_norm():
        if isinstance(shortcut, HcQuantized):
            assert quantized is not None
            quantized.append(shortcut.swizzled)
            return shortcut.rows
        if isinstance(shortcut, HcNormed):
            # Prefill projections still quantize the BF16 input themselves; the
            # optional fused-quantization list stays empty for this case.
            return shortcut.rows
        if apply_pre is None:
            return norm(x[:, 0, :].contiguous())
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        hopper_fused = (
            get_platform().is_sm90
            and not quantize
            and (0 < x.shape[0] <= 96 or 4096 <= x.shape[0] <= 65536)
            and not is_batch_invariant_mode_enabled()
        )
        if (
            x.is_cuda
            and (get_platform().is_blackwell or hopper_fused)
            and x.dtype == torch.bfloat16
            and apply_pre.stride(1) == 1
            and cfg.mult == 4
            and cfg.hidden == 5120
            and norm.weight.dtype == torch.bfloat16
            and not norm.cast_x_before_out_mul
            and norm.variance_size_override is None
        ):
            # One fused form for every row count. Each row's reduction order is
            # fixed regardless of the grid split, so this holds under
            # batch-invariant mode too.
            if quantize and x.shape[0] <= 8:
                from sglang.kernels.ops.layernorm.hc_combine_norm import (
                    hc_combine_norm_mxfp8,
                )

                y, y_q, y_sf = hc_combine_norm_mxfp8(
                    x_flat, apply_pre, norm.weight, norm.variance_epsilon
                )
                quantized.append(Mxfp8SwizzledInput(y_q, y_sf))
                return y
            from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm

            return hc_combine_norm(
                x_flat, apply_pre, norm.weight, norm.variance_epsilon
            )
        return norm(hc_combine(x_flat, apply_pre, cfg.mult, x.dtype))

    y = combine_and_norm()
    if stats_stream is not None and tiny:
        stats_stream.wait_stream(torch.cuda.current_stream())
    return y


# ---------------------------------------------------------------------------
# Step 4 -- rebuild the streams, optionally collapsing the next input too
# ---------------------------------------------------------------------------


def post(
    cfg: HcConfig,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """``R'[i] = post[i] * x + sum_j comb[j][i] * R[j]``, by whichever kernel fits."""
    if x.shape[0] == 0:
        return torch.empty((0, cfg.mult, x.shape[-1]), dtype=x.dtype, device=x.device)

    if _is_npu:
        if not is_npu_arch35():
            return torch.ops.custom.npu_hc_post(x, residual, post_mix, comb)
        # The A5 build of npu_hc_post is batched -- it requires a leading batch axis
        # on every operand.
        return torch.ops.custom.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post_mix.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    if _is_xpu:
        from sglang.srt.models.deepseek_v4 import get_mhc_ops

        return get_mhc_ops().mhc_post(x, residual, post_mix, comb)

    if (
        get_platform().is_blackwell
        and cfg.pre_from_prev
        and cfg.mult == 4
        and cfg.hidden == 5120
        # This 384 is the kernel's own tile limit, unrelated to the collective's
        # push slots; it covers every narrow batch, not one particular regime.
        and x.shape[0] <= 384
    ):
        return mhc_post_split_h(x, residual, post_mix, comb)

    if _is_hip:
        from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip as _hip

        y = _hip.hc_post(cfg, x, residual, post_mix, comb)
        if y is not None:
            return y

    if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
        from flashinfer.mhc import mhc_post

        return mhc_post(x, residual, post_mix, comb)

    if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
        if (
            cfg.pre_from_prev
            and get_platform().is_sm90
            and cfg.mult == 4
            and cfg.hidden == 5120
            and 1 <= x.shape[0] <= 64
        ):
            return mhc_post_split_h(x, residual, post_mix, comb)

        from sglang.kernels.ops.layernorm.mhc import mhc_post

        return mhc_post(x, residual, post_mix, comb)

    if _is_hip:
        from aiter.ops.mhc import mhc_post

        result = torch.empty_like(residual)
        mhc_post(result, x, residual, post_mix, comb)
        return result

    assert residual.shape == (x.shape[0], cfg.mult, x.shape[-1])
    assert post_mix.shape == (x.shape[0], cfg.mult)
    assert comb.shape == (x.shape[0], cfg.mult, cfg.mult)

    @compile_in_capture_mode
    def post_torch_impl(x, residual, post_mix, comb):
        return (
            post_mix.unsqueeze(-1) * x.unsqueeze(1)
            + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
        ).type_as(x)

    return post_torch_impl(x, residual, post_mix, comb)


def post_with_combine(
    cfg: HcConfig,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    next: Optional[HcNextBoundary],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Step 4, folding the next combine + norm in where the wide-tile kernel serves
    this seam; ``normalized`` is None when the next combine must run itself."""
    if (
        next is not None
        and next.norm_fusable
        and cfg.pre_from_prev
        and get_platform().is_blackwell
        and 4096 <= x.shape[0] <= 65536
        and cfg.hidden == 5120
        and cfg.mult == 4
        and get_parallel().attn_dp_size == 1
        and not cfg.cp_prefill
    ):
        from sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill import (
            mhc_post_combine_norm_prefill,
        )

        norm = next.norm
        updated, normalized = mhc_post_combine_norm_prefill(
            x, residual, post_mix, comb, pre, norm.weight, norm.variance_epsilon
        )
        return updated, normalized
    return post(cfg, x, residual, post_mix, comb), None


# ---------------------------------------------------------------------------
# Planning the post
# ---------------------------------------------------------------------------


def can_fuse_post(cfg: HcConfig) -> bool:
    """Whether a fused post is expressible at all for this model, device and
    parallel layout.

    The capability half of the decision; which token ranges should take it, and what
    batch-invariant mode should do about it, are the caller's policy.
    """
    return (
        get_platform().is_blackwell
        and cfg.pre_from_prev
        and cfg.mult == 4
        and cfg.hidden == 5120
        and get_parallel().attn_dp_size == 1
        and not cfg.cp_prefill
    )


# ---------------------------------------------------------------------------
# Running the post
# ---------------------------------------------------------------------------


def _compute_triplet(
    hc: HcSubLayer,
    residual: torch.Tensor,
    stats_stream: Optional[torch.cuda.Stream],
) -> HcTriplet:
    """Issue the triplet (on the side stream, beside the sublayer) and join before
    the post reads it."""
    coefficients = mix_stats(hc, residual, stats_stream)
    if stats_stream is not None:
        torch.cuda.current_stream().wait_stream(stats_stream)
    return coefficients


def _plain_post(
    hc: HcSubLayer,
    y: torch.Tensor,
    residual: torch.Tensor,
    coefficients: HcTriplet,
    *,
    next: Optional[HcNextBoundary],
) -> HcState:
    """Step 4 without a collective; the next seam may still fold in."""
    pre, post_mix, comb = coefficients
    updated, normalized = post_with_combine(
        hc.cfg, y, residual, post_mix, comb, pre, next
    )
    return HcState(updated, pre, None if normalized is None else HcNormed(normalized))


def run_attn_post(
    hc: HcSubLayer,
    out: Union[torch.Tensor, AttnOutput],
    residual: torch.Tensor,
    *,
    stats_stream: Optional[torch.cuda.Stream],
    next: Optional[HcNextBoundary],
    world_size: int,
) -> HcState:
    """The attention post. An `AttnOutput` rides the collective kernel (which also
    folds ``next``'s norm); the attention may decline the handover even when asked,
    so the type is the ground truth."""
    if isinstance(out, AttnOutput):
        from sglang.kernels.ops.communication.all_reduce_mhc import all_reduce_mhc_norm

        assert next is not None
        pre, post_mix, comb = _compute_triplet(hc, residual, stats_stream)
        _, updated, normalized = all_reduce_mhc_norm(
            out.partial,
            residual,
            post_mix,
            comb,
            pre,
            next.norm.weight,
            next.norm.variance_epsilon,
            world_size=world_size,
        )
        return HcState(updated, pre, HcNormed(normalized))
    coefficients = _compute_triplet(hc, residual, stats_stream)
    return _plain_post(hc, out, residual, coefficients, next=next)


def run_moe_post(
    hc: HcSubLayer,
    out: Union[torch.Tensor, MoEOutput],
    residual: torch.Tensor,
    *,
    stats_stream: Optional[torch.cuda.Stream],
    next: Optional[HcNextBoundary],
    world_size: int,
) -> HcState:
    """The MoE post. A deferred finalize the push plane can carry rides the
    collective kernel (finalize + shared add + all-reduce, quantizing ``next``'s
    input when a boundary is given); anything else is finalized here and takes
    the plain post."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        can_fuse_all_reduce,
    )

    if (
        isinstance(out, MoEOutput)
        and is_deferred_finalize(out.routed)
        and can_fuse_all_reduce()
    ):
        pre, post_mix, comb = _compute_triplet(hc, residual, stats_stream)
        args = (
            out.routed.gemm2_out,
            out.routed.expanded_idx_to_permuted_idx,
            out.routed.expert_weights,
            out.routed.top_k,
            out.shared,
            residual,
            post_mix,
            comb,
        )
        if next is not None and next.accepts_mxfp8 and next.norm_fusable:
            from sglang.kernels.ops.communication.all_reduce_mhc import (
                moe_finalize_all_reduce_mhc_quant,
            )

            _, updated, normalized, q, sf = moe_finalize_all_reduce_mhc_quant(
                *args,
                pre,
                next.norm.weight,
                next.norm.variance_epsilon,
                world_size=world_size,
            )
            return HcState(
                updated, pre, HcQuantized(normalized, Mxfp8SwizzledInput(q, sf))
            )
        from sglang.kernels.ops.communication.all_reduce_mhc import (
            moe_finalize_all_reduce_mhc,
        )

        _, updated = moe_finalize_all_reduce_mhc(*args, world_size=world_size)
        return HcState(updated, pre)
    if isinstance(out, MoEOutput):
        out = out.get_all_reduce_merged()
    coefficients = _compute_triplet(hc, residual, stats_stream)
    return _plain_post(hc, out, residual, coefficients, next=next)
