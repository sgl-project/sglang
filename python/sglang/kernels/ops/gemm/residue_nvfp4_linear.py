"""Opaque custom op covering the residue NVFP4 dense-linear dispatch.

Why this exists:
    The residue NVFP4 linear method has distinct kernel chains:
      - mext fold path: scaled_fp4_quant_mext_r1 + CuTeDSL fold GEMM
      - two-GEMM prefill path: one plain NVFP4 GEMM at 2M rows + pair-sum
      - plain path: stock NVFP4 quant + GEMM (no residue)
      - K-ext hybrid: small M uses base-K fold; large M uses masked quant +
        GEMM over K_ext
    Which one a layer gets is NOT purely an M decision: a mext_r1 layer
    (ratio 1.0, weight stored once) has no K-extended weight to fall back to,
    so it takes a residue-preserving path at every M. Selecting between the
    chains at Python level inside apply() makes the routing
    tensor-shape-dependent; torch.compile then specializes the trace at its
    dummy input shape and bakes one branch into every captured CUDA graph.

    Wrapping the whole pipeline (quant + GEMM) in a single opaque custom op
    fixes that: torch.compile captures one node, bakes only per-layer
    constants into its args, and the internal dispatch runs at execution
    time, so layout choice is correct for the actual M of each capture.

Shape derivation:
    All intermediate shapes are computed from x.shape and integer args
    (per-layer constants). Never from a kernel output's shape -- those are
    SymInts whose materialization forces a host sync at trace time.

Backend scope:
    Cutlass-layout NVFP4 backends only (flashinfer cutlass / cute-dsl). The
    linear method enforces that before routing a layer here.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
    MEXT_R1_LAYOUT_ROW_PAIR,
    scaled_fp4_quant_mext_r1,
)
from sglang.srt.utils.custom_op import register_custom_op

# ── how a layer represents its residue ─────────────────────────────────────
# Derived, never passed: the op receives `num_salient` and `is_mext_r1`
# separately, so the information is unambiguous AT THE BOUNDARY. What was
# ambiguous historically was code that branched on `num_salient == 0`, a value
# that means both "plain layer, no residue" and "mext_r1 layer, every channel
# is salient". Branch on the name, not the number.
RESIDUE_NONE = "none"
RESIDUE_K_EXT = "k_ext"
RESIDUE_MEXT_R1 = "mext_r1"


def residue_kind_of(num_salient: int, is_mext_r1: bool) -> str:
    """Which residue representation this layer uses.

    `is_mext_r1` wins: a mext_r1 layer legitimately reports num_salient == 0
    because every channel is salient and the residue lives in the doubled
    token rows, not in a channel subset.
    """
    if is_mext_r1:
        return RESIDUE_MEXT_R1
    return RESIDUE_NONE if int(num_salient) == 0 else RESIDUE_K_EXT


# M threshold for choosing the fold path on a K-extended checkpoint. This is a
# PERFORMANCE crossover between two residue-preserving representations: below
# it, base-K MExt fold; above it, the checkpoint's K-ext quant + GEMM. A
# mext_r1 layer ignores the threshold as a correctness gate (it must apply its
# residue at every M), though the threshold still separates its small-M fold
# from the large-M K-loop/two-GEMM tuner.
#
# Defaults are arch-dependent: the crossover was measured at ~128 on the
# sm100/sm103 CuTeDSL fold and ~64 elsewhere. Read through fold_max_m_for(),
# never off the constants -- warmup and dispatch must agree or the server JITs
# inside cudagraph capture for the sizes warmup missed.
_FOLD_MAX_M_DEFAULT_SM10X = 128
_FOLD_MAX_M_DEFAULT_OTHER = 64
_FOLD_MAX_M_ENV = os.environ.get("SGLANG_RESIDUE_DECODE_MEXT_MAX_M")

# PD-disaggregated decode servers: every forward is small-M anyway, but a
# uniform captured kernel sequence (no M-threshold branch across the captured
# graph set) is worth forcing. At-call cost is zero -- only the dispatch
# expression changes.
_FORCE_FOLD = os.environ.get("SGLANG_RESIDUE_FORCE_DECODE_MEXT", "0") == "1"

# mext_r1 prefill routing above the fold band:
#   auto    FlashInfer-tuned two_gemm-vs-k_loop dispatch (two_gemm on miss)
#   k_loop  force the wrapped-K kernel where precompiled
#   two_gemm  bypass the tuner entirely
_MEXT_PREFILL_PATH = os.environ.get("SGLANG_RESIDUE_MEXT_PREFILL_PATH", "auto")


def fold_max_m_for(arch: str | None) -> int:
    """The M at or below which a fold-eligible non-mext_r1 layer folds.

    ``arch`` is ``'sm100'`` or ``'sm103'`` when a fold kernel is available.
    The environment override is a performance knob only.
    """
    if _FOLD_MAX_M_ENV is not None:
        return int(_FOLD_MAX_M_ENV)
    return _FOLD_MAX_M_DEFAULT_SM10X if arch is not None else _FOLD_MAX_M_DEFAULT_OTHER


# ── arch probes (once per process) ──────────────────────────────────────────
_SM10X_FOLD_ARCH: str | None = None
_SM10X_FOLD_PROBED = False
_SM10X_FOLD_FIRED = [False]


def _sm10x_fold_arch() -> str | None:
    """'sm100'/'sm103' if this GPU uses the sm10x CuTeDSL fold, else None."""
    global _SM10X_FOLD_ARCH, _SM10X_FOLD_PROBED
    if _SM10X_FOLD_PROBED:
        return _SM10X_FOLD_ARCH
    _SM10X_FOLD_PROBED = True
    if os.environ.get("SGLANG_RESIDUE_DISABLE_SM10X_FOLD", "0") == "1":
        _SM10X_FOLD_ARCH = None
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
        from sglang.kernels.ops.gemm.residue_fold.cute_fold.tactics import (
            kernel_arch_for_capability,
        )

        _SM10X_FOLD_ARCH = kernel_arch_for_capability(int(major), int(minor))
    except Exception:
        _SM10X_FOLD_ARCH = None
    return _SM10X_FOLD_ARCH


# ── helpers ─────────────────────────────────────────────────────────────────


def _base_k_weight_view(weight: torch.Tensor, k_base: int) -> torch.Tensor:
    """Base-K prefix of a (possibly ext-K) packed FP4 weight, as a VIEW.

    For a pure mext_r1 layer k_base already spans the whole tensor and this
    returns `weight` unchanged. For an extended_k layer the stored weight is
    [N, k_ext/2] and the fold only needs the first k_base/2 packed columns;
    the narrowed view keeps row stride k_ext/2, which the CuTeDSL folds
    consume directly. No copy, so the ext weight is never duplicated.
    """
    k_base_packed = int(k_base) // 2
    if k_base_packed == weight.shape[1]:
        return weight
    assert (
        k_base_packed < weight.shape[1]
    ), f"k_base {k_base} exceeds stored weight width {weight.shape[1] * 2}"
    return weight[:, :k_base_packed]


def _pad_activation_cols(x_fp4: torch.Tensor, padding_cols: int) -> torch.Tensor:
    """Pad packed FP4 activations to match the weight's K-dim padding."""
    if padding_cols > 0:
        return torch.nn.functional.pad(x_fp4, (0, padding_cols)).contiguous()
    return x_fp4


def _slice_linear_output(
    out: torch.Tensor, original_m: int, output_size: int
) -> torch.Tensor:
    """Remove fold/kernel padding without copying the common exact shape."""
    target_n = int(output_size) if int(output_size) > 0 else out.shape[-1]
    if out.shape[0] != int(original_m) or out.shape[-1] != target_n:
        return out[: int(original_m), :target_n].contiguous()
    return out


def _fp4_gemm(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Plain NVFP4 GEMM through SGLang's fp4 runner (flashinfer mm_fp4)."""
    from sglang.srt.layers.quantization.modelopt_quant import fp4_gemm

    return fp4_gemm(
        x_fp4, weight.T, x_sf, weight_scale.T, alpha, out_dtype, weight.shape[0]
    )


def _plain_fp4_quantize(
    x: torch.Tensor, input_global_scale_inv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    if fp4_quantize is None:
        raise RuntimeError("residue nvfp4 linear requires flashinfer's fp4_quantize")
    return fp4_quantize(x, input_global_scale_inv)


def _run_mext_r1_two_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """mext_r1 prefill: one backend GEMM at 2M rows + pair-sum epilogue.

    out[i] = W . base_i + W . res_i, exactly the fold's math. The mext_r1
    quant's row_pair SF is bytewise the cutlass 128x4 swizzle over the 2M
    rows, so the doubled activation feeds the stock GEMM directly and
    base/residue rows come back interleaved at even/odd indices.

    Every op here has a static compile space: no CuTeDSL, nothing to JIT at
    serving time, and cudagraph capture sees ordinary torch/GEMM calls.
    """
    x_fp4, x_sf = scaled_fp4_quant_mext_r1(
        x, input_global_scale_inv, layout_mode=MEXT_R1_LAYOUT_ROW_PAIR
    )
    two_m, k = x_fp4.shape[0], x.shape[-1]
    num_m_tiles = (two_m + 127) // 128
    num_k_tiles = (k + 63) // 64
    x_sf_v = x_sf.view(torch.float8_e4m3fn).view(num_m_tiles * 128, num_k_tiles * 4)
    y = _fp4_gemm(x_fp4, x_sf_v, weight, weight_scale_base, alpha, output_dtype)
    return y[0::2] + y[1::2]


def _run_mext_fold_gemm_sm10x(
    x_fp4_rowpair: torch.Tensor,
    weight: torch.Tensor,
    x_sf: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """CuTeDSL SM100/SM103 row-pair fold GEMM."""
    from sglang.kernels.ops.gemm.residue_fold import run_fold

    if not _SM10X_FOLD_FIRED[0]:
        _SM10X_FOLD_FIRED[0] = True
        print(
            f"[residue] SM10x MExt fold kernel FIRED: "
            f"act={tuple(x_fp4_rowpair.shape)} weight={tuple(weight.shape)} "
            f"arch={_sm10x_fold_arch()}",
            flush=True,
        )

    major, minor = torch.cuda.get_device_capability()
    return run_fold(
        int(major),
        int(minor),
        weight,
        x_fp4_rowpair,
        weight_scale_base,
        x_sf.view(torch.float8_e4m3fn),
        alpha,
        output_dtype,
    )


def _nvfp4_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_base: torch.Tensor,
    channel_mask: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    num_salient: int,
    weights_padding_cols: int,
    output_size: int,
    fold_eligible: bool,
    is_mext_r1: bool,
) -> torch.Tensor:
    M = x.shape[0]
    N = int(output_size) if int(output_size) > 0 else weight.shape[0]
    return torch.empty(M, N, dtype=x.dtype, device=x.device)


@register_custom_op(
    op_name="residue_nvfp4_linear",
    mutates_args=[],
    fake_impl=_nvfp4_linear_fake,
)
def nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_base: torch.Tensor,
    channel_mask: torch.Tensor,
    alpha: torch.Tensor,
    k_base: int,
    num_salient: int,
    weights_padding_cols: int,
    output_size: int,
    fold_eligible: bool,
    is_mext_r1: bool,
) -> torch.Tensor:
    """Single opaque op covering the residue NVFP4 dense-linear paths.

    Dispatch rule (computed inside the opaque body, so each cudagraph capture
    sees the real M of its capture size):

        use_fold = fold_eligible AND (is_mext_r1 OR force_fold
                                      OR M <= fold_max_m_for(arch))

    Args:
        x: [M, K_base] fp16/bf16 activation.
        weight: [N, K_stored/2] packed FP4 weight (K_stored == K_base for
            mext_r1 / plain layers; K_base + num_salient for extended_k).
        input_global_scale_inv: scalar reciprocal of the input global scale.
        weight_scale: full-K swizzled block scale (plain / K-ext GEMM).
        weight_scale_base: base-K swizzled block scale (fold path; the caller
            passes weight_scale itself when the layer has no separate one).
        channel_mask: salient-channel bitmask (k_ext quant only; 1-byte
            placeholder otherwise).
        alpha: per-tensor output scale.
        k_base: effective K for the fold path GEMM.
        num_salient: salient-channel count (k_ext only; 0 otherwise).
        weights_padding_cols: cutlass weight K-padding in packed bytes; the
            plain/K-ext activation is padded to match.
        output_size: output feature dim; the plain/K-ext output is sliced back
            to it when the weight was N-padded.
        fold_eligible: per-layer constant -- the layer has the fold layout
            prepared (base-K scale + valid k_base).
        is_mext_r1: per-layer constant -- ratio-1.0 M-extension layer. Must
            never fall through to the plain path: it was charged for a
            residue, so it applies one at EVERY M.

    Returns:
        [M, N] output in x.dtype. Caller adds bias and reshapes outside.
    """
    original_m = x.shape[0]
    output_n = weight.shape[0]

    if original_m == 0:
        empty_n = int(output_size) if int(output_size) > 0 else output_n
        return x.new_empty((0, empty_n))

    output_dtype = x.dtype
    fold_max_m = fold_max_m_for(_sm10x_fold_arch())

    use_fold = fold_eligible and (is_mext_r1 or _FORCE_FOLD or original_m <= fold_max_m)

    if use_fold:
        # mext_r1 above the fold band: FlashInfer-tuned two_gemm-vs-k_loop
        # selection per (shape, m-bucket). runners[0]=two_gemm is the
        # designated fallback on any cache miss, and the kloop runner only
        # reports precompiled tiles outside the tuning window -- so this can
        # never JIT while serving. Any tuner-layer error degrades to plain
        # two_gemm rather than breaking the op.
        if is_mext_r1 and original_m > fold_max_m:
            w_view = _base_k_weight_view(weight, k_base)
            if _MEXT_PREFILL_PATH in ("auto", "k_loop"):
                try:
                    from sglang.kernels.ops.gemm.residue_fold.tuners import (
                        tuned_mext_prefill,
                    )

                    return tuned_mext_prefill(
                        x,
                        w_view,
                        input_global_scale_inv,
                        weight_scale_base,
                        alpha,
                        output_dtype,
                        force_kloop=(_MEXT_PREFILL_PATH == "k_loop"),
                    )
                except Exception:  # noqa: BLE001
                    pass
            return _run_mext_r1_two_gemm(
                x,
                w_view,
                input_global_scale_inv,
                weight_scale_base,
                alpha,
                output_dtype,
            )

        if _sm10x_fold_arch() is not None:
            try:
                from sglang.kernels.ops.gemm.residue_fold.tuners import (
                    sm10x_tuner_enabled,
                    tuned_sm10x_fold,
                )

                if sm10x_tuner_enabled():
                    out = tuned_sm10x_fold(
                        x,
                        _base_k_weight_view(weight, k_base),
                        input_global_scale_inv,
                        weight_scale_base,
                        alpha,
                        output_dtype,
                    )
                    return _slice_linear_output(out, int(original_m), int(output_size))
            except ImportError:
                pass

            x_fp4, x_sf = scaled_fp4_quant_mext_r1(
                x, input_global_scale_inv, layout_mode=MEXT_R1_LAYOUT_ROW_PAIR
            )
            out = _run_mext_fold_gemm_sm10x(
                x_fp4,
                _base_k_weight_view(weight, k_base),
                x_sf,
                weight_scale_base,
                alpha,
                output_dtype,
            )
            return _slice_linear_output(out, int(original_m), int(output_size))

        major, minor = torch.cuda.get_device_capability()
        raise RuntimeError(
            "Residue NVFP4 fold supports SM100 and SM103 only; "
            f"got SM{major}{minor}."
        )

    kind = residue_kind_of(num_salient, is_mext_r1)

    if kind == RESIDUE_MEXT_R1:
        # Unreachable today: use_fold is forced true for mext_r1, so we
        # returned above. Assert rather than trust it -- fold_eligible can be
        # False on an unsupported arch, and falling through to the plain path
        # would compute NO residue: not a degraded answer, the wrong one.
        raise RuntimeError(
            "mext_r1 layer reached the non-fold path (fold_eligible="
            f"{bool(fold_eligible)}, M={int(original_m)}). The residue cannot "
            "be applied without the fold kernel and there is no K-extended "
            "weight to fall back to."
        )

    if kind == RESIDUE_NONE:
        x_fp4, x_sf = _plain_fp4_quantize(x, input_global_scale_inv)
        x_fp4 = _pad_activation_cols(x_fp4, int(weights_padding_cols))
        out = _fp4_gemm(x_fp4, x_sf, weight, weight_scale, alpha, output_dtype)
        if out.shape[-1] != int(output_size):
            out = out[..., : int(output_size)].contiguous()
        return out

    # ── K-extension large-M residue path ────────────────────────────────
    # n_ext = K_base + num_salient (per residue contract); M stays as M. All
    # shape math is derived from x.shape and the int num_salient, never from
    # a kernel output's shape.
    from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
        scaled_fp4_quant_with_mask,
    )

    K_base = x.shape[-1]
    n_ext = K_base + int(num_salient)
    num_m_tiles = (original_m + 127) // 128
    num_k_tiles = (n_ext + 63) // 64

    x_fp4, x_sf_block = scaled_fp4_quant_with_mask(
        x, input_global_scale_inv, channel_mask, int(num_salient)
    )
    x_sf_block = x_sf_block.view(torch.float8_e4m3fn).view(
        num_m_tiles * 128, num_k_tiles * 4
    )
    x_fp4 = _pad_activation_cols(x_fp4, int(weights_padding_cols))
    out = _fp4_gemm(x_fp4, x_sf_block, weight, weight_scale, alpha, output_dtype)
    if out.shape[-1] != int(output_size):
        out = out[..., : int(output_size)].contiguous()
    return out
