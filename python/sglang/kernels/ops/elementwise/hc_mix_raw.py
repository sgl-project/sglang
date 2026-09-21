"""Explicitly prepared, static-weight small-T raw-state HC primitives.

Prepare outside execution. The shared branch-major mode reuses the large-T
Down/Inject storage; the standalone two-dimensional weight interface remains.
"""

from dataclasses import dataclass

import torch

from sglang.kernels.ops.elementwise.hc_mix import (
    pad_lowrank,
    permute_pad_up_weight,
)


@dataclass(frozen=True)
class PreparedRawHCWeights:
    down_inject: torch.Tensor
    up: torch.Tensor
    norm_permuted: torch.Tensor
    scratch: torch.Tensor
    hc_count: int
    hidden_size: int
    lowrank: int
    branch_packed: bool = False


@torch.no_grad()
def prepare_raw_hc_weights_from_batched(
    weights, up, norm_weight, *, norm_weight_permuted=None
):
    """Share prepared Down and optional [H,C] RMS storage with large-T."""
    c, h, lowrank = weights.hc_count, weights.hidden_size, weights.lowrank
    matrix = weights.matrix
    if (
        c != 4
        or not 0 < h <= 16384
        or h % 512
        or lowrank <= 0
        or lowrank % 8
        or lowrank + c > pad_lowrank(lowrank)
        or matrix.ndim != 3
        or matrix.shape[:2] != (c, h)
        or matrix.shape[2] < lowrank + c
        or matrix.stride() != (h * matrix.shape[2], 1, h)
        or not matrix.is_cuda
        or matrix.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise ValueError("unsupported shared raw HC weight layout")
    if norm_weight.shape != (c * h,) or norm_weight.dtype != matrix.dtype:
        raise ValueError("RMS checkpoint must match prepared HC dimensions/dtype")
    if norm_weight_permuted is None:
        if norm_weight.device != matrix.device:
            raise ValueError("RMS source must match the prepared HC device")
        norm_weight_permuted = norm_weight.view(c, h).T.contiguous().flatten()
    for tensor, shape in ((up, (c * h, lowrank)), (norm_weight_permuted, (c * h,))):
        if (
            tensor.shape != shape
            or tensor.device != matrix.device
            or tensor.dtype != matrix.dtype
        ):
            raise ValueError("shared HC weights must match dimensions/device/dtype")
    if not norm_weight_permuted.is_contiguous() or norm_weight_permuted.data_ptr() % 16:
        raise ValueError("shared RMS weights must be contiguous and 16-byte aligned")
    return PreparedRawHCWeights(
        matrix,
        permute_pad_up_weight(up, c),
        norm_weight_permuted,
        torch.zeros(
            (24, pad_lowrank(lowrank)), device=matrix.device, dtype=matrix.dtype
        ),
        c,
        h,
        lowrank,
        branch_packed=True,
    )


@torch.no_grad()
def prepare_raw_hc_weights(down, inject, up, norm_weight, hc_count=4):
    if hc_count != 4 or down.ndim != 2:
        raise ValueError("raw HC requires C=4 and a matrix Down weight")
    lowrank, width = down.shape
    h = width // hc_count
    if (
        width != hc_count * h
        or h % 512
        or not 0 < h <= 16384
        or lowrank <= 0
        or lowrank % 8
        or lowrank + hc_count > pad_lowrank(lowrank)
    ):
        raise ValueError("unsupported raw HC hidden/lowrank geometry")
    for w, shape in (
        (inject, (hc_count, width)),
        (up, (width, lowrank)),
        (norm_weight, (width,)),
    ):
        if w.shape != shape or w.dtype != down.dtype or w.device != down.device:
            raise ValueError(
                "raw HC weights must have matching device/dtype and shapes"
            )
    if not down.is_cuda or down.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("raw HC requires CUDA BF16/FP16 weights")
    # FP32 folding, one storage cast. This intentionally changes rounding
    # compared with materializing a low-precision normalized activation.
    packed = torch.cat((down, inject), dim=0)
    folded = (packed.float() * (1.0 + norm_weight.float())).to(down.dtype)
    norm_perm = norm_weight.view(hc_count, h).T.contiguous().flatten()
    return PreparedRawHCWeights(
        folded,
        permute_pad_up_weight(up, hc_count),
        norm_perm,
        torch.zeros((24, pad_lowrank(lowrank)), device=down.device, dtype=down.dtype),
        hc_count,
        h,
        lowrank,
    )


def hc_mix_raw(
    residual,
    sum_sq,
    weights: PreparedRawHCWeights,
    *,
    next_sum_sq,
    rms_eps=1e-6,
    out=None,
    alpha=None,
):
    """Sum-consuming Down+Inject -> Up/Mix; one stream per prepared object."""
    from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
        SplitKTactic,
        run_splitk_dense_gate,
        run_splitk_dense_silu_aux,
    )

    c, h, lowrank = weights.hc_count, weights.hidden_size, weights.lowrank
    if sum_sq is None or next_sum_sq is None:
        raise ValueError("current and next square-sum tensors are required")
    if (
        residual.ndim != 2
        or residual.shape[1] != c * h
        or residual.shape[0] > 24
        or not residual.is_contiguous()
    ):
        raise ValueError("raw HC only accepts contiguous [T,C*H], T<=24")
    rows = residual.shape[0]
    if out is None:
        out = residual.new_empty((rows, h))
    if alpha is None:
        alpha = torch.empty((rows, c), dtype=torch.float32, device=residual.device)
    if rows == 0:
        return out, alpha
    packed_n = lowrank + c
    t_pad = weights.scratch[:rows]
    # Uniform retained sum-state Down policy, not a model/T-value dispatch.
    down_tactic = SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=3)
    run_splitk_dense_silu_aux(
        residual,
        weights.down_inject if weights.branch_packed else weights.down_inject.T,
        t_pad[:, :packed_n],
        alpha,
        True,
        down_tactic,
        1.0 / c,
        lowrank,
        "alpha_only",
        sum_sq=sum_sq,
        rms_eps=rms_eps,
        branch_packed=weights.branch_packed,
    )
    # Structural small-T policy: output-channel tile 64, token tile
    # 8/16/32. No model-name, fixed-H or individual benchmark-T branches.
    up_tactic = SplitKTactic(
        mma_m=64,
        mma_n=8 if rows <= 8 else 16 if rows <= 16 else 32,
        split_k=1,
        ab_stages=3,
    )
    run_splitk_dense_gate(
        t_pad,
        weights.up.T,
        residual,
        out,
        True,
        up_tactic,
        1.0 / c,
        c,
        norm_weight_permuted=weights.norm_permuted,
        sum_sq=sum_sq,
        next_sum_sq=next_sum_sq,
        rms_eps=rms_eps,
    )
    return out, alpha
