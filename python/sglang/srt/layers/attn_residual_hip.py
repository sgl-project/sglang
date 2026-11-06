"""Kimi-K3's gfx950 TP8 column tail into the attention-residual bank: the
column-parallel latent tail's all-gather of the next stream computes the next
attention read too, and stages it on the bank for that read's ``forward()``."""

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.attention import k3_attn_res_read_hip
from sglang.srt.layers.attn_residual import AttnResidual, get_cw
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.k3_ll_comm_hip import LLComm


class StagedRead(msgspec.Struct, frozen=True):
    """An attention read computed ahead of its stage (by the kernel that
    gathered its input), waiting for the read it stands in for."""

    prefix: torch.Tensor
    score_proj: ReplicatedLinear
    normed: torch.Tensor
    wrote_bank: bool


def get_fp32_norm_weight(norm: RMSNorm) -> torch.Tensor:
    """Cached fp32 copy of an output RMSNorm weight, the dtype the fused gfx950
    collective reads consume."""
    weight = getattr(norm, "_attn_res_fp32_weight", None)
    if weight is None:
        weight = norm._attn_res_fp32_weight = norm.weight.float().contiguous()
    return weight


def gather_covers(
    bank: AttnResidual, num_tokens: int, slice_width: int, write: bool
) -> bool:
    """Whether gather_tail_read takes [num_tokens, slice_width] column slices."""
    return not bank.written_on_slice and k3_attn_res_read_hip.gather_shape_covered(
        num_tokens, slice_width, bank.block_residual, bank.num_valid_blocks, write
    )


def gather_tail_read(
    bank: AttnResidual,
    part: torch.Tensor,
    sq: Optional[torch.Tensor],
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor],
    tail_eps: float,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
    write: bool,
    comm: "LLComm",
) -> torch.Tensor:
    """Finish this rank's TP8 column slice of the next stream from the
    column-parallel latent tail's partials, all-gather the stream and
    stage the attention read of it (the next forward() with these scoring
    parameters); returns the stream. The caller checked gather_covers()
    and k3_attn_res_read_hip.tail_covered()."""
    normed, stream = k3_attn_res_read_hip.gather_tail_attn_res_read(
        part,
        sq,
        shared,
        prefix,
        tail_eps,
        bank.block_residual,
        bank.num_valid_blocks,
        get_cw(score_proj, score_norm),
        get_fp32_norm_weight(out_norm),
        score_norm.variance_epsilon,
        out_norm.variance_epsilon,
        comm,
        write_bank=write,
    )
    bank.staged = StagedRead(stream, score_proj, normed, write)
    return stream
