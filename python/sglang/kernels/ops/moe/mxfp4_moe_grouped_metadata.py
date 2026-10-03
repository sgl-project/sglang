from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# The scan gives one expert one thread, so the CTA width caps the expert count; keep in sync with
# kMetadataBlock in mxfp4_moe_grouped_metadata.cu.
MXFP4_MOE_FUSED_METADATA_MAX_EXPERTS = 1024


@cache_once
def _jit_mxfp4_moe_grouped_metadata_module() -> Module:
    return load_jit(
        "mxfp4_moe_grouped_metadata",
        cuda_files=["moe/mxfp4_moe_grouped_metadata.cu"],
        header_only=False,
    )


@register_custom_op(
    op_name="mxfp4_moe_grouped_metadata",
    mutates_args=["expert_offsets", "problem_sizes1", "problem_sizes2", "src2dst"],
)
def mxfp4_moe_grouped_metadata(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    src2dst: torch.Tensor,
    n: int,
    k: int,
) -> None:
    """Fills the per-expert grouped-GEMM metadata and the expert-grouped row permutation.

    :param topk_ids: [num_tokens, topk] int32 expert ids.
    :param expert_offsets: [num_experts + 1] int32 out, exclusive prefix sum of the row counts.
    :param problem_sizes1: [num_experts, 3] int32 out, per-expert (2 * n, rows, k).
    :param problem_sizes2: [num_experts, 3] int32 out, per-expert (k, rows, n).
    :param src2dst: [num_tokens * topk] int32 out, flat topk index -> expert-sorted row.
    :param n: Per-rank intermediate size.
    :param k: Hidden size.
    """
    _jit_mxfp4_moe_grouped_metadata_module().mxfp4_moe_grouped_metadata(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        src2dst,
        n,
        k,
    )
