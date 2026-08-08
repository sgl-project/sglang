"""moe_align with two paths behind one entry point.

Drop-in for ``sglang.kernels.ops.moe.moe_align.moe_align_block_size_out``: same
argument list, same buffer contract, same "+1 offset" convention, plus the AOT
kernel's ``ignore_invalid_expert``. Small batches run one fused launch; above its
capacity a two-launch histogram/scan + scatter path takes over. See
``kernels/jit/csrc/moe/moe_align_v2.cuh``.

Remaining differences from the kernel it replaces:
  - the bucket count is capped at ``CTA_SIZE``, where the AOT/JIT kernel reaches
    8192 via a separate path (only LoRA virtual experts get that wide);
  - ``block_size`` must be a multiple of 4 (every real BLOCK_SIZE_M is);
  - ``topk_ids`` must be 16-byte aligned, which a fresh tensor always is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# blockDim.x in the kernels; one thread owns one bucket, so the bucket count
# (num_experts + 1 at the call site) must not exceed this.
CTA_SIZE = 1024


@cache_once
def _jit_moe_align_v2_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "moe_align_v2",
        *args,
        cuda_files=["moe/moe_align_v2.cuh"],
        cuda_wrappers=[("moe_align_v2", f"sglang::moe_align_v2<{args}>")],
    )


def moe_align_block_size_out(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
    ignore_invalid_expert: bool = False,
) -> None:
    """
    ``num_experts`` is the bucket count ``E + 1``, i.e. exactly what the
    moe_runner call site passes as ``num_experts + 1``.

    NOTE: `pad_sorted_token_ids` is always ignored (treated as true)
    """
    module = _jit_moe_align_v2_module(is_arch_support_pdl())
    module.moe_align_v2(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
        ignore_invalid_expert,
    )
