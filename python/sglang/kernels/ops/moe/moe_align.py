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


# blockDim.x in v2's bucket-scanning kernels: one thread owns one bucket. v1
# reaches 8192 instead, through its per-thread multi-expert path.
V2_MAX_BUCKETS = 1024

# Up to this many pairs, v2 takes a single-CTA path that works purely on the pair
# axis -- one or two pairs per warp, no bucket-wide scan, no vectorized load --
# and is therefore bounded by neither of the two limits around it.
V2_SMALL_NUMEL_LIMIT = 64


def v2_supported(topk_ids: torch.Tensor, num_experts: int, block_size: int) -> bool:
    """The CHECK_HOST list in moe_align_v2.cuh, as a fall-back-to-v1 predicate.

    Kept in sync by hand: violating any of these is a hard error in the kernel,
    not a slow path, so the picker has to know all of them.
    """
    if topk_ids.dtype != torch.int32 or block_size % 4 != 0:
        return False
    if topk_ids.numel() <= V2_SMALL_NUMEL_LIMIT:
        return True
    return (
        num_experts <= V2_MAX_BUCKETS
        and topk_ids.data_ptr() % 16 == 0  # need 16-byte alignment
    )


@cache_once
def _jit_moe_align_v1_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "moe_align_block_size",
        *args,
        cuda_files=["moe/moe_align_v1.cuh"],
        cuda_wrappers=[("run", f"MoeAlignBlockSizeKernel<{args}>::run")],
    )


@cache_once
def _jit_moe_align_v2_module(ignore_invalid_expert: bool) -> Module:
    args = make_cpp_args(ignore_invalid_expert, is_arch_support_pdl())
    return load_jit(
        "moe_align_v2",
        *args,
        cuda_files=["moe/moe_align_v2.cuh"],
        cuda_wrappers=[("run", f"sglang::moe_align_v2<{args}>")],
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
    version: int = 0,  # 0 = auto-pick, 1 = v1, 2 = v2
) -> None:
    """Destination-passing align-block-size; ``num_experts`` is the bucket count.

    CUDA only -- see the ``capabilities`` on its KernelSpec. Registered as the
    JIT backend of ``moe.moe_align_block_size_out``; reach it through
    ``sglang.kernels.ops.moe.moe_align_block_size``.

    v2 is the default wherever it runs; v1 covers what it cannot (wide expert
    counts, non-int32 ids, odd block sizes, unaligned inputs). Pass ``version``
    to pin one, which benchmarks do to compare them.
    """
    if version == 0:
        version = 2 if v2_supported(topk_ids, num_experts, block_size) else 1
    if version == 1:
        assert not ignore_invalid_expert, "TODO: support ignore_invalid_expert=True"
        module = _jit_moe_align_v1_module(topk_ids.dtype)
    else:
        module = _jit_moe_align_v2_module(ignore_invalid_expert)
    module.run(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )
