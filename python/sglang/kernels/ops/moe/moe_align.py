from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_align_module(
    dtype: torch.dtype,
    write_inactive_prefix_count: bool = False,
) -> Module:
    args = make_cpp_args(dtype, write_inactive_prefix_count)
    return load_jit(
        "moe_align_block_size",
        *args,
        cuda_files=["moe/moe_align_kernel.cu"],
        cuda_wrappers=[
            ("moe_align_block_size", f"MoeAlignBlockSizeKernel<{args}>::run"),
        ],
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
    *,
    write_inactive_prefix_count: bool = False,
) -> None:
    """Align +1-offset buckets, optionally publishing a padded-token/block pair.

    ``write_inactive_prefix_count`` requires an aligned int32[2] count output.
    """
    module = _jit_moe_align_module(topk_ids.dtype, write_inactive_prefix_count)
    module.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )
