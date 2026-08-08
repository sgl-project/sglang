from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_align_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_align_block_size",
        *args,
        cuda_files=["moe/moe_align_kernel.cu"],
        cuda_wrappers=[
            ("moe_align_block_size", f"MoeAlignBlockSizeKernel<{args}>::run"),
        ],
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
    """Destination-passing align-block-size; ``num_experts`` is the bucket count.

    CUDA only -- see the ``capabilities`` on its KernelSpec. Registered as the
    JIT backend of ``moe.moe_align_block_size_out``; reach it through
    ``sglang.kernels.ops.moe.moe_align_block_size``.
    """
    assert not ignore_invalid_expert, "TODO: support ignore_invalid_expert=True"
    module = _jit_moe_align_module(topk_ids.dtype)
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
