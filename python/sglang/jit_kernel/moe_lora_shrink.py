from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_lora_shrink_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    arch_env = nullcontext()
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 10:
            arch_env = override_jit_cuda_arch(major, minor, suffix="a")
    with arch_env:
        return load_jit(
            "moe_lora_shrink",
            *args,
            cuda_files=["lora/moe_lora_shrink_kernel.cu"],
            cuda_wrappers=[
                ("moe_lora_shrink", f"MoeLoraShrinkKernel<{args}>::run"),
            ],
        )


def moe_lora_shrink(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_size_m: int,
) -> torch.Tensor:
    """MoE LoRA-A "shrink" grouped GEMM (SPLIT_K == 1).

    Computes, for every routed (token, top-k slot) ``t``::

        output[t] = hidden_states[t // top_k] @ lora_a[expert(t)].T

    Tokens are grouped by virtual expert via the moe_align routing buffers
    (``sorted_token_ids`` / ``expert_ids`` / ``num_tokens_post_padded``), the
    same layout the fused-MoE kernels use. ``block_size_m`` must match the
    ``block_size`` used to build those routing buffers. Only routed output rows
    are written (non-owned / sentinel slots are left untouched), matching the
    Triton SPLIT_K == 1 path.

    Parameters
    ----------
    output                  : [num_tokens * top_k, N] CUDA tensor, written in place.
    hidden_states           : [num_tokens, K] CUDA tensor.
    lora_a                  : [num_virtual_experts, N, K] merged LoRA-A weights
                              (N == lora rank, K == hidden size).
    sorted_token_ids        : int32 routing buffer from moe_align_block_size.
    expert_ids              : int32 per-m-block expert ids (-1 sentinel for padding).
    num_tokens_post_padded  : int32 [1] padded token count.
    top_k                   : router top-k.
    block_size_m            : routing block size (must equal the align block size).

    Returns
    -------
    The ``output`` tensor (written in place).
    """
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"Unsupported dtype {hidden_states.dtype}. The WMMA shrink supports "
            "float16, bfloat16 (tensor-core MMA); float32 is not supported."
        )
    if block_size_m != 16:
        raise RuntimeError(
            f"WMMA shrink requires block_size_m == 16 (one WMMA M-tile), got {block_size_m}"
        )
    if not (hidden_states.dtype == lora_a.dtype == output.dtype):
        raise RuntimeError(
            "hidden_states, lora_a and output must share dtype, got "
            f"{hidden_states.dtype}, {lora_a.dtype}, {output.dtype}"
        )
    rank = lora_a.shape[1]
    if rank not in (16, 32, 64):
        raise RuntimeError(f"m16n8 shrink supports rank 16, 32 or 64, got rank {rank}")
    hidden = lora_a.shape[2]
    if hidden % 256 != 0:
        raise RuntimeError(
            f"m16n8 shrink requires hidden size divisible by 256, got {hidden}"
        )

    module = _jit_moe_lora_shrink_module(hidden_states.dtype)
    module.moe_lora_shrink(
        output,
        hidden_states,
        lora_a,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
    )
    return output
