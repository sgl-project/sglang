"""Compiler boundary for the fused QKV kernel used by Qwen3 on Ascend."""

from typing import Optional, Tuple

import torch
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope


@torch.library.custom_op("sglang::npu_split_qkv_rmsnorm_rope", mutates_args=())
def npu_split_qkv_rmsnorm_rope(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: Optional[float] = None,
    q_weight: Optional[torch.Tensor] = None,
    k_weight: Optional[torch.Tensor] = None,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    is_neox_style: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep the Python device-property query and Triton launcher outside Dynamo.
    return split_qkv_rmsnorm_rope(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )


@npu_split_qkv_rmsnorm_rope.register_fake
def _fake_split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
):
    return (
        input.new_empty((input.shape[0], q_hidden_size)),
        input.new_empty((input.shape[0], kv_hidden_size)),
        input.new_empty((input.shape[0], kv_hidden_size)),
    )
