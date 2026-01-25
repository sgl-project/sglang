from typing import List

import sgl_kernel_npu.norm.split_qkv_rmsnorm_rope as sgl_kernel_npu
import torch

from sglang.srt.utils.custom_op import register_custom_op


def split_qkv_rmsnorm_rope_fake(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hiddem_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
) -> List[torch.Tensor]:
    # TODO: generalize shape
    q = torch.empty(
        (input.shape[0], q_hidden_size), dtype=input.dtype, device=input.device
    )
    k = torch.empty(
        (input.shape[0], kv_hiddem_size), dtype=input.dtype, device=input.device
    )
    v = torch.empty(
        (input.shape[0], kv_hiddem_size), dtype=input.dtype, device=input.device
    )
    return [q, k, v]


@register_custom_op(fake_impl=split_qkv_rmsnorm_rope_fake)
def split_qkv_rmsnorm_rope(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hiddem_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
) -> List[torch.Tensor]:
    q, k, v = sgl_kernel_npu.split_qkv_rmsnorm_rope(
        input,
        sin,
        cos,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hiddem_size,
        head_dim,
        eps,
        q_bias,
        k_bias,
    )
    return [q, k, v]
