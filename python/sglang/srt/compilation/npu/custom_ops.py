from typing import List

import sgl_kernel_npu.norm.split_qkv_rmsnorm_rope
import torch


@torch.library.custom_op("sglang::split_qkv_rmsnorm_rope", mutates_args=())
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
    q, k, v = sgl_kernel_npu.norm.split_qkv_rmsnorm_rope.split_qkv_rmsnorm_rope(
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

@split_qkv_rmsnorm_rope.register_fake
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
    # TODO: generalize shape
    q = torch.empty((128, 2048), dtype=input.dtype, device=input.device)
    k = torch.empty((128, 256), dtype=input.dtype, device=input.device)
    v = torch.empty((128, 256), dtype=input.dtype, device=input.device)
    return [q, k, v]
