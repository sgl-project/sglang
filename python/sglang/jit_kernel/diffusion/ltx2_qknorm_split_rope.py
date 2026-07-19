from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ltx2_qknorm_split_rope_module() -> Module:
    return load_jit(
        "diffusion_ltx2_qknorm_split_rope",
        cuda_files=["diffusion/ltx2_qknorm_split_rope.cuh"],
        cuda_wrappers=[
            (
                "ltx2_qknorm_split_rope_pair",
                "sglang_ltx2_qknorm_split_rope::LTX2QKNormSplitRopeKernel::run",
            )
        ],
    )


def _fake_impl(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q, dtype=torch.bfloat16), torch.empty_like(
        k, dtype=torch.bfloat16
    )


@register_custom_op(
    op_name="diffusion_ltx2_qknorm_split_rope",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _ltx2_qknorm_split_rope_custom_op(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty_like(q, dtype=torch.bfloat16)
    k_out = torch.empty_like(k, dtype=torch.bfloat16)
    module = _jit_ltx2_qknorm_split_rope_module()
    module.ltx2_qknorm_split_rope_pair(
        q_out,
        k_out,
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        float(eps),
        int(num_heads),
        int(head_dim),
    )
    return q_out, k_out


def _supported_side(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> bool:
    return (
        x.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and weight.is_cuda
        and x.device == cos.device == sin.device == weight.device
        and x.dtype == torch.bfloat16
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.ndim == 3
        and cos.ndim == 4
        and sin.ndim == 4
        and x.is_contiguous()
        and cos.shape == sin.shape
        and cos.shape[0] == x.shape[0]
        and cos.shape[1] == num_heads
        and cos.shape[2] == x.shape[1]
        and cos.shape[3] * 2 == head_dim
        and x.shape[2] == num_heads * head_dim
        and x.shape[2] == weight.shape[0]
        and weight.ndim == 1
        and head_dim % 2 == 0
        and x.shape[2] % 4 == 0
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
    )


def _is_sm100_or_newer(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    try:
        return torch.cuda.get_device_capability(x.device)[0] >= 10
    except RuntimeError:
        return False


def can_use_ltx2_qknorm_split_rope_cuda(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> bool:
    return (
        _is_sm100_or_newer(q)
        and _supported_side(
            q,
            q_cos,
            q_sin,
            q_weight,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        and _supported_side(
            k,
            k_cos,
            k_sin,
            k_weight,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    )


def ltx2_qknorm_split_rope_cuda(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    ):
        raise RuntimeError("unsupported input for LTX2 QKNorm split-RoPE CUDA")
    return _ltx2_qknorm_split_rope_custom_op(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        float(eps),
        int(num_heads),
        int(head_dim),
    )
