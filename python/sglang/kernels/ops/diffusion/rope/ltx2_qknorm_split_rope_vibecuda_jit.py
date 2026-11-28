"""Bit-exact SM100/SM103 LTX-2 QK RMSNorm and split-half RoPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kda_kernels.ltx2_qknorm_split_rope_jit import (
    _supported_side,
    _validate_output_aliases,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _require_supported_arch(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    cuda = current_arch().cuda
    if cuda is None or cuda.capability not in ((10, 0), (10, 3)):
        capability = None if cuda is None else cuda.capability
        raise RuntimeError(
            f"vibecuda LTX2 QKNorm split-RoPE requires SM100 or SM103; got {capability}"
        )
    return True


@cache_once
def _jit_ltx2_qknorm_split_rope_vibecuda_module() -> Module:
    cuda = current_arch().cuda
    if cuda is None or cuda.capability not in ((10, 0), (10, 3)):
        capability = None if cuda is None else cuda.capability
        raise RuntimeError(
            f"vibecuda LTX2 QKNorm split-RoPE requires SM100 or SM103; got {capability}"
        )
    return load_jit(
        "diffusion_ltx2_qknorm_split_rope_vibecuda",
        cuda_files=["diffusion/ltx2_qknorm_split_rope_vibecuda.cuh"],
        cuda_wrappers=[
            (
                "ltx2_qknorm_split_rope_pair",
                (
                    "ltx2_qknorm_split_rope_vibecuda::"
                    "VibeCUDALTX2QKNormSplitRopeKernel::run"
                ),
            )
        ],
    )


def _ltx2_qknorm_split_rope_vibecuda_into(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
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
) -> None:
    _jit_ltx2_qknorm_split_rope_vibecuda_module().ltx2_qknorm_split_rope_pair(
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
    op_name="diffusion_ltx2_qknorm_split_rope_vibecuda",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _ltx2_qknorm_split_rope_vibecuda_custom_op(
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
    _ltx2_qknorm_split_rope_vibecuda_into(
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


def can_use_ltx2_qknorm_split_rope_vibecuda(
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
        _require_supported_arch(q)
        and head_dim % 8 == 0
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


def ltx2_qknorm_split_rope_vibecuda(
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
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 8 != 0:
        raise RuntimeError("vibecuda LTX2 QKNorm split-RoPE requires head_dim % 8 == 0")
    if not can_use_ltx2_qknorm_split_rope_vibecuda(
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
        raise RuntimeError(
            "unsupported input for vibecuda LTX2 QKNorm split-RoPE; "
            "requires SM100/SM103 CUDA tensors, contiguous BF16 Q/K and weights, "
            "BF16 unit-inner-stride RoPE tables, and head_dim divisible by 8"
        )
    if out is not None:
        _validate_output_aliases(
            out, (q, q_cos, q_sin, q_weight, k, k_cos, k_sin, k_weight)
        )
        if torch.compiler.is_compiling():
            raise RuntimeError("out= is supported for eager and CUDA Graph calls only")
        _ltx2_qknorm_split_rope_vibecuda_into(
            *out,
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
        return out
    return _ltx2_qknorm_split_rope_vibecuda_custom_op(
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
