"""Fused BF16 QK RMSNorm and full-width NeoX multimodal RoPE."""

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


@cache_once
def _module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "fused_qk_norm_mrope",
        args,
        cuda_files=["attention/fused_qk_norm_mrope.cuh"],
        cuda_wrappers=[("run", f"FusedQKNormMRoPE<{args}>::run")],
    )


def fused_qk_norm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    axis_map: torch.Tensor,
    eps: float,
) -> None:
    """Normalize and rotate Q/K in place; BF16, head size 128, NeoX layout.

    ``axis_map`` maps each of the 64 rotary pairs to one of three position
    rows, supporting both contiguous and interleaved multimodal sections.
    The normalization rounds to BF16 before the BF16 rotary operations.
    """
    _module().run(q, k, q_weight, k_weight, cos_sin_cache, positions, axis_map, eps)
