"""Fused BF16 QK RMSNorm and full-width NeoX multimodal RoPE."""

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


@cache_once
def _module(write_cache: bool = False):
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "fused_qk_norm_mrope",
        args,
        str(write_cache),
        cuda_files=["attention/fused_qk_norm_mrope.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"FusedQKNormMRoPE<{args}>::"
                + ("run_with_cache" if write_cache else "run"),
            )
        ],
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
    *,
    value: torch.Tensor | None = None,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    slots: torch.Tensor | None = None,
) -> None:
    """Normalize and rotate Q/K in place; BF16, head size 128, NeoX layout.

    ``axis_map`` maps each of the 64 rotary pairs to one of three position
    rows, supporting both contiguous and interleaved multimodal sections.
    The normalization rounds to BF16 before the BF16 rotary operations.
    Optional cache arguments write rotated K and unmodified V in the same
    launch. Cache views have logical [page, token, head, 128] dimensions and
    matching strides; negative int64 slots are skipped. Q/K remain in place.
    """
    if value is not None:
        if key_cache is None or value_cache is None or slots is None:
            raise ValueError(
                "Cache writes require value, key_cache, value_cache, and slots"
            )
        _module(True).run(
            q,
            k,
            value,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            axis_map,
            key_cache,
            value_cache,
            slots,
            eps,
        )
    else:
        if key_cache is not None or value_cache is not None or slots is not None:
            raise ValueError("Cache writes require value")
        _module().run(q, k, q_weight, k_weight, cos_sin_cache, positions, axis_map, eps)
