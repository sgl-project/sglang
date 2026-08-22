"""Torch entry point for the MLX block paged attention decode kernel.

Torch owns the tensors on Apple silicon, so this module keeps the MLX crossing
in one place: it validates the Torch-side contract, then hands every input to
``mlx_call``, which fences MPS once, imports the inputs zero-copy, evaluates the
kernel, and exports the result through DLPack.  The kernel itself and its
shape rules live in ``sgl_kernel.metal``; nothing here duplicates them.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from sglang.srt.utils.tensor_bridge import mlx_call

_INSTALL_HINT = (
    "Install the Metal kernels with "
    "`uv run python/sglang/kernels/aot/setup_metal.py install` from the SGLang "
    "repo root in the active environment."
)


@functools.lru_cache(maxsize=1)
def _load_kernel() -> Any:
    try:
        from sgl_kernel import metal
    except ImportError as exc:
        raise ImportError(
            f"sgl_kernel.metal is not importable. {_INSTALL_HINT}"
        ) from exc

    import_error = getattr(metal, "_IMPORT_ERROR", None)
    if getattr(metal, "_metal", None) is None or import_error is not None:
        reason = f" Reason: {import_error}." if import_error is not None else ""
        raise ImportError(
            "sgl_kernel.metal is importable, but the native Metal extension "
            f"or metallib is not available.{reason} {_INSTALL_HINT}"
        ) from import_error
    return metal.block_paged_attention_decode


def _check_mps_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor, got {type(tensor).__name__}")
    if tensor.device.type != "mps":
        raise ValueError(
            f"block_paged_attention_decode requires MPS tensors, got "
            f"{name} on {tensor.device}"
        )
    if not tensor.is_contiguous():
        # A borrowed DLPack import must describe the same layout the kernel
        # indexes; make the caller's copy explicit rather than hiding one here.
        raise ValueError(
            f"block_paged_attention_decode requires contiguous tensors, "
            f"got a non-contiguous {name}"
        )


def block_paged_attention_decode(
    q: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Decode attention over a block-table KV layout, with Torch tensors.

    Args:
        q: Query tensor with shape ``[batch, num_qo_heads, head_dim]``.
        k_blocks: Key blocks with shape
            ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        v_blocks: Value blocks with the same shape and dtype as ``k_blocks``.
        block_tables: ``int32`` physical block IDs with shape
            ``[batch, max_num_blocks]``; ``-1`` entries are padding.
        seq_lens: ``int32`` visible sequence lengths with shape ``[batch]``.
        num_qo_heads: Query head count; must be divisible by ``num_kv_heads``.
        num_kv_heads: Key/value head count.
        head_dim: Per-head dimension; the kernel supports ``head_dim <= 256``.
        block_size: Tokens per KV block.
        sm_scale: Softmax scale applied to the query-key dot products.

    Returns:
        Torch MPS tensor with shape ``[batch, num_qo_heads, head_dim]`` and
        ``q``'s dtype.
    """
    kernel = _load_kernel()

    for name, tensor in (
        ("q", q),
        ("k_blocks", k_blocks),
        ("v_blocks", v_blocks),
        ("block_tables", block_tables),
        ("seq_lens", seq_lens),
    ):
        _check_mps_contiguous(name, tensor)

    if q.dtype != k_blocks.dtype or q.dtype != v_blocks.dtype:
        raise ValueError(
            "block_paged_attention_decode requires a shared q/k_blocks/v_blocks "
            f"dtype, got {q.dtype}, {k_blocks.dtype}, {v_blocks.dtype}"
        )
    if block_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError(
            "block_paged_attention_decode requires int32 block_tables and "
            f"seq_lens, got {block_tables.dtype} and {seq_lens.dtype}"
        )

    def _run(q_a, k_a, v_a, tables_a, lens_a):
        return kernel(
            q_a,
            k_a,
            v_a,
            tables_a,
            lens_a,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            sm_scale=float(sm_scale),
        )

    return mlx_call(_run, q, k_blocks, v_blocks, block_tables, seq_lens)


__all__ = ["block_paged_attention_decode"]
