"""Fused RoPE + Hadamard for the NSA Lightning Indexer.

Replaces the chain
  ``apply_rope_with_cos_sin_cache_inplace(q_rope, k_rope, ...)`` (1 launch)
  ``hadamard_transform(query, scale=1/sqrt(head_dim))``           (1 launch)
  ``hadamard_transform(key,   scale=1/sqrt(head_dim))``           (1 launch)
with a single in-place launch that handles both ``q`` and ``k``.

Public entrypoint: :func:`fused_rope_hadamard`.

Architecture: portable from SM70+ (warp shuffles + 128-bit vec loads); PDL hooks
gate to SM90+ via :func:`is_arch_support_pdl`, which covers SM90 (Hopper),
SM100 (B200), and SM103 (B300).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    KERNEL_PATH,
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_fused_rope_hadamard_module(
    is_neox: bool,
    head_dim: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> Module:
    """JIT-compile (and cache) one specialization per (is_neox, head_dim, rope_dim, dtype).

    PDL is folded into the cache key indirectly: ``is_arch_support_pdl()`` returns
    a per-process constant, so each architecture lane gets its own specialization.
    """
    pdl = is_arch_support_pdl()
    args = make_cpp_args(
        is_neox,
        head_dim,
        rope_dim,
        pdl,
        dtype,
    )
    hadamard_include_dir = (KERNEL_PATH / "csrc" / "fast-hadamard-transform").resolve()
    logger.info(
        "[NSA] JIT-compiling fused_rope_hadamard "
        "(is_neox=%s, head_dim=%d, rope_dim=%d, dtype=%s, pdl=%s)",
        is_neox,
        head_dim,
        rope_dim,
        dtype,
        pdl,
    )
    module = load_jit(
        "fused_rope_hadamard",
        *args,
        cuda_files=["elementwise/fused_rope_hadamard.cuh"],
        cuda_wrappers=[
            ("run", f"FusedRopeHadamardKernel<{args}>::run"),
        ],
        extra_include_paths=[str(hadamard_include_dir)],
    )
    logger.info(
        "[NSA] fused_rope_hadamard JIT compile complete "
        "(is_neox=%s, head_dim=%d, rope_dim=%d, dtype=%s)",
        is_neox,
        head_dim,
        rope_dim,
        dtype,
    )
    return module


@register_custom_op(
    op_name="nsa_fused_rope_hadamard",
    mutates_args=["q", "k"],
)
def fused_rope_hadamard(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool = True,
    rope_dim: int = 0,
) -> None:
    """Fused in-place RoPE (rope-half) + per-head Hadamard (full head_dim).

    Args:
        q: Query tensor of shape ``[num_tokens, num_qo_heads, head_dim]``, fp16/bf16.
            The first ``rope_dim`` columns of each head are RoPE-rotated; the full
            ``head_dim`` is then Hadamard-transformed and scaled by ``1/sqrt(head_dim)``.
            Modified in place.
        k: Key tensor of shape ``[num_tokens, num_kv_heads, head_dim]``, same dtype/device
            as ``q``. Same per-head treatment. Modified in place.
        cos_sin_cache: ``[max_position, rope_dim]`` fp32. The first ``rope_dim/2``
            columns are cos values; the second ``rope_dim/2`` are sin values
            (matching :mod:`sglang.jit_kernel.rope`'s layout).
        positions: ``[num_tokens]`` int32 or int64 position indices.
        is_neox: True for GPT-NeoX-style (paired rotation across the rope-half
            midpoint). False is not yet supported (asserted in the kernel).
        rope_dim: rope dimension; defaults to ``cos_sin_cache.size(-1)``.

    Notes:
        * ``head_dim`` must be a power of two and equal between ``q`` and ``k``.
        * ``rope_dim`` must satisfy ``rope_dim % (2 * (16 / sizeof(dtype))) == 0``
          (so the rope/nope split aligns to vector lanes; e.g. multiple of 16 for bf16).
        * ``head_dim / (16 / sizeof(dtype))`` must be in ``[4, 32]``: the kernel
          uses sub-warp/warp Hadamard butterflies and does not stage across warps
          in this version. For bf16/fp16 this covers ``head_dim ∈ {32, 64, 128, 256}``.
        * ``q`` and ``k`` must share the same per-head stride.

    Example:
        >>> # In nsa_indexer._get_q_k_bf16, replace:
        >>> #   q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
        >>> #   self._update_rope_guarded(query[..., :rope_head_dim], q_rope)
        >>> #   self._update_rope_guarded(key[..., :rope_head_dim], k_rope)
        >>> #   query = rotate_activation(query)
        >>> #   key   = rotate_activation(key)
        >>> # with:
        >>> fused_rope_hadamard(
        ...     q=query,
        ...     k=key,
        ...     cos_sin_cache=self.rotary_emb.cos_sin_cache,
        ...     positions=positions,
        ...     is_neox=self.rotary_emb.is_neox_style,
        ...     rope_dim=self.rope_head_dim,
        ... )
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    head_dim = q.size(-1)
    module = _jit_fused_rope_hadamard_module(is_neox, head_dim, rope_dim, q.dtype)
    module.run(q, k, cos_sin_cache, positions)
