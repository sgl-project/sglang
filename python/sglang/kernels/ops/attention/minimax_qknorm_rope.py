"""Fused per-head Gemma-RMSNorm + partial NeoX RoPE for MiniMax-M3 attention.

In-place over a fused QKV tensor: normalizes + rotates one or more groups of
heads (each group = a contiguous head run sharing one norm weight, all getting
RoPE), leaving every other head (V, index-V) untouched. Consumes the model's
own ``cos_sin_cache`` (fp32) so the rotation matches sglang's RotaryEmbedding
exactly.

Two entry points:

* :func:`minimax_qknorm_rope` -- the original main-attention call
  ``(q | k | v ...)``: Q heads then K heads, both normed + roped.
* :func:`minimax_qknorm_rope_grouped` -- a multi-group launch (up to 4 groups),
  used to fold the main Q/K *and* the sparse-index Q/K of one fused
  qkv+index-qkv GEMM output into a single kernel launch (mirroring the
  multi-branch single-launch design of ``fused_store_kv_index.cuh``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_GROUPS = 4


@cache_once
def _jit_module(pos_dtype, head_dim, rope_dim) -> Module:
    args = make_cpp_args(pos_dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        "fused_gemma_qknorm_rope",
        *args,
        cuda_files=["minimax/fused_gemma_qknorm_rope.cuh"],
        cuda_wrappers=[
            ("fused_gemma_qknorm_rope", f"fused_gemma_qknorm_rope<{args}>"),
            ("fused_gemma_qknorm_rope_fp8", f"fused_gemma_qknorm_rope_fp8<{args}>"),
        ],
    )


@register_custom_op(mutates_args=["qkv"])
def _fused_gemma_qknorm_rope(
    qkv: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    off0: int,
    cnt0: int,
    off1: int,
    cnt1: int,
    off2: int,
    cnt2: int,
    off3: int,
    cnt3: int,
    num_groups: int,
    eps: float,
) -> None:
    # Wrap the tvm-ffi kernel as a custom op so torch.compile / piecewise CUDA
    # graph can trace past the otherwise-opaque FFI call. The launch is
    # graph-capturable (host-side constant offsets/counts), so it stays inside
    # the captured region.
    module = _jit_module(positions.dtype, 128, 64)
    module.fused_gemma_qknorm_rope(
        qkv,
        w0,
        w1,
        w2,
        w3,
        cos_sin_cache,
        positions,
        off0,
        cnt0,
        off1,
        cnt1,
        off2,
        cnt2,
        off3,
        cnt3,
        num_groups,
        eps,
    )


@register_custom_op(mutates_args=["qkv", "fp8_a", "fp8_b"])
def _fused_gemma_qknorm_rope_fp8(
    qkv: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    fp8_a: torch.Tensor,
    fp8_b: torch.Tensor,
    off0: int,
    cnt0: int,
    off1: int,
    cnt1: int,
    off2: int,
    cnt2: int,
    off3: int,
    cnt3: int,
    num_groups: int,
    group_a: int,
    group_b: int,
    eps: float,
) -> None:
    module = _jit_module(positions.dtype, 128, 64)
    module.fused_gemma_qknorm_rope_fp8(
        qkv,
        w0,
        w1,
        w2,
        w3,
        cos_sin_cache,
        positions,
        fp8_a,
        fp8_b,
        off0,
        cnt0,
        off1,
        cnt1,
        off2,
        cnt2,
        off3,
        cnt3,
        num_groups,
        group_a,
        group_b,
        eps,
    )


def _pad_groups(groups: Sequence[Tuple[torch.Tensor, int, int]]):
    # Empty groups are dropped, so a caller finds its group's kernel slot here.
    slot_of = []
    kept = 0
    for _, _, cnt in groups:
        slot_of.append(kept if cnt > 0 else -1)
        kept += 1 if cnt > 0 else 0
    groups = [(w, off, cnt) for (w, off, cnt) in groups if cnt > 0]
    num_groups = len(groups)
    assert 1 <= num_groups <= _MAX_GROUPS, (
        f"need 1..{_MAX_GROUPS} groups, got {num_groups}"
    )
    weights: List[torch.Tensor] = [g[0] for g in groups]
    offsets: List[int] = [int(g[1]) for g in groups]
    counts: List[int] = [int(g[2]) for g in groups]
    # Pad weight slots up to 4 with a dummy (group 0's weight); the kernel never
    # reads padded slots because num_groups bounds the in-kernel group scan.
    while len(weights) < _MAX_GROUPS:
        weights.append(weights[0])
        offsets.append(0)
        counts.append(0)
    return num_groups, weights, offsets, counts, slot_of


def minimax_qknorm_rope_grouped_fp8(
    qkv: torch.Tensor,
    groups: Sequence[Tuple[torch.Tensor, int, int]],
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    *,
    fp8_a: torch.Tensor,
    group_a: int,
    fp8_b: torch.Tensor,
    group_b: int,
) -> torch.Tensor:
    """As :func:`minimax_qknorm_rope_grouped`, also emitting two groups as FP8.

    ``fp8_a`` / ``fp8_b`` are ``[T, head_count, head_dim]`` ``float8_e4m3fn``
    buffers receiving groups ``group_a`` / ``group_b``, bit-equal to the in-place
    BF16 result cast to FP8. The BF16 write-back is unchanged.
    """
    num_groups, weights, offsets, counts, slot_of = _pad_groups(groups)
    slot_a, slot_b = slot_of[group_a], slot_of[group_b]
    assert slot_a >= 0 and slot_b >= 0, "an FP8 output names an empty group"
    _fused_gemma_qknorm_rope_fp8(
        qkv,
        weights[0],
        weights[1],
        weights[2],
        weights[3],
        cos_sin_cache,
        positions,
        fp8_a,
        fp8_b,
        offsets[0],
        counts[0],
        offsets[1],
        counts[1],
        offsets[2],
        counts[2],
        offsets[3],
        counts[3],
        num_groups,
        slot_a,
        slot_b,
        eps,
    )
    return qkv


def minimax_qknorm_rope_grouped(
    qkv: torch.Tensor,
    groups: Sequence[Tuple[torch.Tensor, int, int]],
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused GemmaRMSNorm + partial NeoX RoPE over ``groups``, in place on ``qkv``.

    ``qkv`` is ``[T, total_heads * head_dim]`` (head_dim == 128). Each group is
    ``(weight, head_offset, head_count)``: ``head_count`` consecutive heads
    starting at ``head_offset`` (in head units) are normed with ``weight``
    (a ``[head_dim]`` bf16 tensor, the *raw* Gemma weight -- the kernel applies
    ``1 + weight``) and rotated. Heads outside every group are untouched.

    Up to 4 groups are supported in one launch. The offsets/counts are
    host-side constants, so the launch is CUDA-graph capturable.
    """
    num_groups, weights, offsets, counts, _ = _pad_groups(groups)
    _fused_gemma_qknorm_rope(
        qkv,
        weights[0],
        weights[1],
        weights[2],
        weights[3],
        cos_sin_cache,
        positions,
        offsets[0],
        counts[0],
        offsets[1],
        counts[1],
        offsets[2],
        counts[2],
        offsets[3],
        counts[3],
        num_groups,
        eps,
    )
    return qkv


def minimax_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    nq: int,
    nk: int,
    nv: int,  # deprecated / ignored: V heads are simply left untouched
    eps: float,
) -> torch.Tensor:
    """Main-attention layout ``[q (nq) | k (nk) | v ...]``: norm + rope Q then K."""
    return minimax_qknorm_rope_grouped(
        qkv,
        [(q_weight, 0, nq), (k_weight, nq, nk)],
        cos_sin_cache,
        positions,
        eps,
    )
