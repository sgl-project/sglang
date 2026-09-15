"""Fused RoPE and two-stage fp4 packing for low-ratio index keys and queries.

Keys include RMSNorm and a 68-byte cache store, using the group's first position.
Queries use each token's own position, without RMSNorm or a cache store.
The decode backend uses index_q_rope_pack_weights to also produce FP32 head weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Payload bytes plus one ue8m0 exponent per 32 elements, per compressed token.
SLOT_BYTES = 68
INDEX_PAGE_SIZE = 64


@cache_once
def _jit_index_k_module(
    head_dim: int, rope_dim: int, page_size: int, ratio: int
) -> Module:
    args = make_cpp_args(head_dim, rope_dim, page_size, ratio, is_arch_support_pdl())
    return load_jit(
        make_name("fp4_rope"),
        *args,
        cuda_files=["deepseek_v4/fp4_rope.cuh"],
        cuda_wrappers=[("index_k", f"FlashIndexKKernel<{args}>::run_index_k")],
    )


@cache_once
def _jit_index_q_module(head_dim: int, rope_dim: int) -> Module:
    args = make_cpp_args(head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name("fp4_rope"),
        *args,
        cuda_files=["deepseek_v4/fp4_rope.cuh"],
        cuda_wrappers=[
            ("index_q", f"FlashIndexQKernel<{args}>::run_index_q"),
            ("index_q_weights", f"FlashIndexQKernel<{args}>::run_index_q_weights"),
        ],
    )


def index_k_norm_rope_pack_store(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    cache: torch.Tensor,
    *,
    ratio: int,
) -> None:
    """Normalize, rotate, quantize twice and store one index-K slot per token.

    :param input: ``[num_tokens, index_head_dim]`` bf16 -- ``wk(latent)``,
                  *before* ``k_norm``.
    :param norm_weight: ``[index_head_dim]`` bf16, ``k_norm.weight``.
    :param eps: ``k_norm.eps``.
    :param freqs_cis: ``[max_pos, rope_head_dim]`` fp32, real/imag interleaved --
                      ``torch.view_as_real(freqs).flatten(-2)``. Indexed
                      in-kernel, so pass the whole table rather than a gather.
    :param positions: ``[num_tokens]`` int32 or int64, the token position. The
                      group position is masked out of it in-kernel.
    :param loc: ``[num_tokens]`` int64, the index-K slot. ``0`` is the reserved
                dummy: those rows publish nothing, which covers both padded
                graph rows and, at ratio > 1, rows completing no group.
    :param cache: the layer's index-K buffer, ``[npages, page_size * 68]`` uint8.
    :param ratio: the layer's compress ratio. A power of two.

    .. note:: Two quantization stages, not one. The fake-quant's amax floor is
       ``6 * 2**-126`` and the packer's is ``1e-4``, applied on opposite sides
       of the divide by 6, so the packer can recover an exponent the fake-quant
       gave away and collapsing them is not equivalent.
    """
    head_dim = input.shape[-1]
    _jit_index_k_module(
        head_dim, freqs_cis.shape[-1], cache.shape[1] // SLOT_BYTES, ratio
    ).index_k(input, norm_weight, freqs_cis, positions, loc, cache, float(eps))


def index_q_rope_pack(
    input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate, quantize twice and pack one indexer query per (token, head).

    :param input: ``[num_tokens, heads, index_head_dim]`` bf16 contiguous --
                  ``wq_b(q_lora)`` viewed per head.
    :param freqs_cis: ``[max_pos, rope_head_dim]`` fp32, real/imag interleaved --
                      ``torch.view_as_real(freqs).flatten(-2)``. Indexed
                      in-kernel, so pass the whole table rather than a gather.
    :param positions: ``[num_tokens]`` int32 or int64. A query rotates by its
                      own position, so this is used unmasked.
    :return: ``(payload, scale)`` -- ``[num_tokens * heads, index_head_dim // 2]``
             int8 and ``[num_tokens * heads]`` int32, the four ue8m0 block
             exponents packed little-endian. Exactly what the paged MQA logits
             kernel takes and what the Triton path returns without a cache.

    .. note:: Two quantization stages, not one -- see
       :func:`index_k_norm_rope_pack_store`. Neither can be dropped.
    """
    num_tokens, heads, head_dim = input.shape
    rows = num_tokens * heads
    payload = input.new_empty((rows, head_dim // 2), dtype=torch.int8)
    scale = input.new_empty((rows,), dtype=torch.int32)

    _jit_index_q_module(head_dim, freqs_cis.shape[-1]).index_q(
        input, freqs_cis, positions, payload, scale
    )
    return payload, scale


def index_q_rope_pack_weights(
    input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    head_weights: torch.Tensor,
    weight_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`index_q_rope_pack` plus the indexer's head weights, one launch.

    Head weights match ``head_weights(x).float()``: multiply in fp32,
    round to nearest-even bf16, then widen to fp32.

    :param head_weights: ``[num_tokens, heads]`` bf16, the raw ``weights_proj``
                         output (before the scale).
    :param weight_scale: ``softmax_scale * heads**-0.5``; rounded to fp32 in the
                         kernel exactly as torch rounds a Python scalar for a
                         bf16 tensor multiply.
    :return: ``(payload, scale, weights)`` -- the first two as
             :func:`index_q_rope_pack`, ``weights`` ``[num_tokens, heads]`` fp32.
    """
    num_tokens, heads, head_dim = input.shape
    rows = num_tokens * heads
    payload = input.new_empty((rows, head_dim // 2), dtype=torch.int8)
    scale = input.new_empty((rows,), dtype=torch.int32)
    weights = input.new_empty((num_tokens, heads), dtype=torch.float32)

    _jit_index_q_module(head_dim, freqs_cis.shape[-1]).index_q_weights(
        input,
        freqs_cis,
        positions,
        payload,
        scale,
        head_weights,
        weights,
        float(weight_scale),
    )
    return payload, scale, weights
