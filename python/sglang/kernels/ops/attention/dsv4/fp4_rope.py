"""Fused RoPE and two-stage fp4 packing for low-ratio index keys and queries.

The key path includes RMSNorm and a 68-byte cache store. RoPE uses the group's
first position, positions & ~(ratio - 1), for power-of-two compress ratios.
The query path has no RMSNorm or cache store and uses each token's own position;
it returns the payload and scales consumed by the paged MQA logits kernel.
The query wrapper is available but the backend currently uses the Triton path.
``index_q_rope_pack_weights`` is the query wrapper the decode backend uses: the
same query pack plus the indexer's head weights (``head_weights(x).float()``)
written by the same warp, replacing the Triton packer and two aten launches.
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

    ``_low_ratio_index_topk_decode`` needs ``head_weights(x).float()`` next to
    the packed query: ``weights_proj(x)`` (bf16) times
    ``softmax_scale * n_heads**-0.5``, as fp32. In torch that is two more
    launches (a bf16 multiply and the ``.float()`` copy). The kernel already
    owns one warp per (token, head) row, so lane 0 of each row writes
    ``float(bf16(w * scale))`` -- the same fp32 multiply, the same
    round-to-nearest-even to bf16 -- and the result is bitwise the torch value.

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
