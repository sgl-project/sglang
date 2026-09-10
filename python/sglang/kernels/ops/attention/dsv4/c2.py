"""Fused ratio-2 decode pair-pooling, RMSNorm and optional main-KV write.

Closed-form softmax and FMA contraction can differ from torch by fp32 ulps;
bf16 rounding boundaries can preserve those differences. The pooling tests
use a tolerance, while state updates and stores from a given latent are bitwise.
Positions and state-ring indices describe the per-request decode schedule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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


@cache_once
def _jit_c2_module(head_dim: int, rope_dim: int = 64, page_size: int = 128) -> Module:
    # rope_dim / page_size only shape the store half; the norm-only wrapper is
    # unaffected by them and just takes the defaults.
    args = make_cpp_args(head_dim, rope_dim, page_size, is_arch_support_pdl())
    return load_jit(
        make_name("c2"),
        *args,
        cuda_files=["deepseek_v4/c2.cuh"],
        cuda_wrappers=[
            ("decode", f"FlashC2DecodeKernel<{args}>::run_decode"),
            ("decode_fusion", f"FlashC2DecodeKernel<{args}>::run_decode_fusion"),
        ],
    )


def c2_decode_norm(
    kv_input: torch.Tensor,
    kv_state: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    req: torch.Tensor,
    raw_out_loc: torch.Tensor,
    eps: float,
    *,
    ring_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pair-pool ``kv_input`` against ``kv_state`` and RMSNorm the result.

    :param kv_input: ``[num_tokens, 2 * head_dim]`` fp32, ``| kv | score |``.
    :param kv_state: ``CompressStatePool``'s flat ``KVAndScore`` buffer,
                     ``[size, 2 * head_dim]`` fp32, same ``| kv | score |``
                     layout. A request's pending pair lives at
                     ``req * ring_size + pos % ring_size``, so a completing row
                     reads what ``pos - 1`` left and a pending one writes its
                     own slot -- read and write never touch the same row.
    :param ring_size: ``CompressStatePool.ring_size``, positions per request.
    :param norm_weight: ``[head_dim]`` bf16 -- ``DeepseekV41Compressor.norm``
                        holds its weight in the model dtype, and the multiply is
                        done in fp32 by promoting it, exactly as the module does.
    :param positions: ``[num_tokens]`` int32 or int64. An odd position completes a group
                      with its even predecessor.
    :param req: ``[num_tokens]`` int64 ``req_pool_idx``, the ``kv_state`` row
                this token pairs through.
    :param raw_out_loc: ``[num_tokens]`` int32 or int64, the token's FULL-pool slot
                        (the scheduler's ``out_cache_loc`` is int64).
                        ``0`` marks a padded graph row, which the kernel skips
                        entirely -- reading nothing and writing nothing, so no
                        spare pair-state row is needed.
    :param eps: RMSNorm epsilon.
    :param out: ``[num_tokens, head_dim]`` bf16 destination. Pass a persistent
                buffer under CUDA graphs.
    :return: ``out``, the pre-RoPE post-norm latent.

    .. note:: **Rows at an even position, and padded rows, are not written.**
       They complete no group, so the kernel skips their output row rather than
       paying for a store the caller discards. Anything already in ``out`` on
       those rows survives the call.
    """
    num_tokens, fused_dim = kv_input.shape
    head_dim = fused_dim // 2
    if out is None:
        out = kv_input.new_empty((num_tokens, head_dim), dtype=torch.bfloat16)

    _jit_c2_module(head_dim).decode(
        kv_input,
        kv_state,
        out,
        norm_weight,
        positions,
        req,
        raw_out_loc,
        float(eps),
        int(ring_size),
    )
    return out


def c2_decode_norm_rope_store(
    kv_input: torch.Tensor,
    kv_state: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    req: torch.Tensor,
    raw_out_loc: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    k_cache: torch.Tensor,
    *,
    page_size: int,
    ring_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``c2_decode_norm`` plus the whole main-KV write, in the same launch.

    ``out`` contains the pre-RoPE latent for the index-K branch's ``wk`` projection.
    The cache store uses ``raw_out_loc // 2`` as its slot.

    :param freqs_cis: ``[max_pos, rope_dim]`` fp32, real/imag interleaved --
                      ``torch.view_as_real(freqs).flatten(-2)``. Indexed
                      in-kernel at ``positions - 1``, the position the latent
                      stands for, so there is no gather launch.
    :param k_cache: the compressed KV pool buffer for this layer.
    :param page_size: slots per page of that pool (``page_size // ratio``).
    """
    num_tokens, fused_dim = kv_input.shape
    head_dim = fused_dim // 2
    if out is None:
        out = kv_input.new_empty((num_tokens, head_dim), dtype=torch.bfloat16)

    _jit_c2_module(head_dim, freqs_cis.shape[-1], page_size).decode_fusion(
        kv_input,
        kv_state,
        out,
        norm_weight,
        positions,
        req,
        raw_out_loc,
        float(eps),
        freqs_cis,
        k_cache,
        int(ring_size),
    )
    return out
