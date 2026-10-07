"""Fused ratio-1 and ratio-2 decode compressors: RMSNorm, RoPE and the FlashMLA
cache write in one launch.

Ratio 1 takes the bf16 ``wkv`` projection as is: RoPE uses the token's own
position and the compressed slot equals the FULL slot. Ratio 2 pair-pools the
token against the pending partner in the state ring first. ``out_loc == 0``
marks a padded graph row on both paths, and both return the pre-RoPE latent for
the index-key projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .kv_layout import KVLayout
from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_c1_module(head_dim: int, rope_dim: int, page_size: int, layout: KVLayout):
    args = make_cpp_args(
        head_dim,
        rope_dim,
        page_size,
        layout.cpp_name,
        is_arch_support_pdl(),
    )
    return load_jit(
        make_name("c1_decode"),
        *args,
        cuda_files=["deepseek_v4/c1.cuh"],
        cuda_wrappers=[
            ("decode_fusion", f"FlashCompress1Kernel<{args}>::run_decode_fusion"),
        ],
    )


def c1_decode_norm_rope_store(
    kv_input: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    k_cache: torch.Tensor,
    *,
    page_size: int,
    layout: Union[KVLayout, str] = KVLayout.V4,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMSNorm ``kv_input`` and write the main KV slot, in a single launch.

    :param kv_input: ``[num_tokens, head_dim]`` bf16, the raw ``wkv`` projection
                     output.
    :param norm_weight: ``[head_dim]`` bf16, promoted to fp32 for the multiply.
    :param positions: ``[num_tokens]`` int32 or int64, indexed into ``freqs_cis``
                      as-is: at ratio 1 the latent stands for the token itself,
                      so there is no ``- 1``.
    :param out_loc: ``[num_tokens]`` int32 or int64 ``c1_out_loc``, which at ratio 1
                    equals ``raw_out_loc`` (the scheduler's int64 ``out_cache_loc``).
                    ``0`` marks a padded graph row: its latent is still computed
                    and published, but nothing is written to the cache.
    :param eps: RMSNorm epsilon.
    :param freqs_cis: ``[max_pos, rope_dim]`` fp32, real/imag interleaved --
                      ``torch.view_as_real(freqs).flatten(-2)``.
    :param k_cache: the compressed KV pool buffer for this layer.
    :param page_size: slots per page of that pool (``page_size // ratio``, i.e.
                      the FULL page size at ratio 1).
    :param layout: the pool's :class:`KVLayout`. The fp8 layouts (``V4``,
                   ``V41``) store the fp4 fake-quantized value; ``V41_FP4``
                   stores the e2m1 codes themselves, rounding once.
    :param out: ``[num_tokens, head_dim]`` bf16 destination for the pre-RoPE
                latent. Pass a persistent buffer under CUDA graphs.
    :return: ``out``, the pre-RoPE post-norm latent.
    """
    num_tokens, head_dim = kv_input.shape
    if out is None:
        out = kv_input.new_empty((num_tokens, head_dim))

    layout = KVLayout.parse(layout)
    module = _jit_c1_module(head_dim, freqs_cis.shape[-1], page_size, layout)
    module.decode_fusion(
        kv_input,
        out,
        norm_weight,
        freqs_cis,
        positions,
        out_loc,
        # HIP's fp8_e4m3_t is uint8_t, so the kernel matches the fp8 pool as bytes
        k_cache.view(torch.uint8) if is_hip_runtime() else k_cache,
        float(eps),
    )
    return out


@cache_once
def _jit_c2_module(
    head_dim: int,
    rope_dim: int,
    page_size: int,
    layout: KVLayout,
) -> Module:
    args = make_cpp_args(
        head_dim,
        rope_dim,
        page_size,
        layout.cpp_name,
        is_arch_support_pdl(),
    )
    return load_jit(
        make_name("c2_decode"),
        *args,
        cuda_files=["deepseek_v4/c2.cuh"],
        cuda_wrappers=[
            ("decode_fusion", f"FlashCompress2Kernel<{args}>::run_decode_fusion"),
        ],
    )


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
    draft_len: int = 1,
    layout: Union[KVLayout, str] = KVLayout.V4,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pair-pool ``kv_input`` against ``kv_state``, RMSNorm, and write the main KV slot.

    The cache store uses ``raw_out_loc // 2`` as its slot.

    :param freqs_cis: ``[max_pos, rope_dim]`` fp32, real/imag interleaved --
                      ``torch.view_as_real(freqs).flatten(-2)``. Indexed
                      in-kernel at ``positions - 1``, the position the latent
                      stands for.
    :param k_cache: the compressed KV pool buffer for this layer.
    :param page_size: slots per page of that pool (``page_size // ratio``).
    :param layout: the pool's :class:`KVLayout`. The fp8 layouts (``V4``,
                   ``V41``) store the fp4 fake-quantized value; ``V41_FP4``
                   stores the e2m1 codes themselves, rounding once.
    """
    num_tokens, fused_dim = kv_input.shape
    head_dim = fused_dim // 2
    if out is None:
        out = kv_input.new_empty((num_tokens, head_dim), dtype=torch.bfloat16)

    layout = KVLayout.parse(layout)
    module = _jit_c2_module(head_dim, freqs_cis.shape[-1], page_size, layout)
    module.decode_fusion(
        kv_input,
        kv_state,
        out,
        norm_weight,
        positions,
        req,
        raw_out_loc,
        eps,
        freqs_cis,
        # HIP's fp8_e4m3_t is uint8_t, so the kernel matches the fp8 pool as bytes
        k_cache.view(torch.uint8) if is_hip_runtime() else k_cache,
        ring_size,
        draft_len,
    )
    return out
