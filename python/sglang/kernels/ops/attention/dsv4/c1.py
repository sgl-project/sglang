"""Fused ratio-1 decode RMSNorm, RoPE, fp4 fake-quant and FlashMLA cache write.

The input is bf16 with no pooling. RoPE uses the token's own position, and the
compressed slot equals the FULL slot; out_loc == 0 marks graph padding.
The pre-RoPE latent is also returned for the index-key projection.
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
def _jit_c1_module(head_dim: int, rope_dim: int, page_size: int) -> Module:
    args = make_cpp_args(head_dim, rope_dim, page_size, is_arch_support_pdl())
    return load_jit(
        make_name("c1"),
        *args,
        cuda_files=["deepseek_v4/c1.cuh"],
        cuda_wrappers=[
            ("decode_fusion", f"FlashC1DecodeKernel<{args}>::run_decode_fusion"),
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
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMSNorm ``kv_input`` and write the main KV slot, in a single launch.

    ``out`` contains the pre-RoPE latent for the index-K branch's ``wk`` projection;
    the main KV cache receives the rotated and quantized value.

    :param kv_input: ``[num_tokens, head_dim]`` bf16 -- ``compressor.project(x)``
                     at ratio 1, i.e. the raw ``wkv`` output.
    :param norm_weight: ``[head_dim]`` bf16. ``DeepseekV41Compressor.norm`` holds
                        its weight in the model dtype and the multiply is done in
                        fp32 by promoting it, exactly as the module does.
    :param positions: ``[num_tokens]`` int32 or int64, the token's position.
                      Indexed into ``freqs_cis`` as-is: at ratio 1 the latent
                      stands for the token itself, so there is no ``- 1`` and no
                      gather launch.
    :param out_loc: ``[num_tokens]`` int32 or int64 ``c1_out_loc``, which at ratio 1
                    equals ``raw_out_loc`` (the scheduler's int64 ``out_cache_loc``).
                    ``0`` marks a padded graph row: it computes
                    and publishes its latent, which the caller discards, but
                    writes nothing to the cache.
    :param eps: RMSNorm epsilon.
    :param freqs_cis: ``[max_pos, rope_dim]`` fp32, real/imag interleaved --
                      ``torch.view_as_real(freqs).flatten(-2)``.
    :param k_cache: the compressed KV pool buffer for this layer.
    :param page_size: slots per page of that pool (``page_size // ratio``, i.e.
                      the FULL page size at ratio 1).
    :param out: ``[num_tokens, head_dim]`` bf16 destination for the pre-RoPE
                latent. Pass a persistent buffer under CUDA graphs.
    :return: ``out``, the pre-RoPE post-norm latent.
    """
    num_tokens, head_dim = kv_input.shape
    if out is None:
        out = kv_input.new_empty((num_tokens, head_dim))

    _jit_c1_module(head_dim, freqs_cis.shape[-1], page_size).decode_fusion(
        kv_input,
        out,
        norm_weight,
        freqs_cis,
        positions,
        out_loc,
        k_cache,
        float(eps),
    )
    return out
