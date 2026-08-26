"""Fused QSA (Qwen4-Exp sparse attention) indexer-prep kernels.

``qsa_index_q_norm_rope_store`` fuses, per token, the eager chain
split -> GemmaRMSNorm(index q) -> MRoPE(index q) -> raw-K store ->
RoPE-position store into one kernel launch.

``qsa_index_k_compress_store`` fuses, per completed compress group, the eager
chain gather -> fp32 mean -> GemmaRMSNorm -> MRoPE(group-start position) ->
compressed-cache store into one kernel launch.

Both kernels reproduce the eager numerics step by step (fp32 norm reduction,
per-op rounding to the storage dtype during RoPE, fp32 group mean rounded to
the storage dtype before the norm). Outputs are bit-comparable to the eager
indexer path: the eager RMSNorm (flashinfer's CuTe DSL kernel) reduces sums of
squares in an order that cannot be reproduced exactly, so a small fraction of
rows (~1 in 30k) may flip by one bf16 ulp on a rounding boundary.
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

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_qsa_indexer_module(
    dtype: torch.dtype, head_dim: int, is_neox_style: bool
) -> Module:
    """Compile and cache the JIT QSA indexer module for one specialisation."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: bfloat16, float16"
        )
    if head_dim not in (64, 128, 256):
        raise RuntimeError(
            f"Unsupported index head_dim {head_dim}. Supported: 64, 128, 256"
        )
    args = make_cpp_args(dtype, head_dim, is_neox_style, is_arch_support_pdl())
    return load_jit(
        "qsa_indexer",
        *args,
        cuda_files=["attention/qsa_indexer.cuh"],
        cuda_wrappers=[
            ("q_prep", f"qsa_index_q_prep<{args}>"),
            ("k_compress", f"qsa_index_k_compress<{args}>"),
        ],
    )


def qsa_index_q_norm_rope_store(
    qk: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    axis_map: torch.Tensor,
    weight: torch.Tensor,
    cache_loc: torch.Tensor,
    key_state_buffer: torch.Tensor,
    rope_position_buffer: torch.Tensor,
    num_q_heads: int,
    rotary_dim: int,
    eps: float,
    is_neox_style: bool,
    q_heads_padded: Optional[int] = None,
) -> torch.Tensor:
    """
    Per-token fused index-Q prep.

    Parameters
    ----------
    qk              : CUDA bf16/fp16 [tokens, (num_q_heads + 1) * head_dim],
                      contiguous fused index Q/K projection output
    positions       : CUDA int64 [tokens] or [3, tokens] RoPE positions
                      (a strided trailing slice is accepted)
    cos_sin_cache   : CUDA fp32 [capacity, rotary_dim] RoPE cache
    axis_map        : CUDA int32 [rotary_dim // 2] position-axis per pair index
    weight          : [head_dim] gemma norm weight (kernel applies 1 + w)
    cache_loc       : CUDA int64 [tokens] state slots of each token
    key_state_buffer : CUDA [slots, head_dim] raw-K state buffer (written)
    rope_position_buffer : CUDA int64 [slots, 3] position buffer (written)
    num_q_heads     : number of index query heads
    rotary_dim      : rotated prefix of each head row
    eps             : RMSNorm epsilon
    is_neox_style   : NeoX (True) or GPT-J (False) RoPE pairing
    q_heads_padded  : output head count; heads >= num_q_heads are zero-filled
                      (defaults to num_q_heads)

    Returns
    -------
    CUDA tensor [tokens, q_heads_padded, head_dim]: normed + rotated index Q.
    """
    num_tokens = qk.shape[0]
    head_dim = weight.shape[0]
    if q_heads_padded is None:
        q_heads_padded = num_q_heads
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    q_out = torch.empty(
        (num_tokens, q_heads_padded, head_dim), dtype=qk.dtype, device=qk.device
    )
    module = _jit_qsa_indexer_module(qk.dtype, head_dim, is_neox_style)
    module.q_prep(
        qk,
        q_out,
        weight,
        cos_sin_cache,
        axis_map,
        positions,
        positions.shape[0],
        cache_loc,
        key_state_buffer,
        rope_position_buffer,
        num_q_heads,
        rotary_dim,
        eps,
    )
    return q_out


def qsa_index_k_compress_store(
    key_state_buffer: torch.Tensor,
    group_locs: torch.Tensor,
    rope_position_buffer: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    axis_map: torch.Tensor,
    weight: torch.Tensor,
    write_locs: torch.Tensor,
    compressed_k_buffer: torch.Tensor,
    compress_ratio: int,
    rotary_dim: int,
    eps: float,
    is_neox_style: bool,
) -> None:
    """
    Per-group compressed-K prep (in-place store into ``compressed_k_buffer``).

    Parameters
    ----------
    key_state_buffer : CUDA bf16/fp16 [slots, head_dim] raw-K state buffer
    group_locs       : CUDA int32 [groups, compress_ratio] state slots of each
                       completed group, group-start slot first? (any order;
                       rope position is read from column 0)
    rope_position_buffer : CUDA int64 [slots, 3] per-slot RoPE coordinates
    cos_sin_cache    : CUDA fp32 [capacity, rotary_dim] RoPE cache
    axis_map         : CUDA int32 [rotary_dim // 2] position-axis per pair index
    weight           : [head_dim] gemma norm weight (kernel applies 1 + w)
    write_locs       : CUDA int32 [groups] compressed-cache slots to write
    compressed_k_buffer : CUDA [compressed_slots, head_dim] (written)
    compress_ratio   : raw keys per compressed key
    rotary_dim       : rotated prefix of each head row
    eps              : RMSNorm epsilon
    is_neox_style    : NeoX (True) or GPT-J (False) RoPE pairing
    """
    head_dim = weight.shape[0]
    module = _jit_qsa_indexer_module(
        key_state_buffer.dtype, head_dim, is_neox_style
    )
    module.k_compress(
        key_state_buffer,
        group_locs,
        rope_position_buffer,
        cos_sin_cache,
        axis_map,
        weight,
        write_locs,
        compressed_k_buffer,
        compress_ratio,
        rotary_dim,
        eps,
    )
