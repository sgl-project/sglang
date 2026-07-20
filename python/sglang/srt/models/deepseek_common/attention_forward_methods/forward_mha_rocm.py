"""AMD/ROCm helpers for DeepSeek MHA forward.

Kernel bodies formerly under ``_is_hip`` / ``_use_aiter*`` in ``forward_mha.py``.
Call-site conditions stay in the shared file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.ops.attention.utils import concat_and_cast_mha_k_triton
from sglang.srt.layers.quantization.fp8_utils import (
    materialize_bpreshuffle_fp8_scale_tuple,
)
from sglang.srt.models.deepseek_common.utils import (
    _use_aiter_bpreshuffle_gfx95,
    _use_aiter_gfx95,
)

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import fused_rms_mxfp4_quant


def rocm_normalize_q_for_mha_dsa_fp8(
    attn: DeepseekV2AttentionMLA,
    q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DSA path: fused RMSNorm+FP8 with unquantized q_lora for the indexer."""
    q_quanted, q_lora, _, _ = fused_rms_fp8_group_quant(
        q,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.variance_epsilon,
        None,
        None,
        None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        output_unquantized_inp1=True,
        transpose_scale=False,
    )
    if _use_aiter_bpreshuffle_gfx95:
        q_quanted = materialize_bpreshuffle_fp8_scale_tuple(q_quanted)
    q = attn.q_b_proj(q_quanted)[0].view(-1, attn.num_local_heads, attn.qk_head_dim)
    return q, q_lora


def rocm_normalize_q_for_mha_mxfp4(
    attn: DeepseekV2AttentionMLA,
    q: torch.Tensor,
) -> torch.Tensor:
    """MXFP4: fused RMSNorm + quant, then q_b_proj."""
    q, _, _, _ = fused_rms_mxfp4_quant(
        q,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.variance_epsilon,
        None,
        None,
        None,
    )
    return attn.q_b_proj(q)[0].view(-1, attn.num_local_heads, attn.qk_head_dim)


def rocm_normalize_q_for_mha_fp8(
    attn: DeepseekV2AttentionMLA,
    q: torch.Tensor,
) -> torch.Tensor:
    """FP8 (non-DSA): fused RMSNorm + group quant, then q_b_proj."""
    q, _, _, _ = fused_rms_fp8_group_quant(
        q,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.variance_epsilon,
        None,
        None,
        None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        output_unquantized_inp1=False,
        transpose_scale=False,
    )
    if _use_aiter_bpreshuffle_gfx95:
        q = materialize_bpreshuffle_fp8_scale_tuple(q)
    return attn.q_b_proj(q)[0].view(-1, attn.num_local_heads, attn.qk_head_dim)


def rocm_normalize_kv_a(
    attn: DeepseekV2AttentionMLA,
    kv_a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm+FP8 for kv_a; returns (kv_a_quanted, kv_a_unquant)."""
    kv_a_quanted, kv_a, _, _ = fused_rms_fp8_group_quant(
        kv_a,
        attn.kv_a_layernorm.weight,
        attn.kv_a_layernorm.variance_epsilon,
        None,
        None,
        None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        output_unquantized_inp1=True,  # return unqaunt kv_a
        transpose_scale=False,
    )
    if _use_aiter_bpreshuffle_gfx95:
        kv_a_quanted = materialize_bpreshuffle_fp8_scale_tuple(kv_a_quanted)
    return kv_a_quanted, kv_a


def rocm_kv_b_proj_mxfp4_fp8_prefill(
    attn: DeepseekV2AttentionMLA,
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 weights + FP8 prefill: fused GEMM / split / cat into FP8 k,v."""
    k, v = attn.kv_b_proj(
        (
            kv_a,
            k_pe.expand(-1, attn.num_local_heads, -1),
            attn.qk_nope_head_dim,
            attn.v_head_dim,
            fp8_dtype,
        )
    )[0]
    return k, v


def rocm_concat_mha_k(
    attn: DeepseekV2AttentionMLA,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> torch.Tensor:
    """AITER HIP path for concatenating k_nope and k_pe."""
    k_shape = (k_nope.shape[0], attn.num_local_heads, attn.qk_head_dim)
    k = k_nope.new_empty(*k_shape)
    concat_and_cast_mha_k_triton(k, k_nope, k_pe)
    return k
