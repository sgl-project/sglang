"""AMD/ROCm helpers for DeepSeek absorbed MLA forward.

Branches formerly under ``_is_hip`` / ``_use_aiter*`` in ``forward_mla.py`` live
here so non-AMD builds never import ``aiter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.quantization.fp8_utils import (
    materialize_bpreshuffle_fp8_scale_tuple,
)
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.models.deepseek_common.utils import (
    _use_aiter,
    _use_aiter_bpreshuffle_gfx95,
    _use_aiter_gfx95,
)

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _use_aiter:
    # aiter ROCm/aiter#2958 renamed the public `fused_qk_rmsnorm` in
    # `aiter.ops.fused_qk_norm_rope_cache_quant` to a private `_fused_qk_rmsnorm`
    # and introduced a unified entry point in `aiter.ops.fused_qk_rmsnorm_group_quant`
    # with a different (in-place, kwarg-only, no-return) signature. Probe for the
    # new symbol first so SGLang works with both pre- and post-#2958 aiter without
    # requiring the docker pin to be bumped atomically.
    try:
        from aiter.ops.enum import QuantType as _AiterQuantType
        from aiter.ops.fused_qk_rmsnorm_group_quant import (
            fused_qk_rmsnorm as _aiter_fused_qk_rmsnorm_unified,
        )

        def fused_qk_rmsnorm_bf16(q, q_weight, q_eps, k, k_weight, k_eps):
            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            _aiter_fused_qk_rmsnorm_unified(
                q_out_quantized=q_out,
                k_out=k_out,
                q=q,
                q_weight=q_weight,
                q_epsilon=q_eps,
                k=k,
                k_weight=k_weight,
                k_epsilon=k_eps,
                quant_type=_AiterQuantType.No,
            )
            return q_out, k_out

    except ImportError:
        from aiter.ops.fused_qk_norm_rope_cache_quant import (
            fused_qk_rmsnorm as fused_qk_rmsnorm_bf16,
        )

    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )

if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla


def rocm_normalize_q_kv_a(
    attn: DeepseekV2AttentionMLA,
    q: torch.Tensor,
    k_nope: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fused / AITER Q and KV-a RMSNorm (+ optional quant).

    Caller must only invoke this when ``_use_aiter`` is true.

    Returns:
        ``(q, k_nope, q_lora)`` where ``q_lora`` is set only on the DSA FP8 path
        that requests an unquantized copy; otherwise ``None``.
    """
    q_lora: Optional[torch.Tensor] = None
    if _use_aiter_gfx95 and attn.q_b_proj.weight.dtype == torch.uint8:
        q, _, k_nope, *_ = fused_rms_mxfp4_quant(
            q,
            attn.q_a_layernorm.weight,
            attn.q_a_layernorm.variance_epsilon,
            k_nope,
            attn.kv_a_layernorm.weight,
            attn.kv_a_layernorm.variance_epsilon,
        )
    elif _use_aiter_gfx95 and attn.q_b_proj.weight.dtype == torch.float8_e4m3fn:
        if attn.use_dsa:
            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                q,
                attn.q_a_layernorm.weight,
                attn.q_a_layernorm.variance_epsilon,
                k_nope,
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.variance_epsilon,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                res1=None,
                output_unquantized_inp1=True,
                transpose_scale=False,
            )
            if _use_aiter_bpreshuffle_gfx95:
                q_quanted = materialize_bpreshuffle_fp8_scale_tuple(q_quanted)
            q = q_quanted
        else:
            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                q,
                attn.q_a_layernorm.weight,
                attn.q_a_layernorm.variance_epsilon,
                k_nope,
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.variance_epsilon,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                res1=None,
                output_unquantized_inp1=False,
                transpose_scale=False,
            )
            if _use_aiter_bpreshuffle_gfx95:
                q = materialize_bpreshuffle_fp8_scale_tuple(q)
    else:
        q, k_nope = fused_qk_rmsnorm_bf16(
            q,
            attn.q_a_layernorm.weight,
            attn.q_a_layernorm.variance_epsilon,
            k_nope,
            attn.kv_a_layernorm.weight,
            attn.kv_a_layernorm.variance_epsilon,
        )
    return q, k_nope, q_lora


def rocm_absorb_q_bmm(
    attn: DeepseekV2AttentionMLA,
    q_nope: torch.Tensor,
    *,
    is_capture_mode: bool,
) -> torch.Tensor:
    """Absorb ``q_nope @ w_kc`` on HIP/AITER (pre-transpose layout)."""
    # TODO(haishaw): add bmm_fp8 to ROCm
    if _use_aiter_gfx95 and attn.w_kc.dtype == torch.uint8:
        x = q_nope.transpose(0, 1)
        q_nope_out = torch.empty(
            x.shape[0],
            x.shape[1],
            attn.w_kc.shape[2],
            device=x.device,
            dtype=torch.bfloat16,
        )
        batched_gemm_afp4wfp4_pre_quant(
            x,
            attn.w_kc.transpose(-2, -1),
            attn.w_scale_k.transpose(-2, -1),
            torch.bfloat16,
            q_nope_out,
        )
    else:
        if (_use_aiter_gfx95 and attn.w_kc.dtype == torch.float8_e4m3fn) or (
            is_capture_mode and attn.w_kc.dtype == torch.float8_e4m3fnuz
        ):
            # fp8 Triton kernel: always on gfx950,
            # cudagraph-only on gfx942 (hides launch overhead)
            q_nope_out = (
                batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                    X=q_nope,
                    WQ=attn.w_kc.transpose(-1, -2),
                    w_scale=attn.w_scale,
                    group_size=128,
                    YQ=None,  # allocate (B, M, N)
                    transpose_bm=False,  # (B, M, N)
                    transpose_bm_in=True,  # (M, B, K)
                    dtype=torch.bfloat16,
                )
            )
        else:
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                attn.w_kc.to(torch.bfloat16) * attn.w_scale,
            )
    return q_nope_out


def rocm_absorb_v_bmm(
    attn: DeepseekV2AttentionMLA,
    attn_output: torch.Tensor,
) -> torch.Tensor:
    """Absorb ``attn_output @ w_vc`` (+ optional fused flatten quant) on HIP."""
    # TODO(haishaw): add bmm_fp8 to ROCm
    if _use_aiter_gfx95 and attn.w_vc.dtype == torch.uint8:
        x = attn_output.transpose(0, 1)
        B_heads, M_batch = x.shape[0], x.shape[1]
        N_vdim = attn.w_vc.shape[2]
        # Allocate in (batch, heads, dim) so the post-GEMM
        # transpose+flatten is a free view instead of a copy.
        _bmm_buf = torch.empty(
            M_batch,
            B_heads,
            N_vdim,
            device=x.device,
            dtype=torch.bfloat16,
        )
        attn_bmm_output = _bmm_buf.transpose(0, 1)
        batched_gemm_afp4wfp4_pre_quant(
            x,
            attn.w_vc.transpose(-2, -1),
            attn.w_scale_v.transpose(-2, -1),
            torch.bfloat16,
            attn_bmm_output,
        )
    else:
        _bmm_buf = None
        if _use_aiter_gfx95 and attn.w_kc.dtype == torch.float8_e4m3fn:
            attn_bmm_output = (
                batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                    X=attn_output,
                    WQ=attn.w_vc.transpose(-1, -2),
                    w_scale=attn.w_scale,
                    group_size=128,
                    YQ=None,
                    transpose_bm=False,
                    transpose_bm_in=True,
                    dtype=torch.bfloat16,
                )
            )
        else:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                attn.w_vc.to(torch.bfloat16) * attn.w_scale,
            )

    if _bmm_buf is not None:
        # _bmm_buf is already (batch, heads, dim) contiguous
        if attn.o_proj.weight.dtype == torch.uint8:
            attn_bmm_output = fused_flatten_mxfp4_quant(_bmm_buf)
        elif attn.o_proj.weight.dtype == torch.float8_e4m3fn:
            attn_bmm_output = fused_flatten_fp8_group_quant(
                _bmm_buf,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                transpose_scale=False,
            )
            if _use_aiter_bpreshuffle_gfx95:
                attn_bmm_output = materialize_bpreshuffle_fp8_scale_tuple(
                    attn_bmm_output
                )
        else:
            attn_bmm_output = _bmm_buf.flatten(1, 2)
    elif attn.o_proj.weight.dtype == torch.uint8:
        attn_bmm_output = attn_bmm_output.transpose(0, 1)
        attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
    elif attn.o_proj.weight.dtype == torch.float8_e4m3fn:
        attn_bmm_output = attn_bmm_output.transpose(0, 1)
        attn_bmm_output = fused_flatten_fp8_group_quant(
            attn_bmm_output,
            group_size=128,
            dtype_quant=torch.float8_e4m3fn,
            transpose_scale=False,
        )
        if _use_aiter_bpreshuffle_gfx95:
            attn_bmm_output = materialize_bpreshuffle_fp8_scale_tuple(attn_bmm_output)
    else:
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

    return attn_bmm_output


def rocm_fused_qk_rope_cat_and_cache_mla(
    attn: DeepseekV2AttentionMLA,
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    *,
    out_cache_loc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE + concat + KV-cache write via AITER fused kernel on gfx95."""
    cos = attn.rotary_emb.cos_cache
    sin = attn.rotary_emb.sin_cache
    kv_cache_dtype = (
        fp8_dtype if attn.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
    )
    return fused_qk_rope_cat_and_cache_mla(
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        get_token_to_kv_pool().get_key_buffer(attn.attn_mqa.layer_id),
        out_cache_loc,
        positions,
        cos,
        sin,
        attn.attn_mqa.k_scale,
        attn.rotary_emb.is_neox_style,
        q_out_dtype=kv_cache_dtype,
    )
