"""gfx95 fp8 reads of a GemmaRMSNorm on add_norm's aiter paths, and the output they
zero-fill for the asm split-K GEMM reading them."""

import torch

from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_add_rmsnorm_fp8
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    emit_transposed_bpreshuffle_scale,
    materialize_bpreshuffle_fp8_scale_tuple,
)

if _use_aiter_bpreshuffle_gfx95:
    from sglang.srt.layers.quantization.fp8_utils_hip import asm_split_k_kernel


def gemma_norm_fp8(
    norm,
    hidden_states,
    residual,
    post_residual_addition,
    quant_format,
    keep_bf16=False,
):
    """A gfx95 Gemma norm quantized for ``quant_format`` ("fp8_per_token" or "fp8")
    as the separate norm and aiter quant kernels would. aiter's fused norm + quant
    kernels apply a plain RMSNorm weight, not GemmaRMSNorm's 1 + weight."""
    if residual is not None and post_residual_addition is not None:
        residual = residual + post_residual_addition
    eps = norm.variance_epsilon
    if quant_format == "fp8_per_token":
        return gemma_add_rmsnorm_fp8(
            hidden_states, residual, norm.weight.data, eps, keep_bf16=keep_bf16
        )
    emit_transposed_scale = emit_transposed_bpreshuffle_scale(
        hidden_states.shape[0],
        on_bpreshuffle_gfx95=_use_aiter_bpreshuffle_gfx95,
    )
    zeroed_gemm_out = _zeroed_gemm_out(norm, hidden_states)
    output, residual_out = gemma_add_rmsnorm_fp8(
        hidden_states,
        residual,
        norm.weight.data,
        eps,
        group_size=128,
        transpose_scale=emit_transposed_scale,
        keep_bf16=keep_bf16,
        zeros=zeroed_gemm_out,
    )
    if zeroed_gemm_out is not None:
        # read (once) by aiter_w8a8_block_fp8_linear
        output[0].zeroed_gemm_out = zeroed_gemm_out
    if not emit_transposed_scale and _use_aiter_bpreshuffle_gfx95:
        output = materialize_bpreshuffle_fp8_scale_tuple(output)
    return output, residual_out


def _zeroed_gemm_out(norm, hidden_states):
    """Output buffer of the asm split-K GEMM that reads this norm's fp8, for the norm to
    zero-fill instead of the memset node aiter's launcher puts before that GEMM."""
    # dynamo cannot trace the tensor-attribute handoff; the asm kernel writes bf16 only
    if (
        norm.fp8_out_gemm_shape is None
        or torch.compiler.is_compiling()
        or hidden_states.dtype != torch.bfloat16
    ):
        return None
    m = hidden_states.shape[0]
    n, k = norm.fp8_out_gemm_shape
    if asm_split_k_kernel(m, n, k) is None:
        return None
    return hidden_states.new_empty((m, n))
