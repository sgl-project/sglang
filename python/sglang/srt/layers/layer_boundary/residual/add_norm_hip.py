"""gfx95 fp8 reads of a GemmaRMSNorm on add_norm's aiter paths."""

from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_add_rmsnorm_fp8
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    emit_transposed_bpreshuffle_scale,
    materialize_bpreshuffle_fp8_scale_tuple,
)


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
    output, residual_out = gemma_add_rmsnorm_fp8(
        hidden_states,
        residual,
        norm.weight.data,
        eps,
        group_size=128,
        transpose_scale=emit_transposed_scale,
        keep_bf16=keep_bf16,
    )
    if not emit_transposed_scale and _use_aiter_bpreshuffle_gfx95:
        output = materialize_bpreshuffle_fp8_scale_tuple(output)
    return output, residual_out
